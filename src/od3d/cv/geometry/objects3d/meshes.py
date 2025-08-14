import logging
from enum import Enum
from pathlib import Path
from typing import List

import torch
from od3d.cv.geometry.transform import proj3d2d_broadcast
from pytorch3d.io import IO
from pytorch3d.io import load_ply
from pytorch3d.io import save_ply
from pytorch3d.renderer import MeshRasterizer
from pytorch3d.renderer import RasterizationSettings
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.mesh import TexturesUV as PT3DTexturesUV
from pytorch3d.renderer.mesh import TexturesVertex as PT3DTexturesVertex
from pytorch3d.renderer.mesh.utils import interpolate_face_attributes
from pytorch3d.structures import packed_to_list as pt3d_packed_to_list
from pytorch3d.structures.meshes import Meshes as PT3DMeshes

logger = logging.getLogger(__name__)
from dataclasses import dataclass
from od3d.cv.geometry.transform import (
    tform4x4,
    tform4x4_broadcast,
    inv_tform4x4,
    add_homog_dim,
    transf3d_broadcast,
)
from od3d.cv.visual.sample import sample_pxl2d_grid
from od3d.cv.geometry.grid import get_pxl2d
from typing import Union
import open3d as o3d
import numpy as np
from od3d.cv.geometry.objects3d.objects3d import (
    PROJECT_MODALITIES,
    OD3D_Objects3D,
    FEATS_DISTR,
)

from typing import Optional


class MESH_RENDER_MODALITIES(str, Enum):
    DEPTH = "depth"
    MASK = "mask"
    RGB = "rgb"
    RGBA = "rgba"
    FEATS = "feats"
    MASK_VERTS_VSBL = "mask_verts_vsbl"
    VERTS_NCDS = "verts_ncds"
    VERTS_ONEHOT = "verts_onehot"


class MESH_RENDER_MODALITIES_GAUSSIAN_SPLAT(str, Enum):
    RGB = MESH_RENDER_MODALITIES.RGB
    FEATS = MESH_RENDER_MODALITIES.FEATS
    VERTS_NCDS = MESH_RENDER_MODALITIES.VERTS_NCDS


class Mesh:
    def __init__(self, verts, faces, rgb=None, feats=None):
        self.verts = verts
        self.faces = faces
        self.rgb = rgb
        self.feats = feats
        self.device = verts.device

    @staticmethod
    def convert_to_textureVertex(
        textures_uv: PT3DTexturesUV,
        meshes: PT3DMeshes,
    ) -> PT3DTexturesVertex:
        # note: this is a workaround, since the model textures_uv contains multiple values per vertex, but textures_vertex only one
        verts_colors_packed = torch.zeros_like(meshes.verts_packed())
        verts_colors_packed[
            meshes.faces_packed()
        ] = textures_uv.faces_verts_textures_packed()  # (*)
        return PT3DTexturesVertex(
            pt3d_packed_to_list(verts_colors_packed, meshes.num_verts_per_mesh()),
        )

    @staticmethod
    def load_from_file(fpath: Path, device="cpu", scale=1.0):
        io = IO()
        mesh = io.load_mesh(fpath, device=device)
        verts = mesh[0].verts_list()[0] * scale
        faces = mesh[0].faces_list()[0]
        if mesh[0].textures is not None:
            verts_rgb = Mesh.convert_to_textureVertex(
                textures_uv=mesh[0].textures,
                meshes=mesh[0],
            ).verts_features_list()[0]
        else:
            verts_rgb = None
        return Mesh(verts=verts, faces=faces, rgb=verts_rgb)

    @staticmethod
    def load_from_file_ply(fpath: Path):
        verts, faces = load_ply(fpath)
        return Mesh(verts=verts, faces=faces)

    def write_to_file(self, fpath: Path):
        logger.info(f"writing mesh to {fpath}")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        save_ply(fpath, verts=self.verts, faces=self.faces)

    def verts_count(self):
        return self.verts.shape[0]

    @staticmethod
    def from_o3d(mesh_o3d: o3d.geometry.TriangleMesh, device="cpu"):
        vertices = torch.from_numpy(np.asarray(mesh_o3d.vertices)).to(
            dtype=torch.float,
            device=device,
        )
        faces = torch.from_numpy(np.asarray(mesh_o3d.triangles)).to(
            dtype=torch.long,
            device=device,
        )
        return Mesh(verts=vertices, faces=faces)

    # .TriangleMesh(vertices=vertices, triangles=triangles)

    def to_o3d(self):
        import open3d

        vertices = open3d.utility.Vector3dVector(self.verts.detach().cpu().numpy())
        faces = open3d.utility.Vector3iVector(self.faces.detach().cpu().numpy())
        o3d_obj_mesh = open3d.geometry.TriangleMesh(vertices=vertices, triangles=faces)
        return o3d_obj_mesh

    @staticmethod
    def create_sphere(
        center3d: torch.Tensor([0.0, 0.0, 0.0]),
        radius: float = 1.0,
        device="cpu",
    ):
        return Mesh.from_o3d(
            o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(
                center3d.detach().cpu().numpy(),
            ),
            device=device,
        )

    @property
    def verts_ncds(self):
        return (self.verts - self.verts.min(dim=0).values[None,]) / (
            self.verts.max(dim=0).values[None,] - self.verts.min(dim=0).values[None,]
        )

    @staticmethod
    def create_plane_as_cone(
        center3d: torch.Tensor = torch.Tensor([0.0, 0.0, 0.0]),
        radius: float = 1.0,
        height: float = 1.0,
        device="cpu",
    ):
        R = o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz((np.pi, 0.0, 0.0))
        # note: height becomes larger with lower resolution
        plane3d_open3d = (
            o3d.geometry.TriangleMesh.create_cone(
                radius=radius,
                height=height,
                resolution=100,
                split=1,
                create_uv_map=False,
            )
            .rotate(R=R, center=(0, 0, 0))
            .translate(center3d.detach().cpu().numpy())
        )
        plane3d_open3d.paint_uniform_color([0.2, 0.2, 0.4])
        return Mesh.from_o3d(plane3d_open3d, device=device)

    # TODO: create ray
    # ray_range = scene_size
    # ray = open3d.geometry.TriangleMesh.create_arrow(cylinder_radius=1.0 * particle_size,
    #                                                 cone_radius=1.5 * particle_size, cylinder_height=ray_range,
    #                                                 cone_height=4.0 * particle_size)
    # ray.transform(inv_tform4x4(cam_tform4x4_obj).detach().cpu().numpy())


class Meshes(OD3D_Objects3D):
    feats_objects: Optional[torch.Tensor]

    def __init__(
        self,
        verts: List[torch.Tensor],
        faces: List[torch.Tensor],
        feat_dim=128,
        objects_count=0,
        feats_objects=False,
        feats_requires_grad=True,
        feat_clutter=False,
        feats_distribution=FEATS_DISTR.VON_MISES_FISHER,
        rgb: List[torch.Tensor] = None,
        verts_requires_grad=False,
        geodesic_prob_sigma=0.2,
        gaussian_splat_enabled=False,
        gaussian_splat_opacity=0.7,
        gaussian_splat_pts3d_size_rel_to_neighbor_dist=0.5,
        pt3d_raster_perspective_correct=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__(
            feat_dim=feat_dim,
            feat_clutter=feat_clutter,
            feats_requires_grad=feats_requires_grad,
            feats_distribution=feats_distribution,
            device=device,
            dtype=dtype,
        )

        self.meshes_count = len(verts)
        self.verts = torch.nn.Parameter(
            torch.cat([_verts for _verts in verts], dim=0).to(**factory_kwargs),
            requires_grad=verts_requires_grad,
        )
        self.faces = torch.nn.Parameter(
            torch.cat([_faces for _faces in faces], dim=0).to(**factory_kwargs),
            requires_grad=False,
        )
        self.device = self.verts.device

        self.gaussian_splat_enabled = gaussian_splat_enabled
        self.gaussian_splat_opacity = gaussian_splat_opacity
        self.gaussian_splat_pts3d_size_rel_to_neighbor_dist = (
            gaussian_splat_pts3d_size_rel_to_neighbor_dist
        )
        self.pt3d_raster_perspective_correct = pt3d_raster_perspective_correct

        self.geodesic_prob_sigma = geodesic_prob_sigma
        self._geodesic_dist = None

        self.verts_counts = [_verts.shape[0] for _verts in verts]
        self.faces_counts = [_faces.shape[0] for _faces in faces]
        self.verts_counts_acc_from_0 = [0] + [
            sum(self.verts_counts[: i + 1]) for i in range(self.meshes_count)
        ]
        self.faces_counts_acc_from_0 = [0] + [
            sum(self.faces_counts[: i + 1]) for i in range(self.meshes_count)
        ]

        self.verts_count = self.verts_counts_acc_from_0[-1]
        self.verts_counts_max = max(self.verts_counts)
        self.faces_counts_max = max(self.faces_counts)

        self.mask_verts_not_padded = torch.ones(
            size=[len(self), self.verts_counts_max],
            dtype=torch.bool,
            device=self.verts.device,
        )

        for i in range(len(self)):
            self.mask_verts_not_padded[i, self.verts_counts[i] :] = False

        if rgb is not None:
            self.rgb = torch.nn.Parameter(
                torch.cat([_rgb for _rgb in rgb], dim=0),
                requires_grad=False,
            )
        else:
            self.register_parameter("rgb", None)

        if feats_objects:
            self.feats_objects = torch.nn.Parameter(
                torch.cat(
                    [
                        torch.empty(
                            size=[self.verts_counts[i], self.feat_dim],
                            **factory_kwargs,
                        )
                        for i in range(len(self))
                    ],
                    dim=0,
                ),
                requires_grad=feats_requires_grad,
            )
        else:
            self.register_parameter("feats_objects", None)

        self.init_pt3d()
        self.pre_rendered_feats = None
        self.pre_rendered_modalities = {}

        self.reset_parameters()

    def to_o3d(self):
        import open3d

        vertices = open3d.utility.Vector3dVector(self.verts.detach().cpu().numpy())
        faces = open3d.utility.Vector3iVector(self.faces.detach().cpu().numpy())
        o3d_obj_mesh = open3d.geometry.TriangleMesh(vertices=vertices, triangles=faces)

        if self.rgb is not None:
            vertex_colors = open3d.utility.Vector3dVector(
                self.rgb.detach().cpu().numpy(),
            )
            o3d_obj_mesh.vertex_colors = vertex_colors
        return o3d_obj_mesh

    @staticmethod
    def create_sphere(verts_count=1000, radius=1.0, device="cpu"):
        # verts_count = 2 * resolution * (resolution-1) + 2
        # (verts_count - 2) / 2  = resolution **2 - resolution
        # (verts_count - 2) / 2  <= resolution **2
        import math

        resolution = int(math.floor(math.sqrt((verts_count - 2.0) / 2.0)))
        o3d_mesh = o3d.geometry.TriangleMesh.create_sphere(
            radius=radius,
            resolution=resolution,
            create_uv_map=True,
        )
        return Meshes.from_o3d(o3d_mesh)

    @staticmethod
    def from_o3d(
        mesh_o3d: Union[List, o3d.geometry.TriangleMesh],
        device="cpu",
        **kwargs,
    ):
        if not isinstance(mesh_o3d, List):
            meshes_o3d = [mesh_o3d]
        else:
            meshes_o3d = mesh_o3d

        vertices = []
        faces = []
        for _mesh_o3d in meshes_o3d:
            vertices.append(
                torch.from_numpy(np.asarray(_mesh_o3d.vertices)).to(
                    dtype=torch.float,
                    device=device,
                ),
            )
            faces.append(
                torch.from_numpy(np.asarray(_mesh_o3d.triangles)).to(
                    dtype=torch.long,
                    device=device,
                ),
            )
        return Meshes(verts=vertices, faces=faces, **kwargs)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if self.feats_objects is not None:
            import math

            # equals kaiming uniform
            bound = 1 / math.sqrt(self.feat_dim) if self.feat_dim > 0 else 0
            # torch.nn.init.uniform_(self.feats_objects, a=-bound, b=bound)
            # torch.nn.init.uniform_(self.feats_objects, a=0., b=1.) # note: somehow better at least without head
            torch.nn.init.normal_(self.feats_objects)

        self.normalize_feats()

    def normalize_feats(self):
        super().normalize_feats()
        if self.feats_objects is not None:
            with torch.no_grad():
                self.feats_objects.copy_(
                    self.feats_objects
                    / (self.feats_objects.norm(dim=-1, keepdim=True) + 1e-10),
                )

    def get_limits(self):
        meshes_limits = []
        for i in range(len(self)):
            mesh_verts = self.get_verts_with_mesh_id(mesh_id=i)
            mesh_limits = torch.stack(
                [mesh_verts.min(dim=0)[0], mesh_verts.max(dim=0)[0]],
            )
            meshes_limits.append(mesh_limits)
        meshes_limits = torch.stack(meshes_limits, dim=0)
        return meshes_limits

    def get_ranges(self):
        meshes_limits = self.get_limits()
        meshes_range = meshes_limits[:, 1, :] - meshes_limits[:, 0, :]
        return meshes_range

    def init_pt3d(self):
        self.pt3dmeshes = PT3DMeshes(
            verts=[self.get_verts_with_mesh_id(i) for i in range(self.meshes_count)],
            faces=[self.get_faces_with_mesh_id(i) for i in range(self.meshes_count)],
        )

    def write_to_file(self, fpath: Path):
        fpath.parent.mkdir(parents=True, exist_ok=True)
        save_ply(fpath, verts=self.verts, faces=self.faces)

    @dataclass
    class PreRendered:
        cams_tform4x4_obj: torch.Tensor
        cams_intr4x4: torch.Tensor
        imgs_sizes: torch.Tensor
        broadcast_batch_and_cams: bool
        meshes_ids: torch.Tensor
        down_sample_rate: float
        rendering: torch.Tensor

    @staticmethod
    def read_from_ply_files(
        fpaths_meshes: List[Path],
        fpaths_meshes_tforms: List[Path] = None,
        feat_dim=128,
        objects_count=0,
        feats_objects=False,
        feat_clutter=False,
        feats_distribution=FEATS_DISTR.VON_MISES_FISHER,
        verts_requires_grad=False,
        geodesic_prob_sigma=0.2,
        gaussian_splat_enabled=False,
        gaussian_splat_opacity=0.7,
        feats_requires_grad=True,
        gaussian_splat_pts3d_size_rel_to_neighbor_dist=0.5,
        pt3d_raster_perspective_correct=False,
        device=None,
        dtype=None,
    ):
        meshes = []
        for i, fpath_mesh in enumerate(fpaths_meshes):
            mesh = Mesh.load_from_file(fpath=fpath_mesh, device=device)
            if fpaths_meshes_tforms is not None and fpaths_meshes_tforms[i] is not None:
                mesh_tform = torch.load(fpaths_meshes_tforms[i]).to(device)
                mesh.verts = transf3d_broadcast(pts3d=mesh.verts, transf4x4=mesh_tform)
            meshes.append(mesh)
        return Meshes.read_from_meshes(
            meshes=meshes,
            feat_dim=feat_dim,
            objects_count=objects_count,
            feats_objects=feats_objects,
            feat_clutter=feat_clutter,
            feats_distribution=feats_distribution,
            verts_requires_grad=verts_requires_grad,
            geodesic_prob_sigma=geodesic_prob_sigma,
            gaussian_splat_enabled=gaussian_splat_enabled,
            gaussian_splat_opacity=gaussian_splat_opacity,
            gaussian_splat_pts3d_size_rel_to_neighbor_dist=gaussian_splat_pts3d_size_rel_to_neighbor_dist,
            pt3d_raster_perspective_correct=pt3d_raster_perspective_correct,
            device=device,
            dtype=dtype,
            feats_requires_grad=feats_requires_grad,
        )

    @staticmethod
    def read_from_meshes(
        meshes: List[Mesh],
        feat_dim=128,
        objects_count=0,
        feats_objects=False,
        feat_clutter=False,
        feats_distribution=FEATS_DISTR.VON_MISES_FISHER,
        verts_requires_grad=False,
        feats_requires_grad=True,
        geodesic_prob_sigma=0.2,
        gaussian_splat_enabled=False,
        gaussian_splat_opacity=0.7,
        gaussian_splat_pts3d_size_rel_to_neighbor_dist=0.5,
        pt3d_raster_perspective_correct=False,
        device=None,
        dtype=None,
    ):
        if device is None:
            device = meshes[0].verts.device
        verts = [mesh.verts.to(device=device) for mesh in meshes]
        faces = [mesh.faces.to(device=device) for mesh in meshes]

        if meshes[0].rgb is not None:
            rgb = [mesh.rgb.to(device=device) for mesh in meshes]
        else:
            rgb = None
        return Meshes(
            verts=verts,
            faces=faces,
            rgb=rgb,
            feat_dim=feat_dim,
            objects_count=objects_count,
            feats_objects=feats_objects,
            feat_clutter=feat_clutter,
            feats_distribution=feats_distribution,
            verts_requires_grad=verts_requires_grad,
            geodesic_prob_sigma=geodesic_prob_sigma,
            gaussian_splat_enabled=gaussian_splat_enabled,
            gaussian_splat_opacity=gaussian_splat_opacity,
            gaussian_splat_pts3d_size_rel_to_neighbor_dist=gaussian_splat_pts3d_size_rel_to_neighbor_dist,
            pt3d_raster_perspective_correct=pt3d_raster_perspective_correct,
            device=device,
            dtype=dtype,
            feats_requires_grad=feats_requires_grad,
        )

    @staticmethod
    def load_by_name(name: str, device="cpu", faces_count=None):
        if name == "bunny":
            bunny_data = o3d.data.BunnyMesh()
            bunny_mesh_open3d = o3d.io.read_triangle_mesh(bunny_data.path)
            if faces_count is not None:
                bunny_mesh_open3d = bunny_mesh_open3d.simplify_quadric_decimation(
                    faces_count,
                )
            bunny_mesh = Meshes.read_from_meshes(
                [Mesh.from_o3d(bunny_mesh_open3d, device=device)],
            )
            bunny_rot = torch.Tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ).to(device=device)
            bunny_mesh.verts.data = transf3d_broadcast(
                pts3d=bunny_mesh.verts,
                transf4x4=bunny_rot,
            )
            return bunny_mesh
        elif name == "cuboid":
            from od3d.cv.geometry.primitives import Cuboids

            cuboids = Cuboids.create_dense_from_limits(
                limits=torch.Tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]]),
                device=device,
            )
            return cuboids

        else:
            raise ValueError(f"Unknown mesh name: {name}")

    def __add__(self, meshes2):
        meshes = []
        for mesh_id in list(range(len(self))):
            meshes.append(self.get_meshes_with_ids(meshes_ids=[mesh_id]))
        for mesh_id in list(range(len(meshes2))):
            meshes.append(meshes2.get_meshes_with_ids(meshes_ids=[mesh_id]))
        return Meshes.read_from_meshes(meshes=meshes, device=self.device)

    @staticmethod
    def get_faces_from_verts(verts, ball_radius=0.3):
        import open3d
        import numpy as np
        from od3d.cv.geometry.transform import (
            transf3d_broadcast,
            transf4x4_from_spherical,
        )

        verts_rot = transf3d_broadcast(
            pts3d=verts,
            transf4x4=transf4x4_from_spherical(
                azim=torch.Tensor([0.05]),
                elev=torch.Tensor([0.05]),
                theta=torch.Tensor([0.05]),
                dist=torch.Tensor([1.0]),
            ),
        )
        verts_centered = verts_rot - verts_rot.mean(dim=-2, keepdim=True)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(verts_centered)
        pcd.normals = open3d.utility.Vector3dVector(verts_centered)
        pcd.estimate_normals()
        # mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd)
        mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd=pcd,
            radii=open3d.utility.DoubleVector([ball_radius]),
        )
        faces = torch.from_numpy(np.asarray(mesh.triangles))
        return faces

    def __len__(self):
        return self.meshes_count

    def _apply(self, fn):
        super()._apply(fn)
        self.init_pt3d()

    def get_mesh_with_id(self, mesh_id):
        verts = self.get_verts_with_mesh_id(mesh_id)
        faces = self.get_faces_with_mesh_id(mesh_id)

        if self.rgb is None:
            rgb = None
        else:
            rgb = self.get_rgb_with_mesh_id(mesh_id)

        if self.feats_objects is None:
            feats_objects = None
        else:
            feats_objects = self.get_feats_with_mesh_id(mesh_id=mesh_id)
        return Mesh(verts=verts, faces=faces, feats=feats_objects, rgb=rgb)

    def get_meshes_with_ids(self, meshes_ids=None, clone=False):
        if meshes_ids is None:
            meshes_ids = list(range(len(self)))

        verts = [
            self.get_verts_with_mesh_id(mesh_id=mesh_id, clone=clone)
            for mesh_id in meshes_ids
        ]
        faces = [
            self.get_faces_with_mesh_id(mesh_id=mesh_id, clone=clone)
            for mesh_id in meshes_ids
        ]

        if self.rgb is None:
            rgb = None
        else:
            rgb = [
                self.get_rgb_with_mesh_id(mesh_id=mesh_id, clone=clone)
                for mesh_id in meshes_ids
            ]
        if self.feats is None:
            feats = None
        else:
            feats = [
                self.get_feats_with_mesh_id(mesh_id=mesh_id, clone=clone)
                for mesh_id in meshes_ids
            ]
        return Meshes(verts=verts, faces=faces, rgb=rgb, feats=feats)

    def get_geodesic_dist(self):
        if self._geodesic_dist is None:
            import gdist

            meshes_ids = list(range(len(self)))
            geodesic_dist = (
                torch.ones(
                    size=(self.verts_count, self.verts_count),
                    device=self.device,
                )
                * torch.inf
            )
            for mesh_id in meshes_ids:
                verts = self.get_verts_with_mesh_id(mesh_id=mesh_id)
                faces = self.get_faces_with_mesh_id(mesh_id=mesh_id)
                mesh_verts_geodesic_dist = gdist.local_gdist_matrix(
                    vertices=verts.detach().cpu().to(torch.float64).numpy(),
                    triangles=faces.cpu().detach().to(torch.int32).numpy(),
                    max_distance=99999,
                )
                # convert  scipy.sparse._csc.csc_matrix to torch.Tensor, fill sparse with torch.inf
                mesh_verts_geodesic_dist = torch.from_numpy(
                    mesh_verts_geodesic_dist.toarray(),
                ).to(
                    dtype=torch.float32,
                    device=self.device,
                )
                mesh_verts_geodesic_dist = (
                    mesh_verts_geodesic_dist / mesh_verts_geodesic_dist.max()
                )
                mesh_verts_geodesic_dist[mesh_verts_geodesic_dist == 0] = torch.inf
                mesh_verts_geodesic_dist[
                    torch.arange(mesh_verts_geodesic_dist.shape[0]).to(self.device),
                    torch.arange(
                        mesh_verts_geodesic_dist.shape[0],
                    ).to(self.device),
                ] = 0.0

                # mesh_verts_geodesic_dist = torch.from_numpy(mesh_verts_geodesic_dist.toarray()).to(dtype=torch.float32, device=self.device)

                geodesic_dist[
                    self.verts_counts_acc_from_0[
                        mesh_id
                    ] : self.verts_counts_acc_from_0[mesh_id + 1],
                    self.verts_counts_acc_from_0[
                        mesh_id
                    ] : self.verts_counts_acc_from_0[mesh_id + 1],
                ] = mesh_verts_geodesic_dist

            # euclidean_dist = torch.cdist(self.verts[None,], self.verts[None,])[0]
            # geodesic_dist = torch.ones_like(euclidean_dist) * torch.inf
            # would need to get verts nearest neighbors and iteratively propagating geodesic_dist
            self._geodesic_dist = geodesic_dist

        return self._geodesic_dist

    def get_geodesic_prob(self):
        _geodesic_dist = self.get_geodesic_dist().clone()
        _geodesic_prob = torch.exp(
            input=-0.5 * (_geodesic_dist / (self.geodesic_prob_sigma + 1e-10)) ** 2,
        )
        # replace inf with 0
        _geodesic_prob[torch.isinf(_geodesic_dist)] = 0.0
        _geodesic_prob = _geodesic_prob / _geodesic_prob.sum(dim=-1, keepdim=True)
        return _geodesic_prob

    def get_geodesic_prob_with_noise(self):
        geodesic_prob_with_noise = torch.eye(
            self.verts_count + 1,
            device=self.device,
        )
        geodesic_prob_with_noise[:-1, :-1] = self.get_geodesic_prob()
        return geodesic_prob_with_noise

    def get_verts_ncds_with_mesh_id(self, mesh_id):
        verts3d = self.get_verts_with_mesh_id(mesh_id)
        verts3d_ncds = (verts3d - verts3d.min(dim=0).values[None,]) / (
            verts3d.max(dim=0).values[None,] - verts3d.min(dim=0).values[None,]
        )
        return verts3d_ncds

    def get_verts_ncds_cat_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        verts3d_ncds = []
        for mesh_id in mesh_ids:
            verts3d_ncds.append(self.get_verts_ncds_with_mesh_id(mesh_id=mesh_id))
        verts3d_ncds = torch.cat(verts3d_ncds, dim=0)
        return verts3d_ncds

    def get_verts_cat_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        verts3d = []
        for mesh_id in mesh_ids:
            verts3d.append(self.get_verts_with_mesh_id(mesh_id=mesh_id))
        verts3d = torch.cat(verts3d, dim=0)
        return verts3d

    def get_faces_cat_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        faces = []
        for mesh_id in mesh_ids:
            faces.append(self.get_faces_with_mesh_id(mesh_id=mesh_id))
        faces = torch.cat(faces, dim=0)
        return faces

    def get_verts_ncds_from_faces_with_mesh_id(self, mesh_id):
        verts3d_ncds = self.get_verts_ncds_with_mesh_id(mesh_id)
        feats_from_faces = verts3d_ncds[self.get_faces_with_mesh_id(mesh_id)]
        return feats_from_faces

    def get_feats_from_faces_with_mesh_id(self, mesh_id):
        # return self.feats_from_faces[self.faces_counts_acc_from_0[mesh_id]: self.faces_counts_acc_from_0[mesh_id+1]]
        return self.get_feats_with_mesh_id(mesh_id)[
            self.get_faces_with_mesh_id(mesh_id)
        ]

    def get_rgb_with_mesh_id(self, mesh_id, clone=False):
        if not clone:
            return self.rgb[
                self.verts_counts_acc_from_0[mesh_id] : self.verts_counts_acc_from_0[
                    mesh_id + 1
                ]
            ]
        else:
            return self.rgb[
                self.verts_counts_acc_from_0[mesh_id] : self.verts_counts_acc_from_0[
                    mesh_id + 1
                ]
            ].clone()

    def get_verts_with_mesh_id(self, mesh_id, clone=False):
        if not clone:
            return self.verts[
                self.verts_counts_acc_from_0[mesh_id] : self.verts_counts_acc_from_0[
                    mesh_id + 1
                ]
            ]
        else:
            return self.verts[
                self.verts_counts_acc_from_0[mesh_id] : self.verts_counts_acc_from_0[
                    mesh_id + 1
                ]
            ].clone()

    def get_mesh_ids_for_verts(self):
        mesh_ids = torch.LongTensor(size=(0,)).to(device=self.device)
        for mesh_id in range(self.meshes_count):
            mesh_ids = torch.cat(
                [
                    mesh_ids,
                    torch.LongTensor([mesh_id] * self.verts_counts[mesh_id]).to(
                        device=self.device,
                    ),
                ],
                dim=0,
            )
        return mesh_ids

    def get_feats_with_mesh_id(self, mesh_id, clone=False):
        if not clone:
            return self.feats_objects[
                self.verts_counts_acc_from_0[mesh_id] : self.verts_counts_acc_from_0[
                    mesh_id + 1
                ]
            ]
        else:
            return self.feats_objects[
                self.verts_counts_acc_from_0[mesh_id] : self.verts_counts_acc_from_0[
                    mesh_id + 1
                ]
            ].clone()

    def get_faces_with_mesh_id(self, mesh_id, clone=False):
        if not clone:
            return self.faces[
                self.faces_counts_acc_from_0[mesh_id] : self.faces_counts_acc_from_0[
                    mesh_id + 1
                ]
            ]
        else:
            return self.faces[
                self.faces_counts_acc_from_0[mesh_id] : self.faces_counts_acc_from_0[
                    mesh_id + 1
                ]
            ].clone()

    def get_faces_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_faces_with_pad(
            tensor=self.get_faces_with_mesh_id(mesh_id),
            mesh_id=mesh_id,
        )

    def get_tensor_verts_with_pad(self, tensor, mesh_id):
        pad = torch.Size([self.verts_counts_max - self.verts_counts[mesh_id]])
        return torch.cat(
            [
                tensor,
                torch.zeros(
                    size=pad + tensor.shape[1:],
                    dtype=tensor.dtype,
                    device=tensor.device,
                ),
            ],
            dim=0,
        )

    def get_tensor_faces_with_pad(self, tensor, mesh_id):
        pad = torch.Size([self.faces_counts_max - self.faces_counts[mesh_id]])
        return torch.cat(
            [
                tensor,
                torch.zeros(
                    size=pad + tensor.shape[1:],
                    dtype=tensor.dtype,
                    device=tensor.device,
                ),
            ],
            dim=0,
        )

    def get_feats_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_verts_with_pad(
            tensor=self.get_feats_with_mesh_id(mesh_id),
            mesh_id=mesh_id,
        )

    def get_verts_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_verts_with_pad(
            tensor=self.get_verts_with_mesh_id(mesh_id),
            mesh_id=mesh_id,
        )

    def get_verts_ncds_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_verts_with_pad(
            tensor=self.get_verts_ncds_with_mesh_id(mesh_id),
            mesh_id=mesh_id,
        )

    def get_rgb_padded_with_mesh_id(self, mesh_id):
        return self.get_tensor_verts_with_pad(
            tensor=self.get_rgb_with_mesh_id(mesh_id),
            mesh_id=mesh_id,
        )

    # get_rgb_padded_with_mesh_id
    # def to(self, device):
    #    if self.device != device:
    #        self.verts = [v.to(device=device) for v in self.verts]
    #        self.faces = [f.to(device=device) for f in self.faces]
    #        if self.rgb is not None:
    #            self.rgb = [i.to(device=device) for i in self.rgb]
    #        if self.feats is not None:
    #            self.feats = [f.to(device=device) for f in self.feats]
    #            self.feats_from_faces = [self.feats[mesh_id][self.faces[mesh_id]] for mesh_id in range(len(self))]

    #           self.device = device
    def set_feats_cat_with_pad(self, feats):
        vts_ct_max = self.verts_counts_max
        device = self.verts.device
        self.feats = torch.nn.Parameter(
            torch.cat(
                [
                    feats[i * vts_ct_max : i * vts_ct_max + self.verts_counts[i]].to(
                        device=device,
                    )
                    for i in range(len(self))
                ],
                dim=0,
            ),
            requires_grad=True,
        )
        self.feats_from_faces = torch.nn.Parameter(
            torch.cat(
                [
                    self.get_feats_with_mesh_id(mesh_id)[
                        self.get_faces_with_mesh_id(mesh_id)
                    ]
                    for mesh_id in range(len(self))
                ],
                dim=0,
            ),
        )

    def set_feats_cat(self, feats):
        self.feats = torch.nn.Parameter(feats, requires_grad=True)
        self.feats_from_faces = torch.nn.Parameter(
            torch.cat(
                [
                    self.get_feats_with_mesh_id(mesh_id)[
                        self.get_faces_with_mesh_id(mesh_id)
                    ]
                    for mesh_id in range(len(self))
                ],
                dim=0,
            ),
        )

    def get_verts_ncds_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        return torch.stack(
            [self.get_verts_ncds_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids],
            dim=0,
        )

    def get_rgb_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        return torch.stack(
            [self.get_rgb_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids],
            dim=0,
        )

    def get_verts_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        return torch.stack(
            [self.get_verts_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids],
            dim=0,
        )

    def get_feats_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        return torch.stack(
            [self.get_feats_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids],
            dim=0,
        )

    def get_faces_stacked_with_mesh_ids(self, mesh_ids=None):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))
        return torch.stack(
            [self.get_faces_padded_with_mesh_id(mesh_id) for mesh_id in mesh_ids],
            dim=0,
        )

    # def add_feats_cat(self, feats):
    #    raise Not I
    #    self.feats = [feats[self.verts_counts_acc_from_0[i] : self.verts_counts_acc_from_0[i+1]].to(device=self.device) for i in range(len(self))]
    #    self.feats_from_faces = [self.feats[mesh_id][self.faces[mesh_id]] for mesh_id in range(len(self))]

    # def add_feats
    # def verts_stacked(self, mesh_ids: list=None):
    #    if mesh_ids is None:
    #        mesh_ids = list(range(len(self)))

    # verts_stacked = torch.zeros(size=(len(mesh_ids), self.verts_counts_max(), 3), device=self.device)
    # for mesh_id in mesh_ids:
    #    verts_stacked[mesh_id, : len(self.verts[mesh_id])] = self.verts[mesh_id]

    #    return torch.stack([self.verts[mesh_id] for mesh_id in mesh_ids], dim=0)

    # def feats_cat(self, mesh_ids: list=None):
    #    if mesh_ids is None:
    #        mesh_ids = list(range(len(self)))
    #
    #    return torch.cat([self.feats[mesh_id] for mesh_id in mesh_ids], dim=0)

    # def get_feats_ids_stacked(self, mesh_ids: list=None):
    #    if mesh_ids is None:
    #        mesh_ids = list(range(len(self)))

    #    device = self.verts.device
    #    return torch.stack([torch.arange(mesh_id*self.verts_counts_max, (mesh_id+1)*self.verts_counts_max, device=device) for mesh_id in mesh_ids], dim=0)

    # def get_verts_and_noise_ids_cat(self, mesh_ids: list=None, count_noise_ids=5):
    #   if mesh_ids is None:
    #        mesh_ids = list(range(len(self)))

    #    device = self.verts.device
    #    noise_ids = torch.ones(size=(count_noise_ids,), dtype=torch.long, device=device) * self.verts_counts_acc_from_0[-1]
    #    verts_ids = [torch.arange(self.verts_counts_acc_from_0[mesh_id], self.verts_counts_acc_from_0[mesh_id+1], device=device) for mesh_id in mesh_ids]
    #    return torch.cat([torch.cat([verts_ids[i], noise_ids], dim=0) for i in range(len(mesh_ids))], dim=0)

    def get_verts_and_noise_ids_stacked(self, mesh_ids: list = None, count_noise_ids=5):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))

        device = self.verts.device
        noise_ids = (
            torch.ones(size=(count_noise_ids,), dtype=torch.long, device=device)
            * self.verts_counts_acc_from_0[-1]
        )
        verts_ids = [
            torch.arange(
                self.verts_counts_acc_from_0[mesh_id],
                self.verts_counts_acc_from_0[mesh_id] + self.verts_counts_max,
                device=device,
            )
            for mesh_id in mesh_ids
        ]
        return torch.stack(
            [torch.cat([verts_ids[i], noise_ids], dim=0) for i in range(len(mesh_ids))],
            dim=0,
        )

    def get_verts_and_noise_ids_stacked_without_acc(
        self,
        mesh_ids: list = None,
        count_noise_ids=5,
    ):
        if mesh_ids is None:
            mesh_ids = list(range(len(self)))

        device = self.verts.device
        noise_ids = (
            torch.ones(size=(count_noise_ids,), dtype=torch.long, device=device)
            * self.verts_counts_max
        )
        verts_ids = [
            torch.arange(0, self.verts_counts_max, device=device)
            for mesh_id in mesh_ids
        ]
        return torch.stack(
            [torch.cat([verts_ids[i], noise_ids], dim=0) for i in range(len(mesh_ids))],
            dim=0,
        )

    def normals3d(self, meshes_ids: Union[torch.LongTensor, List] = None):
        """
        Args:
            meshes_ids (Union[torch.LongTensor, List]): len(mesh_ids) == B
        Returns:
            normals3d (torch.Tensor): BxNx3
        """

        if isinstance(meshes_ids, List):
            meshes_ids = torch.LongTensor(meshes_ids)

        if isinstance(meshes_ids, torch.LongTensor):
            meshes_ids = meshes_ids.clone()

        pt3dmeshes = self.pt3dmeshes[meshes_ids]
        return pt3dmeshes.verts_normals_padded()

    def verts2d(
        self,
        cams_tform4x4_obj,
        cams_intr4x4,
        imgs_sizes,
        mesh_ids: Union[torch.LongTensor, List],
        down_sample_rate=1.0,
        broadcast_batch_and_cams=False,
    ):
        """
        Args:
            cams_tform4x4_obj (torch.Tensor): Bx4x4
            cams_intr4x4 (torch.Tensor): Bx4x4
            imgs_sizes (torch.Tensor): Bx2 / 2 (height, width)
            mesh_ids (list): len(mesh_ids) == B
        Returns:
            verts2d (torch.Tensor): BxNx2
        """
        # meshes_count = mesh_ids.shape[0]
        # cams_count = cams_tform4x4_obj.shape[0]

        if isinstance(mesh_ids, List):
            mesh_ids = torch.LongTensor(mesh_ids)

        if isinstance(mesh_ids, torch.LongTensor):
            mesh_ids = mesh_ids.clone()

        meshes_count = mesh_ids.shape[0]
        if cams_tform4x4_obj.dim() == 4:
            cams_count = cams_tform4x4_obj.shape[1]
        elif cams_tform4x4_obj.dim() == 3:
            cams_count = cams_tform4x4_obj.shape[0]
        else:
            raise ValueError(f"Set `cams_tform4x4_obj.dim()` must be 3 or 4")

        if broadcast_batch_and_cams:
            mesh_ids = mesh_ids
            if cams_tform4x4_obj.dim() == 3:
                cams_tform4x4_obj = cams_tform4x4_obj[None, :]
            if cams_intr4x4.dim() == 3:
                cams_intr4x4 = cams_intr4x4[None, :]
            cams_tform4x4_obj = cams_tform4x4_obj.expand(
                meshes_count,
                cams_count,
                4,
                4,
            ).reshape(-1, 4, 4)
            cams_intr4x4 = cams_intr4x4.expand(meshes_count, cams_count, 4, 4).reshape(
                -1,
                4,
                4,
            )
            mesh_ids = mesh_ids[:, None].expand(meshes_count, cams_count).reshape(-1)
            # render_count = meshes_count * cams_count
        else:
            if meshes_count != cams_count:
                raise ValueError(
                    f"Set `broadcast_batch_and_cams=True` to allow different number of cameras and meshes",
                )

        B = cams_tform4x4_obj.shape[0]
        # if imgs_sizes.dim() == 2:
        #    imgs_sizes = imgs_sizes[None,].expand(B, imgs_sizes.shape[0], imgs_sizes.shape[1])
        print(cams_intr4x4.dtype, cams_tform4x4_obj.dtype)
        cams_tform4x4_obj = cams_tform4x4_obj.to(dtype=cams_intr4x4.dtype, device=cams_intr4x4.device)
        print(cams_intr4x4.dtype, cams_tform4x4_obj.dtype)
        cams_proj4x4_obj = torch.bmm(cams_intr4x4, cams_tform4x4_obj)
        verts3d = self.get_verts_stacked_with_mesh_ids(mesh_ids=mesh_ids)
        verts2d = proj3d2d_broadcast(verts3d, proj4x4=cams_proj4x4_obj[:, None])
        mask_verts_vsbl = self.render_feats(
            cams_tform4x4_obj=cams_tform4x4_obj,
            cams_intr4x4=cams_intr4x4,
            imgs_sizes=imgs_sizes,
            meshes_ids=mesh_ids,
            modality=MESH_RENDER_MODALITIES.MASK_VERTS_VSBL,
            down_sample_rate=down_sample_rate,
        )
        mask_verts_vsbl *= (verts2d <= (imgs_sizes[None, None] - 1)).all(dim=-1)
        mask_verts_vsbl *= (verts2d >= 0).all(dim=-1)

        # Fill masking areas with zero
        verts2d[~mask_verts_vsbl] = 0
        # verts2d.clamp()

        if broadcast_batch_and_cams:
            verts2d = verts2d.reshape(meshes_count, cams_count, *verts2d.shape[1:])
            mask_verts_vsbl = mask_verts_vsbl.reshape(
                meshes_count,
                cams_count,
                *mask_verts_vsbl.shape[1:],
            )

        verts2d /= down_sample_rate

        return verts2d, mask_verts_vsbl

    def show(
        self,
        fpath: Path = None,
        return_visualization=False,
        viewpoints_count=1,
        meshes_add_translation=True,
    ):
        from od3d.cv.visual.show import show_scene

        return show_scene(
            meshes=self,
            fpath=fpath,
            return_visualization=return_visualization,
            viewpoints_count=viewpoints_count,
            meshes_add_translation=meshes_add_translation,
        )

    """
    def show(self, pts3d=[], meshes_ids=None):
        from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs
        from pytorch3d.structures import Pointclouds
        if meshes_ids is None:
            meshes_ids = list(range(len(self.pt3dmeshes)))
        pcls = Pointclouds(points=pts3d)
        verts = Pointclouds(points=self.get_verts_stacked_with_mesh_ids(mesh_ids=meshes_ids))
        fig = plot_scene({
            "Meshes":
                {
                    # **{f"mesh{i + 1}": self.pt3dmeshes[i] for i in meshes_ids},
                    **{f"verts{i + 1}": verts[i] for i in meshes_ids},
                    **{f"pcl{i + 1}": pcls[i] for i in range(len(pcls))}
                }
        }, axis_args=AxisArgs(backgroundcolor="rgb(200, 200, 230)", showgrid=True, zeroline=True, showline=True,
                                   showaxeslabels=True, showticklabels=True))
        fig.show()
        input('bla')
    """

    def del_pre_rendered(self):
        self.pre_rendered_modalities.clear()
        torch.cuda.empty_cache()

    def visualize(self, pcl=None):
        from od3d.cv.visual.show import show_imgs
        import numpy as np

        device = self.verts.device
        cxy = 250.0
        fxy = 500.0
        down_sample_rate = 2.0
        imgs_sizes = torch.LongTensor([512, 512]).to(device=device)

        azim = torch.linspace(
            start=eval("-np.pi / 2"),
            end=eval("np.pi / 2"),
            steps=5,
        ).to(
            device=device,
        )  # 12
        elev = torch.linspace(
            start=eval("np.pi / 4"),
            end=eval("np.pi / 4"),
            steps=1,
        ).to(
            device=device,
        )  # start=-torch.pi / 6, end=torch.pi / 3, steps=4
        theta = torch.linspace(
            start=eval("0."),
            end=eval("0."),
            steps=1,
        ).to(
            device=device,
        )  # -torch.pi / 6, end=torch.pi / 6, steps=3

        # dist = torch.linspace(start=eval(config_sample.uniform.dist.min), end=eval(config_sample.uniform.dist.max), steps=config_sample.uniform.dist.steps).to(
        #    device=self.device)
        dist = torch.linspace(start=1.0, end=1.0, steps=1).to(device=device)

        azim_shape = azim.shape
        elev_shape = elev.shape
        theta_shape = theta.shape
        dist_shape = dist.shape
        in_shape = azim_shape + elev_shape + theta_shape + dist_shape
        azim = azim[:, None, None, None].expand(in_shape).reshape(-1)
        elev = elev[None, :, None, None].expand(in_shape).reshape(-1)
        theta = theta[None, None, :, None].expand(in_shape).reshape(-1)
        dist = dist[None, None, None, :].expand(in_shape).reshape(-1)
        from od3d.cv.geometry.transform import transf4x4_from_spherical

        cams_tform4x4_obj = transf4x4_from_spherical(
            azim=azim,
            elev=elev,
            theta=theta,
            dist=dist,
        )

        T = cams_tform4x4_obj.shape[0]
        M = len(self)
        # M x T x 4 x 4
        pre_rendered_cams_tform4x4_obj = (
            cams_tform4x4_obj[None, :]
            .expand(
                M,
                T,
                *cams_tform4x4_obj[0].shape,
            )
            .clone()
        )
        pre_rendered_cams_tform4x4_obj[:, :, :3, 3] = 0.0
        pre_rendered_meshes_size = (
            self.get_verts_stacked_with_mesh_ids().flatten(1).max(dim=-1)[0]
        )
        pre_rendered_meshes_dist = (pre_rendered_meshes_size * fxy) / (
            500.0 * 0.8 - cxy
        )  # u = (x / z) * fx + cx  -> z = (fx * x) / (u - cx)
        pre_rendered_meshes_dist = pre_rendered_meshes_dist[:, None].expand(M, T)
        pre_rendered_cams_tform4x4_obj[:, :, 2, 3] = pre_rendered_meshes_dist

        # 1 x 1 x 4 x 4
        pre_rendered_cams_intr4x4 = (
            torch.eye(4)[None, None].to(device=device).expand(M, 1, 4, 4)
        )
        pre_rendered_cams_intr4x4[:, :, 0, 0] = fxy
        pre_rendered_cams_intr4x4[:, :, 1, 1] = fxy
        pre_rendered_cams_intr4x4[:, :, :2, 2] = cxy

        pre_rendered_meshes_ids = torch.arange(M).to(device=device)
        rendering = []
        for m in range(M):
            rendering.append(
                self.render_feats(
                    cams_tform4x4_obj=pre_rendered_cams_tform4x4_obj[m : m + 1],
                    cams_intr4x4=pre_rendered_cams_intr4x4[m : m + 1],
                    imgs_sizes=imgs_sizes,
                    meshes_ids=pre_rendered_meshes_ids[m : m + 1],
                    modality=MESH_RENDER_MODALITIES.VERTS_NCDS,
                    broadcast_batch_and_cams=True,
                    down_sample_rate=down_sample_rate,
                ),
            )
        rendering = torch.cat(rendering, dim=0)

        if pcl is not None:
            pxl2d_pre_rendered = (
                proj3d2d_broadcast(
                    pts3d=pcl[:, None, None],
                    proj4x4=tform4x4_broadcast(
                        pre_rendered_cams_intr4x4,
                        pre_rendered_cams_tform4x4_obj,
                    ),
                )
                / down_sample_rate
            )
            from od3d.cv.visual.mask import mask_from_pxl2d

            pxl2d_mask = mask_from_pxl2d(
                pxl2d=pxl2d_pre_rendered,
                dim_pxl=3,
                dim_pts=0,
                H=int(imgs_sizes[0] // down_sample_rate),
                W=int(imgs_sizes[1] // down_sample_rate),
            )
            rendering[pxl2d_mask[:, :, None, :, :].repeat(1, 1, 3, 1, 1)] = 1.0
            # rendering[:, :, :, ]
            # sample_pxl2d_grid(rendering.reshape(-1, C, H, W),
            #                  pxl2d=pxl2d_pre_rendered.reshape(-1, H, W, 2)).reshape(B, T, C, H, W)
        return show_imgs(rendering)

    def get_pre_rendered_feats(
        self,
        modality: MESH_RENDER_MODALITIES,
        cams_tform4x4_obj,
        cams_intr4x4,
        imgs_sizes,
        meshes_ids=None,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
    ):
        if modality not in self.pre_rendered_modalities.keys():
            cxy = 250.0
            fxy = 500.0
            T = cams_tform4x4_obj.shape[1]
            M = len(self)
            # M x T x 4 x 4
            pre_rendered_cams_tform4x4_obj = (
                cams_tform4x4_obj[:1, :].expand(M, *cams_tform4x4_obj[0].shape).clone()
            )
            pre_rendered_cams_tform4x4_obj[:, :, :3, 3] = 0.0
            pre_rendered_meshes_size = (
                self.get_verts_stacked_with_mesh_ids().flatten(1).max(dim=-1)[0]
            )
            pre_rendered_meshes_dist = (pre_rendered_meshes_size * fxy) / (
                500.0 * 0.7 - cxy
            )  # u = (x / z) * fx + cx  -> z = (fx * x) / (u - cx)
            pre_rendered_meshes_dist = pre_rendered_meshes_dist[:, None].expand(M, T)
            pre_rendered_cams_tform4x4_obj[:, :, 2, 3] = pre_rendered_meshes_dist

            # 1 x 1 x 4 x 4
            pre_rendered_cams_intr4x4 = (
                cams_intr4x4[:1, :1].clone().expand(M, 1, *cams_intr4x4[0, 0].shape)
            )
            pre_rendered_cams_intr4x4[:, :, 0, 0] = fxy
            pre_rendered_cams_intr4x4[:, :, 1, 1] = fxy
            pre_rendered_cams_intr4x4[:, :, :2, 2] = cxy

            pre_rendered_meshes_ids = torch.arange(M).to(device=meshes_ids.device)

            rendering = []
            for m in range(M):
                rendering.append(
                    self.render_feats(
                        cams_tform4x4_obj=pre_rendered_cams_tform4x4_obj[m : m + 1],
                        cams_intr4x4=pre_rendered_cams_intr4x4[m : m + 1],
                        imgs_sizes=imgs_sizes,
                        meshes_ids=pre_rendered_meshes_ids[m : m + 1],
                        modality=modality,
                        broadcast_batch_and_cams=broadcast_batch_and_cams,
                        down_sample_rate=down_sample_rate,
                    ),
                )
            rendering = torch.cat(rendering, dim=0)

            self.pre_rendered_modalities[modality] = Meshes.PreRendered(
                cams_tform4x4_obj=pre_rendered_cams_tform4x4_obj,
                cams_intr4x4=pre_rendered_cams_intr4x4,
                imgs_sizes=imgs_sizes,
                broadcast_batch_and_cams=broadcast_batch_and_cams,
                meshes_ids=pre_rendered_meshes_ids,
                down_sample_rate=down_sample_rate,
                rendering=rendering,
            )

        else:
            assert (
                self.pre_rendered_modalities[modality].cams_tform4x4_obj[
                    meshes_ids,
                    :,
                    :3,
                    :3,
                ]
                == cams_tform4x4_obj[:, :, :3, :3]
            ).all()
            assert (
                self.pre_rendered_modalities[modality].imgs_sizes == imgs_sizes
            ).all()
            assert (
                self.pre_rendered_modalities[modality].broadcast_batch_and_cams
                == broadcast_batch_and_cams
            )
            assert (
                self.pre_rendered_modalities[modality].down_sample_rate
                == down_sample_rate
            )

        pre_rendered_feats = self.pre_rendered_modalities[modality].rendering[
            meshes_ids
        ]
        pre_rendered_cam_intr4x4 = self.pre_rendered_modalities[modality].cams_intr4x4[
            meshes_ids
        ]
        pre_rendered_cam_tform4x4_obj = self.pre_rendered_modalities[
            modality
        ].cams_tform4x4_obj[meshes_ids]

        B, T, C, H, W = pre_rendered_feats.shape

        pxl2d_cams = (
            get_pxl2d(
                H=H,
                W=W,
                dtype=pre_rendered_feats.dtype,
                device=pre_rendered_feats.device,
                B=None,
            )
            * self.pre_rendered_modalities[modality].down_sample_rate
        )
        pxl2d_cams = pxl2d_cams.expand(*pre_rendered_feats.shape[:2], *pxl2d_cams.shape)
        pts3d_homog_cams = (
            transf3d_broadcast(
                pts3d=add_homog_dim(pxl2d_cams, dim=4),
                transf4x4=cams_intr4x4.pinverse()[:, :, None, None],
            )
            * cams_tform4x4_obj[:, :, 2, 3, None, None, None]
        )
        pts3d_pre_rendered = transf3d_broadcast(
            pts3d=pts3d_homog_cams,
            transf4x4=tform4x4(
                pre_rendered_cam_tform4x4_obj,
                inv_tform4x4(cams_tform4x4_obj),
            )[:, :, None, None],
        )
        pxl2d_pre_rendered = (
            proj3d2d_broadcast(
                pts3d=pts3d_pre_rendered,
                proj4x4=pre_rendered_cam_intr4x4[:, :, None, None],
            )
            / self.pre_rendered_modalities[modality].down_sample_rate
        )
        cams_features = sample_pxl2d_grid(
            pre_rendered_feats.reshape(-1, C, H, W),
            pxl2d=pxl2d_pre_rendered.reshape(-1, H, W, 2),
        ).reshape(B, T, C, H, W)

        return cams_features

    def get_pre_rendered_masks(
        self,
        cams_tform4x4_obj,
        cams_intr4x4,
        imgs_sizes,
        meshes_ids=None,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
    ):
        assert self.pre_rendered_masks_verts_vsbl_cams_tform4x4_obj == cams_tform4x4_obj
        assert self.pre_rendered_masks_verts_vsbl_cams_intr4x4 == cams_intr4x4
        assert self.pre_rendered_masks_verts_vsbl_imgs_sizes == imgs_sizes
        assert self.pre_rendered_masks_verts_vsbl_meshes_ids == meshes_ids
        assert (
            self.pre_rendered_masks_verts_vsbl_broadcast_batch_and_cams
            == broadcast_batch_and_cams
        )
        assert self.pre_rendered_masks_verts_vsbl_down_sample_rate == down_sample_rate

        if self.pre_rendered_feats is None:
            self.pre_rendered_masks_verts_vsbl_cams_tform4x4_obj = cams_tform4x4_obj
            self.pre_rendered_masks_verts_vsbl_cams_intr4x4 = cams_intr4x4
            self.pre_rendered_masks_verts_vsbl_imgs_sizes = imgs_sizes
            self.pre_rendered_feats_meshes_ids = meshes_ids
            self.pre_rendered_feats_broadcast_batch_and_cams = broadcast_batch_and_cams
            self.pre_rendered_down_sample_rate = down_sample_rate
            self.pre_rendered_feats = self.render_feats(
                cams_tform4x4_obj=self.pre_rendered_feats_cams_tform4x4_obj,
                cams_intr4x4=self.pre_rendered_feats_cams_intr4x4,
                imgs_sizes=self.pre_rendered_feats_imgs_sizes,
                meshes_ids=self.pre_rendered_feats_meshes_ids,
                modality=MESH_RENDER_MODALITIES.MASK_VERTS_VSBL,
                broadcast_batch_and_cams=self.pre_rendered_feats_broadcast_batch_and_cams,
                down_sample_rate=self.pre_rendered_down_sample_rate,
            )

        return self.pre_rendered_feats

    # def get_pre_rendered_masks(self):

    def render_feats(
        self,
        cams_tform4x4_obj,
        cams_intr4x4,
        imgs_sizes,
        meshes_ids=None,
        modality=MESH_RENDER_MODALITIES.FEATS,
        broadcast_batch_and_cams=False,
        down_sample_rate=1.0,
    ):
        # imgs_size: (height, width)
        dtype = cams_tform4x4_obj.dtype
        device = cams_tform4x4_obj.device

        self.to(device)

        if down_sample_rate != 1.0:
            cams_intr4x4 = cams_intr4x4.clone()
            if cams_intr4x4.dim() == 2:
                cams_intr4x4[:2] /= down_sample_rate
            elif cams_intr4x4.dim() == 3:
                cams_intr4x4[:, :2] /= down_sample_rate
            elif cams_intr4x4.dim() == 4:
                cams_intr4x4[:, :, :2] /= down_sample_rate
            else:
                raise NotImplementedError
            imgs_sizes = imgs_sizes.clone() // down_sample_rate
        else:
            cams_intr4x4 = cams_intr4x4.clone()
            imgs_sizes = imgs_sizes.clone()

        if meshes_ids is None:
            meshes_ids = torch.LongTensor(list(range(len(self)))).to(device=device)

        if isinstance(meshes_ids, torch.LongTensor):
            meshes_ids = meshes_ids.clone().to(device=device)

        meshes_count = meshes_ids.shape[0]
        if cams_tform4x4_obj.dim() == 4:
            cams_count = cams_tform4x4_obj.shape[1]
        elif cams_tform4x4_obj.dim() == 3:
            cams_count = cams_tform4x4_obj.shape[0]
        else:
            raise ValueError(f"Set `cams_tform4x4_obj.dim()` must be 3 or 4")

        if broadcast_batch_and_cams:
            meshes_ids = meshes_ids
            if cams_tform4x4_obj.dim() == 3:
                cams_tform4x4_obj = cams_tform4x4_obj[None, :]
            if cams_intr4x4.dim() == 3:
                cams_intr4x4 = cams_intr4x4[None, :]
            cams_tform4x4_obj = cams_tform4x4_obj.expand(
                meshes_count,
                cams_count,
                4,
                4,
            ).reshape(-1, 4, 4)
            cams_intr4x4 = cams_intr4x4.expand(meshes_count, cams_count, 4, 4).reshape(
                -1,
                4,
                4,
            )
            meshes_ids = (
                meshes_ids[:, None].expand(meshes_count, cams_count).reshape(-1)
            )
            render_count = meshes_count * cams_count
        else:
            if meshes_count != cams_count:
                raise ValueError(
                    f"Set `broadcast_batch_and_cams=True` to allow different number of cameras and meshes",
                )
            render_count = meshes_count

        if self.gaussian_splat_enabled and modality in [
            MESH_RENDER_MODALITIES.VERTS_NCDS,
            MESH_RENDER_MODALITIES.RGB,
            MESH_RENDER_MODALITIES.FEATS,
            MESH_RENDER_MODALITIES.MASK,
        ]:  # MESH_RENDER_MODALITIES.FEATS:
            # from od3d.cv.render.gaussian_splats import render_gaussians
            from od3d.cv.render.gaussians_splats_v2 import render_gaussians

            pts3d = (
                self.get_verts_stacked_with_mesh_ids(mesh_ids=meshes_ids)
                .to(device)
                .clone()
                .detach()
            )
            if modality == MESH_RENDER_MODALITIES.VERTS_NCDS:
                feats = self.get_verts_ncds_stacked_with_mesh_ids(
                    mesh_ids=meshes_ids,
                ).to(device)
            elif modality == MESH_RENDER_MODALITIES.RGB:
                feats = self.get_rgb_stacked_with_mesh_ids(mesh_ids=meshes_ids).to(
                    device,
                )
            elif modality == MESH_RENDER_MODALITIES.MASK:
                feats = torch.ones_like(pts3d[..., 0:1])
            else:
                feats = self.get_feats_stacked_with_mesh_ids(mesh_ids=meshes_ids).to(
                    device,
                )

            pts3d_mask = self.mask_verts_not_padded.to(device)[meshes_ids]
            mesh_feats2d_rendered = render_gaussians(
                cams_tform4x4_obj=cams_tform4x4_obj,
                cams_intr4x4=cams_intr4x4,
                imgs_size=imgs_sizes,
                pts3d=pts3d,
                pts3d_mask=pts3d_mask,
                feats=feats,
                opacity=self.gaussian_splat_opacity,
                pts3d_size_rel_to_neighbor_dist=self.gaussian_splat_pts3d_size_rel_to_neighbor_dist,
            )

            if broadcast_batch_and_cams:
                # logger.info(cams_intr4x4.reshape(meshes_count, cams_count, 4, 4)[:, 0])
                # logger.info(cams_tform4x4_obj.reshape(meshes_count, cams_count, 4, 4)[:, 0])
                mesh_feats2d_rendered = mesh_feats2d_rendered.reshape(
                    meshes_count,
                    cams_count,
                    *mesh_feats2d_rendered.shape[-3:],
                )
                # from od3d.cv.visual.show import show_imgs
                # show_imgs(mesh_feats2d_rendered)
            else:
                pass
            return mesh_feats2d_rendered
        # self.to(device)

        # num_cams = cams_tform4x4_obj.shape[0]

        # cams_tform4x4_obj = cams_tform4x4_obj.repeat_interleave(num_meshes, dim=0)
        # cams_intr4x4 = cams_intr4x4.repeat_interleave(num_meshes, dim=0)
        # imgs_sizes = imgs_sizes.repeat_interleave(num_meshes, dim=0)
        t3d_tform_pscl3d = torch.Tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).to(device=device, dtype=dtype)
        # t3d_cam_tform_obj = torch.matmul(t3d_tform_pscl3d, cams_tform4x4_obj)
        t3d_cam_tform_obj = (
            t3d_tform_pscl3d[None]
            .expand(cams_tform4x4_obj.shape)
            .bmm(cams_tform4x4_obj)
        )
        R = t3d_cam_tform_obj[..., :3, :3].permute(0, 2, 1)
        t = t3d_cam_tform_obj[..., :3, 3]
        focal_length = torch.stack(
            [cams_intr4x4[..., 0, 0], cams_intr4x4[..., 1, 1]],
            dim=-1,
        )
        principal_point = torch.stack(
            [cams_intr4x4[..., 0, 2], cams_intr4x4[..., 1, 2]],
            dim=-1,
        )

        cameras = PerspectiveCameras(
            device=device,
            R=R,
            T=t,
            focal_length=focal_length,
            principal_point=principal_point,
            in_ndc=False,
            image_size=imgs_sizes[None,].expand(render_count, 2),
        )

        # K=self.K_4x4[None,]) #, K=K) # , K=K , znear=0.001, zfar=100000,
        #  znear=0.001, zfar=100000, fov=10
        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
        # the difference between naive and coarse-to-fine rasterization.

        raster_settings = RasterizationSettings(
            image_size=[int(imgs_sizes[0]), int(imgs_sizes[1])],
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None,
            perspective_correct=self.pt3d_raster_perspective_correct,
            cull_backfaces=False,  # cull_backfaces=True
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )

        pt3dmeshes = self.pt3dmeshes[meshes_ids]
        fragments = rasterizer(pt3dmeshes)
        # pix_to_face: BxHxWx1, zbuf: BxHxWx1, bary_coords: BxHxWx1x3, dists: BxHxWx1
        if modality == MESH_RENDER_MODALITIES.MASK:
            mask = fragments.zbuf.permute(0, 3, 1, 2) > 0.0
            if broadcast_batch_and_cams:
                mask = mask.reshape(meshes_count, cams_count, *mask.shape[-3:])

            return mask

        if modality == MESH_RENDER_MODALITIES.DEPTH:
            depth = fragments.zbuf.permute(0, 3, 1, 2)
            if broadcast_batch_and_cams:
                depth = depth.reshape(meshes_count, cams_count, *depth.shape[-3:])

            return depth

        if modality == MESH_RENDER_MODALITIES.MASK_VERTS_VSBL:
            B = fragments.pix_to_face.shape[0]
            faces_ids = torch.cat(
                [self.get_faces_with_mesh_id(mesh_id) for mesh_id in meshes_ids],
                dim=0,
            )
            # verts_ids_vsbl = torch.cat([self.get_faces_with_mesh_id(mesh_id) for mesh_id in meshes_ids], dim=0) [fragments.pix_to_face.reshape(B, -1)].reshape(B, -1)  # .unique(dim=1)
            verts_vsbl_mask = torch.zeros(
                size=(B, self.verts_counts_max),
                dtype=torch.bool,
                device=device,
            )
            for b in range(B):
                # logger.info(f'meshes_ids {meshes_ids}')
                faces_ids_vsbl = fragments.pix_to_face[b]
                faces_ids_vsbl = faces_ids_vsbl.unique()
                faces_ids_vsbl = faces_ids_vsbl[faces_ids_vsbl >= 0]
                verts_ids_vsbl = faces_ids[faces_ids_vsbl].unique()
                verts_vsbl_mask[b, verts_ids_vsbl] = 1

            return verts_vsbl_mask

        if modality == MESH_RENDER_MODALITIES.FEATS:
            feats_from_faces = torch.cat(
                [
                    self.get_feats_from_faces_with_mesh_id(mesh_id)
                    for mesh_id in meshes_ids
                ],
                dim=0,
            )
        else:
            feats_from_faces = torch.cat(
                [
                    self.get_verts_ncds_from_faces_with_mesh_id(mesh_id)
                    for mesh_id in meshes_ids
                ],
                dim=0,
            )

        mesh_feats2d_rendered = interpolate_face_attributes(
            fragments.pix_to_face,
            fragments.bary_coords,
            feats_from_faces,
        )[:, ..., 0, :].permute(0, 3, 1, 2)
        # mask = fragments.pix_to_face >= 0
        # mesh_feats2d_prob = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

        if broadcast_batch_and_cams:
            mesh_feats2d_rendered = mesh_feats2d_rendered.reshape(
                meshes_count,
                cams_count,
                *mesh_feats2d_rendered.shape[-3:],
            )

        return mesh_feats2d_rendered

    def render_batch(
        self,
        cams_tform4x4_obj,
        cams_intr4x4,
        imgs_sizes,
        objects_ids=None,
        modalities: Union[
            PROJECT_MODALITIES,
            List[PROJECT_MODALITIES],
        ] = PROJECT_MODALITIES.FEATS,
        add_clutter=False,
        add_other_objects=False,
    ):
        """
        Render the objects in the scene with the given camera parameters.
        Args:
            cams_tform4x4_obj: (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (B, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
            objects_ids: (B, ) tensor of objects ids to render.
            modalities: (List(PROJECT_MODALITIES)) the modalities to render.
            add_clutter: bool, determines wether to add clutter features to the rendered features.
        Returns:
            mods2d_rendered (Dict[PROJECT_MODALITIES, torch.Tensor]): (B, F, H, W) dict of rendered modalities.
        """

        device = cams_tform4x4_obj.device
        dtype = cams_tform4x4_obj.dtype
        render_count = cams_tform4x4_obj.shape[0]

        # cams_tform4x4_obj = cams_tform4x4_obj.repeat_interleave(num_meshes, dim=0)
        # cams_intr4x4 = cams_intr4x4.repeat_interleave(num_meshes, dim=0)
        # imgs_sizes = imgs_sizes.repeat_interleave(num_meshes, dim=0)
        t3d_tform_pscl3d = torch.Tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).to(device=device, dtype=dtype)
        # t3d_cam_tform_obj = torch.matmul(t3d_tform_pscl3d, cams_tform4x4_obj)
        t3d_cam_tform_obj = (
            t3d_tform_pscl3d[None]
            .expand(cams_tform4x4_obj.shape)
            .bmm(cams_tform4x4_obj)
        )
        R = t3d_cam_tform_obj[..., :3, :3].permute(0, 2, 1)
        t = t3d_cam_tform_obj[..., :3, 3]
        focal_length = torch.stack(
            [cams_intr4x4[..., 0, 0], cams_intr4x4[..., 1, 1]],
            dim=-1,
        )
        principal_point = torch.stack(
            [cams_intr4x4[..., 0, 2], cams_intr4x4[..., 1, 2]],
            dim=-1,
        )

        cameras = PerspectiveCameras(
            device=device,
            R=R,
            T=t,
            focal_length=focal_length,
            principal_point=principal_point,
            in_ndc=False,
            image_size=imgs_sizes[None,].expand(render_count, 2),
        )

        # K=self.K_4x4[None,]) #, K=K) # , K=K , znear=0.001, zfar=100000,
        #  znear=0.001, zfar=100000, fov=10
        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
        # the difference between naive and coarse-to-fine rasterization.

        raster_settings = RasterizationSettings(
            image_size=[int(imgs_sizes[0]), int(imgs_sizes[1])],
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=None,
            perspective_correct=self.pt3d_raster_perspective_correct,
            cull_backfaces=False,  # cull_backfaces=True
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )

        pt3dmeshes = self.pt3dmeshes[objects_ids]
        fragments = rasterizer(pt3dmeshes)

        mods2d_rendered = {}
        for modality in modalities:
            if modality == PROJECT_MODALITIES.MASK:
                mod2d_rendered = fragments.zbuf.permute(0, 3, 1, 2) > 0.0

            elif (
                modality == PROJECT_MODALITIES.ONEHOT
                or modality == PROJECT_MODALITIES.ONEHOT_SMOOTH
            ):
                if modality == PROJECT_MODALITIES.ONEHOT:
                    if add_other_objects:
                        num_classes = self.verts_count
                        verts_ids_from_faces = torch.cat(
                            [
                                self.get_faces_with_mesh_id(object_id)
                                + self.verts_counts_acc_from_0[object_id]
                                for b, object_id in enumerate(objects_ids)
                            ],
                            dim=0,
                        )
                    else:
                        num_classes = self.verts_counts_max
                        verts_ids_from_faces = torch.cat(
                            [
                                self.get_faces_with_mesh_id(object_id)
                                for b, object_id in enumerate(objects_ids)
                            ],
                            dim=0,
                        )

                    if add_clutter:
                        num_classes += 1

                    verts_one_hot_from_faces = torch.nn.functional.one_hot(
                        verts_ids_from_faces,
                        num_classes=num_classes,
                    ).to(device, dtype)

                elif modality == PROJECT_MODALITIES.ONEHOT_SMOOTH:
                    from od3d.cv.select import batched_index_select

                    verts_one_hot_from_faces = torch.cat(
                        [
                            self.get_smooth_label_from_object_id(
                                object_id=object_id,
                                add_other_objects=add_other_objects,
                                add_clutter=add_clutter,
                                device=device,
                            )[self.get_faces_with_mesh_id(object_id).to(device=device)]
                            for b, object_id in enumerate(objects_ids)
                        ],
                        dim=0,
                    )

                mod2d_rendered = interpolate_face_attributes(
                    fragments.pix_to_face,
                    fragments.bary_coords,
                    verts_one_hot_from_faces,
                )[:, ..., 0, :].permute(0, 3, 1, 2)

                if add_clutter:
                    mask = (fragments.zbuf.permute(0, 3, 1, 2) > 0.0).expand(
                        *mod2d_rendered.shape,
                    )
                    clutter_onehot = (
                        self.get_label_clutter(
                            add_other_objects=add_other_objects,
                            one_hot=True,
                            device=device,
                        )[:, :, None, None]
                        .expand(*mod2d_rendered.shape)
                        .to(dtype)
                    )
                    mod2d_rendered[~mask] = clutter_onehot[~mask]

            elif modality == PROJECT_MODALITIES.DEPTH:
                mod2d_rendered = fragments.zbuf.permute(0, 3, 1, 2)

            elif modality == PROJECT_MODALITIES.MASK_VERTS_VSBL:
                B = fragments.pix_to_face.shape[0]
                faces_ids = torch.cat(
                    [
                        self.get_faces_with_mesh_id(object_id)
                        for object_id in objects_ids
                    ],
                    dim=0,
                )
                # verts_ids_vsbl = torch.cat([self.get_faces_with_mesh_id(mesh_id) for mesh_id in meshes_ids], dim=0) [fragments.pix_to_face.reshape(B, -1)].reshape(B, -1)  # .unique(dim=1)
                verts_vsbl_mask = torch.zeros(
                    size=(B, self.verts_counts_max),
                    dtype=torch.bool,
                    device=device,
                )
                for b in range(B):
                    # logger.info(f'meshes_ids {meshes_ids}')
                    faces_ids_vsbl = fragments.pix_to_face[b]
                    faces_ids_vsbl = faces_ids_vsbl.unique()
                    faces_ids_vsbl = faces_ids_vsbl[faces_ids_vsbl >= 0]
                    verts_ids_vsbl = faces_ids[faces_ids_vsbl].unique()
                    verts_vsbl_mask[b, verts_ids_vsbl] = 1
                mod2d_rendered = verts_vsbl_mask

            elif modality == PROJECT_MODALITIES.FEATS:
                feats_from_faces = torch.cat(
                    [
                        self.get_feats_from_faces_with_mesh_id(object_id)
                        for object_id in objects_ids
                    ],
                    dim=0,
                )
                mod2d_rendered = interpolate_face_attributes(
                    fragments.pix_to_face,  # B x H x W x 1
                    fragments.bary_coords,  # B x H x W x 1 x 3
                    feats_from_faces,  # F x 3 x C
                )[:, ..., 0, :].permute(0, 3, 1, 2)
                # mask = fragments.pix_to_face >= 0
                # mesh_feats2d_prob = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
            elif modality == PROJECT_MODALITIES.PT3D_NCDS:
                feats_from_faces = torch.cat(
                    [
                        self.get_verts_ncds_from_faces_with_mesh_id(object_id)
                        for object_id in objects_ids
                    ],
                    dim=0,
                )
                mod2d_rendered = interpolate_face_attributes(
                    fragments.pix_to_face,
                    fragments.bary_coords,
                    feats_from_faces,
                )[:, ..., 0, :].permute(0, 3, 1, 2)
                # mask = fragments.pix_to_face >= 0
                # mesh_feats2d_prob = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

            else:
                raise ValueError(f"Unknown modality {modality}")

            mods2d_rendered[modality] = mod2d_rendered

        return mods2d_rendered

    def sample_batch(
        self,
        cams_tform4x4_obj,
        cams_intr4x4,
        imgs_sizes,
        objects_ids=None,
        modalities: Union[
            PROJECT_MODALITIES,
            List[PROJECT_MODALITIES],
        ] = PROJECT_MODALITIES.FEATS,
        add_clutter=False,
        add_other_objects=False,
        device=None,
        dtype=None,
        sample_clutter=False,
        sample_other_objects=False,
    ):
        """
        Sample the objects' projection.
        Args:
            cams_tform4x4_obj: (B, 4, 4) tensor of camera poses in the object frame.
            cams_intr4x4: (B, 4, 4) tensor of camera intrinsics.
            imgs_sizes: (2,) tensor of (H, W)
            objects_ids: (B, ) tensor of objects ids to render.
            modalities: (List(PROJECT_MODALITIES)) the modalities to render.
            add_clutter: bool, determines wether to add clutter features to the sampled features.
        Returns:
            mods1d_sampled (Dict[PROJECT_MODALITIES, torch.Tensor]): (B, V, F) dict of projected modalities.
        """

        mods1d_sampled = {}
        for modality in modalities:
            if modality in mods1d_sampled.keys():
                continue

            if (
                modality == PROJECT_MODALITIES.MASK
                or modality == PROJECT_MODALITIES.MASK_VERTS_VSBL
            ):
                mask_verts_vsbl = self.render(
                    cams_tform4x4_obj=cams_tform4x4_obj,
                    cams_intr4x4=cams_intr4x4,
                    imgs_sizes=imgs_sizes,
                    objects_ids=objects_ids,
                    modalities=PROJECT_MODALITIES.MASK_VERTS_VSBL,
                    add_clutter=False,
                    add_other_objects=False,
                ).to(device)

                pxl2d_verts = self.sample(
                    cams_tform4x4_obj=cams_tform4x4_obj,
                    cams_intr4x4=cams_intr4x4,
                    imgs_sizes=imgs_sizes,
                    objects_ids=objects_ids,
                    modalities=PROJECT_MODALITIES.PXL2D,
                    add_clutter=False,
                    add_other_objects=False,
                    sample_clutter=False,
                    sample_other_objects=False,
                ).to(device)

                mask_verts_vsbl *= (
                    pxl2d_verts <= (imgs_sizes[None, None].to(device) - 1)
                ).all(dim=-1)
                mask_verts_vsbl *= (pxl2d_verts >= 0).all(dim=-1)

                # if add_clutter or add_other_objects:
                #     clutter_count = 1 if add_clutter else 0
                #     B = mask_verts_vsbl.shape[0]
                #     if add_other_objects:
                #         mask_verts_vsbl_all = torch.zeros((B, self.verts_count + clutter_count),
                #                                           device=mask_verts_vsbl.device, dtype=mask_verts_vsbl.dtype)
                #         for b in range(B):
                #             mask_verts_vsbl_all[b, self.verts_counts_acc_from_0[objects_ids[b]]:
                #                                    self.verts_counts_acc_from_0[objects_ids[b]+1]] = \
                #                 mask_verts_vsbl[b, :self.verts_counts[objects_ids[b]]]
                #         mask_verts_vsbl = mask_verts_vsbl_all
                #     else:
                #         mask_verts_vsbl_all = torch.zeros((B, self.verts_counts_max + clutter_count),
                #                                           device=mask_verts_vsbl.device, dtype=mask_verts_vsbl.dtype)
                #         mask_verts_vsbl_all[:, :self.verts_count_max] = mask_verts_vsbl
                #         mask_verts_vsbl = mask_verts_vsbl_all
                mods1d_sampled[modality] = mask_verts_vsbl
            elif modality == PROJECT_MODALITIES.FEATS:
                if add_other_objects:
                    B = len(objects_ids)
                    F = self.feat_dim
                    mods1d_sampled[modality] = self.feats_objects.clone()[
                        None,
                        :,
                        :,
                    ].repeat(
                        B,
                        1,
                        1,
                    )  # (B, O*V, F)
                else:
                    mods1d_sampled[modality] = self.get_feats_stacked_with_mesh_ids(
                        mesh_ids=objects_ids,
                    )  # (B, V, F)
                if add_clutter:
                    B = mods1d_sampled[modality].shape[0]
                    F = mods1d_sampled[modality].shape[-1]
                    mods1d_sampled[modality] = torch.cat(
                        [
                            mods1d_sampled[modality],
                            self.feat_clutter.clone()[None, None].expand(B, 1, F),
                        ],
                        dim=1,
                    )  # (B, V+1, F)
            elif modality == PROJECT_MODALITIES.ID:
                mods1d_sampled[modality] = self.get_labels_ids(
                    objects_ids=objects_ids,
                    add_other_objects=add_other_objects,
                    sample_clutter=sample_clutter,
                    sample_other_objects=sample_other_objects,
                    device=device,
                )

            elif (
                modality == PROJECT_MODALITIES.ONEHOT
                or modality == PROJECT_MODALITIES.ONEHOT_SMOOTH
            ):
                labels_onehot = self.get_labels_onehot(
                    smooth=modality == PROJECT_MODALITIES.ONEHOT_SMOOTH,
                    objects_ids=objects_ids,
                    add_other_objects=add_other_objects,
                    add_clutter=add_clutter,
                    sample_clutter=sample_clutter,
                    sample_other_objects=sample_other_objects,
                    device=device,
                )
                mods1d_sampled[modality] = labels_onehot.permute(0, 2, 1)

            elif modality == PROJECT_MODALITIES.PXL2D:
                cams_proj4x4_obj = torch.bmm(cams_intr4x4, cams_tform4x4_obj)

                if sample_other_objects:
                    verts3d = self.get_verts_stacked_with_mesh_ids(mesh_ids=None).to(
                        device,
                    )
                    verts3d = verts3d[None,].expand(len(objects_ids), *verts3d.shape)
                else:
                    verts3d = self.get_verts_stacked_with_mesh_ids(
                        mesh_ids=objects_ids,
                    ).to(device)

                if sample_clutter:
                    B = verts3d.shape[0]
                    verts3d = torch.cat(
                        [
                            verts3d,
                            torch.zeros(
                                (B, 1, 3),
                                device=verts3d.device,
                                dtype=verts3d.dtype,
                            ),
                        ],
                        dim=1,
                    )

                pxl2d = proj3d2d_broadcast(verts3d, proj4x4=cams_proj4x4_obj[:, None])
                mods1d_sampled[modality] = pxl2d
            elif modality == PROJECT_MODALITIES.PT3D:
                if sample_other_objects:
                    verts3d = self.get_verts_stacked_with_mesh_ids(mesh_ids=None).to(
                        device,
                    )
                    verts3d = verts3d[None,].expand(len(objects_ids), *verts3d.shape)
                else:
                    verts3d = self.get_verts_stacked_with_mesh_ids(
                        mesh_ids=objects_ids,
                    ).to(device)

                if sample_clutter:
                    B = verts3d.shape[0]
                    verts3d = torch.cat(
                        [
                            verts3d,
                            torch.zeros(
                                (B, 1, 3),
                                device=verts3d.device,
                                dtype=verts3d.dtype,
                            ),
                        ],
                        dim=1,
                    )
                mods1d_sampled[modality] = verts3d
            elif modality == PROJECT_MODALITIES.PT3D_NCDS:
                if sample_other_objects:
                    verts3d = self.get_verts_ncds_stacked_with_mesh_ids(
                        mesh_ids=None,
                    ).to(device)
                    verts3d = verts3d[None,].expand(len(objects_ids), *verts3d.shape)
                else:
                    verts3d = self.get_verts_ncds_stacked_with_mesh_ids(
                        mesh_ids=objects_ids,
                    ).to(device)

                if sample_clutter:
                    B = verts3d.shape[0]
                    verts3d = torch.cat(
                        [
                            verts3d,
                            torch.zeros(
                                (B, 1, 3),
                                device=verts3d.device,
                                dtype=verts3d.dtype,
                            ),
                        ],
                        dim=1,
                    )
                mods1d_sampled[modality] = verts3d
            else:
                raise ValueError(f"Unknown modality {modality}")

        return mods1d_sampled

    def get_label_max(self, add_other_objects=True, add_clutter=True):
        if add_other_objects:
            if add_clutter:
                return self.verts_count
            else:
                return self.verts_count - 1
        else:
            if add_clutter:
                return self.verts_counts_max
            else:
                return self.verts_counts_max - 1

    def get_label_clutter(self, add_other_objects=False, one_hot=False, device=None):
        clutter_label = self.get_label_max(
            add_other_objects=add_other_objects,
            add_clutter=True,
        )
        clutter_label = torch.LongTensor([clutter_label]).to(device)

        if one_hot:
            clutter_label = torch.nn.functional.one_hot(clutter_label, num_classes=-1)
        return clutter_label

    def update_feats_moving_average(
        self,
        labels,
        labels_mask,
        feats,
        alpha,
        objects_ids=None,
        add_clutter=True,
        add_other_objects=True,
    ):
        """
        Args:
            labels (torch.Tensor): BxN (or BxVxN)
            labels_mask (torch.Tensor): BxN
            feats (torch.Tensor): BxNxF
            alpha (float): the moving average factor.
        """
        device = feats.device
        dtype = feats.dtype
        feats_count = len(self.feats_objects) + 1
        if (
            not hasattr(self, "feats_moving_average")
            or self.feats_moving_average is None
        ):
            self.feats_moving_average = torch.zeros(
                (feats_count, self.feat_dim),
                dtype=dtype,
                device=device,
            )
            import math

            # equals kaiming uniform
            bound = 1 / math.sqrt(self.feat_dim) if self.feat_dim > 0 else 0
            torch.nn.init.uniform_(self.feats_moving_average, a=-bound, b=-bound)

        # V x F
        # feats_update = torch.zeros((feats_count, self.feat_dim), dtype=dtype, device=device)

        labels = labels.clone()
        if labels.dim() == 2:
            if not add_other_objects:
                for b in range(objects_ids.shape[0]):
                    labels[b] = labels[b] + self.verts_counts_acc_from_0[objects_ids[b]]
            labels[labels >= feats_count] = feats_count - 1
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes=feats_count)
        else:
            raise NotImplementedError
            # labels_onehot = labels.permute(0, 2, 1)
            # if not add_other_objects or not add_clutter:
            #     B = labels_onehot.shape[0]
            #     N = labels_onehot.shape[1]
            #     labels_onehot_ext = torch.zeros((B, N, feats_count), dtype=torch.bool, device=device)
            #
            #     if not add_other_objects:
            #         labels_onehot_ext[:, :, self.verts_counts_acc_from_0[objects_ids+1]:self.verts_counts_acc_from_0[objects_ids+2]] = labels_onehot
            #     labels_onehot = labels_onehot_ext

        feats_update = torch.einsum(
            "nf,nv->vf",
            feats[labels_mask],
            labels_onehot[labels_mask] * 1.0,
        ) / (labels_onehot[labels_mask].sum(dim=0)[:, None] + 1e-10)
        feats_update = feats_update.detach()
        feats_update_mask = labels_onehot[labels_mask].sum(dim=0) > 0
        self.feats_moving_average[feats_update_mask] = (
            alpha * self.feats_moving_average[feats_update_mask]
            + (1.0 - alpha) * feats_update[feats_update_mask]
        )

        self.feats_objects.data = self.feats_moving_average[:-1].clone()
        self.feat_clutter.data = self.feats_moving_average[-1].clone()
        self.normalize_feats()

    def update_feats_total_average(
        self,
        labels,
        labels_mask,
        feats,
        objects_ids=None,
        add_clutter=True,
        add_other_objects=True,
    ):
        device = feats.device
        dtype = feats.dtype
        feats_count = len(self.feats_objects) + 1
        if (
            not hasattr(self, "feats_total_average_sum")
            or self.feats_total_average_sum is None
        ):
            self.feats_total_average_sum = torch.zeros(
                (feats_count, self.feat_dim),
                dtype=dtype,
                device=device,
            )
            self.feats_total_count = torch.zeros(
                (feats_count,),
                dtype=torch.long,
                device=device,
            )

        labels = labels.clone()
        if labels.dim() == 2:
            if not add_other_objects:
                for b in range(objects_ids.shape[0]):
                    labels[b] = labels[b] + self.verts_counts_acc_from_0[objects_ids[b]]
            labels[labels >= feats_count] = feats_count - 1
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes=feats_count)
        else:
            raise NotImplementedError
            # labels_onehot = labels.permute(0, 2, 1)
            # if not add_other_objects or not add_clutter:
            #     B = labels_onehot.shape[0]
            #     N = labels_onehot.shape[1]
            #     labels_onehot_ext = torch.zeros((B, N, feats_count), dtype=torch.bool, device=device)
            #
            #     if not add_other_objects:
            #         labels_onehot_ext[:, :, self.verts_counts_acc_from_0[objects_ids + 1]:self.verts_counts_acc_from_0[
            #             objects_ids + 2]] = labels_onehot
            #     labels_onehot = labels_onehot_ext

        feats_update = torch.einsum(
            "nf,nv->vf",
            feats[labels_mask],
            labels_onehot[labels_mask] * 1.0,
        ) / (labels_onehot[labels_mask].sum(dim=0)[:, None] + 1e-10)
        feats_update = feats_update.detach()
        feats_update_mask = labels_onehot[labels_mask].sum(dim=0) > 0
        self.feats_total_average_sum[feats_update_mask] = (
            self.feats_total_average_sum[feats_update_mask]
            + feats_update[feats_update_mask]
        )
        self.feats_total_count[feats_update_mask] += 1

        self.feats_objects.data = self.feats_total_average_sum[:-1].clone()
        self.feat_clutter.data = self.feats_total_average_sum[-1].clone()
        self.normalize_feats()

    def get_labels_ids(
        self,
        objects_ids=None,
        add_other_objects=True,
        sample_clutter=False,
        sample_other_objects=True,
        device=None,
    ):
        """
        Args:
            objects_ids (torch.Tensor): B
            add_other_objects (bool): whether to add other objects.
            sample_clutter (bool): whether to sample clutter.
            sample_other_objects (bool): whether to sample other objects.
        Returns:
            labels_ids (torch.Tensor): BxV(*O)(+1) depending on sample clutter/sample other objects
                                       from 0 to V(*O)(+1) or 0 to V(+1) if not add other objects.
        """
        if objects_ids is None:
            objects_ids = list(range(len(self)))

        if device is None:
            device = self.device

        if add_other_objects:
            if not sample_other_objects:
                labels_ids = self.get_verts_and_noise_ids_stacked(
                    mesh_ids=objects_ids,
                    count_noise_ids=0,
                ).to(
                    device,
                )  # (B, V(+1))
            else:
                labels_ids = self.get_verts_and_noise_ids_stacked(
                    mesh_ids=None,
                    count_noise_ids=0,
                ).to(device)
                labels_ids = labels_ids[None,].expand(
                    len(objects_ids),
                    *labels_ids.shape,
                )
        else:
            if not sample_other_objects:
                labels_ids = self.get_verts_and_noise_ids_stacked_without_acc(
                    mesh_ids=objects_ids,
                    count_noise_ids=0,
                ).to(
                    device,
                )  # (B, V(+1))
            else:
                labels_ids = self.get_verts_and_noise_ids_stacked_without_acc(
                    mesh_ids=None,
                    count_noise_ids=0,
                ).to(
                    device,
                )
                labels_ids = labels_ids[None,].expand(
                    len(objects_ids),
                    *labels_ids.shape,
                )

        if sample_clutter:
            B = labels_ids.shape[0]
            labels_ids = torch.cat(
                [
                    labels_ids,
                    self.get_label_clutter(add_other_objects=add_other_objects)
                    .expand(B, 1)
                    .to(device),
                ],
                dim=1,
            )  # (B, V+1, F)
        return labels_ids

    def get_labels_onehot(
        self,
        smooth=False,
        objects_ids=None,
        add_other_objects=True,
        add_clutter=True,
        sample_clutter=False,
        sample_other_objects=True,
        device=None,
    ):
        """
        Args:
            objects_ids (torch.Tensor): B
            smooth (bool): whether to use smooth labels.
        Returns:
            labels_onehot (torch.Tensor): VxV
        """

        if device is None:
            device = self.device

        if objects_ids is None:
            objects_ids = list(range(len(self)))

        if not smooth:
            labels_ids = self.get_labels_ids(
                objects_ids=objects_ids,
                add_other_objects=add_other_objects,
                sample_clutter=sample_clutter,
                sample_other_objects=sample_other_objects,
                device=device,
            )  # BxV(*O)(+1) in range 0 to V(*O)(+1) or 0 to V(+1) if not add other objects.
            label_max = self.get_label_max(
                add_other_objects=add_other_objects,
                add_clutter=add_clutter,
            )
            labels_onehot = torch.eye(label_max + 1, device=device)

            labels_ids_out_of_range = labels_ids >= labels_onehot.shape[0]
            labels_ids[labels_ids_out_of_range] = 0
            labels_onehot = labels_onehot[labels_ids]
            labels_onehot[labels_ids_out_of_range] = 0
        else:
            labels_ids = self.get_labels_ids(
                objects_ids=objects_ids,
                add_other_objects=True,
                sample_clutter=sample_clutter,
                sample_other_objects=sample_other_objects,
                device=device,
            )  # BxV(*O)(+1) in range 0 to V(*O)(+1) or 0 to V(+1) if not add other objects.
            if add_other_objects:
                if add_clutter:
                    labels_onehot = (
                        self.get_geodesic_prob_with_noise().clone().to(device=device)
                    )

                else:
                    labels_onehot = self.get_geodesic_prob().clone().to(device=device)

                labels_ids_out_of_range = labels_ids >= labels_onehot.shape[0]
                labels_ids[labels_ids_out_of_range] = 0
                labels_onehot = labels_onehot[labels_ids]
                labels_onehot[labels_ids_out_of_range] = 0
            else:
                from od3d.cv.select import batched_index_select

                labels_onehot = self.get_smooth_label_from_objects_ids(
                    objects_ids=objects_ids,
                    add_other_objects=add_other_objects,
                    add_clutter=add_clutter,
                    device=device,
                )
                labels_onehot = batched_index_select(
                    index=labels_ids,
                    input=labels_onehot,
                    dim=1,
                )

        return labels_onehot

    def get_smooth_label_from_objects_ids(
        self,
        objects_ids=None,
        add_other_objects=True,
        add_clutter=True,
        device=None,
    ):
        if objects_ids is None:
            objects_ids = list(range(len(self)))

        label_max = self.get_label_max(
            add_other_objects=add_other_objects,
            add_clutter=add_clutter,
        )
        labels_onehot_smooth = torch.zeros(
            (len(objects_ids), label_max + 1, label_max + 1),
            device=device,
        )

        for b, object_id in enumerate(objects_ids):
            if add_clutter:
                labels_onehot_all = self.get_geodesic_prob_with_noise()
            else:
                labels_onehot_all = self.get_geodesic_prob()

            if add_other_objects:
                labels_onehot_smooth[
                    b,
                    : self.verts_counts[object_id],
                    : labels_onehot_all.shape[-1],
                ] = labels_onehot_all[
                    self.verts_counts_acc_from_0[
                        object_id
                    ] : self.verts_counts_acc_from_0[object_id + 1]
                ]
            else:
                labels_onehot_smooth[
                    b,
                    : self.verts_counts[object_id],
                    : self.verts_counts[object_id],
                ] = labels_onehot_all[
                    self.verts_counts_acc_from_0[
                        object_id
                    ] : self.verts_counts_acc_from_0[object_id + 1],
                    self.verts_counts_acc_from_0[
                        object_id
                    ] : self.verts_counts_acc_from_0[object_id + 1],
                ]

            if add_clutter:
                labels_onehot_smooth[b, -1, -1] = 1.0

        return labels_onehot_smooth

    def get_smooth_label_from_object_id(
        self,
        object_id,
        add_other_objects=True,
        add_clutter=True,
        device=None,
    ):
        if device is None:
            device = self.device

        labels_onehot_all = self.get_geodesic_prob().clone().to(device)

        if add_other_objects:
            labels_onehot_smooth = labels_onehot_all[
                self.verts_counts_acc_from_0[object_id] : self.verts_counts_acc_from_0[
                    object_id + 1
                ]
            ]
        else:
            labels_onehot_smooth = labels_onehot_all[
                self.verts_counts_acc_from_0[object_id] : self.verts_counts_acc_from_0[
                    object_id + 1
                ],
                self.verts_counts_acc_from_0[object_id] : self.verts_counts_acc_from_0[
                    object_id + 1
                ],
            ]

        if add_clutter:
            _labels_onehot_smooth = torch.zeros(
                (labels_onehot_smooth.shape[0], labels_onehot_smooth.shape[1] + 1),
            ).to(labels_onehot_smooth.device, labels_onehot_smooth.dtype)
            _labels_onehot_smooth[:, :-1] = labels_onehot_smooth
            labels_onehot_smooth = _labels_onehot_smooth
        return labels_onehot_smooth
