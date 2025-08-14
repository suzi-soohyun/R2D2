import logging
logger = logging.getLogger(__name__)

from od3d.datasets.frame import OD3D_Frame
from od3d.datasets.frame_meta import OD3D_FrameMeta

# from od3d.datasets.frame_meta import OD3D_FrameMeta
# from od3d.datasets.frame import OD3D_Frame, OD3D_FrameCamIntr4x4Mixin, OD3D_FrameCategoryMixin
from od3d.datasets.object import (
    OD3D_Object,
    OD3D_SequenceSfMTypeMixin,
    OD3D_SEQUENCE_SFM_TYPES,
    OD3D_PCLTypeMixin,
    OD3D_PCL_TYPES,
    OD3D_MeshTypeMixin,
    OD3D_MeshFeatsTypeMixin,
    OD3D_MESH_TYPES,
    OD3D_FrameModalitiesMixin,
    OD3D_TformObjMixin,
    OD3D_TFROM_OBJ_TYPES,
    OD3D_MESH_FEATS_DIST_REDUCE_TYPES,
)
from od3d.data.ext_dicts import rollup_flattened_dict
from dataclasses import dataclass, field
from typing import List
import numpy as np
from pathlib import Path
import torch
from od3d.cv.reconstruction.clean import (
    get_pcl_clean_with_masks,
)
from od3d.cv.io import write_pts3d_with_colors_and_normals
from od3d.datasets.object import OD3D_CAM_TFORM_OBJ_TYPES
from torch.utils.data import Dataset
import os
import re
from od3d.cv.io import read_pts3d_with_colors_and_normals
import open3d
from od3d.cv.geometry.objects3d.meshes import Mesh
from od3d.cv.geometry.downsample import random_sampling, voxel_downsampling

from od3d.cv.io import get_default_device
from od3d.cv.geometry.transform import (
    transf3d_broadcast,
    transf3d_normal_broadcast,
    inv_tform4x4,
    tform4x4,
    tform4x4_broadcast,
    transf3d,
)
from od3d.datasets.object import OD3D_TFROM_OBJ_TYPES
import pytorch3d
from od3d.cv.reconstruction import camera_alignment
import random
import trimesh
from PIL import Image
from od3d.cv.reconstruction.backproject_co3d import (
    _load_16bit_png_depth,
    backproject,
    read_from_depth_binary_array,
    backproject_with_rgb,
    backproject_with_feat,
)
import time
import torch.nn.functional as F
import os, sys

@dataclass
class OD3D_Sequence(OD3D_FrameModalitiesMixin, OD3D_Object, Dataset):
    frame_type = OD3D_Frame
    _frames_names = None
    _frames_names_unique = None
    transform = None

    @property
    def first_frame(self):
        return self.get_frame_by_index(index=0)

    @property
    def frames_names_unique(self):
        if self._frames_names_unique is None:
            dict_nested_frames = rollup_flattened_dict({self.name_unique: None})
            dict_nested_frames = OD3D_FrameMeta.complete_nested_metas(
                path_meta=self.path_meta,
                dict_nested_metas=dict_nested_frames,
            )
            self._frames_names_unique = OD3D_FrameMeta.unroll_nested_metas(
                dict_nested_meta=dict_nested_frames,
            )
        return self._frames_names_unique

    @property
    def frames_names(self):
        if self._frames_names is None:
            self._frames_names = [
                frame_name.split("/")[-1] for frame_name in self.frames_names_unique
            ]
        return self._frames_names

    @staticmethod
    def get_subset_frames_names_uniform(frames_names, count_max_per_sequence=None):
        if count_max_per_sequence is not None:
            frames_names = [
                frames_names[fid]
                for fid in np.linspace(0, len(frames_names) - 1, count_max_per_sequence)
                .astype(int)
                .tolist()
            ]
        return frames_names

    @property
    def frames_count(self):
        return len(self.frames_names)

    def get_frames(self, frames_ids=None, start_index=None, end_index=None):
        if frames_ids is None:
            frames_ids = list(range(self.frames_count))
        else:
            frames_ids = list(range(start_index, end_index))
        frames = [self.get_frame_by_index(frame_id) for frame_id in frames_ids]
        return frames

    def __len__(self):
        return self.frames_count

    def __getitem__(self, idx):
        frame = self.get_frame_by_index(idx)
        frame.item_id = idx
        return self.transform(frame)

    def collate_fn(
        self,
        frames: List[OD3D_Frame],
        device="cpu",
        dtype=torch.float32,
        modalities=None,
    ):
        if modalities is None:
            modalities = self.modalities
        from od3d.datasets.frames import OD3D_Frames

        frames = OD3D_Frames.get_frames_from_list(
            frames,
            modalities=modalities,
            dtype=dtype,
            device=device,
        )
        return frames

    def get_dataloader(self, batch_size=1, shuffle=False, transform=None):
        self.transform = transform
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def get_frame_by_index(self, index: int):
        return self.get_frame_by_name_unique(self.frames_names_unique[index])

    def get_frame_by_name_unique(self, frame_name_unique: str):
        from dataclasses import fields

        frame_fields_names = [field.name for field in fields(self.frame_type)]
        sequence_fields = fields(self)
        all_attrs_except_name_unique = {
            field.name: getattr(self, field.name)
            for field in sequence_fields
            if field.name != "name_unique" and field.name in frame_fields_names
        }
        return self.frame_type(
            name_unique=frame_name_unique,
            **all_attrs_except_name_unique,
        )

    def visualize(self):
        from od3d.cv.visual.show import show_scene

        logger.info(self.name_unique)
        tform_obj_type = self.tform_obj_type
        cams_tform4x4_world, cams_intr4x4, cams_imgs = self.read_cams(
            cams_count=20,
            show_imgs=True,
            tform_obj_type=tform_obj_type,
        )
        cams_viewpoints = inv_tform4x4(torch.stack(cams_tform4x4_world))[:, :3, 3]

        pts3d, pts3d_colors, pts3d_normals = self.read_pcl(
            tform_obj_type=tform_obj_type,
        )

        mesh_feats_viewpoints = self.read_mesh_feats_viewpoint(
            tform_obj_type=tform_obj_type,
        )
        if isinstance(mesh_feats_viewpoints, list):
            mesh_feats_viewpoints = torch.cat(mesh_feats_viewpoints, dim=0)

        mesh = self.get_mesh()
        logger.info(f"mesh has {len(mesh.verts)} vertices and {len(mesh.faces)} faces.")

        show_scene(
            cams_tform4x4_world=cams_tform4x4_world,
            cams_intr4x4=cams_intr4x4,
            cams_imgs=cams_imgs,
            pts3d_colors=[pts3d_colors],
            pts3d=[pts3d, mesh_feats_viewpoints, cams_viewpoints],
            meshes=[mesh],
        )

    def read_cams(
        self,
        cam_tform4x4_obj_type: OD3D_CAM_TFORM_OBJ_TYPES = None,
        tform_obj_type=None,
        cams_count=5,
        show_imgs=True,
    ):
        cams_tform4x4_world = []
        cams_intr4x4 = []
        cams_imgs = []
        frames_count = len(self.frames_names)
        if cams_count == -1:
            step_size = 1
        else:
            step_size = (frames_count // cams_count) + 1
        for c in range(0, frames_count, step_size):
            frame = self.get_frame_by_index(c)
            cams_tform4x4_world.append(
                frame.read_cam_tform4x4_obj(
                    cam_tform4x4_obj_type=cam_tform4x4_obj_type,
                    tform_obj_type=tform_obj_type,
                ),
            )

            cams_intr4x4.append(frame.read_cam_intr4x4())
            if show_imgs:
                cams_imgs.append(frame.get_rgb())
        return cams_tform4x4_world, cams_intr4x4, cams_imgs

    def get_cams(
        self,
        cam_tform4x4_obj_type: OD3D_CAM_TFORM_OBJ_TYPES = None,
        tform_obj_type=None,
        cams_count=5,
        show_imgs=True,
    ):
        cams_tform4x4_world = []
        cams_intr4x4 = []
        cams_imgs = []
        frames_count = len(self.frames_names)
        if cams_count == -1:
            step_size = 1
        else:
            step_size = (frames_count // cams_count) + 1
        for c in range(0, frames_count, step_size):
            frame = self.get_frame_by_index(c)
            cams_tform4x4_world.append(
                frame.get_cam_tform4x4_obj(
                    cam_tform4x4_obj_type=cam_tform4x4_obj_type,
                    tform_obj_type=tform_obj_type,
                ),
            )

            cams_intr4x4.append(frame.get_cam_intr4x4())
            if show_imgs:
                cams_imgs.append(frame.get_rgb())
        return cams_tform4x4_world, cams_intr4x4, cams_imgs


@dataclass
class OD3D_SequenceCategoryMixin(OD3D_Sequence):
    # frame_type = OD3D_FrameCategoryMixin
    all_categories: List[str]
    map_categories_to_od3d = None

    @property
    def category(self):
        return self.meta.category

    @property
    def category_id(self):
        return self.all_categories.index(self.category)


@dataclass
class OD3D_SequenceSfMMixin(OD3D_SequenceSfMTypeMixin, OD3D_Sequence):
    # frame_type = OD3D_FrameCamIntr4x4Mixin

    def get_min_HW(self):
        return None, None

    def get_sfm_HW(self):
        return None, None

    @property
    def path_sfm(self):
        return self.path_sfm_root.joinpath(self.name_unique)

    @property
    def path_sfm_root(self):
        return self.path_preprocess.joinpath("sfm", f"{self.sfm_type}")

    # @property
    # def path_sfm_cams_tform4x4_obj(self):
    #     return self.path_sfm.joinpath(self.dname_sfm_cams_tform4x4_obj)
    @property
    def path_sfm_cams_tform4x4_obj(self):
        return self.path_sfm.joinpath(
            f"Partial_Ratio_{100* self.partial_ratio}_Percent_start_frame_{self.start_frame_id}_flip_sfm_{self.flip_sfm}"
        )

    @property
    def dname_sfm_cams_tform4x4_obj(self):
        return "cam_tform4x4_obj"

    @property
    def fpath_sfm_pcl(self):
        return self.path_sfm.joinpath(self.fname_sfm_pcl)

    @property
    def fname_sfm_pcl(self):
        return "pcl.ply"

    @property
    def fpath_sfm_rays_center3d(self):
        if self.sfm_type == "colmap50":
            return self.path_sfm.joinpath(
                f"Partial_Ratio_{100* self.partial_ratio}_Percent_start_frame_{self.start_frame_id}_flip_sfm_{self.flip_sfm}",
                self.fname_sfm_rays_center3d,
            )
        else:
            return self.path_sfm.joinpath(self.fname_sfm_rays_center3d)

    @property
    def fname_sfm_rays_center3d(self):
        return "rays_center3d.pt"

    # def get_sfm_cam_tform4x4_obj(self, frame_name):
    #     return torch.load(self.path_sfm_cams_tform4x4_obj.joinpath(f"{frame_name}.pt"))
    def get_sfm_cam_tform4x4_obj(self, frame_name):
        import os

        if os.path.exists(
            self.path_sfm_cams_tform4x4_obj.joinpath(
                os.path.join("extrinsic", f"frame{str(frame_name).zfill(6)}.pt")
            )
        ):
            return torch.load(
                self.path_sfm_cams_tform4x4_obj.joinpath(
                    os.path.join("extrinsic", f"frame{str(frame_name).zfill(6)}.pt")
                )
            )
        else:
            if os.path.exists(
                self.path_sfm_cams_tform4x4_obj.joinpath(
                    os.path.join(
                        "extrinsic", f"frame{str(int(frame_name) + 1).zfill(6)}.pt"
                    )
                )
            ):
                return torch.load(
                    self.path_sfm_cams_tform4x4_obj.joinpath(
                        os.path.join(
                            "extrinsic", f"frame{str(int(frame_name) + 1).zfill(6)}.pt"
                        )
                    )
                )

    def get_sfm_rays_center3d(self):
        return torch.load(self.fpath_sfm_rays_center3d)

    def preprocess_sfm(self, override=False):
        path_sfm = self.path_sfm.joinpath(
            f"Partial_Ratio_{100* self.partial_ratio}_Percent_start_frame_{self.start_frame_id}_flip_sfm_{self.flip_sfm}"
        )

        if not override and path_sfm.exists():
            logger.info(f"path sfm already exists at {path_sfm}")
            return
        else:
            logger.info(
                f"preprocessing sfm for {self.name_unique} with type {self.sfm_type}",
            )

        if self.sfm_type == OD3D_SEQUENCE_SFM_TYPES.DROID:
            path_in = self.path_raw.joinpath("frames", self.name_unique)
            path_out_root = (
                self.path_sfm_root
            )  #  self.path_preprocess.joinpath('droid_slam')
            rpath_out = Path(self.name_unique)

            # note: this is only required if the frames have different sizes
            H, W = self.get_sfm_HW()
            if H is not None and W is not None:
                path_out = path_out_root.joinpath(rpath_out)
                path_in = path_out.joinpath("images")

                import torchvision

                for f_id in range(len(self.frames_names)):
                    frame = self.get_frame_by_index(f_id)
                    rgb = frame.rgb[:, :H, :W].clone()
                    torchvision.io.image.write_jpeg(
                        rgb,
                        filename=str(path_in.joinpath(f"{f_id:05d}" + ".jpg")),
                    )

            from od3d.cv.reconstruction.droid_slam import run_droid_slam

            run_droid_slam(
                path_rgbs=path_in,
                path_out_root=path_out_root,
                rpath_out=rpath_out,
                cam_intr4x4=self.first_frame.get_cam_intr4x4(),
                pcl_fname=self.fname_sfm_pcl,
                rays_center3d_fname=self.fname_sfm_rays_center3d,
                cam_tform_obj_dname=self.dname_sfm_cams_tform4x4_obj,
            )
        elif self.sfm_type == OD3D_SEQUENCE_SFM_TYPES.META:
            from od3d.cv.geometry.fit.rays_center3d import fit_rays_center3d

            logger.info("only need to preprocess rays center3d for meta sfm type")

            frames = self.get_frames()
            device = get_default_device()
            cams_tform4x4_obj = torch.stack(
                [
                    frame.read_cam_tform4x4_obj(tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW)
                    for frame in frames
                ],
                dim=0,
            ).to(device=device)
            center3d = fit_rays_center3d(cams_tform4x4_obj=cams_tform4x4_obj)
            self.fpath_sfm_rays_center3d.parent.mkdir(parents=True, exist_ok=True)
            torch.save(center3d.detach().cpu(), f=self.fpath_sfm_rays_center3d)
            return
        else:
            raise NotImplementedError(f"sfm_type {self.sfm_type} not implemented")

@dataclass
class OD3D_SequencePartialMixin(OD3D_SequenceSfMMixin, OD3D_Sequence):
    # frame_type = OD3D_FrameCamIntr4x4Mixin
    partial_ratio: float
    start_frame_id: int
    use_sph: bool
    use_flipped_feature: bool
    use_sd: bool
    flip_sfm: bool
    mixing_ratio: float

    @property
    def frames_names_unique(self):
        if self._frames_names_unique is None:
            dict_nested_frames = rollup_flattened_dict({self.name_unique: None})
            dict_nested_frames = OD3D_FrameMeta.complete_nested_metas(
                path_meta=self.path_meta,
                dict_nested_metas=dict_nested_frames,
            )
            self._frames_names_unique = OD3D_FrameMeta.unroll_nested_metas(
                dict_nested_meta=dict_nested_frames,
            )
        return self._frames_names_unique[
            self.start_frame_id : self.start_frame_id
            + int(len(self._frames_names_unique) * self.partial_ratio)
        ]

    @property
    def frames_names(self):
        if self._frames_names is None:
            self._frames_names = [
                frame_name.split("/")[-1] for frame_name in self.frames_names_unique
            ]
        return self._frames_names

    @staticmethod
    def get_subset_frames_names_uniform(frames_names, count_max_per_sequence=None):
        if count_max_per_sequence is not None:
            frames_names = [
                frames_names[fid]
                for fid in np.linspace(0, len(frames_names) - 1, count_max_per_sequence)
                .astype(int)
                .tolist()
            ]
        return frames_names

    @property
    def frames_count(self):
        return len(self.frames_names)

    def get_frames(self):
        frames_ids = list(range(self.frames_count))

        frames = [self.get_frame_by_index(frame_id) for frame_id in frames_ids]
        return frames

    def __len__(self):
        return self.frames_count

    def __getitem__(self, idx):
        frame = self.get_frame_by_index(idx)
        frame.item_id = idx
        return self.transform(frame)

    def collate_fn(
        self,
        frames: List[OD3D_Frame],
        device="cpu",
        dtype=torch.float32,
        modalities=None,
    ):
        if modalities is None:
            modalities = self.modalities
        from od3d.datasets.frames import OD3D_Frames

        frames = OD3D_Frames.get_frames_from_list(
            frames,
            modalities=modalities,
            dtype=dtype,
            device=device,
        )
        return frames

    def get_dataloader_partial(self, batch_size=1, shuffle=False, transform=None):
        self.transform = transform
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def get_frame_by_index(self, index: int):
        return self.get_frame_by_name_unique(self.frames_names_unique[index])

    def get_frame_by_name_unique(self, frame_name_unique: str):
        from dataclasses import fields

        frame_fields_names = [field.name for field in fields(self.frame_type)]
        sequence_fields = fields(self)
        all_attrs_except_name_unique = {
            field.name: getattr(self, field.name)
            for field in sequence_fields
            if field.name != "name_unique" and field.name in frame_fields_names
        }
        return self.frame_type(
            name_unique=frame_name_unique,
            **all_attrs_except_name_unique,
        )

    def visualize(self):
        from od3d.cv.visual.show import show_scene

        logger.info(self.name_unique)
        tform_obj_type = self.tform_obj_type
        cams_tform4x4_world, cams_intr4x4, cams_imgs = self.read_cams(
            cams_count=20,
            show_imgs=True,
            tform_obj_type=tform_obj_type,
        )
        cams_viewpoints = inv_tform4x4(torch.stack(cams_tform4x4_world))[:, :3, 3]

        pts3d, pts3d_colors, pts3d_normals = self.read_pcl(
            tform_obj_type=tform_obj_type,
        )

        mesh_feats_viewpoints = self.read_mesh_feats_viewpoint(
            tform_obj_type=tform_obj_type,
        )
        if isinstance(mesh_feats_viewpoints, list):
            mesh_feats_viewpoints = torch.cat(mesh_feats_viewpoints, dim=0)

        mesh = self.get_mesh()
        logger.info(f"mesh has {len(mesh.verts)} vertices and {len(mesh.faces)} faces.")

        show_scene(
            cams_tform4x4_world=cams_tform4x4_world,
            cams_intr4x4=cams_intr4x4,
            cams_imgs=cams_imgs,
            pts3d_colors=[pts3d_colors],
            pts3d=[pts3d, mesh_feats_viewpoints, cams_viewpoints],
            meshes=[mesh],
        )

    def read_cams(
        self,
        cam_tform4x4_obj_type: OD3D_CAM_TFORM_OBJ_TYPES = None,
        tform_obj_type=None,
        cams_count=5,
        show_imgs=True,
    ):
        cams_tform4x4_world = []
        cams_intr4x4 = []
        cams_imgs = []
        frames_count = len(self.frames_names)
        if cams_count == -1:
            step_size = 1
        else:
            step_size = (frames_count // cams_count) + 1
        for c in range(0, frames_count, step_size):
            frame = self.get_frame_by_index(c)
            cams_tform4x4_world.append(
                frame.read_cam_tform4x4_obj(
                    cam_tform4x4_obj_type=cam_tform4x4_obj_type,
                    tform_obj_type=tform_obj_type,
                ),
            )

            cams_intr4x4.append(frame.read_cam_intr4x4())
            if show_imgs:
                cams_imgs.append(frame.get_rgb())
        return cams_tform4x4_world, cams_intr4x4, cams_imgs

    def preprocess_sfm(self, override=False):
        # if not override and self.path_sfm.exists():
        #     logger.info(f"path sfm already exists at {self.path_sfm}")
        #     return
        path_sfm = self.path_sfm.joinpath(
            f"Partial_Ratio_{100* self.partial_ratio}_Percent_start_frame_{self.start_frame_id}_flip_sfm_{self.flip_sfm}",
            "dense",
            "stereo",
            "depth_maps",
        )

        if not override and path_sfm.exists():
            logger.info(f"path sfm already exists at {path_sfm}")
            return
        else:
            logger.info(
                f"preprocessing sfm for {self.name_unique} with type {self.sfm_type}",
            )

        if self.sfm_type == OD3D_SEQUENCE_SFM_TYPES.DROID:
            path_in = self.path_raw.joinpath("frames", self.name_unique)
            path_out_root = (
                self.path_sfm_root
            )  #  self.path_preprocess.joinpath('droid_slam')
            rpath_out = Path(self.name_unique)

            # note: this is only required if the frames have different sizes
            H, W = self.get_sfm_HW()
            if H is not None and W is not None:
                path_out = path_out_root.joinpath(rpath_out)
                path_in = path_out.joinpath("images")

                import torchvision

                for f_id in range(len(self.frames_names)):
                    frame = self.get_frame_by_index(f_id)
                    rgb = frame.rgb[:, :H, :W].clone()
                    torchvision.io.image.write_jpeg(
                        rgb,
                        filename=str(path_in.joinpath(f"{f_id:05d}" + ".jpg")),
                    )

            from od3d.cv.reconstruction.droid_slam import run_droid_slam

            run_droid_slam(
                path_rgbs=path_in,
                path_out_root=path_out_root,
                rpath_out=rpath_out,
                cam_intr4x4=self.first_frame.get_cam_intr4x4(),
                pcl_fname=self.fname_sfm_pcl,
                rays_center3d_fname=self.fname_sfm_rays_center3d,
                cam_tform_obj_dname=self.dname_sfm_cams_tform4x4_obj,
            )
        elif self.sfm_type == OD3D_SEQUENCE_SFM_TYPES.META:
            from od3d.cv.geometry.fit.rays_center3d import fit_rays_center3d

            logger.info("only need to preprocess rays center3d for meta sfm type")

            frames = self.get_frames()
            device = get_default_device()
            cams_tform4x4_obj = torch.stack(
                [
                    frame.read_cam_tform4x4_obj(tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW)
                    for frame in frames
                ],
                dim=0,
            ).to(device=device)

            center3d = fit_rays_center3d(cams_tform4x4_obj=cams_tform4x4_obj)
            self.fpath_sfm_rays_center3d.parent.mkdir(parents=True, exist_ok=True)
            torch.save(center3d.detach().cpu(), f=self.fpath_sfm_rays_center3d)
            return

        elif self.sfm_type == OD3D_SEQUENCE_SFM_TYPES.COLMAP50:
            from od3d.cv.reconstruction.colmap_pipeline import ColmapPipeline

            logger.info("calling COLMAP for partial view videos")
            colmap_pipeline_for_partial_view = ColmapPipeline(
                self.path_raw.joinpath(self.name_unique),
                self.path_sfm,
                self.frames_count,
                ratio=self.partial_ratio,
                start_frame_id=self.start_frame_id,
                flip_sfm=self.flip_sfm,
            )
            colmap_pipeline_for_partial_view.run_colmap(logger)
        else:
            logger.info("************YOU SHOULD NOT SEE THIS MESSAGE!!!************")

from typing import List, Dict


@dataclass
class OD3D_Multiple_SequencesPartialMixin:
    sequence_dict: Dict[str, List[OD3D_SequencePartialMixin]] = field(
        default_factory=dict
    )

    def add_sequence(self, sequence: OD3D_SequencePartialMixin):
        """Add a sequence to the appropriate category list in the dictionary."""
        if sequence.category not in self.sequence_dict:
            self.sequence_dict[sequence.category] = []
        self.sequence_dict[sequence.category].append(sequence)

    def get_sequences_by_category(
        self, category: str
    ) -> List[OD3D_SequencePartialMixin]:
        """Retrieve a list of sequences by category."""
        return self.sequence_dict.get(category, [])

    def get_sequence_partial_ratio(self):
        """Retrieve the partial ratio of the first sequence in the first category."""
        if self.sequence_dict:
            first_category = next(iter(self.sequence_dict))
            if self.sequence_dict[first_category]:
                return self.sequence_dict[first_category][0].partial_ratio
        return None

    def display_sequences(self):
        """Optional method to display all sequences grouped by categories."""
        for category, sequences in self.sequence_dict.items():
            print(f"Category: {category}")
            for sequence in sequences:
                print(sequence)

    def get_sequences_length(self, category):
        return len(self.sequence_dict[category])


@dataclass
class OD3D_SequencePCLMixin(
    OD3D_TformObjMixin,
    OD3D_PCLTypeMixin,
    OD3D_SequenceSfMMixin,
):
    pts3d = None
    pts3d_colors = None
    pts3d_normals = None

    @property
    def fpath_pcl(self):
        return self.get_fpath_pcl()

    def get_fpath_pcl(self, pcl_type=None):
        if pcl_type is None:
            pcl_type = self.pcl_type

        if pcl_type == OD3D_PCL_TYPES.META:
            return self.path_raw.joinpath(self.meta.rfpath_pcl)
        elif pcl_type == OD3D_PCL_TYPES.SFM:
            return self.fpath_sfm_pcl
        elif pcl_type == OD3D_PCL_TYPES.SFM_MASK:
            base_path = self.path_preprocess.joinpath(
                "pcl",
                str(pcl_type),
                self.sfm_type,
                self.name_unique,
                f"Partial_Ratio_{100* self.partial_ratio}_Percent_start_frame_{self.start_frame_id}_flip_sfm_{self.flip_sfm}",
            )
            return base_path.joinpath("pcl.ply")
        else:
            return self.path_preprocess.joinpath(
                "pcl",
                f"{pcl_type}",
                f"{self.sfm_type}",
                self.name_unique,
                "pcl.ply",
            )

    def read_pcl(
        self,
        fpath_pcl=None,
        pcl_type=None,
        device="cuda:0",
        tform_obj_type: OD3D_TFROM_OBJ_TYPES = None,
    ):
        if fpath_pcl is None:
            fpath_pcl = self.get_fpath_pcl(pcl_type=pcl_type)
        if os.path.exists(fpath_pcl):
            pts3d, pts3d_colors, pts3d_normals = read_pts3d_with_colors_and_normals(
                fpath=fpath_pcl,
                device=device,
            )

            tform_obj = self.get_tform_obj(tform_obj_type=tform_obj_type)
            if tform_obj is not None:
                tform_obj = tform_obj.to(device=device)
                pts3d = transf3d_broadcast(pts3d=pts3d, transf4x4=tform_obj)
                pts3d_normals = transf3d_normal_broadcast(
                    normals3d=pts3d_normals,
                    transf4x4=tform_obj,
                )

            if (pcl_type is None or pcl_type == self.pcl_type) and (
                tform_obj_type == self.tform_obj_type or tform_obj_type is None
            ):
                self.pts3d, self.pts3d_colors, self.pts3d_normals = (
                    pts3d,
                    pts3d_colors,
                    pts3d_normals,
                )
            return pts3d, pts3d_colors, pts3d_normals
        else:
            return None, None, None

    def get_pcl(
        self,
        pcl_type=None,
        clone=False,
        device="cpu",
        tform_obj_type: OD3D_TFROM_OBJ_TYPES = None,
    ):
        if (
            self.pts3d is not None
            and (pcl_type is None or pcl_type == self.pcl_type)
            and (tform_obj_type is None or tform_obj_type == self.tform_obj_type)
        ):
            pts3d, pts3d_colors, pts3d_normals = (
                self.pts3d,
                self.pts3d_colors,
                self.pts3d_normals,
            )
        else:
            pts3d, pts3d_colors, pts3d_normals = self.read_pcl(
                pcl_type=pcl_type,
                device=device,
                tform_obj_type=tform_obj_type,
            )
        if not clone:
            return pts3d, pts3d_colors, pts3d_normals
        else:
            return pts3d.clone(), pts3d_colors.clone(), pts3d_normals.clone()

    def preprocess_pcl(self, override=False):
        if self.pcl_type == OD3D_PCL_TYPES.META:
            logger.info("no need to preprocess pcl for meta pcl type")
            return
        elif self.pcl_type == OD3D_PCL_TYPES.SFM:
            logger.info("no need to preprocess pcl for sfm pcl type")
            return
        elif (
            # self.pcl_type == OD3D_PCL_TYPES.SFM_MASK
            # or
            self.pcl_type
            == OD3D_PCL_TYPES.META_MASK
        ):
            # if self.pcl_type == OD3D_PCL_TYPES.SFM_MASK:
            #     pcl_type_in = OD3D_PCL_TYPES.SFM
            # elif self.pcl_type == OD3D_PCL_TYPES.META_MASK:
            pcl_type_in = OD3D_PCL_TYPES.META
            # else:
            #     raise NotImplementedError

            fpath_pcl_out = self.get_fpath_pcl(pcl_type=self.pcl_type)

            if not override and fpath_pcl_out.exists():
                logger.info(f"fpath sfm mask pcl already exists at {fpath_pcl_out}")
                return

            frames = self.get_frames()
            device = get_default_device()

            H, W = self.get_min_HW()
            # note: this is only required if the frames have different sizes
            if H is not None and W is not None:
                masks = torch.stack(
                    [frame.read_mask()[:, :H, :W] for frame in frames],
                    dim=0,
                ).to(device=device)
            else:
                masks = torch.stack([frame.read_mask() for frame in frames], dim=0).to(
                    device=device,
                )

            cams_intr4x4 = torch.stack(
                [frame.read_cam_intr4x4() for frame in frames],
                dim=0,
            ).to(device=device)
            tforms_list = []

            # Iterate over each frame
            for frame in frames:
                # Set the partial_ratio for the current frame
                frame.partial_ratio = self.partial_ratio

                # Read the transformation and append it to the list
                tforms_list.append(
                    frame.read_cam_tform4x4_obj(tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW)
                )

            # Stack the transformations and move to the specified device
            cams_tform4x4_obj = torch.stack(tforms_list, dim=0).to(device=device)

            pts3d, pts3d_colors, pts3d_normals = self.read_pcl(
                pcl_type=pcl_type_in,
                device=device,
                tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW,
            )
            pts3d, pts3d_mask = get_pcl_clean_with_masks(
                pcl=pts3d,
                masks=masks,
                cams_intr4x4=cams_intr4x4,
                cams_tform4x4_obj=cams_tform4x4_obj,
                pts3d_prob_thresh=0.6,
                pts3d_max_count=20000,
                pts3d_count_min=10,
                return_mask=True,
            )
            pts3d_colors = pts3d_colors[pts3d_mask]
            pts3d_normals = pts3d_normals[pts3d_mask]

            write_pts3d_with_colors_and_normals(
                fpath=fpath_pcl_out,
                pts3d=pts3d.detach().cpu(),
                pts3d_colors=pts3d_colors.detach().cpu(),
                pts3d_normals=pts3d_normals.detach().cpu(),
            )

        elif (
            self.pcl_type
            == OD3D_PCL_TYPES.SFM_MASK
            # or self.pcl_type == OD3D_PCL_TYPES.META_MASK
        ):
            # if self.pcl_type == OD3D_PCL_TYPES.SFM_MASK:
            pcl_type_in = OD3D_PCL_TYPES.SFM_MASK
            # elif self.pcl_type == OD3D_PCL_TYPES.META_MASK:
            #     pcl_type_in = OD3D_PCL_TYPES.META
            # else:
            #     raise NotImplementedError

            def extract_indices_from_filenames(directory):
                # Define the regex pattern to match the file names and extract indices
                pattern = re.compile(r"frame(\d+)\.pt")
                # List directory contents
                files = os.listdir(directory)
                # Extract indices from file names using the regex pattern
                indices = [
                    int(pattern.search(file).group(1))
                    for file in files
                    if pattern.search(file)
                ]
                return indices

            device = get_default_device()
            # dirname = f'Partial_Ratio_{100* self.partial_ratio}_Percent'
            dirname = f"Partial_Ratio_{100* self.partial_ratio}_Percent_start_frame_{self.start_frame_id}_flip_sfm_{self.flip_sfm}"
            fpath_pcl_out = self.path_preprocess.joinpath(
                "pcl",
                f"{self.pcl_type}",
                f"{self.sfm_type}",
                self.name_unique,
                dirname,
                "pcl.ply",
            )

            if not override and fpath_pcl_out.exists():
                logger.info(f"fpath sfm mask pcl already exists at {fpath_pcl_out}")
                return

            base_path = self.path_preprocess.joinpath(
                "sfm", self.sfm_type, self.name_unique
            )
            if not os.path.exists(base_path.joinpath(dirname, "dense", "images")):
                return

            # if len(os.listdir(base_path.joinpath(dirname, 'dense', 'images'))) == len(os.listdir(base_path.joinpath(dirname, 'images'))) and len(os.listdir(base_path.joinpath(dirname).joinpath(f'dense/stereo/depth_maps/')))!=0:
            indices = sorted(
                extract_indices_from_filenames(
                    base_path.joinpath(dirname).joinpath("extrinsic")
                )
            )
            frames = self.get_frames()

            H, W = self.get_min_HW()
            # note: this is only required if the frames have different sizes
            if H is not None and W is not None:
                masks = torch.stack(
                    [frame.read_mask()[:, :H, :W] for frame in frames],
                    dim=0,
                ).to(device=device)
            else:
                masks = torch.stack([frame.read_mask() for frame in frames], dim=0).to(
                    device=device,
                )

            cams_intr4x4 = (
                torch.stack(
                    [
                        torch.load(
                            base_path.joinpath(dirname)
                            .joinpath("intrinsic")
                            .joinpath(f"frame{str(index).zfill(6)}.pt")
                        )
                        for index in indices
                    ],
                    dim=0,
                )
                .to(device=device)
                .to(dtype=torch.float32)
            )

            cams_tform4x4_obj = (
                torch.stack(
                    [
                        torch.load(
                            base_path.joinpath(dirname)
                            .joinpath("extrinsic")
                            .joinpath(f"frame{str(index).zfill(6)}.pt")
                        )
                        for index in indices
                    ],
                    dim=0,
                )
                .to(device=device)
                .to(dtype=torch.float32)
            )
            points_3d_list = []
            for k in range(len(indices)):
                index = indices[k]
                start = time.time()
                depth_path = base_path.joinpath(dirname).joinpath(
                    f"dense/stereo/depth_maps/frame{str(index).zfill(6)}.jpg.geometric.bin"
                )
                if not os.path.exists(depth_path):
                    continue
                depth_mask_path = self.path_raw.joinpath(self.name_unique).joinpath(
                    f"depth_masks/frame{str(index).zfill(6)}.png"
                )
                rgb_path = base_path.joinpath(dirname).joinpath(
                    f"dense/images/frame{str(index).zfill(6)}.jpg"
                )
                depth = read_from_depth_binary_array(depth_path)
                depth_mask = Image.open(depth_mask_path)
                H = depth.shape[0]
                W = depth.shape[1]
                rgb = np.array(Image.open(rgb_path).resize((W, H), Image.NEAREST))
                depth_mask = np.array(depth_mask.resize((W, H), Image.NEAREST))
                if self.flip_sfm:
                    depth_mask = np.flip(depth_mask, axis=1)

                masked_depth = depth * depth_mask
                masked_rgb = rgb * np.expand_dims(depth_mask, axis=2)
                points_3d = backproject_with_rgb(
                    masked_depth,
                    masked_rgb,
                    cams_intr4x4[k].cpu().numpy(),
                    np.linalg.inv(cams_tform4x4_obj[k].cpu().numpy()),
                )
                points_3d_list.append(points_3d)
                end = time.time()
                print("time is ", end - start)

            points_3d_list = np.vstack(points_3d_list)
            pts3d_max_count = 20000
            index_list = np.random.randint(0, len(points_3d_list), size=pts3d_max_count)
            points_3d_list = points_3d_list[index_list]

            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points_3d_list[:, :3])
            pcd.colors = open3d.utility.Vector3dVector(points_3d_list[:, 3:] / 255)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.5)
            folder_path = str(
                self.path_preprocess.joinpath(
                    "pcl",
                    f"{self.pcl_type}",
                    f"{self.sfm_type}",
                    self.name_unique,
                    dirname,
                ),
            )
            os.makedirs(folder_path, exist_ok=True)
            open3d.io.write_point_cloud(os.path.join(folder_path, "pcl.ply"), cl)
            # else:
            #     logger.info('No Valid SFM detected!')


    def get_fpath_tform_obj(self, tform_obj_type=None):
        if tform_obj_type is None:
            tform_obj_type = self.tform_obj_type
        if tform_obj_type == OD3D_TFROM_OBJ_TYPES.LABEL3D_CUBOID_META:
            return self.path_preprocess.joinpath(
                "tform_obj",
                f"{tform_obj_type}",
                f"{self.pcl_type}",
                f"{self.sfm_type}",
                self.name_unique,
                f"Partial_Ratio_{100* self.partial_ratio}_Percent_start_frame_{self.start_frame_id}_flip_sfm_{self.flip_sfm}",
                "tform_obj.pt",
            )
        else:
            return self.path_preprocess.joinpath(
                "tform_obj",
                f"{tform_obj_type}",
                f"{self.pcl_type}",
                f"{self.sfm_type}",
                self.name_unique,
                "tform_obj.pt",
            )

    def get_sequence_by_name_unique(self, sequence_name_unique: str):
        from dataclasses import fields

        frame_fields = fields(self)
        sequence_fields_names = [field.name for field in fields(self.__class__)]
        all_attrs_except_name_unique = {
            field.name: getattr(self, field.name)
            for field in frame_fields
            if field.name != "name_unique" and field.name in sequence_fields_names
        }
        return self.__class__(
            name_unique=sequence_name_unique,
            **all_attrs_except_name_unique,
        )

    def get_sequence_by_name_unique_with_start_id(self, sequence_name_unique: str, id):
        from dataclasses import fields

        frame_fields = fields(self)
        sequence_fields_names = [field.name for field in fields(self.__class__)]
        all_attrs_except_name_unique = {
            field.name: getattr(self, field.name)
            for field in frame_fields
            if field.name != "name_unique" and field.name in sequence_fields_names
        }
        return self.__class__(
            name_unique=sequence_name_unique,
            start_frame_id=id,
            **all_attrs_except_name_unique,
        )

    def preprocess_tform_obj(self, override=False, tform_obj_type=None):
        if tform_obj_type is None:
            tform_obj_type = self.tform_obj_type

        from od3d.cv.label.axis import label_axis_in_pcl

        fpath_tform_obj = self.get_fpath_tform_obj(tform_obj_type=tform_obj_type)
        if fpath_tform_obj.exists() and not override:
            logger.info(
                f"Label tform_obj already exists {fpath_tform_obj}, override disabled.",
            )
            return

        if tform_obj_type == OD3D_TFROM_OBJ_TYPES.RAW:
            logger.info(f"No need to preprocess tform_obj for raw tform_obj type")
            return
        elif tform_obj_type == OD3D_TFROM_OBJ_TYPES.LABEL3D_ZSP:
            from od3d.io import read_json
            from od3d.cv.geometry.transform import inv_tform4x4, tform4x4

            ref_seq_name_unique = (
                self.category
                + "/"
                + sorted(
                    list(
                        Path("third_party/zero-shot-pose/data/class_labels")
                        .joinpath(self.category)
                        .iterdir(),
                    ),
                )[0].stem
            )
            ref_seq = self.get_sequence_by_name_unique(ref_seq_name_unique)
            ref_seq.preprocess_tform_obj(
                override=False,
                tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D_CUBOID,
            )
            label3d_cuboid_tform_obj_ref = ref_seq.get_tform_obj(
                tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D_CUBOID,
            )
            scale = (
                label3d_cuboid_tform_obj_ref[:3, :3]
                .norm(dim=-1, keepdim=True)
                .mean(dim=-2, keepdim=True)
            )
            label3d_cuboid_tform_obj_ref[:3] = label3d_cuboid_tform_obj_ref[:3] / scale

            fpath_zsp = Path("third_party/zero-shot-pose/data/class_labels").joinpath(
                self.name_unique + ".json",
            )
            zsp_tform_obj = inv_tform4x4(
                torch.from_numpy(np.array(read_json(fpath_zsp)["trans"])),
            )
            zsp_tform_obj = zsp_tform_obj.to(torch.float)
            scale = (
                zsp_tform_obj[:3, :3]
                .norm(dim=-1, keepdim=True)
                .mean(dim=-2, keepdim=True)
            )
            zsp_tform_obj[:3] = zsp_tform_obj[:3] / scale

            fpath_zsp_ref = Path(
                "third_party/zero-shot-pose/data/class_labels",
            ).joinpath(ref_seq_name_unique + ".json")
            zsp_tform_obj_ref = inv_tform4x4(
                torch.from_numpy(np.array(read_json(fpath_zsp_ref)["trans"])),
            )
            zsp_tform_obj_ref = zsp_tform_obj_ref.to(torch.float)
            scale = (
                zsp_tform_obj_ref[:3, :3]
                .norm(dim=-1, keepdim=True)
                .mean(dim=-2, keepdim=True)
            )
            zsp_tform_obj_ref[:3] = zsp_tform_obj_ref[:3] / scale

            zsp_obj_ref_tform_obj = tform4x4(
                inv_tform4x4(zsp_tform_obj_ref),
                zsp_tform_obj,
            )

            label_tform_obj = tform4x4(
                label3d_cuboid_tform_obj_ref,
                zsp_obj_ref_tform_obj,
            )
            # note: as zsp does not offer scale, we cannot retrieve actual translation, therefore we use this pcl center
            pts3d, pts3d_colors, pts3d_normals = self.read_pcl(
                tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW,
            )
            label_tform_obj[:3, 3] = -pts3d.mean(dim=0)

            self.write_tform_obj(
                tform_obj=label_tform_obj,
                fpath_tform_obj=fpath_tform_obj,
            )

        elif tform_obj_type == OD3D_TFROM_OBJ_TYPES.LABEL3D_ZSP_CUBOID:
            from od3d.datasets.enum import OD3D_CATEGORIES_SIZES_IN_M
            from od3d.cv.geometry.fit.cuboid import fit_cuboid_to_pts3d
            from od3d.cv.geometry.transform import tform4x4

            size = OD3D_CATEGORIES_SIZES_IN_M[
                self.map_categories_to_od3d[self.category]
            ]

            self.preprocess_tform_obj(
                override=override,
                tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D_ZSP,
            )

            tform_obj = self.get_tform_obj(
                tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D_ZSP,
            )

            pts3d, pts3d_colors, pts3d_normals = self.read_pcl(
                tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW,
            )

            _, obj_cuboid_tform_obj = fit_cuboid_to_pts3d(
                pts3d=pts3d,
                size=size,
                optimize_rot=False,
                optimize_transl=True,
                tform_obj_label=tform_obj,
                optimize_steps=100,
            )

            self.write_tform_obj(
                tform_obj=obj_cuboid_tform_obj,
                fpath_tform_obj=fpath_tform_obj,
            )

        elif tform_obj_type == OD3D_TFROM_OBJ_TYPES.LABEL3D:
            fpath_tform_obj.parent.mkdir(parents=True, exist_ok=True)
            cams_tform4x4_world, cams_intr4x4, cams_imgs = self.read_cams(
                cams_count=4,
                show_imgs=True,
                tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW,
            )
            pts3d, pts3d_colors, pts3d_normals = self.read_pcl(
                tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW,
            )

            while True:
                from od3d.cv.geometry.transform import inv_tform4x4

                prev_tform_obj = self.get_tform_obj(tform_obj_type=tform_obj_type)
                # prev_tform_obj = torch.eye(4).to(device=prev_tform_obj.device)
                axis_pts3d = label_axis_in_pcl(
                    pts3d=pts3d,
                    pts3d_colors=pts3d_colors,
                    prev_labeled_pcl_tform_pcl=prev_tform_obj,
                    cams_tform4x4_world=cams_tform4x4_world,
                    cams_intr4x4=cams_intr4x4,
                    cams_imgs=cams_imgs,
                )

                from od3d.cv.geometry.fit.axis_tform_from_pts3d import (
                    axis_tform4x4_obj_from_pts3d,
                )

                if axis_pts3d is None or axis_pts3d.shape != (3, 2, 3):
                    if prev_tform_obj is not None:
                        logger.warning("not overriding previous tform_obj")
                        break

                    logger.warning(
                        f"axis_pts3d is None or axis_pts3d.shape != (3, 2, 3) {axis_pts3d.shape if axis_pts3d is not None else None}",
                    )
                    continue

                tform_obj = axis_tform4x4_obj_from_pts3d(axis_pts3d=axis_pts3d)
                tform_obj[:3, 3] = -pts3d.mean(dim=0)

                if not (torch.linalg.det(tform_obj[:3, :3]) - 1.0).abs() <= 1e-5:
                    logger.warning(
                        f"determinant is not close to 1. {torch.linalg.det(tform_obj[:3, :3])}",
                    )
                    continue

                self.write_tform_obj(
                    tform_obj=tform_obj,
                    fpath_tform_obj=fpath_tform_obj,
                )
                break
        elif tform_obj_type == OD3D_TFROM_OBJ_TYPES.LABEL3D_CUBOID:
            from od3d.datasets.enum import OD3D_CATEGORIES_SIZES_IN_M
            from od3d.cv.geometry.fit.cuboid import fit_cuboid_to_pts3d
            from od3d.cv.geometry.transform import tform4x4

            size = OD3D_CATEGORIES_SIZES_IN_M[
                self.map_categories_to_od3d[self.category]
            ]

            self.preprocess_tform_obj(
                override=override,
                tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D,
            )

            tform_obj = self.get_tform_obj(tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D)
            pts3d, pts3d_colors, pts3d_normals = self.read_pcl(
                tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW,
            )

            # from od3d.cv.visual.show import show_scene
            # show_scene(pts3d=[pts3d], pts3d_colors=[pts3d_colors], pts3d_normals=[pts3d_normals])

            # pts3d_label3d = transf3d_broadcast(pts3d=pts3d, transf4x4=)
            # from od3d.cv.geometry.downsample import voxel_downsampling
            # pts3d_label3d = voxel_downsampling(pts3d_label3d, K=100)
            _, obj_cuboid_tform_obj = fit_cuboid_to_pts3d(
                pts3d=pts3d,
                size=size,
                optimize_rot=False,
                optimize_transl=True,
                optimize_steps=100,
                tform_obj_label=tform_obj,
            )
            logger.info(obj_cuboid_tform_obj)

            # tform_obj = tform4x4(obj_cuboid_tform_obj, tform_obj)

            logger.info(f"write at {fpath_tform_obj}")
            self.write_tform_obj(
                tform_obj=obj_cuboid_tform_obj,
                fpath_tform_obj=fpath_tform_obj,
            )

        elif tform_obj_type == OD3D_TFROM_OBJ_TYPES.LABEL3D_CUBOID_META:
            from copy import deepcopy
            from od3d.cv.geometry.transform import tform4x4, inv_tform4x4

            def extract_indices_from_filenames(directory):
                pattern = re.compile(r"frame(\d+)\.pt")
                files = os.listdir(directory)
                indices = [
                    int(pattern.search(file).group(1))
                    for file in files
                    if pattern.search(file)
                ]

                return indices

            dirname = f"Partial_Ratio_{100* self.partial_ratio}_Percent_start_frame_{self.start_frame_id}_flip_sfm_{self.flip_sfm}"
            sequence_meta = deepcopy(self)
            sequence_meta.cam_tform4x4_obj_type = OD3D_CAM_TFORM_OBJ_TYPES.META
            sequence_meta.pcl_type = OD3D_PCL_TYPES.META_MASK
            sequence_meta.sfm_type = OD3D_SEQUENCE_SFM_TYPES.META
            # sequence_meta.preprocess_tform_obj(
            #     override=override,
            #     tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D_CUBOID,
            # )
            device = get_default_device()
            base_path = self.path_preprocess.joinpath(
                "sfm", self.sfm_type, self.name_unique
            )
            # if len(os.listdir(base_path.joinpath(dirname, 'dense', 'images'))) != len(os.listdir(base_path.joinpath(dirname, 'images'))):
            #     return
            indices = sorted(
                extract_indices_from_filenames(
                    base_path.joinpath(dirname).joinpath("extrinsic")
                )
            )
            if len(indices) == 0:
                return
            cams_intr4x4 = (
                torch.stack(
                    [
                        torch.load(
                            base_path.joinpath(dirname)
                            .joinpath("intrinsic")
                            .joinpath(f"frame{str(index).zfill(6)}.pt")
                        )
                        for index in indices
                    ],
                    dim=0,
                )
                .to(device=device)
                .to(dtype=torch.float32)
            )

            cams_tform4x4_obj = (
                torch.stack(
                    [
                        torch.load(
                            base_path.joinpath(dirname)
                            .joinpath("extrinsic")
                            .joinpath(f"frame{str(index).zfill(6)}.pt")
                        )
                        for index in indices
                    ],
                    dim=0,
                )
                .to(device=device)
                .to(dtype=torch.float32)
            )

            co3d_cam_pose_meta, co3d_intr4x4_meta, _ = sequence_meta.read_cams(
                cams_count=-1,
                show_imgs=True,
                tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW,
            )

            # co3d_cam_pose = torch.stack(co3d_cam_pose[:self.frames_count], dim = 0).to(device= device)
            # co3d_cam_pose = torch.stack(
            #     [co3d_cam_pose_meta[index] for index in range(self.frames_count)],
            #     dim=0,
            # ).to(device=device).to(dtype=torch.float32)
            # co3d_intr4x4 = torch.stack(
            #     [co3d_intr4x4_meta[index] for index in  range(self.frames_count)],
            #     dim=0,
            # ).to(device=device).to(dtype=torch.float32)
            co3d_cam_pose = (
                torch.stack(
                    [co3d_cam_pose_meta[k] for k in range(len(indices))],
                    dim=0,
                )
                .to(device=device)
                .to(dtype=torch.float32)
            )
            co3d_intr4x4 = (
                torch.stack(
                    [co3d_intr4x4_meta[k] for k in range(len(indices))],
                    dim=0,
                )
                .to(device=device)
                .to(dtype=torch.float32)
            )

            inv_co3d_cam_pose = torch.linalg.inv(
                co3d_cam_pose
            )  # transform co3d cam to ref world
            inv_cams_tform4x4_obj = torch.linalg.inv(
                cams_tform4x4_obj
            )  # transform from co3d cam to my colmap world

            intr_scale = (
                torch.mean(cams_intr4x4, dim=0)[0, 0]
                / torch.mean(co3d_intr4x4, dim=0)[0, 0]
            )
            (
                est_transformations,
                rotation,
                scale,
                shift,
            ) = camera_alignment.calculate_offset(
                inv_cams_tform4x4_obj, inv_co3d_cam_pose
            )  # transform from my colmap to co3d colmap
            print("scale is ", scale)
            transformtion = torch.eye(4).cuda()
            transformtion[:3, :3] = rotation * scale * intr_scale
            transformtion[:3, -1] = shift * scale * intr_scale

            from_co3d_colmap_to_reference_mesh_trafo = torch.load(
                sequence_meta.get_fpath_tform_obj(
                    tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D_CUBOID
                )
            ).to(device)
            from_my_colmap_to_reference_mesh_trafo = tform4x4(
                from_co3d_colmap_to_reference_mesh_trafo, transformtion
            )
            logger.info("Writing tform_obj for Label_3D_Cuboid_Meta...")
            self.write_tform_obj(
                tform_obj=from_my_colmap_to_reference_mesh_trafo,
                fpath_tform_obj=fpath_tform_obj,
            )
            # print('from_my_colmap_to_reference_mesh_trafo is ', from_my_colmap_to_reference_mesh_trafo)
            # torch.save(from_my_colmap_to_reference_mesh_trafo,str(fpath_tform_obj) )
            # x = torch.load(str(fpath_tform_obj))
            # print('saved tensor is ', x)

        else:
            raise NotImplementedError(
                f"tform_obj_type {tform_obj_type} not implemented",
            )

        # if axis_pcl is not None and axis_pcl.shape == (3, 2, 3):
        #
        #     logger.info(f'storing axis labeled, cuboid tform, and cuboid ')
        #     torch.save(axis_pcl, f=fpath_axis_droid_slam)
        #
        #     self.fpath_labeled_obj_tform_obj.parent.mkdir(parents=True, exist_ok=True)
        #     torch.save(pcl_labeled_tform_pcl, self.fpath_labeled_obj_tform_obj)
        #
        #     size = OD3D_CATEGORIES_SIZES_IN_M[MAP_CATEGORIES_CO3D_TO_OD3D[self.category]]
        #     pcl_labeled_tform_pts3d = transf3d_broadcast(
        #         pts3d=self.get_pcl(),
        #         transf4x4=pcl_labeled_tform_pcl)
        #     pcl_labeled_cuboid, pcl_labeled_cuboid_tform_pcl_labeled = \
        #         fit_cuboid_to_pts3d(pts3d=pcl_labeled_tform_pts3d, size=size, optimize_rot=False,
        #                             optimize_transl=True)
        #
        #     self.fpath_obj_labeled_cuboid.parent.mkdir(parents=True, exist_ok=True)
        #     pcl_labeled_cuboid.write_to_file(fpath=self.fpath_obj_labeled_cuboid)
        #
        #     self.fpath_labeled_cuboid_obj_tform_labeled_obj.parent.mkdir(parents=True, exist_ok=True)
        #     torch.save(pcl_labeled_cuboid_tform_pcl_labeled.detach().cpu(),
        #                f=self.fpath_labeled_cuboid_obj_tform_labeled_obj)
        # else:
        #     logger.info(f'not storing labeled axis.')


from od3d.cv.geometry.objects3d.meshes import Meshes


@dataclass
class OD3D_SequenceMeshMixin(
    OD3D_MeshFeatsTypeMixin,
    OD3D_MeshTypeMixin,
    OD3D_SequencePCLMixin,
):
    mesh = None
    mesh_feats = None
    mesh_feats_viewpoint = None

    def get_mesh_type_unique(self, mesh_type=None):
        if mesh_type is None:
            mesh_type = self.mesh_type
        if mesh_type == OD3D_MESH_TYPES.META:
            return Path("").joinpath(f"{mesh_type}")
        else:
            return Path("").joinpath(
                f"{mesh_type}",
                f"{self.pcl_type}",
                f"{self.sfm_type}",
            )

    def get_fpath_mesh(self, mesh_type=None):
        if mesh_type is None:
            mesh_type = self.mesh_type
        if mesh_type == OD3D_MESH_TYPES.META:
            return self.path_raw.joinpath(self.meta.rfpath_mesh)
        else:
            return self.path_preprocess.joinpath(
                "mesh",
                self.get_mesh_type_unique(mesh_type),
                self.name_unique,
                f"Partial_Ratio_{100* self.partial_ratio}_Percent_start_frame_{self.start_frame_id}_flip_sfm_{self.flip_sfm}",
                "mesh.ply",
            )

    def write_aligned_mesh_and_tform_obj(
        self,
        mesh: Meshes,
        aligned_obj_tform_obj: torch.Tensor,
        aligned_name: str,
    ):
        mesh_type = f"aligned_N_{aligned_name}"
        fpath_mesh_aligned = self.get_fpath_mesh(mesh_type=mesh_type)
        mesh.write_to_file(fpath=fpath_mesh_aligned)

        tform_obj_type = mesh_type
        fpath_tform_obj_aligned = self.get_fpath_tform_obj(
            tform_obj_type=tform_obj_type,
        )

        tform_obj = self.get_tform_obj()
        if tform_obj is not None:
            tform_obj = tform_obj.to(device=aligned_obj_tform_obj.device)
            aligned_obj_tform_obj = tform4x4(
                aligned_obj_tform_obj.detach().clone(),
                tform_obj,
            )
        fpath_tform_obj_aligned.parent.mkdir(parents=True, exist_ok=True)
        torch.save(aligned_obj_tform_obj.detach().cpu(), f=fpath_tform_obj_aligned)

    @property
    def fpath_mesh(self):
        return self.get_fpath_mesh()

    def read_mesh(self, mesh_type=None, device="cpu", tform_obj_type=None):
        path = self.get_fpath_mesh(mesh_type=mesh_type)
        # print('path is ', path)
        if not os.path.exists(path):
            print("the seq without mesh is ", self)
            return
        mesh = Mesh.load_from_file(
            fpath=self.get_fpath_mesh(mesh_type=mesh_type),
            device=device,
        )

        tform_obj = self.get_tform_obj(tform_obj_type=tform_obj_type)
        if tform_obj is not None:
            tform_obj = tform_obj.to(device=device)
            mesh.verts = transf3d_broadcast(pts3d=mesh.verts, transf4x4=tform_obj)

        if (mesh_type is None or mesh_type == self.mesh_type) and (
            tform_obj_type is None or tform_obj_type == self.tform_obj_type
        ):
            self.mesh = mesh
        return mesh

    def get_mesh(self, mesh_type=None, clone=False, device="cpu", tform_obj_type=None):
        if (
            self.mesh is not None
            and (mesh_type is None or mesh_type == self.mesh_type)
            and (tform_obj_type is None or tform_obj_type == self.tform_obj_type)
        ):
            mesh = self.mesh
        else:
            mesh = self.read_mesh(
                mesh_type=mesh_type,
                device=device,
                tform_obj_type=tform_obj_type,
            )

        if not clone:
            return mesh
        else:
            return mesh.clone()

    def preprocess_mesh(self, override=False):
        if self.fpath_mesh.exists() and not override:
            logger.warning(f"mesh already exists {self.fpath_mesh}")
            return
        else:
            logger.info(
                f"preprocessing mesh for {self.name_unique} with type {self.mesh_type}",
            )

        match = re.match(r"([a-z]+)([0-9]+)", self.mesh_type, re.I)
        if match and len(match.groups()) == 2:
            mesh_type, mesh_vertices_count = match.groups()
            mesh_vertices_count = int(mesh_vertices_count)
        else:
            msg = f"could not retrieve mesh type and vertices count from mesh name {self.mesh_type}"
            raise Exception(msg)

        # fpath_pcl = self.get_fpath_pcl(pcl_source=self.pcl_source)
        # if not fpath_pcl.exists():
        #     self.preprocess_pcl(override=True)
        #
        # if fpath_pcl.exists():
        #     pts3d, pts3d_colors, pts3d_normals = read_pts3d_with_colors_and_normals(fpath_pcl)
        # else:
        #     logger.warning(f'could not preprocess mesh, due to fpath to pcl {fpath_pcl} does not exists.')
        #     return None

        device = get_default_device()
        pts3d, pts3d_colors, pts3d_normals = self.read_pcl(
            tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW,
            device=device,
        )
        if pts3d is None:
            return
        N = pts3d.shape[0]
        if N < 4:
            logger.warning(
                f"Could not estimate mesh for sequence {self.name_unique} due to too few points in raw pcl {N}",
            )
            return

        o3d_pcl = open3d.geometry.PointCloud()
        o3d_pcl.points = open3d.utility.Vector3dVector(pts3d.detach().cpu().numpy())
        o3d_pcl.normals = open3d.utility.Vector3dVector(
            pts3d_normals.detach().cpu().numpy(),
        )  # invalidate existing normals
        o3d_pcl.colors = open3d.utility.Vector3dVector(
            pts3d_colors.detach().cpu().numpy(),
        )

        ## DEBUG BLOCK START
        # open3d.visualization.draw_geometries([o3d_pcl])
        # scams = 30
        # frames = self.get_frames()
        # H, W = frames[0].H, frames[0].W
        # rgb = torch.stack([frame.rgb[:, :int(H * 0.9), :int(W * 0.9)] for frame in frames], dim=0).to(device=self.device)
        # cams_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0).to(device=self.device)
        # cams_tform4x4_obj = torch.stack([frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.PCL) for frame in frames], dim=0).to(device=self.device)
        # show_scene(pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj[::scams], cams_intr4x4=cams_intr4x4[::scams], cams_imgs=rgb[::scams])
        ## DEBUG BLOCK END

        # #### OPTION 1: CONVEX HULL
        if mesh_type == "convex":
            o3d_obj_mesh, _ = o3d_pcl.compute_convex_hull()
            o3d_obj_mesh.compute_vertex_normals()
            logger.info(o3d_obj_mesh)
            o3d_obj_mesh = o3d_obj_mesh.remove_unreferenced_vertices()
            logger.info(o3d_obj_mesh)
            o3d_obj_mesh = o3d_obj_mesh.simplify_quadric_decimation(mesh_vertices_count)
            logger.info(o3d_obj_mesh)
            obj_mesh = Mesh.from_o3d(o3d_obj_mesh, device=device)

        elif mesh_type == "poisson":
            # #### OPTION 2: POISSON (requires normals)

            (
                o3d_obj_mesh,
                densities,
            ) = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                o3d_pcl,
                depth=9,
                linear_fit=False,
            )
            vertices_to_remove = densities < np.quantile(densities, 0.05)
            o3d_obj_mesh.remove_vertices_by_mask(vertices_to_remove)
            logger.info(o3d_obj_mesh)
            o3d_obj_mesh = o3d_obj_mesh.remove_unreferenced_vertices()
            logger.info(o3d_obj_mesh)
            o3d_obj_mesh = o3d_obj_mesh.simplify_quadric_decimation(mesh_vertices_count)
            logger.info(o3d_obj_mesh)
            obj_mesh = Mesh.from_o3d(o3d_obj_mesh, device=device)

        elif mesh_type == "alpha":
            # #### OPTION 3: ALPHA_SHAPE
            pts3d = random_sampling(pts3d, pts3d_max_count=10000)  # 11 GB
            quantile = max(0.01, 3.0 / len(pts3d))
            particle_size = (
                torch.cdist(pts3d[None,], pts3d[None,])
                .quantile(dim=-1, q=quantile)
                .mean()
            )
            alpha = particle_size

            o3d_obj_mesh = None
            while (
                o3d_obj_mesh is None
                or not o3d_obj_mesh.is_watertight()
                or not o3d_obj_mesh.is_vertex_manifold()
            ):
                try:
                    o3d_obj_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                        o3d_pcl,
                        alpha,
                    )
                except Exception as e:
                    logger.warning(f"alpha {alpha} failed with {e}")

                if o3d_obj_mesh is not None:
                    logger.info(o3d_obj_mesh)
                    o3d_obj_mesh = o3d_obj_mesh.remove_unreferenced_vertices()
                    logger.info(o3d_obj_mesh)
                    faces_count = mesh_vertices_count * 2

                    o3d_obj_mesh_downsampled = o3d_obj_mesh
                    vertices_count = len(o3d_obj_mesh_downsampled.vertices)
                    while vertices_count > mesh_vertices_count:
                        faces_count = int(faces_count * 0.9)
                        o3d_obj_mesh_downsampled = (
                            o3d_obj_mesh.simplify_quadric_decimation(
                                target_number_of_triangles=faces_count,
                            )
                        )
                        logger.info(o3d_obj_mesh_downsampled)
                        vertices_count = len(o3d_obj_mesh_downsampled.vertices)

                    obj_mesh = Mesh.from_o3d(o3d_obj_mesh_downsampled, device=device)
                alpha = alpha * 1.3

        elif mesh_type == "alphauniform":
            # #### OPTION 3: ALPHA_SHAPE
            pts3d = random_sampling(pts3d, pts3d_max_count=10000)  # 11 GB
            quantile = max(0.01, 3.0 / len(pts3d))
            particle_size = (
                torch.cdist(pts3d[None,], pts3d[None,])
                .quantile(dim=-1, q=quantile)
                .mean()
            )
            vertices_count = mesh_vertices_count + 1
            alpha = particle_size
            o3d_obj_mesh = None
            while vertices_count > mesh_vertices_count:
                try:
                    o3d_obj_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                        o3d_pcl,
                        alpha,
                    )
                except Exception as e:
                    logger.warning(f"alpha {alpha} failed with {e}")

                from od3d.cv.geometry.mesh_simplification import simplify_mesh

                logger.info(o3d_obj_mesh)
                if (
                    o3d_obj_mesh is not None
                    and o3d_obj_mesh.is_watertight()
                    and o3d_obj_mesh.is_vertex_manifold()
                ):
                    pass
                else:
                    alpha = alpha * 1.3
                    continue
                assert o3d_obj_mesh.is_watertight()
                o3d_obj_mesh_downsampled = simplify_mesh(
                    o3d_obj_mesh,
                    mesh_vertices_count=mesh_vertices_count,
                    isotropic=True,
                    valence_aware=True,
                )
                logger.info(o3d_obj_mesh_downsampled)
                if (
                    o3d_obj_mesh_downsampled.is_watertight()
                    and o3d_obj_mesh_downsampled.is_vertex_manifold()
                ):
                    vertices_count = len(o3d_obj_mesh_downsampled.vertices)
                else:
                    alpha = alpha * 1.3

            assert o3d_obj_mesh_downsampled.is_watertight()
            obj_mesh = Mesh.from_o3d(o3d_obj_mesh_downsampled, device=device)

        elif mesh_type == "alphawrapuniform":
            # #### OPTION 3: ALPHA_SHAPE
            pts3d = random_sampling(pts3d, pts3d_max_count=10000)  # 11 GB
            quantile = max(0.01, 3.0 / len(pts3d))
            particle_size = (
                torch.cdist(pts3d[None,], pts3d[None,])
                .quantile(dim=-1, q=quantile)
                .mean()
            )
            vertices_count = mesh_vertices_count + 1
            alpha = particle_size
            offset = particle_size / 100.0
            while vertices_count > mesh_vertices_count:
                from CGAL.CGAL_Kernel import Point_3
                from CGAL.CGAL_Alpha_wrap_3 import alpha_wrap_3
                from CGAL.CGAL_Polyhedron_3 import Polyhedron_3

                cgal_pts3d = [
                    Point_3(pt[0].item(), pt[1].item(), pt[2].item()) for pt in pts3d
                ]
                cgal_poly = Polyhedron_3()
                alpha_wrap_3(cgal_pts3d, alpha.item(), offset.item(), cgal_poly)
                cgal_poly.points()

                vertices = []
                for v in cgal_poly.vertices():
                    vertices.append([v.point().x(), v.point().y(), v.point().z()])
                vertices = torch.Tensor(vertices)

                # Get faces
                faces = []
                for f in cgal_poly.facets():
                    edge = f.facet_begin()
                    edge = edge.next()
                    face_vertices = []
                    for i in range(f.facet_degree()):
                        vertex = torch.Tensor(
                            [
                                edge.vertex().point().x(),
                                edge.vertex().point().y(),
                                edge.vertex().point().z(),
                            ],
                        )
                        vertex_id = torch.where((vertices == vertex).all(dim=-1))[0]
                        face_vertices.append(vertex_id)
                        edge = edge.next()
                    # Assuming each facet is a triangle
                    assert len(face_vertices) == 3
                    faces.append(face_vertices)
                faces = torch.Tensor(faces).long()

                vertices = open3d.utility.Vector3dVector(
                    vertices.detach().cpu().numpy(),
                )
                faces = open3d.utility.Vector3iVector(faces.detach().cpu().numpy())

                o3d_obj_mesh = open3d.geometry.TriangleMesh(
                    vertices=vertices,
                    triangles=faces,
                )

                from od3d.cv.geometry.mesh_simplification import simplify_mesh

                logger.info(o3d_obj_mesh)
                if o3d_obj_mesh.is_watertight() and o3d_obj_mesh.is_vertex_manifold():
                    pass
                else:
                    alpha = alpha * 1.3
                    continue
                assert o3d_obj_mesh.is_watertight()
                o3d_obj_mesh_downsampled = simplify_mesh(
                    o3d_obj_mesh,
                    mesh_vertices_count=mesh_vertices_count,
                    isotropic=True,
                    valence_aware=True,
                )
                logger.info(o3d_obj_mesh_downsampled)
                if (
                    o3d_obj_mesh_downsampled.is_watertight()
                    and o3d_obj_mesh_downsampled.is_vertex_manifold()
                ):
                    vertices_count = len(o3d_obj_mesh_downsampled.vertices)
                else:
                    alpha = alpha * 1.3

            assert o3d_obj_mesh_downsampled.is_watertight()
            obj_mesh = Mesh.from_o3d(o3d_obj_mesh_downsampled, device=device)

        elif mesh_type == "alphawrap":
            # #### OPTION 3: ALPHAWRAP_SHAPE
            pts3d = random_sampling(pts3d, pts3d_max_count=10000)  # 11 GB
            quantile = max(0.01, 3.0 / len(pts3d))
            particle_size = (
                torch.cdist(pts3d[None,], pts3d[None,])
                .quantile(dim=-1, q=quantile)
                .mean()
            )
            alpha = particle_size
            offset = particle_size / 100.0
            vertices_count = mesh_vertices_count + 1
            while vertices_count > mesh_vertices_count:
                from CGAL.CGAL_Kernel import Point_3
                from CGAL.CGAL_Alpha_wrap_3 import alpha_wrap_3
                from CGAL.CGAL_Polyhedron_3 import Polyhedron_3

                cgal_pts3d = [
                    Point_3(pt[0].item(), pt[1].item(), pt[2].item()) for pt in pts3d
                ]
                cgal_poly = Polyhedron_3()
                alpha_wrap_3(cgal_pts3d, alpha.item(), offset.item(), cgal_poly)
                cgal_poly.points()

                vertices = []
                for v in cgal_poly.vertices():
                    vertices.append([v.point().x(), v.point().y(), v.point().z()])
                vertices = torch.Tensor(vertices)

                # Get faces
                faces = []
                for f in cgal_poly.facets():
                    edge = f.facet_begin()
                    edge = edge.next()
                    face_vertices = []
                    for i in range(f.facet_degree()):
                        vertex = torch.Tensor(
                            [
                                edge.vertex().point().x(),
                                edge.vertex().point().y(),
                                edge.vertex().point().z(),
                            ],
                        )
                        vertex_id = torch.where((vertices == vertex).all(dim=-1))[0]
                        face_vertices.append(vertex_id)
                        edge = edge.next()
                    # Assuming each facet is a triangle
                    assert len(face_vertices) == 3
                    faces.append(face_vertices)
                faces = torch.Tensor(faces).long()

                vertices = open3d.utility.Vector3dVector(
                    vertices.detach().cpu().numpy(),
                )
                faces = open3d.utility.Vector3iVector(faces.detach().cpu().numpy())

                o3d_obj_mesh = open3d.geometry.TriangleMesh(
                    vertices=vertices,
                    triangles=faces,
                )

                assert o3d_obj_mesh.is_watertight()

                logger.info(o3d_obj_mesh)
                vertices_count = len(o3d_obj_mesh.vertices)

                alpha = alpha * 1.3

            logger.info(o3d_obj_mesh)
            obj_mesh = Mesh.from_o3d(o3d_obj_mesh, device=device)

            assert o3d_obj_mesh.is_watertight()

        elif mesh_type == "voxel":
            #### OPTION 4: VOXEL GRID
            from pytorch3d.ops.marching_cubes import marching_cubes

            voxel_grid, voxel_grid_range, voxel_grid_offset = voxel_downsampling(
                pts3d_cls=pts3d,
                K=mesh_vertices_count * 2,
                return_voxel_grid=True,
                min_steps=2,
            )
            # vol_batch(N, D, H, W) ->  (X, Y, Z).permute(2, 0, 1)
            verts, faces = marching_cubes(
                vol_batch=voxel_grid.permute(2, 0, 1)[None,] * 1.0,
                return_local_coords=True,
            )
            faces = faces[0].to(device=device)
            verts = (verts[0].to(device=device) + 1) / 2.0

            obj_mesh = Mesh(
                verts=voxel_grid_offset[None,] + voxel_grid_range[None,] * verts,
                faces=faces,
            )

        elif mesh_type == "cuboid":
            from od3d.cv.geometry.fit.cuboid import fit_cuboid_to_pts3d

            if (
                self.get_tform_obj(tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D_CUBOID)
                is not None
            ):
                tform_obj = self.get_tform_obj(
                    tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D_CUBOID,
                    device=device,
                )

            elif (
                self.get_tform_obj(
                    tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D_ZSP_CUBOID,
                )
                is not None
            ):
                tform_obj = self.get_tform_obj(
                    tform_obj_type=OD3D_TFROM_OBJ_TYPES.LABEL3D_ZSP_CUBOID,
                    device=device,
                )

            else:
                msg = f"Could not find tform_obj for cuboid type"
                raise NotImplementedError(msg)

            cannical_pts3d = transf3d_broadcast(pts3d=pts3d, transf4x4=tform_obj)

            cuboids, _ = fit_cuboid_to_pts3d(
                pts3d=cannical_pts3d,
                optimize_rot=False,
                optimize_transl=False,
                vertices_max_count=mesh_vertices_count,
            )

            cuboids.verts.data = transf3d_broadcast(
                pts3d=cuboids.verts,
                transf4x4=inv_tform4x4(tform_obj),
            )
            obj_mesh = cuboids.get_mesh_with_id(0)
        else:
            msg = f"Unknown mesh type {mesh_type}"
            raise Exception(msg)

        # visualization...
        ## DEBUG BLOCK START
        # open3d.visualization.draw(o3d_pcl)
        # open3d.visualization.draw(o3d_obj_mesh)
        ## DEBUG BLOCK END

        obj_mesh.write_to_file(fpath=self.fpath_mesh)

        ## DEBUG BLOCK START
        # scams = 30
        # frames = self.get_frames()
        # H, W = frames[0].H, frames[0].W
        # rgb = torch.stack([frame.rgb[:, :int(H * 0.9), :int(W * 0.9)] for frame in frames], dim=0).to(device=self.device)
        # cams_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0).to(device=self.device)
        # cams_tform4x4_obj = torch.stack([frame.cam_tform4x4_obj for frame in frames], dim=0).to(device=self.device)
        # show_scene(meshes=[obj_mesh], pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj[::scams], cams_intr4x4=cams_intr4x4[::scams], cams_imgs=rgb[::scams])
        # ## DEBUG BLOCK END

    def get_fpath_mesh_feats(
        self, mesh_type=None, mesh_feats_type=None, sph_type=None, ratio=None
    ):
        if mesh_type is None:
            mesh_type = self.mesh_type
        if mesh_feats_type is None:
            mesh_feats_type = self.mesh_feats_type
        if sph_type is None:
            sph_type = self.use_sph
        if ratio is None:
            ratio = self.partial_ratio
        return self.path_preprocess.joinpath(
            "feats",
            f"{mesh_feats_type}",
            f"{mesh_type}",
            f"{self.pcl_type}",
            f"{self.sfm_type}",
            self.name_unique,
            f"Partial_Ratio_{100* ratio}_Percent_start_frame_{self.start_frame_id}_use_sph_{sph_type}_use_flipped_feature_{self.use_flipped_feature}_use_sd_{self.use_sd}_flip_sfm_{self.flip_sfm}_mixing_ratio_{100 * self.mixing_ratio}",
            "mesh_feats.pt",
        )

    def get_fpath_mesh_feats_viewpoint(self, mesh_type=None, mesh_feats_type=None):
        if mesh_type is None:
            mesh_type = self.mesh_type
        if mesh_feats_type is None:
            mesh_feats_type = self.mesh_feats_type

        return self.path_preprocess.joinpath(
            "feats",
            f"{mesh_feats_type}",
            f"{mesh_type}",
            f"{self.pcl_type}",
            f"{self.sfm_type}",
            self.name_unique,
            f"Partial_Ratio_{100* self.partial_ratio}_Percent_start_frame_{self.start_frame_id}_use_sph_{self.use_sph}_use_flipped_feature_{self.use_flipped_feature}_use_sd_{self.use_sd}_flip_sfm_{self.flip_sfm}_mixing_ratio_{100 * self.mixing_ratio}",
            "mesh_feats_viewpoint.pt",
        )
    
    @property
    def fpath_mesh_feats(self):
        return self.get_fpath_mesh_feats()

    @property
    def fpath_mesh_feats_viewpoint(self):
        return self.get_fpath_mesh_feats_viewpoint()
    
    def read_mesh_feats(
        self, mesh_type=None, mesh_feats_type=None, cache=True, sph_type=None
    ):
        fpath_mesh_feats = self.get_fpath_mesh_feats(
            mesh_type=mesh_type,
            mesh_feats_type=mesh_feats_type,
            sph_type=sph_type,
        )
        if not os.path.exists(fpath_mesh_feats):
            print("the path is not existing! ", fpath_mesh_feats)
            return
        mesh_feats = torch.load(fpath_mesh_feats)
        if (
            cache is True
            and (mesh_type is None or mesh_type == self.mesh_type)
            and (mesh_feats_type is None or mesh_feats_type == self.mesh_feats_type)
        ):
            self.mesh_feats = mesh_feats
        return mesh_feats

    def read_mesh_feats_viewpoint(
        self,
        mesh_type=None,
        mesh_feats_type=None,
        tform_obj_type=None,
        device="cpu",
    ):
        fpath_mesh_feats_viewpoint = self.get_fpath_mesh_feats_viewpoint(
            mesh_type=mesh_type,
            mesh_feats_type=mesh_feats_type,
        )
        mesh_feats_viewpoint = torch.load(fpath_mesh_feats_viewpoint)
        if isinstance(mesh_feats_viewpoint, list):
            mesh_feats_viewpoint = [f.to(device=device) for f in mesh_feats_viewpoint]
        else:
            mesh_feats_viewpoint = mesh_feats_viewpoint.to(device=device)

        tform_obj = self.get_tform_obj(tform_obj_type=tform_obj_type)
        if tform_obj is not None:
            tform_obj = tform_obj.to(device=device)
            if isinstance(mesh_feats_viewpoint, list):
                mesh_feats_viewpoint = [
                    transf3d_broadcast(pts3d=f, transf4x4=tform_obj)
                    for f in mesh_feats_viewpoint
                ]
            else:
                mesh_feats_viewpoint = transf3d_broadcast(
                    pts3d=mesh_feats_viewpoint,
                    transf4x4=tform_obj,
                )

        if (
            (mesh_type is None or mesh_type == self.mesh_type)
            and (mesh_feats_type is None or mesh_feats_type == self.mesh_feats_type)
            and (tform_obj_type is None or tform_obj_type == self.tform_obj_type)
        ):
            self.mesh_feats_viewpoint = mesh_feats_viewpoint
        return mesh_feats_viewpoint

    def get_mesh_feats(self, mesh_type=None, mesh_feats_type=None, clone=False):
        if (
            (mesh_type is None or mesh_type == self.mesh_type)
            and (mesh_feats_type is None or mesh_feats_type == self.mesh_feats_type)
            and self.mesh_feats is not None
        ):
            mesh_feats = self.mesh_feats
        else:
            mesh_feats = self.read_mesh_feats(
                mesh_type=mesh_type,
                mesh_feats_type=mesh_feats_type,
            )

        if not clone:
            return mesh_feats
        else:
            return mesh_feats.clone()

    def get_mesh_feats_viewpoint(
        self,
        mesh_type=None,
        mesh_feats_type=None,
        tform_obj_type=None,
        clone=False,
    ):
        if (
            (mesh_type is None or mesh_type == self.mesh_type)
            and (mesh_feats_type is None or mesh_feats_type == self.mesh_feats_type)
            and (tform_obj_type is None or tform_obj_type == self.tform_obj_type)
            and self.mesh_feats_viewpoint is not None
        ):
            mesh_feats_viewpoint = self.mesh_feats_viewpoint
        else:
            mesh_feats_viewpoint = self.read_mesh_feats_viewpoint(
                mesh_type=mesh_type,
                mesh_feats_type=mesh_feats_type,
                tform_obj_type=tform_obj_type,
            )

        if not clone:
            return mesh_feats_viewpoint
        else:
            return mesh_feats_viewpoint.clone()

    def preprocess_dino_feats(self, batch_size, override=False):
        from od3d.models.model import OD3D_Model
        from od3d.cv.transforms.transform import OD3D_Transform
        from od3d.cv.transforms.sequential import SequentialTransform
        from tqdm import tqdm
        import re
        device = get_default_device()

        # e.g.: 'M_dinov2_frozen_base_T_centerzoom512_R_acc'
        match = re.match(
            r"M_([a-z0-9_]+)_T_([a-z0-9_]+)_R_([a-z0-9_]+)",
            self.mesh_feats_type,
            re.I,
        )
        if match and len(match.groups()) == 3:
            model_name, transform_name, reduce_type = match.groups()
        else:
            msg = f"could not retrieve model, transform, and reduce type from mesh feats type {self.mesh_feats_type}"
            raise Exception(msg)

        # if self.mesh_feats_type == FEATURE_TYPES.
        model = OD3D_Model.create_by_name(model_name)
        model.cuda()
        model.eval()
        transform = SequentialTransform(
            [OD3D_Transform.create_by_name(transform_name), model.transform],
        )
        dataloader = self.get_dataloader_partial(
            batch_size=batch_size,
            shuffle=False,
            transform=transform,
        )
        
        total_dino_features_list = []
        for batch in tqdm(iter(dataloader)):
            B = len(batch)
            batch.to(device=device)
            mask = (batch.mask > 0.5).float()
            imgs = batch.rgb
            logger.info(f"rgb image shape {imgs.shape}")
            logger.info(f"mask image shape {mask.shape}")
            
            dino_feat = model(imgs)
            B, C_dino, H, W = dino_feat.shape
            logger.info(f"dino feature map shape {dino_feat.shape}")
            mask_resized = F.interpolate(mask, size=(H, W), mode='nearest')
            dino_features_list = []
                
            from od3d.datasets.pca_util import mask_features
            for b in range(B):
                masked_dino_features, _ = mask_features(dino_feat[b], mask_resized[b])
                logger.info(f"masked dino features: {masked_dino_features.shape}")
                dino_features_list.append(masked_dino_features)
            total_dino_features_list.append(torch.cat(dino_features_list, dim=0))
            
        total_dino_features_tensor = torch.cat(total_dino_features_list)
        logger.info(f"concatenated dino features shape: {total_dino_features_tensor.shape}")
        return total_dino_features_tensor


    def preprocess_raw_feats(self, seq_root_path, dino_feats, dino_mean, dino_proj, batch_size, visualization, override=False):
        from od3d.models.model import OD3D_Model
        from od3d.cv.transforms.transform import OD3D_Transform
        from od3d.cv.transforms.sequential import SequentialTransform
        from tqdm import tqdm
        import re
        from od3d.SphericalMaps.get_feature import my_get_feature
        from od3d.SphericalMaps.dino_mapper import MyDINOMapper
        from od3d.datasets.pca_util import mask_features, visualize_features

        device = get_default_device()
 
        # e.g.: 'M_dinov2_frozen_base_T_centerzoom512_R_acc'
        match = re.match(
            r"M_([a-z0-9_]+)_T_([a-z0-9_]+)_R_([a-z0-9_]+)",
            self.mesh_feats_type,
            re.I,
        )
        if match and len(match.groups()) == 3:
            model_name, transform_name, reduce_type = match.groups()
        else:
            msg = f"could not retrieve model, transform, and reduce type from mesh feats type {self.mesh_feats_type}"
            raise Exception(msg)
        
        if not self.use_sph:
            logger.info("Not using sph features")
            return False
      
        model = OD3D_Model.create_by_name(model_name)
        model.cuda()
        model.eval()
            
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if self.use_sph == "save_raw_features":
            sph_mapper = MyDINOMapper(
                backbone="dinov2_vitb14_frozen_base_no_norm", n_cats=114
            )
            sph_mapper.load_checkpoint(
                f'{root}/SphericalMaps/exps/olaf/exp_002_in3d_wo_sym_and_co3d_200.pth',
                device=device,
            )
            sph_mapper.to(device)
        sph_mapper = sph_mapper.to(device)
        
        transform = SequentialTransform(
            [OD3D_Transform.create_by_name(transform_name), model.transform],
        )
        dataloader = self.get_dataloader_partial(
            batch_size=batch_size,
            shuffle=False,
            transform=transform,
        )
        s_pixel, e_pixel = 0, 0
        idx = 0
        total_dino_features_list = []
        total_sph_features_list = []
        for batch in tqdm(iter(dataloader)):
            B = len(batch)
            batch.to(device=device)
            mask = (batch.mask > 0.5).float()
            imgs = batch.rgb
            logger.info(f"rgb image shape {imgs.shape}")
            logger.info(f"mask image shape {mask.shape}")

            sph_feats = my_get_feature(imgs, sph_mapper)
            logger.info(f"spherical feature map shape {sph_feats.shape}")
            B, C, H, W = sph_feats.shape
            mask_resized = F.interpolate(mask, size=(H, W), mode='nearest')
            for b in range(B):
                masked_sph_features, mask_coords = mask_features(sph_feats[b], mask_resized[b])
                n_pixel = torch.count_nonzero(mask_resized[b]).item()
                e_pixel = s_pixel + n_pixel
                masked_dino_features = dino_feats[s_pixel:e_pixel,:]
                logger.info(f"start: {s_pixel}, end: {e_pixel}, n_pixel: {n_pixel}")
                s_pixel = e_pixel
                
                masked_dino_features = masked_dino_features.cpu()
                masked_dino_features = masked_dino_features.cpu()
                
                logger.info(f"masked dino features: {masked_dino_features.shape}")
                logger.info(f"masked sph features: {masked_sph_features.shape}")
                                
                img_root_path = self.path_preprocess.joinpath(
                    "feats_img",
                    self.name_unique,
                )
                if not os.path.exists(img_root_path):
                    os.makedirs(img_root_path)
                
                if visualization:
                    reconstructed_dino = torch.einsum("ND,CD->NC", masked_dino_features, dino_proj) + dino_mean
                    visualize_features(reconstructed_dino, masked_sph_features, mask_coords, (H, W), idx, output_dir=img_root_path)
                    idx += 1
                
                total_dino_features_list.append(masked_dino_features)
                total_sph_features_list.append(masked_sph_features)
            
        total_dino_features_tensor = torch.cat(total_dino_features_list, dim=0)
        total_sph_features_tensor = torch.cat(total_sph_features_list, dim=0)

        logger.info(f"concatenated dino features shape: {total_dino_features_tensor.shape}")
        logger.info(f"concatenated sph features shape: {total_sph_features_tensor.shape}")
        
        if not os.path.exists(seq_root_path):
            os.makedirs(seq_root_path)
        
        torch.save(total_dino_features_tensor, f=os.path.join(seq_root_path, "dino_feats.pt"))
        logger.info(f"save dino feat at {seq_root_path}/dino_feats.pt")
        torch.save(total_sph_features_tensor, f=os.path.join(seq_root_path, "sph_feats.pt"))
        logger.info(f"save sph feat at {seq_root_path}/sph_feats.pt")
        
        del total_dino_features_tensor, total_sph_features_tensor
        torch.cuda.empty_cache()
        

    def preprocess_mesh_feats(self, category, sequence_name_unique, batch_size, feats_type="dino", override=False):
        # TODO: check if this works
        from od3d.models.model import OD3D_Model
        from od3d.cv.transforms.transform import OD3D_Transform
        from od3d.cv.transforms.sequential import SequentialTransform
        from od3d.cv.geometry.transform import inv_tform4x4
        from tqdm import tqdm
        import re
        from od3d.cv.geometry.objects3d.meshes import Meshes
        from od3d.cv.visual.sample import sample_pxl2d_pts
        device = get_default_device()

        # e.g.: 'M_dinov2_frozen_base_T_centerzoom512_R_acc'
        match = re.match(
            r"M_([a-z0-9_]+)_T_([a-z0-9_]+)_R_([a-z0-9_]+)",
            self.mesh_feats_type,
            re.I,
        )
        if match and len(match.groups()) == 3:
            model_name, transform_name, reduce_type = match.groups()
        else:
            msg = f"could not retrieve model, transform, and reduce type from mesh feats type {self.mesh_feats_type}"
            raise Exception(msg)

        seq_root_path = self.path_preprocess.joinpath(
            "raw_feats",
            category,
            sequence_name_unique
        )
        if feats_type == "dino":
            raw_feats = torch.load(f"{seq_root_path}/dino_feats.pt", map_location=device)
        elif feats_type == "sph":
            raw_feats = torch.load(f"{seq_root_path}/sph_feats.pt", map_location=device)

        logger.info(f"{feats_type}_feats shape: {raw_feats.shape}")

        # if self.mesh_feats_type == FEATURE_TYPES.
        model = OD3D_Model.create_by_name(model_name)
        model.cuda()
        model.eval()
        transform = SequentialTransform(
            [OD3D_Transform.create_by_name(transform_name), model.transform],
        )
        dataloader = self.get_dataloader_partial(
            batch_size=batch_size,
            shuffle=False,
            transform=transform,
        )  # 11 GB

        down_sample_rate = model.downsample_rate
        feature_dim = raw_feats.shape[-1]
        self.mesh = None
        mesh = (
            self.get_mesh()
        )  # already transfered the mesh coordiniate into ref mesh coordinaate system
        if mesh is None:
            return
        meshes = Meshes.read_from_meshes([mesh], device=device)

        meshes_verts_aggregated_features = [
            torch.zeros((0, feature_dim), device="cpu"),
        ] * meshes.verts.shape[0]

        meshes_verts_aggregated_viewpoints = [
            torch.zeros((0, 3), device="cpu"),
        ] * meshes.verts.shape[0]
        vertices_count = len(meshes_verts_aggregated_features)
        print("vertices_count", vertices_count)

        s_idx = 0
        for batch in tqdm(iter(dataloader)):
            # INFO: not all vertices of the mesh are visible in all frames due to occlusions or the image not caputring that
            # part of the object. If a vertex is not visible in an image, the image should not contribute neither to the mean, 
            # nor the covariance calculation. 
            B = len(batch)
            batch.to(device=device)
            mask = (batch.mask > 0.5).float()
            mask_resized = F.interpolate(mask, size=(32, 32), mode='nearest')
            batch.cam_tform4x4_obj = batch.cam_tform4x4_obj.detach()

            vts2d, vts2d_mask = meshes.verts2d(
                cams_intr4x4=batch.cam_intr4x4,
                cams_tform4x4_obj=batch.cam_tform4x4_obj,
                imgs_sizes=batch.size,
                mesh_ids=[0] * B,
                down_sample_rate=down_sample_rate,
            )
           
            from od3d.cv.visual.resize import resize
            
            rgb_mask_low_res = resize(
                batch.rgb_mask,
                scale_factor=1.0 / down_sample_rate,
            )
            
            vts2d_mask *= sample_pxl2d_pts(rgb_mask_low_res, pxl2d=vts2d)[:, :, 0]
            batch_cam_tform4x4_obj_raw = batch.cam_tform4x4_obj
            tform_obj = self.get_tform_obj(device=device)
            if tform_obj is not None:
                batch_cam_tform4x4_obj_raw = tform4x4_broadcast(
                    batch_cam_tform4x4_obj_raw,
                    tform_obj[None,],
                )

            viewpoints3d = (inv_tform4x4(batch_cam_tform4x4_obj_raw)[:, :3, 3])[
                :,
                None,
            ].expand(*vts2d_mask.shape, 3)
            viewpoints3d = viewpoints3d[vts2d_mask]

            N = vts2d.shape[1]
            batch_vts_ids = meshes.get_verts_and_noise_ids_stacked(
                [0] * B,
                count_noise_ids=0,
            )

            batch_vts_ids = torch.cat(
                [batch_vts_ids[:, :N][vts2d_mask], batch_vts_ids[:, N:].reshape(-1)],
                dim=0,
            )
            net_feats = []
            s_idx, e_idx = 0, 0
            for b in range(B):
                n_pixels = int(mask_resized[b].sum().item())
                e_idx = s_idx + n_pixels
                pxl2d=vts2d[b]
                from od3d.cv.visual.sample import sample_pxl2d_pts_with_features
                img_feats = sample_pxl2d_pts_with_features(
                    raw_feats[s_idx:e_idx,:],
                    pxl2d,
                    mask_resized[b],
                )
                s_idx = e_idx
                net_feats.append(img_feats)
            net_feats = torch.stack(net_feats, dim=0)
            C = net_feats.shape[-1]
            
            # N x C
            net_feats = torch.cat(
                [net_feats[:, :N][vts2d_mask], net_feats[:, N:].reshape(-1, C)],
                dim=0,
            )

            for b, vertex_id in enumerate(batch_vts_ids):
                # logger.info(f"net_feats[b : b + 1]: {net_feats[b : b + 1].shape}")
                # logger.info(f"meshes_verts_aggregated_features[vertex_id]: {meshes_verts_aggregated_features[vertex_id].shape}")
                
                meshes_verts_aggregated_features[vertex_id] = torch.cat(
                    [
                        net_feats[b : b + 1].detach().cpu(),
                        meshes_verts_aggregated_features[vertex_id].detach().cpu(),
                    ],
                    dim=0,
                )
                
                meshes_verts_aggregated_viewpoints[vertex_id] = torch.cat(
                    [
                        viewpoints3d[b : b + 1].detach().cpu(),
                        meshes_verts_aggregated_viewpoints[vertex_id].detach().cpu(),
                    ],
                    dim=0,
                )
                  
        mesh_root_path = self.path_preprocess.joinpath(
            "raw_mesh_feats",
            category,
            sequence_name_unique,
            feats_type,
        )
        if not os.path.exists(mesh_root_path):
            os.makedirs(mesh_root_path)
            
        logger.info(f"type of meshes_verts_aggregated_features: {type(meshes_verts_aggregated_features)}")
        logger.info(f"type of meshes_verts_aggregated_viewpoints: {type(meshes_verts_aggregated_viewpoints)}")
        
        if reduce_type == "acc":
            torch.save(meshes_verts_aggregated_features, f=f"{mesh_root_path}/mesh_feats.pt")
            torch.save(
                meshes_verts_aggregated_viewpoints,
                f=f"{mesh_root_path}/mesh_feats_viewpoint.pt",
            )

            logger.info(f"save {feats_type} mesh feats at {mesh_root_path}/mesh_feats.pt")
            logger.info(f"save {feats_type} mesh feats viewpoint at {mesh_root_path}/mesh_feats_viewpoint.pt")

            meshes_verts_aggregated_features_avg = None
            meshes_verts_aggregated_features_avg = torch.stack(
                [
                    agg_feats[(agg_feats != 0).any(dim=1)] # remove masking vertex and background
                    .mean(dim=0)
                    for agg_feats in meshes_verts_aggregated_features
                ],
                dim=0,
            )            
            ## save mean
            if len(meshes_verts_aggregated_features) > 0:
                torch.save(
                    meshes_verts_aggregated_features_avg.detach().cpu(),
                    f=f"{mesh_root_path}/mesh_feats_mean.pt"
                )
            del meshes_verts_aggregated_features_avg
            logger.info(f"save {feats_type} mean of mesh feats at {mesh_root_path}/mesh_feats_mean.pt")

            mesh_verts_aggregated_features_cov = None
            mesh_verts_aggregated_features_cov = torch.stack(
                [
                    torch.cov(agg_feats[(agg_feats != 0).any(dim=1)].T) # remove masking vertex and background
                    for agg_feats in meshes_verts_aggregated_features
                ],
                dim=0
            )
            if len(mesh_verts_aggregated_features_cov) > 0:
                torch.save(
                    mesh_verts_aggregated_features_cov,
                    f=f"{mesh_root_path}/mesh_feats_cov.pt"
                )
            del mesh_verts_aggregated_features_cov
            logger.info(f"save {feats_type} covariance matrix of mesh feats at {mesh_root_path}/mesh_feats_cov.pt")
        else:
            logger.warning(f"Unknown mesh feature reduce_type {reduce_type}.")

        del dataloader
        del model
        torch.cuda.empty_cache()


    def preprocess_mesh_feats_baseline(self, batch_size, override=False):
        from od3d.models.model import OD3D_Model
        from od3d.cv.transforms.transform import OD3D_Transform
        from od3d.cv.transforms.sequential import SequentialTransform
        from od3d.cv.geometry.transform import inv_tform4x4
        from tqdm import tqdm
        import re
        from od3d.cv.geometry.objects3d.meshes import Meshes
        from od3d.cv.visual.sample import sample_pxl2d_pts
        from od3d.SphericalMaps.get_feature import get_feature, my_get_feature
        from od3d.SphericalMaps.dino_mapper import DINOMapper, MyDINOMapper

        if (
            not override
            and self.fpath_mesh_feats.exists()
            and self.fpath_mesh_feats_viewpoint.exists()
        ):
            logger.info(f"mesh feats already exist at {self.fpath_mesh_feats}")
            return

        if override and (
            self.fpath_mesh_feats.exists() or self.fpath_mesh_feats_viewpoint.exists()
        ):
            logger.info(f"overriding mesh feats at {self.fpath_mesh_feats}")
            # self.remove_mesh_feats_preprocess_dependent_files()

        device = get_default_device()

        # e.g.: 'M_dinov2_frozen_base_T_centerzoom512_R_acc'
        match = re.match(
            r"M_([a-z0-9_]+)_T_([a-z0-9_]+)_R_([a-z0-9_]+)",
            self.mesh_feats_type,
            re.I,
        )
        if match and len(match.groups()) == 3:
            model_name, transform_name, reduce_type = match.groups()
        else:
            msg = f"could not retrieve model, transform, and reduce type from mesh feats type {self.mesh_feats_type}"
            raise Exception(msg)
 
        # if self.mesh_feats_type == FEATURE_TYPES.
        model = OD3D_Model.create_by_name(model_name)
        model.cuda()
        model.eval()
        transform = SequentialTransform(
            [OD3D_Transform.create_by_name(transform_name), model.transform],
        )
        dataloader = self.get_dataloader_partial(
            batch_size=batch_size,
            shuffle=False,
            transform=transform,
        )  # 11 GB

        down_sample_rate = model.downsample_rate
        if self.use_sph:
            feature_dim = 3
            if self.use_sph == "sph_excludes_co3d_with_dino":
                feature_dim = 3 + model.out_dim
                if self.use_sd:
                    feature_dim += 384
        else:
            feature_dim = model.out_dim
            if self.use_sd: 
                feature_dim = feature_dim + 384
                # feature_dim = 768
        self.mesh = None
        mesh = (
            self.get_mesh()
        )  # already transfered the mesh coordiniate into ref mesh coordinaate system
        if mesh is None:
            return
        meshes = Meshes.read_from_meshes([mesh], device=device)

        ## DEBUG BLOCK START
        # cams_tform4x4_world, cams_intr4x4, cams_imgs = self.get_cams(CAM_TFORM_OBJ_SOURCES.PCL)
        # show_scene(meshes=meshes, cams_tform4x4_world=cams_tform4x4_world, cams_intr4x4=cams_intr4x4, cams_imgs=cams_imgs )
        ## DEBUG BLOCK END

        meshes_verts_aggregated_features = [
            torch.zeros((0, feature_dim), device="cpu"),
        ] * meshes.verts.shape[0]

        meshes_verts_aggregated_features_test = [
            torch.zeros((0, feature_dim), device="cpu"),
        ] * meshes.verts.shape[0]

        meshes_verts_aggregated_viewpoints = [
            torch.zeros((0, 3), device="cpu"),
        ] * meshes.verts.shape[0]
        vertices_count = len(meshes_verts_aggregated_features)
        print("vertices_count", vertices_count)
   
        for batch in tqdm(iter(dataloader)):
            B = len(batch)  # 6, 3, 512, 512
            batch.to(device=device)            
            batch.cam_tform4x4_obj = batch.cam_tform4x4_obj.detach()

            vts2d, vts2d_mask = meshes.verts2d(
                cams_intr4x4=batch.cam_intr4x4,
                cams_tform4x4_obj=batch.cam_tform4x4_obj,
                imgs_sizes=batch.size,
                mesh_ids=[0] * B,
                down_sample_rate=down_sample_rate,
            )
           
            from od3d.cv.visual.resize import resize
            
            rgb_mask_low_res = resize(
                batch.rgb_mask,
                scale_factor=1.0 / down_sample_rate,
            )
            
            vts2d_mask *= sample_pxl2d_pts(rgb_mask_low_res, pxl2d=vts2d)[:, :, 0]
         
            batch_cam_tform4x4_obj_raw = batch.cam_tform4x4_obj
            tform_obj = self.get_tform_obj(device=device)
            if tform_obj is not None:
                batch_cam_tform4x4_obj_raw = tform4x4_broadcast(
                    batch_cam_tform4x4_obj_raw,
                    tform_obj[None,],
                )

            viewpoints3d = (inv_tform4x4(batch_cam_tform4x4_obj_raw)[:, :3, 3])[
                :,
                None,
            ].expand(*vts2d_mask.shape, 3)
            viewpoints3d = viewpoints3d[vts2d_mask]
            
            N = vts2d.shape[1]
            if not self.use_sph:
                # B x C x H x W
                feats2d_net = model(batch.rgb)
                if self.flip_sfm:
                    feats2d_net = model(torch.flip(batch.rgb, dims=[3]))
                    # feats2d_net = torch.flip(feats2d_net, dims = [3])
            else:
                root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if self.use_sph == "sph_baseline":
                    sph_mapper = MyDINOMapper(
                        backbone="dinov2_vitb14_frozen_base_no_norm", n_cats=18
                    )
                    sph_mapper.load_checkpoint(
                        f'{root}/SphericalMaps/exps/olaf/baseline_sph_200.pth',
                        device=device,
                    )
                if self.use_sph == "sph_sam2":
                    sph_mapper = MyDINOMapper(
                        backbone="dinov2_vitb14_frozen_base_no_norm", n_cats=229
                    )
                    sph_mapper.load_checkpoint(
                        f'{root}/SphericalMaps/exps/olaf/in3d_wo_sym_sam2_170.pth',
                        device=device,
                    )
                if self.use_sph == "sph_coco":
                    sph_mapper = MyDINOMapper(
                        backbone="dinov2_vitb14_frozen_base_no_norm", n_cats=28
                    )
                    sph_mapper.load_checkpoint(
                       f'{root}/SphericalMaps/exps/olaf/in3d_coco_mrcnn_200.pth',
                        device=device,
                    )
                if self.use_sph == "sph_my_baseline":
                    sph_mapper = MyDINOMapper(
                        backbone="dinov2_vitb14_frozen_base_no_norm", n_cats=18
                    )
                    sph_mapper.load_checkpoint(
                        f'{root}/SphericalMaps/exps/baseline_od3d_modelexp_cuda:0_2024_08_18_15:56:55/ckpts/200.pth',
                        device=device,
                    )
                if self.use_sph == "sph_excludes_co3d":
                    sph_mapper = MyDINOMapper(
                        backbone="dinov2_vitb14_frozen_base_no_norm", n_cats=114
                    )
                    sph_mapper.load_checkpoint(
                        f'{root}/SphericalMaps/exps/olaf/exp_002_in3d_wo_sym_and_co3d_200.pth',
                        device=device,
                    )
                    sph_mapper.to(device)
                if self.use_sph == "sph_excludes_co3d_with_dino":
                    sph_mapper = MyDINOMapper(
                        backbone="dinov2_vitb14_frozen_base_no_norm", n_cats=114
                    )
                    sph_mapper.load_checkpoint(
                        f'{root}/SphericalMaps/exps/olaf/exp_002_in3d_wo_sym_and_co3d_200.pth',
                        device=device,
                    )
                    sph_mapper.to(device)

                sph_mapper = sph_mapper.to(device)
                from od3d.SphericalMaps.sd_dino.utils.utils_correspondence import resize
               
                if self.use_sph == "sph_excludes_co3d_with_dino":
                    feats2d_net = my_get_feature(batch.rgb, sph_mapper)
                    feats2d_net = feats2d_net / torch.norm(feats2d_net, dim=1, keepdim=True)
                
                    mixing_ratio = self.mixing_ratio
                    feats2d_net_dino = model(batch.rgb)

                    feats2d_net_dino = feats2d_net_dino / torch.norm(
                        feats2d_net_dino, dim=1, keepdim=True
                    )
                    feats2d_net = torch.concat(
                        [
                            feats2d_net * torch.sqrt(torch.tensor(mixing_ratio)),
                            feats2d_net_dino
                            * torch.sqrt(torch.tensor(1.0 - mixing_ratio)),
                        ],
                        dim=1,
                    )
                    feats2d_net = feats2d_net / torch.norm(
                        feats2d_net, dim=1, keepdim=True
                    )
                    assert torch.allclose(
                        torch.norm(feats2d_net, dim=1).cuda(),
                        torch.ones(
                            torch.norm(feats2d_net, dim=1).shape[0], 32, 32
                        ).cuda(),
                        atol=1e-6,
                    ), "Vectors are not properly normalized"
                    # print(' ')
                if self.flip_sfm:
                    feats2d_net = my_get_feature(
                        torch.flip(batch.rgb, dims=[3]), sph_mapper
                    )

            noise2d = torch.ones(size=(vts2d.shape[0], 0, 2), device=device)

            # B x F+N x C
            net_feats = sample_pxl2d_pts(
                feats2d_net,
                pxl2d=torch.cat([vts2d, noise2d], dim=1),
            )
            
            # visualize points sampled
            # from od3d.cv.visual.show import show_img
            # from od3d.cv.visual.draw import draw_pixels
            # img = batch.rgb[0].clone()
            # img = draw_pixels(img, vts2d[0] * down_sample_rate, colors=meshes.get_verts_ncds_with_mesh_id(mesh_id=0))
            # show_img(img)

            C = net_feats.shape[2]
            # args: X: Bx3xHxW, keypoint_positions: BxNx2, obj_mask: BxHxW ensures that noise is sampled outside of object mask
            # returns: BxF+NxC

            # net_feats = net_feats[:, :].reshape(-1, net_feats.shape[-1])
            batch_vts_ids = meshes.get_verts_and_noise_ids_stacked(
                [0] * B,
                count_noise_ids=0,
            )

            # N,
            batch_vts_ids = torch.cat(
                [batch_vts_ids[:, :N][vts2d_mask], batch_vts_ids[:, N:].reshape(-1)],
                dim=0,
            )

            # N x C
            net_feats = torch.cat(
                [net_feats[:, :N][vts2d_mask], net_feats[:, N:].reshape(-1, C)],
                dim=0,
            )

            
            mesh_root_path = self.path_preprocess.joinpath(
                "raw_mesh_feats",
                self.name_unique,
            )
            if not os.path.exists(mesh_root_path):
                os.makedirs(mesh_root_path)
            
            
            for b, vertex_id in enumerate(batch_vts_ids):
                #logger.info(f"net_feats[b : b + 1]: {net_feats[b : b + 1].shape}")
                #logger.info(f"meshes_verts_aggregated_features[vertex_id]: {meshes_verts_aggregated_features[vertex_id].shape}")
                
                meshes_verts_aggregated_features[vertex_id] = torch.cat(
                    [
                        net_feats[b : b + 1].detach().cpu(),
                        meshes_verts_aggregated_features[vertex_id].detach().cpu(),
                    ],
                    dim=0,
                )
                meshes_verts_aggregated_viewpoints[vertex_id] = torch.cat(
                    [
                        viewpoints3d[b : b + 1].detach().cpu(),
                        meshes_verts_aggregated_viewpoints[vertex_id].detach().cpu(),
                    ],
                    dim=0,
                )
                
        logger.info(f"type of meshes_verts_aggregated_features: {type(meshes_verts_aggregated_features)}")
        logger.info(f"type of meshes_verts_aggregated_viewpoints: {type(meshes_verts_aggregated_viewpoints)}")
        
        logger.info(f"save mesh feats at {self.fpath_mesh_feats}")
        logger.info(f"save mesh feats viewpoint at {self.fpath_mesh_feats_viewpoint}")
        
        if reduce_type == "acc":
            if not self.fpath_mesh_feats.parent.exists():
                self.fpath_mesh_feats.parent.mkdir(parents=True, exist_ok=True)
            torch.save(meshes_verts_aggregated_features, f=f"{mesh_root_path}/mesh_feats.pt")
            torch.save(
                meshes_verts_aggregated_viewpoints,
                f=f"{mesh_root_path}/mesh_feats_viewpoint.pt",
            )
            # meshes_verts_agdgregated_features.clear()
            avg_path = self.get_fpath_mesh_feats(
                mesh_feats_type="M_dinov2_vitb14_frozen_base_T_centerzoom512_R_avg"
            )
            if not avg_path.parent.exists():
                avg_path.parent.mkdir(parents=True, exist_ok=True)
            meshes_verts_aggregated_features_avg = torch.stack(
                [
                    agg_feats.mean(dim=0)
                    for agg_feats in meshes_verts_aggregated_features
                ],
                dim=0,
            )
            torch.save(
                meshes_verts_aggregated_features_avg.detach().cpu(),
                f=avg_path,
            )

            del meshes_verts_aggregated_features_avg
        elif reduce_type == "avg":
            if not self.fpath_mesh_feats.parent.exists():
                self.fpath_mesh_feats.parent.mkdir(parents=True, exist_ok=True)
            meshes_verts_aggregated_features_avg = torch.stack(
                [
                    agg_feats.mean(dim=0)
                    for agg_feats in meshes_verts_aggregated_features
                ],
                dim=0,
            )
            torch.save(
                meshes_verts_aggregated_features_avg.detach().cpu(),
                f=self.fpath_mesh_feats,
            )
            meshes_verts_aggregated_viewpoints_avg = torch.stack(
                [
                    agg_viewpoints.mean(dim=0)
                    for agg_viewpoints in meshes_verts_aggregated_viewpoints
                ],
                dim=0,
            )
            torch.save(
                meshes_verts_aggregated_viewpoints_avg.detach().cpu(),
                f=self.fpath_mesh_feats_viewpoint,
            )

            del meshes_verts_aggregated_features_avg
        elif reduce_type == "avg_norm":
            if not self.fpath_mesh_feats.parent.exists():
                self.fpath_mesh_feats.parent.mkdir(parents=True, exist_ok=True)
            meshes_verts_aggregated_features_avg_norm = torch.nn.functional.normalize(
                torch.stack(
                    [
                        agg_feats.mean(dim=0)
                        for agg_feats in meshes_verts_aggregated_features
                    ],
                    dim=0,
                ),
                dim=-1,
            )
            torch.save(
                meshes_verts_aggregated_features_avg_norm.detach().cpu(),
                f=self.fpath_mesh_feats,
            )
            del meshes_verts_aggregated_features_avg_norm
            meshes_verts_aggregated_viewpoints_avg = torch.stack(
                [
                    agg_viewpoints.mean(dim=0)
                    for agg_viewpoints in meshes_verts_aggregated_viewpoints
                ],
                dim=0,
            )
            torch.save(
                meshes_verts_aggregated_viewpoints_avg.detach().cpu(),
                f=self.fpath_mesh_feats_viewpoint,
            )

        elif reduce_type == "min50":
            if not self.fpath_mesh_feats.parent.exists():
                self.fpath_mesh_feats.parent.mkdir(parents=True, exist_ok=True)

            meshes_verts_aggregated_features_padded = torch.nn.utils.rnn.pad_sequence(
                meshes_verts_aggregated_features,
                padding_value=torch.nan,
                batch_first=True,
            )
            meshes_verts_aggregated_viewpoints_padded = torch.nn.utils.rnn.pad_sequence(
                meshes_verts_aggregated_viewpoints,
                padding_value=torch.nan,
                batch_first=True,
            )

            meshes_verts_aggregated_features_dists = torch.cdist(
                meshes_verts_aggregated_features_padded,
                meshes_verts_aggregated_features_padded,
            )
            c = torch.nanquantile(meshes_verts_aggregated_features_dists, q=0.5, dim=-1)
            vals, indices = c.nan_to_num(torch.inf).min(dim=-1)
            from od3d.cv.select import batched_index_select

            mesh_verts_aggregated_features_min50 = batched_index_select(
                input=meshes_verts_aggregated_features_padded,
                index=indices[:, None],
                dim=1,
            )[:, 0]
            mesh_verts_aggregated_viewpoints_min50 = batched_index_select(
                input=meshes_verts_aggregated_viewpoints_padded,
                index=indices[:, None],
                dim=1,
            )[:, 0]

            torch.save(
                mesh_verts_aggregated_features_min50.detach().cpu(),
                f=self.fpath_mesh_feats,
            )
            torch.save(
                mesh_verts_aggregated_viewpoints_min50.detach().cpu(),
                f=self.fpath_mesh_feats_viewpoint,
            )

        else:
            logger.warning(f"Unknown mesh feature reduce_type {reduce_type}.")

        del dataloader
        del model
        torch.cuda.empty_cache()

    def get_fpath_mesh_feats_dist(
        self,
        sequence: OD3D_Sequence,
        mesh_type=None,
        mesh_feats_type=None,
        sph_type=None,
    ):
        if mesh_type is None:
            mesh_type = self.mesh_type
        if mesh_feats_type is None:
            mesh_feats_type = self.mesh_feats_type
        if sph_type is None:
            sph_type = self.use_sph
        return self.path_preprocess.joinpath(
            "feats_dist",
            f"{self.mesh_feats_dist_reduce_type}",
            f"{mesh_feats_type}",
            f"{mesh_type}",
            f"{self.pcl_type}",
            f"{self.sfm_type}",
            self.name_unique,
            sequence.name_unique,
            f"Partial_Ratio_{100* self.partial_ratio}_Percent_start_frame_{self.start_frame_id}_ref_flip_sfm_{self.flip_sfm}_src_flip_sfm_{sequence.flip_sfm}_use_sph_{self.use_sph}_mixing_ratio_{self.mixing_ratio}",
            # f'Partial_Ratio_{100* self.partial_ratio}_Percent_start_frame_{self.start_frame_id}_ref_flip_sfm_{self.flip_sfm}_src_flip_sfm_{sequence.flip_sfm}',
            "mesh_feats_dist.pt",
        )

    def get_fpath_mesh_feats_dist_mixture(
        self,
        sequence: OD3D_Sequence,
        mesh_type=None,
        mesh_feats_type=None,
        sph_type=None,
    ):
        if mesh_type is None:
            mesh_type = self.mesh_type
        if mesh_feats_type is None:
            mesh_feats_type = self.mesh_feats_type
        if sph_type is None:
            sph_type = self.use_sph
        return self.path_preprocess.joinpath(
            "feats_dist",
            f"{self.mesh_feats_dist_reduce_type}",
            f"{mesh_feats_type}",
            f"{mesh_type}",
            f"{self.pcl_type}",
            f"{self.sfm_type}",
            self.name_unique,
            sequence.name_unique,
            f"Partial_Ratio_ref_{100* self.partial_ratio}_src_{100 * sequence.partial_ratio}_use_sph_{self.use_sph}_mixing_ratio_{self.mixing_ratio}",
            # f'Partial_Ratio_{100* self.partial_ratio}_Percent_start_frame_{self.start_frame_id}_ref_flip_sfm_{self.flip_sfm}_src_flip_sfm_{sequence.flip_sfm}',
            "mesh_feats_dist.pt",
        )

    def get_fpath_mesh_feats_dist_multi_start(
        self,
        sequence: OD3D_Sequence,
        mesh_type=None,
        mesh_feats_type=None,
    ):
        if mesh_type is None:
            mesh_type = self.mesh_type
        if mesh_feats_type is None:
            mesh_feats_type = self.mesh_feats_type

        return self.path_preprocess.joinpath(
            "feats_dist",
            f"{self.mesh_feats_dist_reduce_type}",
            f"{mesh_feats_type}",
            f"{mesh_type}",
            f"{self.pcl_type}",
            f"{self.sfm_type}",
            self.name_unique,
            sequence.name_unique,
            f"Partial_Ratio_{100* self.partial_ratio}_Percent_ref_start_{self.start_frame_id}_src_start_{sequence.start_frame_id}_use_sph_{self.use_sph}_mixing_ratio_{self.mixing_ratio}",
            # f'Partial_Ratio_{100* self.partial_ratio}_Percent_start_frame_{self.start_frame_id}_ref_flip_sfm_{self.flip_sfm}_src_flip_sfm_{sequence.flip_sfm}',
            "mesh_feats_dist.pt",
        )

    def read_mesh_feats_dist(
        self,
        sequence: OD3D_Sequence,
        mesh_type=None,
        mesh_feats_type=None,
        sph_type=None,
    ):
        fpath_mesh_feats_dist = self.get_fpath_mesh_feats_dist(
            sequence,
            mesh_type=mesh_type,
            mesh_feats_type=mesh_feats_type,
            sph_type=sph_type,
        )
        mesh_feats_dist = torch.load(fpath_mesh_feats_dist)
        return mesh_feats_dist

    def read_mesh_feats_dist_mixture(
        self,
        sequence: OD3D_Sequence,
        mesh_type=None,
        mesh_feats_type=None,
        sph_type=None,
    ):
        fpath_mesh_feats_dist = self.get_fpath_mesh_feats_dist_mixture(
            sequence,
            mesh_type=mesh_type,
            mesh_feats_type=mesh_feats_type,
            sph_type=sph_type,
        )
        mesh_feats_dist = torch.load(fpath_mesh_feats_dist)
        return mesh_feats_dist

    def read_mesh_feats_dist_multi_start(
        self,
        sequence: OD3D_Sequence,
        mesh_type=None,
        mesh_feats_type=None,
    ):
        fpath_mesh_feats_dist = self.get_fpath_mesh_feats_dist_multi_start(
            sequence,
            mesh_type=mesh_type,
            mesh_feats_type=mesh_feats_type,
        )
        mesh_feats_dist = torch.load(fpath_mesh_feats_dist)
        return mesh_feats_dist

    def preprocess_mesh_feats_dist_mixture(
        self, sequence: OD3D_Sequence, override=False
    ):
        fpath_dist_verts_mesh_feats = self.get_fpath_mesh_feats_dist_mixture(sequence)
        if not override and fpath_dist_verts_mesh_feats.exists():
            logger.info(
                f"mesh feats dist already exist at {fpath_dist_verts_mesh_feats}",
            )
            return

        if override and fpath_dist_verts_mesh_feats.exists():
            logger.info(f"overriding mesh feats dist at {fpath_dist_verts_mesh_feats}")

        device = get_default_device()

        seq1_feats = self.read_mesh_feats(cache=False)
        if seq1_feats is None:
            return
        seq2_feats = sequence.read_mesh_feats(
            cache=False,
            mesh_type=self.mesh_type,
            mesh_feats_type=self.mesh_feats_type,
        )
        if seq2_feats is None:
            return

        from od3d.cv.cluster.embed import pca

        dist_verts_mesh_feats_reduce_type = self.mesh_feats_dist_reduce_type
        match = re.match(
            r"(pca)?([0-9]*)_?([a-z_]*)",
            dist_verts_mesh_feats_reduce_type,
            re.I,
        )

        if match:
            embed_type, embed_dim, reduce_type = match.groups()
            if len(embed_dim) > 0:
                embed_dim = int(embed_dim)
            # print(embed_type, embed_dim, reduce_type)
        else:
            msg = f"could not retrieve embed_type, embed_dim, and reduce type from mesh feats type {dist_verts_mesh_feats_reduce_type}"
            raise Exception(msg)

        # perform pca on seq1_feats and seq2_feats and visualize
        if isinstance(seq1_feats, list):
            seq1_verts_count = len(seq1_feats)
            seq2_verts_count = len(seq2_feats)
            dist_verts_seq1_seq2 = (
                torch.ones(size=(seq1_verts_count, seq2_verts_count)).to(
                    device=device,
                )
                * torch.inf
            )

            # Vertices1+2 x Viewpoints x F
            seq12_feats_padded = torch.nn.utils.rnn.pad_sequence(
                seq1_feats + seq2_feats,
                batch_first=True,
                padding_value=torch.nan,
            ).to(device=device)
            F = seq12_feats_padded.shape[-1]
            V = seq12_feats_padded.shape[-2]
            seq12_feats_padded_mask = ~seq12_feats_padded.isnan().all(dim=-1)

            if embed_type is None:
                pass
            elif embed_type == "pca":
                seq12_feats_padded_embed = torch.zeros(
                    size=(seq12_feats_padded.shape[:-1] + (embed_dim,)),
                ).to(
                    device=device,
                    dtype=seq12_feats_padded.dtype,
                )
                seq12_feats_padded_embed[:] = torch.nan
                seq12_feats_padded_embed[seq12_feats_padded_mask] = pca(
                    seq12_feats_padded[seq12_feats_padded_mask],
                    C=embed_dim,
                )
                seq12_feats_padded = seq12_feats_padded_embed
                F = embed_dim
            else:
                logger.warning(f"unknown embed type {embed_type}")

            P = seq1_verts_count  # ensures that 11 GB are enough
            logger.info(
                f"seq1 verts {seq1_verts_count}, seq2 verts {seq2_verts_count}, seq1 partial {(seq1_verts_count // P)}, viewpoints max {V}",
            )

            for p in range(P):
                if p < P - 1:
                    seq1_verts_partial = torch.arange(seq1_verts_count)[
                        (seq1_verts_count // P) * p : (seq1_verts_count // P) * (p + 1)
                    ].to(device=device)
                else:
                    seq1_verts_partial = torch.arange(seq1_verts_count)[
                        (seq1_verts_count // P) * p :
                    ].to(
                        device=device,
                    )
                seq1_verts_partial_count = len(seq1_verts_partial)
                # logger.info(seq1_verts_partial)
                # Vertices1 x Viewpoints x F
                seq1_feats_padded = seq12_feats_padded[
                    seq1_verts_partial
                ].clone()  # 1, 69, 384
                seq2_feats_padded = seq12_feats_padded[
                    seq1_verts_count:
                ].clone()  # 452, 69, 384

                seq1_feats_padded_mask = seq12_feats_padded_mask[
                    seq1_verts_partial
                ].clone()
                seq2_feats_padded_mask = seq12_feats_padded_mask[
                    seq1_verts_count:
                ].clone()

                # Vertices1 x Viewpoints x Vertices2 x Viewpoints
                if reduce_type.startswith("negdot"):
                    dists_verts_feats_seq1_seq2 = -torch.einsum(
                        "bnf,bkf->bnk",
                        seq1_feats_padded.reshape(-1, F)[None,],
                        seq2_feats_padded.reshape(-1, F)[None,],
                    ).reshape(
                        seq1_verts_partial_count,
                        V,
                        seq2_verts_count,
                        V,
                    )
                else:
                    dists_verts_feats_seq1_seq2 = torch.cdist(
                        seq1_feats_padded.reshape(-1, F)[None,],
                        seq2_feats_padded.reshape(-1, F)[None,],
                    ).reshape(
                        seq1_verts_partial_count,
                        V,
                        seq2_verts_count,
                        V,
                    )
                dists_verts_feats_seq1_seq2_mask = (
                    seq1_feats_padded_mask[:, :, None, None]
                    * seq2_feats_padded_mask[None, None, :, :]
                )
                dist_verts_seq1_seq2_inf_mask = (
                    dists_verts_feats_seq1_seq2_mask.permute(0, 2, 1, 3)
                    .flatten(
                        2,
                    )
                    .sum(
                        dim=-1,
                    )
                    == 0.0
                )

                if (
                    reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN
                    or reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.NEGDOT_MIN
                ):
                    # replace nan values with inf
                    dists_verts_feats_seq1_seq2 = (
                        dists_verts_feats_seq1_seq2.nan_to_num(torch.inf)
                    )
                    dist_verts_seq1_seq2[seq1_verts_partial] = (
                        dists_verts_feats_seq1_seq2.permute(
                            0,
                            2,
                            1,
                            3,
                        )
                        .flatten(
                            2,
                        )
                        .min(dim=-1)
                        .values
                    )
                elif (
                    reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.AVG
                    or reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.NEGDOT_AVG
                ):
                    dists_verts_feats_seq1_seq2 = (
                        dists_verts_feats_seq1_seq2.nan_to_num(0.0)
                    )
                    dists_verts_feats_seq1_seq2_mask = (
                        dists_verts_feats_seq1_seq2_mask.nan_to_num(0.0)
                    )

                    dist_verts_seq1_seq2_partial = (
                        dists_verts_feats_seq1_seq2.permute(0, 2, 1, 3).flatten(
                            2,
                        )
                        * dists_verts_feats_seq1_seq2_mask.permute(0, 2, 1, 3).flatten(
                            2,
                        )
                    ).sum(dim=-1) / (
                        dists_verts_feats_seq1_seq2_mask.permute(
                            0,
                            2,
                            1,
                            3,
                        )
                        .flatten(2)
                        .sum(
                            dim=-1,
                        )
                        + 1e-10
                    )
                    dist_verts_seq1_seq2_partial[
                        dist_verts_seq1_seq2_inf_mask
                    ] = torch.inf
                    dist_verts_seq1_seq2[
                        seq1_verts_partial
                    ] = dist_verts_seq1_seq2_partial
                    del dist_verts_seq1_seq2_partial
                elif (
                    reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG
                    or reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.NEGDOT_MIN_AVG
                ):
                    dists_verts_feats_seq1_seq2 = (
                        dists_verts_feats_seq1_seq2.nan_to_num(torch.inf)
                    )
                    dists_verts_feats_seq1_seq2_mask = (
                        dists_verts_feats_seq1_seq2_mask.nan_to_num(0.0)
                    )
                    dist_verts_seq1_seq2_partial = (
                        (
                            dists_verts_feats_seq1_seq2.permute(
                                0,
                                2,
                                1,
                                3,
                            )
                            .min(
                                dim=-1,
                            )
                            .values.nan_to_num(
                                posinf=0.0,
                            )
                            * dists_verts_feats_seq1_seq2_mask.permute(
                                0,
                                2,
                                1,
                                3,
                            )[:, :, :, 0]
                        ).sum(dim=-1)
                        + (
                            dists_verts_feats_seq1_seq2.permute(
                                0,
                                2,
                                1,
                                3,
                            )
                            .min(
                                dim=-2,
                            )
                            .values.nan_to_num(
                                posinf=0.0,
                            )
                            * dists_verts_feats_seq1_seq2_mask.permute(
                                0,
                                2,
                                1,
                                3,
                            )[:, :, 0, :]
                        ).sum(dim=-1)
                    ) / (
                        dists_verts_feats_seq1_seq2_mask.permute(0, 2, 1, 3)[
                            :,
                            :,
                            0,
                            :,
                        ].sum(
                            dim=-1,
                        )
                        + dists_verts_feats_seq1_seq2_mask.permute(
                            0,
                            2,
                            1,
                            3,
                        )[
                            :,
                            :,
                            :,
                            0,
                        ].sum(dim=-1)
                        + 1e-10
                    )
                    dist_verts_seq1_seq2_partial[
                        dist_verts_seq1_seq2_inf_mask
                    ] = torch.inf
                    dist_verts_seq1_seq2[
                        seq1_verts_partial
                    ] = dist_verts_seq1_seq2_partial
                    del dist_verts_seq1_seq2_partial
                else:
                    logger.warning(f"Unknown reduce type {reduce_type}.")

                del dists_verts_feats_seq1_seq2
                del dists_verts_feats_seq1_seq2_mask
                del dist_verts_seq1_seq2_inf_mask
                del seq1_verts_partial
            del seq12_feats_padded
            del seq12_feats_padded_mask
            seq1_feats.clear()
            seq2_feats.clear()
        else:
            if reduce_type.startswith("negdot"):
                dist_verts_seq1_seq2 = -torch.einsum(
                    "nf,kf->nk",
                    seq1_feats.to(device=device),
                    seq2_feats.to(device=device),
                )
            else:
                dist_verts_seq1_seq2 = torch.cdist(
                    seq1_feats.to(device=device),
                    seq2_feats.to(device=device),
                )

        if not fpath_dist_verts_mesh_feats.parent.exists():
            fpath_dist_verts_mesh_feats.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dist_verts_seq1_seq2.detach().cpu(), fpath_dist_verts_mesh_feats)
        logger.info(f"save mesh feats dist at {fpath_dist_verts_mesh_feats}")
        del dist_verts_seq1_seq2
        del seq1_feats
        del seq2_feats
        torch.cuda.empty_cache()

    def preprocess_mesh_feats_dist(self, sequence: OD3D_Sequence, override=False):
        if self.start_frame_id == sequence.start_frame_id:
            fpath_dist_verts_mesh_feats = self.get_fpath_mesh_feats_dist(sequence)
        else:
            fpath_dist_verts_mesh_feats = self.get_fpath_mesh_feats_dist_multi_start(
                sequence
            )
        if not override and fpath_dist_verts_mesh_feats.exists():
            logger.info(
                f"mesh feats dist already exist at {fpath_dist_verts_mesh_feats}",
            )
            return

        if override and fpath_dist_verts_mesh_feats.exists():
            logger.info(f"overriding mesh feats dist at {fpath_dist_verts_mesh_feats}")

        device = get_default_device()

        seq1_feats = self.read_mesh_feats(cache=False)
        if seq1_feats is None:
            return
        seq2_feats = sequence.read_mesh_feats(
            cache=False,
            mesh_type=self.mesh_type,
            mesh_feats_type=self.mesh_feats_type,
        )
        if seq2_feats is None:
            return

        from od3d.cv.cluster.embed import pca

        dist_verts_mesh_feats_reduce_type = self.mesh_feats_dist_reduce_type
        match = re.match(
            r"(pca)?([0-9]*)_?([a-z_]*)",
            dist_verts_mesh_feats_reduce_type,
            re.I,
        )

        if match:
            embed_type, embed_dim, reduce_type = match.groups()
            if len(embed_dim) > 0:
                embed_dim = int(embed_dim)
            # print(embed_type, embed_dim, reduce_type)
        else:
            msg = f"could not retrieve embed_type, embed_dim, and reduce type from mesh feats type {dist_verts_mesh_feats_reduce_type}"
            raise Exception(msg)

        # perform pca on seq1_feats and seq2_feats and visualize
        if isinstance(seq1_feats, list):
            seq1_verts_count = len(seq1_feats)
            seq2_verts_count = len(seq2_feats)
            dist_verts_seq1_seq2 = (
                torch.ones(size=(seq1_verts_count, seq2_verts_count)).to(
                    device=device,
                )
                * torch.inf
            )

            # Vertices1+2 x Viewpoints x F
            seq12_feats_padded = torch.nn.utils.rnn.pad_sequence(
                seq1_feats + seq2_feats,
                batch_first=True,
                padding_value=torch.nan,
            ).to(
                device=device
            )  # 904, 69, 384
            F = seq12_feats_padded.shape[-1]  # 384
            V = seq12_feats_padded.shape[-2]  # 69
            seq12_feats_padded_mask = ~seq12_feats_padded.isnan().all(dim=-1)

            if embed_type is None:
                pass
            elif embed_type == "pca":
                seq12_feats_padded_embed = torch.zeros(
                    size=(seq12_feats_padded.shape[:-1] + (embed_dim,)),
                ).to(
                    device=device,
                    dtype=seq12_feats_padded.dtype,
                )
                seq12_feats_padded_embed[:] = torch.nan
                seq12_feats_padded_embed[seq12_feats_padded_mask] = pca(
                    seq12_feats_padded[seq12_feats_padded_mask],
                    C=embed_dim,
                )
                seq12_feats_padded = seq12_feats_padded_embed
                F = embed_dim
            else:
                logger.warning(f"unknown embed type {embed_type}")

            P = seq1_verts_count  # ensures that 11 GB are enough    # 452
            logger.info(
                f"seq1 verts {seq1_verts_count}, seq2 verts {seq2_verts_count}, seq1 partial {(seq1_verts_count // P)}, viewpoints max {V}",
            )

            for p in range(P):
                if p < P - 1:
                    seq1_verts_partial = torch.arange(seq1_verts_count)[
                        (seq1_verts_count // P) * p : (seq1_verts_count // P) * (p + 1)
                    ].to(device=device)
                else:
                    seq1_verts_partial = torch.arange(seq1_verts_count)[
                        (seq1_verts_count // P) * p :
                    ].to(
                        device=device,
                    )
                seq1_verts_partial_count = len(seq1_verts_partial)
                # logger.info(seq1_verts_partial)
                # Vertices1 x Viewpoints x F
                seq1_feats_padded = seq12_feats_padded[
                    seq1_verts_partial
                ].clone()  # 1, 69, 384
                seq2_feats_padded = seq12_feats_padded[
                    seq1_verts_count:
                ].clone()  # 452, 69, 384

                seq1_feats_padded_mask = seq12_feats_padded_mask[
                    seq1_verts_partial
                ].clone()  # 1, 69
                seq2_feats_padded_mask = seq12_feats_padded_mask[
                    seq1_verts_count:
                ].clone()  # 452, 69

                # Vertices1 x Viewpoints x Vertices2 x Viewpoints
                if reduce_type.startswith("negdot"):
                    dists_verts_feats_seq1_seq2 = -torch.einsum(
                        "bnf,bkf->bnk",
                        seq1_feats_padded.reshape(-1, F)[None,],
                        seq2_feats_padded.reshape(-1, F)[None,],
                    ).reshape(
                        seq1_verts_partial_count,
                        V,
                        seq2_verts_count,
                        V,
                    )
                else:
                    dists_verts_feats_seq1_seq2 = torch.cdist(
                        seq1_feats_padded.reshape(-1, F)[None,],
                        seq2_feats_padded.reshape(-1, F)[None,],
                    ).reshape(
                        seq1_verts_partial_count,
                        V,
                        seq2_verts_count,
                        V,
                    )
                dists_verts_feats_seq1_seq2_mask = (
                    seq1_feats_padded_mask[:, :, None, None]
                    * seq2_feats_padded_mask[None, None, :, :]
                )
                dist_verts_seq1_seq2_inf_mask = (
                    dists_verts_feats_seq1_seq2_mask.permute(0, 2, 1, 3)
                    .flatten(
                        2,
                    )
                    .sum(
                        dim=-1,
                    )
                    == 0.0
                )

                if (
                    reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN
                    or reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.NEGDOT_MIN
                ):
                    # replace nan values with inf
                    dists_verts_feats_seq1_seq2 = (
                        dists_verts_feats_seq1_seq2.nan_to_num(torch.inf)
                    )
                    dist_verts_seq1_seq2[seq1_verts_partial] = (
                        dists_verts_feats_seq1_seq2.permute(
                            0,
                            2,
                            1,
                            3,
                        )
                        .flatten(
                            2,
                        )
                        .min(dim=-1)
                        .values
                    )
                elif (
                    reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.AVG
                    or reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.NEGDOT_AVG
                ):
                    dists_verts_feats_seq1_seq2 = (
                        dists_verts_feats_seq1_seq2.nan_to_num(0.0)
                    )
                    dists_verts_feats_seq1_seq2_mask = (
                        dists_verts_feats_seq1_seq2_mask.nan_to_num(0.0)
                    )

                    dist_verts_seq1_seq2_partial = (
                        dists_verts_feats_seq1_seq2.permute(0, 2, 1, 3).flatten(
                            2,
                        )
                        * dists_verts_feats_seq1_seq2_mask.permute(0, 2, 1, 3).flatten(
                            2,
                        )
                    ).sum(dim=-1) / (
                        dists_verts_feats_seq1_seq2_mask.permute(
                            0,
                            2,
                            1,
                            3,
                        )
                        .flatten(2)
                        .sum(
                            dim=-1,
                        )
                        + 1e-10
                    )
                    dist_verts_seq1_seq2_partial[
                        dist_verts_seq1_seq2_inf_mask
                    ] = torch.inf
                    dist_verts_seq1_seq2[
                        seq1_verts_partial
                    ] = dist_verts_seq1_seq2_partial
                    del dist_verts_seq1_seq2_partial
                elif (
                    reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG
                    or reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.NEGDOT_MIN_AVG
                ):
                    dists_verts_feats_seq1_seq2 = (
                        dists_verts_feats_seq1_seq2.nan_to_num(torch.inf)
                    )
                    dists_verts_feats_seq1_seq2_mask = (
                        dists_verts_feats_seq1_seq2_mask.nan_to_num(0.0)
                    )
                    dist_verts_seq1_seq2_partial = (
                        (
                            dists_verts_feats_seq1_seq2.permute(
                                0,
                                2,
                                1,
                                3,
                            )
                            .min(
                                dim=-1,
                            )
                            .values.nan_to_num(
                                posinf=0.0,
                            )
                            * dists_verts_feats_seq1_seq2_mask.permute(
                                0,
                                2,
                                1,
                                3,
                            )[:, :, :, 0]
                        ).sum(dim=-1)
                        + (
                            dists_verts_feats_seq1_seq2.permute(
                                0,
                                2,
                                1,
                                3,
                            )
                            .min(
                                dim=-2,
                            )
                            .values.nan_to_num(
                                posinf=0.0,
                            )
                            * dists_verts_feats_seq1_seq2_mask.permute(
                                0,
                                2,
                                1,
                                3,
                            )[:, :, 0, :]
                        ).sum(dim=-1)
                    ) / (
                        dists_verts_feats_seq1_seq2_mask.permute(0, 2, 1, 3)[
                            :,
                            :,
                            0,
                            :,
                        ].sum(
                            dim=-1,
                        )
                        + dists_verts_feats_seq1_seq2_mask.permute(
                            0,
                            2,
                            1,
                            3,
                        )[
                            :,
                            :,
                            :,
                            0,
                        ].sum(dim=-1)
                        + 1e-10
                    )
                    dist_verts_seq1_seq2_partial[
                        dist_verts_seq1_seq2_inf_mask
                    ] = torch.inf
                    dist_verts_seq1_seq2[
                        seq1_verts_partial
                    ] = dist_verts_seq1_seq2_partial
                    del dist_verts_seq1_seq2_partial
                else:
                    logger.warning(f"Unknown reduce type {reduce_type}.")

                del dists_verts_feats_seq1_seq2
                del dists_verts_feats_seq1_seq2_mask
                del dist_verts_seq1_seq2_inf_mask
                del seq1_verts_partial
            del seq12_feats_padded
            del seq12_feats_padded_mask
            seq1_feats.clear()
            seq2_feats.clear()
        else:
            if reduce_type.startswith("negdot"):
                dist_verts_seq1_seq2 = -torch.einsum(
                    "nf,kf->nk",
                    seq1_feats.to(device=device),
                    seq2_feats.to(device=device),
                )
            else:
                dist_verts_seq1_seq2 = torch.cdist(
                    seq1_feats.to(device=device),
                    seq2_feats.to(device=device),
                )

        if not fpath_dist_verts_mesh_feats.parent.exists():
            fpath_dist_verts_mesh_feats.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dist_verts_seq1_seq2.detach().cpu(), fpath_dist_verts_mesh_feats)
        logger.info(f"save mesh feats dist at {fpath_dist_verts_mesh_feats}")
        del dist_verts_seq1_seq2
        del seq1_feats
        del seq2_feats
        torch.cuda.empty_cache()
