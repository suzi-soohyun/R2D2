import logging

logger = logging.getLogger(__name__)
from od3d.methods.method import OD3D_Method
from od3d.datasets.dataset import OD3D_Dataset
from od3d.benchmark.results import OD3D_Results
from od3d.datasets.co3d import CO3D
from omegaconf import DictConfig
from od3d.cv.visual.show import show_scene
import math
import torch
import os
import numpy as np

torch.multiprocessing.set_sharing_strategy("file_system")
from od3d.cv.geometry.objects3d.meshes import Meshes
from pathlib import Path
from od3d.cv.geometry.transform import tform4x4
from od3d.cv.metric.pose import get_pose_diff_in_rad
from od3d.cv.io import image_as_wandb_image
from od3d.models.model import OD3D_Model
from od3d.cv.transforms.transform import OD3D_Transform
from od3d.cv.transforms.sequential import SequentialTransform
from typing import Dict
from od3d.data.ext_enum import ExtEnum
import json
import matplotlib.pyplot as plt
import yaml
import time
from itertools import accumulate

plt.switch_backend("Agg")
from od3d.cv.optimization.ransac import ransac
from od3d.cv.geometry.fit.tform4x4 import fit_tform4x4, score_tform4x4_fit
import pickle
from functools import partial
from pathlib import Path
from od3d.cv.geometry.transform import inv_tform4x4, tform4x4
from od3d.cv.geometry.transform import transf3d_broadcast
from od3d.cv.visual.show import plot_score_rotation_error

# from mast3r.sequence import Correspondences
from sklearn.cluster import KMeans
from od3d.cv.cluster.spectral_clustering import (
    spectral_clustering,
    spectral_clustering_mean_shift,
)
from od3d.methods.nemo_align3d.utils import *
import shutil
import pandas as pd
import open3d

# threshold = 0.0
threshold = 0.0
temperature = 10.0


# temperature = 1.0
class VISUAL_MODALITIES(str, ExtEnum):
    PRED_VERTS_NCDS_IN_RGB = "pred_verts_ncds_in_rgb"
    GT_VERTS_NCDS_IN_RGB = "gt_verts_ncds_in_rgb"
    PRED_VS_GT_VERTS_NCDS_IN_RGB = "pred_vs_gt_verts_ncds_in_rgb"
    NET_FEATS_NEAREST_VERTS = "net_feats_nearest_verts"
    SIM_PXL = "sim_pxl"
    SAMPLES = "samples"


class SIM_FEATS_MESH_WITH_IMAGE(str, ExtEnum):
    VERTS2D = "verts2d"
    RENDERED = "rendered"


def plot_shared_region3D(tensor1, tensor2, path):
    cosine_sim = torch.mm(tensor1, tensor2.t())

    overlap_indices = (cosine_sim > threshold).nonzero(as_tuple=True)

    vectors1 = tensor1[overlap_indices[0]].numpy()
    vectors2 = tensor2[overlap_indices[1]].numpy()
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot of vectors from the first tensor
    ax.scatter(
        vectors1[:, 0], vectors1[:, 1], vectors1[:, 2], color="blue", label="Ref seq"
    )

    # Scatter plot of vectors from the second tensor
    ax.scatter(
        vectors2[:, 0], vectors2[:, 1], vectors2[:, 2], color="red", label="Src seq"
    )
    print("start")
    # Optionally, add lines to visualize them better
    for v in vectors1:
        # print('v', v)
        ax.quiver(
            0,
            0,
            0,
            v[0],
            v[1],
            v[2],
            length=1.0,
            normalize=True,
            arrow_length_ratio=0.05,
            color="blue",
        )

    for v in vectors2:
        ax.quiver(
            0,
            0,
            0,
            v[0],
            v[1],
            v[2],
            length=1.0,
            normalize=True,
            arrow_length_ratio=0.05,
            color="red",
        )

    # Setting plot limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Title, legend, and grid
    ax.set_title("3D Normalized Vectors Visualization")
    ax.legend(loc="upper right")
    ax.grid(True)

    # Show the plot
    plt.show()
    plt.savefig(path)
    ref_indices = overlap_indices[0]
    src_indices = overlap_indices[1]
    print(f"The shared region has {len(ref_indices) } vertices. ")
    return ref_indices, src_indices


def plot_shared_region2D(tensor1, tensor2, path):
    cosine_sim = torch.mm(tensor1, tensor2.t())

    overlap_indices = (cosine_sim > threshold).nonzero(as_tuple=True)

    vectors1 = tensor1[overlap_indices[0]].numpy()[:, :2]
    vectors2 = tensor2[overlap_indices[1]].numpy()[:, :2]
    vectors1 = vectors1 / np.linalg.norm(vectors1, axis=1, keepdims=True)
    vectors2 = vectors2 / np.linalg.norm(vectors2, axis=1, keepdims=True)
    # Plotting
    fig, ax = plt.subplots()
    ax.quiver(
        np.zeros_like(vectors1[:, 0]),
        np.zeros_like(vectors1[:, 1]),
        vectors1[:, 0],
        vectors1[:, 1],
        color="blue",
        scale=1,
        scale_units="xy",
        angles="xy",
        label="Ref Seq",
    )
    ax.quiver(
        np.zeros_like(vectors2[:, 0]),
        np.zeros_like(vectors2[:, 1]),
        vectors2[:, 0],
        vectors2[:, 1],
        color="red",
        scale=1,
        scale_units="xy",
        angles="xy",
        label="Src Seq",
    )
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    # Title, legend, and grid
    ax.set_title("2D Normalized Vectors Visualization")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.show()
    plt.savefig(path)
    ref_indices = overlap_indices[0]
    src_indices = overlap_indices[1]
    # print(f'The shared region has {len(ref_indices) } vertices. ')
    return ref_indices, src_indices


def plot_vectors3D(tensor1, tensor2, path):
    vectors1 = tensor1.numpy()
    vectors2 = tensor2.numpy()

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot of vectors from the first tensor
    ax.scatter(
        vectors1[:, 0], vectors1[:, 1], vectors1[:, 2], color="blue", label="Ref seq"
    )

    # Scatter plot of vectors from the second tensor
    ax.scatter(
        vectors2[:, 0], vectors2[:, 1], vectors2[:, 2], color="red", label="Src seq"
    )

    # Optionally, add lines to visualize them better
    for v in vectors1:
        ax.quiver(
            0,
            0,
            0,
            v[0],
            v[1],
            v[2],
            length=1.0,
            normalize=True,
            arrow_length_ratio=0.05,
            color="blue",
        )

    for v in vectors2:
        ax.quiver(
            0,
            0,
            0,
            v[0],
            v[1],
            v[2],
            length=1.0,
            normalize=True,
            arrow_length_ratio=0.05,
            color="red",
        )

    # Setting plot limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Title, legend, and grid
    ax.set_title("3D Normalized Vectors Visualization")
    ax.legend(loc="upper right")
    ax.grid(True)

    # Show the plot
    plt.show()
    plt.savefig(path)


def plot_vectors2D(tensor1, tensor2, path):
    vectors1 = tensor1.numpy()
    vectors2 = tensor2.numpy()

    # Ensure only the first two components are considered (in case of higher dimensions)
    vectors1 = vectors1[:, :2]
    vectors2 = vectors2[:, :2]

    # Normalize the vectors
    vectors1 = vectors1 / np.linalg.norm(vectors1, axis=1, keepdims=True)
    vectors2 = vectors2 / np.linalg.norm(vectors2, axis=1, keepdims=True)

    # Setup the plot
    fig, ax = plt.subplots()
    ax.quiver(
        np.zeros_like(vectors1[:, 0]),
        np.zeros_like(vectors1[:, 1]),
        vectors1[:, 0],
        vectors1[:, 1],
        color="blue",
        scale=1,
        scale_units="xy",
        angles="xy",
        label="Tensor 1 Vectors",
    )
    ax.quiver(
        np.zeros_like(vectors2[:, 0]),
        np.zeros_like(vectors2[:, 1]),
        vectors2[:, 0],
        vectors2[:, 1],
        color="red",
        scale=1,
        scale_units="xy",
        angles="xy",
        label="Tensor 2 Vectors",
    )

    # Set plot limits and aspects
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")

    # Title, legend, and grid
    ax.set_title("2D Normalized Vectors Visualization")
    ax.legend(loc="upper right")
    ax.grid(True)

    # Save and show the plot
    plt.savefig(path)
    plt.show()


def find_closest_vertices(meshA, meshB):
    closest_indices = []
    for vertexA in meshA:
        # Compute the squared Euclidean distances to avoid the square root for efficiency
        distances = ((meshB - vertexA) ** 2).sum(1)
        # Find the index of the minimum distance
        closest_idx = distances.argmin()
        closest_indices.append(closest_idx.item())
    return closest_indices  # find the vertex of mesh B to each mesh A


def extract_rotation_from_sim3(sim3_matrix):
    # Step 1: Extract the top-left 3x3 part, this is sR
    sR = sim3_matrix[:3, :3]

    # Step 2: Calculate the scale s using the Frobenius norm
    # Frobenius norm of the scaled rotation matrix sR is s * sqrt(3)
    s = torch.norm(sR) / torch.sqrt(torch.tensor(3.0))

    # Step 3: Divide by s to get the rotation matrix R
    R = sR / s

    return R, s


def apply_softmax_entire_matrix(weighted_matrics, temperature):
    diagonal_mask = torch.eye(weighted_matrics.shape[0], dtype=torch.bool)
    weighted_matrics[diagonal_mask] = 1
    off_diagonal_mask = (~diagonal_mask).cuda()

    weighted_matrics_flat = weighted_matrics[off_diagonal_mask].view(-1)
    weighted_matrics_softmax = torch.nn.functional.softmax(
        weighted_matrics_flat / temperature, dim=0
    )  # temperature to make the probability less sharper
    weighted_matrics[off_diagonal_mask] = weighted_matrics_softmax.view(
        weighted_matrics[off_diagonal_mask].shape
    )
    return weighted_matrics


def apply_rowwise_softmax_excluding_diagonal(weighted_matrics, temperature):
    # Mask to identify diagonal elements
    diagonal_mask = torch.eye(weighted_matrics.shape[0], dtype=torch.bool).to(
        weighted_matrics.device
    )
    # Keep diagonal values safe (store original or set them very high to ensure they turn to 1 after softmax)
    diagonal_values = weighted_matrics.diagonal().clone()
    # Set diagonal elements to a very low value to exclude them effectively from softmax
    weighted_matrics[diagonal_mask] = float("-inf")
    # Apply softmax row-wise
    weighted_matrics = torch.nn.functional.softmax(
        weighted_matrics / temperature, dim=1
    )
    # Restore diagonal values to 1
    weighted_matrics[diagonal_mask] = 1

    return weighted_matrics


class NeMo_Align3D(OD3D_Method):
    def setup(self):
        pass

    def __init__(
        self,
        config: DictConfig,
        logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)

        self.device = "cuda:0"

        # init Network
        self.net = OD3D_Model(config.model)

        self.transform_train = SequentialTransform(
            [
                OD3D_Transform.subclasses[
                    config.train.transform.class_name
                ].create_from_config(config=config.train.transform),
                self.net.transform,
            ],
        )
        self.transform_test = SequentialTransform(
            [
                OD3D_Transform.subclasses[
                    config.test.transform.class_name
                ].create_from_config(config=config.test.transform),
                self.net.transform,
            ],
        )
        # if config.train.transform.random_color:
        #     self.transform_train = torchvision.transforms.Compose([
        #         RandomCenterZoom3D(**config.train.transform.random_center_zoom3d),
        #         #RGB_Random(),
        #         self.net.transform,
        #     ])
        # else:
        #     self.transform_train = torchvision.transforms.Compose([
        #         RandomCenterZoom3D(**config.train.transform.random_center_zoom3d),
        #         self.net.transform,
        #     ])

        # self.transform_test = torchvision.transforms.Compose([
        #    CenterZoom3D(**config.test.transform),
        #    self.net.transform
        # ])

        self.meshes = None
        self.sequences_unique_names = None
        self.meshes = None

        # init Meshes / Features
        self.total_params = sum(p.numel() for p in self.net.parameters())
        self.net.eval()
        self.net.cpu()

        self.down_sample_rate = self.net.downsample_rate

        self.dir_tmp = Path("/tmp").joinpath(self.__class__.__name__)
        self.dir_tmp.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        datasets_train: Dict[str, CO3D],
        datasets_val: Dict[str, OD3D_Dataset],
    ):
        score_metric_name = "pose/acc_pi18"  # 'pose/acc_pi18' 'pose/acc_pi6'
        score_ckpt_val = 0.0
        score_latest = 0.0

        dataset_src: CO3D = datasets_train["src"]
        dataset_ref: CO3D = datasets_train["labeled"]
        from od3d.cv.geometry._mesh import Meshes

        categories = dataset_src.categories
        # num_instances = self.config.instances_num
        # src_sequences = dataset_src.get_sequences()[:self.config.instances_num]
        # ref_sequences = dataset_ref.get_sequences()[:self.config.instances_num]
        src_sequences = dataset_src.get_sequences()
        ref_sequences = dataset_ref.get_sequences()
        for sequence in src_sequences + ref_sequences:
            if self.config.preprocess.pcl.enabled:
                sequence.preprocess_pcl(override=self.config.preprocess.pcl.override)
            if self.config.preprocess.tform_obj.enabled:
                sequence.preprocess_tform_obj(
                    override=self.config.preprocess.tform_obj.override,
                )
            if self.config.preprocess.mesh.enabled:
                sequence.preprocess_mesh(override=self.config.preprocess.mesh.override)
            if self.config.preprocess.mesh_feats.enabled:
                sequence.preprocess_mesh_feats(
                    override=self.config.preprocess.mesh_feats.override,
                )

        if self.config.preprocess.mesh_feats_dist.enabled:
            for src_sequence in src_sequences:
                for ref_sequence in ref_sequences:
                    if src_sequence.category == ref_sequence.category:
                        src_sequence.preprocess_mesh_feats_dist(
                            sequence=ref_sequence,
                            override=self.config.preprocess.mesh_feats_dist.override,
                        )
                        ref_sequence.preprocess_mesh_feats_dist(
                            sequence=src_sequence,
                            override=self.config.preprocess.mesh_feats_dist.override,
                        )

        # tform4x4(inv_tform4x4(src_frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D)), src_frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.DROID_SLAM))
        logger.info("loading mesh feats...")
        # self.sequences_mesh_feats = [seq.feats for seq in self.sequences]
        src_sequences_unique_names = [seq.name_unique for seq in src_sequences]
        ref_sequences_unique_names = [seq.name_unique for seq in ref_sequences]
        # sequences_unique_names = [seq.name_unique for seq in sequences]

        category_dict_ref_seqs = {}
        result_dict = {}
        # Populate the dictionary
        for sequence in ref_sequences_unique_names:
            category, seq_name = sequence.split("/", 1)  # Split on the first '/'
            if category not in category_dict_ref_seqs:
                category_dict_ref_seqs[category] = []
            category_dict_ref_seqs[category].append(seq_name)

        src_map_seq_to_cat = torch.LongTensor(
            [
                categories.index(name.split("/")[0])
                for name in src_sequences_unique_names
            ],
        )
        ref_map_seq_to_cat = torch.LongTensor(
            [
                categories.index(name.split("/")[0])
                for name in ref_sequences_unique_names
            ],
        )

        categories_count = len(categories)

        src_instances_count_per_category = [
            (src_map_seq_to_cat == c).sum().item() for c in range(categories_count)
        ]
        ref_instances_count_per_category = [
            (ref_map_seq_to_cat == c).sum().item() for c in range(categories_count)
        ]

        logger.info("loading meshes...")
        src_meshes = Meshes.load_from_meshes(
            [seq.read_mesh() for seq in src_sequences],
            device=self.device,
        )
        ref_meshes = Meshes.load_from_meshes(
            [seq.read_mesh() for seq in ref_sequences],
            device=self.device,
        )

        src_instances_count = len(src_meshes)
        ref_instances_count = len(ref_meshes)

        dtype = src_meshes.verts.dtype

        src_sequences_mesh_ids_for_verts = src_meshes.get_mesh_ids_for_verts()
        ref_sequences_mesh_ids_for_verts = ref_meshes.get_mesh_ids_for_verts()

        results_diff_log_rot = {}
        pose_graph_results_diff_log_rot = {}
        pose_graph_init_transformations = {}
        all_pred_ref_pts_offset = {}
        all_pred_ref_tform_src = {}
        all_pred_ref_tform_src_rotational = {}
        weighted_matrix = {}
        scale_matrix = {}
        all_pred_pose_dist_geo = {}
        all_pred_pose_dist_appear = {}
        all_pred_pose_dist_geo_diff = {}
        all_pred_pose_dist_appear_diff = {}
        for cat_id, category in enumerate(categories):
            logger.info(f"category id {cat_id} name {category}")
            src_instance_ids = torch.LongTensor(list(range(src_instances_count)))
            ref_instance_ids = torch.LongTensor(list(range(ref_instances_count)))

            results_diff_log_rot[category] = torch.zeros(
                size=(
                    ref_instances_count_per_category[cat_id],
                    src_instances_count_per_category[cat_id],
                ),
            ).to(
                device=self.device,
                dtype=dtype,
            )  # [1,1]
            pose_graph_results_diff_log_rot[category] = torch.zeros(
                size=(
                    ref_instances_count_per_category[cat_id],
                    src_instances_count_per_category[cat_id],
                ),
            ).to(
                device=self.device,
                dtype=dtype,
            )  # [1,1]
            pose_graph_init_transformations[category] = torch.zeros(
                size=(
                    ref_instances_count_per_category[cat_id],
                    src_instances_count_per_category[cat_id],
                ),
            ).to(
                device=self.device,
                dtype=dtype,
            )  # [1,1]
            all_pred_pose_dist_geo_diff[category] = torch.zeros(
                size=(
                    ref_instances_count_per_category[cat_id],
                    src_instances_count_per_category[cat_id],
                ),
            ).to(
                device=self.device,
                dtype=dtype,
            )
            all_pred_pose_dist_appear_diff[category] = torch.zeros(
                size=(
                    ref_instances_count_per_category[cat_id],
                    src_instances_count_per_category[cat_id],
                ),
            ).to(
                device=self.device,
                dtype=dtype,
            )
            all_pred_ref_tform_src[category] = torch.zeros(
                size=(
                    ref_instances_count_per_category[cat_id],
                    src_instances_count_per_category[cat_id],
                    4,
                    4,
                ),
            ).to(
                device=self.device,
                dtype=dtype,
            )  # [4,4]
            all_pred_ref_tform_src_rotational[category] = torch.zeros(
                size=(
                    ref_instances_count_per_category[cat_id],
                    src_instances_count_per_category[cat_id],
                    4,
                    4,
                ),
            ).to(
                device=self.device,
                dtype=dtype,
            )  # [4,4]
            weighted_matrix[category] = torch.zeros(
                size=(
                    ref_instances_count_per_category[cat_id],
                    src_instances_count_per_category[cat_id],
                ),
            ).to(
                device=self.device,
                dtype=dtype,
            )  # [1,1]
            scale_matrix[category] = torch.zeros(
                size=(
                    ref_instances_count_per_category[cat_id],
                    src_instances_count_per_category[cat_id],
                ),
            ).to(
                device=self.device,
                dtype=dtype,
            )  # [1,1]
            all_pred_pose_dist_geo[category] = torch.zeros(
                size=(
                    ref_instances_count_per_category[cat_id],
                    src_instances_count_per_category[cat_id],
                ),
            ).to(
                device=self.device,
                dtype=dtype,
            )  # 1,1

            all_pred_ref_pts_offset[category] = torch.zeros(
                size=(
                    src_instances_count_per_category[cat_id],
                    sum(ref_meshes.verts_counts),
                    3,
                ),
            ).to(
                device=self.device,
                dtype=dtype,
            )  # 1, 452, 3

            all_pred_pose_dist_appear[category] = torch.zeros(
                size=(
                    ref_instances_count_per_category[cat_id],
                    src_instances_count_per_category[cat_id],
                ),
            ).to(
                device=self.device,
                dtype=dtype,
            )  # 1,1

        for i in range(self.config.global_optimization_steps):
            for cat_id, category in enumerate(categories):
                annotated_idx_list = []
                src_instance_ids = torch.LongTensor(list(range(src_instances_count)))
                ref_instance_ids = torch.LongTensor(list(range(ref_instances_count)))

                src_mesh_ids = src_instance_ids[src_map_seq_to_cat == cat_id]
                ref_mesh_ids = ref_instance_ids[ref_map_seq_to_cat == cat_id]

                rot_deg_diff_list = []
                pts_list = []
                pts_color_list = []

                for r, ref_mesh_id in enumerate(ref_mesh_ids):
                    ref_seq_pcl = ref_sequences[ref_mesh_id]
                    pts3d, pts3d_colors, pts3d_normals = ref_seq_pcl.read_pcl()
                    pts_list.append(pts3d)
                    pts_color_list.append(pts3d_colors)
                    for s, src_mesh_id in enumerate(src_mesh_ids):
                        # src_sequences[src_mesh_id].show(show_imgs=True)
                        src_vertices_mask = (
                            src_sequences_mesh_ids_for_verts == src_mesh_id
                        )
                        pts_src = src_meshes.verts[src_vertices_mask].clone()
                        src_seq_pcl = src_sequences[src_mesh_id]
                        (
                            pts3d_src,
                            pts3d_colors_src,
                            pts3d_normals_src,
                        ) = src_seq_pcl.read_pcl()
                        if r > 0 and (
                            self.config.use_only_first_reference
                            or self.config.global_optimization_steps > 1
                        ):
                            pred_ref_tform_src = tform4x4(
                                inv_tform4x4(all_pred_ref_tform_src[category][0, r]),
                                all_pred_ref_tform_src[category][0, s],
                            )
                            all_pred_ref_tform_src[category][r, s] = pred_ref_tform_src
                            all_pred_pose_dist_geo[category][r, s] = (
                                all_pred_pose_dist_geo[category][0, r]
                                + all_pred_pose_dist_geo[category][0, s]
                            )
                            all_pred_pose_dist_appear[category][r, s] = (
                                all_pred_pose_dist_appear[category][0, r]
                                + all_pred_pose_dist_appear[category][0, s]
                            )
                            # logger.info(pred_ref_tform_src)
                        else:
                            if self.config.global_optimization_steps > 1 and i > 0:
                                # ref_vertices_mask = self.sequences_mesh_ids_for_verts == ref_mesh_id
                                # pts = self.meshes.verts.clone().detach()
                                # pts_ref = pts[ref_vertices_mask].clone()
                                # TODO: reference points from multiple point clouds have different scale and therefore problematic to fit with correspondences over multiple points
                                pts_ref = torch.cat(
                                    [
                                        transf3d_broadcast(
                                            pts3d=ref_meshes.verts[
                                                ref_sequences_mesh_ids_for_verts
                                                == _ref_mesh_id
                                            ].clone(),
                                            transf4x4=all_pred_ref_tform_src[category][
                                                0,
                                                _r,
                                            ],
                                        )
                                        if _ref_mesh_id != src_mesh_id
                                        else torch.zeros((0, 3), device=self.device)
                                        for _r, _ref_mesh_id in enumerate(ref_mesh_ids)
                                    ],
                                    dim=0,
                                )

                                dist_src_ref = torch.cat(
                                    [
                                        src_sequences[src_mesh_id]
                                        .read_mesh_feats_dist(
                                            ref_sequences[_ref_mesh_id],
                                        )
                                        .to(device=self.device, dtype=dtype)
                                        if _ref_mesh_id != src_mesh_id
                                        else torch.zeros(
                                            (pts_src.shape[0], 0),
                                            device=self.device,
                                        )
                                        for _ref_mesh_id in ref_mesh_ids
                                    ],
                                    dim=-1,
                                )

                                logger.info(
                                    f"category: {category}, pts-src: {pts_src.shape}, pts-ref: {pts_ref.shape}",
                                )

                                # division by two to normalize to 0. - 1.
                                dist_src_ref = dist_src_ref / 2.0

                                # four points required, otherwise rotation yields an ambiguity. like planes without normals
                                ref_tform4x4_src = ransac(
                                    pts=pts_src,
                                    fit_func=partial(
                                        fit_tform4x4,
                                        pts_ref=pts_ref,
                                        dist_app_ref=dist_src_ref,
                                    ),
                                    score_func=partial(
                                        score_tform4x4_fit,
                                        pts_ref=pts_ref,
                                        dist_app_ref=dist_src_ref,
                                        dist_app_weight=self.config.dist_appear_weight,
                                        geo_cyclic_weight_temp=self.config.geo_cyclic_weight_temp,
                                        app_cyclic_weight_temp=self.config.app_cyclic_weight_temp,
                                        score_perc=self.config.ransac.score_perc,
                                    ),
                                    fits_count=self.config.ransac.samples,
                                    fit_pts_count=4,
                                )

                                _, pose_dist_geo, pose_dist_appear = score_tform4x4_fit(
                                    pts=pts_src,
                                    tform4x4=ref_tform4x4_src[None,],
                                    pts_ref=pts_ref,
                                    dist_app_ref=dist_src_ref,
                                    return_dists=True,
                                    dist_app_weight=self.config.dist_appear_weight,
                                    geo_cyclic_weight_temp=self.config.geo_cyclic_weight_temp,
                                    app_cyclic_weight_temp=self.config.app_cyclic_weight_temp,
                                    score_perc=self.config.ransac.score_perc,
                                )
                                # logger.info(f'sim: geo: {pose_dist_geo}, app: {pose_dist_appear}')
                                all_pred_pose_dist_geo[category][r, s] = pose_dist_geo
                                all_pred_pose_dist_appear[category][
                                    r,
                                    s,
                                ] = pose_dist_appear
                                pred_ref_tform_src = ref_tform4x4_src.clone()
                                # pred_ref_tform_src[:3, :3] /= torch.linalg.norm(pred_ref_tform_src[:3, :3], dim=-1,
                                #                                                keepdim=True)

                                all_pred_ref_tform_src[category][
                                    r,
                                    s,
                                ] = pred_ref_tform_src
                            else:
                                print("ref_mesh_id ", ref_mesh_id)
                                print("src_mesh_id ", src_mesh_id)
                                ref_vertices_mask = (
                                    ref_sequences_mesh_ids_for_verts == ref_mesh_id
                                )
                                pts_ref = ref_meshes.verts[ref_vertices_mask].clone()

                                logger.info(
                                    f"category: {category}, pts-src: {pts_src.shape}, pts-ref: {pts_ref.shape}",
                                )
                                logger.info(f"ref seq {ref_sequences_unique_names[r]} ")
                                logger.info(f"src_seq {src_sequences_unique_names[s]} ")
                                ref_start = ref_sequences[ref_mesh_id].start_frame_id
                                partial_ratio_for_saving = self.config.ratio
                                sequences_annotated = list(
                                    dataset_ref.dict_nested_frames_annotated[
                                        category
                                    ].keys()
                                )
                                # if ref_sequences_unique_names[r].split('/')[1] in sequences_annotated:
                                #     annotated_idx_list.append(r)

                                # ref_pcl = ref_sequences[ref_mesh_id].read_pcl()
                                # src_pcl = src_sequences[ref_mesh_id].read_pcl()

                                for i in range(len(category_dict_ref_seqs[category])):
                                    if (
                                        category_dict_ref_seqs[category][i]
                                        in sequences_annotated
                                    ):
                                        annotated_idx_list.append(i)
                                use_sph = src_sequences[src_mesh_id].use_sph
                                sfm_type_for_saving = src_sequences[
                                    src_mesh_id
                                ].sfm_type
                                pcl_type_for_saving = src_sequences[
                                    src_mesh_id
                                ].pcl_type

                                use_sd = src_sequences[src_mesh_id].use_sd
                                src_name = src_sequences[src_mesh_id].name
                                ref_name = ref_sequences[ref_mesh_id].name
                                mixing_ratio = ref_sequences[ref_mesh_id].mixing_ratio

                                dist_ref_src = (
                                    ref_sequences[ref_mesh_id]
                                    .read_mesh_feats_dist(
                                        src_sequences[src_mesh_id],
                                    )
                                    .to(device=self.device, dtype=dtype)
                                )  # 452, 452

                                ref_mesh_feats_acc = ref_sequences[
                                    ref_mesh_id
                                ].read_mesh_feats()
                                (
                                    ref_sph_feat_cluster_mean,
                                    ref_original_indices,
                                    ref_mesh_feat_filtered,
                                ) = preprocess_mesh_feats_acc(ref_mesh_feats_acc)
                                pts_ref = pts_ref[ref_original_indices].cuda()

                                src_mesh_feats_acc = src_sequences[
                                    src_mesh_id
                                ].read_mesh_feats()
                                (
                                    src_sph_feat_cluster_mean,
                                    src_original_indices,
                                    src_mesh_feat_filtered,
                                ) = preprocess_mesh_feats_acc(src_mesh_feats_acc)
                                pts_src = pts_src[src_original_indices].cuda()

                                dist_ref_src = dist_ref_src[ref_original_indices, :][
                                    :, src_original_indices
                                ]

                                weight_selected_model = 0

                                if r != s:
                                    (
                                        ref_indices,
                                        src_indices,
                                    ) = get_shared_region_mesh_vertex_feat_cluster_mean(
                                        ref_sph_feat_cluster_mean,
                                        src_sph_feat_cluster_mean,
                                        threshold,
                                    )
                                    ref_indices = torch.unique(
                                        torch.tensor(ref_indices)
                                    )
                                    src_indices = torch.unique(
                                        torch.tensor(src_indices)
                                    )
                                    if len(ref_indices) == 0 or len(src_indices) == 0:
                                        weight_selected_model = -1.0
                                        src_tform4x4_ref = torch.eye(4)
                                        continue

                                    dist_ref_src = dist_ref_src[ref_indices, :][
                                        :, src_indices
                                    ]
                                    pts_ref = pts_ref[ref_indices].cuda()
                                    pts_src = pts_src[src_indices].cuda()

                                    ref_sph_feat_cluster_mean = filter_mesh_feats_acc(
                                        ref_sph_feat_cluster_mean, ref_indices
                                    )
                                    ref_mesh_feat_filtered = filter_mesh_feats_acc(
                                        ref_mesh_feat_filtered, ref_indices
                                    )

                                    src_sph_feat_cluster_mean = filter_mesh_feats_acc(
                                        src_sph_feat_cluster_mean, src_indices
                                    )
                                    src_mesh_feat_filtered = filter_mesh_feats_acc(
                                        src_mesh_feat_filtered, src_indices
                                    )

                                (
                                    src_tform4x4_ref,
                                    models,
                                    scores,
                                    best_correspondence,
                                    best_ref_correspondence,
                                    best_geo_dist,
                                    best_appear_dist,
                                    src_tform4x4_ref_score,
                                    pts_ids,
                                    pts_ref_ids,
                                    proposal_dist_ref_geo_avg,
                                    proposal_dist_ref_appear_avg,
                                ) = ransac(
                                    pts=pts_ref,
                                    fit_func=partial(
                                        fit_tform4x4,
                                        pts_ref=pts_src,
                                        dist_ref=dist_ref_src,
                                        # pts_ref_label = src_mesh_feats_label,
                                        # pts_label = ref_mesh_feats_label,
                                    ),
                                    score_func=partial(
                                        score_tform4x4_fit,
                                        pts_ref=pts_src,
                                        dist_app_ref=dist_ref_src,
                                        dist_app_weight=self.config.dist_appear_weight,
                                        geo_cyclic_weight_temp=self.config.geo_cyclic_weight_temp,
                                        app_cyclic_weight_temp=self.config.app_cyclic_weight_temp,
                                        score_perc=self.config.ransac.score_perc,
                                    ),
                                    fits_count=self.config.ransac.samples,
                                    fit_pts_count=4,
                                    return_pts_id=True,
                                    # ref_sph_feat = src_mesh_feats_acc[src_indices],
                                    # src_sph_feat = ref_mesh_feats_acc[ref_indices],
                                )
                                from od3d.cv.visual.correspondences_vis import (
                                    save_visualization_mesh,
                                    convert_pts_mesh,
                                    save_pgo_init_meshes,
                                    save_pgo_init_meshes_transformed,
                                    vis_heatmap_from_vertices,
                                    get_closest_correspondence_model,
                                    save_visualization_mesh_with_color,
                                    compare_sph_dino,
                                    compare_sph_dino_best_id,
                                    MeshVisualizer,
                                )

                                weight_ransac = 1.0
                                # if r!=s:
                                #     ref_name = ref_sequences_unique_names[r].split('/')[1]
                                #     src_name = src_sequences_unique_names[s].split('/')[1]
                                #     weight_ransac = compare_sph_dino(models, pts_src_ids = pts_ref_ids , pts_ref_ids = pts_ids, sph_ref= ref_sph_feat_cluster_mean, sph_src =  src_sph_feat_cluster_mean,
                                #                         scores = scores, r =ref_name, s = src_name, category = category,
                                #                          sph_src_filtered = src_mesh_feat_filtered,
                                #                         sph_ref_filtered = ref_mesh_feat_filtered
                                #     )
                                #     compare_sph_dino_best_id(model = src_tform4x4_ref, pts_src_id = best_ref_correspondence, pts_ref_id = best_correspondence,
                                #                              sph_src = src_sph_feat_cluster_mean,
                                #                              sph_ref = ref_sph_feat_cluster_mean,
                                #                              pts_ref = pts_ref,
                                #                              pts_src = pts_src,
                                #                              sph_src_filtered = src_mesh_feat_filtered,
                                #                              sph_ref_filtered = ref_mesh_feat_filtered)

                                transformed_pts_src = (
                                    (inv_tform4x4(src_tform4x4_ref))
                                    @ torch.cat(
                                        (
                                            pts_src,
                                            torch.ones(pts_src.size(0), 1).cuda(),
                                        ),
                                        dim=1,
                                    ).T
                                ).T
                                closest_indices_in_src = find_closest_vertices(
                                    pts_ref, transformed_pts_src[:, :3]
                                )

                                value_list = []
                                for i in range(len(pts_ref)):
                                    value = get_max_cosine_similarity_vertex_pairs(
                                        ref_sph_feat_cluster_mean[i],
                                        src_sph_feat_cluster_mean[
                                            closest_indices_in_src[i]
                                        ],
                                    )
                                    value_list.append(value)

                                weight_selected_model = torch.mean(
                                    torch.tensor(value_list, dtype=torch.float32)
                                )

                                transformation_score = 1 / (
                                    -src_tform4x4_ref_score.item()
                                )
                                # final_weight =  transformation_score * weight_selected_model
                                # final_weight =   weight_selected_model.clone()
                                final_weight = 1.0
                                if self.config.weight.sph_weight:
                                    final_weight *= weight_selected_model
                                if self.config.weight.ransac_estimation:
                                    final_weight *= weight_ransac
                                if self.config.weight.transformation_score:
                                    final_weight *= transformation_score

                                weighted_matrix[category][r, s] = final_weight
                                # final_weight =  weight_selected_model * transformation_score * weight_ransac
                                logger.info(
                                    f"The ESTIMATE of RANSAC is {weight_ransac} "
                                )
                                logger.info(
                                    f"THE ESTIMATE of SELECTED MODEL is {weight_selected_model}"
                                )
                                logger.info(
                                    f"THE TRANSFORMATION SCORE IS {transformation_score}"
                                )

                                logger.info(f"THE FINAL WEIGHT IS {final_weight}")

                                exp_name = self.config.name
                                vi_mesh_path = (
                                    "/CT/3D_DST_Scene/work/od3d/scripts/vis_meshes"
                                )
                                vi_mesh_path = os.path.join(vi_mesh_path, f"{exp_name}")
                                os.makedirs(vi_mesh_path, exist_ok=True)
                                vi_mesh_path = os.path.join(
                                    vi_mesh_path,
                                    f"ransac_estimation_{self.config.weight.ransac_estimation}_transformation_score_{self.config.weight.transformation_score}_sph_{self.config.weight.sph_weight}",
                                )
                                os.makedirs(vi_mesh_path, exist_ok=True)
                                category_path = os.path.join(
                                    vi_mesh_path, f"category_{category}"
                                )
                                os.makedirs(category_path, exist_ok=True)

                                ratio_path = os.path.join(
                                    category_path,
                                    f"ratio_{int(100 *partial_ratio_for_saving )}",
                                )
                                os.makedirs(ratio_path, exist_ok=True)
                                threshold_path = os.path.join(
                                    ratio_path, f"threshold_{int(100 * threshold )}"
                                )
                                # threshold_path = os.path.join(ratio_path, f'threshold_{int(100 * threshold )}')
                                # threshold_path = os.path.join(ratio_path, f'temperature_{temperature}')
                                os.makedirs(threshold_path, exist_ok=True)

                                feature_path = os.path.join(
                                    threshold_path,
                                    f"use_sph_{use_sph}_mixing_ratio_{mixing_ratio}",
                                )
                                # if os.path.exists(feature_path):
                                #     shutil.rmtree(feature_path)
                                os.makedirs(feature_path, exist_ok=True)
                                category_ratio_use_sph_path = os.path.join(
                                    feature_path, f"id_start_{ref_start}"
                                )
                                os.makedirs(category_ratio_use_sph_path, exist_ok=True)
                                sfm_pcl_type_path = os.path.join(
                                    category_ratio_use_sph_path,
                                    f"sfm_type_{sfm_type_for_saving}_pcl_type_{pcl_type_for_saving}",
                                )
                                os.makedirs(sfm_pcl_type_path, exist_ok=True)
                                folder_path = os.path.join(
                                    sfm_pcl_type_path,
                                    f"ref_id_{ref_name}_src_id_{src_name}_aligned",
                                )
                                os.makedirs(folder_path, exist_ok=True)
                                os.makedirs(
                                    os.path.join(
                                        sfm_pcl_type_path,
                                        f"ref_id_{ref_name}_src_id_{src_name}",
                                    ),
                                    exist_ok=True,
                                )

                                if r != s:
                                    mesh_vis_folder_path_before_alignment = (
                                        os.path.join(
                                            sfm_pcl_type_path,
                                            f"ref_id_{ref_name}_src_id_{src_name}",
                                            "before_alignment",
                                        )
                                    )
                                    os.makedirs(
                                        mesh_vis_folder_path_before_alignment,
                                        exist_ok=True,
                                    )
                                    save_visualization_mesh(
                                        pts=pts_src,
                                        pts_ref=pts_ref,
                                        pts_ids=best_correspondence,
                                        pts_ref_ids=best_ref_correspondence,
                                        filename=os.path.join(
                                            sfm_pcl_type_path,
                                            f"ref_id_{ref_name}_src_id_{src_name}",
                                        ),
                                    )

                                    pts_ref_o3d = open3d.geometry.PointCloud()
                                    pts_cloned = pts3d.clone().cpu().numpy()
                                    pts_color_cloned = (
                                        pts3d_colors.clone().cpu().numpy()
                                    )
                                    pts_ref_o3d.points = open3d.utility.Vector3dVector(
                                        pts_cloned
                                    )
                                    pts_ref_o3d.colors = open3d.utility.Vector3dVector(
                                        pts_color_cloned
                                    )

                                    pts_src_o3d = open3d.geometry.PointCloud()
                                    pts_cloned = pts3d_src.clone().cpu().numpy()
                                    pts_color_cloned = (
                                        pts3d_colors_src.clone().cpu().numpy()
                                    )
                                    pts_src_o3d.points = open3d.utility.Vector3dVector(
                                        pts_cloned
                                    )
                                    pts_src_o3d.colors = open3d.utility.Vector3dVector(
                                        pts_color_cloned
                                    )

                                    mesh_vis = MeshVisualizer(
                                        pts_ref_o3d,
                                        pts_src_o3d,
                                        pts_src[best_ref_correspondence],
                                        pts_ref[best_correspondence],
                                        mesh_vis_folder_path_before_alignment,
                                    )
                                    mesh_vis.save()
                                    # MeshVisualizer(pts_ref_o3d, pts_src_o3d.transform(nv_tform4x4(src_tform4x4_ref)), )
                                    tmp_transformation = (
                                        inv_tform4x4(src_tform4x4_ref).cpu().numpy()
                                    )
                                    transformed_pts = pts_src_o3d.transform(
                                        tmp_transformation
                                    )
                                    mesh_vis_folder_path_after_alignment = os.path.join(
                                        sfm_pcl_type_path,
                                        f"ref_id_{ref_name}_src_id_{src_name}",
                                        "after_alignment",
                                    )
                                    os.makedirs(
                                        mesh_vis_folder_path_after_alignment,
                                        exist_ok=True,
                                    )
                                    open3d.io.write_point_cloud(
                                        os.path.join(
                                            mesh_vis_folder_path_after_alignment,
                                            "tranformed_src_pts.ply",
                                        ),
                                        transformed_pts,
                                    )
                                    open3d.io.write_point_cloud(
                                        os.path.join(
                                            mesh_vis_folder_path_after_alignment,
                                            "ref_pts.ply",
                                        ),
                                        pts_ref_o3d,
                                    )

                                    save_visualization_mesh(
                                        pts=transformed_pts_src[:, :3],
                                        pts_ref=pts_ref,
                                        pts_ids=best_correspondence,
                                        pts_ref_ids=best_ref_correspondence,
                                        filename=folder_path,
                                    )

                                    # save_visualization_mesh_with_color(pts= pts_src, pts_ref= pts_ref, pts_ids= best_correspondence, pts_ref_ids= best_ref_correspondence,
                                    #     filename= os.path.join(sfm_pcl_type_path, f'ref_id_{ref_name}_src_id_{src_name}'),
                                    #     original_pts_ref = pts3d,
                                    #     original_pts_color_ref = pts3d_colors,
                                    #     original_pts_src =  pts3d_src,
                                    #     original_pts_color_src = pts3d_colors_src,)

                                    # save_visualization_mesh_with_color(pts= transformed_pts_src[:,:3], pts_ref= pts_ref, pts_ids= best_correspondence, pts_ref_ids= best_ref_correspondence,
                                    # filename= folder_path,
                                    # original_pts_ref = pts3d,
                                    # original_pts_color_ref = pts3d_colors,
                                    # original_pts_src =  pts3d_src,
                                    # original_pts_color_src = pts3d_colors_src)

                                from od3d.cv.optimization.gradient_descent import (
                                    gradient_descent_se3,
                                )

                                if self.config.refine_optimization_steps > 0:
                                    (
                                        src_tform4x4_ref,
                                        ref_pts_offset,
                                    ) = gradient_descent_se3(
                                        pts=pts_ref,
                                        models=src_tform4x4_ref,
                                        score_func=partial(
                                            score_tform4x4_fit,
                                            pts_ref=pts_src,
                                            dist_app_ref=dist_ref_src,
                                            dist_app_weight=self.config.dist_appear_weight,
                                            geo_cyclic_weight_temp=self.config.geo_cyclic_weight_temp,
                                            app_cyclic_weight_temp=self.config.app_cyclic_weight_temp,
                                            score_perc=self.config.ransac.score_perc,
                                        ),
                                        steps=self.config.refine_optimization_steps,
                                        lr=self.config.refine_lr,
                                        pts_weight=self.config.refine_pts_weight,
                                        arap_weight=self.config.refine_arap_weight,
                                        arap_geo_std=self.config.refine_arap_geo_std,
                                        reg_weight=self.config.refine_reg_weight,
                                        return_pts_offset=True,
                                    )

                                    all_pred_ref_pts_offset[category][s][
                                        ref_vertices_mask
                                    ] = ref_pts_offset

                                _, pose_dist_geo, pose_dist_appear = score_tform4x4_fit(
                                    pts=pts_ref,
                                    tform4x4=src_tform4x4_ref[None,],
                                    pts_ref=pts_src,
                                    dist_app_ref=dist_ref_src,
                                    return_dists=True,
                                    dist_app_weight=self.config.dist_appear_weight,
                                    geo_cyclic_weight_temp=self.config.geo_cyclic_weight_temp,
                                    app_cyclic_weight_temp=self.config.app_cyclic_weight_temp,
                                    score_perc=self.config.ransac.score_perc,
                                )

                                # logger.info(f'sim: geo: {pose_dist_geo}, app: {pose_dist_appear}')
                                all_pred_pose_dist_geo[category][r, s] = pose_dist_geo
                                all_pred_pose_dist_appear[category][
                                    r,
                                    s,
                                ] = pose_dist_appear
                                pred_ref_tform_src = src_tform4x4_ref.clone()
                                pred_ref_tform_src = inv_tform4x4(
                                    src_tform4x4_ref,
                                ).clone()
                                all_pred_ref_tform_src[category][
                                    r,
                                    s,
                                ] = pred_ref_tform_src

                                normalized_rotation, scale = extract_rotation_from_sim3(
                                    pred_ref_tform_src
                                )
                                scale_matrix[category][r, s] = scale
                                all_pred_ref_tform_src_rotational[category][
                                    r,
                                    s,
                                ][:3, :3] = normalized_rotation
                                all_pred_ref_tform_src_rotational[category][
                                    r,
                                    s,
                                ][-1, -1] = 1

                        gt_ref_tform_src = torch.eye(4).to(device=self.device)

                        diff_rot_angle_rad = get_pose_diff_in_rad(
                            pred_tform4x4=pred_ref_tform_src,
                            gt_tform4x4=gt_ref_tform_src,
                        )
                        # diff_rot_angle_rad_best_sph = get_pose_diff_in_rad(
                        #     pred_tform4x4=inv_tform4x4(best_model_sph ),
                        #     gt_tform4x4=gt_ref_tform_src,
                        # )

                        # logger.info(f'diff_rot_angle_rad is  {diff_rot_angle_rad}')
                        logger.info(
                            f"diff rot degree for the baseline is {180 * diff_rot_angle_rad / torch.pi}"
                        )
                        print(
                            "------------------------------------------------------------------------------------------------------------"
                        )
                        results_diff_log_rot[category][r, s] = diff_rot_angle_rad

                        if r != s:
                            rot_deg_diff_list.append(
                                180 * diff_rot_angle_rad / torch.pi
                            )

                from od3d.cv.optimization.masked_posegraph import (
                    WeightedPoseGraphOptimizationRotationMatrixMasking,
                    WeightedPoseGraphOptimizationRotation,
                )

                os.makedirs(os.path.join(feature_path, "pgo_init"), exist_ok=True)
                torch.save(
                    all_pred_ref_tform_src_rotational[category],
                    os.path.join(feature_path, "pgo_init", "input_alignments.pt"),
                )
                weighted_matrics = weighted_matrix[category].clone()
                torch.save(
                    weighted_matrics,
                    os.path.join(feature_path, "pgo_init", f"weighted_matrics.pt"),
                )

                show_weight = weighted_matrics[annotated_idx_list, :][
                    :, annotated_idx_list
                ]
                print("The weighted_matrices", show_weight)

                # with open( os.path.join(feature_path, 'pgo_init', f'seq_list.txt'), 'w') as f:
                #     for item in ref_sequences_unique_names:
                #         f.write("%s\n" % item)
                with open(
                    os.path.join(feature_path, "pgo_init", f"seq_list.json"), "w"
                ) as json_file:
                    json.dump(ref_sequences_unique_names, json_file)
                # diagonal_mask = torch.eye(weighted_matrics.shape[0], dtype = torch.bool)
                # weighted_matrics[diagonal_mask] = 1
                # off_diagonal_mask = (~diagonal_mask).cuda()

                # weighted_matrics_flat = weighted_matrics[off_diagonal_mask].view(-1)
                # weighted_matrics_softmax = torch.nn.functional.softmax( weighted_matrics_flat /temperature , dim=0) # temperature to make the probability less sharper
                # #weighted_matrics_softmax = torch.nn.functional.softmax( weighted_matrics_flat / 1.0, dim=0)
                # #weighted_matrics_softmax = torch.nn.functional.softmax( weighted_matrics_flat / 0.1, dim=0)
                # #weighted_matrics_softmax = torch.nn.functional.softmax( weighted_matrics_flat / 0.01, dim=0)
                # weighted_matrics[off_diagonal_mask] = weighted_matrics_softmax.view(weighted_matrics[off_diagonal_mask].shape)

                weighted_matrics = apply_softmax_entire_matrix(
                    weighted_matrics, temperature
                )
                # weighted_matrics = apply_rowwise_softmax_excluding_diagonal(weighted_matrics, temperature)

                annotated_idx_list = np.unique(annotated_idx_list)
                print("annotated idx list ", annotated_idx_list)
                torch.save(
                    torch.tensor(annotated_idx_list),
                    os.path.join(feature_path, "pgo_init", "annotated_idx.pt"),
                )
                print(
                    "result before pgo ",
                    results_diff_log_rot[category][annotated_idx_list, :][
                        :, annotated_idx_list
                    ]
                    * 180
                    / torch.pi,
                )

                print(
                    "The weighted_matrices",
                    weighted_matrics[annotated_idx_list, :][
                        :, annotated_idx_list
                    ].shape,
                )
                # print('The sum of weighted_matrices', torch.sum(weighted_matrics[off_diagonal_mask]))
                # labels, n_clusters  = spectral_clustering_mean_shift(weighted_matrics[annotated_idx_list, :][:, annotated_idx_list].cpu().numpy())
                # print('the number of clusters ', n_clusters)
                # print('labels ', labels)

                # pgo = WeightedPoseGraphOptimization(all_pred_ref_tform_src[category], weighted_matrics=weighted_matrics)
                # pgo = WeightedPoseGraphOptimizationRotationMatrix(all_pred_ref_tform_src_rotational[category], weighted_matrices= weighted_matrics)
                pgo = WeightedPoseGraphOptimizationRotationMatrixMasking(
                    all_pred_ref_tform_src_rotational[category],
                    weighted_matrices=weighted_matrics,
                    seq_list=ref_sequences_unique_names,
                )
                # pgo = WeightedPoseGraphOptimizationRotation(all_pred_ref_tform_src_rotational[category], weighted_matrices= weighted_matrics, seq_list =  ref_sequences_unique_names)

                updated_rad = pgo.optimization()
                node_order = pgo.return_node_order()
                print("node order ", node_order)

                pgo_folder = os.path.join(feature_path, "pgo_init")
                os.makedirs(pgo_folder, exist_ok=True)
                pgo.save_edges_to_file(os.path.join(pgo_folder, "pgo_graph_init.json"))
                pose_graph_init = pgo.return_pose_graph_init()
                pose_node = pgo.return_pose_node_init()
                print("pose_graph_init ", pose_graph_init)
                print("pose_node ", pose_node)

                pose_node_optimized = pgo.return_pose_node_optimized()
                print("pose node optimized ", pose_node_optimized)
                name_list = []
                for i in range(len(ref_sequences)):
                    name = ref_sequences[i].name
                    name_list.append(name)

                # save_pgo_init_meshes(pose_graph_init, ref_meshes, ref_sequences_mesh_ids_for_verts, pgo_folder)
                save_pgo_init_meshes_transformed(
                    pose_node, pts_list, pts_color_list, pgo_folder, name_list
                )

                pgo_folder_optimized = os.path.join(feature_path, "pgo_optimized")
                save_pgo_init_meshes_transformed(
                    pose_node_optimized,
                    pts_list,
                    pts_color_list,
                    pgo_folder_optimized,
                    name_list,
                )
                init_transformation_rad = pgo.return_pose_graph_init_transformations()
                pose_graph_init_transformations[category] = init_transformation_rad
                print(
                    "pose graph intialization ",
                    pose_graph_init_transformations[category][annotated_idx_list, :][
                        :, annotated_idx_list
                    ]
                    * 180.0
                    / torch.pi,
                )

                bac = BaselineAlignmentToCanonicalization(
                    all_pred_ref_tform_src_rotational[category]
                )
                baseline_results = bac.convert()
                for baseline_results_idx in range(len(baseline_results)):
                    baseline_results_folder = os.path.join(
                        feature_path, f"baseline_{baseline_results_idx}"
                    )
                    save_pgo_init_meshes_transformed(
                        baseline_results[baseline_results_idx],
                        pts_list,
                        pts_color_list,
                        baseline_results_folder,
                        name_list,
                    )

                os.makedirs(os.path.join(pgo_folder, "node_order"), exist_ok=True)
                for i in range(len(node_order)):
                    vertices_mask = ref_sequences_mesh_ids_for_verts == int(
                        node_order[i]
                    )
                    pts_ref = ref_meshes.verts[vertices_mask].clone()

                    convert_pts_mesh(
                        pts_ref,
                        os.path.join(pgo_folder, "node_order", f"order_{i}.ply"),
                    )

                pose_graph_results_diff_log_rot[category] = updated_rad
                print(
                    "pose_graph_results_diff_log_rot ",
                    pose_graph_results_diff_log_rot[category][annotated_idx_list, :][
                        :, annotated_idx_list
                    ]
                    * 180.0
                    / torch.pi,
                )

                category_results = OD3D_Results()
                exclude_diagonal = dataset_src.name == dataset_ref.name
                category_results[f"rot_diff_rad"] = (
                    results_diff_log_rot[category][annotated_idx_list, :][
                        :, annotated_idx_list
                    ][
                        torch.eye(len(annotated_idx_list)).to(
                            device=self.device,
                        )
                        == 0
                    ]
                    .reshape(
                        len(annotated_idx_list),
                        len(annotated_idx_list) - 1,
                    )
                    .permute(1, 0)
                )
                category_results[f"pose_graph_rot_diff_rad"] = (
                    pose_graph_results_diff_log_rot[category][annotated_idx_list, :][
                        :, annotated_idx_list
                    ][
                        torch.eye(len(annotated_idx_list)).to(
                            device=self.device,
                        )
                        == 0
                    ]
                    .reshape(
                        len(annotated_idx_list),
                        len(annotated_idx_list) - 1,
                    )
                    .permute(1, 0)
                )
                category_results[f"pose_graph_init_rot_diff_rad"] = (
                    pose_graph_init_transformations[category][annotated_idx_list, :][
                        :, annotated_idx_list
                    ][
                        torch.eye(len(annotated_idx_list)).to(
                            device=self.device,
                        )
                        == 0
                    ]
                    .reshape(
                        len(annotated_idx_list),
                        len(annotated_idx_list) - 1,
                    )
                    .permute(1, 0)
                )
                category_results_mean = category_results.add_prefix(category)
                category_results_mean = category_results_mean.mean()
                category_results_mean.log()
                print("category_results_mean ", category_results_mean)
                result_dict[category] = {}
                result_dict[category]["pi_6_accuracy_pg"] = category_results_mean[
                    f"pose/prefix/{category}_pose_graph_acc_pi6"
                ]
                result_dict[category]["pi_12_accuracy_pg"] = category_results_mean[
                    f"pose/prefix/{category}_pose_graph_acc_pi12"
                ]
                result_dict[category]["pi_18_accuracy_pg"] = category_results_mean[
                    f"pose/prefix/{category}_pose_graph_acc_pi18"
                ]
                result_dict[category]["error_mean_pg"] = category_results_mean[
                    f"pose/prefix/{category}_pose_graph_err_mean"
                ]
                result_dict[category]["error_median_pg"] = category_results_mean[
                    f"pose/prefix/{category}_pose_graph_err_median"
                ]

                result_dict[category]["pi_6_accuracy"] = category_results_mean[
                    f"pose/prefix/{category}_acc_pi6"
                ]
                result_dict[category]["pi_12_accuracy"] = category_results_mean[
                    f"pose/prefix/{category}_acc_pi12"
                ]
                result_dict[category]["pi_18_accuracy"] = category_results_mean[
                    f"pose/prefix/{category}_acc_pi18"
                ]
                result_dict[category]["error_mean"] = category_results_mean[
                    f"pose/prefix/{category}_err_mean"
                ]
                result_dict[category]["error_median"] = category_results_mean[
                    f"pose/prefix/{category}_err_median"
                ]

                result_dict[category]["pi_6_accuracy_pg_init"] = category_results_mean[
                    f"pose/prefix/{category}_pose_graph_init_acc_pi6"
                ]
                result_dict[category]["pi_12_accuracy_pg_init"] = category_results_mean[
                    f"pose/prefix/{category}_pose_graph_init_acc_pi12"
                ]
                result_dict[category]["pi_18_accuracy_pg_init"] = category_results_mean[
                    f"pose/prefix/{category}_pose_graph_init_acc_pi18"
                ]
                result_dict[category]["error_mean_pg_init"] = category_results_mean[
                    f"pose/prefix/{category}_pose_graph_init_err_mean"
                ]
                result_dict[category]["error_median_pg_init"] = category_results_mean[
                    f"pose/prefix/{category}_pose_graph_init_err_median"
                ]

                print("result_dict[category] ", result_dict[category])
                df = pd.DataFrame(result_dict[category])
                json_str = df.to_json(orient="records", lines=False, indent=4)
                print("json_str ", json_str)
                with open(os.path.join(feature_path, f"results.json"), "w") as file:
                    file.write(json_str)
                # df.to_json()
                # Convert the result_dict to a DataFrame
                # df = pd.DataFrame.from_dict(result_dict[category], orient='index')
                # print('THE DATA FRAME IS ')
                # print(df)
                # df.to_csv(os.path.join('results',f'{int(100 *partial_ratio_for_saving)}_ratio_updated_weight_trafo_score',f'filename_{category}.csv'), index=False)
                # # Reset index to turn the index into a regular column, if preferred
                # df.reset_index(inplace=True)
                # df.rename(columns={'index': 'Category'}, inplace=True)

                # Convert DataFrame to LaTeX code
                # latex_code = df.to_latex(index=False, header=True, column_format='lcccccccccc', caption='Performance Metrics', label='tab:performance_metrics')
                # print('THE LATEX CODE IS ')
                # print(latex_code)
                del pgo, weighted_matrics, updated_rad

                # Optionally clear any residual memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear memory cache if running on GPU


    def test(self, dataset: OD3D_Dataset, config_inference: DictConfig = None):
        return OD3D_Results()
