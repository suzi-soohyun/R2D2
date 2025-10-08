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

import matplotlib.pyplot as plt
import pickle

plt.switch_backend("Agg")
from od3d.cv.optimization.ransac import ransac
from od3d.cv.geometry.fit.tform4x4 import fit_tform4x4, score_tform4x4_fit

from functools import partial
from pathlib import Path
from od3d.cv.geometry.transform import inv_tform4x4, tform4x4
from od3d.cv.geometry.transform import transf3d_broadcast
from od3d.cv.visual.show import plot_score_rotation_error


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

        self.dir_tmp = Path("/storage/user/jiso/tmp").joinpath(self.__class__.__name__)
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
        result_dict = {}
        src_sequences = dataset_src.get_sequences()
        ref_sequences = dataset_ref.get_sequences()
        # for sequence in src_sequences + ref_sequences:
        #     if self.config.preprocess.pcl.enabled:
        #         sequence.preprocess_pcl(override=self.config.preprocess.pcl.override)
        #     if self.config.preprocess.tform_obj.enabled:
        #         sequence.preprocess_tform_obj(
        #             override=self.config.preprocess.tform_obj.override,
        #         )
        #     if self.config.preprocess.mesh.enabled:
        #         sequence.preprocess_mesh(override=self.config.preprocess.mesh.override)
        #     if self.config.preprocess.mesh_feats.enabled:
        #         sequence.preprocess_mesh_feats(
        #             override=self.config.preprocess.mesh_feats.override,
        #         )

        # if self.config.preprocess.mesh_feats_dist.enabled:
        #     for src_sequence in src_sequences:
        #         for ref_sequence in ref_sequences:
        #             if src_sequence.category == ref_sequence.category:
        #                 src_sequence.preprocess_mesh_feats_dist(
        #                     sequence=ref_sequence,
        #                     override=self.config.preprocess.mesh_feats_dist.override,
        #                 )
        #                 ref_sequence.preprocess_mesh_feats_dist(
        #                     sequence=src_sequence,
        #                     override=self.config.preprocess.mesh_feats_dist.override,
        #                 )

        # tform4x4(inv_tform4x4(src_frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D)), src_frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.DROID_SLAM))
        logger.info("loading mesh feats...")
        # self.sequences_mesh_feats = [seq.feats for seq in self.sequences]
        src_sequences_unique_names = [seq.name_unique for seq in src_sequences]
        ref_sequences_unique_names = [seq.name_unique for seq in ref_sequences]
        # sequences_unique_names = [seq.name_unique for seq in sequences]
        
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
        all_pred_ref_pts_offset = {}
        all_pred_ref_tform_src = {}
        all_pred_pose_dist_geo = {}
        all_pred_pose_dist_appear = {}
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
                src_instance_ids = torch.LongTensor(list(range(src_instances_count)))
                ref_instance_ids = torch.LongTensor(list(range(ref_instances_count)))

                src_mesh_ids = src_instance_ids[src_map_seq_to_cat == cat_id]
                ref_mesh_ids = ref_instance_ids[ref_map_seq_to_cat == cat_id]

                rot_deg_diff_list = []
                for r, ref_mesh_id in enumerate(ref_mesh_ids):
                    for s, src_mesh_id in enumerate(src_mesh_ids):
                        if ref_sequences_unique_names[r] == src_sequences_unique_names[s]:
                            continue
                        # src_sequences[src_mesh_id].show(show_imgs=True)
                        print("ref seq ", ref_sequences_unique_names[r])
                        print("src_seq ", src_sequences_unique_names[s])
                        src_vertices_mask = (
                            src_sequences_mesh_ids_for_verts == src_mesh_id
                        )
                        # flipped_src_vertices_mask = (
                        #      flipped_src_sequences_mesh_ids_for_verts == src_mesh_id
                        # )
                        # src_vertices = torch.arange(src_vertices_count).to(device=self.device)[src_vertices_mask]
                        pts_src = src_meshes.verts[src_vertices_mask].clone()
                        # pt_src_rgb = src_meshes.rgb[src_vertices_mask].clone()
                        # pts_src_flip = flipped_src_meshes.verts[flipped_src_vertices_mask].clone()
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
                                print("dist_src_ref:", dist_src_ref)
                                # division by two to normalize to 0. - 1.
                                dist_src_ref = dist_src_ref / 2.0
                                print("dist_src_ref:", dist_src_ref)

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
                                logger.info(f'sim: geo: {pose_dist_geo}, app: {pose_dist_appear}')
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
                                ##HERE
                                ref_vertices_mask = (
                                    ref_sequences_mesh_ids_for_verts == ref_mesh_id
                                )
                                pts_ref = ref_meshes.verts[ref_vertices_mask].clone()

                                logger.info(
                                    f"category: {category}, pts-src: {pts_src.shape}, pts-ref: {pts_ref.shape}",
                                )
                                
                                # Optimal Transport
                                from od3d.datasets.ot.optimal_transport import calculate_distance_matrix, save_num_matches, load_vertices
                                
                                root_path = ref_sequences[ref_mesh_id].path_preprocess
                                category = ref_sequences[ref_mesh_id].name_unique.split('/')[0]
                                sequence1 = ref_sequences[ref_mesh_id].name_unique.split('/')[1]
                                sequence2 = src_sequences[src_mesh_id].name_unique.split('/')[1]                             
                                
                                from od3d.datasets.ot.optimal_transport import ot_based_ransac
                                (
                                    src_tform4x4_ref,
                                    best_correspondence,
                                    best_ref_correspondence,
                                    src_tform4x4_ref_score,
                                    dist_ref_src,
                                ) = ot_based_ransac(
                                    root_path=root_path,
                                    category=category,
                                    sequence1=sequence1,
                                    sequence2=sequence2,
                                )

                                print(dist_ref_src.shape)
                                print("ref_tform4x4_src:", src_tform4x4_ref)
                                src_tform4x4_ref = src_tform4x4_ref.to(device=self.device)

                                # from od3d.cv.visual.correspondences_vis import (
                                #     save_visualization_mesh,
                                #     save_visualization_mesh_with_color,
                                # )

                                # partial_ratio_for_saving = src_sequences[
                                #     src_mesh_id
                                # ].partial_ratio
                                # start_frame_id_for_saving = src_sequences[
                                #     src_mesh_id
                                # ].start_frame_id
                                # use_sph = src_sequences[src_mesh_id].use_sph
                                # sfm_type_for_saving = src_sequences[
                                #     src_mesh_id
                                # ].sfm_type
                                # pcl_type_for_saving = src_sequences[
                                #     src_mesh_id
                                # ].pcl_type

                                # use_sd = src_sequences[src_mesh_id].use_sd
                                src_name = src_sequences[src_mesh_id].name
                                ref_name = ref_sequences[ref_mesh_id].name
                                # vi_mesh_path = (
                                #     "/storage/user/jiso/output/vis_meshes"
                                # )
                                # category_ratio_use_sph_path = os.path.join(
                                #     vi_mesh_path,
                                #     f"{category}_partial_ratio_{partial_ratio_for_saving}_start_frame_{start_frame_id_for_saving}_use_sph_{use_sph}_use_sd_{use_sd}",
                                # )
                                # os.makedirs(category_ratio_use_sph_path, exist_ok=True)
                                # sfm_pcl_type_path = os.path.join(
                                #     category_ratio_use_sph_path,
                                #     f"sfm_type_{sfm_type_for_saving}_pcl_type_{pcl_type_for_saving}",
                                # )
                                # os.makedirs(sfm_pcl_type_path, exist_ok=True)
                                # folder_path = os.path.join(
                                #     sfm_pcl_type_path,
                                #     f"ref_id_{ref_name}_src_id_{src_name}_aligned",
                                # )
                                # folder_path_flip = os.path.join(
                                #     sfm_pcl_type_path,
                                #     f"ref_id_{ref_name}_src_id_{src_name}_aligned_flip",
                                # )
                                # os.makedirs(folder_path, exist_ok=True)
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

                                (
                                    src_pts3d,
                                    src_pts3d_colors,
                                    src_pts3d_normals,
                                ) = src_sequences[src_mesh_id].read_pcl()
                                (
                                    ref_pts3d,
                                    ref_pts3d_colors,
                                    ref_pts3d_normals,
                                ) = ref_sequences[ref_mesh_id].read_pcl()

                                # save_visualization_mesh(
                                #     pts=pts_src,
                                #     pts_ref=pts_ref,
                                #     pts_ids=best_correspondence,
                                #     pts_ref_ids=best_ref_correspondence,
                                #     filename=os.path.join(
                                #         sfm_pcl_type_path,
                                #         f"ref_id_{ref_name}_src_id_{src_name}",
                                #     ),
                                # )

                                # save_visualization_mesh(
                                #     pts=transformed_pts_src[:, :3],
                                #     pts_ref=pts_ref,
                                #     pts_ids=best_correspondence,
                                #     pts_ref_ids=best_ref_correspondence,
                                #     filename=folder_path,
                                # )

                                # save_visualization_mesh_with_color(
                                #     pts=pts_src,
                                #     pts_ref=pts_ref,
                                #     pts_ids=best_correspondence,
                                #     pts_ref_ids=best_ref_correspondence,
                                #     filename=os.path.join(
                                #         sfm_pcl_type_path,
                                #         f"ref_id_{ref_name}_src_id_{src_name}",
                                #     ),
                                #     original_pts_ref=ref_pts3d,
                                #     original_pts_color_ref=ref_pts3d_colors,
                                #     original_pts_src=src_pts3d,
                                #     original_pts_color_src=src_pts3d_colors,
                                # )

                                # save_visualization_mesh_with_color(pts= transformed_pts_src[:,:3], pts_ref= pts_ref, pts_ids= best_correspondence, pts_ref_ids= best_ref_correspondence, filename= folder_path)

                                # os.makedirs(
                                #     os.path.join(sfm_pcl_type_path, f"{category}"),
                                #     exist_ok=True,
                                # )
                                # np.save( os.path.join(sfm_pcl_type_path, f'{category}',f'dino_{ref_name}.npy'), ref_mesh_feat_attached)
                                # np.save( os.path.join(sfm_pcl_type_path, f'{category}',f'dino_{src_name}.npy'), src_mesh_feat_attached)
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

                        gt_ref_tform_src = torch.eye(4).to(device=self.device)

                        diff_rot_angle_rad = get_pose_diff_in_rad(
                            pred_tform4x4=pred_ref_tform_src,
                            gt_tform4x4=gt_ref_tform_src,
                        )

                        # logger.info(f'diff_rot_angle_rad is  {diff_rot_angle_rad}')
                        logger.info(f"ref seq {ref_sequences_unique_names[r]}")
                        logger.info(f"src_seq {src_sequences_unique_names[s]}")
                        logger.info(
                            f"diff rot degree for the optimal transport is {180 * diff_rot_angle_rad / torch.pi}"
                        )
                        # logger.info(f'diff_rot_degree mast3r is {180 * diff_rot_angle_rad_mast3r / torch.pi}')
                        print(
                            "------------------------------------------------------------------------------------------------------------"
                        )
                        results_diff_log_rot[category][r, s] = diff_rot_angle_rad
                        # mast3r_results_diff_log_rot[category][r,s] = diff_rot_angle_rad_mast3r
                        # results_intrinsic_orientation[category][r, s] = product_result
                        if r != s:
                            rot_deg_diff_list.append(
                                180 * diff_rot_angle_rad / torch.pi
                            )

        rot_deg_diff_list_numpy = np.array(
            [tensor.item() for tensor in rot_deg_diff_list]
        )
        plt.hist(rot_deg_diff_list_numpy, bins=12, edgecolor="black")
        plt.title("Rotational Degree Error Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        import io

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        import wandb

        def add_table_to_wandb(matrix_list, title_list, category):
            # Create a figure with three subplots
            fig, axes = plt.subplots(1, 5, figsize=(30, 6))  # Adjusted subplot creation
            # Loop through the list of matrices and titles
            for ax, matrix, title in zip(axes, matrix_list, title_list):
                cax = ax.matshow(matrix, cmap="viridis")
                fig.colorbar(cax, ax=ax)
                ax.set_xlabel("Column Index")
                ax.set_ylabel("Row Index")
                ax.set_title(title)
            # Log to wandb
            results = OD3D_Results()
            results[f"{category}"] = wandb.Image(fig)
            results.log_with_prefix(category)
            # Close the plot to prevent memory issues
            plt.close(fig)

        matrix_list = []
        matrix_list.append(results_diff_log_rot[category].cpu().numpy())
        matrix_list.append(all_pred_pose_dist_geo[category].cpu().numpy())
        matrix_list.append(all_pred_pose_dist_appear[category].cpu().numpy())

        title_list = []
        title_list.append("rotational_degree_error")
        title_list.append("geo distance")
        title_list.append("appearance distance")

        add_table_to_wandb(matrix_list, title_list, category)
        pi6_list = []
        pi12_list = []
        pi18_list = []
        mean_list = []
        # results_ref = OD3D_Results()
        for cat_id, category in enumerate(categories):
            category_results = OD3D_Results()
            exclude_diagonal = dataset_src.name == dataset_ref.name

            if exclude_diagonal:
                # excluding diagonal entries as these are predicted transformation between same instance
                if self.config.gt_cam_tform_obj_source is not None:
                    category_results[f"rot_diff_rad"] = (
                        results_diff_log_rot[category][
                            torch.eye(src_instances_count_per_category[cat_id]).to(
                                device=self.device,
                            )
                            == 0
                        ]
                        .reshape(
                            ref_instances_count_per_category[cat_id],
                            src_instances_count_per_category[cat_id] - 1,
                        )
                        .permute(1, 0)
                    )
                category_results[f"sim"] = 1.0 - all_pred_pose_dist_geo[category][
                    torch.eye(src_instances_count_per_category[cat_id]).to(
                        device=self.device,
                    )
                    == 0
                ].reshape(
                    ref_instances_count_per_category[cat_id],
                    src_instances_count_per_category[cat_id] - 1,
                ).permute(
                    1,
                    0,
                )
                category_results[f"pose_sim_geo"] = 1.0 - all_pred_pose_dist_geo[
                    category
                ][
                    torch.eye(src_instances_count_per_category[cat_id]).to(
                        device=self.device,
                    )
                    == 0
                ].reshape(
                    ref_instances_count_per_category[cat_id],
                    src_instances_count_per_category[cat_id] - 1,
                ).permute(
                    1,
                    0,
                )
                category_results[f"pose_sim_appear"] = 1.0 - all_pred_pose_dist_appear[
                    category
                ][
                    torch.eye(src_instances_count_per_category[cat_id]).to(
                        device=self.device,
                    )
                    == 0
                ].reshape(
                    ref_instances_count_per_category[cat_id],
                    src_instances_count_per_category[cat_id] - 1,
                ).permute(
                    1,
                    0,
                )
            else:
                if self.config.gt_cam_tform_obj_source is not None:
                    category_results[f"rot_diff_rad"] = results_diff_log_rot[
                        category
                    ].permute(1, 0)
                category_results[f"sim"] = 1.0 - all_pred_pose_dist_geo[
                    category
                ].permute(1, 0)
                category_results[f"pose_sim_geo"] = 1.0 - all_pred_pose_dist_geo[
                    category
                ].permute(1, 0)
                category_results[f"pose_sim_appear"] = 1.0 - all_pred_pose_dist_appear[
                    category
                ].permute(1, 0)

            # results += category_results  # .mean()
            category_results_mean = category_results.add_prefix(category)
            category_results_mean = category_results_mean.mean()
            category_results_mean.log()
            result_dict[category] = {}
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
            # with open(f'baseline_result.pickle', 'wb') as handle:
            #     pickle.dump(result_dict, handle, protocol= pickle.HIGHEST_PROTOCOL)
            logger.info(f"category_results_mean is,  {category_results_mean}")

     
    def test(self, dataset: OD3D_Dataset, config_inference: DictConfig = None):
        return OD3D_Results()
