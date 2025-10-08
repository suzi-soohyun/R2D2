import os

import numpy as np
import torch
from od3d.cv.metric.pose import get_pose_diff_in_rad
from od3d.cv.optimization.masked_posegraph import (
    WeightedPoseGraphOptimizationRotationMatrixMasking,
)
from scipy.linalg import eig
from scipy.optimize import minimize
from scipy.special import ive  # Modified Bessel function of the first kind, order v
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

MIN_SAMPLE = 1


def filter(tensor):
    isnan = torch.isnan(tensor)
    row_with_all_nan = isnan.all(dim=1)
    valid_rows = ~row_with_all_nan
    filtered_tensor = tensor[valid_rows]
    original_indices = torch.arange(tensor.size(0))[valid_rows]  # Keep original indices
    return filtered_tensor, original_indices


def filter_mesh_feats_acc(list, ref_indices):
    result = []
    for i in ref_indices:
        x = list[i]
        result.append(x)

    return result


def get_max_cossine_similarity(tensor1, tensor2):
    tensor1 = tensor1 / tensor1.norm(dim=1, keepdim=True)
    tensor2 = tensor2 / tensor2.norm(dim=1, keepdim=True)
    cosine_similarities = torch.mm(tensor1, tensor2.t())
    return cosine_similarities.max()


def get_min_cossine_similarity(tensor1, tensor2):
    tensor1 = tensor1 / tensor1.norm(dim=1, keepdim=True)
    tensor2 = tensor2 / tensor2.norm(dim=1, keepdim=True)
    cosine_similarities = torch.mm(tensor1, tensor2.t())
    return cosine_similarities.min()


def get_mean_cossine_similarity(tensor1, tensor2):
    tensor1 = tensor1 / tensor1.norm(dim=1, keepdim=True)
    tensor2 = tensor2 / tensor2.norm(dim=1, keepdim=True)
    cosine_similarities = torch.mm(tensor1, tensor2.t())
    return torch.mean(cosine_similarities)


def preprocess_mesh_feats_acc(mesh_feat_acc):
    mesh_feats_acc_cluster = MeshFeatsVertexClustering(mesh_feat_acc)
    (
        mesh_feats_acc_cluster_mean,
        original_indices,
        mesh_feats_filtered,
    ) = mesh_feats_acc_cluster.preprocess()
    return (
        mesh_feats_acc_cluster_mean,
        torch.tensor(np.array(original_indices)).cuda(),
        mesh_feats_filtered,
    )


def get_shared_region_mesh_vertex_feat_cluster_mean(list_A, list_B, threshold):
    list_A_index_list = []
    list_B_index_list = []
    for i in range(len(list_A)):
        # Check if list_A[i] is empty using .size
        if list_A[i].size == 0:
            continue  # Skip this iteration if list_A[i] is empty

        for j in range(len(list_B)):
            # Check if list_B[j] is empty using .size
            if list_B[j].size == 0:
                continue  # Skip this iteration if list_B[j] is empty

            # Calculate the maximum dot product between features and check against the threshold
            # if np.min(list_A[i] @ list_B[j].T) > threshold:
            if np.max(list_A[i] @ list_B[j].T) > threshold:
                list_A_index_list.append(i)
                list_B_index_list.append(j)

    return list_A_index_list, list_B_index_list


def get_max_cosine_similarity_vertex_pairs(vertex_A_feat, vertex_B_feat):
    tmp = -2.0
    for i in range(len(vertex_A_feat)):
        for j in range(len(vertex_B_feat)):
            if vertex_A_feat[i].T @ vertex_B_feat[j] > tmp:
                tmp = vertex_A_feat[i].T @ vertex_B_feat[j]
                # assert -1 <= tmp <= 1, "Cosine similarity out of range [-1, 1]"
    return tmp


def get_min_cosine_similarity_vertex_pairs(vertex_A_feat, vertex_B_feat):
    tmp = 2.0
    for i in range(len(vertex_A_feat)):
        for j in range(len(vertex_B_feat)):
            if vertex_A_feat[i].T @ vertex_B_feat[j] < tmp:
                tmp = vertex_A_feat[i].T @ vertex_B_feat[j]
                # assert -1 <= tmp <= 1, "Cosine similarity out of range [-1, 1]"
    return tmp


class MeshFeatsVertexClustering:
    def __init__(self, mesh_feats):
        self.mesh_feats = mesh_feats
        self.mesh_feats_cluster_mean = []
        self.original_indices = []
        self.mesh_feats_filtered = []

    def preprocess(self):
        for i in range(len(self.mesh_feats)):
            vertex_mesh_feat = self.mesh_feats[i]

            # Convert to NumPy array and ensure it's not empty
            if vertex_mesh_feat.shape[0] == 0:
                # print(f"Skipping empty vertex feature at index {i}.")
                continue

            self.original_indices.append(i)
            vertex_sph_feat = vertex_mesh_feat[:, :3].numpy()

            # Normalize the vectors to unit length unless they are zero vectors
            norms = np.linalg.norm(vertex_sph_feat, axis=1, keepdims=True)
            vertex_sph_feat /= norms
            self.mesh_feats_filtered.append(vertex_sph_feat)

            similarity = cosine_similarity(vertex_sph_feat)
            distance = np.clip(1 - similarity, 0, None)

            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=0.1, min_samples=MIN_SAMPLE, metric="precomputed")
            clusters = dbscan.fit_predict(distance)

            mean_vectors = []
            # Calculate the mean direction of each cluster
            for label in np.unique(clusters):
                if label == -1:  # Exclude noise points
                    continue
                # if label > 0:
                #     print('look!!!!')
                # Select data points belonging to the current cluster
                members = vertex_sph_feat[clusters == label]
                if members.size == 0:
                    print(f"No members in cluster {label} at index {i}.")
                    continue

                # Calculate the mean by summing and normalizing
                mean_vector = np.mean(members, axis=0)
                mean_vector /= np.linalg.norm(mean_vector)
                mean_vectors.append(mean_vector)
            mean_vectors = np.array(mean_vectors)
            self.mesh_feats_cluster_mean.append(mean_vectors)

        return (
            self.mesh_feats_cluster_mean,
            self.original_indices,
            self.mesh_feats_filtered,
        )


class TemperatureScaling:
    def __init__(
        self, rotation_matrics, weighted_matrics, temperature, annotated_idx, seq_list
    ):
        self.rotation_matrics = rotation_matrics
        self.weighted_matrics = weighted_matrics
        self.temperature = temperature
        self.annotated_idx = annotated_idx
        self.seq_list = seq_list

    def preprocess_weight_entire_matrix(self):
        diagonal_mask = torch.eye(self.weighted_matrics.shape[0], dtype=torch.bool)
        self.weighted_matrics[diagonal_mask] = 1
        off_diagonal_mask = (~diagonal_mask).cuda()
        weighted_matrics_flat = self.weighted_matrics[off_diagonal_mask].view(-1)
        weighted_matrics_softmax = torch.nn.functional.softmax(
            weighted_matrics_flat / self.temperature, dim=0
        )
        self.weighted_matrics[off_diagonal_mask] = weighted_matrics_softmax.view(
            self.weighted_matrics[off_diagonal_mask].shape
        )
        return self.weighted_matrics

    # def preprocess_weight_row_wise(self):
    #     diagonal_mask = torch.eye(self.weighted_matrics.shape[0], dtype=torch.bool).to(self.weighted_matrics.device)
    #     # Keep diagonal values safe (store original or set them very high to ensure they turn to 1 after softmax)
    #     diagonal_values = self.weighted_matrics.diagonal().clone()
    #     # Set diagonal elements to a very low value to exclude them effectively from softmax
    #     self.weighted_matrics[diagonal_mask] = float('-inf')
    #     # Apply softmax row-wise
    #     self.weighted_matrics = torch.nn.functional.softmax(self.weighted_matrics / temperature, dim=1)
    #     # Restore diagonal values to 1
    #     self.weighted_matrics[diagonal_mask] = 1

    #     return self.weighted_matrics

    @staticmethod
    def calculate(rad):
        num_pi_6 = 0
        num_pi_12 = 0
        num_pi_18 = 0
        errors = 0
        for i in range(rad.shape[0]):
            for j in range(rad.shape[1]):
                if i != j:
                    errors += rad[i, j]
                    if rad[i, j] < torch.pi / 6:
                        num_pi_6 += 1
                        if rad[i, j] < torch.pi / 12:
                            num_pi_12 += 1
                            if rad[i, j] < torch.pi / 18:
                                num_pi_18 += 1

        pi_6_acc = num_pi_6 / (rad.shape[0] * (rad.shape[0] - 1))
        pi_12_acc = num_pi_12 / (rad.shape[0] * (rad.shape[0] - 1))
        pi_18_acc = num_pi_18 / (rad.shape[0] * (rad.shape[0] - 1))
        error_mean = errors / (rad.shape[0] * (rad.shape[0] - 1))

        print("pi_6_acc ", pi_6_acc)
        print("pi_12_acc ", pi_12_acc)
        print("pi_18_acc ", pi_18_acc)
        print("error mean ", error_mean * 180 / torch.pi)

    def run_pgo(self):
        # print('Wihout PGO')
        # self.calculate(self.rotation_matrics)
        # print('---------------------------------------------------------')
        # self.preprocess_weight()
        # self.preprocess_weight_row_wise()
        self.preprocess_weight_entire_matrix()
        pgo = WeightedPoseGraphOptimizationRotationMatrixMasking(
            self.rotation_matrics,
            weighted_matrices=self.weighted_matrics,
            seq_list=self.seq_list,
        )
        # pgo = WeightedPoseGraphOptimizationRotationMatrixMaskingTop10(self.rotation_matrics, weighted_matrices= self.weighted_matrics)

        updated_rad = pgo.optimization()
        updated_rad = updated_rad[self.annotated_idx, :][:, self.annotated_idx]
        # print('updated_rad ', updated_rad)

        init_rad = pgo.return_pose_graph_init_transformations()
        init_rad = init_rad[self.annotated_idx, :][:, self.annotated_idx]
        print("PGO INIT")
        self.calculate(init_rad)
        print("---------------------------------------------------------")
        print("PGO UPDATED")
        self.calculate(updated_rad)


class BaselineAlignmentToCanonicalization:
    def __init__(self, baseline_alignments):
        self.baseline_alignments = baseline_alignments

    def convert(self):
        result = []
        for i in range(self.baseline_alignments.shape[0]):
            pose_node = {}
            alignments = self.baseline_alignments[i]
            for j in range(self.baseline_alignments.shape[0]):
                pose_node[f"{j}"] = torch.linalg.inv(alignments[j])
            result.append(pose_node)

        return result


if __name__ == "__main__":
    # category = ''
    import json

    category = "toytruck"
    # for temperature in [1.0, 5.0, 10.0,15.0]:
    # for temperature in [1.0, 10.0]:
    temperature = 10.0
    print("THE TEMPERATURE IS ", temperature)
    base_path = "/storage/user/jiso/output"
    weight_path = f"scripts/vis_meshes/align_with_unannotated/category_{category}/ratio_25/threshold_90/use_sph_sph_excludes_co3d_with_dino_mixing_ratio_0.2/pgo_init/weighted_matrics.pt"
    rotation_path = f"scripts/vis_meshes/align_with_unannotated/category_{category}/ratio_25/threshold_90/use_sph_sph_excludes_co3d_with_dino_mixing_ratio_0.2/pgo_init/input_alignments.pt"
    annotated_idx_path = f"scripts/vis_meshes/align_with_unannotated/category_{category}/ratio_25/threshold_90/use_sph_sph_excludes_co3d_with_dino_mixing_ratio_0.2/pgo_init/annotated_idx.pt"
    seq_list_path = f"scripts/vis_meshes/align_with_unannotated/category_{category}/ratio_25/threshold_90/use_sph_sph_excludes_co3d_with_dino_mixing_ratio_0.2/pgo_init/seq_list.json"
    with open(os.path.join(base_path, seq_list_path)) as json_file:
        seq_list = json.load(json_file)

    weight = torch.load(os.path.join(base_path, weight_path))
    input_alignment = torch.load(os.path.join(base_path, rotation_path))
    annotated_idx_list = torch.load(os.path.join(base_path, annotated_idx_path))
    tmp = TemperatureScaling(
        rotation_matrics=input_alignment,
        weighted_matrics=weight,
        temperature=temperature,
        annotated_idx=annotated_idx_list,
        seq_list=seq_list,
    )
    pgo_run = tmp.run_pgo()
    print(
        "------------------------------------------------------------------------------------------------------------------"
    )
