import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from od3d.cv.geometry.transform import inv_tform4x4
from od3d.cv.metric.pose import get_pose_diff_in_rad
from od3d.methods.nemo_align3d.utils import get_max_cosine_similarity_vertex_pairs
from od3d.methods.nemo_align3d.utils import get_min_cosine_similarity_vertex_pairs
from scipy.spatial import cKDTree

# THRESHOLD = 0.90
# VAR_MAX = 0.30
THRESHOLD = 0.80
VAR_MAX = 0.30


def plot(angle_list, max_value_list_mean_list, min_value_list_mean_list):
    # Assuming angle_list, max_value_list_mean_list, and min_value_list_mean_list
    # are already tensors and are on the appropriate device (like CPU), if not, ensure to move them first.
    angle_array = np.array(angle_list.cpu().numpy())
    max_values = np.array(max_value_list_mean_list.cpu().numpy())
    min_values = np.array(min_value_list_mean_list.cpu().numpy())

    plt.figure(figsize=(10, 4))  # Adjust the size as necessary

    # Plot Max Dist vs. Angle
    plt.subplot(1, 2, 1)  # This plots on the first half of the figure
    plt.scatter(
        max_values, angle_array, color="red", alpha=0.5, label="Max Value vs. Angle"
    )
    plt.title("Max Value vs. Angle")
    plt.ylabel("Angle Difference (radians)")
    plt.xlabel("Max Cosine Similarity Mean")
    plt.legend()
    plt.savefig("sph_max_angle.png")

    # Plot Min Dist vs. Angle
    plt.subplot(1, 2, 2)  # This plots on the second half of the figure
    plt.scatter(
        min_values, angle_array, color="blue", alpha=0.5, label="Min Value vs. Angle"
    )
    plt.title("Min Value vs. Angle")
    plt.ylabel("Angle Difference (radians)")
    plt.xlabel("Min Cosine Similarity Mean")
    plt.legend()
    plt.savefig("sph_min_angle.png")

    plt.tight_layout()  # Adjust the layout to make room for everything
    plt.show()


def cal_var(V):
    if V.shape[0] < 2:
        # raise ValueError("Insufficient data points for covariance calculation (at least 2 points required).")
        return 0, 0, 0

    # Mean center the data
    V_centered = V - np.mean(V, axis=0)
    # Compute the covariance matrix
    covariance_matrix = np.cov(V_centered, rowvar=False)
    # Calculate trace variance
    trace_variance = np.trace(covariance_matrix)
    # Calculate average variance
    average_variance = np.mean(np.diag(covariance_matrix))
    # Calculate largest eigenvalue (take the real part to avoid numerical issues)
    eigenvalues = np.linalg.eigvals(covariance_matrix)
    largest_variance = np.max(np.real(eigenvalues))

    return trace_variance, average_variance, largest_variance


def compare_sph_dino(
    models,
    pts_src_ids,
    pts_ref_ids,
    sph_src,
    sph_ref,
    scores,
    r,
    s,
    category,
    sph_src_filtered,
    sph_ref_filtered,
):
    gt_ref_tform_src = torch.eye(4).cuda()
    diff_rot_angle_rad_list = []
    max_value_list_mean_list = []
    min_value_list_mean_list = []
    score_list = []
    selected_model = []
    for i in range(len(models)):
        model = models[i]
        diff_rot_angle_rad = get_pose_diff_in_rad(
            pred_tform4x4=inv_tform4x4(model),
            gt_tform4x4=gt_ref_tform_src,
        )
        pts_src_id = pts_src_ids[i]
        pts_ref_id = pts_ref_ids[i]
        max_value_list = []
        min_value_list = []

        var_src_list = []
        var_ref_list = []
        # for each model, try to constraint the cossine similarity for each correspondence pairs
        for j in range(4):
            feat_src = sph_src_filtered[pts_src_id[j]]
            feat_ref = sph_ref_filtered[pts_ref_id[j]]

            trace_variance_src, average_variance_src, largest_variance_src = cal_var(
                feat_src
            )
            trace_variance_ref, average_variance_ref, largest_variance_ref = cal_var(
                feat_ref
            )

            var_src_list.append(trace_variance_src)
            var_ref_list.append(trace_variance_ref)
            # if trace_variance_src > VAR_MAX or trace_variance_ref > VAR_MAX:
            #     break

            max_value = get_max_cosine_similarity_vertex_pairs(
                sph_src[pts_src_id[j]], sph_ref[pts_ref_id[j]]
            )
            min_value = get_min_cosine_similarity_vertex_pairs(
                sph_src[pts_src_id[j]], sph_ref[pts_ref_id[j]]
            )

            max_value_list.append(max_value)
            min_value_list.append(min_value)

        count_above_threshold = sum(1 for value in max_value_list if value > THRESHOLD)
        # count_above_threshold = sum(1 for value in min_value_list if value > THRESHOLD)
        count_low_var = sum(
            1
            for var_src, var_ref in zip(var_src_list, var_ref_list)
            if var_src < VAR_MAX and var_ref < VAR_MAX
        )
        # if count_above_threshold ==4 and  count_low_var == 4:
        if count_low_var == 4:
            # print('max_value_list ', max_value_list)
            # print('diff_rot_angle_rad ', diff_rot_angle_rad)
            # print('--------------------------------------')
            # print('pts_ref ', )
            diff_rot_angle_rad_list.append(diff_rot_angle_rad)
            score_list.append(scores[i])
            selected_model.append(models[i])

    plt.figure(figsize=(10, 4))  # Adjust the size as necessary
    # print('scores list ratio ', len(score_list)/ len(models))
    # Plot Max Dist vs. Angle
    plt.subplot(1, 2, 1)  # This plots on the first half of the figure
    score_list = torch.tensor(score_list)
    diff_rot_angle_rad_list = torch.tensor(diff_rot_angle_rad_list)
    score_list = score_list.cpu().numpy()
    diff_rot_angle_rad_list = diff_rot_angle_rad_list.cpu().numpy()
    plt.scatter(
        score_list,
        diff_rot_angle_rad_list,
        color="red",
        alpha=0.5,
        label="Max Value vs. Angle",
    )
    plt.title("Score Value vs. Angle")
    plt.ylabel("Angle Difference (radians)")
    plt.xlabel("Scores")
    plt.legend()
    os.makedirs(
        os.path.join("test", f"cate_{category}_theshold_{THRESHOLD}"), exist_ok=True
    )
    plt.savefig(
        os.path.join(
            "test",
            f"cate_{category}_theshold_{THRESHOLD}",
            f"scores_angles_ref_name_{r}_src_name_{s}.png",
        )
    )
    if len(score_list) > 0:
        my_best_score_id = np.argmax(score_list)
        print(
            "MY BEST diff_rot_angle_rad_list is ",
            diff_rot_angle_rad_list[my_best_score_id] * 180 / np.pi,
        )

    # return len(score_list)/ len(models), selected_model[my_best_score_id]
    return len(score_list) / len(models)

    # diff_rot_angle_rad_list = torch.tensor(diff_rot_angle_rad_list)
    # max_value_list_mean_list = torch.tensor(max_value_list_mean_list)
    # min_value_list_mean_list = torch.tensor(min_value_list_mean_list)

    # plot(diff_rot_angle_rad_list,max_value_list_mean_list, min_value_list_mean_list)


def compare_sph_dino_best_id(
    model,
    pts_src_id,
    pts_ref_id,
    sph_src,
    sph_ref,
    pts_ref,
    pts_src,
    sph_src_filtered,
    sph_ref_filtered,
):
    gt_ref_tform_src = torch.eye(4).cuda()

    diff_rot_angle_rad = get_pose_diff_in_rad(
        pred_tform4x4=inv_tform4x4(model),
        gt_tform4x4=gt_ref_tform_src,
    )
    for id in range(4):
        max_value = get_max_cosine_similarity_vertex_pairs(
            sph_src[pts_src_id[id]], sph_ref[pts_ref_id[id]]
        )
        min_value = get_min_cosine_similarity_vertex_pairs(
            sph_src[pts_src_id[id]], sph_ref[pts_ref_id[id]]
        )
        pts_ref_corr = pts_ref[pts_ref_id[id]]
        pts_src_corr = pts_src[pts_src_id[id]]

        feat_src = sph_src_filtered[pts_src_id[id]]
        feat_ref = sph_ref_filtered[pts_ref_id[id]]

        trace_variance_src, average_variance_src, largest_variance_src = cal_var(
            feat_src
        )
        trace_variance_ref, average_variance_ref, largest_variance_ref = cal_var(
            feat_ref
        )

        print("max value ", max_value)
        print("min value ", min_value)
        print("pts ref corr ", pts_ref_corr)
        print("pts src corr ", pts_src_corr)
        print("trace_variance_src ", trace_variance_src)
        print("trace_variance_ref ", trace_variance_ref)

        print("average_variance_src ", average_variance_src)
        print("average_variance_ref ", average_variance_ref)

        print("largest_variance_src ", largest_variance_src)
        print("largest_variance_ref ", largest_variance_ref)

        print("--------------------------------------------------------------")


def plot_relationships(
    dist_list, feat_dist_list, angle_list, scores, geo_score, semantic_score
):
    # Convert all lists to numpy arrays for easier handling
    dist_array = np.array(dist_list.cpu())
    feat_dist_array = np.array(feat_dist_list.cpu())
    angle_array = np.array(angle_list.cpu())
    scores_array = np.array(scores.cpu())
    geo_score = np.array(geo_score.cpu())
    semantic_score = np.array(semantic_score.cpu())
    # Creating plots
    plt.figure(figsize=(20, 4))

    # Distance vs. Angle
    plt.subplot(1, 5, 1)
    plt.scatter(dist_array, angle_array, c="r", alpha=0.5, label="Dist vs. Angle")
    plt.xlabel("Distance Sum")
    plt.ylabel("Angle Difference (radians)")
    plt.title("Distance vs. Angle Difference")
    plt.legend()
    plt.savefig("distance_angle.png")

    # Feature Distance vs. Angle
    plt.subplot(1, 5, 2)
    plt.scatter(
        feat_dist_array, angle_array, c="b", alpha=0.5, label="Feat Dist vs. Angle"
    )
    plt.xlabel("Feature Distance Sum")
    plt.ylabel("Angle Difference (radians)")
    plt.title("Feature Distance vs. Angle Difference")
    plt.legend()
    plt.savefig("feat_distance_angle.png")

    # Scores vs. Angle
    plt.subplot(1, 5, 3)
    plt.scatter(scores_array, angle_array, c="g", alpha=0.5, label="Scores vs. Angle")
    plt.xlabel("Scores")
    plt.ylabel("Angle Difference (radians)")
    plt.title("Scores vs. Angle Difference")
    plt.legend()
    plt.savefig("scores_angle.png")

    plt.subplot(1, 5, 4)
    plt.scatter(geo_score, angle_array, c="g", alpha=0.5, label="Geo Scores vs. Angle")
    plt.xlabel("Geo Scores")
    plt.ylabel("Angle Difference (radians)")
    plt.title("Geo Scores vs. Angle Difference")
    plt.legend()
    plt.savefig("geo_scores_angle.png")

    plt.subplot(1, 5, 5)
    plt.scatter(
        semantic_score, angle_array, c="g", alpha=0.5, label="Geo Scores vs. Angle"
    )
    plt.xlabel("Semantic Scores")
    plt.ylabel("Angle Difference (radians)")
    plt.title("Sem Scores vs. Angle Difference")
    plt.legend()
    plt.savefig("sem_scores_angle.png")

    plt.tight_layout()
    plt.show()


def get_closest_correspondence_model(
    models,
    best_correspondence,
    best_ref_correspondence,
    pts,
    pts_ref,
    dist_ref_src,
    scores,
    geo_score,
    semantic_score,
):
    dist_list = []
    feat_dist_list = []
    angle_list = []
    for i in range(len(models)):
        model = models[i]
        best_correspondence_id = best_correspondence[i]
        best_ref_correspondence_id = best_ref_correspondence[i]
        dist = pts[best_correspondence_id] - pts_ref[best_ref_correspondence_id]
        feat_dist_sum = 0
        for j in range(4):
            feat_dist = dist_ref_src[best_correspondence_id[j]][
                best_ref_correspondence_id[j]
            ]
            feat_dist_sum += feat_dist
        feat_dist_list.append(feat_dist_sum)
        dist_sum = torch.sum(torch.norm(dist, dim=1))
        dist_list.append(dist_sum.item())

        gt_ref_tform_src = torch.eye(4).cuda()

        diff_rot_angle_rad = get_pose_diff_in_rad(
            pred_tform4x4=inv_tform4x4(model),
            gt_tform4x4=gt_ref_tform_src,
        )
        angle_list.append(diff_rot_angle_rad)

    dist_list = torch.tensor(dist_list)
    feat_dist_list = torch.tensor(feat_dist_list)
    angle_list = torch.tensor(angle_list)
    plot_relationships(
        dist_list, feat_dist_list, angle_list, scores, geo_score, semantic_score
    )
    # feat_dist_list = torch.tensor(feat_dist_list)
    # dist_tensor = torch.tensor(dist_list)
    # _, indices = torch.topk(dist_tensor, 10, largest=False, sorted=True)
    # closest_models = [models[idx] for idx in indices]
    # for idx in indices:
    #     save_visualization_mesh(pts, pts_ref, best_ref_correspondence[idx], best_correspondence[idx], 'special_case'  )
    # return closest_models, indices.tolist()


def vis_heatmap_from_vertices(
    dist_ref_src, pts_ref, pts_src, pts_ref_ids, pts_ids, filename
):
    for i in range(len(pts_ref_ids)):
        pts_ref_id = pts_ref_ids[i]
        pts_src_id = pts_ids[i]
        dists = dist_ref_src[pts_ref_id].T.unsqueeze(1)
        pts_src = torch.concat([pts_src, dists], axis=1)
        pts_src_numpy = pts_src.cpu().numpy()
        np.save(os.path.join(filename, f"pts_src_with_dists_{i}.npy"), pts_src_numpy)


def save_visualization_mesh(pts, pts_ref, filename):
    os.makedirs(filename, exist_ok=True)
    # Convert tensors to numpy arrays
    pts_cloned = pts.clone().cpu().numpy()
    pts_ref_cloned = pts_ref.clone().cpu().numpy()
    # Create point clouds
    pts_point_cloud = o3d.geometry.PointCloud()
    pts_point_cloud.points = o3d.utility.Vector3dVector(pts_cloned)
    pts_point_cloud.colors = o3d.utility.Vector3dVector(
        np.tile([1, 0, 0], (pts.shape[0], 1))
    )  # Color each point as red

    pts_ref_point_cloud = o3d.geometry.PointCloud()
    pts_ref_point_cloud.points = o3d.utility.Vector3dVector(pts_ref_cloned)
    pts_ref_point_cloud.colors = o3d.utility.Vector3dVector(
        np.tile([0, 1, 0], (pts_ref.shape[0], 1))
    )  # Color each point as green

    # Save the point clouds to .ply files
    combined_pcd = pts_point_cloud + pts_ref_point_cloud
   
    o3d.io.write_point_cloud(os.path.join(filename, "aligned_meshes.ply"), combined_pcd)

    print(f"Saved point clouds to {filename}")

    # for i in range(len(pts_ids)):
    #     start_point = pts_cloned[pts_ref_ids[i]]
    #     end_point = pts_ref_cloned[pts_ids[i]]
    #     points = np.array([start_point, end_point])
    #     lines = np.array([[0, 1]])
    #     line_set = o3d.geometry.LineSet()
    #     line_set.points = o3d.utility.Vector3dVector(points)
    #     line_set.lines = o3d.utility.Vector2iVector(lines)

    #     # Optionally, define colors for the line
    #     colors = [[1, 0, 0]]  # Red color for the line
    #     line_set.colors = o3d.utility.Vector3dVector(colors)

    #     # Save the LineSet to a .ply file
    #     o3d.io.write_line_set(os.path.join(filename, f"line_segment_{i}.ply"), line_set)


def get_closest_points_color(pts, original_pts, original_pts_color):
    original_pts = original_pts.cpu().numpy()
    pts = pts.cpu().numpy()
    original_pts_color = original_pts_color.cpu().numpy()
    tree = cKDTree(original_pts)
    # Find the nearest neighbor in B for each element in A
    distances, indices = tree.query(pts, k=1)  # k=1 for the nearest neighbor
    return original_pts_color[indices]


def save_visualization_mesh_with_color(
    root_path,
    category,
    ref_name,
    src_name,
    pts_ref,
    pts_color_ref,
    pts_src,
    pts_color_src,
    pts_ref_ids,
    pts_src_ids,
    transformed_pts_src,
    baseline=False,
):
    seqname=f"ref_id_{ref_name}_src_id_{src_name}"
    folder_path = os.path.join(root_path, category, seqname)
    os.makedirs(folder_path, exist_ok=True)

    pts_point_cloud = o3d.geometry.PointCloud()
    pts_point_cloud.points = o3d.utility.Vector3dVector(
        pts_ref.clone().cpu().numpy()
    )
    pts_point_cloud.colors = o3d.utility.Vector3dVector(
        pts_color_ref.clone().cpu().numpy()
    )

    pts_ref_point_cloud = o3d.geometry.PointCloud()
    pts_ref_point_cloud.points = o3d.utility.Vector3dVector(
        pts_ref.clone().cpu().numpy()
    )
    pts_ref_point_cloud.colors = o3d.utility.Vector3dVector(
        pts_color_src.clone().cpu().numpy()
    )

    o3d.io.write_point_cloud(os.path.join(folder_path, "ref_colored.ply"), pts_point_cloud)
    o3d.io.write_point_cloud(
        os.path.join(folder_path, "src_colored.ply"), pts_ref_point_cloud
    )
    
    if baseline:
        pts_ref_ids = pts_ref[pts_ref_ids]
        pts_src_ids = pts_src[pts_src_ids]

    print(f"Saved point clouds to {folder_path}")
    vtx = {
        'category': category,
        'seq1': {
            'name': ref_name,
            'pose': pts_ref,
            'color': pts_color_ref,
        },
        'seq2': {
            'name': src_name,
            'pose': pts_src,
            'color': pts_color_src,
            'transformed_pose': transformed_pts_src,
        },
        'matches': {
            'match0': pts_ref_ids,
            'match1': pts_src_ids,
        }
    }
    
    vtx_numpy = tensor_to_numpy(vtx)
    import pickle
    with open(f"{folder_path}/matches_dict.pkl", "wb") as f:
        pickle.dump(vtx_numpy, f)

def tensor_to_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()  # move to CPU and convert to numpy
    elif isinstance(obj, dict):
        return {k: tensor_to_numpy(v) for k, v in obj.items()}
    else:
        return obj

def convert_pts_mesh(pts, file_path):
    pts_cloned = pts.clone().cpu().numpy()
    # Create point clouds
    pts_point_cloud = o3d.geometry.PointCloud()
    pts_point_cloud.points = o3d.utility.Vector3dVector(pts_cloned)
    pts_point_cloud.colors = o3d.utility.Vector3dVector(
        np.tile([0.3, 0.3, 0.3], (pts.shape[0], 1))
    )  # Color each point as gray

    # Save the point clouds to .ply files
    o3d.io.write_point_cloud(file_path, pts_point_cloud)


def compute_l2_distance(A, B):
    # A shape: [B, N1, C]
    # B shape: [B, N2, C]

    # Expand A and B to [B, N1, 1, C] and [B, 1, N2, C] respectively
    A_expanded = A.unsqueeze(2)  # Adds a dimension at position 2
    B_expanded = B.unsqueeze(1)  # Adds a dimension at position 1

    # Compute the squared differences, sum over the last dimension (feature dimension)
    # and take the square root to get the L2 distance
    distance = torch.sqrt(torch.sum((A_expanded - B_expanded) ** 2, dim=-1))

    return distance


def print_geo_feature_diff(pts_ids, pts_ref_ids, src_geo_feature, ref_geo_feature):
    distances = compute_l2_distance(ref_geo_feature, src_geo_feature)
    print("distance max", torch.amax(distances))
    print("distance min", torch.amin(distances))

    for i in range(len(pts_ids)):
        feature_diff = torch.norm(
            ref_geo_feature[:, pts_ids[i], :] - src_geo_feature[:, pts_ref_ids[i], :]
        )
        print(f"feature diff for {i}", feature_diff)


def find_nearest_neighbor_geo_feature(
    pts, pts_ref, pts_ids, pts_ref_ids, src_geo_features, ref_geo_feature, filename
):
    distances = compute_l2_distance(ref_geo_feature, src_geo_features)  # B, N1, N2
    for i in range(len(pts_ids)):
        values, indices = torch.topk(
            distances[:, pts_ids[i], :], 5, largest=False, sorted=True
        )
        for j in range(5):
            start_point = pts_ref[pts_ids[i]]
            end_point = pts[indices[0][j]]
            points = np.array([start_point.cpu().numpy(), end_point.cpu().numpy()])
            lines = np.array([[0, 1]])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)

            # Optionally, define colors for the line
            colors = [[1, 0, 0]]  # Red color for the line
            line_set.colors = o3d.utility.Vector3dVector(colors)

            # Save the LineSet to a .ply file
            o3d.io.write_line_set(
                os.path.join(filename, f"ref_{i}_src_{j}.ply"), line_set
            )


def save_pgo_init_meshes(pgo_init, meshes, mesh_ids, saved_folder):
    for i in range(len(pgo_init)):
        edge = pgo_init[i]
        source = int(edge["source"])
        target = int(edge["target"])
        ref_vertices_mask = mesh_ids == source
        src_vertices_mask = mesh_ids == target
        pts_source = meshes.verts[ref_vertices_mask].clone()
        pts_target = meshes.verts[src_vertices_mask].clone()
        iteration_folder = os.path.join(saved_folder, f"iteration_{i}")
        os.makedirs(iteration_folder, exist_ok=True)
        convert_pts_mesh(
            pts_source, os.path.join(iteration_folder, f"old_node_id_{source}.ply")
        )
        convert_pts_mesh(
            pts_target, os.path.join(iteration_folder, f"new_node_id_{target}.ply")
        )


def save_pgo_init_meshes_transformed(
    pose_node, pts_list, pts_color_list, saved_folder, name_list
):
    iteration_folder = os.path.join(saved_folder, "init_canonicalized")
    if os.path.exists(iteration_folder):
        shutil.rmtree(iteration_folder)
    os.makedirs(iteration_folder, exist_ok=True)

    for i, node in enumerate(pose_node.keys()):
        name = name_list[int(node)]
        # rotation = pose_node[node][:3, :3].cpu().numpy()  # Move rotation matrix to CPU and convert to numpy

        pts = (
            pts_list[int(node)].cpu().numpy()
        )  # Ensure points are on CPU and in numpy format
        pts_color = (
            pts_color_list[int(node)].cpu().numpy()
        )  # Ensure colors are on CPU and in numpy format

        # Normalize color values if necessary
        if np.max(pts_color) > 1.0:
            pts_color = pts_color / 255.0  # Assuming the color range is 0-255

        # Create Open3D point cloud object
        pts_point_cloud = o3d.geometry.PointCloud()
        pts_point_cloud.points = o3d.utility.Vector3dVector(pts)
        pts_point_cloud.colors = o3d.utility.Vector3dVector(pts_color)

        # Apply transformation
        # transformation = np.eye(4)
        # transformation[:3, :3] = rotation
        transformation = np.linalg.inv(pose_node[node].cpu().numpy())
        pts_point_cloud.transform(transformation)

        # Save the transformed point cloud with original colors
        file_path = os.path.join(
            iteration_folder, f"pose_node_{i}_name_{name}_id_{int(node)}.ply"
        )
        o3d.io.write_point_cloud(file_path, pts_point_cloud)

        print(f"Saved transformed point cloud: {file_path}")

        # transformed_pt_source = (rotation @ pts_source.T).T
        # iteration_folder = os.path.join(saved_folder, f'init_canonicalized')
        # os.makedirs(iteration_folder, exist_ok= True)
        # # convert_pts_mesh(pts_source, os.path.join(iteration_folder, f'node_{i}.ply' ))
        # convert_pts_mesh( transformed_pt_source, os.path.join(iteration_folder, f'name_{name}.ply' ))


class MeshVisualizer:
    def __init__(
        self, ref_mesh, src_mesh, ref_correspondence, src_correspondence, output_dir
    ):
        self.ref_mesh = ref_mesh
        self.src_mesh = src_mesh
        self.ref_correspondence = ref_correspondence
        self.src_correspondence = src_correspondence
        self.output_dir = output_dir

    def save(self):
        o3d.io.write_point_cloud(
            os.path.join(self.output_dir, "ref_mesh.ply"), self.ref_mesh
        )
        o3d.io.write_point_cloud(
            os.path.join(self.output_dir, "src_mesh.ply"), self.src_mesh
        )
        torch.save(
            self.ref_correspondence,
            os.path.join(self.output_dir, "ref_correspondence.pt"),
        )
        torch.save(
            self.src_correspondence,
            os.path.join(self.output_dir, "src_correspondence.pt"),
        )
