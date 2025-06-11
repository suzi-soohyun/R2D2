import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d
import torch
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_DIR / "third_party" / "mast3r"))
from mast3r.fast_nn import fast_reciprocal_NNs


def _load_16bit_png_depth(depth_png):
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def read_from_depth_binary_array(depth_binary_path):
    with open(depth_binary_path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid,
            delimiter="&",
            max_rows=1,
            usecols=(0, 1, 2),
            dtype=int,
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    depth_map = array
    min_depth, max_depth = np.percentile(
        depth_map,
        [25, 85],
    )
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth
    return np.transpose(array, (1, 0, 2)).squeeze()


def backproject(depth, intrinsics, cam2world):
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth
    x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.stack((x, y, z), axis=-1)
    points_3d_flatten = points.reshape(-1, 3)
    points_3d_flatten = points_3d_flatten[points_3d_flatten[:, 2] > 0]

    ones = np.ones((points_3d_flatten.shape[0], 1))
    points_homogeneous = np.concatenate((points_3d_flatten, ones), axis=1)
    points_word_homogeneous = (cam2world @ points_homogeneous.T).T
    points_world = points_word_homogeneous[:, :3] / points_word_homogeneous[:, 3:]
    return points_world


def backproject_with_rgb(depth, rgb, intrinsics, cam2world):
    height, width = depth.shape

    # Generate grid of coordinates (u, v)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth  # Depth values

    # Compute x, y coordinates based on intrinsics
    x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]

    # Stack x, y, z to form the 3D points in camera space
    points = np.stack((x, y, z), axis=-1)
    points_3d_flatten = points.reshape(-1, 3)

    # Filter out points where depth is non-positive
    valid = points_3d_flatten[:, 2] > 0
    points_3d_flatten = points_3d_flatten[valid]
    rgb_flatten = rgb.reshape(-1, 3)[valid]

    # Convert to homogeneous coordinates
    ones = np.ones((points_3d_flatten.shape[0], 1))
    points_homogeneous = np.concatenate((points_3d_flatten, ones), axis=1)

    # Transform to world coordinates using the camera-to-world matrix
    points_world_homogeneous = (cam2world @ points_homogeneous.T).T
    points_world = points_world_homogeneous[:, :3] / points_world_homogeneous[:, 3:]

    # Combine points with their corresponding colors
    colored_points_world = np.concatenate((points_world, rgb_flatten), axis=1)
    return colored_points_world


def visualize_tensor(feature_tensor, name):
    # Reshape: Combine spatial dimensions into one dimension, result is [768, 1024]
    reshaped_features = feature_tensor.reshape(
        768, -1
    ).T  # Transpose to get [1024, 768]

    # PCA: Reduce dimensions to 3 for RGB visualization
    pca = PCA(n_components=3)
    reduced_features = pca.fit_transform(reshaped_features)
    # Normalize for RGB
    colors = (reduced_features - np.min(reduced_features, axis=0)) / (
        np.max(reduced_features, axis=0) - np.min(reduced_features, axis=0)
    )
    # Reshape back to 2D spatial structure [32, 32, 3]
    image_rgb = colors.reshape(feature_tensor.shape[-2], feature_tensor.shape[-1], 3)

    # Plotting
    plt.imshow(image_rgb)
    plt.title("Visualized Feature Tensor as RGB")
    plt.axis("off")
    plt.show()
    plt.savefig(name)


def backproject_with_feat(
    depth,
    feat,
    intrinsics,
    cam2world,
    mesh_vertices,
    meshes_verts_aggregated_features_test,
):
    height, width = depth.shape

    # Generate grid of coordinates (u, v)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth  # Depth values

    # Compute x, y coordinates based on intrinsics
    x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]

    feat_resized = torch.nn.functional.interpolate(
        feat.unsqueeze(0), (height, width), mode="nearest"
    )
    feat_resized = feat_resized.detach().cpu().numpy()
    feat = feat.detach().cpu().numpy()
    # visualize_tensor(feat, 'raw_feat_map.png')
    # visualize_tensor(feat_resized, 'resized_feated_map.png')
    # Stack x, y, z to form the 3D points in camera space
    points = np.stack((x, y, z), axis=-1)
    points_3d_flatten = points.reshape(-1, 3)

    # Filter out points where depth is non-positive
    valid = points_3d_flatten[:, 2] > 0
    points_3d_flatten = points_3d_flatten[valid]
    if len(points_3d_flatten) != 0:
        feat_flatten = np.transpose(feat_resized.reshape(feat.shape[0], -1))[valid]

        # Convert to homogeneous coordinates
        ones = np.ones((points_3d_flatten.shape[0], 1))
        points_homogeneous = np.concatenate((points_3d_flatten, ones), axis=1)

        # Transform to world coordinates using the camera-to-world matrix
        points_world_homogeneous = (cam2world @ points_homogeneous.T).T
        points_world = points_world_homogeneous[:, :3] / points_world_homogeneous[:, 3:]

        # Combine points with their corresponding colors
        feated_points_world = np.concatenate((points_world, feat_flatten), axis=1)
        print("feated_points_world.shape[0] is ", feated_points_world.shape[0])
        num_samples = min(feated_points_world.shape[0], 400)
        indices = np.random.choice(
            feated_points_world.shape[0], num_samples, replace=False
        )
        selected_points_world = points_world[indices]
        selected_feats = feat_flatten[indices]

        tree = cKDTree(mesh_vertices)
        distances, indices = tree.query(selected_points_world)
        closest_vertices = mesh_vertices[indices]
        # np.save('feated_points_world.npy', feated_points_world[indices])
        for i in range(indices.shape[0]):
            indice = indices[i]
            feature = torch.tensor(selected_feats[i]).unsqueeze(0)
            meshes_verts_aggregated_features_test[indice] = torch.cat(
                [
                    feature,
                    meshes_verts_aggregated_features_test[indice].detach().cpu(),
                ],
                dim=0,
            )


# Save the points as a .ply file
def save_as_ply(points, filename):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    open3d.io.write_point_cloud(filename, pcd)


def read_from_depth_binary_array(depth_binary_path):
    with open(depth_binary_path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid,
            delimiter="&",
            max_rows=1,
            usecols=(0, 1, 2),
            dtype=int,
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    depth_map = array
    min_depth, max_depth = np.percentile(
        depth_map,
        [5, 95],
    )
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth
    return np.transpose(array, (1, 0, 2)).squeeze()


def tell_left_from_right_sd_feature(
    ref_sd_feature,
    src_sd_feature,
    src_sd_feature_flipped,
    mask_ref,
    mask_src,
    mask_src_flipped,
    device="cuda",
):
    matches_im0, matches_im1 = fast_reciprocal_NNs(
        ref_sd_feature,
        src_sd_feature,
        subsample_or_initxy1=8,
        device=device,
        dist="dot",
        block_size=2**13,
    )
