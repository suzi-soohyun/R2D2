import open3d as o3d
import numpy as np
from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord

def visualize_correspondences_matching_after_ot(
    seq1_pose, seq1_color,
    seq2_pose, seq2_color,
    match0, match1,
    image_path="correspondences_offscreen.png",
    image_width=1024,
    image_height=768
):

    # Shift second point cloud
    shift = np.array([10, 0, 0])
    seq2_pose_shifted = seq2_pose + shift
    match1_shifted = match1 + shift

    # Prepare geometries
    pcd_seq1 = o3d.geometry.PointCloud()
    pcd_seq1.points = o3d.utility.Vector3dVector(seq1_pose)
    pcd_seq1.colors = o3d.utility.Vector3dVector(seq1_color)

    pcd_seq2 = o3d.geometry.PointCloud()
    pcd_seq2.points = o3d.utility.Vector3dVector(seq2_pose_shifted)
    pcd_seq2.colors = o3d.utility.Vector3dVector(seq2_color)

    # Create line set for correspondences
    points = []
    lines = []
    for i in range(len(match0)):
        points.append(match0[i])
        points.append(match1_shifted[i])
        lines.append([2 * i, 2 * i + 1])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(lines))  # Green

    # Set up offscreen renderer
    renderer = OffscreenRenderer(image_width, image_height)
    material = MaterialRecord()
    material.shader = "defaultUnlit"

    renderer.scene.add_geometry("pcd1", pcd_seq1, material)
    renderer.scene.add_geometry("pcd2", pcd_seq2, material)
    renderer.scene.add_geometry("lines", line_set, material)

    # Camera setup: center and zoom out
    center = np.mean(np.vstack([seq1_pose, seq2_pose_shifted]), axis=0)
    renderer.setup_camera(60.0, center, center + [0, 0, 1], [0, 1, 0])

    # Render and save
    img = renderer.render_to_image()
    o3d.io.write_image(image_path, img)
    print(f"Saved correspondence visualization to: {image_path}")


def visualize_transformed_meshes_after_ot(seq1_pose, seq2_pose, best_T, save_path):
    seq1_color = np.ones((seq1_pose.shape[0], 3)) * np.array([1, 0, 0])
    seq2_color = np.ones((seq2_pose.shape[0], 3)) * np.array([0, 1, 0])

    seq2_pose_homogeneous = np.hstack((seq2_pose, np.ones((seq2_pose.shape[0], 1))))
    transformed_match1 = (seq2_pose_homogeneous @ best_T.T)[:, :-1]

    pcd_seq1 = o3d.geometry.PointCloud()
    pcd_seq1.points = o3d.utility.Vector3dVector(seq1_pose)
    pcd_seq1.colors = o3d.utility.Vector3dVector(seq1_color)

    pcd_seq2 = o3d.geometry.PointCloud()
    pcd_seq2.points = o3d.utility.Vector3dVector(transformed_match1)
    pcd_seq2.colors = o3d.utility.Vector3dVector(seq2_color)

    combined_pcd = pcd_seq1 + pcd_seq2
    o3d.io.write_point_cloud(save_path, combined_pcd)
    print(f"Combined point cloud saved to: {save_path}")
 

    
