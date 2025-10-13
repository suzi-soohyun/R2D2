import numpy as np
import pickle
import os, sys

def umeyama(src, dst, estimate_scale=True):
    """Estimate N-D similarity transformation with or without scaling.
    Taken from skimage!

    homo_src = np.hstack((src, np.ones((len(src), 1))))
    homo_dst = np.hstack((src, np.ones((len(dst), 1))))

    homo_dst = T @ homo_src, where T is the returned transformation

    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    """
    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.eye(4), 1
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    
    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    # Apply scale to rotation and compute translation accordingly
    T[:dim, :dim] *= scale
    T[:dim, dim] = dst_mean - T[:dim, :dim] @ src_mean

    return T, scale

# TODO: more iterations, run umeyama at the end with best_inliers
def run_ransac(match0, match1, threshold = 2, minimal_correspondences = 4, iter = 1000):
    print("threshold:", threshold)
    best_num_inliers = 0
    best_inliers = []
    best_T = None
    for i in range(iter):
        correspondence_subset = randomly_select_n_correspondences(match0, match1, minimal_correspondences)
        T_21, scale = umeyama(correspondence_subset[1], correspondence_subset[0])
        match1_homogeneous = np.hstack([match1, np.ones((match1.shape[0], 1))])
        transformed_mesh2 = match1_homogeneous @ T_21.T
        transformed_mesh2 = transformed_mesh2[:, :-1] 
        num_inliers = 0
        inliers = []
        for idx, (pt1, pt2) in enumerate(zip(match0, transformed_mesh2)):
            distance = np.linalg.norm(pt1 - pt2)
            if distance < threshold:
                inliers.append(idx)
                num_inliers += 1
        
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inliers = inliers
            best_T = T_21

    print("Before:", len(match0), ", After:", best_num_inliers) 
    if best_num_inliers < minimal_correspondences:
       return None, None
 
    return best_inliers, best_T


def randomly_select_n_correspondences(match0, match1, n):
    assert len(match0) == len(match1), "match0 and match1 must have the same length"
    
    indices = np.random.choice(len(match0), size=n, replace=False)
    match0_selected = match0[indices, :]
    match1_selected = match1[indices, :]
    return (match0_selected, match1_selected)


def decide_threshold(point_cloud1, point_cloud2):
    max_width1 = np.max(point_cloud1[:, 0]) - np.min(point_cloud1[:, 0])
    max_height1 = np.max(point_cloud1[:, 1]) - np.min(point_cloud1[:, 1])
    max_depth1 = np.max(point_cloud1[:, 2]) - np.min(point_cloud1[:, 2])
    print(f"Max Width1: {max_width1}")
    print(f"Max Height1: {max_height1}")
    print(f"Max Depth1: {max_depth1}")
    max_width2 = np.max(point_cloud2[:, 0]) - np.min(point_cloud2[:, 0])
    max_height2 = np.max(point_cloud2[:, 1]) - np.min(point_cloud2[:, 1])
    max_depth2 = np.max(point_cloud2[:, 2]) - np.min(point_cloud2[:, 2])
    print(f"Max Width2: {max_width2}")
    print(f"Max Height2: {max_height2}")
    print(f"Max Depth2: {max_depth2}")
    threshold = int(min(max_width1, max_height1, max_depth1, max_width2, max_height2, max_depth2)) / 2
    print(threshold)
    return threshold