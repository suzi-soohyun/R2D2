import torch
import os
import numpy as np
import torch.nn.functional as F

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

# Sinkhorn iterations for optimal transport
def sinkhorn_iterations(Z, log_mu, log_nu, iters):
    # Perform Sinkhorn Normalization in log-space for numerical stability
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

# Optimal transport using Sinkhorn algorithm
def log_optimal_transport(scores, alpha, iters):
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # Adjust for normalization
    return Z

# Original code from Super-Glue
def extract_matches_with_sinkhorn(scores, threshold: float = 0.1):
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)

    #non_zero_elements = mutual0 & (mscores0 != 0)
    #print(mscores0)
    
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > threshold)
    valid1 = mutual1 & valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        
    num_matches = valid0.sum().item()  # Or you could use valid1.sum() as they should match
    return indices0, indices1, num_matches

def save_num_matches(scores, alpha, iters, threshold):
    num_matches_list = []
    matched_indices_list = []
    for iteration in range(iters):
        scores_ot = log_optimal_transport(scores, alpha, iteration + 1)
        match0, match1, num_matches = extract_matches_with_sinkhorn(scores_ot, threshold)
        num_matches_list.append(num_matches)
        if iteration % 10 == 0:  # Optional: print every 100 iterations
            print(f"Iteration {iteration}, Number of matches: {num_matches}")
            
        # Extract match indices
        i0 = match0[0]  # shape (N,)
        matches0 = torch.nonzero(i0 >= 0).squeeze(1)
        matched_indices = torch.stack([matches0, i0[matches0]], dim=1)
        matched_indices_list.append(matched_indices)
    return num_matches_list, matched_indices_list, scores_ot.squeeze(0)

def calculate_distance_matrix(root_path, category, sequence1, sequence2, feature_type):
    seq1_feats = torch.load(os.path.join(root_path, "mesh_feats", category, sequence1, feature_type, 'mesh_feats_mean.pt'))
    seq2_feats = torch.load(os.path.join(root_path, "mesh_feats", category, sequence2, feature_type, 'mesh_feats_mean.pt'))        

    len_seq1 = len(seq1_feats)
    len_seq2 = len(seq2_feats)
    distance_matrices = np.full((len_seq1, len_seq2), np.inf)
    if feature_type == "dino":
        for i in range(len_seq1):
            for j in range(len_seq2):
                if torch.isnan(seq1_feats[i]).any() or torch.isnan(seq2_feats[j]).any():
                    continue
                dist = torch.cdist(seq1_feats[i].unsqueeze(0), seq2_feats[j].unsqueeze(0), p=2).detach().cpu()
                distance_matrices[i, j] = dist
    elif feature_type == "sph":
        for i in range(len_seq1):
            for j in range(len_seq2):
                if torch.isnan(seq1_feats[i]).any() or torch.isnan(seq2_feats[j]).any():
                    continue
                cos_sim = F.cosine_similarity(seq1_feats[i].unsqueeze(0), seq2_feats[j].unsqueeze(0))
                cos_dist = 1 - cos_sim
                distance_matrices[i, j] = cos_dist
    del seq1_feats, seq2_feats
    return distance_matrices


def load_vertices(root_path, category, sequence):
    fpath = os.path.join(root_path, 'mesh/alpha500/meta_mask/meta', category, sequence,
                        'Partial_Ratio_100.0_Percent_start_frame_0_flip_sfm_False', 'mesh_vertex.pt')
    vtx = torch.load(fpath)
    positions = vtx[:, 0, :]
    colors = vtx[:, 1, :]
    return positions, colors

def ot_based_ransac(
    root_path,
    category,
    sequence1,
    sequence2,
    seq1_pose,
    seq2_pose,
):
    from od3d.datasets.ot.ransac_for_ot import run_ransac, decide_threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dino_feats_dist = calculate_distance_matrix(root_path, category, sequence1, sequence2, "dino")
    sph_feats_dist = calculate_distance_matrix(root_path, category, sequence1, sequence2, "sph")
    total_dist = dino_feats_dist + sph_feats_dist
    dist_matrix = torch.tensor(total_dist, dtype=torch.float32).to(device)
    scores = 1 - dist_matrix.unsqueeze(0)

    alpha = torch.tensor(1.0).to(device)
    iters = 50
    threshold = 0.001
    num_matches_list, matched_indices_list, _ = save_num_matches(scores, alpha, iters, threshold)

    match0, match1 = [], []
    for i, j in matched_indices_list[-1]:
        match0.append(seq1_pose[i.item()])
        match1.append(seq2_pose[j.item()])
    match0 = np.array(match0)
    match1 = np.array(match1)

    if match0.shape[0] < 4:
        raise RuntimeError("Not enough matches for transformation estimation.")

    # threshold = decide_threshold(match0, match1)
    best_inliers, best_T = run_ransac(match0, match1, threshold=0.2, minimal_correspondences=4, iter=2000)
    best_model = torch.from_numpy(best_T).float()  # shape: (4, 4)
    best_ref_correspondence = torch.tensor(match0[best_inliers]).float()
    best_src_correspondence = torch.tensor(match1[best_inliers]).float()
    best_score = torch.tensor(len(best_inliers)).float()

    return (
        best_model,
        best_ref_correspondence,
        best_src_correspondence,
        best_score,
        dist_matrix,
    )
