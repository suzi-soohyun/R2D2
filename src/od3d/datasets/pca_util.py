from typing import Optional, Union, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import os, sys
import torch.nn.functional as F

from od3d.cv.visual import mask

def pca(features: torch.Tensor, q: int, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply PCA to the features.
    
    Args:
        features (torch.Tensor): Input features of shape [N, C].
        q (int): Number of principal components to keep.
    
    Returns:
        tuple: Tuple containing:
            - offset (torch.Tensor): Mean of the features.
            - projection (torch.Tensor): PCA projection matrix of shape [C, C_out].
    """
    if q is None:
        q = features.shape[-1] // 2  # Use all components if q is not specified
    
    # center the data
    mean = features.mean(dim=0, keepdim=True)
    features = features - mean
    
    U, S, V = torch.pca_lowrank(features, q=q, center=False)
    
    if kwargs.get("plot", False):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(S.cpu().numpy(), marker='o')
        plt.title('Singular Values')
        plt.xlabel('Component Index')
        plt.ylabel('Singular Value')
        plt.grid()
        plt.savefig("pca_singular_values.png")
    
    if kwargs.get("C_out", None) is not None:
        C_out = kwargs["C_out"]
    elif kwargs.get("p", None) is not None:
        S_cumsum = (S **2).cumsum(dim=0) # cumulative sum of singular values [V1, V1 + V2, ..., V1 + ... + Vq]
        C_out = (S_cumsum <= kwargs["p"]).sum().item()  # number of components that keep at least p% of the variance
    else:
        C_out = q // 2
        
    if C_out > q:
        raise ValueError(f"C_out ({C_out}) must be less than or equal to q ({q}).")
    print(f"Reducing PCA components from {features.shape[-1]} to {C_out}.")
    precentage_information = (S[:C_out] ** 2).sum() / (S ** 2).sum()
    print(f"Percentage of information kept: {precentage_information:.2%}")

    projection =  V[:, :C_out]
    pca_features = torch.einsum("NC,CD->ND", features - mean, projection)
    
    return pca_features, mean, projection


def mask_features(dino_features: torch.Tensor, sph_features: torch.Tensor, mask: torch.Tensor):
    dino_feat_permuted = dino_features.permute(1, 2, 0)
    sph_feat_permuted = sph_features.permute(1, 2, 0)

    mask_permuted = mask.permute(1, 2, 0)
    mask_bool = mask_permuted.squeeze(-1) > 0.5
    
    masked_dino_features = dino_feat_permuted[mask_bool]
    masked_sph_features = sph_feat_permuted[mask_bool]
    mask_coords = mask_bool.nonzero(as_tuple=False)
    
    return masked_dino_features, masked_sph_features, mask_coords


def apply_mask_to_full_size(features: torch.Tensor, mask: torch.Tensor, image_size: Tuple[int, int] = (32, 32)):
    H, W = image_size
    full_image_features = torch.full((H, W, features.shape[1]), float('nan'), device=features.device)
    mask_permuted = mask.to(features.device).permute(1, 2, 0)
    mask_bool = mask_permuted.squeeze(-1) > 0.5
    
    full_image_features[mask_bool] = features
    return full_image_features.unsqueeze(0)



def visualize_features(dino_feat: torch.Tensor, sph_feat: torch.Tensor, mask: torch.Tensor, 
                       image_size: Tuple[int, int] = (32, 32), idx: int = 0, output_dir: str = None):

    dino_feat = dino_feat[:, :3]
    dino_features_norm = (dino_feat - dino_feat.min(dim=0).values) / (dino_feat.max(dim=0).values - dino_feat.min(dim=0).values + 1e-5)
    sph_features_norm = (sph_feat - sph_feat.min(dim=0).values) / (sph_feat.max(dim=0).values - sph_feat.min(dim=0).values + 1e-5)

    dino_img = torch.zeros((image_size[0], image_size[1], 3), dtype=torch.float32, device=dino_feat.device)
    sph_img = torch.zeros((image_size[0], image_size[1], 3), dtype=torch.float32, device=sph_feat.device)
    
    dino_img[mask[:, 0], mask[:, 1]] = dino_features_norm
    sph_img[mask[:, 0], mask[:, 1]] = sph_features_norm

    assert output_dir is not None, "Output directory must be specified."

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(dino_img.cpu().numpy())
    plt.title(f'DINO Features after PCA')

    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(sph_img.cpu().detach().numpy())
    plt.title(f'Spherical Features')

    plt.tight_layout(pad=4.0)
    plt.savefig(f"{output_dir}/masked_features_{idx}.png")