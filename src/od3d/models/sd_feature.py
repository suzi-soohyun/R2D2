import itertools
from contextlib import ExitStack

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.config import instantiate
from detectron2.config import LazyCall as L
from detectron2.data import detection_utils as utils
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.evaluation import inference_context
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import random_color
from detectron2.utils.visualizer import Visualizer
from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
from odise import model_zoo
from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise
from odise.data import get_openseg_labels
from odise.modeling.wrapper import OpenPanopticInference
from PIL import Image

COCO_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 1
]
COCO_THING_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 1]
COCO_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 0
]
COCO_STUFF_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 0]

ADE_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 1
]
ADE_THING_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 1]
ADE_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 0
]
ADE_STUFF_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 0]

LVIS_CLASSES = get_openseg_labels("lvis_1203", True)
# use beautiful coco colors
LVIS_COLORS = list(
    itertools.islice(
        itertools.cycle([c["color"] for c in COCO_CATEGORIES]), len(LVIS_CLASSES)
    ),
)


class StableDiffusionSeg:
    def __init__(self, model, metadata, aug, instance_mode=ColorMode.IMAGE):
        """
        Args:
            model (nn.Module):
            metadata (MetadataCatalog): image metadata.
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.model = model
        self.metadata = metadata
        self.aug = aug
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

    def get_features(self, original_image, caption=None, pca=None):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            features (dict):
                the output of the model for one image only.
        """
        height, width = original_image.shape[:2]
        aug_input = T.AugInput(original_image, sem_seg=None)
        self.aug(aug_input)
        image = aug_input.image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        if caption is not None:
            features = self.model.get_features([inputs], caption, pca=pca)
        else:
            features = self.model.get_features([inputs], pca=pca)
        return features

    def predict(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        height, width = original_image.shape[:2]
        aug_input = T.AugInput(original_image, sem_seg=None)
        self.aug(aug_input)
        image = aug_input.image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        predictions = self.model([inputs])[0]
        return predictions


def build_demo_classes_and_metadata(vocab, label_list):
    extra_classes = []

    if vocab:
        for words in vocab.split(";"):
            extra_classes.append([word.strip() for word in words.split(",")])
    extra_colors = [
        random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))
    ]

    demo_thing_classes = extra_classes
    demo_stuff_classes = []
    demo_thing_colors = extra_colors
    demo_stuff_colors = []

    if "COCO" in label_list:
        demo_thing_classes += COCO_THING_CLASSES
        demo_stuff_classes += COCO_STUFF_CLASSES
        demo_thing_colors += COCO_THING_COLORS
        demo_stuff_colors += COCO_STUFF_COLORS
    if "ADE" in label_list:
        demo_thing_classes += ADE_THING_CLASSES
        demo_stuff_classes += ADE_STUFF_CLASSES
        demo_thing_colors += ADE_THING_COLORS
        demo_stuff_colors += ADE_STUFF_COLORS
    if "LVIS" in label_list:
        demo_thing_classes += LVIS_CLASSES
        demo_thing_colors += LVIS_COLORS

    MetadataCatalog.pop("odise_demo_metadata", None)
    demo_metadata = MetadataCatalog.get("odise_demo_metadata")
    demo_metadata.thing_classes = [c[0] for c in demo_thing_classes]
    demo_metadata.stuff_classes = [
        *demo_metadata.thing_classes,
        *[c[0] for c in demo_stuff_classes],
    ]
    demo_metadata.thing_colors = demo_thing_colors
    demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
    demo_metadata.stuff_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.stuff_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.thing_classes))
    }

    demo_classes = demo_thing_classes + demo_stuff_classes

    return demo_classes, demo_metadata


import sys


def load_model(
    config_path="Panoptic/odise_label_coco_50e.py",
    seed=42,
    diffusion_ver="v1-3",
    image_size=1024,
    num_timesteps=0,
    block_indices=(2, 5, 8, 11),
    decoder_only=True,
    encoder_only=False,
    resblock_only=False,
):
    cfg = model_zoo.get_config(config_path, trained=True)

    cfg.model.backbone.feature_extractor.init_checkpoint = "sd://" + diffusion_ver
    cfg.model.backbone.feature_extractor.steps = (num_timesteps,)
    cfg.model.backbone.feature_extractor.unet_block_indices = block_indices
    cfg.model.backbone.feature_extractor.encoder_only = encoder_only
    cfg.model.backbone.feature_extractor.decoder_only = decoder_only
    cfg.model.backbone.feature_extractor.resblock_only = resblock_only
    cfg.model.overlap_threshold = 0
    seed_all_rng(seed)

    cfg.dataloader.test.mapper.augmentations = [
        L(T.ResizeShortestEdge)(
            short_edge_length=image_size, sample_style="choice", max_size=2560
        ),
    ]
    dataset_cfg = cfg.dataloader.test

    aug = instantiate(dataset_cfg.mapper).augmentations

    model = instantiate_odise(cfg.model)
    model.to(cfg.train.device)
    ODISECheckpointer(model).load(cfg.train.init_checkpoint)

    return model, aug


def inference(model, aug, image, vocab, label_list):
    demo_classes, demo_metadata = build_demo_classes_and_metadata(vocab, label_list)
    with ExitStack() as stack:
        inference_model = OpenPanopticInference(
            model=model,
            labels=demo_classes,
            metadata=demo_metadata,
            semantic_on=False,
            instance_on=False,
            panoptic_on=True,
        )
        stack.enter_context(inference_context(inference_model))
        stack.enter_context(torch.no_grad())

        demo = StableDiffusionSeg(inference_model, demo_metadata, aug)
        pred = demo.predict(np.array(image))
        return (pred, demo_classes)


def get_features(model, aug, image, vocab, label_list, caption=None, pca=False):
    demo_classes, demo_metadata = build_demo_classes_and_metadata(vocab, label_list)
    with ExitStack() as stack:
        inference_model = OpenPanopticInference(
            model=model,
            labels=demo_classes,
            metadata=demo_metadata,
            semantic_on=False,
            instance_on=False,
            panoptic_on=True,
        )
        stack.enter_context(inference_context(inference_model))
        stack.enter_context(torch.no_grad())

        demo = StableDiffusionSeg(inference_model, demo_metadata, aug)
        if caption is not None:
            features = demo.get_features(np.array(image), caption, pca=pca)
        else:
            features = demo.get_features(np.array(image), pca=pca)
        return features


def pca_process(features):
    # Get the feature tensors
    size_s5 = features["s5"].shape[-1]
    size_s4 = features["s4"].shape[-1]
    size_s3 = features["s3"].shape[-1]

    s5 = features["s5"].reshape(features["s5"].shape[0], features["s5"].shape[1], -1)
    s4 = features["s4"].reshape(features["s4"].shape[0], features["s4"].shape[1], -1)
    s3 = features["s3"].reshape(features["s3"].shape[0], features["s3"].shape[1], -1)

    # Define the target dimensions
    target_dims = {"s5": 128, "s4": 128, "s3": 128}

    # Apply PCA to each tensor using Faiss CPU
    for name, tensor in zip(["s5", "s4", "s3"], [s5, s4, s3]):
        target_dim = target_dims[name]

        # Transpose the tensor so that the last dimension is the number of features
        tensor = tensor.permute(0, 2, 1)

        # # Norm the tensor
        # tensor = tensor / tensor.norm(dim=-1, keepdim=True)

        # Initialize a Faiss PCA object
        pca = faiss.PCAMatrix(tensor.shape[-1], target_dim)

        # Train the PCA object
        pca.train(tensor[0].cpu().numpy())

        # Apply PCA to the data
        transformed_tensor_np = pca.apply(tensor[0].cpu().numpy())

        # Convert the transformed data back to a tensor
        transformed_tensor = torch.tensor(
            transformed_tensor_np, device=tensor.device
        ).unsqueeze(0)

        # Store the transformed tensor in the features dictionary
        features[name] = transformed_tensor

    # Reshape the tensors back to their original shapes
    features["s5"] = (
        features["s5"]
        .permute(0, 2, 1)
        .reshape(features["s5"].shape[0], -1, size_s5, size_s5)
    )
    features["s4"] = (
        features["s4"]
        .permute(0, 2, 1)
        .reshape(features["s4"].shape[0], -1, size_s4, size_s4)
    )
    features["s3"] = (
        features["s3"]
        .permute(0, 2, 1)
        .reshape(features["s3"].shape[0], -1, size_s3, size_s3)
    )
    # Upsample s5 spatially by a factor of 2
    upsampled_s5 = torch.nn.functional.interpolate(
        features["s5"], scale_factor=2, mode="bilinear", align_corners=False
    )

    # Concatenate upsampled_s5 and s4 to create a new s5
    features["s5"] = torch.cat((upsampled_s5, features["s4"]), dim=1)

    # Set s3 as the new s4
    features["s4"] = features["s3"]

    # Remove s3 from the features dictionary
    del features["s3"]

    return features


def process_features_and_mask(
    model, aug, image, category=None, input_text=None, mask=True, pca=False, raw=False
):
    input_image = image
    caption = input_text
    vocab = ""
    label_list = ["COCO"]
    category_convert_dict = {
        "aeroplane": "airplane",
        "motorbike": "motorcycle",
        "pottedplant": "potted plant",
        "tvmonitor": "tv",
    }
    if type(category) is not list and category in category_convert_dict:
        category = category_convert_dict[category]
    elif type(category) is list:
        category = [
            category_convert_dict[cat] if cat in category_convert_dict else cat
            for cat in category
        ]
    features = get_features(
        model, aug, input_image, vocab, label_list, caption, pca=(pca or raw)
    )
    if pca:
        features = pca_process(features)
    if raw:
        return features
    features_gether_s4_s5 = torch.cat(
        [
            features["s4"],
            F.interpolate(
                features["s5"], size=(features["s4"].shape[-2:]), mode="bilinear"
            ),
        ],
        dim=1,
    )

    if mask:
        (pred, classes) = inference(model, aug, input_image, vocab, label_list)
        seg_map = pred["panoptic_seg"][0]
        target_mask_id = []
        for item in pred["panoptic_seg"][1]:
            item["category_name"] = classes[item["category_id"]]
            if category in item["category_name"]:
                target_mask_id.append(item["id"])
        resized_seg_map_s4 = F.interpolate(
            seg_map.unsqueeze(0).unsqueeze(0).float(),
            size=(features["s4"].shape[-2:]),
            mode="nearest",
        )
        # to do adjust size
        binary_seg_map = torch.zeros_like(resized_seg_map_s4)
        for i in target_mask_id:
            binary_seg_map += (resized_seg_map_s4 == i).float()
        if len(target_mask_id) == 0 or binary_seg_map.sum() < 6:
            binary_seg_map = torch.ones_like(resized_seg_map_s4)
        features_gether_s4_s5 = features_gether_s4_s5 * binary_seg_map
        # set where mask is 0 to inf
        features_gether_s4_s5[
            (binary_seg_map == 0).repeat(1, features_gether_s4_s5.shape[1], 1, 1)
        ] = -1

    return features_gether_s4_s5


def get_mask(model, aug, image, category=None, input_text=None):
    model.backbone.feature_extractor.decoder_only = False
    model.backbone.feature_extractor.encoder_only = False
    model.backbone.feature_extractor.resblock_only = False
    input_image = image
    caption = input_text
    vocab = ""
    label_list = ["COCO"]
    category_convert_dict = {
        "aeroplane": "airplane",
        "motorbike": "motorcycle",
        "pottedplant": "potted plant",
        "tvmonitor": "tv",
    }
    if type(category) is not list and category in category_convert_dict:
        category = category_convert_dict[category]
    elif type(category) is list:
        category = [
            category_convert_dict[cat] if cat in category_convert_dict else cat
            for cat in category
        ]

    (pred, classes) = inference(model, aug, input_image, vocab, label_list)
    seg_map = pred["panoptic_seg"][0]
    target_mask_id = []
    for item in pred["panoptic_seg"][1]:
        item["category_name"] = classes[item["category_id"]]
        if type(category) is list:
            for cat in category:
                if cat in item["category_name"]:
                    target_mask_id.append(item["id"])
        else:
            if category in item["category_name"]:
                target_mask_id.append(item["id"])
    resized_seg_map_s4 = seg_map.float()
    binary_seg_map = torch.zeros_like(resized_seg_map_s4)
    for i in target_mask_id:
        binary_seg_map += (resized_seg_map_s4 == i).float()
    if len(target_mask_id) == 0 or binary_seg_map.sum() < 6:
        binary_seg_map = torch.ones_like(resized_seg_map_s4)

    return binary_seg_map


def co_pca(features1, dim=[128, 128, 128]):
    processed_features1 = {}
    s5_size = features1["s5"].shape[-1]
    s4_size = features1["s4"].shape[-1]
    s3_size = features1["s3"].shape[-1]
    # Get the feature tensors
    s5_1 = features1["s5"].reshape(
        features1["s5"].shape[0], features1["s5"].shape[1], -1
    )
    s4_1 = features1["s4"].reshape(
        features1["s4"].shape[0], features1["s4"].shape[1], -1
    )
    s3_1 = features1["s3"].reshape(
        features1["s3"].shape[0], features1["s3"].shape[1], -1
    )
    # Define the target dimensions
    target_dims = {"s5": dim[0], "s4": dim[1], "s3": dim[2]}

    # Compute the PCA
    for name, tensors in zip(
        ["s5", "s4", "s3"], [[s5_1, s5_1], [s4_1, s4_1], [s3_1, s3_1]]
    ):
        target_dim = target_dims[name]

        # Concatenate the features
        features = torch.cat(tensors, dim=-1)  # along the spatial dimension
        features = features.permute(0, 2, 1)  # Bx(t_x+t_y)x(d)
        mean = torch.mean(features[0], dim=0, keepdim=True)
        centered_features = features[0] - mean
        U, S, V = torch.pca_lowrank(centered_features, q=target_dim)
        reduced_features = torch.matmul(
            centered_features, V[:, :target_dim]
        )  # (t_x+t_y)x(d)
        features = reduced_features.unsqueeze(0).permute(0, 2, 1)  # Bx(d)x(t_x+t_y)

        # Split the features
        processed_features1[name] = features[
            :, :, : features.shape[-1] // 2
        ]  # Bx(d)x(t_x)

    # reshape the features
    processed_features1["s5"] = processed_features1["s5"].reshape(
        processed_features1["s5"].shape[0], -1, s5_size, s5_size
    )
    processed_features1["s4"] = processed_features1["s4"].reshape(
        processed_features1["s4"].shape[0], -1, s4_size, s4_size
    )
    processed_features1["s3"] = processed_features1["s3"].reshape(
        processed_features1["s3"].shape[0], -1, s3_size, s3_size
    )

    # Upsample s5 spatially by a factor of 2
    processed_features1["s5"] = F.interpolate(
        processed_features1["s5"],
        size=(processed_features1["s4"].shape[-2:]),
        mode="bilinear",
        align_corners=False,
    )

    # Concatenate upsampled_s5 and s4 to create a new s5
    processed_features1["s5"] = torch.cat(
        [processed_features1["s4"], processed_features1["s5"]], dim=1
    )

    # Set s3 as the new s4
    processed_features1["s4"] = processed_features1["s3"]

    # Remove s3 from the features dictionary
    processed_features1.pop("s3")

    # current order are layer 8, 5, 2
    features1_gether_s4_s5 = torch.cat(
        [
            processed_features1["s4"],
            F.interpolate(
                processed_features1["s5"],
                size=(processed_features1["s4"].shape[-2:]),
                mode="bilinear",
            ),
        ],
        dim=1,
    )
    return features1_gether_s4_s5


if __name__ == "__main__":
    from mast3r.mast3r.fast_nn import fast_reciprocal_NNs
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    img_1 = np.array(
        Image.open(
            "/CT/3D_DST_Scene/work/od3d/datasets/CO3D_V2_Preprocess/sfm/colmap50/hydrant/167_18177_33907/Partial_Ratio_25.0_Percent_start_frame_0_flip_sfm_False/images/frame000003.jpg"
        ).resize((960, 960), Image.ANTIALIAS)
    )
    img_2 = np.array(
        Image.open(
            "/CT/3D_DST_Scene/work/od3d/datasets/CO3D_V2_Preprocess/sfm/colmap50/hydrant/167_18177_33907/Partial_Ratio_25.0_Percent_start_frame_0_flip_sfm_False/images/frame000003.jpg"
        ).resize((960, 960), Image.ANTIALIAS)
    )

    # mask_1 = np.array(Image.open('/CT/CO3D-dataset/nobackup/CO3D_V2/hydrant/167_18177_33907/masks/frame000001.png').resize((960, 960), Image.NEAREST))
    # mask_2 = np.array(Image.open('/CT/CO3D-dataset/nobackup/CO3D_V2/hydrant/167_18177_33907/masks/frame000005.png').resize((960, 960), Image.NEAREST))

    # mask_1[mask_1 > 0] =1
    # mask_2[mask_2 > 0] =1

    # img_1 = img_1 * np.expand_dims(mask_1, axis=-1)
    # img_2 = img_2 * np.expand_dims(mask_2, axis=-1)

    sd_model, sd_aug = load_model(
        diffusion_ver="v1-3",
        image_size=960,
        num_timesteps=100,
        block_indices=[2, 5, 8, 11],
    )
    features1 = process_features_and_mask(
        sd_model, sd_aug, img_1, input_text=None, mask=False, raw=True
    )
    features1 = co_pca(features1, [256, 256, 256])
    features1 = features1 / features1.norm(dim=1)
    torch.save(features1, "features1.pt")
    # features1 = torch.load('features1.pt')
    # #features1 = torch.nn.functional.interpolate(features1, (960,960), mode = 'nearest')
    # # features1 = features1.view(768, -1).permute(1,0).detach().cpu().numpy()
    pca = PCA(n_components=3)
    reduced_data_1 = pca.fit_transform(features1)
    reduced_image_1 = reduced_data_1.reshape(60, 60, 3)
    # Normalize the reduced image to be between 0 and 1
    normalized_image_1 = (reduced_image_1 - reduced_image_1.min()) / (
        reduced_image_1.max() - reduced_image_1.min()
    )

    features2 = process_features_and_mask(
        sd_model, sd_aug, img_2, input_text=None, mask=False, raw=True
    )
    features2 = co_pca(features2, [256, 256, 256])
    features2 = features2 / features2.norm(dim=1)
    torch.save(features2, "features2.pt")
    # features2 = torch.load('features2.pt')
    # load fearure

    W1, H1 = img_1.shape[1], img_1.shape[0]

    mask_1 = np.array(
        Image.open(
            "/CT/CO3D-dataset/nobackup/CO3D_V2/hydrant/167_18177_33907/masks/frame000003.png"
        ).resize((int(W1 / 8), int(H1 / 8)))
    )
    mask_2 = np.array(
        Image.open(
            "/CT/CO3D-dataset/nobackup/CO3D_V2/hydrant/167_18177_33907/masks/frame000003.png"
        ).resize((int(W1 / 8), int(H1 / 8)))
    )

    mask_1[mask_1 > 0] = 1
    mask_2[mask_2 > 0] = 1

    features1_resized = torch.nn.functional.interpolate(
        features1, size=(int(H1 / 8), int(W1 / 8)), mode="bilinear"
    )
    features2_resized = torch.nn.functional.interpolate(
        features2, size=(int(H1 / 8), int(W1 / 8)), mode="bilinear"
    )

    features1_resized = features1_resized.permute(0, 2, 3, 1).squeeze(0)
    features2_resized = features2_resized.permute(0, 2, 3, 1).squeeze(0)

    matches_im0, matches_im1 = fast_reciprocal_NNs(
        features1_resized,
        features2_resized,
        subsample_or_initxy1=8,
        device="cuda",
        dist="dot",
        block_size=2**13,
    )
    plt.figure()

    img_1 = np.array(
        Image.open(
            "/CT/3D_DST_Scene/work/od3d/datasets/CO3D_V2_Preprocess/sfm/colmap50/hydrant/167_18177_33907/Partial_Ratio_25.0_Percent_start_frame_0_flip_sfm_False/images/frame000003.jpg"
        ).resize((int(W1 / 8), int(H1 / 8)), Image.ANTIALIAS)
    )
    img_2 = np.array(
        Image.open(
            "/CT/3D_DST_Scene/work/od3d/datasets/CO3D_V2_Preprocess/sfm/colmap50/hydrant/167_18177_33907/Partial_Ratio_25.0_Percent_start_frame_0_flip_sfm_False/images/frame000003.jpg"
        ).resize((int(W1 / 8), int(H1 / 8)), Image.ANTIALIAS)
    )

    plt.imshow(np.concatenate([np.array(img_1), np.array(img_2)], axis=1))
    for i in range(len(matches_im0)):
        x0, y0 = matches_im0[i]
        x1, y1 = matches_im1[i]
        plt.plot([int(y0), int(y1) + int(W1 / 8)], [int(x0), int(x1)], "-+")
    plt.savefig("match.png")
    # binary_1 = mask_1[matches_im0[:,1], matches_im0[:,0]] > 1
    # binary_2 = mask_2[matches_im1[:,1], matches_im1[:,0]] > 1

    # matches_im0 = matches_im0[binary_1 & binary_2]
    # matches_im1 = matches_im1[binary_1 & binary_2]

    print("done")
    # # try to get the nearest neighbor of feature in the feature space
    # features1_resized = features1_resized.view(768, -1)  # C, L1
    # features2_resized = features2_resized.view(768, -1)  # C, L2

    # features1_resized_expanded = features1_resized.unsqueeze(2)
    # features2_resized_expanded = features2_resized.unsqueeze(1)

    # result = features1_resized_expanded - features2_resized_expanded

    # result_dist = torch.norm(result, dim=0) # L1, L2
    # result_dist = result_dist.view(1, -1)
    # lowest_values, indices = torch.topk(result_dist, 10, largest=False) # top 10 lowest value

    # print('done')

    # plt.figure()
    # img_2 = img_2.resize((int(W1/32), int(H1/32)))
    # plt.imshow(np.concatenate([np.array(img_1), np.array(img_2)], axis =1))
    # L1 = features1_resized.shape[1]
    # L2 = features2_resized.shape[1]
    # indices = indices[0]
    # for i in range(len(indices)):
    #     y1 = int(indices[i] / L2)
    #     y1_h1 = int(y1/(int(W1/32)))
    #     y1_w1 = y1 - y1_h1 * (int(W1/32))
    #     x1 = indices[i] - y1 * L2
    #     x1_h1 = int(x1/(int(W1/32)))
    #     x1_w1 = x1 - x1_h1 * (int(W1/32))
    #     #plt.plot([y1_w1, int(x1_w1) + int(W1/8)], [y1_h1, x1_h1], 'r-')
    #     plt.plot([int(y1_w1), int(x1_w1) + int(W1/32)], [int(y1_h1), int(x1_h1)],'-+' )
    # plt.savefig('match.png')
