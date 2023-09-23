import os
import pickle as pkl
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open_clip
import torch
import tyro
from gradslam.datasets import (
    ICLDataset,
    ReplicaDataset,
    ScannetDataset,
    load_dataset_config,
)
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.pointclouds import Pointclouds
from gradslam.structures.rgbdimages import RGBDImages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from tqdm import tqdm, trange
from typing_extensions import Literal

# Check if an os environment variable is set for fastsam path
FASTSAM_PYTHON_PATH = None
if not os.environ.get("FASTSAM_PYTHON_PATH"):
    warnings.warn(
        "FASTSAM_PYTHON_PATH not set. Please set it to the path of FastSAM repo."
    )
FASTSAM_PYTHON_PATH = "/home/krishna/code/FastSAM"
sys.path.append(FASTSAM_PYTHON_PATH)
from fastsam import FastSAM, FastSAMPrompt


@dataclass
class ProgramArgs:
    # Torch device to run computation on (E.g., "cpu")
    device: str = "cuda"

    # SAM checkpoint and model params
    sam_model_path: Union[str, Path] = (
        Path(FASTSAM_PYTHON_PATH) / "weights" / "FastSAM-x.pt"
    )
    # Resize image so that the largest side is this sam_imgsz pixels
    sam_imgsz: int = 512
    # IoU threshold (TODO: elaborate)
    sam_iou: float = 0.9
    # Mask confidence threshold
    sam_conf: float = 0.4
    # Device to run FastSAM on
    sam_device: str = "cuda:0"
    # Draw high-res segmentation masks
    sam_retina: bool = True
    # Draw contours around masks
    sam_with_contours: bool = True
    # Better quality using morphologyEx
    sam_better_quality: bool = False
    # Save mask visualizations
    sam_save_viz: bool = True

    # gradslam mode ("incremental" vs "batch")
    mode: Literal["incremental", "batch"] = "incremental"

    # Path to the data config (.yaml) file
    dataconfig_path: str = "dataconfigs/replica.yaml"
    # Path to the dataset directory
    data_dir: Union[str, Path] = Path.home() / "data" / "nice-slam-data" / "Replica"
    # Sequence from the dataset to load
    sequence: str = "room0"
    # Start frame index
    start_idx: int = 0
    # End frame index
    end_idx: int = -1
    # Stride (number of frames to skip between successive fusion steps)
    stride: int = 100
    # Desired image width and height
    desired_height: int = 240
    desired_width: int = 320

    # CLIP model config
    open_clip_model = "ViT-H-14"
    open_clip_pretrained_dataset = "laion2b_s32b_b79k"

    # Directory to save extracted features
    save_dir: str = "saved-feat"


def get_dataset(dataconfig_path, basedir, sequence, **kwargs):
    config_dict = load_dataset_config(dataconfig_path)
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def crop_bbox_from_img(img, _fastsam_everything_results, idx=0, no_bg=False):
    bbox_xyxy = _fastsam_everything_results[0].boxes.xyxy[idx]
    _img = img
    if no_bg:
        _mask = _fastsam_everything_results[0].masks.data[idx]
        _img = _img * _mask[..., None].astype(np.float32)
        _img = _img.astype(np.uint8)
    # Check if bbox is torch tensor or numpy array or list
    if isinstance(bbox_xyxy, torch.Tensor):
        x1, y1, x2, y2 = [
            int(_item) for _item in bbox_xyxy.detach().cpu().numpy().tolist()
        ]
    elif isinstance(bbox_xyxy, np.ndarray):
        x1, y1, x2, y2 = [int(_item) for _item in bbox_xyxy.tolist()]
    elif isinstance(bbox_xyxy, list):
        x1, y1, x2, y2 = [int(_item) for _item in bbox_xyxy]
    else:
        raise TypeError(f"Unknown type for bbox_xyxy: {type(bbox_xyxy)}")
    return _img[y1:y2, x1:x2]


def batch_crop_bbox_from_img(img, _fastsam_everything_results, no_bg=False):
    num_boxes = _fastsam_everything_results[0].boxes.xyxy.shape[0]
    cropped_imgs = []

    for idx in range(num_boxes):
        cropped_img = crop_bbox_from_img(
            img, _fastsam_everything_results, idx=idx, no_bg=no_bg
        )
        cropped_imgs.append(cropped_img)
    return cropped_imgs


def main():
    torch.autograd.set_grad_enabled(False)

    args = tyro.cli(ProgramArgs)

    # dataconfig = load_dataset_config(args.dataconfig_path)
    dataset = get_dataset(
        dataconfig_path=args.dataconfig_path,
        basedir=args.data_dir,
        sequence=args.sequence,
        start=args.start_idx,
        end=args.end_idx,
        stride=args.stride,
        desired_height=args.desired_height,
        desired_width=args.desired_width,
    )

    # Load FastSAM model
    print("Loading FastSAM model...")
    fastsam_model = FastSAM(args.sam_model_path)

    print(
        f"Initializing OpenCLIP model: {args.open_clip_model}"
        f" pre-trained on {args.open_clip_pretrained_dataset}..."
    )
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        args.open_clip_model, args.open_clip_pretrained_dataset
    )
    clip_model.cuda()
    clip_model.half()
    clip_model.eval()

    os.makedirs(args.save_dir, exist_ok=True)

    print("Compute pixel-aligned 2D features...")
    for idx in trange(len(dataset)):
        imgfile = dataset.color_paths[idx]
        img = cv2.imread(imgfile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sam_masks = fastsam_model(
            imgfile,
            device=args.sam_device,
            retina_masks=args.sam_retina,
            imgsz=args.sam_imgsz,
            conf=args.sam_conf,
            iou=args.sam_iou,
        )
        cropped_imgs = batch_crop_bbox_from_img(img, sam_masks, no_bg=False)

        # global CLIP feature
        preprocessed_full_img = clip_preprocess(Image.fromarray(img))
        global_feat = clip_model.encode_image(
            preprocessed_full_img[None, ...].half().cuda()
        )
        feat_dim = global_feat.shape[-1]
        global_feat = torch.nn.functional.normalize(global_feat, dim=-1)

        # Convert to PIL and preprocess
        preprocessed_images = [
            clip_preprocess(Image.fromarray(img_))[None, ...].half().to(args.device)
            for img_ in cropped_imgs
        ]
        batch_preprocessed = torch.cat(preprocessed_images, axis=0)
        local_feat = clip_model.encode_image(batch_preprocessed.half())
        local_feat = torch.nn.functional.normalize(local_feat, dim=-1)

        similarity_scores = []
        semiglobal_feat = []
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        for local_feat_idx in range(local_feat.shape[0]):
            similarity_scores.append(
                cosine_similarity(global_feat, local_feat[local_feat_idx])
            )
        softmax_scores = torch.nn.functional.softmax(
            torch.cat(similarity_scores), dim=0
        )
        for local_feat_idx in range(local_feat.shape[0]):
            w = softmax_scores[local_feat_idx][None, ..., None]
            wfeat = w * global_feat + (1 - w) * local_feat[local_feat_idx]
            semiglobal_feat.append(torch.nn.functional.normalize(wfeat, dim=-1))

        prompt_process = FastSAMPrompt(imgfile, sam_masks, device=args.sam_device)
        masks = prompt_process.everything_prompt()
        # masks is N, H, W; convert to N, 1, H, W
        masks = masks[:, None, ...]
        # Downsample masks to target shape (nearest neighbor interpolation)
        masks = torch.nn.functional.interpolate(
            masks.float(), [args.desired_height, args.desired_width], mode="nearest"
        ).bool()
        masks = masks.squeeze(-3)

        mask_areas = []
        for maskidx in range(len(masks)):
            xywh = sam_masks[0].boxes[maskidx].xywh[0]
            mask_areas.append(xywh[2].item() * xywh[3].item())

        # Sort the mask indices in decreasing order of area
        sorted_mask_indices = np.argsort(np.array(mask_areas))[::-1]

        outfeat = torch.zeros(
            args.desired_height,
            args.desired_width,
            feat_dim,
            dtype=torch.half,
            device=args.device,
        )

        for maskidx in sorted_mask_indices.tolist():
            nonzero_inds = torch.argwhere(masks[maskidx])
            # Store the semiglobal feature for this mask
            outfeat[nonzero_inds[:, 0], nonzero_inds[:, 1]] = semiglobal_feat[
                maskidx
            ].half()

        # tokenizer = open_clip.get_tokenizer(args.open_clip_model)
        # while True:
        #     # Prompt user whether or not to continue
        #     prompt_text = input("Type a prompt ('q' to quit): ")
        #     if prompt_text == "q":
        #         break

        #     text = tokenizer([prompt_text])
        #     textfeat = clip_model.encode_text(text.cuda())
        #     textfeat = torch.nn.functional.normalize(textfeat, dim=-1)
        #     textfeat = textfeat.unsqueeze(0)

        #     fig, ax = plt.subplots(1, 2)
        #     img_to_show = img
        #     im0 = ax[0].imshow(img_to_show)
        #     ax[0].axis("off")

        #     _simfunc = torch.nn.CosineSimilarity(dim=-1)
        #     _sim = _simfunc(outfeat.float().cuda(), textfeat)
        #     _sim = (_sim - _sim.min()) / (_sim.max() - _sim.min() + 1e-12)
        #     im1 = ax[1].matshow(_sim.detach().cpu().numpy())
        #     ax[1].axis("off")
        #     divider = make_axes_locatable(ax[1])
        #     cax = divider.append_axes("right", size="5%", pad=0.05)

        #     plt.show()

        savefile = os.path.join(
            args.save_dir,
            os.path.splitext(os.path.basename(dataset.color_paths[idx]))[0] + ".pt",
        )
        torch.save(outfeat.detach().cpu(), savefile)


if __name__ == "__main__":
    main()
