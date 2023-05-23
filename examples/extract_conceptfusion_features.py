import os
import pickle as pkl
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
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from tqdm import tqdm, trange
from typing_extensions import Literal


@dataclass
class ProgramArgs:
    # Torch device to run computation on (E.g., "cpu")
    device: str = "cuda"

    # SAM checkpoint and model params
    checkpoint_path: Union[str, Path] = (
        Path.home()
        / "code"
        / "gradslam-foundation"
        / "examples"
        / "checkpoints"
        / "sam_vit_h_4b8939.pth"
    )
    model_type = "vit_h"
    # Ignore masks that have valid pixels less than this fraction (of the image area)
    bbox_area_thresh: float = 0.0005
    # Number of query points (grid size) to be sampled by SAM
    points_per_side: int = 32

    # gradslam mode ("incremental" vs "batch")
    mode: Literal["incremental", "batch"] = "incremental"

    # Path to the data config (.yaml) file
    dataconfig_path: str = "dataconfigs/icl.yaml"
    # Path to the dataset directory
    data_dir: Union[str, Path] = Path.home() / "data" / "icl"
    # Sequence from the dataset to load
    sequence: str = "living_room_traj1_frei_png"
    # Start frame index
    start_idx: int = 0
    # End frame index
    end_idx: int = -1
    # Stride (number of frames to skip between successive fusion steps)
    stride: int = 20
    # Desired image width and height
    desired_height: int = 120
    desired_width: int = 160

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

    sam = sam_model_registry[args.model_type](checkpoint=Path(args.checkpoint_path))
    sam.to(device=args.device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,
        pred_iou_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    print("Extracting SAM masks...")
    for idx in trange(len(dataset)):
        imgfile = dataset.color_paths[idx]
        img = cv2.imread(imgfile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(img)
        cur_mask = masks[0]["segmentation"]
        _savefile = os.path.join(
            args.save_dir,
            os.path.splitext(os.path.basename(dataset.color_paths[idx]))[0] + ".pkl",
        )
        with open(_savefile, "wb") as f:
            pkl.dump(masks, f, protocol=pkl.HIGHEST_PROTOCOL)

    print(
        f"Initializing OpenCLIP model: {args.open_clip_model}"
        f" pre-trained on {args.open_clip_pretrained_dataset}..."
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.open_clip_model, args.open_clip_pretrained_dataset
    )
    model.cuda()
    model.eval()

    print("Computing pixel-aligned features...")
    for idx in trange(len(dataset)):
        maskfile = os.path.join(
            args.save_dir,
            os.path.splitext(os.path.basename(dataset.color_paths[idx]))[0] + ".pkl",
        )
        with open(maskfile, "rb") as f:
            masks = pkl.load(f)
        
        imgfile = dataset.color_paths[idx]
        img = cv2.imread(imgfile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = img.shape[0], img.shape[1]
        
        global_feat = None
        with torch.cuda.amp.autocast():
            # print("Extracting global CLIP features...")
            _img = preprocess(Image.open(imgfile)).unsqueeze(0)
            global_feat = model.encode_image(_img.cuda())
            global_feat /= global_feat.norm(dim=-1, keepdim=True)
            # tqdm.write(f"Image feature dims: {global_feat.shape} \n")
        global_feat = global_feat.half().cuda()
        global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, 1024)
        feat_dim = global_feat.shape[-1]
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        feat_per_roi = []
        roi_nonzero_inds = []
        similarity_scores = []
        for maskidx in range(len(masks)):
            _x, _y, _w, _h = tuple(masks[maskidx]["bbox"])  # xywh bounding box
            seg = masks[maskidx]["segmentation"]
            nonzero_inds = torch.argwhere(torch.from_numpy(masks[maskidx]["segmentation"]))
            # Note: Image is (H, W, 3). In SAM output, y coords are along height, x along width
            img_roi = img[_y : _y + _h, _x : _x + _w, :]
            img_roi = Image.fromarray(img_roi)
            img_roi = preprocess(img_roi).unsqueeze(0).cuda()
            roifeat = model.encode_image(img_roi)
            roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(nonzero_inds)
            _sim = cosine_similarity(global_feat, roifeat)
            similarity_scores.append(_sim)
        
        similarity_scores = torch.cat(similarity_scores)
        softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)
        outfeat = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, feat_dim, dtype=torch.half)
        for maskidx in range(len(masks)):
            _weighted_feat = softmax_scores[maskidx] * global_feat + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
            _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] += _weighted_feat[0].detach().cpu().half()
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] = torch.nn.functional.normalize(
                outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]].float(), dim=-1
            ).half()

        outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
        outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
        outfeat = torch.nn.functional.interpolate(outfeat, [args.desired_height, args.desired_width], mode="nearest")
        outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
        outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
        outfeat = outfeat[0].half() # --> H, W, feat_dim

        # tokenizer = open_clip.get_tokenizer(args.open_clip_model)
        # while True:
        #     # Prompt user whether or not to continue
        #     prompt_text = input("Type a prompt ('q' to quit): ")
        #     if prompt_text == "q":
        #         break

        #     text = tokenizer([prompt_text])
        #     textfeat = model.encode_text(text.cuda())
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
