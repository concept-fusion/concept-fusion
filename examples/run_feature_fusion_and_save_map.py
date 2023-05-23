"""
Script to run feature fusion and save maps to disk.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
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
from models_common import get_model
from tqdm import trange
from typing_extensions import Literal


@dataclass
class ProgramArgs:
    """Commandline args for this script"""

    # Script mode
    # Determines whether the script extracts and saves features, or
    # loads the saved features and performs fusion
    mode: Literal["extract", "fusion"] = "fusion"

    # Model params
    model_type: Literal["lseg", "dino-n3f", "vit"] = "vit"
    checkpoint_path: Union[str, Path] = (
        Path(__file__).parent / "checkpoints" / "lseg_minimal_e200.ckpt"
    )

    # Hardware accelerator
    device: str = "cuda:0"

    # Dataset args
    # Path to config file
    dataconfig_path: Union[str, Path] = Path("dataconfigs") / "icl.yaml"
    # Path to base dir of dataset
    dataset_path: Union[str, Path] = Path("~/data/icl").expanduser()
    # trajectory to sample from
    sequence: Union[str, List[str]] = "living_room_traj1_frei_png"
    # length of sequence to sample (determined by start, end, and stride)
    frame_start: int = 0
    frame_end: int = -1  # -1 to process until end of sequence
    # number of frames to skip between successive samples from trajectory
    stride: int = 20

    # Image dims for feeding into the feature extraction model
    # Height of RGB-D images in loaded sequence
    image_height: int = 120
    # Width of RGB-D images in loaded sequence
    image_width: int = 160

    # Desired feature map size
    desired_feature_height: int = 120
    desired_feature_width: int = 160

    # Directory to temporarily store embeddings to
    feat_dir: str = "saved-feat"
    # This is a placeholder -- will be set by the script once the model is run
    embedding_dim: int = 512  # dimensionality of embeddings

    # Odometry file (in format created by "realtime/compute_and_save_o3d_odom.py")
    odomfile: str = "data/azurekinect/poses.txt"

    # Directory to save pointclouds to
    dir_to_save_map: str = "saved-map"


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


def extract_and_save_features(args):

    if os.path.exists(args.feat_dir):
        # First, delete all files in the temp dir where we will house our features
        for _file in os.scandir(args.feat_dir):
            os.remove(_file.path)

    os.makedirs(args.feat_dir, exist_ok=True)

    # Get dataset
    dataset = get_dataset(
        dataconfig=args.dataconfig_path,
        basedir=args.dataset_path,
        sequence=args.sequence,
        desired_height=args.image_height,
        desired_width=args.image_width,
        start=args.frame_start,
        end=args.frame_end,
        stride=args.stride,
        load_embeddings=False,  # We will not read in embeddings; we will compute them
    )

    # Get model
    model = get_model(
        args.model_type,
        ckpt=args.checkpoint_path,
        upsample=True,
        desired_height=args.desired_feature_height,
        desired_width=args.desired_feature_width,
        device=args.device,
    )

    for idx in trange(len(dataset)):
        _color, *_ = dataset[idx]
        _color = _color / 255.0
        _color = _color.permute(2, 0, 1).unsqueeze(0)
        _feat = model(_color)

        _savefile = os.path.join(
            args.feat_dir,
            os.path.splitext(os.path.basename(dataset.color_paths[idx]))[0] + ".pt",
        )
        torch.save(_feat.detach().cpu(), _savefile)


def run_fusion_and_save_map(args):
    # Get dataset
    dataset = get_dataset(
        dataconfig_path=args.dataconfig_path,
        basedir=args.dataset_path,
        sequence=args.sequence,
        desired_height=args.image_height,
        desired_width=args.image_width,
        start=args.frame_start,
        end=args.frame_end,
        stride=args.stride,
        load_embeddings=False,  # We will not read in embeddings; we will compute them
        odomfile=args.odomfile,
    )

    slam = PointFusion(odom="gt", dsratio=1, device=args.device, use_embeddings=True)

    frame_cur, frame_prev = None, None
    pointclouds = Pointclouds(
        device=args.device,
    )

    print("Running PointFusion (incremental mode)...")

    for idx in trange(len(dataset)):
        _color, _depth, intrinsics, _pose, *_ = dataset[idx]
        _loadfile = os.path.join(
            args.feat_dir,
            os.path.splitext(os.path.basename(dataset.color_paths[idx]))[0] + ".pt",
        )
        _embedding = torch.load(_loadfile)
        _embedding = _embedding.float()
        _embedding = torch.nn.functional.interpolate(
            _embedding.permute(2, 0, 1).unsqueeze(0).float(),
            [args.image_height, args.image_width],
            mode="nearest",
        )[0].permute(1, 2, 0).half().cuda()
        # _embedding = torch.nn.functional.interpolate(
        #     _embedding, [args.image_height, args.image_width], mode="nearest"
        # )
        # _embedding = _embedding.permute(0, 2, 3, 1).half().cuda()

        frame_cur = RGBDImages(
            _color.unsqueeze(0).unsqueeze(0),
            _depth.unsqueeze(0).unsqueeze(0),
            intrinsics.unsqueeze(0).unsqueeze(0),
            _pose.unsqueeze(0).unsqueeze(0),
            embeddings=_embedding.unsqueeze(0).unsqueeze(0),
        )
        pointclouds, _ = slam.step(pointclouds, frame_cur, frame_prev)
        # frame_prev = frame_cur
        torch.cuda.empty_cache()

    os.makedirs(args.dir_to_save_map, exist_ok=True)
    pointclouds.save_to_h5(args.dir_to_save_map)


if __name__ == "__main__":

    args = tyro.cli(ProgramArgs)

    if args.mode == "extract":
        extract_and_save_features(args)
    elif args.mode == "fusion":
        run_fusion_and_save_map(args)
