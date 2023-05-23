"""
Script to provide a unified interface to access models
"""

import warnings
from pathlib import Path
from typing import Union

import torch
import torchvision
from typing_extensions import Literal


def upsample_feat_vec(feat, target_shape):
    return torch.nn.functional.interpolate(
        feat, target_shape, mode="bilinear", align_corners=True
    )


class LSegFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        ckpt_path: Union[str, Path],
        backbone: Literal[
            "clip_vitl16_384", "clipRN50x16_vitl16_384", "clip_vitb32_384"
        ] = "clip_vitl16_384",
        num_features: int = 256,
        crop_size: int = 480,
        arch_option: int = 0,
        block_depth: int = 0,
        activation: str = "lrelu",
        scale_factor: float = 1.0,
        upsample: bool = False,
        device: str = "cuda:0",
        **kwargs,
    ):
        from lseg import LSegNet

        super().__init__()
        self.model = LSegNet(
            backbone=backbone,
            features=num_features,
            crop_size=crop_size,
            arch_option=arch_option,
            block_depth=block_depth,
            activation=activation,
            scale_factor=scale_factor,
        )
        self.model.load_state_dict(torch.load(str(ckpt_path)), strict=False)
        self.model.eval()
        self.model.to(device)
        if upsample == True:
            if "desired_height" in kwargs.keys():
                self.desired_height = kwargs["desired_height"]
                if "desired_width" in kwargs.keys():
                    self.desired_width = kwargs["desired_width"]
                    self.upsample = True
                else:
                    warnings.warn(
                        "Ignoring upsample arguments as they are incomplete. "
                        "Missing `desired_width`."
                    )
            else:
                warnings.warn(
                    "Ignoring upsample arguments as they are incomplete. "
                    "Missing `desired_height`."
                )
        else:
            self.upsample = False

    def forward(self, img):
        feat = self.model.forward(img)
        if self.upsample:
            feat = upsample_feat_vec(feat, [self.desired_height, self.desired_width])
        return feat


class DINOFeatureExtractorN3F(torch.nn.Module):
    """Uses the DINO feature extractor based on the N3F (neural feature fusion fields)
    paper.
    """

    def __init__(
        self,
        ckpt_path: Union[str, Path],
        patch_size: int = 8,
        upsample=False,
        device="cuda:0",
        **kwargs,
    ):
        from dinofeat import DINO

        super().__init__()
        self.model = DINO(patch_size=patch_size, device=device)
        self.model.load_checkpoint(ckpt_path)
        self.input_image_transform = self.get_input_image_transform()
        if upsample == True:
            if "desired_height" in kwargs.keys():
                self.desired_height = kwargs["desired_height"]
                if "desired_width" in kwargs.keys():
                    self.desired_width = kwargs["desired_width"]
                    self.upsample = True
                else:
                    warnings.warn(
                        "Ignoring upsample arguments as they are incomplete. "
                        "Missing `desired_width`."
                    )
            else:
                warnings.warn(
                    "Ignoring upsample arguments as they are incomplete. "
                    "Missing `desired_height`."
                )
        else:
            self.upsample = False

    def get_input_image_transform(self):
        _NORM_MEAN = [0.485, 0.456, 0.406]
        _NORM_STD = [0.229, 0.224, 0.225]
        return torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD)]
        )

    def forward(self, img):
        img = self.input_image_transform(img)
        feat = self.model.extract_features(img, transform=False, upsample=False)
        if self.upsample:
            feat = upsample_feat_vec(feat, [self.desired_height, self.desired_width])
        return feat


class VITFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        model_type="dino_vits8",
        stride=4,
        device="cuda:0",
        load_size=224,
        upsample=False,
        **kwargs,
    ):
        from dino_feature_extractor import ViTExtractor

        super().__init__()
        self.extractor = ViTExtractor(model_type, stride, device=device)
        self.load_size = load_size
        self.input_image_transform = self.get_input_image_transform()
        if upsample == True:
            if "desired_height" in kwargs.keys():
                self.desired_height = kwargs["desired_height"]
                if "desired_width" in kwargs.keys():
                    self.desired_width = kwargs["desired_width"]
                    self.upsample = True
                else:
                    warnings.warn(
                        "Ignoring upsample arguments as they are incomplete. "
                        "Missing `desired_width`."
                    )
            else:
                warnings.warn(
                    "Ignoring upsample arguments as they are incomplete. "
                    "Missing `desired_height`."
                )
        else:
            self.upsample = False
        # Layer to extract feature maps from
        self.layer_idx_to_extract_from = 11
        if "layer" in kwargs.keys():
            self.layer_idx_to_extract_from = kwargs["layer"]
        # Type of attention component to create descriptors from
        self.facet = "key"
        if "facet" in kwargs.keys():
            self.facet = kwargs["facet"]
        # Whether or not to create a binned descriptor
        self.binned = False
        if "binned" in kwargs.keys():
            self.binned = kwargs["binned"]

    def get_input_image_transform(self):
        _NORM_MEAN = [0.485, 0.456, 0.406]
        _NORM_STD = [0.229, 0.224, 0.225]
        return torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD)]
        )

    def forward(self, img, apply_default_input_transform=True):
        img = torchvision.transforms.functional.resize(img, self.load_size)
        if apply_default_input_transform:
            # Default input image transfoms
            img = self.input_image_transform(img)
        feat = self.extractor.extract_descriptors(
            img, self.layer_idx_to_extract_from, self.facet, self.binned
        )
        feat = feat.reshape(
            self.extractor.num_patches[0],
            self.extractor.num_patches[1],
            feat.shape[-1],
        )
        feat = feat.permute(2, 0, 1)
        feat = feat.unsqueeze(0)
        if self.upsample:
            feat = upsample_feat_vec(feat, [self.desired_height, self.desired_width])
        return feat


def get_model(model_type, ckpt=None, upsample=False, **kwargs):
    if model_type.lower() == "lseg":
        return LSegFeatureExtractor(ckpt, upsample=upsample, **kwargs)
    elif model_type.lower() == "dino-n3f":
        return DINOFeatureExtractorN3F(ckpt, upsample=upsample, **kwargs)
    elif model_type.lower() == "vit":
        return VITFeatureExtractor(upsample=upsample, **kwargs)
    else:
        raise ValueError(f"Invalid model_type {model_type}.")


if __name__ == "__main__":

    # lseg_feature_extractor = LSegFeatureExtractor(
    #     "checkpoints/lseg_minimal_e200.ckpt",
    #     # upsample=True,
    #     # desired_height=480,
    #     # desired_width=640,
    # )

    # dino_feature_extractor = DINOFeatureExtractorN3F(
    #     "checkpoints/dino_vitbase8_pretrain.pth",
    #     # upsample=True,
    #     # desired_height=240,
    #     # desired_width=320,
    # )

    # dino_feature_extractor = VITFeatureExtractor(
    #     # upsample=True,
    #     # desired_height=240,
    #     # desired_width=320,
    # )

    img = torch.rand(1, 3, 480, 640).cuda()
    # feat = lseg_feature_extractor(img)
    feat = dino_feature_extractor(img)
    print(feat.shape)
