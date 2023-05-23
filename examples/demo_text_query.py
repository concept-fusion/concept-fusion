"""
Script to run gradslam over a sequence from ICL and visualize self-similarity
over fused feature maps (assumes we point-and-click a 3D map point to compute
self-similarity scores againt the other points in the map).
"""

import json
import pathlib
from dataclasses import dataclass
from typing import List, Union

import matplotlib
import numpy as np
import open3d as o3d
import open_clip
import torch
import tyro
from gradslam.structures.pointclouds import Pointclouds
from typing_extensions import Literal


@dataclass
class ProgramArgs:
    """Commandline args for this script"""

    # Path to saved pointcloud to visualize
    load_path: str

    device: str = "cuda:0"

    # Similarity computation and visualization params
    viz_type: Literal["topk", "thresh"] = "thresh"
    similarity_thresh: float = 0.6
    topk: int = 10000

    # CLIP model config
    open_clip_model = "ViT-H-14"
    open_clip_pretrained_dataset = "laion2b_s32b_b79k"

    def to_dict(self) -> dict:
        """Convert the ProgramArgs object to a dictionary"""
        attrs = {}
        for attr in vars(self):
            temp = getattr(self, attr)
            if isinstance(temp, pathlib.PosixPath):
                temp = str(temp)
            attrs[attr] = temp
        return attrs

    def to_json(self) -> str:
        """Convert the ProgramArgs object to a JSON string"""
        return json.dumps(self.to_dict(), indent=2)


if __name__ == "__main__":

    # Process command-line args (if any)
    args = tyro.cli(ProgramArgs)

    pointclouds = Pointclouds.load_pointcloud_from_h5(args.load_path)
    pointclouds.to(args.device)

    print(f"Map embeddings: {pointclouds.embeddings_padded.shape}")

    print(
        f"Initializing OpenCLIP model: {args.open_clip_model}"
        f" pre-trained on {args.open_clip_pretrained_dataset}..."
    )
    model, _, _ = open_clip.create_model_and_transforms(
        args.open_clip_model, args.open_clip_pretrained_dataset
    )
    model.cuda()
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.open_clip_model)

    while True:
        # Prompt user whether or not to continue
        prompt_text = input("Type a prompt ('q' to quit): ")
        if prompt_text == "q":
            break

        text = tokenizer([prompt_text])
        textfeat = model.encode_text(text.cuda())
        textfeat = torch.nn.functional.normalize(textfeat, dim=-1)
        textfeat = textfeat.unsqueeze(0)

        # Normalize the map
        map_embeddings = pointclouds.embeddings_padded.cuda()
        map_embeddings_norm = torch.nn.functional.normalize(map_embeddings, dim=2)
        print(f"map_embeddings_norm: {map_embeddings_norm.shape}")
        print(f"textfeat: {textfeat.shape}")

        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        similarity = cosine_similarity(
            map_embeddings_norm, textfeat
        )

        pcd = pointclouds.open3d(0)
        map_colors = np.asarray(pcd.colors)

        if args.viz_type == "topk":
            # Viz topk points
            _, topk_ind = torch.topk(similarity, args.topk)
            map_colors[topk_ind.detach().cpu().numpy()] = np.array([1.0, 0.5, 0.0])
        elif args.viz_type == "thresh":
            # Viz thresholded "relative" attention scores
            similarity = (similarity + 1.0) / 2.0  # scale from [-1, 1] to [0, 1]
            # similarity = similarity.clamp(0., 1.)
            similarity_rel = (similarity - similarity.min()) / (
                similarity.max() - similarity.min() + 1e-12
            )
            similarity_rel[similarity_rel < args.similarity_thresh] = 0.0
            cmap = matplotlib.cm.get_cmap("jet")
            similarity_colormap = cmap(similarity_rel[0].detach().cpu().numpy())[:, :3]
            print(map_colors.shape, similarity_colormap.shape)
            map_colors = 0.5 * map_colors + 0.5 * similarity_colormap

        # Assign colors and display GUI
        pcd.colors = o3d.utility.Vector3dVector(map_colors)
        o3d.visualization.draw_geometries([pcd])
