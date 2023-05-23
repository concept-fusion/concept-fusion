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
    similarity_thresh: float = 0.8
    topk: int = 10000

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


def pick_points_from_gui(pcd: o3d.geometry.PointCloud) -> np.array:
    """Return a set of clicked points from the visualizer

    Args:
        pcd (o3d.geometry.PointCloud): Open3D pointcloud to visualize

    Returns:
        _type_: _description_
    """
    print("")
    print("==> Please pick a point using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("==> Afther picking a point, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


if __name__ == "__main__":

    # Process command-line args (if any)
    args = tyro.cli(ProgramArgs)

    pointclouds = Pointclouds.load_pointcloud_from_h5(args.load_path)
    pointclouds.to(args.device)

    print(f"Map embeddings: {pointclouds.embeddings_padded.shape}")

    while True:

        pcd = pointclouds.open3d(0)
        picked_points_idx = pick_points_from_gui(pcd)
        # Consider only the first picked point
        picked_points_idx = picked_points_idx[0]
        print({f"picked_points_idx": picked_points_idx})

        # Normalize the map
        map_embeddings = pointclouds.embeddings_padded
        map_embeddings_norm = torch.nn.functional.normalize(map_embeddings, dim=2)

        selected_embedding = (
            map_embeddings_norm[0, picked_points_idx].clone().detach().unsqueeze(0)
        )

        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        similarity = cosine_similarity(
            map_embeddings_norm, selected_embedding.unsqueeze(1)
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

        # Prompt user whether or not to continue
        to_continue = input("Click another point? ('q' to quit)")
        if to_continue == "q":
            break
