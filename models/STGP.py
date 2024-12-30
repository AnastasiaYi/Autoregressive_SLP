import torch
import torch.nn as nn
import networkx as nx
import copy
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.video2pose import extract_keypoints_from_folder
from utils.helper import create_graph
from torch.nn import Module


def downsample_graph(graph):
    """
    Downsample the graph.
    Removes nodes systematically while preserving connectivity.
    """
    nodes = list(graph.nodes)
    downsampled_nodes = nodes[::2]
    downsampled_graph = nx.Graph()
    downsampled_graph.add_nodes_from(downsampled_nodes)

    # Add edges to preserve connectivity
    for u in downsampled_nodes:
        for v in downsampled_nodes:
            if u != v and nx.has_path(graph, u, v):
                path = nx.shortest_path(graph, u, v)
                if len(path) == 2:  # Direct neighbors
                    downsampled_graph.add_edge(u, v)

    return downsampled_graph


def upsample_graph(coarse_graph, fine_graph):
    """
    Upsample the graph by interpolating features for finer nodes.
    """
    upsampled_graph = nx.Graph()
    upsampled_graph.add_nodes_from(fine_graph.nodes)

    for u, v in fine_graph.edges:
        if u in coarse_graph.nodes and v in coarse_graph.nodes:
            upsampled_graph.add_edge(u, v)  # Retain original edges
        elif u in coarse_graph.nodes or v in coarse_graph.nodes:
            # Add interpolated edges for finer nodes
            upsampled_graph.add_edge(u, v)

    return upsampled_graph


class STGPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, base_graph, levels, kernel_size=3, stride=1, padding=1):
        super(STGPBlock, self).__init__()
        self.levels = levels
        self.base_graph = base_graph

        # Build graph pyramid
        self.graph_pyramid = self.build_graph_pyramid(base_graph, levels)

        # Spatial graph convolutions at each level
        self.spatial_convs = nn.ModuleList([
            self.create_spatial_conv(in_channels if i == 0 else out_channels, out_channels, nx.to_numpy_array(g))
            for i, g in enumerate(self.graph_pyramid)
        ])

        # Temporal convolution
        self.temporal_conv = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # Residual connection
        self.residual = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        ) if in_channels != out_channels else nn.Identity()

    def build_graph_pyramid(self, base_graph, levels):
        """
        Build a pyramid of downsampled graphs.
        """
        pyramid = [base_graph]
        current_graph = base_graph
        for _ in range(levels - 1):
            downsampled_graph = downsample_graph(current_graph)
            pyramid.append(downsampled_graph)
            current_graph = downsampled_graph
        return pyramid

    def create_spatial_conv(self, in_channels, out_channels, adjacency_matrix):
        """
        Create a spatial convolution for a specific adjacency matrix.
        """
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            lambda x: torch.matmul(adjacency_matrix.to(x.device), x)
        )

    def forward(self, x):
        """
        Forward pass through the STGP block.
        x: Tensor of shape (batch_size, num_nodes, time_steps, in_channels)
        """
        batch_size, num_nodes, time_steps, in_channels = x.size()
        x = x.permute(0, 3, 1, 2)  # (batch_size, in_channels, num_nodes, time_steps)

        # Apply spatial convolutions at each level of the pyramid
        for spatial_conv in self.spatial_convs:
            x = spatial_conv(x)

        # Apply temporal convolution
        x = self.temporal_conv(x.permute(0, 2, 3, 1))  # (batch_size, time_steps, num_nodes, out_channels)

        # Residual connection
        x_residual = self.residual(x)
        return x + x_residual

class STGPEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, base_graph, num_blocks):
        super(STGPEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            STGPBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                base_graph=base_graph,
                levels=num_blocks
            )
            for i in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class STGPDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, base_graph, num_blocks):
        super(STGPDecoder, self).__init__()
        self.blocks = nn.ModuleList([
            STGPBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                base_graph=base_graph,
                levels=num_blocks
            )
            for i in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


if __name__=="__main__":
    # Run  python -m models.STGP  from project root
    folder_path = './models/trail_vid/'
    image_files = sorted(os.listdir(folder_path))
    keypoints = extract_keypoints_from_folder(folder_path, image_files)
    keypoints = [tuple(k) for k in keypoints]
    base_graph = create_graph(keypoints)

    downsampled_graph = downsample_graph(base_graph)
    print(f"Original graph: {len(base_graph.nodes)} nodes, {len(base_graph.edges)} edges")
    print(f"Downsampled graph: {len(downsampled_graph.nodes)} nodes, {len(downsampled_graph.edges)} edges")

    # Test upsample_graph function
    upsampled_graph = upsample_graph(downsampled_graph, base_graph)
    print(f"Upsampled graph: {len(upsampled_graph.nodes)} nodes, {len(upsampled_graph.edges)} edges")
