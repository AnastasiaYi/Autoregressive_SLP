import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import networkx as nx
import copy
import numpy as np

def create_mediapipe_hand_graph():
    # Create a graph for 21 keypoints per hand, totaling 42 nodes.
    graph = nx.Graph()
    graph.add_nodes_from(range(42))  # 42 nodes for both hands
    
    # Define edges based on MediaPipe keypoint connections for a single hand
    hand_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]
    
    # Add edges for both hands (21 nodes each)
    for hand in range(2):
        offset = hand * 21
        for edge in hand_edges:
            graph.add_edge(edge[0] + offset, edge[1] + offset)
    
    return graph

class GraphPyramid:
    def __init__(self, skeleton_graph):
        # Store the original skeleton graph and build an initial pyramid with only the original graph
        self.original_graph = skeleton_graph.copy()
        self.current = self.original_graph.copy()

    def downsample_graph(graph):
        """
        Downsample the graph by one level, removing nodes while updating position information to maintain spatial coherence.
        Args:
            graph (networkx.Graph): The graph to downsample, with nodes having "position" attributes (x, y coordinates).
        Returns:
            networkx.Graph: The downsampled graph after one level, with updated positions.
        """
        downsampled_graph = graph.copy()
        nodes_to_remove = []

        for node in list(downsampled_graph.nodes):
            if len(downsampled_graph[node]) > 1:
                neighbors = list(downsampled_graph.neighbors(node))
                nodes_to_remove.append(node)
                
                # Calculate the average position of the neighboring nodes to replace the removed node's position
                avg_position = np.mean([downsampled_graph.nodes[neighbor]["position"] for neighbor in neighbors], axis=0)
                
                # Assign the averaged position to one of the neighbors (or spread among neighbors)
                for neighbor in neighbors:
                    downsampled_graph.nodes[neighbor]["position"] = avg_position
                
                # Add edges between neighbors to preserve connectivity
                downsampled_graph.add_edges_from((neighbor, n) for neighbor in neighbors for n in neighbors if n != neighbor)
                downsampled_graph.remove_node(node)

        return downsampled_graph

# Graph Convolution Layer
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        h = torch.matmul(adj, x)  # Apply adjacency matrix to input
        h = self.fc(h)
        return self.relu(h)

class STGPBlock(nn.Module):
    def __init__(self, in_features, out_features, graph_pyramid):
        super(STGPBlock, self).__init__()
        self.input_projection = nn.Linear(in_features, out_features)
        self.graph_conv = GraphConvLayer(in_features, out_features)
        self.temporal_conv = nn.Conv1d(out_features, out_features, kernel_size=1)  # Temporal 1x1 Convolution
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.graph_pyramid = graph_pyramid

    def forward(self, x):
        # Obtain the downsampled graph and adjacency matrix
        downsampled_graph = self.graph_pyramid.downsample()
        adj_matrix = torch.tensor(nx.to_numpy_array(downsampled_graph), dtype=torch.float32)
        
        # Select only the features corresponding to the nodes in the downsampled graph
        downsampled_nodes = list(downsampled_graph.nodes)
        x = x[downsampled_nodes]  # Match x to the downsampled graph's nodes

        x_proj = self.input_projection(x)

        # Graph
        out = self.graph_conv(x_proj, adj_matrix)
        
        # Temporal
        out = out.permute(1, 0).unsqueeze(0)  # Prepare for temporal convolution (batch, channel, time)
        out = self.batch_norm(self.temporal_conv(out)).squeeze(0).permute(1, 0)

        return out + x_proj

class STGPNetwork(nn.Module):
    def __init__(self, num_blocks, in_features, out_features, base_pyramid):
        super(STGPNetwork, self).__init__()
        self.blocks = nn.ModuleList([
            STGPBlock(in_features, out_features, base_pyramid) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            block.graph_pyramid.reset()

        for block in self.blocks:
            x = block(x)

        return x

# Example Usage with MediaPipe Structure
if __name__ == "__main__":
    # Initialize the graph with MediaPipe hand keypoints
    mediapipe_hand_graph = create_mediapipe_hand_graph()
    base_pyramid = GraphPyramid(mediapipe_hand_graph)
    
    # Define input tensors and model
    num_blocks = 3
    in_features, out_features = 2, 32  # V=42, C=2 for 2D data; adjust C if using 3D data
    stgp_network = STGPNetwork(num_blocks, in_features, out_features, base_pyramid)

    x = torch.randn(42, in_features)

    # Forward pass through STGP Network
    output = stgp_network(x)

    print("STGP Network output shape:", output.shape)


