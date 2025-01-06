import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import networkx as nx
import os
import numpy as np
from video2pose import extract_keypoints_from_folder
from helper import get_annotation_by_folder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from utils.helper import create_graph
from torch_geometric.utils.convert import from_networkx

class PoseSkeletonDataset(Dataset):
    def __init__(self, root_dir, annotation_path=None, transform=None):
        self.root_dir = root_dir #Train/test folder
        self.transform = transform
        self.annotation_path = annotation_path
        self.video_folders = [os.path.join(root_dir, folder) for folder in sorted(os.listdir(root_dir))]
    
    def __len__(self):
        return len(self.video_folders)
    
    def __getitem__(self, idx):
        video_path = self.video_folders[idx]
        image_files = sorted(os.listdir(video_path))

        keypoints = extract_keypoints_from_folder(image_files, transform=self.transform) #17 for body, 21 for left hand and 21 for right hand.
        keypoints = [tuple(k) for k in keypoints]
        base_graph = create_graph(keypoints)
        torch_graph = from_networkx(base_graph)

        if self.annotation_path:
            label = get_annotation_by_folder(video_path, self.annotation_path)
            return torch_graph, label
        
        return torch_graph
    
    def collate_fn(batch):
        sequences, labels = zip(*batch)
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        return padded_sequences, labels
        

