import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import video2pose

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.video_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_name = os.path.join(self.data_dir, self.video_files[idx])
        pose_sequence = video2pose.extract_pose_sequence(video_name)
        if self.transform:
            pose_sequence = self.transform(pose_sequence)
        
        if self.mode == 'train':
            # Assuming you have labels for training data
            label = self.get_label(video_name)
            return pose_sequence, label
        else:
            return pose_sequence

    def get_label(self, video_name):
        # Implement this method to return the label for the given video
        # For example, you could extract the label from the video file name or a separate file
        pass

def get_dataloader(data_dir, batch_size=32, shuffle=True, transform=None):
    dataset = CustomDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader