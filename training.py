import torch
import torch.nn as nn
from utils.dataset import PoseSkeletonDataset
from utils.loss import Loss
from models.STGP import STGPEncoder, STGPDecoder, Codebook


dataset = PoseSkeletonDataset(root_dir='./data/train', annotation_path='./data/annotation.csv')
encoder = STGPEncoder(in_channels=2, out_channels=2, num_blocks=2)
decoder = STGPDecoder(in_channels=128, out_channels=64, num_blocks=2)
codebook = Codebook(codebook_size=512, embedding_dim=128)