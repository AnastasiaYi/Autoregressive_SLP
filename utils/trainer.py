import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from utils.helper import load_config
from utils.dataset import PoseSkeletonDataset
from utils.loss import Loss

class Stage1Trainer:
    def __init__(self, dataset, model, cfg_path="./Configs/Base.yaml", device='gpu'):
        cfg=load_config(cfg_path)
        self.model = model.to(device)
        self.device = device
        self.num_epochs = cfg['epochs']
        self.batch_size = cfg['batch_size']
        self.learning_rate = cfg['learning_rate']
        self.dataset = dataset
        self.loss_list = Loss(cfg['loss'])
        self.alpha = cfg['alpha']
        self.beta = cfg['beta']

        self.dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def collate_fn(self, batch):
        """
        Custom collate function to pad sequences within a batch to the same length.
        """
        sequences = [item for item in batch]
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        return padded_sequences.to(self.device)

    def train_epoch(self):
        """
        Train the model for one epoch and return the average loss.
        """
        self.model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Training")):
            self.optimizer.zero_grad()
            
            # Forward pass
            reconstructed, mu, logvar = self.model(batch)
            
            # Compute loss
            # TODO: Implement the loss function
            epoch_loss += loss.item()
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
        
        return epoch_loss / len(self.dataloader)
    
    def train(self):
        """
        Perform the training over the specified number of epochs.
        """
        for epoch in range(self.num_epochs):
            avg_loss = self.train_epoch()
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}")
        print("Training complete.")

# Usage example:
# Assuming `vae_model` is an instance of your VAE model and `pose_skeleton_dataset` is the dataset.
# trainer = Trainer(model=vae_model, dataset=pose_skeleton_dataset, batch_size=16, learning_rate=1e-3, num_epochs=10, device='cuda')
# trainer.train()


