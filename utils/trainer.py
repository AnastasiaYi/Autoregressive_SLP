import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from utils.helper import load_config

class TrainerStage1:
    def __init__(self, dataset, encoder, decoder, codebook, cfg_path="./Configs/Base.yaml", device='gpu'):
        cfg=load_config(cfg_path)
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.codebook = codebook.to(device)
        self.device = device
        self.num_epochs = cfg['epochs']
        self.batch_size = cfg['batch_size']
        self.learning_rate = cfg['learning_rate']
        self.dataset = dataset
        self.alpha = cfg['alpha']
        self.beta = cfg['beta']

        self.dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Loss
        self.criterion_reconstruction = nn.MSELoss()
        self.criterion_diversity = nn.CrossEntropyLoss()

    def train_epoch(self):
        self.encoder.train()
        self.decoder.train()
        self.codebook.train()

        epoch_loss = 0.0
        for batch in self.dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            encoded = self.encoder(batch)
            quantized, soft_assignment = self.codebook(encoded)
            reconstructed = self.decoder(quantized)

            reconstruction_loss = self.criterion_reconstruction(reconstructed, batch)
            diversity_loss = -torch.sum(soft_assignment * torch.log(soft_assignment + 1e-6)) / batch.size(0)

            loss = reconstruction_loss + 0.1 * diversity_loss  # Adjust diversity weight as needed

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(self.dataloader)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
