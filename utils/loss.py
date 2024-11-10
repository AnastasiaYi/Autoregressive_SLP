from torch import nn
import torch.nn.functional as F
import torch

class Loss(nn.Module):
    def __init__(name, self):
        super(Loss, self).__init__()
        self.name = name
    
    def forward(self, predicted=None, true=None, codebook_probs=None, decoder_hidden_states=None, encoder_hidden_states=None):
        if self.name == 'L2':
            L = true.size(1)
            loss_pose = F.mse_loss(predicted, true, reduction='sum') / L
            return loss_pose
        elif self.name == 'diversity':
            loss_div = -torch.sum(codebook_probs * torch.log(codebook_probs + 1e-8))
            return loss_div
        elif self.name == 'cross_entropy':
            loss_ce = F.cross_entropy(predicted.view(-1, predicted.size(-1)), true.view(-1), reduction='mean')
            return loss_ce
        elif self.name == 'latent_alignment':
            M = decoder_hidden_states.size(1)  # Sequence length
            loss_latent = F.mse_loss(decoder_hidden_states, encoder_hidden_states, reduction='sum') / M
            return loss_latent
        
        