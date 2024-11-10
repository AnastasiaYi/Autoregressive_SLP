import torch
import torch.nn.functional as F

def codebook_loss_single(P, P_hat, c, c_hat, alpha):
    """
    Computes the Codebook loss for a single sample.

    Args:
        P (torch.Tensor): Original pose sequence (sequence_length, features).
        P_hat (torch.Tensor): Predicted pose sequence (sequence_length, features).
        c (torch.Tensor): Counter values (sequence_length, 1).
        c_hat (torch.Tensor): Predicted counter values (sequence_length, 1).
        alpha (float): Scaling factor for the counter loss.

    Returns:
        torch.Tensor: The codebook loss for a single sample.
    """
    pose_loss = F.mse_loss(P, P_hat, reduction='mean')
    counter_loss = F.mse_loss(c, c_hat, reduction='mean')
    return pose_loss + alpha * counter_loss

def supervised_contrastive_loss_single(z, pos_indices, neg_indices, tau):
    """
    Computes the Supervised Contrastive loss for a single sample.

    Args:
        z (torch.Tensor): Encoded representation (features,).
        pos_indices (list): Indices for positive samples.
        neg_indices (list): Indices for negative samples.
        tau (float): Temperature parameter for contrastive loss.

    Returns:
        torch.Tensor: The supervised contrastive loss for a single sample.
    """
    pos_exp_sum = sum(torch.exp(torch.dot(z, z[a]) / tau) for a in pos_indices)
    neg_exp_sum = sum(torch.exp(torch.dot(z, z[b]) / tau) for b in neg_indices)

    # Avoid division by zero
    if neg_exp_sum > 0:
        return -torch.log(pos_exp_sum / neg_exp_sum)
    else:
        return torch.tensor(0.0)

import torch
import torch.nn.functional as F

def reconstruction_loss(y_pred, y_true):
    """
    Computes the reconstruction loss (L2 loss).
    
    Args:
        y_pred (torch.Tensor): The predicted values.
        y_true (torch.Tensor): The ground truth values.
        
    Returns:
        torch.Tensor: The L2 loss.
    """
    return F.mse_loss(y_pred, y_true)

def sign_loss(y_pred, y_true):
    """
    Computes the sign loss.
    
    Args:
        y_pred (torch.Tensor): The predicted sign values.
        y_true (torch.Tensor): The ground truth sign values.
        
    Returns:
        torch.Tensor: The sign loss.
    """
    return -torch.sum(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

