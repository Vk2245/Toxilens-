"""
Graph Neural Network (GNN) model for Tox21 toxicity prediction.

This module implements a multi-task GNN using AttentiveFP architecture with
joint correlation loss to leverage cross-assay relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentiveFP, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Tuple
import numpy as np


class ToxGNN(nn.Module):
    """
    Multi-task Graph Neural Network for toxicity prediction.
    
    Uses AttentiveFP architecture with 4 graph convolution layers,
    global pooling, and 12 output heads for Tox21 assays.
    """
    
    def __init__(
        self,
        node_feat_dim: int = 39,  # From graph_builder.py
        edge_feat_dim: int = 10,  # From graph_builder.py
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_tasks: int = 12,
        dropout: float = 0.3
    ):
        """
        Initialize ToxGNN model.
        
        Args:
            node_feat_dim: Dimension of node features
            edge_feat_dim: Dimension of edge features
            hidden_dim: Hidden dimension for graph convolutions
            num_layers: Number of AttentiveFP layers
            num_tasks: Number of prediction tasks (12 for Tox21)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        
        # AttentiveFP layers
        self.attentivefp = AttentiveFP(
            in_channels=node_feat_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=edge_feat_dim,
            num_layers=num_layers,
            num_timesteps=2,
            dropout=dropout
        )
        
        # Classification head
        # AttentiveFP outputs graph-level embeddings of size hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_tasks)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features (num_nodes, node_feat_dim)
                - edge_index: Edge connectivity (2, num_edges)
                - edge_attr: Edge features (num_edges, edge_feat_dim)
                - batch: Batch assignment (num_nodes,)
        
        Returns:
            Logits tensor of shape (batch_size, num_tasks)
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # AttentiveFP forward pass - returns graph-level embeddings directly
        # Shape: (batch_size, hidden_dim)
        graph_embedding = self.attentivefp(x, edge_index, edge_attr, batch)
        
        # Classification
        graph_embedding = self.dropout(graph_embedding)
        logits = self.fc(graph_embedding)
        
        return logits


def compute_correlation_matrix(predictions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute correlation matrix between assay predictions.
    
    Args:
        predictions: Predicted probabilities (batch_size, num_tasks)
        labels: Ground truth labels (batch_size, num_tasks)
        mask: Binary mask for valid labels (batch_size, num_tasks)
    
    Returns:
        Correlation matrix (num_tasks, num_tasks)
    """
    num_tasks = predictions.shape[1]
    
    # Apply mask to get valid predictions
    valid_preds = []
    for i in range(num_tasks):
        task_mask = mask[:, i]
        if task_mask.sum() > 1:  # Need at least 2 samples
            valid_preds.append(predictions[task_mask, i])
        else:
            valid_preds.append(torch.zeros(2, device=predictions.device))  # Dummy
    
    # Compute correlation matrix
    corr_matrix = torch.zeros(num_tasks, num_tasks, device=predictions.device)
    
    for i in range(num_tasks):
        for j in range(i, num_tasks):
            if len(valid_preds[i]) > 1 and len(valid_preds[j]) > 1:
                # Pearson correlation
                pred_i = valid_preds[i]
                pred_j = valid_preds[j]
                
                # Ensure same length
                min_len = min(len(pred_i), len(pred_j))
                pred_i = pred_i[:min_len]
                pred_j = pred_j[:min_len]
                
                if min_len > 1:
                    corr = torch.corrcoef(torch.stack([pred_i, pred_j]))[0, 1]
                    if not torch.isnan(corr):
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
    
    return corr_matrix


def joint_correlation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_correlation: torch.Tensor,
    lambda_corr: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute joint loss combining masked BCE and correlation consistency.
    
    Args:
        logits: Model predictions (batch_size, num_tasks)
        labels: Ground truth labels with NaN for missing (batch_size, num_tasks)
        target_correlation: Target correlation matrix from training data (num_tasks, num_tasks)
        lambda_corr: Weight for correlation loss
    
    Returns:
        Tuple of (total_loss, bce_loss, corr_loss)
    """
    # Create mask for valid labels (not NaN)
    mask = ~torch.isnan(labels)
    
    # Replace NaN with 0 for computation (will be masked out)
    labels_clean = torch.where(mask, labels, torch.zeros_like(labels))
    
    # Compute masked BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(
        logits,
        labels_clean,
        reduction='none'
    )
    
    # Apply mask and compute mean
    bce_loss = (bce_loss * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
    
    # Compute predicted probabilities
    probs = torch.sigmoid(logits)
    
    # Compute correlation matrix of predictions
    pred_correlation = compute_correlation_matrix(probs, labels_clean, mask)
    
    # Correlation consistency loss (Frobenius norm)
    corr_loss = F.mse_loss(pred_correlation, target_correlation)
    
    # Total loss
    total_loss = bce_loss + lambda_corr * corr_loss
    
    return total_loss, bce_loss, corr_loss


if __name__ == "__main__":
    # Test model instantiation
    model = ToxGNN(
        node_feat_dim=39,
        edge_feat_dim=10,
        hidden_dim=256,
        num_layers=4,
        num_tasks=12,
        dropout=0.3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model architecture:\n{model}")
