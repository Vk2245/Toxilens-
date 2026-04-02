"""
Graph Neural Network (GNN) model for Tox21 toxicity prediction.

This module provides a wrapper for the trained ToxGNN model that predicts
toxicity across 12 Tox21 assays using molecular graph representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentiveFP
from torch_geometric.data import Data
from pathlib import Path
from typing import Union

import numpy as np


class ToxGNN(nn.Module):
    """
    Multi-task Graph Neural Network for toxicity prediction.
    
    Uses AttentiveFP architecture with 4 graph convolution layers,
    global pooling, and 12 output heads for Tox21 assays.
    
    This is the inference version copied from ml/models/gnn.py.
    """
    
    def __init__(
        self,
        node_feat_dim: int = 133,
        edge_feat_dim: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_tasks: int = 12,
        dropout: float = 0.3
    ):
        """
        Initialize ToxGNN model.
        
        Args:
            node_feat_dim: Dimension of node features (default: 133)
            edge_feat_dim: Dimension of edge features (default: 7)
            hidden_dim: Hidden dimension for graph convolutions (default: 256)
            num_layers: Number of AttentiveFP layers (default: 4)
            num_tasks: Number of prediction tasks (default: 12 for Tox21)
            dropout: Dropout probability (default: 0.3)
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


class GNNModelWrapper:
    """
    Wrapper for loading and running inference with trained ToxGNN model.
    
    This class handles model loading from checkpoint files and provides
    a simple predict() interface with GPU support.
    
    Examples:
        >>> model = GNNModelWrapper("ml/artifacts/gnn_best.pt", device="cuda")
        >>> graph = mol_to_graph(mol)  # PyTorch Geometric Data object
        >>> logits = model.predict(graph)
        >>> logits.shape
        (12,)
    """
    
    def __init__(self, checkpoint_path: Union[str, Path], device: str = 'cpu'):
        """
        Initialize GNN model wrapper by loading from checkpoint.
        
        Args:
            checkpoint_path: Path to GNN checkpoint file (.pt)
            device: Device for inference ('cpu' or 'cuda')
            
        Raises:
            FileNotFoundError: If checkpoint file not found
            ValueError: If checkpoint is invalid or incompatible
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Set device
        self.device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract hyperparameters
        hyperparams = checkpoint.get('hyperparameters', {})
        node_feat_dim = hyperparams.get('node_feat_dim', 133)
        edge_feat_dim = hyperparams.get('edge_feat_dim', 7)
        hidden_dim = hyperparams.get('hidden_dim', 256)
        num_layers = hyperparams.get('num_layers', 4)
        num_tasks = hyperparams.get('num_tasks', 12)
        dropout = hyperparams.get('dropout', 0.3)
        
        # Initialize model
        self.model = ToxGNN(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_tasks=num_tasks,
            dropout=dropout
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        self.model.eval()
        
        self.num_tasks = num_tasks
    
    @torch.no_grad()
    def predict(self, graph: Data) -> np.ndarray:
        """
        Predict toxicity logits for a molecular graph.
        
        This method:
        1. Moves the graph to the appropriate device (CPU/GPU)
        2. Runs forward pass through the GNN
        3. Returns logits as numpy array
        
        Args:
            graph: PyTorch Geometric Data object representing molecular graph
        
        Returns:
            Logits array of shape (12,) for 12 Tox21 assays
            
        Examples:
            >>> graph = mol_to_graph(mol)
            >>> logits = model.predict(graph)
            >>> probs = 1 / (1 + np.exp(-logits))  # Convert to probabilities
        """
        # Move graph to device
        graph = graph.to(self.device)
        
        # Ensure batch attribute exists (required for AttentiveFP)
        if not hasattr(graph, 'batch') or graph.batch is None:
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)
        
        # Forward pass
        logits = self.model(graph)
        
        # Return as numpy array
        return logits.cpu().numpy()[0]
