"""
GNN model training script for Tox21 toxicity prediction.

This script trains a multi-task AttentiveFP GNN with joint correlation loss
to leverage molecular graph structure and cross-assay relationships.
"""

import os
import sys
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.models.gnn import ToxGNN, joint_correlation_loss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_processed_data(data_path: str) -> Dict:
    """Load preprocessed Tox21 data."""
    logger.info(f"Loading processed data from {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded data with {len(data['smiles'])} molecules")
    logger.info(f"Train: {len(data['train_idx'])}, Val: {len(data['val_idx'])}, Test: {len(data['test_idx'])}")
    
    return data


def create_dataloaders(
    graphs: List[Data],
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int = 32
) -> tuple:
    """
    Create PyTorch Geometric DataLoaders for train/val/test splits.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info("Creating DataLoaders...")
    
    # Create dataset with labels attached to each graph
    train_data = []
    for idx in train_idx:
        graph = graphs[idx].clone()
        # Ensure labels are 2D: (1, num_tasks)
        graph.y = torch.tensor(labels[idx], dtype=torch.float32).unsqueeze(0)
        train_data.append(graph)
    
    val_data = []
    for idx in val_idx:
        graph = graphs[idx].clone()
        graph.y = torch.tensor(labels[idx], dtype=torch.float32).unsqueeze(0)
        val_data.append(graph)
    
    test_data = []
    for idx in test_idx:
        graph = graphs[idx].clone()
        graph.y = torch.tensor(labels[idx], dtype=torch.float32).unsqueeze(0)
        test_data.append(graph)
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def train_epoch(
    model: ToxGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    target_correlation: torch.Tensor,
    device: torch.device,
    lambda_corr: float = 0.1
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_bce = 0.0
    total_corr = 0.0
    num_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        # Reshape labels from (batch_size*1, num_tasks) to (batch_size, num_tasks)
        batch_size = batch.batch.max().item() + 1
        labels = batch.y.view(batch_size, -1)
        
        # Forward pass
        logits = model(batch)
        
        # Compute loss
        loss, bce_loss, corr_loss = joint_correlation_loss(
            logits,
            labels,
            target_correlation,
            lambda_corr=lambda_corr
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_bce += bce_loss.item()
        total_corr += corr_loss.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'bce_loss': total_bce / num_batches,
        'corr_loss': total_corr / num_batches
    }


@torch.no_grad()
def evaluate(
    model: ToxGNN,
    loader: DataLoader,
    target_correlation: torch.Tensor,
    device: torch.device,
    lambda_corr: float = 0.1
) -> Dict:
    """Evaluate model on validation/test set."""
    model.eval()
    
    all_logits = []
    all_labels = []
    total_loss = 0.0
    total_bce = 0.0
    total_corr = 0.0
    num_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        # Reshape labels from (batch_size*1, num_tasks) to (batch_size, num_tasks)
        batch_size = batch.batch.max().item() + 1
        labels = batch.y.view(batch_size, -1)
        
        # Forward pass
        logits = model(batch)
        
        # Compute loss
        loss, bce_loss, corr_loss = joint_correlation_loss(
            logits,
            labels,
            target_correlation,
            lambda_corr=lambda_corr
        )
        
        # Accumulate
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        total_loss += loss.item()
        total_bce += bce_loss.item()
        total_corr += corr_loss.item()
        num_batches += 1
    
    # Concatenate all predictions
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute probabilities
    all_probs = torch.sigmoid(all_logits).numpy()
    all_labels_np = all_labels.numpy()
    
    # Compute per-assay AUROC
    num_tasks = all_probs.shape[1]
    aucs = []
    
    for i in range(num_tasks):
        # Filter out missing labels
        mask = ~np.isnan(all_labels_np[:, i])
        if mask.sum() > 0 and len(np.unique(all_labels_np[mask, i])) > 1:
            auc = roc_auc_score(all_labels_np[mask, i], all_probs[mask, i])
            aucs.append(auc)
        else:
            aucs.append(np.nan)
    
    mean_auc = np.nanmean(aucs)
    
    return {
        'loss': total_loss / num_batches,
        'bce_loss': total_bce / num_batches,
        'corr_loss': total_corr / num_batches,
        'mean_auc': mean_auc,
        'per_assay_auc': aucs
    }


def main():
    """Main training pipeline."""
    
    # Hyperparameters
    HIDDEN_DIM = 256
    NUM_LAYERS = 4
    NUM_TASKS = 12
    DROPOUT = 0.3
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    PATIENCE = 15
    LAMBDA_CORR = 0.1
    T_MAX = 50
    
    # Paths
    data_path = "ml/data/processed/tox21_processed.pkl"
    artifacts_dir = "ml/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    data = load_processed_data(data_path)
    
    # Get target correlation matrix
    target_correlation = torch.tensor(data['label_correlation'], dtype=torch.float32).to(device)
    logger.info(f"Target correlation matrix shape: {target_correlation.shape}")
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data['graphs'],
        data['labels'],
        data['train_idx'],
        data['val_idx'],
        data['test_idx'],
        batch_size=BATCH_SIZE
    )
    
    # Get actual feature dimensions from data
    sample_graph = data['graphs'][0]
    node_feat_dim = sample_graph.x.shape[1]
    edge_feat_dim = sample_graph.edge_attr.shape[1]
    
    logger.info(f"Node feature dim: {node_feat_dim}, Edge feature dim: {edge_feat_dim}")
    
    # Initialize model
    model = ToxGNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_tasks=NUM_TASKS,
        dropout=DROPOUT
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX)
    
    # Training loop
    best_val_auc = 0.0
    patience_counter = 0
    
    logger.info(f"\nStarting training for up to {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, target_correlation, device, LAMBDA_CORR
        )
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, target_correlation, device, LAMBDA_CORR
        )
        
        # Step scheduler
        scheduler.step()
        
        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_metrics['loss']:.4f} (BCE: {train_metrics['bce_loss']:.4f}, Corr: {train_metrics['corr_loss']:.4f}) | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val AUC: {val_metrics['mean_auc']:.4f}"
        )
        
        # Early stopping
        if val_metrics['mean_auc'] > best_val_auc:
            best_val_auc = val_metrics['mean_auc']
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'hyperparameters': {
                    'node_feat_dim': node_feat_dim,
                    'edge_feat_dim': edge_feat_dim,
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'num_tasks': NUM_TASKS,
                    'dropout': DROPOUT,
                    'lambda_corr': LAMBDA_CORR
                }
            }, f"{artifacts_dir}/gnn_best.pt")
            
            logger.info(f"  → New best model saved (Val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model for final evaluation
    logger.info("\nLoading best model for final evaluation...")
    checkpoint = torch.load(f"{artifacts_dir}/gnn_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_metrics = evaluate(
        model, test_loader, target_correlation, device, LAMBDA_CORR
    )
    
    logger.info("\n" + "="*60)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("="*60)
    
    assay_names = data.get('assay_names', [f'Assay_{i}' for i in range(NUM_TASKS)])
    for i, (assay_name, auc) in enumerate(zip(assay_names, test_metrics['per_assay_auc'])):
        if not np.isnan(auc):
            logger.info(f"  {assay_name}: Test AUROC = {auc:.4f}")
    
    logger.info(f"\nMean Test AUROC: {test_metrics['mean_auc']:.4f}")
    
    # Check if target met
    if test_metrics['mean_auc'] >= 0.80:
        logger.info(f"✓ Mean AUROC {test_metrics['mean_auc']:.4f} meets target (≥0.80)")
    else:
        logger.warning(f"✗ Mean AUROC {test_metrics['mean_auc']:.4f} below target (≥0.80)")
    
    # Save metadata
    metadata = {
        'assay_names': assay_names,
        'test_results': {name: auc for name, auc in zip(assay_names, test_metrics['per_assay_auc']) if not np.isnan(auc)},
        'mean_test_auc': float(test_metrics['mean_auc']),
        'best_val_auc': float(best_val_auc),
        'best_epoch': int(checkpoint['epoch']),
        'hyperparameters': checkpoint['hyperparameters'],
        'n_train': len(data['train_idx']),
        'n_val': len(data['val_idx']),
        'n_test': len(data['test_idx'])
    }
    
    with open(f"{artifacts_dir}/gnn_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\nTraining complete! Model saved to: {artifacts_dir}/gnn_best.pt")


if __name__ == "__main__":
    main()
