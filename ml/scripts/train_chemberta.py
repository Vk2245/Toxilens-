"""
ChemBERTa-2 fine-tuning script for Tox21 toxicity prediction.

This script fine-tunes the pretrained ChemBERTa-zinc-base-v1 model
with a 12-class multi-label classification head for Tox21 assays.
"""

import os
import sys
import logging
import pickle
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Tox21Dataset(Dataset):
    """Dataset for Tox21 SMILES and labels."""
    
    def __init__(self, smiles: list, labels: np.ndarray, tokenizer, max_length: int = 512):
        self.smiles = smiles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        labels = self.labels[idx]
        
        # Tokenize SMILES
        encoding = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert labels to tensor, replace NaN with -1 for masking
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        labels_tensor = torch.where(
            torch.isnan(labels_tensor),
            torch.tensor(-1.0),
            labels_tensor
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels_tensor
        }


class ChemBERTaForMultiLabelClassification(nn.Module):
    """ChemBERTa model with multi-label classification head."""
    
    def __init__(self, model_name: str, num_labels: int = 12, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute BCE loss with masking for missing labels.
    
    Args:
        logits: Model predictions (batch_size, num_labels)
        labels: Ground truth with -1 for missing labels (batch_size, num_labels)
    
    Returns:
        Masked BCE loss
    """
    # Create mask (1 for valid labels, 0 for missing)
    mask = (labels != -1).float()
    
    # Replace -1 with 0 for computation
    labels_clean = torch.where(labels == -1, torch.zeros_like(labels), labels)
    
    # Compute BCE loss
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels_clean,
        reduction='none'
    )
    
    # Apply mask and compute mean
    loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
    
    return loss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler
) -> float:
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(input_ids, attention_mask)
            loss = masked_bce_loss(logits, labels)
        
        # Backward pass with gradient scaling
        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict:
    """Evaluate model on validation/test set."""
    model.eval()
    
    all_logits = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = masked_bce_loss(logits, labels)
        
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        total_loss += loss.item()
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
        # Filter out missing labels (-1)
        mask = all_labels_np[:, i] != -1
        if mask.sum() > 0 and len(np.unique(all_labels_np[mask, i])) > 1:
            auc = roc_auc_score(all_labels_np[mask, i], all_probs[mask, i])
            aucs.append(auc)
        else:
            aucs.append(np.nan)
    
    mean_auc = np.nanmean(aucs)
    
    return {
        'loss': total_loss / num_batches,
        'mean_auc': mean_auc,
        'per_assay_auc': aucs
    }


def main():
    """Main training pipeline."""
    
    # Hyperparameters
    MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
    NUM_LABELS = 12
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 8
    WARMUP_RATIO = 0.1
    MAX_LENGTH = 512
    PATIENCE = 3
    
    # Paths
    data_path = "ml/data/processed/tox21_processed.pkl"
    artifacts_dir = "ml/artifacts"
    output_dir = f"{artifacts_dir}/chemberta_finetuned"
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading processed data from {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded data with {len(data['smiles'])} molecules")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']
    
    train_smiles = [data['canonical_smiles'][i] for i in train_idx]
    val_smiles = [data['canonical_smiles'][i] for i in val_idx]
    test_smiles = [data['canonical_smiles'][i] for i in test_idx]
    
    train_labels = data['labels'][train_idx]
    val_labels = data['labels'][val_idx]
    test_labels = data['labels'][test_idx]
    
    train_dataset = Tox21Dataset(train_smiles, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = Tox21Dataset(val_smiles, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = Tox21Dataset(test_smiles, test_labels, tokenizer, MAX_LENGTH)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize model
    logger.info(f"Loading pretrained model from {MODEL_NAME}")
    model = ChemBERTaForMultiLabelClassification(MODEL_NAME, NUM_LABELS).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # Training loop
    best_val_auc = 0.0
    patience_counter = 0
    
    logger.info(f"\nStarting training for up to {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val AUC: {val_metrics['mean_auc']:.4f}"
        )
        
        # Early stopping
        if val_metrics['mean_auc'] > best_val_auc:
            best_val_auc = val_metrics['mean_auc']
            patience_counter = 0
            
            # Save best model
            model.encoder.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            torch.save({
                'epoch': epoch,
                'classifier_state_dict': model.classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
            }, f"{output_dir}/classifier.pt")
            
            logger.info(f"  → New best model saved (Val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model for final evaluation
    logger.info("\nLoading best model for final evaluation...")
    model.encoder = AutoModel.from_pretrained(output_dir).to(device)
    checkpoint = torch.load(f"{output_dir}/classifier.pt", map_location=device, weights_only=False)
    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, device)
    
    logger.info("\n" + "="*60)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("="*60)
    
    assay_names = data.get('assay_names', [f'Assay_{i}' for i in range(NUM_LABELS)])
    for i, (assay_name, auc) in enumerate(zip(assay_names, test_metrics['per_assay_auc'])):
        if not np.isnan(auc):
            logger.info(f"  {assay_name}: Test AUROC = {auc:.4f}")
    
    logger.info(f"\nMean Test AUROC: {test_metrics['mean_auc']:.4f}")
    
    # Check if target met
    if test_metrics['mean_auc'] >= 0.78:
        logger.info(f"✓ Mean AUROC {test_metrics['mean_auc']:.4f} meets target (≥0.78)")
    else:
        logger.warning(f"✗ Mean AUROC {test_metrics['mean_auc']:.4f} below target (≥0.78)")
    
    # Save metadata
    metadata = {
        'model_name': MODEL_NAME,
        'assay_names': assay_names,
        'test_results': {name: auc for name, auc in zip(assay_names, test_metrics['per_assay_auc']) if not np.isnan(auc)},
        'mean_test_auc': float(test_metrics['mean_auc']),
        'best_val_auc': float(best_val_auc),
        'best_epoch': int(checkpoint['epoch']),
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'warmup_ratio': WARMUP_RATIO,
            'max_length': MAX_LENGTH
        },
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx)
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\nTraining complete! Model saved to: {output_dir}/")


if __name__ == "__main__":
    main()
