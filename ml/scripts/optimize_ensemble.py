"""
Ensemble weight optimization script.

This script optimizes the weights for combining ChemBERTa-2, GNN, and LightGBM
predictions using Nelder-Mead optimization on the validation set.
"""

import os
import sys
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.models.ensemble import EnsembleModel, logit_fusion, logits_to_probs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def softmax(x: np.ndarray) -> np.ndarray:
    """Apply softmax to ensure weights sum to 1."""
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / exp_x.sum()


def load_individual_predictions(
    data_path: str,
    chemberta_path: str,
    gnn_path: str,
    lgbm_path: str,
    split: str = 'val'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load predictions from all three models for a given split.
    
    Args:
        data_path: Path to processed data
        chemberta_path: Path to ChemBERTa model
        gnn_path: Path to GNN checkpoint
        lgbm_path: Path to LightGBM artifacts
        split: 'train', 'val', or 'test'
    
    Returns:
        Tuple of (predictions, labels)
            predictions: (num_samples, 3, num_tasks) - logits from each model
            labels: (num_samples, num_tasks) - ground truth labels
    """
    logger.info(f"Loading {split} predictions from individual models...")
    
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get split indices
    if split == 'train':
        indices = data['train_idx']
    elif split == 'val':
        indices = data['val_idx']
    elif split == 'test':
        indices = data['test_idx']
    else:
        raise ValueError(f"Invalid split: {split}")
    
    labels = data['labels'][indices]
    num_samples = len(indices)
    num_tasks = labels.shape[1]
    
    # Initialize predictions array
    predictions = np.zeros((num_samples, 3, num_tasks))
    
    # Load ensemble model (we'll use it to get individual predictions)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create temporary weights file (we'll use equal weights for now)
    temp_weights_path = "ml/artifacts/temp_weights.json"
    with open(temp_weights_path, 'w') as f:
        json.dump({'weights': [1.0, 1.0, 1.0]}, f)
    
    try:
        ensemble = EnsembleModel(
            chemberta_path=chemberta_path,
            gnn_path=gnn_path,
            lgbm_path=lgbm_path,
            weights_path=temp_weights_path,
            device=device
        )
        
        # Get predictions for each sample
        for i, idx in enumerate(tqdm(indices, desc=f"Computing {split} predictions")):
            smiles = data['canonical_smiles'][idx]
            graph = data['graphs'][idx]
            
            # Concatenate features (descriptors + morgan + maccs)
            features = np.concatenate([
                data['descriptors'][idx],
                data['morgan_fp'][idx],
                data['maccs_fp'][idx]
            ])
            
            # Get individual model logits
            chemberta_logits = ensemble.predict_chemberta(smiles)
            gnn_logits = ensemble.predict_gnn(graph)
            lgbm_logits = ensemble.predict_lgbm(features)
            
            predictions[i, 0, :] = chemberta_logits
            predictions[i, 1, :] = gnn_logits
            predictions[i, 2, :] = lgbm_logits
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_weights_path):
            os.remove(temp_weights_path)
    
    return predictions, labels


def compute_mean_auroc(
    predictions: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Compute mean AUROC for ensemble with given weights.
    
    Args:
        predictions: (num_samples, 3, num_tasks) - logits from each model
        labels: (num_samples, num_tasks) - ground truth labels
        weights: (3,) - weights for each model
    
    Returns:
        Mean AUROC across all tasks
    """
    # Apply softmax to weights
    weights_norm = softmax(weights)
    
    # Fuse predictions
    num_samples, num_models, num_tasks = predictions.shape
    fused_logits = np.zeros((num_samples, num_tasks))
    
    for i in range(num_samples):
        fused_logits[i] = logit_fusion(predictions[i], weights_norm)
    
    # Convert to probabilities
    fused_probs = logits_to_probs(fused_logits)
    
    # Compute per-task AUROC
    aucs = []
    for task_idx in range(num_tasks):
        # Filter out missing labels
        mask = ~np.isnan(labels[:, task_idx])
        if mask.sum() > 0 and len(np.unique(labels[mask, task_idx])) > 1:
            auc = roc_auc_score(labels[mask, task_idx], fused_probs[mask, task_idx])
            aucs.append(auc)
    
    return np.mean(aucs) if aucs else 0.0


def objective_function(
    weights: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Objective function for optimization (negative mean AUROC).
    
    Args:
        weights: (3,) - weights for each model
        predictions: (num_samples, 3, num_tasks) - logits from each model
        labels: (num_samples, num_tasks) - ground truth labels
    
    Returns:
        Negative mean AUROC (for minimization)
    """
    mean_auc = compute_mean_auroc(predictions, labels, weights)
    return -mean_auc  # Negative because we minimize


def main():
    """Main optimization pipeline."""
    
    # Paths
    data_path = "ml/data/processed/tox21_processed.pkl"
    chemberta_path = "ml/artifacts/chemberta_finetuned"
    gnn_path = "ml/artifacts/gnn_best.pt"
    lgbm_path = "ml/artifacts"
    output_path = "ml/artifacts/ensemble_weights.json"
    
    # Check if all models exist
    if not os.path.exists(chemberta_path):
        logger.error(f"ChemBERTa model not found at {chemberta_path}")
        logger.error("Please train ChemBERTa model first (python ml/scripts/train_chemberta.py)")
        return
    
    if not os.path.exists(gnn_path):
        logger.error(f"GNN model not found at {gnn_path}")
        logger.error("Please train GNN model first (python ml/scripts/train_gnn.py)")
        return
    
    if not os.path.exists(f"{lgbm_path}/lgbm_metadata.json"):
        logger.error(f"LightGBM models not found at {lgbm_path}")
        logger.error("Please train LightGBM models first (python ml/scripts/train_lgbm.py)")
        return
    
    # Load validation predictions
    val_predictions, val_labels = load_individual_predictions(
        data_path, chemberta_path, gnn_path, lgbm_path, split='val'
    )
    
    logger.info(f"Validation predictions shape: {val_predictions.shape}")
    logger.info(f"Validation labels shape: {val_labels.shape}")
    
    # Compute baseline performance (equal weights)
    equal_weights = np.array([1.0, 1.0, 1.0])
    baseline_auc = compute_mean_auroc(val_predictions, val_labels, equal_weights)
    logger.info(f"\nBaseline (equal weights): Val AUC = {baseline_auc:.4f}")
    
    # Compute individual model performance
    logger.info("\nIndividual model performance:")
    for i, model_name in enumerate(['ChemBERTa', 'GNN', 'LightGBM']):
        weights = np.zeros(3)
        weights[i] = 1.0
        auc = compute_mean_auroc(val_predictions, val_labels, weights)
        logger.info(f"  {model_name}: Val AUC = {auc:.4f}")
    
    # Optimize weights using Nelder-Mead
    logger.info("\nOptimizing ensemble weights...")
    
    initial_weights = np.array([1.0, 1.0, 1.0])
    
    result = minimize(
        objective_function,
        initial_weights,
        args=(val_predictions, val_labels),
        method='Nelder-Mead',
        options={'maxiter': 100, 'disp': True}
    )
    
    # Get optimized weights
    optimized_weights_raw = result.x
    optimized_weights = softmax(optimized_weights_raw)
    
    logger.info(f"\nOptimization complete!")
    logger.info(f"Optimized weights (raw): {optimized_weights_raw}")
    logger.info(f"Optimized weights (normalized): {optimized_weights}")
    logger.info(f"  ChemBERTa: {optimized_weights[0]:.4f}")
    logger.info(f"  GNN: {optimized_weights[1]:.4f}")
    logger.info(f"  LightGBM: {optimized_weights[2]:.4f}")
    
    # Compute optimized performance
    optimized_auc = compute_mean_auroc(val_predictions, val_labels, optimized_weights_raw)
    logger.info(f"\nOptimized ensemble: Val AUC = {optimized_auc:.4f}")
    logger.info(f"Improvement over baseline: {optimized_auc - baseline_auc:+.4f}")
    
    # Save weights
    weights_data = {
        'weights': optimized_weights.tolist(),
        'weights_raw': optimized_weights_raw.tolist(),
        'val_auc': float(optimized_auc),
        'baseline_auc': float(baseline_auc),
        'improvement': float(optimized_auc - baseline_auc),
        'model_order': ['ChemBERTa', 'GNN', 'LightGBM']
    }
    
    with open(output_path, 'w') as f:
        json.dump(weights_data, f, indent=2)
    
    logger.info(f"\nWeights saved to: {output_path}")
    
    # Evaluate on test set
    logger.info("\nEvaluating optimized ensemble on test set...")
    test_predictions, test_labels = load_individual_predictions(
        data_path, chemberta_path, gnn_path, lgbm_path, split='test'
    )
    
    test_auc = compute_mean_auroc(test_predictions, test_labels, optimized_weights_raw)
    logger.info(f"Test AUC: {test_auc:.4f}")
    
    # Check if targets met
    if test_auc >= 0.80:
        logger.info(f"✓ Mean AUROC {test_auc:.4f} meets target (≥0.80)")
    else:
        logger.warning(f"✗ Mean AUROC {test_auc:.4f} below target (≥0.80)")
    
    # Check improvement over best individual model
    best_individual_auc = max([
        compute_mean_auroc(test_predictions, test_labels, np.array([1.0, 0.0, 0.0])),
        compute_mean_auroc(test_predictions, test_labels, np.array([0.0, 1.0, 0.0])),
        compute_mean_auroc(test_predictions, test_labels, np.array([0.0, 0.0, 1.0]))
    ])
    
    improvement = test_auc - best_individual_auc
    logger.info(f"Best individual model: {best_individual_auc:.4f}")
    logger.info(f"Ensemble improvement: {improvement:+.4f}")
    
    if improvement >= 0.02:
        logger.info(f"✓ Improvement {improvement:.4f} meets target (≥0.02)")
    else:
        logger.warning(f"✗ Improvement {improvement:.4f} below target (≥0.02)")
    
    # Update weights file with test results
    weights_data['test_auc'] = float(test_auc)
    weights_data['best_individual_auc'] = float(best_individual_auc)
    weights_data['test_improvement'] = float(improvement)
    
    with open(output_path, 'w') as f:
        json.dump(weights_data, f, indent=2)
    
    logger.info(f"\nOptimization complete! Final weights saved to: {output_path}")


if __name__ == "__main__":
    main()
