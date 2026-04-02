"""
LightGBM model training script for Tox21 toxicity prediction.

This script trains 12 separate LightGBM classifiers (one per assay) using
molecular descriptors and fingerprints. It handles class imbalance with
per-assay weights and missing labels with masking.
"""

import os
import sys
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

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


def prepare_features(data: Dict) -> Tuple[np.ndarray, StandardScaler]:
    """
    Prepare feature matrix by concatenating descriptors and fingerprints.
    
    Returns:
        Tuple of (features, scaler)
    """
    logger.info("Preparing feature matrix...")
    
    # Concatenate descriptors + Morgan + MACCS
    descriptors = data['descriptors']  # (n_samples, 200)
    morgan_fp = data['morgan_fp']      # (n_samples, 2048)
    maccs_fp = data['maccs_fp']        # (n_samples, 167)
    
    features = np.concatenate([descriptors, morgan_fp, maccs_fp], axis=1)
    logger.info(f"Feature matrix shape: {features.shape}")
    
    # Handle inf and very large values
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Clip extreme values
    features = np.clip(features, -1e6, 1e6)
    
    logger.info(f"Cleaned features - min: {features.min():.2f}, max: {features.max():.2f}")
    
    # Fit scaler on training data only
    train_idx = data['train_idx']
    scaler = StandardScaler()
    scaler.fit(features[train_idx])
    
    # Transform all data
    features_scaled = scaler.transform(features)
    
    return features_scaled, scaler


def train_single_assay(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weight: float,
    assay_name: str
) -> lgb.Booster:
    """
    Train a single LightGBM classifier for one assay.
    
    Args:
        X_train: Training features
        y_train: Training labels (with NaN for missing)
        X_val: Validation features
        y_val: Validation labels (with NaN for missing)
        class_weight: Weight for positive class
        assay_name: Name of the assay
        
    Returns:
        Trained LightGBM booster
    """
    # Filter out missing labels
    train_mask = ~np.isnan(y_train)
    val_mask = ~np.isnan(y_val)
    
    X_train_clean = X_train[train_mask]
    y_train_clean = y_train[train_mask]
    X_val_clean = X_val[val_mask]
    y_val_clean = y_val[val_mask]
    
    logger.info(f"  {assay_name}: Train samples={len(y_train_clean)}, Val samples={len(y_val_clean)}")
    
    # Check if we have both classes
    if len(np.unique(y_train_clean)) < 2:
        logger.warning(f"  {assay_name}: Only one class present, skipping")
        return None
    
    # Compute sample weights
    sample_weights = np.ones(len(y_train_clean))
    sample_weights[y_train_clean == 1] = class_weight
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_train_clean,
        label=y_train_clean,
        weight=sample_weights
    )
    
    val_data = lgb.Dataset(
        X_val_clean,
        label=y_val_clean,
        reference=train_data
    )
    
    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    # Train with early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0)  # Suppress iteration logs
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )
    
    # Evaluate
    train_pred = model.predict(X_train_clean)
    val_pred = model.predict(X_val_clean)
    
    train_auc = roc_auc_score(y_train_clean, train_pred)
    val_auc = roc_auc_score(y_val_clean, val_pred)
    
    logger.info(f"  {assay_name}: Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}, Best iteration={model.best_iteration}")
    
    return model


def evaluate_models(
    models: List[lgb.Booster],
    X_test: np.ndarray,
    y_test: np.ndarray,
    assay_names: List[str]
) -> Dict[str, float]:
    """
    Evaluate all models on test set.
    
    Returns:
        Dictionary of assay_name -> AUROC
    """
    logger.info("\nEvaluating on test set...")
    
    results = {}
    aucs = []
    
    for i, (model, assay_name) in enumerate(zip(models, assay_names)):
        if model is None:
            logger.warning(f"  {assay_name}: No model (skipped during training)")
            continue
        
        # Filter missing labels
        test_mask = ~np.isnan(y_test[:, i])
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask, i]  # Fix: index the column correctly
        
        if len(y_test_clean) == 0 or len(np.unique(y_test_clean)) < 2:
            logger.warning(f"  {assay_name}: Insufficient test data")
            continue
        
        # Predict
        y_pred = model.predict(X_test_clean)
        
        # Compute AUROC
        auc = roc_auc_score(y_test_clean, y_pred)
        results[assay_name] = auc
        aucs.append(auc)
        
        logger.info(f"  {assay_name}: Test AUROC = {auc:.4f}")
    
    # Compute mean AUROC
    mean_auc = np.mean(aucs)
    logger.info(f"\nMean Test AUROC: {mean_auc:.4f}")
    
    return results


def main():
    """Main training pipeline."""
    
    # Paths
    data_path = "ml/data/processed/tox21_processed.pkl"
    artifacts_dir = "ml/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Load data
    data = load_processed_data(data_path)
    
    # Prepare features
    features, scaler = prepare_features(data)
    
    # Get splits
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']
    
    X_train = features[train_idx]
    X_val = features[val_idx]
    X_test = features[test_idx]
    
    y_train = data['labels'][train_idx]
    y_val = data['labels'][val_idx]
    y_test = data['labels'][test_idx]
    
    # Get class weights and assay names
    class_weights = data['class_weights']
    assay_names = data.get('assay_names', [f'Assay_{i}' for i in range(y_train.shape[1])])
    
    logger.info(f"\nTraining {len(assay_names)} LightGBM classifiers...")
    
    # Train one model per assay
    models = []
    for i, assay_name in enumerate(tqdm(assay_names, desc="Training assays")):
        model = train_single_assay(
            X_train,
            y_train[:, i],
            X_val,
            y_val[:, i],
            class_weights[i],
            assay_name
        )
        models.append(model)
    
    # Evaluate on test set
    test_results = evaluate_models(models, X_test, y_test, assay_names)
    
    # Check if mean AUROC meets target
    mean_auc = np.mean(list(test_results.values()))
    if mean_auc >= 0.80:
        logger.info(f"✓ Mean AUROC {mean_auc:.4f} meets target (≥0.80)")
    else:
        logger.warning(f"✗ Mean AUROC {mean_auc:.4f} below target (≥0.80)")
    
    # Save models
    logger.info(f"\nSaving models to {artifacts_dir}/")
    
    # Save individual models
    for i, (model, assay_name) in enumerate(zip(models, assay_names)):
        if model is not None:
            model_path = f"{artifacts_dir}/lgbm_{assay_name}.txt"
            model.save_model(model_path)
    
    # Save scaler
    with open(f"{artifacts_dir}/lgbm_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    metadata = {
        'assay_names': assay_names,
        'class_weights': class_weights.tolist(),
        'test_results': test_results,
        'mean_test_auc': mean_auc,
        'feature_dim': features.shape[1],
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx)
    }
    
    with open(f"{artifacts_dir}/lgbm_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Training complete!")
    logger.info(f"Models saved to: {artifacts_dir}/")
    logger.info(f"Mean Test AUROC: {mean_auc:.4f}")


if __name__ == "__main__":
    main()
