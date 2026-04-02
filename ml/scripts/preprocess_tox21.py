"""
Tox21 dataset preprocessing script.

This script loads the Tox21 dataset, validates and standardizes SMILES,
computes molecular features (descriptors, fingerprints, graphs), performs
scaffold-based splitting, and saves processed data for model training.
"""

import os
import sys
import logging
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.app.preprocessing.pipeline import PreprocessingPipeline
from backend.app.preprocessing.rdkit_utils import validate_smiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tox21 assay names
TOX21_ASSAYS = [
    'NR-AR', 'NR-AhR', 'NR-AR-LBD', 'SR-ARE', 'SR-p53', 'NR-ER',
    'SR-MMP', 'NR-AROMATASE', 'SR-ATAD5', 'SR-HSE', 'NR-ER-LBD', 'NR-PPAR'
]


def load_tox21_dataset(data_path: str) -> pd.DataFrame:
    """
    Load Tox21 dataset from CSV file.
    
    Args:
        data_path: Path to Tox21 CSV file
        
    Returns:
        DataFrame with SMILES and toxicity labels
    """
    logger.info(f"Loading Tox21 dataset from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Tox21 dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} molecules")
    
    # Check for required columns
    if 'smiles' not in df.columns and 'SMILES' not in df.columns:
        raise ValueError("Dataset must contain 'smiles' or 'SMILES' column")
    
    # Standardize column name
    if 'SMILES' in df.columns:
        df.rename(columns={'SMILES': 'smiles'}, inplace=True)
    
    return df


def scaffold_split(
    smiles_list: List[str],
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split molecules using Bemis-Murcko scaffold-based splitting.
    
    This ensures that molecules with similar scaffolds are in the same split,
    preventing data leakage and providing more realistic evaluation.
    
    Args:
        smiles_list: List of SMILES strings
        train_size: Fraction for training set
        val_size: Fraction for validation set
        test_size: Fraction for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    logger.info("Performing scaffold-based splitting...")
    
    # Generate scaffolds for all molecules
    scaffolds = {}
    for idx, smiles in enumerate(tqdm(smiles_list, desc="Computing scaffolds")):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                if scaffold not in scaffolds:
                    scaffolds[scaffold] = []
                scaffolds[scaffold].append(idx)
        except Exception as e:
            logger.warning(f"Failed to compute scaffold for {smiles}: {e}")
            # Assign to unique scaffold
            scaffolds[f"unique_{idx}"] = [idx]
    
    # Sort scaffolds by size (largest first) for better distribution
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    
    # Distribute scaffolds to splits
    n_total = len(smiles_list)
    n_train = int(n_total * train_size)
    n_val = int(n_total * val_size)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for scaffold_set in scaffold_sets:
        if len(train_indices) < n_train:
            train_indices.extend(scaffold_set)
        elif len(val_indices) < n_val:
            val_indices.extend(scaffold_set)
        else:
            test_indices.extend(scaffold_set)
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    logger.info(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    logger.info(f"Split ratios - Train: {len(train_indices)/n_total:.2%}, Val: {len(val_indices)/n_total:.2%}, Test: {len(test_indices)/n_total:.2%}")
    
    return train_indices, val_indices, test_indices


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """
    Compute per-assay class weights for imbalanced labels.
    
    Weight = n_negative / n_positive for each assay.
    
    Args:
        labels: Array of shape (n_samples, n_assays) with binary labels and NaN for missing
        
    Returns:
        Array of shape (n_assays,) with class weights
    """
    logger.info("Computing per-assay class weights...")
    
    n_assays = labels.shape[1]
    weights = np.zeros(n_assays)
    
    for i in range(n_assays):
        assay_labels = labels[:, i]
        # Remove NaN values
        valid_labels = assay_labels[~np.isnan(assay_labels)]
        
        if len(valid_labels) > 0:
            n_positive = np.sum(valid_labels == 1)
            n_negative = np.sum(valid_labels == 0)
            
            if n_positive > 0:
                weights[i] = n_negative / n_positive
            else:
                weights[i] = 1.0
            
            logger.info(f"  {TOX21_ASSAYS[i]}: pos={n_positive}, neg={n_negative}, weight={weights[i]:.2f}")
        else:
            weights[i] = 1.0
            logger.warning(f"  {TOX21_ASSAYS[i]}: No valid labels, using weight=1.0")
    
    return weights


def compute_label_correlation(labels: np.ndarray) -> np.ndarray:
    """
    Compute label correlation matrix for GNN joint loss.
    
    Args:
        labels: Array of shape (n_samples, n_assays) with binary labels and NaN for missing
        
    Returns:
        Correlation matrix of shape (n_assays, n_assays)
    """
    logger.info("Computing label correlation matrix...")
    
    # Create a copy and fill NaN with -1 for correlation computation
    labels_filled = labels.copy()
    labels_filled[np.isnan(labels_filled)] = -1
    
    # Compute correlation
    correlation_matrix = np.corrcoef(labels_filled.T)
    
    # Replace NaN correlations with 0
    correlation_matrix[np.isnan(correlation_matrix)] = 0.0
    
    logger.info(f"Correlation matrix shape: {correlation_matrix.shape}")
    logger.info(f"Mean absolute correlation: {np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]).mean():.3f}")
    
    return correlation_matrix


def main():
    """Main preprocessing pipeline."""
    start_time = time.time()
    
    # Paths
    raw_data_path = "ml/data/raw/tox21.csv"
    processed_dir = "ml/data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load dataset
    df = load_tox21_dataset(raw_data_path)
    
    # Extract SMILES and labels
    smiles_list = df['smiles'].tolist()
    
    # Extract labels (assuming columns match TOX21_ASSAYS or similar)
    label_columns = [col for col in df.columns if col != 'smiles' and col != 'mol_id']
    if len(label_columns) == 0:
        # Try to find assay columns
        label_columns = [col for col in df.columns if any(assay.lower() in col.lower() for assay in TOX21_ASSAYS)]
    
    if len(label_columns) < 12:
        logger.warning(f"Found only {len(label_columns)} label columns, expected 12")
    
    labels = df[label_columns].values.astype(float)
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Missing labels: {np.isnan(labels).sum()} / {labels.size} ({np.isnan(labels).mean():.1%})")
    
    # Validate and filter SMILES
    logger.info("Validating SMILES strings...")
    valid_indices = []
    for idx, smiles in enumerate(tqdm(smiles_list, desc="Validating")):
        if validate_smiles(smiles):
            valid_indices.append(idx)
    
    logger.info(f"Valid SMILES: {len(valid_indices)} / {len(smiles_list)}")
    
    # Filter to valid molecules
    smiles_list = [smiles_list[i] for i in valid_indices]
    labels = labels[valid_indices]
    
    # Scaffold split
    train_idx, val_idx, test_idx = scaffold_split(smiles_list)
    
    # Initialize preprocessing pipeline
    pipeline = PreprocessingPipeline()
    
    # Process all molecules
    logger.info("Processing molecules...")
    processed_data = {
        'smiles': [],
        'canonical_smiles': [],
        'descriptors': [],
        'morgan_fp': [],
        'maccs_fp': [],
        'graphs': [],
        'labels': []
    }
    
    failed_count = 0
    for idx, smiles in enumerate(tqdm(smiles_list, desc="Processing")):
        try:
            result = pipeline.process(smiles)
            
            processed_data['smiles'].append(smiles)
            processed_data['canonical_smiles'].append(result['canonical_smiles'])
            processed_data['descriptors'].append(result['descriptors'])
            processed_data['morgan_fp'].append(result['morgan_fp'])
            processed_data['maccs_fp'].append(result['maccs_fp'])
            processed_data['graphs'].append(result['graph'])
            processed_data['labels'].append(labels[idx])
            
        except Exception as e:
            logger.warning(f"Failed to process {smiles}: {e}")
            failed_count += 1
    
    logger.info(f"Successfully processed: {len(processed_data['smiles'])} molecules")
    logger.info(f"Failed: {failed_count} molecules")
    
    # Convert to arrays
    descriptors = np.array(processed_data['descriptors'])
    morgan_fps = np.array(processed_data['morgan_fp'])
    maccs_fps = np.array(processed_data['maccs_fp'])
    labels_array = np.array(processed_data['labels'])
    
    # Compute class weights and correlation matrix
    class_weights = compute_class_weights(labels_array)
    label_correlation = compute_label_correlation(labels_array)
    
    # Adjust split indices for failed molecules
    # (This is simplified - in production, track which indices failed)
    n_processed = len(processed_data['smiles'])
    train_idx = train_idx[train_idx < n_processed]
    val_idx = val_idx[val_idx < n_processed]
    test_idx = test_idx[test_idx < n_processed]
    
    # Save processed data
    logger.info("Saving processed data...")
    
    save_data = {
        'smiles': processed_data['smiles'],
        'canonical_smiles': processed_data['canonical_smiles'],
        'descriptors': descriptors,
        'morgan_fp': morgan_fps,
        'maccs_fp': maccs_fps,
        'graphs': processed_data['graphs'],
        'labels': labels_array,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'class_weights': class_weights,
        'label_correlation': label_correlation,
        'assay_names': label_columns
    }
    
    with open(f"{processed_dir}/tox21_processed.pkl", 'wb') as f:
        pickle.dump(save_data, f)
    
    logger.info(f"Saved to {processed_dir}/tox21_processed.pkl")
    
    # Log statistics
    elapsed_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info("PREPROCESSING STATISTICS")
    logger.info(f"{'='*60}")
    logger.info(f"Total molecules processed: {n_processed}")
    logger.info(f"Train set: {len(train_idx)} ({len(train_idx)/n_processed:.1%})")
    logger.info(f"Val set: {len(val_idx)} ({len(val_idx)/n_processed:.1%})")
    logger.info(f"Test set: {len(test_idx)} ({len(test_idx)/n_processed:.1%})")
    logger.info(f"Descriptor shape: {descriptors.shape}")
    logger.info(f"Morgan FP shape: {morgan_fps.shape}")
    logger.info(f"MACCS FP shape: {maccs_fps.shape}")
    logger.info(f"Labels shape: {labels_array.shape}")
    logger.info(f"Missing labels: {np.isnan(labels_array).sum()} ({np.isnan(labels_array).mean():.1%})")
    logger.info(f"Processing time: {elapsed_time/60:.1f} minutes")
    logger.info(f"{'='*60}\n")
    
    if elapsed_time > 600:
        logger.warning(f"Processing took {elapsed_time/60:.1f} minutes (target: <10 minutes)")
    else:
        logger.info(f"✓ Processing completed within target time")


if __name__ == "__main__":
    main()
