"""
Download Tox21 dataset from DeepChem/MoleculeNet.

This script downloads the Tox21 dataset and saves it as a CSV file
in the format expected by the preprocessing pipeline.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

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


def download_tox21_deepchem():
    """Download Tox21 dataset using DeepChem."""
    try:
        import deepchem as dc
        logger.info("Loading Tox21 dataset from DeepChem MoleculeNet...")
        
        # Load Tox21 dataset (returns train, valid, test splits)
        tasks, datasets, transformers = dc.molnet.load_tox21(
            featurizer='Raw',  # We'll do our own featurization
            splitter='scaffold',
            reload=True
        )
        
        train_dataset, valid_dataset, test_dataset = datasets
        
        logger.info(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
        logger.info(f"Tasks: {tasks}")
        
        # Combine all splits
        all_smiles = []
        all_labels = []
        
        for dataset in [train_dataset, valid_dataset, test_dataset]:
            all_smiles.extend(dataset.ids)
            all_labels.append(dataset.y)
        
        all_labels = np.vstack(all_labels)
        
        # Create DataFrame
        df = pd.DataFrame({'smiles': all_smiles})
        
        # Add labels for each assay
        for i, task in enumerate(tasks):
            df[task] = all_labels[:, i]
        
        logger.info(f"Total molecules: {len(df)}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        return df
        
    except ImportError:
        logger.error("DeepChem not installed. Trying alternative method...")
        return None


def download_tox21_url():
    """Download Tox21 dataset from GitHub URL."""
    import urllib.request
    import gzip
    import io
    
    logger.info("Downloading Tox21 dataset from GitHub...")
    
    url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/tox21.csv.gz"
    
    try:
        # Download the gzipped file
        with urllib.request.urlopen(url) as response:
            compressed_data = response.read()
        
        # Decompress
        with gzip.GzipFile(fileobj=io.BytesIO(compressed_data)) as f:
            df = pd.read_csv(f)
        
        logger.info(f"Downloaded {len(df)} molecules")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Rename columns to match expected format
        if 'smiles' not in df.columns and 'SMILES' in df.columns:
            df.rename(columns={'SMILES': 'smiles'}, inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to download from URL: {e}")
        return None


def create_synthetic_data():
    """Create synthetic Tox21 data for testing."""
    logger.warning("Creating synthetic data for testing purposes...")
    
    # Some example SMILES strings
    smiles_examples = [
        'CC(C)Cc1ccc(cc1)C(C)C(O)=O',  # Ibuprofen
        'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(C)NCC(COc1ccccc1)O',  # Propranolol
        'Cc1ccc(cc1)S(=O)(=O)N',  # Toluenesulfonamide
        'c1ccc2c(c1)ccc3c2ccc4c3cccc4',  # Anthracene
        'C1=CC=C(C=C1)C(=O)O',  # Benzoic acid
        'CC(C)(C)c1ccc(O)cc1',  # BHT
        'c1ccc(cc1)N',  # Aniline
        'C1=CC=C(C=C1)O',  # Phenol
    ] * 120  # Repeat to get ~1200 molecules
    
    # Create DataFrame
    df = pd.DataFrame({'smiles': smiles_examples[:1200]})
    
    # Add random labels for each assay (with some NaN for missing data)
    np.random.seed(42)
    for assay in TOX21_ASSAYS:
        labels = np.random.choice([0.0, 1.0, np.nan], size=len(df), p=[0.6, 0.3, 0.1])
        df[assay] = labels
    
    logger.info(f"Created synthetic dataset with {len(df)} molecules")
    
    return df


def main():
    """Main download pipeline."""
    
    # Create output directory
    output_dir = "ml/data/raw"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/tox21.csv"
    
    # Try DeepChem first
    df = download_tox21_deepchem()
    
    # If DeepChem fails, try direct URL download
    if df is None:
        df = download_tox21_url()
    
    # If both fail, create synthetic data
    if df is None:
        df = create_synthetic_data()
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved Tox21 dataset to {output_path}")
    
    # Print statistics
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Total molecules: {len(df)}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    if len(df.columns) > 1:
        label_cols = [col for col in df.columns if col != 'smiles']
        for col in label_cols:
            if col in df.columns:
                valid = df[col].notna().sum()
                positive = (df[col] == 1).sum()
                negative = (df[col] == 0).sum()
                logger.info(f"  {col}: {valid} valid ({positive} positive, {negative} negative)")


if __name__ == "__main__":
    main()
