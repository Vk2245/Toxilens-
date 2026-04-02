"""
Standalone Tox21 dataset preprocessing script.

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
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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


def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def standardize_smiles(smiles: str) -> str:
    """Standardize SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Remove salts, neutralize charges
    from rdkit.Chem.MolStandardize import rdMolStandardize
    clean_mol = rdMolStandardize.Cleanup(mol)
    
    # Canonicalize
    canonical_smiles = Chem.MolToSmiles(clean_mol, canonical=True)
    return canonical_smiles


def compute_descriptors(mol: Chem.Mol) -> np.ndarray:
    """Compute molecular descriptors."""
    desc_list = []
    
    # Basic descriptors
    desc_list.append(Descriptors.MolWt(mol))
    desc_list.append(Descriptors.MolLogP(mol))
    desc_list.append(Descriptors.TPSA(mol))
    desc_list.append(Descriptors.NumHDonors(mol))
    desc_list.append(Descriptors.NumHAcceptors(mol))
    desc_list.append(Descriptors.NumRotatableBonds(mol))
    desc_list.append(Descriptors.NumAromaticRings(mol))
    desc_list.append(Descriptors.NumSaturatedRings(mol))
    desc_list.append(Descriptors.FractionCSP3(mol))
    desc_list.append(Descriptors.NumAliphaticRings(mol))
    
    # Add more descriptors to reach ~200
    desc_names = [name for name, _ in Descriptors.descList]
    for name in desc_names[:200]:  # Take first 200 descriptors
        try:
            func = getattr(Descriptors, name)
            value = func(mol)
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                desc_list.append(value)
        except:
            desc_list.append(0.0)
    
    # Pad or truncate to exactly 200
    if len(desc_list) < 200:
        desc_list.extend([0.0] * (200 - len(desc_list)))
    else:
        desc_list = desc_list[:200]
    
    return np.array(desc_list, dtype=np.float32)


def compute_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprint."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def compute_maccs_keys(mol: Chem.Mol) -> np.ndarray:
    """Compute MACCS keys."""
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.float32)


def mol_to_graph_simple(mol: Chem.Mol) -> Dict:
    """
    Convert molecule to simple graph representation (without PyTorch).
    Returns dict with node features, edge indices, and edge features.
    """
    # Node features
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
        ]
        node_features.append(features)
    
    # Edge features
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        bond_type = bond.GetBondTypeAsDouble()
        is_conjugated = int(bond.GetIsConjugated())
        is_in_ring = int(bond.IsInRing())
        
        # Add both directions
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        
        edge_feat = [bond_type, is_conjugated, is_in_ring]
        edge_features.append(edge_feat)
        edge_features.append(edge_feat)
    
    return {
        'node_features': np.array(node_features, dtype=np.float32),
        'edge_indices': np.array(edge_indices, dtype=np.int64).T if edge_indices else np.zeros((2, 0), dtype=np.int64),
        'edge_features': np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 3), dtype=np.float32)
    }


def scaffold_split(
    smiles_list: List[str],
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split molecules using Bemis-Murcko scaffold-based splitting."""
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
            scaffolds[f"unique_{idx}"] = [idx]
    
    # Sort scaffolds by size
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
    
    return train_indices, val_indices, test_indices


def compute_label_correlation(labels: np.ndarray) -> np.ndarray:
    """Compute label correlation matrix."""
    logger.info("Computing label correlation matrix...")
    
    # Fill NaN with -1 for correlation computation
    labels_filled = labels.copy()
    labels_filled[np.isnan(labels_filled)] = -1
    
    # Compute correlation
    correlation_matrix = np.corrcoef(labels_filled.T)
    
    # Replace NaN with 0
    correlation_matrix[np.isnan(correlation_matrix)] = 0.0
    
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
    logger.info(f"Loading Tox21 dataset from {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    logger.info(f"Loaded {len(df)} molecules")
    
    # Standardize column name
    if 'SMILES' in df.columns:
        df.rename(columns={'SMILES': 'smiles'}, inplace=True)
    
    # Extract SMILES and labels
    smiles_list = df['smiles'].tolist()
    
    # Find label columns
    label_columns = [col for col in df.columns if col not in ['smiles', 'mol_id']]
    logger.info(f"Label columns: {label_columns}")
    
    labels = df[label_columns].values.astype(float)
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Missing labels: {np.isnan(labels).sum()} / {labels.size} ({np.isnan(labels).mean():.1%})")
    
    # Validate SMILES
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
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failed_count += 1
                continue
            
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            descriptors = compute_descriptors(mol)
            morgan_fp = compute_morgan_fingerprint(mol)
            maccs_fp = compute_maccs_keys(mol)
            graph = mol_to_graph_simple(mol)
            
            processed_data['smiles'].append(smiles)
            processed_data['canonical_smiles'].append(canonical_smiles)
            processed_data['descriptors'].append(descriptors)
            processed_data['morgan_fp'].append(morgan_fp)
            processed_data['maccs_fp'].append(maccs_fp)
            processed_data['graphs'].append(graph)
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
    
    # Compute label correlation
    label_correlation = compute_label_correlation(labels_array)
    
    # Adjust split indices
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


if __name__ == "__main__":
    main()
