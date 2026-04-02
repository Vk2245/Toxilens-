"""
Molecular fingerprint computation module.

This module computes molecular fingerprints (Morgan/ECFP4 and MACCS keys)
for use as features in the LightGBM toxicity prediction model. Fingerprints
provide binary structural representations that capture molecular similarity.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys


def compute_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Compute Morgan fingerprint (ECFP4) for a molecule.
    
    Morgan fingerprints are circular fingerprints that encode structural features
    by iteratively aggregating atom neighborhoods. With radius=2, this produces
    ECFP4 (Extended Connectivity Fingerprint with diameter 4).
    
    Args:
        mol: RDKit molecule object
        radius: Radius for Morgan fingerprint (default: 2 for ECFP4)
        n_bits: Number of bits in the fingerprint vector (default: 2048)
        
    Returns:
        Binary fingerprint as numpy array of shape (n_bits,)
        
    Raises:
        ValueError: If molecule is None or invalid
        
    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> fp = compute_morgan_fingerprint(mol)
        >>> fp.shape
        (2048,)
        >>> fp.dtype
        dtype('int64')
    """
    if mol is None:
        raise ValueError("Molecule cannot be None")
    
    # Generate Morgan fingerprint as bit vector
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    
    # Convert to numpy array
    arr = np.zeros((n_bits,), dtype=np.int64)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    
    return arr


def compute_maccs_keys(mol: Chem.Mol) -> np.ndarray:
    """
    Compute MACCS keys for a molecule.
    
    MACCS (Molecular ACCess System) keys are 167 predefined structural keys
    representing the presence or absence of specific substructures. These
    keys are widely used in pharmaceutical research for similarity searching.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Binary fingerprint as numpy array of shape (167,)
        
    Raises:
        ValueError: If molecule is None or invalid
        
    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> maccs = compute_maccs_keys(mol)
        >>> maccs.shape
        (167,)
        >>> maccs.dtype
        dtype('int64')
    """
    if mol is None:
        raise ValueError("Molecule cannot be None")
    
    # Generate MACCS keys
    fp = MACCSkeys.GenMACCSKeys(mol)
    
    # Convert to numpy array
    arr = np.zeros((167,), dtype=np.int64)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    
    return arr
