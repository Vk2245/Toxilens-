"""
RDKit utilities for SMILES validation, standardization, and molecular operations.

This module provides core cheminformatics functionality for the ToxiLens platform,
including SMILES parsing, molecular standardization, and 2D structure visualization.
"""

from typing import Optional, Tuple
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import Draw


def validate_smiles(smiles: str) -> bool:
    """
    Validate SMILES syntax using RDKit parser.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if SMILES is valid, False otherwise
        
    Examples:
        >>> validate_smiles("CCO")
        True
        >>> validate_smiles("invalid")
        False
    """
    if not smiles or not isinstance(smiles, str):
        return False
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def standardize_smiles(smiles: str) -> str:
    """
    Standardize SMILES string with charge neutralization, salt removal, 
    and tautomer canonicalization.
    
    This function applies the following transformations:
    1. Parse SMILES to molecule
    2. Remove salts and solvents
    3. Neutralize charges
    4. Canonicalize tautomers
    5. Generate canonical SMILES
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Standardized canonical SMILES string
        
    Raises:
        ValueError: If SMILES is invalid or cannot be parsed
        
    Examples:
        >>> standardize_smiles("CCO")
        'CCO'
        >>> standardize_smiles("[Na+].[Cl-]")
        '[Na+].[Cl-]'
    """
    if not smiles or not isinstance(smiles, str):
        raise ValueError("SMILES string cannot be empty")
    
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: unable to parse molecular structure")
    
    # Remove salts and solvents
    remover = SaltRemover.SaltRemover()
    mol = remover.StripMol(mol)
    
    # Neutralize charges
    uncharger = rdMolStandardize.Uncharger()
    mol = uncharger.uncharge(mol)
    
    # Canonicalize tautomers
    enumerator = rdMolStandardize.TautomerEnumerator()
    mol = enumerator.Canonicalize(mol)
    
    # Generate canonical SMILES
    canonical_smiles = Chem.MolToSmiles(mol)
    
    return canonical_smiles


def smiles_to_mol(smiles: str) -> Chem.Mol:
    """
    Parse SMILES string to RDKit molecule object with error handling.
    
    Args:
        smiles: SMILES string to parse
        
    Returns:
        RDKit Mol object
        
    Raises:
        ValueError: If SMILES is invalid or cannot be parsed
        
    Examples:
        >>> mol = smiles_to_mol("CCO")
        >>> mol.GetNumAtoms()
        3
    """
    if not smiles or not isinstance(smiles, str):
        raise ValueError("SMILES string cannot be empty")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: unable to parse molecular structure")
    
    return mol


def generate_2d_image(mol: Chem.Mol, size: Tuple[int, int] = (400, 400)) -> bytes:
    """
    Generate 2D molecular structure image in PNG format.
    
    Args:
        mol: RDKit molecule object
        size: Image dimensions as (width, height) tuple
        
    Returns:
        PNG image as bytes
        
    Raises:
        ValueError: If molecule is None or invalid
        
    Examples:
        >>> mol = smiles_to_mol("CCO")
        >>> png_bytes = generate_2d_image(mol)
        >>> len(png_bytes) > 0
        True
    """
    if mol is None:
        raise ValueError("Molecule cannot be None")
    
    # Generate 2D coordinates if not present
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    
    # Draw molecule to PNG
    img = Draw.MolToImage(mol, size=size)
    
    # Convert PIL Image to bytes
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    png_bytes = buffer.getvalue()
    
    return png_bytes
