"""
Preprocessing pipeline module integrating all preprocessing components.

This module provides the PreprocessingPipeline class that orchestrates the complete
preprocessing workflow: SMILES validation, standardization, descriptor computation,
fingerprint generation, graph construction, and 2D image generation.
"""

import logging
import time
from typing import Dict, Any

import numpy as np
from rdkit import Chem

from backend.app.preprocessing.rdkit_utils import (
    validate_smiles,
    standardize_smiles,
    smiles_to_mol,
    generate_2d_image
)
from backend.app.preprocessing.descriptors import compute_descriptors
from backend.app.preprocessing.fingerprints import (
    compute_morgan_fingerprint,
    compute_maccs_keys
)
from backend.app.preprocessing.graph_builder import mol_to_graph


logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Integrated preprocessing pipeline for molecular data.
    
    This class orchestrates the complete preprocessing workflow, transforming
    raw SMILES strings into all required representations for model inference:
    - Standardized canonical SMILES
    - RDKit molecule object
    - 200+ molecular descriptors
    - Morgan fingerprints (2048-bit ECFP4)
    - MACCS keys (167-bit)
    - PyTorch Geometric graph representation
    - 2D molecular structure image (PNG)
    
    The pipeline handles errors gracefully and logs processing times for
    performance monitoring.
    
    Examples:
        >>> pipeline = PreprocessingPipeline()
        >>> result = pipeline.process("CCO")
        >>> result.keys()
        dict_keys(['mol', 'canonical_smiles', 'descriptors', 'morgan_fp', 
                   'maccs_fp', 'graph', 'image_png'])
    """
    
    def __init__(self):
        """Initialize preprocessing pipeline."""
        logger.info("PreprocessingPipeline initialized")
    
    def process(self, smiles: str) -> Dict[str, Any]:
        """
        Process SMILES string into all required representations.
        
        This method performs the complete preprocessing workflow:
        1. Validate SMILES syntax
        2. Standardize molecule (neutralize charges, remove salts, canonicalize)
        3. Compute molecular descriptors (200+ features)
        4. Compute Morgan fingerprints (2048-bit, radius=2)
        5. Compute MACCS keys (167-bit)
        6. Build molecular graph (PyTorch Geometric format)
        7. Generate 2D structure image (PNG)
        
        All errors are caught and re-raised with descriptive messages.
        Processing time is measured and logged.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Dictionary containing:
                - mol: RDKit Mol object
                - canonical_smiles: Standardized canonical SMILES string
                - descriptors: np.ndarray of shape (200,) with molecular descriptors
                - morgan_fp: np.ndarray of shape (2048,) with Morgan fingerprint
                - maccs_fp: np.ndarray of shape (167,) with MACCS keys
                - graph: torch_geometric.data.Data object with molecular graph
                - image_png: bytes containing PNG image of 2D structure
                
        Raises:
            ValueError: If SMILES is invalid, empty, or cannot be processed
            Exception: For other processing errors with descriptive messages
            
        Examples:
            >>> pipeline = PreprocessingPipeline()
            >>> result = pipeline.process("CCO")
            >>> result['canonical_smiles']
            'CCO'
            >>> result['descriptors'].shape
            (200,)
            >>> result['morgan_fp'].shape
            (2048,)
            >>> result['maccs_fp'].shape
            (167,)
        """
        start_time = time.time()
        
        try:
            # Step 1: Validate SMILES
            logger.debug(f"Validating SMILES: {smiles}")
            if not validate_smiles(smiles):
                raise ValueError(f"Invalid SMILES: unable to parse molecular structure")
            
            # Step 2: Standardize SMILES
            logger.debug("Standardizing SMILES")
            canonical_smiles = standardize_smiles(smiles)
            
            # Step 3: Convert to RDKit molecule
            logger.debug("Converting to RDKit molecule")
            mol = smiles_to_mol(canonical_smiles)
            
            # Step 4: Compute descriptors
            logger.debug("Computing molecular descriptors")
            descriptors = compute_descriptors(mol)
            
            # Step 5: Compute Morgan fingerprints
            logger.debug("Computing Morgan fingerprints")
            morgan_fp = compute_morgan_fingerprint(mol, radius=2, n_bits=2048)
            
            # Step 6: Compute MACCS keys
            logger.debug("Computing MACCS keys")
            maccs_fp = compute_maccs_keys(mol)
            
            # Step 7: Build molecular graph
            logger.debug("Building molecular graph")
            graph = mol_to_graph(mol)
            
            # Step 8: Generate 2D image
            logger.debug("Generating 2D structure image")
            image_png = generate_2d_image(mol, size=(400, 400))
            
            # Measure processing time
            elapsed_time = time.time() - start_time
            logger.info(f"Preprocessing completed in {elapsed_time*1000:.2f}ms for SMILES: {smiles}")
            
            return {
                'mol': mol,
                'canonical_smiles': canonical_smiles,
                'descriptors': descriptors,
                'morgan_fp': morgan_fp,
                'maccs_fp': maccs_fp,
                'graph': graph,
                'image_png': image_png
            }
            
        except ValueError as e:
            # Re-raise validation errors with original message
            elapsed_time = time.time() - start_time
            logger.error(f"Preprocessing failed after {elapsed_time*1000:.2f}ms: {str(e)}")
            raise
            
        except Exception as e:
            # Catch all other errors and provide descriptive message
            elapsed_time = time.time() - start_time
            error_msg = f"Preprocessing failed: {type(e).__name__}: {str(e)}"
            logger.error(f"{error_msg} (after {elapsed_time*1000:.2f}ms)")
            raise Exception(error_msg) from e
