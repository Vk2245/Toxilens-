"""
Unit tests for preprocessing pipeline module.
"""

import pytest
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data

from backend.app.preprocessing.pipeline import PreprocessingPipeline


class TestPreprocessingPipeline:
    """Test suite for PreprocessingPipeline class."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return PreprocessingPipeline()
    
    def test_process_valid_smiles(self, pipeline):
        """Test processing a valid SMILES string."""
        result = pipeline.process("CCO")
        
        # Check all expected keys are present
        assert 'mol' in result
        assert 'canonical_smiles' in result
        assert 'descriptors' in result
        assert 'morgan_fp' in result
        assert 'maccs_fp' in result
        assert 'graph' in result
        assert 'image_png' in result
        
        # Check types
        assert isinstance(result['mol'], Chem.Mol)
        assert isinstance(result['canonical_smiles'], str)
        assert isinstance(result['descriptors'], np.ndarray)
        assert isinstance(result['morgan_fp'], np.ndarray)
        assert isinstance(result['maccs_fp'], np.ndarray)
        assert isinstance(result['graph'], Data)
        assert isinstance(result['image_png'], bytes)
        
        # Check shapes
        assert result['descriptors'].shape == (200,)
        assert result['morgan_fp'].shape == (2048,)
        assert result['maccs_fp'].shape == (167,)
        
        # Check canonical SMILES is valid
        assert result['canonical_smiles'] == 'CCO'
        
        # Check molecule has correct number of atoms
        assert result['mol'].GetNumAtoms() == 3
        
        # Check graph has correct structure
        assert result['graph'].x.shape[0] == 3  # 3 atoms
        assert result['graph'].edge_index.shape[0] == 2  # COO format
        
        # Check image is non-empty
        assert len(result['image_png']) > 0
    
    def test_process_aromatic_smiles(self, pipeline):
        """Test processing an aromatic SMILES string."""
        result = pipeline.process("c1ccccc1")  # Benzene
        
        assert result['canonical_smiles'] == 'c1ccccc1'
        assert result['mol'].GetNumAtoms() == 6
        assert result['descriptors'].shape == (200,)
    
    def test_process_complex_molecule(self, pipeline):
        """Test processing a complex drug-like molecule."""
        # Aspirin
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        result = pipeline.process(smiles)
        
        assert 'canonical_smiles' in result
        assert result['mol'].GetNumAtoms() == 13
        assert result['descriptors'].shape == (200,)
        assert result['morgan_fp'].shape == (2048,)
        assert result['maccs_fp'].shape == (167,)
    
    def test_process_invalid_smiles(self, pipeline):
        """Test that invalid SMILES raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SMILES"):
            pipeline.process("invalid_smiles_123")
    
    def test_process_empty_smiles(self, pipeline):
        """Test that empty SMILES raises ValueError."""
        with pytest.raises(ValueError):
            pipeline.process("")
    
    def test_process_none_smiles(self, pipeline):
        """Test that None SMILES raises ValueError."""
        with pytest.raises(ValueError):
            pipeline.process(None)
    
    def test_process_with_salt(self, pipeline):
        """Test processing SMILES with salt (should be removed)."""
        # Aspirin sodium salt - the salt remover will keep the main molecule
        result = pipeline.process("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
        
        # Should standardize and remove salt
        assert 'canonical_smiles' in result
        assert result['mol'] is not None
        # The sodium should be removed, leaving just the aspirin molecule
        assert '[Na+]' not in result['canonical_smiles']
    
    def test_descriptors_are_numeric(self, pipeline):
        """Test that all descriptors are numeric values."""
        result = pipeline.process("CCO")
        descriptors = result['descriptors']
        
        # Check all values are numeric (not NaN or inf)
        assert np.all(np.isfinite(descriptors))
        assert descriptors.dtype == np.float64
    
    def test_fingerprints_are_binary(self, pipeline):
        """Test that fingerprints contain binary values."""
        result = pipeline.process("CCO")
        
        morgan_fp = result['morgan_fp']
        maccs_fp = result['maccs_fp']
        
        # Check Morgan fingerprint is binary
        assert np.all((morgan_fp == 0) | (morgan_fp == 1))
        
        # Check MACCS keys are binary
        assert np.all((maccs_fp == 0) | (maccs_fp == 1))
    
    def test_graph_structure(self, pipeline):
        """Test that graph has correct structure."""
        result = pipeline.process("CCO")  # Ethanol: 3 atoms, 2 bonds
        graph = result['graph']
        
        # Check node features
        assert graph.x.shape[0] == 3  # 3 atoms
        assert graph.x.shape[1] > 0  # Has node features
        
        # Check edge structure (undirected, so 2 edges per bond)
        assert graph.edge_index.shape[0] == 2  # COO format
        assert graph.edge_index.shape[1] == 4  # 2 bonds * 2 directions
        
        # Check edge attributes
        assert graph.edge_attr.shape[0] == 4  # 4 directed edges
        assert graph.edge_attr.shape[1] == 7  # Edge feature dimension
    
    def test_image_is_png(self, pipeline):
        """Test that generated image is valid PNG."""
        result = pipeline.process("CCO")
        image_png = result['image_png']
        
        # Check PNG magic bytes
        assert image_png[:8] == b'\x89PNG\r\n\x1a\n'
    
    def test_processing_time_logged(self, pipeline, caplog):
        """Test that processing time is logged."""
        import logging
        caplog.set_level(logging.INFO)
        
        pipeline.process("CCO")
        
        # Check that processing time was logged
        assert any("Preprocessing completed" in record.message for record in caplog.records)
        assert any("ms" in record.message for record in caplog.records)
    
    def test_error_handling_with_logging(self, pipeline, caplog):
        """Test that errors are logged with timing information."""
        import logging
        caplog.set_level(logging.ERROR)
        
        with pytest.raises(ValueError):
            pipeline.process("invalid_smiles_xyz")
        
        # Check that error was logged with timing
        assert any("Preprocessing failed" in record.message for record in caplog.records)
        assert any("ms" in record.message for record in caplog.records)
    
    def test_idempotence(self, pipeline):
        """Test that processing the same SMILES twice gives same canonical form."""
        result1 = pipeline.process("CCO")
        result2 = pipeline.process("CCO")
        
        assert result1['canonical_smiles'] == result2['canonical_smiles']
        assert np.array_equal(result1['descriptors'], result2['descriptors'])
        assert np.array_equal(result1['morgan_fp'], result2['morgan_fp'])
    
    def test_different_smiles_same_molecule(self, pipeline):
        """Test that different SMILES for same molecule give same canonical form."""
        # Different representations of ethanol
        result1 = pipeline.process("CCO")
        result2 = pipeline.process("OCC")
        
        assert result1['canonical_smiles'] == result2['canonical_smiles']
        assert np.array_equal(result1['descriptors'], result2['descriptors'])
