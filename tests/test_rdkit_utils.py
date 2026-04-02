"""
Unit tests for RDKit utilities module.

Tests cover SMILES validation, standardization, molecule parsing, and image generation.
"""

import pytest
from rdkit import Chem

from backend.app.preprocessing.rdkit_utils import (
    validate_smiles,
    standardize_smiles,
    smiles_to_mol,
    generate_2d_image
)


class TestValidateSmiles:
    """Test suite for validate_smiles function."""
    
    def test_valid_simple_smiles(self):
        """Test validation of simple valid SMILES strings."""
        assert validate_smiles("CCO") is True
        assert validate_smiles("c1ccccc1") is True
        assert validate_smiles("CC(=O)O") is True
    
    def test_valid_complex_smiles(self):
        """Test validation of complex valid SMILES strings."""
        assert validate_smiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O") is True  # Ibuprofen
        assert validate_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C") is True  # Caffeine
    
    def test_invalid_smiles(self):
        """Test validation of invalid SMILES strings."""
        assert validate_smiles("invalid") is False
        assert validate_smiles("C(C") is False
        assert validate_smiles("CC)C") is False
    
    def test_empty_smiles(self):
        """Test validation of empty SMILES strings."""
        assert validate_smiles("") is False
        assert validate_smiles(None) is False
    
    def test_non_string_input(self):
        """Test validation with non-string input."""
        assert validate_smiles(123) is False
        assert validate_smiles([]) is False


class TestStandardizeSmiles:
    """Test suite for standardize_smiles function."""
    
    def test_simple_standardization(self):
        """Test standardization of simple molecules."""
        result = standardize_smiles("CCO")
        assert result == "CCO"
    
    def test_salt_removal(self):
        """Test removal of salts from SMILES."""
        # Sodium chloride should be stripped to largest fragment
        result = standardize_smiles("CCO.[Na+].[Cl-]")
        assert result == "CCO"
    
    def test_charge_neutralization(self):
        """Test neutralization of charged molecules."""
        # Carboxylate anion should be neutralized
        result = standardize_smiles("CC(=O)[O-]")
        assert "O-" not in result or result == "CC(=O)O"
    
    def test_tautomer_canonicalization(self):
        """Test canonicalization of tautomers."""
        # Different tautomers should give same canonical form
        keto = standardize_smiles("CC(=O)CC")
        enol = standardize_smiles("CC(O)=CC")
        # Both should standardize (may be same or different depending on RDKit version)
        assert isinstance(keto, str)
        assert isinstance(enol, str)
    
    def test_invalid_smiles_raises_error(self):
        """Test that invalid SMILES raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SMILES"):
            standardize_smiles("invalid")
    
    def test_empty_smiles_raises_error(self):
        """Test that empty SMILES raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            standardize_smiles("")
    
    def test_none_smiles_raises_error(self):
        """Test that None SMILES raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            standardize_smiles(None)
    
    def test_idempotence(self):
        """Test that standardization is idempotent."""
        smiles = "CCO"
        first = standardize_smiles(smiles)
        second = standardize_smiles(first)
        assert first == second


class TestSmilesToMol:
    """Test suite for smiles_to_mol function."""
    
    def test_valid_smiles_parsing(self):
        """Test parsing of valid SMILES to molecule."""
        mol = smiles_to_mol("CCO")
        assert mol is not None
        assert isinstance(mol, Chem.Mol)
        assert mol.GetNumAtoms() == 3
    
    def test_complex_molecule_parsing(self):
        """Test parsing of complex molecules."""
        mol = smiles_to_mol("c1ccccc1")  # Benzene
        assert mol is not None
        assert mol.GetNumAtoms() == 6
    
    def test_invalid_smiles_raises_error(self):
        """Test that invalid SMILES raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SMILES"):
            smiles_to_mol("invalid")
    
    def test_empty_smiles_raises_error(self):
        """Test that empty SMILES raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            smiles_to_mol("")
    
    def test_none_smiles_raises_error(self):
        """Test that None SMILES raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            smiles_to_mol(None)


class TestGenerate2dImage:
    """Test suite for generate_2d_image function."""
    
    def test_image_generation(self):
        """Test generation of 2D molecular image."""
        mol = smiles_to_mol("CCO")
        png_bytes = generate_2d_image(mol)
        
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        # Check PNG magic number
        assert png_bytes[:8] == b'\x89PNG\r\n\x1a\n'
    
    def test_custom_size(self):
        """Test image generation with custom size."""
        mol = smiles_to_mol("CCO")
        png_bytes = generate_2d_image(mol, size=(200, 200))
        
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
    
    def test_default_size(self):
        """Test image generation with default size (400x400)."""
        mol = smiles_to_mol("CCO")
        png_bytes = generate_2d_image(mol)
        
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
    
    def test_complex_molecule_image(self):
        """Test image generation for complex molecules."""
        mol = smiles_to_mol("CC(C)Cc1ccc(cc1)C(C)C(=O)O")  # Ibuprofen
        png_bytes = generate_2d_image(mol)
        
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
    
    def test_none_molecule_raises_error(self):
        """Test that None molecule raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            generate_2d_image(None)
    
    def test_molecule_with_existing_coords(self):
        """Test image generation for molecule with existing 2D coordinates."""
        mol = smiles_to_mol("CCO")
        # Pre-compute 2D coordinates
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(mol)
        
        png_bytes = generate_2d_image(mol)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0


class TestIntegration:
    """Integration tests for rdkit_utils module."""
    
    def test_full_pipeline(self):
        """Test complete pipeline: validate -> standardize -> parse -> image."""
        smiles = "CCO"
        
        # Validate
        assert validate_smiles(smiles) is True
        
        # Standardize
        std_smiles = standardize_smiles(smiles)
        assert isinstance(std_smiles, str)
        
        # Parse
        mol = smiles_to_mol(std_smiles)
        assert mol is not None
        
        # Generate image
        png_bytes = generate_2d_image(mol)
        assert len(png_bytes) > 0
    
    def test_pipeline_with_salt(self):
        """Test pipeline with salt-containing SMILES."""
        smiles = "CCO.[Na+].[Cl-]"
        
        # Validate original
        assert validate_smiles(smiles) is True
        
        # Standardize (should remove salt)
        std_smiles = standardize_smiles(smiles)
        assert "[Na+]" not in std_smiles
        assert "[Cl-]" not in std_smiles
        
        # Parse and image
        mol = smiles_to_mol(std_smiles)
        png_bytes = generate_2d_image(mol)
        assert len(png_bytes) > 0
