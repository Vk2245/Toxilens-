"""
Unit tests for molecular descriptor computation module.

Tests cover descriptor computation for various molecular structures
and edge cases.
"""

import pytest
import numpy as np
from rdkit import Chem

from backend.app.preprocessing.descriptors import compute_descriptors


class TestComputeDescriptors:
    """Test suite for compute_descriptors function."""
    
    def test_simple_molecule(self):
        """Test descriptor computation for simple molecule (ethanol)."""
        mol = Chem.MolFromSmiles("CCO")
        descriptors = compute_descriptors(mol)
        
        assert isinstance(descriptors, np.ndarray)
        assert descriptors.shape == (200,)
        assert descriptors.dtype == np.float64
    
    def test_aromatic_molecule(self):
        """Test descriptor computation for aromatic molecule (benzene)."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        descriptors = compute_descriptors(mol)
        
        assert descriptors.shape == (200,)
        # Benzene should have aromatic rings
        assert np.any(descriptors > 0)
    
    def test_complex_molecule(self):
        """Test descriptor computation for complex molecule (ibuprofen)."""
        mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        descriptors = compute_descriptors(mol)
        
        assert descriptors.shape == (200,)
        assert not np.any(np.isnan(descriptors))
        assert not np.any(np.isinf(descriptors))
    
    def test_caffeine(self):
        """Test descriptor computation for caffeine."""
        mol = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        descriptors = compute_descriptors(mol)
        
        assert descriptors.shape == (200,)
        assert not np.any(np.isnan(descriptors))
    
    def test_none_molecule_raises_error(self):
        """Test that None molecule raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            compute_descriptors(None)
    
    def test_descriptor_values_reasonable(self):
        """Test that descriptor values are in reasonable ranges."""
        mol = Chem.MolFromSmiles("CCO")
        descriptors = compute_descriptors(mol)
        
        # Molecular weight should be around 46 for ethanol
        mw = descriptors[0]
        assert 40 < mw < 50
        
        # All descriptors should be finite
        assert np.all(np.isfinite(descriptors))
    
    def test_different_molecules_different_descriptors(self):
        """Test that different molecules produce different descriptors."""
        mol1 = Chem.MolFromSmiles("CCO")  # Ethanol
        mol2 = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        
        desc1 = compute_descriptors(mol1)
        desc2 = compute_descriptors(mol2)
        
        # Descriptors should be different
        assert not np.allclose(desc1, desc2)
    
    def test_same_molecule_same_descriptors(self):
        """Test that same molecule produces same descriptors."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCO")
        
        desc1 = compute_descriptors(mol1)
        desc2 = compute_descriptors(mol2)
        
        # Descriptors should be identical
        assert np.allclose(desc1, desc2)
    
    def test_charged_molecule(self):
        """Test descriptor computation for charged molecule."""
        mol = Chem.MolFromSmiles("CC(=O)[O-]")  # Acetate anion
        descriptors = compute_descriptors(mol)
        
        assert descriptors.shape == (200,)
        assert not np.any(np.isnan(descriptors))
    
    def test_molecule_with_heteroatoms(self):
        """Test descriptor computation for molecule with heteroatoms."""
        mol = Chem.MolFromSmiles("CCN(C)C")  # Triethylamine
        descriptors = compute_descriptors(mol)
        
        assert descriptors.shape == (200,)
        assert not np.any(np.isnan(descriptors))
    
    def test_molecule_with_multiple_rings(self):
        """Test descriptor computation for molecule with multiple rings."""
        mol = Chem.MolFromSmiles("C1CCC2CCCCC2C1")  # Decalin
        descriptors = compute_descriptors(mol)
        
        assert descriptors.shape == (200,)
        assert not np.any(np.isnan(descriptors))
    
    def test_molecule_with_functional_groups(self):
        """Test descriptor computation for molecule with various functional groups."""
        mol = Chem.MolFromSmiles("CC(=O)OC")  # Methyl acetate (ester)
        descriptors = compute_descriptors(mol)
        
        assert descriptors.shape == (200,)
        assert not np.any(np.isnan(descriptors))
    
    def test_halogenated_molecule(self):
        """Test descriptor computation for halogenated molecule."""
        mol = Chem.MolFromSmiles("CCCl")  # Chloroethane
        descriptors = compute_descriptors(mol)
        
        assert descriptors.shape == (200,)
        assert not np.any(np.isnan(descriptors))
    
    def test_nitro_compound(self):
        """Test descriptor computation for nitro compound."""
        mol = Chem.MolFromSmiles("c1ccc(cc1)[N+](=O)[O-]")  # Nitrobenzene
        descriptors = compute_descriptors(mol)
        
        assert descriptors.shape == (200,)
        assert not np.any(np.isnan(descriptors))


class TestDescriptorProperties:
    """Test suite for descriptor properties and invariants."""
    
    def test_descriptor_count(self):
        """Test that exactly 200 descriptors are returned."""
        mol = Chem.MolFromSmiles("CCO")
        descriptors = compute_descriptors(mol)
        
        assert len(descriptors) == 200
    
    def test_descriptor_dtype(self):
        """Test that descriptors are float64."""
        mol = Chem.MolFromSmiles("CCO")
        descriptors = compute_descriptors(mol)
        
        assert descriptors.dtype == np.float64
    
    def test_no_nan_values(self):
        """Test that descriptors contain no NaN values."""
        test_smiles = [
            "CCO",
            "c1ccccc1",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC(=O)O"
        ]
        
        for smiles in test_smiles:
            mol = Chem.MolFromSmiles(smiles)
            descriptors = compute_descriptors(mol)
            assert not np.any(np.isnan(descriptors)), f"NaN found for {smiles}"
    
    def test_no_inf_values(self):
        """Test that descriptors contain no infinite values."""
        test_smiles = [
            "CCO",
            "c1ccccc1",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC(=O)O"
        ]
        
        for smiles in test_smiles:
            mol = Chem.MolFromSmiles(smiles)
            descriptors = compute_descriptors(mol)
            assert not np.any(np.isinf(descriptors)), f"Inf found for {smiles}"
    
    def test_descriptor_reproducibility(self):
        """Test that descriptor computation is reproducible."""
        mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        
        desc1 = compute_descriptors(mol)
        desc2 = compute_descriptors(mol)
        desc3 = compute_descriptors(mol)
        
        assert np.allclose(desc1, desc2)
        assert np.allclose(desc2, desc3)


class TestSpecificDescriptors:
    """Test suite for specific descriptor values."""
    
    def test_molecular_weight(self):
        """Test that molecular weight descriptor is correct."""
        mol = Chem.MolFromSmiles("CCO")
        descriptors = compute_descriptors(mol)
        
        # First descriptor should be molecular weight
        mw = descriptors[0]
        # Ethanol MW = 46.07
        assert 45 < mw < 47
    
    def test_hydrogen_bond_donors(self):
        """Test hydrogen bond donor count."""
        mol = Chem.MolFromSmiles("CCO")  # Ethanol has 1 H-bond donor
        descriptors = compute_descriptors(mol)
        
        # 5th descriptor is NumHDonors
        hbd = descriptors[4]
        assert hbd == 1
    
    def test_hydrogen_bond_acceptors(self):
        """Test hydrogen bond acceptor count."""
        mol = Chem.MolFromSmiles("CCO")  # Ethanol has 1 H-bond acceptor
        descriptors = compute_descriptors(mol)
        
        # 6th descriptor is NumHAcceptors
        hba = descriptors[5]
        assert hba == 1
    
    def test_aromatic_rings(self):
        """Test aromatic ring count."""
        mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene has 1 aromatic ring
        descriptors = compute_descriptors(mol)
        
        # 9th descriptor is NumAromaticRings
        aromatic_rings = descriptors[8]
        assert aromatic_rings == 1
    
    def test_rotatable_bonds(self):
        """Test rotatable bond count."""
        mol = Chem.MolFromSmiles("CCCC")  # Butane
        descriptors = compute_descriptors(mol)
        
        # 7th descriptor is NumRotatableBonds
        # RDKit counts rotatable bonds as single bonds not in rings
        # For butane (C-C-C-C), RDKit counts 1 rotatable bond (the middle C-C bond)
        rotatable = descriptors[6]
        assert rotatable == 1


class TestIntegration:
    """Integration tests for descriptor computation."""
    
    def test_batch_computation(self):
        """Test descriptor computation for multiple molecules."""
        smiles_list = [
            "CCO",
            "c1ccccc1",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC(=O)O"
        ]
        
        descriptors_list = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            descriptors = compute_descriptors(mol)
            descriptors_list.append(descriptors)
        
        # All should have same shape
        assert all(d.shape == (200,) for d in descriptors_list)
        
        # All should be different
        for i in range(len(descriptors_list)):
            for j in range(i + 1, len(descriptors_list)):
                assert not np.allclose(descriptors_list[i], descriptors_list[j])
    
    def test_descriptor_matrix_creation(self):
        """Test creation of descriptor matrix for multiple molecules."""
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        
        descriptor_matrix = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            descriptors = compute_descriptors(mol)
            descriptor_matrix.append(descriptors)
        
        descriptor_matrix = np.array(descriptor_matrix)
        
        assert descriptor_matrix.shape == (3, 200)
        assert descriptor_matrix.dtype == np.float64
