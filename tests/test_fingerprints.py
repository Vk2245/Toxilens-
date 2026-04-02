"""
Unit tests for molecular fingerprint computation module.

Tests cover Morgan fingerprint and MACCS keys computation for various
molecular structures and edge cases.
"""

import pytest
import numpy as np
from rdkit import Chem

from backend.app.preprocessing.fingerprints import (
    compute_morgan_fingerprint,
    compute_maccs_keys
)


class TestComputeMorganFingerprint:
    """Test suite for compute_morgan_fingerprint function."""
    
    def test_simple_molecule(self):
        """Test Morgan fingerprint computation for simple molecule (ethanol)."""
        mol = Chem.MolFromSmiles("CCO")
        fp = compute_morgan_fingerprint(mol)
        
        assert isinstance(fp, np.ndarray)
        assert fp.shape == (2048,)
        assert fp.dtype == np.int64
    
    def test_default_parameters(self):
        """Test Morgan fingerprint with default parameters (radius=2, n_bits=2048)."""
        mol = Chem.MolFromSmiles("CCO")
        fp = compute_morgan_fingerprint(mol)
        
        assert fp.shape == (2048,)
        # Should have some bits set
        assert np.sum(fp) > 0
    
    def test_custom_radius(self):
        """Test Morgan fingerprint with custom radius."""
        # Use a larger molecule to see differences with different radii
        mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")  # Ibuprofen
        fp_r1 = compute_morgan_fingerprint(mol, radius=1)
        fp_r2 = compute_morgan_fingerprint(mol, radius=2)
        fp_r3 = compute_morgan_fingerprint(mol, radius=3)
        
        # Different radii should produce different fingerprints for larger molecules
        # (small molecules may have identical fingerprints at different radii)
        assert not np.array_equal(fp_r1, fp_r2)
        assert not np.array_equal(fp_r2, fp_r3)
    
    def test_custom_n_bits(self):
        """Test Morgan fingerprint with custom bit length."""
        mol = Chem.MolFromSmiles("CCO")
        fp_1024 = compute_morgan_fingerprint(mol, n_bits=1024)
        fp_2048 = compute_morgan_fingerprint(mol, n_bits=2048)
        fp_4096 = compute_morgan_fingerprint(mol, n_bits=4096)
        
        assert fp_1024.shape == (1024,)
        assert fp_2048.shape == (2048,)
        assert fp_4096.shape == (4096,)
    
    def test_aromatic_molecule(self):
        """Test Morgan fingerprint for aromatic molecule (benzene)."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        fp = compute_morgan_fingerprint(mol)
        
        assert fp.shape == (2048,)
        assert np.sum(fp) > 0
    
    def test_complex_molecule(self):
        """Test Morgan fingerprint for complex molecule (ibuprofen)."""
        mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        fp = compute_morgan_fingerprint(mol)
        
        assert fp.shape == (2048,)
        assert np.sum(fp) > 0
    
    def test_caffeine(self):
        """Test Morgan fingerprint for caffeine."""
        mol = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        fp = compute_morgan_fingerprint(mol)
        
        assert fp.shape == (2048,)
        assert np.sum(fp) > 0
    
    def test_none_molecule_raises_error(self):
        """Test that None molecule raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            compute_morgan_fingerprint(None)
    
    def test_binary_values(self):
        """Test that fingerprint contains only binary values (0 or 1)."""
        mol = Chem.MolFromSmiles("CCO")
        fp = compute_morgan_fingerprint(mol)
        
        # All values should be 0 or 1
        assert np.all((fp == 0) | (fp == 1))
    
    def test_different_molecules_different_fingerprints(self):
        """Test that different molecules produce different fingerprints."""
        mol1 = Chem.MolFromSmiles("CCO")  # Ethanol
        mol2 = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        
        fp1 = compute_morgan_fingerprint(mol1)
        fp2 = compute_morgan_fingerprint(mol2)
        
        # Fingerprints should be different
        assert not np.array_equal(fp1, fp2)
    
    def test_same_molecule_same_fingerprint(self):
        """Test that same molecule produces same fingerprint."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCO")
        
        fp1 = compute_morgan_fingerprint(mol1)
        fp2 = compute_morgan_fingerprint(mol2)
        
        # Fingerprints should be identical
        assert np.array_equal(fp1, fp2)
    
    def test_charged_molecule(self):
        """Test Morgan fingerprint for charged molecule."""
        mol = Chem.MolFromSmiles("CC(=O)[O-]")  # Acetate anion
        fp = compute_morgan_fingerprint(mol)
        
        assert fp.shape == (2048,)
        assert np.sum(fp) > 0
    
    def test_molecule_with_heteroatoms(self):
        """Test Morgan fingerprint for molecule with heteroatoms."""
        mol = Chem.MolFromSmiles("CCN(C)C")  # Triethylamine
        fp = compute_morgan_fingerprint(mol)
        
        assert fp.shape == (2048,)
        assert np.sum(fp) > 0
    
    def test_halogenated_molecule(self):
        """Test Morgan fingerprint for halogenated molecule."""
        mol = Chem.MolFromSmiles("CCCl")  # Chloroethane
        fp = compute_morgan_fingerprint(mol)
        
        assert fp.shape == (2048,)
        assert np.sum(fp) > 0


class TestComputeMaccsKeys:
    """Test suite for compute_maccs_keys function."""
    
    def test_simple_molecule(self):
        """Test MACCS keys computation for simple molecule (ethanol)."""
        mol = Chem.MolFromSmiles("CCO")
        maccs = compute_maccs_keys(mol)
        
        assert isinstance(maccs, np.ndarray)
        assert maccs.shape == (167,)
        assert maccs.dtype == np.int64
    
    def test_aromatic_molecule(self):
        """Test MACCS keys for aromatic molecule (benzene)."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        maccs = compute_maccs_keys(mol)
        
        assert maccs.shape == (167,)
        # Should have some keys set
        assert np.sum(maccs) > 0
    
    def test_complex_molecule(self):
        """Test MACCS keys for complex molecule (ibuprofen)."""
        mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        maccs = compute_maccs_keys(mol)
        
        assert maccs.shape == (167,)
        assert np.sum(maccs) > 0
    
    def test_caffeine(self):
        """Test MACCS keys for caffeine."""
        mol = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        maccs = compute_maccs_keys(mol)
        
        assert maccs.shape == (167,)
        assert np.sum(maccs) > 0
    
    def test_none_molecule_raises_error(self):
        """Test that None molecule raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            compute_maccs_keys(None)
    
    def test_binary_values(self):
        """Test that MACCS keys contain only binary values (0 or 1)."""
        mol = Chem.MolFromSmiles("CCO")
        maccs = compute_maccs_keys(mol)
        
        # All values should be 0 or 1
        assert np.all((maccs == 0) | (maccs == 1))
    
    def test_different_molecules_different_maccs(self):
        """Test that different molecules produce different MACCS keys."""
        mol1 = Chem.MolFromSmiles("CCO")  # Ethanol
        mol2 = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        
        maccs1 = compute_maccs_keys(mol1)
        maccs2 = compute_maccs_keys(mol2)
        
        # MACCS keys should be different
        assert not np.array_equal(maccs1, maccs2)
    
    def test_same_molecule_same_maccs(self):
        """Test that same molecule produces same MACCS keys."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCO")
        
        maccs1 = compute_maccs_keys(mol1)
        maccs2 = compute_maccs_keys(mol2)
        
        # MACCS keys should be identical
        assert np.array_equal(maccs1, maccs2)
    
    def test_charged_molecule(self):
        """Test MACCS keys for charged molecule."""
        mol = Chem.MolFromSmiles("CC(=O)[O-]")  # Acetate anion
        maccs = compute_maccs_keys(mol)
        
        assert maccs.shape == (167,)
        assert np.sum(maccs) > 0
    
    def test_molecule_with_heteroatoms(self):
        """Test MACCS keys for molecule with heteroatoms."""
        mol = Chem.MolFromSmiles("CCN(C)C")  # Triethylamine
        maccs = compute_maccs_keys(mol)
        
        assert maccs.shape == (167,)
        assert np.sum(maccs) > 0
    
    def test_halogenated_molecule(self):
        """Test MACCS keys for halogenated molecule."""
        mol = Chem.MolFromSmiles("CCCl")  # Chloroethane
        maccs = compute_maccs_keys(mol)
        
        assert maccs.shape == (167,)
        assert np.sum(maccs) > 0
    
    def test_nitro_compound(self):
        """Test MACCS keys for nitro compound."""
        mol = Chem.MolFromSmiles("c1ccc(cc1)[N+](=O)[O-]")  # Nitrobenzene
        maccs = compute_maccs_keys(mol)
        
        assert maccs.shape == (167,)
        assert np.sum(maccs) > 0


class TestFingerprintProperties:
    """Test suite for fingerprint properties and invariants."""
    
    def test_morgan_reproducibility(self):
        """Test that Morgan fingerprint computation is reproducible."""
        mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        
        fp1 = compute_morgan_fingerprint(mol)
        fp2 = compute_morgan_fingerprint(mol)
        fp3 = compute_morgan_fingerprint(mol)
        
        assert np.array_equal(fp1, fp2)
        assert np.array_equal(fp2, fp3)
    
    def test_maccs_reproducibility(self):
        """Test that MACCS keys computation is reproducible."""
        mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        
        maccs1 = compute_maccs_keys(mol)
        maccs2 = compute_maccs_keys(mol)
        maccs3 = compute_maccs_keys(mol)
        
        assert np.array_equal(maccs1, maccs2)
        assert np.array_equal(maccs2, maccs3)
    
    def test_morgan_sparsity(self):
        """Test that Morgan fingerprints are sparse (mostly zeros)."""
        mol = Chem.MolFromSmiles("CCO")
        fp = compute_morgan_fingerprint(mol)
        
        # For small molecules, fingerprints should be sparse
        sparsity = np.sum(fp == 0) / len(fp)
        assert sparsity > 0.5  # At least 50% zeros
    
    def test_maccs_fixed_length(self):
        """Test that MACCS keys always have 167 bits."""
        test_smiles = [
            "C",  # Methane (smallest)
            "CCO",
            "c1ccccc1",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
        ]
        
        for smiles in test_smiles:
            mol = Chem.MolFromSmiles(smiles)
            maccs = compute_maccs_keys(mol)
            assert maccs.shape == (167,), f"Wrong shape for {smiles}"


class TestFingerprintComparison:
    """Test suite for comparing Morgan and MACCS fingerprints."""
    
    def test_morgan_vs_maccs_different_lengths(self):
        """Test that Morgan and MACCS have different lengths."""
        mol = Chem.MolFromSmiles("CCO")
        
        morgan = compute_morgan_fingerprint(mol)
        maccs = compute_maccs_keys(mol)
        
        assert len(morgan) == 2048
        assert len(maccs) == 167
        assert len(morgan) != len(maccs)
    
    def test_both_capture_structural_features(self):
        """Test that both fingerprints capture structural differences."""
        mol1 = Chem.MolFromSmiles("CCO")  # Ethanol
        mol2 = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        
        morgan1 = compute_morgan_fingerprint(mol1)
        morgan2 = compute_morgan_fingerprint(mol2)
        maccs1 = compute_maccs_keys(mol1)
        maccs2 = compute_maccs_keys(mol2)
        
        # Both should distinguish between the molecules
        assert not np.array_equal(morgan1, morgan2)
        assert not np.array_equal(maccs1, maccs2)
    
    def test_concatenation_for_ml(self):
        """Test concatenation of Morgan and MACCS for ML features."""
        mol = Chem.MolFromSmiles("CCO")
        
        morgan = compute_morgan_fingerprint(mol)
        maccs = compute_maccs_keys(mol)
        
        # Concatenate as would be done for LightGBM model
        combined = np.concatenate([morgan, maccs])
        
        assert combined.shape == (2048 + 167,)
        assert combined.shape == (2215,)
        assert combined.dtype == np.int64


class TestIntegration:
    """Integration tests for fingerprint computation."""
    
    def test_batch_computation(self):
        """Test fingerprint computation for multiple molecules."""
        smiles_list = [
            "CCO",
            "c1ccccc1",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC(=O)O"
        ]
        
        morgan_list = []
        maccs_list = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            morgan_list.append(compute_morgan_fingerprint(mol))
            maccs_list.append(compute_maccs_keys(mol))
        
        # All should have correct shapes
        assert all(fp.shape == (2048,) for fp in morgan_list)
        assert all(fp.shape == (167,) for fp in maccs_list)
    
    def test_fingerprint_matrix_creation(self):
        """Test creation of fingerprint matrices for multiple molecules."""
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        
        morgan_matrix = []
        maccs_matrix = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            morgan_matrix.append(compute_morgan_fingerprint(mol))
            maccs_matrix.append(compute_maccs_keys(mol))
        
        morgan_matrix = np.array(morgan_matrix)
        maccs_matrix = np.array(maccs_matrix)
        
        assert morgan_matrix.shape == (3, 2048)
        assert maccs_matrix.shape == (3, 167)
        assert morgan_matrix.dtype == np.int64
        assert maccs_matrix.dtype == np.int64
    
    def test_combined_feature_matrix(self):
        """Test creation of combined descriptor+fingerprint feature matrix."""
        from backend.app.preprocessing.descriptors import compute_descriptors
        
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        
        feature_matrix = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            
            descriptors = compute_descriptors(mol)
            morgan = compute_morgan_fingerprint(mol)
            maccs = compute_maccs_keys(mol)
            
            # Combine all features as would be done for LightGBM
            features = np.concatenate([descriptors, morgan, maccs])
            feature_matrix.append(features)
        
        feature_matrix = np.array(feature_matrix)
        
        # 200 descriptors + 2048 Morgan + 167 MACCS = 2415 features
        assert feature_matrix.shape == (3, 2415)
    
    def test_tanimoto_similarity_calculation(self):
        """Test Tanimoto similarity calculation using Morgan fingerprints."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCCO")  # Similar (propanol)
        mol3 = Chem.MolFromSmiles("c1ccccc1")  # Different (benzene)
        
        fp1 = compute_morgan_fingerprint(mol1)
        fp2 = compute_morgan_fingerprint(mol2)
        fp3 = compute_morgan_fingerprint(mol3)
        
        # Tanimoto similarity = intersection / union
        def tanimoto(fp_a, fp_b):
            intersection = np.sum(fp_a & fp_b)
            union = np.sum(fp_a | fp_b)
            return intersection / union if union > 0 else 0.0
        
        sim_1_2 = tanimoto(fp1, fp2)
        sim_1_3 = tanimoto(fp1, fp3)
        
        # Ethanol and propanol should be more similar than ethanol and benzene
        assert sim_1_2 > sim_1_3


class TestEdgeCases:
    """Test suite for edge cases and special molecules."""
    
    def test_single_atom_molecule(self):
        """Test fingerprints for single atom molecule."""
        mol = Chem.MolFromSmiles("C")  # Methane
        
        morgan = compute_morgan_fingerprint(mol)
        maccs = compute_maccs_keys(mol)
        
        assert morgan.shape == (2048,)
        assert maccs.shape == (167,)
        assert np.sum(morgan) > 0
        assert np.sum(maccs) > 0
    
    def test_large_molecule(self):
        """Test fingerprints for large molecule."""
        # Taxol (paclitaxel) - large complex molecule
        taxol_smiles = "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
        mol = Chem.MolFromSmiles(taxol_smiles)
        
        morgan = compute_morgan_fingerprint(mol)
        maccs = compute_maccs_keys(mol)
        
        assert morgan.shape == (2048,)
        assert maccs.shape == (167,)
        # Large molecule should have more bits set
        assert np.sum(morgan) > 50
        assert np.sum(maccs) > 20
    
    def test_molecule_with_stereochemistry(self):
        """Test fingerprints for molecule with stereochemistry."""
        mol = Chem.MolFromSmiles("C[C@H](O)CC")  # (S)-2-butanol
        
        morgan = compute_morgan_fingerprint(mol)
        maccs = compute_maccs_keys(mol)
        
        assert morgan.shape == (2048,)
        assert maccs.shape == (167,)
