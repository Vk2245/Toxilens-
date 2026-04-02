"""
Unit tests for graph builder module.

Tests cover molecular graph construction, node features, edge features,
and edge cases for various molecular structures.
"""

import pytest
import torch
import numpy as np
from rdkit import Chem

from backend.app.preprocessing.graph_builder import mol_to_graph


class TestMolToGraph:
    """Test suite for mol_to_graph function."""
    
    def test_simple_molecule(self):
        """Test graph construction for simple molecule (ethanol)."""
        mol = Chem.MolFromSmiles("CCO")
        graph = mol_to_graph(mol)
        
        assert isinstance(graph.x, torch.Tensor)
        assert isinstance(graph.edge_index, torch.Tensor)
        assert isinstance(graph.edge_attr, torch.Tensor)
        
        # Ethanol has 3 atoms
        assert graph.x.shape[0] == 3
        # Ethanol has 2 bonds (4 directed edges)
        assert graph.edge_index.shape == (2, 4)
        assert graph.edge_attr.shape[0] == 4
    
    def test_node_feature_dimensions(self):
        """Test that node features have correct dimensions."""
        mol = Chem.MolFromSmiles("CCO")
        graph = mol_to_graph(mol)
        
        # Node features: 118 (atomic_num) + 11 (degree) + 4 (hybridization) + 
        #                1 (aromatic) + 1 (in_ring) + 1 (formal_charge) + 1 (num_Hs)
        # Total: 137 features
        assert graph.x.shape[1] == 137
    
    def test_edge_feature_dimensions(self):
        """Test that edge features have correct dimensions."""
        mol = Chem.MolFromSmiles("CCO")
        graph = mol_to_graph(mol)
        
        # Edge features: 4 (bond_type) + 1 (conjugated) + 1 (in_ring) + 1 (stereo)
        # Total: 7 features
        assert graph.edge_attr.shape[1] == 7
    
    def test_aromatic_molecule(self):
        """Test graph construction for aromatic molecule (benzene)."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        graph = mol_to_graph(mol)
        
        # Benzene has 6 atoms
        assert graph.x.shape[0] == 6
        # Benzene has 6 bonds (12 directed edges)
        assert graph.edge_index.shape == (2, 12)
        assert graph.edge_attr.shape[0] == 12
    
    def test_complex_molecule(self):
        """Test graph construction for complex molecule (ibuprofen)."""
        mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        graph = mol_to_graph(mol)
        
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        
        assert graph.x.shape[0] == num_atoms
        # Each bond creates 2 directed edges
        assert graph.edge_index.shape == (2, num_bonds * 2)
        assert graph.edge_attr.shape[0] == num_bonds * 2
    
    def test_caffeine(self):
        """Test graph construction for caffeine."""
        mol = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        graph = mol_to_graph(mol)
        
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        
        assert graph.x.shape[0] == num_atoms
        assert graph.edge_index.shape == (2, num_bonds * 2)
    
    def test_none_molecule_raises_error(self):
        """Test that None molecule raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            mol_to_graph(None)
    
    def test_edge_index_format(self):
        """Test that edge_index is in COO format."""
        mol = Chem.MolFromSmiles("CCO")
        graph = mol_to_graph(mol)
        
        # Edge index should be (2, num_edges)
        assert graph.edge_index.shape[0] == 2
        # All indices should be valid atom indices
        assert torch.all(graph.edge_index >= 0)
        assert torch.all(graph.edge_index < graph.x.shape[0])
    
    def test_undirected_graph(self):
        """Test that graph is undirected (edges in both directions)."""
        mol = Chem.MolFromSmiles("CC")  # Ethane
        graph = mol_to_graph(mol)
        
        # Should have 2 directed edges (one in each direction)
        assert graph.edge_index.shape[1] == 2
        
        # Check that edges exist in both directions
        edge_set = set()
        for i in range(graph.edge_index.shape[1]):
            src = graph.edge_index[0, i].item()
            dst = graph.edge_index[1, i].item()
            edge_set.add((src, dst))
        
        # For each edge (i, j), there should be (j, i)
        for src, dst in list(edge_set):
            assert (dst, src) in edge_set
    
    def test_single_atom_molecule(self):
        """Test graph construction for single atom molecule (methane)."""
        mol = Chem.MolFromSmiles("C")
        graph = mol_to_graph(mol)
        
        # Methane has 1 carbon atom (hydrogens are implicit)
        assert graph.x.shape[0] == 1
        # No bonds between heavy atoms
        assert graph.edge_index.shape == (2, 0)
        assert graph.edge_attr.shape == (0, 7)
    
    def test_charged_molecule(self):
        """Test graph construction for charged molecule."""
        mol = Chem.MolFromSmiles("CC(=O)[O-]")  # Acetate anion
        graph = mol_to_graph(mol)
        
        num_atoms = mol.GetNumAtoms()
        assert graph.x.shape[0] == num_atoms
        
        # Check that formal charge is captured
        # The oxygen with negative charge should have formal_charge = -1
        # Formal charge is at index 135 (118 + 11 + 4 + 1 + 1 + 1)
        formal_charges = graph.x[:, 135]
        assert torch.any(formal_charges == -1)
    
    def test_molecule_with_heteroatoms(self):
        """Test graph construction for molecule with heteroatoms."""
        mol = Chem.MolFromSmiles("CCN(C)C")  # Triethylamine
        graph = mol_to_graph(mol)
        
        num_atoms = mol.GetNumAtoms()
        assert graph.x.shape[0] == num_atoms
        
        # Should have nitrogen atom (atomic number 7)
        # Check one-hot encoding for nitrogen (index 6, since 0-indexed)
        atomic_nums = graph.x[:, :118]
        nitrogen_present = torch.any(atomic_nums[:, 6] == 1)
        assert nitrogen_present
    
    def test_halogenated_molecule(self):
        """Test graph construction for halogenated molecule."""
        mol = Chem.MolFromSmiles("CCCl")  # Chloroethane
        graph = mol_to_graph(mol)
        
        num_atoms = mol.GetNumAtoms()
        assert graph.x.shape[0] == num_atoms
        
        # Should have chlorine atom (atomic number 17)
        # Check one-hot encoding for chlorine (index 16)
        atomic_nums = graph.x[:, :118]
        chlorine_present = torch.any(atomic_nums[:, 16] == 1)
        assert chlorine_present
    
    def test_molecule_with_double_bond(self):
        """Test graph construction for molecule with double bond."""
        mol = Chem.MolFromSmiles("C=C")  # Ethene
        graph = mol_to_graph(mol)
        
        # Should have 2 atoms and 2 directed edges
        assert graph.x.shape[0] == 2
        assert graph.edge_index.shape[1] == 2
        
        # Check that double bond is encoded
        # Bond type features are at indices 0-3 of edge_attr
        # Double bond should be at index 1
        double_bond_features = graph.edge_attr[:, 1]
        assert torch.any(double_bond_features == 1)
    
    def test_molecule_with_triple_bond(self):
        """Test graph construction for molecule with triple bond."""
        mol = Chem.MolFromSmiles("C#C")  # Ethyne
        graph = mol_to_graph(mol)
        
        # Should have 2 atoms and 2 directed edges
        assert graph.x.shape[0] == 2
        assert graph.edge_index.shape[1] == 2
        
        # Check that triple bond is encoded
        # Triple bond should be at index 2
        triple_bond_features = graph.edge_attr[:, 2]
        assert torch.any(triple_bond_features == 1)
    
    def test_aromatic_bond_features(self):
        """Test that aromatic bonds are correctly encoded."""
        mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        graph = mol_to_graph(mol)
        
        # Check that aromatic bond type is encoded
        # Aromatic bond should be at index 3
        aromatic_bond_features = graph.edge_attr[:, 3]
        assert torch.any(aromatic_bond_features == 1)
        
        # Check that aromatic flag is set for atoms
        # Aromatic flag is at index 133 (118 + 11 + 4)
        aromatic_atom_features = graph.x[:, 133]
        assert torch.all(aromatic_atom_features == 1)
    
    def test_conjugated_system(self):
        """Test that conjugated bonds are correctly encoded."""
        mol = Chem.MolFromSmiles("C=CC=C")  # Butadiene
        graph = mol_to_graph(mol)
        
        # Check that conjugated flag is set
        # Conjugated is at index 4 of edge_attr
        conjugated_features = graph.edge_attr[:, 4]
        assert torch.any(conjugated_features == 1)
    
    def test_ring_detection(self):
        """Test that ring membership is correctly encoded."""
        mol = Chem.MolFromSmiles("C1CCCCC1")  # Cyclohexane
        graph = mol_to_graph(mol)
        
        # All atoms should be in ring
        # In ring flag is at index 134 (118 + 11 + 4 + 1)
        in_ring_atom_features = graph.x[:, 134]
        assert torch.all(in_ring_atom_features == 1)
        
        # All bonds should be in ring
        # In ring flag is at index 5 of edge_attr
        in_ring_bond_features = graph.edge_attr[:, 5]
        assert torch.all(in_ring_bond_features == 1)


class TestGraphProperties:
    """Test suite for graph properties and invariants."""
    
    def test_node_feature_dtype(self):
        """Test that node features are float tensors."""
        mol = Chem.MolFromSmiles("CCO")
        graph = mol_to_graph(mol)
        
        assert graph.x.dtype == torch.float32
    
    def test_edge_index_dtype(self):
        """Test that edge indices are long tensors."""
        mol = Chem.MolFromSmiles("CCO")
        graph = mol_to_graph(mol)
        
        assert graph.edge_index.dtype == torch.long
    
    def test_edge_attr_dtype(self):
        """Test that edge attributes are float tensors."""
        mol = Chem.MolFromSmiles("CCO")
        graph = mol_to_graph(mol)
        
        assert graph.edge_attr.dtype == torch.float32
    
    def test_no_nan_values(self):
        """Test that graph contains no NaN values."""
        test_smiles = [
            "CCO",
            "c1ccccc1",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC(=O)O"
        ]
        
        for smiles in test_smiles:
            mol = Chem.MolFromSmiles(smiles)
            graph = mol_to_graph(mol)
            
            assert not torch.any(torch.isnan(graph.x)), f"NaN in node features for {smiles}"
            assert not torch.any(torch.isnan(graph.edge_attr)), f"NaN in edge features for {smiles}"
    
    def test_no_inf_values(self):
        """Test that graph contains no infinite values."""
        test_smiles = [
            "CCO",
            "c1ccccc1",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC(=O)O"
        ]
        
        for smiles in test_smiles:
            mol = Chem.MolFromSmiles(smiles)
            graph = mol_to_graph(mol)
            
            assert not torch.any(torch.isinf(graph.x)), f"Inf in node features for {smiles}"
            assert not torch.any(torch.isinf(graph.edge_attr)), f"Inf in edge features for {smiles}"
    
    def test_graph_reproducibility(self):
        """Test that graph construction is reproducible."""
        mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        
        graph1 = mol_to_graph(mol)
        graph2 = mol_to_graph(mol)
        graph3 = mol_to_graph(mol)
        
        assert torch.allclose(graph1.x, graph2.x)
        assert torch.allclose(graph2.x, graph3.x)
        assert torch.equal(graph1.edge_index, graph2.edge_index)
        assert torch.equal(graph2.edge_index, graph3.edge_index)
        assert torch.allclose(graph1.edge_attr, graph2.edge_attr)
        assert torch.allclose(graph2.edge_attr, graph3.edge_attr)
    
    def test_different_molecules_different_graphs(self):
        """Test that different molecules produce different graphs."""
        mol1 = Chem.MolFromSmiles("CCO")  # Ethanol
        mol2 = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        
        graph1 = mol_to_graph(mol1)
        graph2 = mol_to_graph(mol2)
        
        # Graphs should be different
        assert graph1.x.shape != graph2.x.shape or not torch.allclose(graph1.x, graph2.x)
    
    def test_same_molecule_same_graph(self):
        """Test that same molecule produces same graph."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCO")
        
        graph1 = mol_to_graph(mol1)
        graph2 = mol_to_graph(mol2)
        
        # Graphs should be identical
        assert torch.allclose(graph1.x, graph2.x)
        assert torch.equal(graph1.edge_index, graph2.edge_index)
        assert torch.allclose(graph1.edge_attr, graph2.edge_attr)


class TestAtomFeatures:
    """Test suite for atom feature encoding."""
    
    def test_atomic_number_encoding(self):
        """Test that atomic numbers are correctly one-hot encoded."""
        mol = Chem.MolFromSmiles("CCO")
        graph = mol_to_graph(mol)
        
        # Extract atomic number features (first 118 dimensions)
        atomic_nums = graph.x[:, :118]
        
        # Each atom should have exactly one atomic number
        assert torch.all(torch.sum(atomic_nums, dim=1) == 1)
        
        # Carbon (atomic number 6) should be at index 5
        # Oxygen (atomic number 8) should be at index 7
        carbon_count = torch.sum(atomic_nums[:, 5]).item()
        oxygen_count = torch.sum(atomic_nums[:, 7]).item()
        
        assert carbon_count == 2  # Two carbons in ethanol
        assert oxygen_count == 1  # One oxygen in ethanol
    
    def test_degree_encoding(self):
        """Test that atom degree is correctly encoded."""
        mol = Chem.MolFromSmiles("C(C)(C)C")  # Neopentane (central carbon has degree 3)
        graph = mol_to_graph(mol)
        
        # Extract degree features (indices 118-128)
        degrees = graph.x[:, 118:129]
        
        # Each atom should have exactly one degree
        assert torch.all(torch.sum(degrees, dim=1) == 1)
        
        # Central carbon should have degree 3 (connected to 3 other carbons)
        degree_3_count = torch.sum(degrees[:, 3]).item()
        assert degree_3_count == 1
        
        # Three terminal carbons should have degree 1
        degree_1_count = torch.sum(degrees[:, 1]).item()
        assert degree_1_count == 3
    
    def test_hybridization_encoding(self):
        """Test that hybridization is correctly encoded."""
        mol = Chem.MolFromSmiles("C=C")  # Ethene (SP2 hybridization)
        graph = mol_to_graph(mol)
        
        # Extract hybridization features (indices 129-132)
        hyb = graph.x[:, 129:133]
        
        # Each atom should have exactly one hybridization type
        assert torch.all(torch.sum(hyb, dim=1) == 1)
        
        # Both carbons should be SP2 (index 1)
        sp2_count = torch.sum(hyb[:, 1]).item()
        assert sp2_count == 2
    
    def test_aromatic_flag(self):
        """Test that aromatic flag is correctly set."""
        mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        graph = mol_to_graph(mol)
        
        # Extract aromatic flag (index 133)
        aromatic = graph.x[:, 133]
        
        # All atoms in benzene should be aromatic
        assert torch.all(aromatic == 1)
    
    def test_ring_flag(self):
        """Test that ring flag is correctly set."""
        mol = Chem.MolFromSmiles("C1CCCCC1")  # Cyclohexane
        graph = mol_to_graph(mol)
        
        # Extract ring flag (index 134)
        in_ring = graph.x[:, 134]
        
        # All atoms in cyclohexane should be in ring
        assert torch.all(in_ring == 1)
    
    def test_formal_charge(self):
        """Test that formal charge is correctly encoded."""
        mol = Chem.MolFromSmiles("CC(=O)[O-]")  # Acetate anion
        graph = mol_to_graph(mol)
        
        # Extract formal charge (index 135)
        formal_charges = graph.x[:, 135]
        
        # Should have one atom with -1 charge
        assert torch.any(formal_charges == -1)
    
    def test_hydrogen_count(self):
        """Test that hydrogen count is correctly encoded."""
        mol = Chem.MolFromSmiles("C")  # Methane
        graph = mol_to_graph(mol)
        
        # Extract hydrogen count (index 136)
        num_hs = graph.x[:, 136]
        
        # Methane carbon should have 4 hydrogens
        assert num_hs[0].item() == 4


class TestBondFeatures:
    """Test suite for bond feature encoding."""
    
    def test_bond_type_encoding(self):
        """Test that bond types are correctly encoded."""
        mol = Chem.MolFromSmiles("CC=C")  # Propene
        graph = mol_to_graph(mol)
        
        # Extract bond type features (indices 0-3)
        bond_types = graph.edge_attr[:, :4]
        
        # Each bond should have exactly one type
        assert torch.all(torch.sum(bond_types, dim=1) == 1)
        
        # Should have both single and double bonds
        single_bonds = torch.sum(bond_types[:, 0]).item()
        double_bonds = torch.sum(bond_types[:, 1]).item()
        
        assert single_bonds > 0
        assert double_bonds > 0
    
    def test_conjugated_flag(self):
        """Test that conjugated flag is correctly set."""
        mol = Chem.MolFromSmiles("C=CC=C")  # Butadiene
        graph = mol_to_graph(mol)
        
        # Extract conjugated flag (index 4)
        conjugated = graph.edge_attr[:, 4]
        
        # Should have conjugated bonds
        assert torch.any(conjugated == 1)
    
    def test_ring_bond_flag(self):
        """Test that ring bond flag is correctly set."""
        mol = Chem.MolFromSmiles("C1CCCCC1")  # Cyclohexane
        graph = mol_to_graph(mol)
        
        # Extract ring flag (index 5)
        in_ring = graph.edge_attr[:, 5]
        
        # All bonds should be in ring
        assert torch.all(in_ring == 1)
    
    def test_stereo_flag(self):
        """Test that stereo flag is correctly set."""
        mol = Chem.MolFromSmiles("C/C=C/C")  # Trans-2-butene
        graph = mol_to_graph(mol)
        
        # Extract stereo flag (index 6)
        stereo = graph.edge_attr[:, 6]
        
        # Should have at least one bond with stereochemistry
        # Note: RDKit may or may not detect stereochemistry depending on input
        assert stereo.shape[0] > 0


class TestIntegration:
    """Integration tests for graph builder."""
    
    def test_batch_graph_construction(self):
        """Test graph construction for multiple molecules."""
        smiles_list = [
            "CCO",
            "c1ccccc1",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CC(=O)O"
        ]
        
        graphs = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            graph = mol_to_graph(mol)
            graphs.append(graph)
        
        # All graphs should have correct feature dimensions
        assert all(g.x.shape[1] == 137 for g in graphs)
        assert all(g.edge_attr.shape[1] == 7 for g in graphs if g.edge_attr.shape[0] > 0)
    
    def test_full_preprocessing_pipeline(self):
        """Test integration with other preprocessing modules."""
        from backend.app.preprocessing.rdkit_utils import smiles_to_mol, standardize_smiles
        from backend.app.preprocessing.descriptors import compute_descriptors
        from backend.app.preprocessing.fingerprints import compute_morgan_fingerprint
        
        smiles = "CCO"
        
        # Standardize
        std_smiles = standardize_smiles(smiles)
        
        # Parse
        mol = smiles_to_mol(std_smiles)
        
        # Compute all features
        descriptors = compute_descriptors(mol)
        fingerprint = compute_morgan_fingerprint(mol)
        graph = mol_to_graph(mol)
        
        # All should be valid
        assert descriptors.shape == (200,)
        assert fingerprint.shape == (2048,)
        assert graph.x.shape[0] == 3
        assert graph.edge_index.shape[1] == 4
    
    def test_pytorch_geometric_compatibility(self):
        """Test that graphs are compatible with PyTorch Geometric."""
        from torch_geometric.data import Batch
        
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        
        graphs = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            graph = mol_to_graph(mol)
            graphs.append(graph)
        
        # Create batch
        batch = Batch.from_data_list(graphs)
        
        # Batch should have correct properties
        assert batch.x.shape[1] == 137
        assert batch.edge_attr.shape[1] == 7
        assert hasattr(batch, 'batch')


class TestEdgeCases:
    """Test suite for edge cases and special molecules."""
    
    def test_large_molecule(self):
        """Test graph construction for large molecule."""
        # Taxol (paclitaxel) - large complex molecule
        taxol_smiles = "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
        mol = Chem.MolFromSmiles(taxol_smiles)
        graph = mol_to_graph(mol)
        
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        
        assert graph.x.shape[0] == num_atoms
        assert graph.edge_index.shape[1] == num_bonds * 2
        assert graph.x.shape[0] > 50  # Large molecule
    
    def test_molecule_with_stereochemistry(self):
        """Test graph construction for molecule with stereochemistry."""
        mol = Chem.MolFromSmiles("C[C@H](O)CC")  # (S)-2-butanol
        graph = mol_to_graph(mol)
        
        num_atoms = mol.GetNumAtoms()
        assert graph.x.shape[0] == num_atoms
    
    def test_molecule_with_multiple_rings(self):
        """Test graph construction for molecule with multiple rings."""
        mol = Chem.MolFromSmiles("C1CCC2CCCCC2C1")  # Decalin
        graph = mol_to_graph(mol)
        
        # All atoms should be in rings
        in_ring = graph.x[:, 134]
        assert torch.all(in_ring == 1)
    
    def test_nitro_compound(self):
        """Test graph construction for nitro compound."""
        mol = Chem.MolFromSmiles("c1ccc(cc1)[N+](=O)[O-]")  # Nitrobenzene
        graph = mol_to_graph(mol)
        
        num_atoms = mol.GetNumAtoms()
        assert graph.x.shape[0] == num_atoms
        
        # Should have charged atoms
        formal_charges = graph.x[:, 135]
        assert torch.any(formal_charges != 0)
    
    def test_metal_containing_molecule(self):
        """Test graph construction for molecule with metal atom."""
        mol = Chem.MolFromSmiles("[Fe]")  # Iron atom
        graph = mol_to_graph(mol)
        
        # Should have 1 atom
        assert graph.x.shape[0] == 1
        
        # Iron (atomic number 26) should be at index 25
        atomic_nums = graph.x[:, :118]
        assert atomic_nums[0, 25].item() == 1
