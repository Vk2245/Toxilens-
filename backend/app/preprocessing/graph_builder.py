"""
Graph builder module for converting RDKit molecules to PyTorch Geometric graphs.

This module provides functionality to convert molecular structures into graph
representations suitable for graph neural network (GNN) models. Each atom becomes
a node with features, and each bond becomes an edge with features.
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data


def mol_to_graph(mol: Chem.Mol) -> Data:
    """
    Convert RDKit molecule to PyTorch Geometric Data object.
    
    This function constructs a molecular graph where atoms are nodes and bonds
    are edges. Node features encode atomic properties, and edge features encode
    bond properties.
    
    Node features (per atom):
        - atomic_num: One-hot encoding of atomic number (118 elements)
        - degree: Atom degree (0-10)
        - hybridization: SP, SP2, SP3, or other
        - is_aromatic: Boolean flag
        - in_ring: Boolean flag
        - formal_charge: Formal charge on atom
        - num_Hs: Number of hydrogen atoms
    
    Edge features (per bond):
        - bond_type: Single, double, triple, or aromatic
        - is_conjugated: Boolean flag
        - is_in_ring: Boolean flag
        - stereo: Stereochemistry information
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        PyTorch Geometric Data object with:
            - x: Node feature matrix (num_atoms, num_node_features)
            - edge_index: Edge connectivity in COO format (2, num_edges)
            - edge_attr: Edge feature matrix (num_edges, num_edge_features)
            
    Raises:
        ValueError: If molecule is None or invalid
        
    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> graph = mol_to_graph(mol)
        >>> graph.x.shape
        torch.Size([3, 133])
        >>> graph.edge_index.shape
        torch.Size([2, 4])
    """
    if mol is None:
        raise ValueError("Molecule cannot be None")
    
    # Extract atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = _get_atom_features(atom)
        atom_features.append(features)
    
    # Convert to tensor
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Extract bond features and connectivity
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        # Get bond indices (both directions for undirected graph)
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Add both directions
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        
        # Get bond features
        bond_feat = _get_bond_features(bond)
        
        # Add features for both directions
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)
    
    # Handle molecules with no bonds (single atoms)
    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 7), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data


def _get_atom_features(atom: Chem.Atom) -> list:
    """
    Extract feature vector for a single atom.
    
    Features:
        - Atomic number (one-hot, 118 elements)
        - Degree (0-10)
        - Hybridization (SP, SP2, SP3, other)
        - Is aromatic
        - In ring
        - Formal charge
        - Number of hydrogens
    
    Args:
        atom: RDKit Atom object
        
    Returns:
        List of atom features (length 133)
    """
    features = []
    
    # Atomic number (one-hot encoding for elements 1-118)
    atomic_num = atom.GetAtomicNum()
    atomic_num_onehot = [0] * 118
    if 1 <= atomic_num <= 118:
        atomic_num_onehot[atomic_num - 1] = 1
    features.extend(atomic_num_onehot)
    
    # Degree (0-10)
    degree = atom.GetDegree()
    degree_onehot = [0] * 11
    if 0 <= degree <= 10:
        degree_onehot[degree] = 1
    features.extend(degree_onehot)
    
    # Hybridization (SP, SP2, SP3, other)
    hybridization = atom.GetHybridization()
    hyb_features = [
        int(hybridization == rdchem.HybridizationType.SP),
        int(hybridization == rdchem.HybridizationType.SP2),
        int(hybridization == rdchem.HybridizationType.SP3),
        int(hybridization not in [rdchem.HybridizationType.SP, 
                                   rdchem.HybridizationType.SP2, 
                                   rdchem.HybridizationType.SP3])
    ]
    features.extend(hyb_features)
    
    # Is aromatic
    features.append(int(atom.GetIsAromatic()))
    
    # In ring
    features.append(int(atom.IsInRing()))
    
    # Formal charge
    features.append(atom.GetFormalCharge())
    
    # Number of hydrogens
    features.append(atom.GetTotalNumHs())
    
    return features


def _get_bond_features(bond: Chem.Bond) -> list:
    """
    Extract feature vector for a single bond.
    
    Features:
        - Bond type (single, double, triple, aromatic)
        - Is conjugated
        - Is in ring
        - Stereo (none, E, Z, other)
    
    Args:
        bond: RDKit Bond object
        
    Returns:
        List of bond features (length 7)
    """
    features = []
    
    # Bond type (one-hot: single, double, triple, aromatic)
    bond_type = bond.GetBondType()
    bond_type_features = [
        int(bond_type == rdchem.BondType.SINGLE),
        int(bond_type == rdchem.BondType.DOUBLE),
        int(bond_type == rdchem.BondType.TRIPLE),
        int(bond_type == rdchem.BondType.AROMATIC)
    ]
    features.extend(bond_type_features)
    
    # Is conjugated
    features.append(int(bond.GetIsConjugated()))
    
    # Is in ring
    features.append(int(bond.IsInRing()))
    
    # Stereo (none, E, Z, other)
    stereo = bond.GetStereo()
    features.append(int(stereo != rdchem.BondStereo.STEREONONE))
    
    return features
