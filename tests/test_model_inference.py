"""
Unit tests for backend model inference modules.

These tests verify the structure and interfaces of the model wrappers.
Note: These tests will fail if model artifacts are not present.
"""

import pytest
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path

from backend.app.models.descriptor_model import DescriptorModel
from backend.app.models.gnn_model import ToxGNN, GNNModelWrapper
from backend.app.models.transformer_model import ChemBERTaModel
from backend.app.models.ensemble_model import (
    EnsembleModel,
    probs_to_logits,
    logits_to_probs,
    logit_fusion
)


class TestUtilityFunctions:
    """Test utility functions for logit/probability conversion."""
    
    def test_probs_to_logits(self):
        """Test probability to logit conversion."""
        probs = np.array([0.1, 0.5, 0.9])
        logits = probs_to_logits(probs)
        
        # Check shape
        assert logits.shape == probs.shape
        
        # Check conversion is correct
        # logit(0.5) should be 0
        assert np.isclose(logits[1], 0.0, atol=1e-6)
        
        # logit(0.9) should be positive
        assert logits[2] > 0
        
        # logit(0.1) should be negative
        assert logits[0] < 0
    
    def test_logits_to_probs(self):
        """Test logit to probability conversion."""
        logits = np.array([-2.0, 0.0, 2.0])
        probs = logits_to_probs(logits)
        
        # Check shape
        assert probs.shape == logits.shape
        
        # Check all probabilities in [0, 1]
        assert np.all((probs >= 0) & (probs <= 1))
        
        # Check conversion is correct
        # sigmoid(0) should be 0.5
        assert np.isclose(probs[1], 0.5, atol=1e-6)
        
        # sigmoid(2) should be > 0.5
        assert probs[2] > 0.5
        
        # sigmoid(-2) should be < 0.5
        assert probs[0] < 0.5
    
    def test_roundtrip_conversion(self):
        """Test that probs -> logits -> probs is identity."""
        original_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        logits = probs_to_logits(original_probs)
        recovered_probs = logits_to_probs(logits)
        
        # Check roundtrip is accurate
        assert np.allclose(original_probs, recovered_probs, atol=1e-6)
    
    def test_logit_fusion(self):
        """Test weighted logit fusion."""
        # Create mock logits from 3 models
        logits_list = np.array([
            [1.0, 2.0, 3.0],  # Model 1
            [1.5, 2.5, 3.5],  # Model 2
            [0.5, 1.5, 2.5],  # Model 3
        ])
        
        # Equal weights
        weights = np.array([1.0, 1.0, 1.0])
        fused = logit_fusion(logits_list, weights)
        
        # Check shape
        assert fused.shape == (3,)
        
        # Check fusion is average with equal weights
        expected = np.mean(logits_list, axis=0)
        assert np.allclose(fused, expected, atol=1e-6)
        
        # Test with different weights
        weights = np.array([0.5, 0.3, 0.2])
        fused = logit_fusion(logits_list, weights)
        
        # Check weights are normalized
        expected = np.average(logits_list, axis=0, weights=weights / weights.sum())
        assert np.allclose(fused, expected, atol=1e-6)


class TestToxGNN:
    """Test ToxGNN model architecture."""
    
    def test_model_initialization(self):
        """Test ToxGNN can be initialized."""
        model = ToxGNN(
            node_feat_dim=133,
            edge_feat_dim=7,
            hidden_dim=256,
            num_layers=4,
            num_tasks=12,
            dropout=0.3
        )
        
        assert model.node_feat_dim == 133
        assert model.edge_feat_dim == 7
        assert model.hidden_dim == 256
        assert model.num_tasks == 12
    
    def test_forward_pass(self):
        """Test ToxGNN forward pass with dummy data."""
        model = ToxGNN(
            node_feat_dim=133,
            edge_feat_dim=7,
            hidden_dim=256,
            num_layers=4,
            num_tasks=12,
            dropout=0.3
        )
        model.eval()
        
        # Create dummy graph data
        num_nodes = 10
        num_edges = 18
        
        x = torch.randn(num_nodes, 133)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 7)
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        
        # Forward pass
        with torch.no_grad():
            logits = model(data)
        
        # Check output shape
        assert logits.shape == (1, 12)


class TestModelWrappers:
    """Test model wrapper classes (will skip if artifacts not present)."""
    
    def test_descriptor_model_interface(self):
        """Test DescriptorModel interface (will fail if artifacts missing)."""
        artifacts_path = Path("ml/artifacts")
        
        if not artifacts_path.exists():
            pytest.skip("Model artifacts not found")
        
        metadata_path = artifacts_path / "lgbm_metadata.json"
        scaler_path = artifacts_path / "lgbm_scaler.pkl"
        
        if not metadata_path.exists() or not scaler_path.exists():
            pytest.skip("LightGBM artifacts not found")
        
        # Initialize model
        model = DescriptorModel(
            model_path=artifacts_path,
            scaler_path=scaler_path
        )
        
        # Test prediction interface
        descriptors = np.random.rand(200)
        fingerprints = np.random.rand(2215)
        
        probs = model.predict(descriptors, fingerprints)
        
        # Check output
        assert probs.shape == (12,)
        assert np.all((probs >= 0) & (probs <= 1))
    
    def test_gnn_model_interface(self):
        """Test GNNModelWrapper interface (will fail if artifacts missing)."""
        checkpoint_path = Path("ml/artifacts/gnn_best.pt")
        
        if not checkpoint_path.exists():
            pytest.skip("GNN checkpoint not found")
        
        # Initialize model
        model = GNNModelWrapper(checkpoint_path, device='cpu')
        
        # Create dummy graph
        num_nodes = 10
        num_edges = 18
        
        x = torch.randn(num_nodes, 133)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 7)
        
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Test prediction
        logits = model.predict(graph)
        
        # Check output
        assert logits.shape == (12,)
    
    def test_chemberta_model_interface(self):
        """Test ChemBERTaModel interface (will fail if artifacts missing)."""
        model_path = Path("ml/artifacts/chemberta_finetuned")
        
        if not model_path.exists():
            pytest.skip("ChemBERTa model not found")
        
        # Initialize model
        model = ChemBERTaModel(model_path, device='cpu')
        
        # Test prediction
        probs = model.predict("CCO")  # Ethanol
        
        # Check output
        assert probs.shape == (12,)
        assert np.all((probs >= 0) & (probs <= 1))
    
    def test_ensemble_model_interface(self):
        """Test EnsembleModel interface (will fail if artifacts missing)."""
        artifacts_path = Path("ml/artifacts")
        
        if not artifacts_path.exists():
            pytest.skip("Model artifacts not found")
        
        chemberta_path = artifacts_path / "chemberta_finetuned"
        gnn_path = artifacts_path / "gnn_best.pt"
        weights_path = artifacts_path / "ensemble_weights.json"
        
        if not all([chemberta_path.exists(), gnn_path.exists(), weights_path.exists()]):
            pytest.skip("Ensemble artifacts not found")
        
        # Initialize ensemble
        ensemble = EnsembleModel(
            chemberta_path=chemberta_path,
            gnn_path=gnn_path,
            lgbm_path=artifacts_path,
            weights_path=weights_path,
            device='cpu'
        )
        
        # Create dummy inputs
        smiles = "CCO"
        descriptors = np.random.rand(200)
        fingerprints = np.random.rand(2215)
        
        num_nodes = 10
        num_edges = 18
        x = torch.randn(num_nodes, 133)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 7)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Test prediction
        result = ensemble.predict(smiles, graph, descriptors, fingerprints)
        
        # Check output structure
        assert 'probabilities' in result
        assert 'logits' in result
        assert 'individual_probs' in result
        assert 'individual_logits' in result
        
        # Check shapes
        assert result['probabilities'].shape == (12,)
        assert result['logits'].shape == (12,)
        assert result['individual_probs'].shape == (3, 12)
        assert result['individual_logits'].shape == (3, 12)
        
        # Check probability bounds
        assert np.all((result['probabilities'] >= 0) & (result['probabilities'] <= 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
