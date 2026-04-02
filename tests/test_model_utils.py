"""
Unit tests for model utility functions.

These tests verify the utility functions without requiring model artifacts.
"""

import pytest
import numpy as np

from backend.app.models.ensemble_model import (
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
    
    def test_probs_to_logits_edge_cases(self):
        """Test edge cases for probability to logit conversion."""
        # Test with values close to 0 and 1
        probs = np.array([0.001, 0.999])
        logits = probs_to_logits(probs)
        
        # Should not produce inf or nan
        assert not np.any(np.isinf(logits))
        assert not np.any(np.isnan(logits))
        
        # Test with exact 0 and 1 (should be clipped)
        probs = np.array([0.0, 1.0])
        logits = probs_to_logits(probs)
        
        # Should not produce inf or nan due to clipping
        assert not np.any(np.isinf(logits))
        assert not np.any(np.isnan(logits))
    
    def test_logits_to_probs_edge_cases(self):
        """Test edge cases for logit to probability conversion."""
        # Test with very large and very small logits
        logits = np.array([-100.0, 100.0])
        probs = logits_to_probs(logits)
        
        # Should produce valid probabilities
        assert np.all((probs >= 0) & (probs <= 1))
        
        # Very negative logit should give probability close to 0
        assert probs[0] < 0.01
        
        # Very positive logit should give probability close to 1
        assert probs[1] > 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
