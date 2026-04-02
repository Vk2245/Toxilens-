"""Unit tests for risk scoring module."""

import numpy as np
import pytest
from backend.app.models.risk_scorer import compute_composite_risk, classify_risk_level


class TestComputeCompositeRisk:
    """Tests for compute_composite_risk function."""
    
    def test_equal_weights_simple_average(self):
        """Test that equal weights produce simple average."""
        probs = np.array([0.5] * 12)
        score = compute_composite_risk(probs)
        assert score == 0.5
    
    def test_equal_weights_varied_probabilities(self):
        """Test equal weights with varied probabilities."""
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.5])
        expected = np.mean(probs)
        score = compute_composite_risk(probs)
        assert np.isclose(score, expected)
    
    def test_custom_weights(self):
        """Test weighted average with custom weights."""
        probs = np.array([1.0] + [0.0] * 11)
        weights = np.array([0.5] + [0.5/11] * 11)
        score = compute_composite_risk(probs, weights)
        assert np.isclose(score, 0.5)
    
    def test_all_zeros(self):
        """Test with all zero probabilities."""
        probs = np.zeros(12)
        score = compute_composite_risk(probs)
        assert score == 0.0
    
    def test_all_ones(self):
        """Test with all maximum probabilities."""
        probs = np.ones(12)
        score = compute_composite_risk(probs)
        assert score == 1.0
    
    def test_invalid_length(self):
        """Test that invalid array length raises ValueError."""
        probs = np.array([0.5] * 10)
        with pytest.raises(ValueError, match="Expected 12 assay probabilities"):
            compute_composite_risk(probs)
    
    def test_invalid_probability_range_negative(self):
        """Test that negative probabilities raise ValueError."""
        probs = np.array([-0.1] + [0.5] * 11)
        with pytest.raises(ValueError, match="must be in range"):
            compute_composite_risk(probs)
    
    def test_invalid_probability_range_above_one(self):
        """Test that probabilities > 1 raise ValueError."""
        probs = np.array([1.1] + [0.5] * 11)
        with pytest.raises(ValueError, match="must be in range"):
            compute_composite_risk(probs)
    
    def test_invalid_weights_length(self):
        """Test that invalid weights length raises ValueError."""
        probs = np.array([0.5] * 12)
        weights = np.array([0.1] * 10)
        with pytest.raises(ValueError, match="Expected 12 weights"):
            compute_composite_risk(probs, weights)
    
    def test_invalid_weights_sum(self):
        """Test that weights not summing to 1.0 raise ValueError."""
        probs = np.array([0.5] * 12)
        weights = np.array([0.1] * 12)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            compute_composite_risk(probs, weights)
    
    def test_list_input(self):
        """Test that list inputs are converted to arrays."""
        probs = [0.5] * 12
        score = compute_composite_risk(probs)
        assert score == 0.5


class TestClassifyRiskLevel:
    """Tests for classify_risk_level function."""
    
    def test_high_risk_above_threshold(self):
        """Test HIGH classification for scores > 0.6."""
        assert classify_risk_level(0.61) == "HIGH"
        assert classify_risk_level(0.7) == "HIGH"
        assert classify_risk_level(0.9) == "HIGH"
        assert classify_risk_level(1.0) == "HIGH"
    
    def test_medium_risk_upper_boundary(self):
        """Test MEDIUM classification at upper boundary (0.6)."""
        assert classify_risk_level(0.6) == "MEDIUM"
    
    def test_medium_risk_lower_boundary(self):
        """Test MEDIUM classification at lower boundary (0.35)."""
        assert classify_risk_level(0.35) == "MEDIUM"
    
    def test_medium_risk_middle(self):
        """Test MEDIUM classification in middle of range."""
        assert classify_risk_level(0.4) == "MEDIUM"
        assert classify_risk_level(0.5) == "MEDIUM"
        assert classify_risk_level(0.55) == "MEDIUM"
    
    def test_low_risk_below_threshold(self):
        """Test LOW classification for scores < 0.35."""
        assert classify_risk_level(0.34) == "LOW"
        assert classify_risk_level(0.2) == "LOW"
        assert classify_risk_level(0.1) == "LOW"
        assert classify_risk_level(0.0) == "LOW"
    
    def test_boundary_precision(self):
        """Test boundary conditions with high precision."""
        # Just above 0.6 should be HIGH
        assert classify_risk_level(0.600001) == "HIGH"
        # Just below 0.35 should be LOW
        assert classify_risk_level(0.349999) == "LOW"
    
    def test_invalid_score_negative(self):
        """Test that negative scores raise ValueError."""
        with pytest.raises(ValueError, match="must be in range"):
            classify_risk_level(-0.1)
    
    def test_invalid_score_above_one(self):
        """Test that scores > 1 raise ValueError."""
        with pytest.raises(ValueError, match="must be in range"):
            classify_risk_level(1.1)


class TestIntegration:
    """Integration tests combining both functions."""
    
    def test_high_risk_workflow(self):
        """Test complete workflow for high-risk molecule."""
        probs = np.array([0.8, 0.9, 0.7, 0.6, 0.5, 0.8, 0.9, 0.7, 0.6, 0.8, 0.7, 0.9])
        score = compute_composite_risk(probs)
        risk_level = classify_risk_level(score)
        assert risk_level == "HIGH"
        assert score > 0.6
    
    def test_medium_risk_workflow(self):
        """Test complete workflow for medium-risk molecule."""
        probs = np.array([0.4, 0.5, 0.3, 0.6, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6, 0.4, 0.3])
        score = compute_composite_risk(probs)
        risk_level = classify_risk_level(score)
        assert risk_level == "MEDIUM"
        assert 0.35 <= score <= 0.6
    
    def test_low_risk_workflow(self):
        """Test complete workflow for low-risk molecule."""
        probs = np.array([0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.3])
        score = compute_composite_risk(probs)
        risk_level = classify_risk_level(score)
        assert risk_level == "LOW"
        assert score < 0.35
