"""Risk scoring module for computing composite toxicity risk scores and classifications."""

import numpy as np
from typing import Literal


def compute_composite_risk(assay_probabilities: np.ndarray, weights: np.ndarray = None) -> float:
    """
    Compute composite risk score as weighted average of 12 assay probabilities.
    
    Implements Requirement 2.7: THE ML_Ensemble SHALL compute a Composite_Risk_Score 
    as the weighted average of 12 assay probabilities.
    
    Args:
        assay_probabilities: Array of 12 toxicity probabilities, one per Tox21 assay.
                           Values should be in range [0, 1].
        weights: Optional array of 12 weights for weighted average. If None, uses
                equal weights (simple average). Weights should sum to 1.0.
    
    Returns:
        Composite risk score in range [0, 1].
    
    Raises:
        ValueError: If assay_probabilities is not length 12 or contains invalid values.
    """
    assay_probabilities = np.asarray(assay_probabilities)
    
    if assay_probabilities.shape != (12,):
        raise ValueError(f"Expected 12 assay probabilities, got {assay_probabilities.shape}")
    
    if np.any((assay_probabilities < 0) | (assay_probabilities > 1)):
        raise ValueError("Assay probabilities must be in range [0, 1]")
    
    if weights is None:
        # Equal weights (simple average)
        weights = np.ones(12) / 12
    else:
        weights = np.asarray(weights)
        if weights.shape != (12,):
            raise ValueError(f"Expected 12 weights, got {weights.shape}")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {weights.sum()}")
    
    composite_score = np.dot(assay_probabilities, weights)
    return float(composite_score)


def classify_risk_level(composite_risk_score: float) -> Literal["HIGH", "MEDIUM", "LOW"]:
    """
    Classify risk level based on composite risk score thresholds.
    
    Implements Requirements:
    - 2.8: WHEN the Composite_Risk_Score is greater than 0.6, 
           THE Platform SHALL classify the Risk_Level as HIGH
    - 2.9: WHEN the Composite_Risk_Score is between 0.35 and 0.6 inclusive, 
           THE Platform SHALL classify the Risk_Level as MEDIUM
    - 2.10: WHEN the Composite_Risk_Score is less than 0.35, 
            THE Platform SHALL classify the Risk_Level as LOW
    
    Args:
        composite_risk_score: Composite risk score in range [0, 1].
    
    Returns:
        Risk level classification: "HIGH", "MEDIUM", or "LOW".
    
    Raises:
        ValueError: If composite_risk_score is not in range [0, 1].
    """
    if not (0 <= composite_risk_score <= 1):
        raise ValueError(f"Composite risk score must be in range [0, 1], got {composite_risk_score}")
    
    if composite_risk_score > 0.6:
        return "HIGH"
    elif composite_risk_score >= 0.35:
        return "MEDIUM"
    else:
        return "LOW"
