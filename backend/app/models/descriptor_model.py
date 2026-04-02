"""
Descriptor-based toxicity prediction model using LightGBM.

This module provides a wrapper for the trained LightGBM models that predict
toxicity across 12 Tox21 assays using molecular descriptors and fingerprints.
"""

import pickle
import json
from pathlib import Path
from typing import Union

import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler


class DescriptorModel:
    """
    LightGBM-based toxicity prediction model.
    
    This class loads 12 separate LightGBM classifiers (one per Tox21 assay)
    and a StandardScaler for feature normalization. It predicts toxicity
    probabilities from concatenated molecular descriptors and fingerprints.
    
    The model expects:
    - 200 molecular descriptors (RDKit descriptors)
    - 2048 Morgan fingerprint bits (ECFP4, radius=2)
    - 167 MACCS keys
    Total: 2415 features
    
    Examples:
        >>> model = DescriptorModel(
        ...     model_path="ml/artifacts",
        ...     scaler_path="ml/artifacts/lgbm_scaler.pkl"
        ... )
        >>> descriptors = np.random.rand(200)
        >>> fingerprints = np.random.rand(2215)  # 2048 + 167
        >>> probs = model.predict(descriptors, fingerprints)
        >>> probs.shape
        (12,)
    """
    
    def __init__(self, model_path: Union[str, Path], scaler_path: Union[str, Path]):
        """
        Initialize DescriptorModel by loading LightGBM models and scaler.
        
        Args:
            model_path: Path to directory containing LightGBM model files
                       (lgbm_<assay_name>.txt) and metadata (lgbm_metadata.json)
            scaler_path: Path to StandardScaler pickle file (lgbm_scaler.pkl)
            
        Raises:
            FileNotFoundError: If model files or scaler file not found
            ValueError: If model files are invalid or incompatible
        """
        model_path = Path(model_path)
        scaler_path = Path(scaler_path)
        
        # Load metadata to get assay names
        metadata_path = model_path / "lgbm_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            self.assay_names = metadata['assay_names']
        
        # Load individual LightGBM models for each assay
        self.models = []
        for assay_name in self.assay_names:
            model_file = model_path / f"lgbm_{assay_name}.txt"
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            model = lgb.Booster(model_file=str(model_file))
            self.models.append(model)
        
        # Load StandardScaler
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.num_tasks = len(self.models)
    
    def predict(self, descriptors: np.ndarray, fingerprints: np.ndarray) -> np.ndarray:
        """
        Predict toxicity probabilities for all 12 Tox21 assays.
        
        This method:
        1. Concatenates descriptors and fingerprints into a single feature vector
        2. Scales features using the loaded StandardScaler
        3. Runs prediction through all 12 LightGBM models
        4. Returns probabilities for the positive class (toxic)
        
        Args:
            descriptors: Molecular descriptors array of shape (200,)
            fingerprints: Concatenated fingerprints array of shape (2215,)
                         containing Morgan (2048) + MACCS (167) fingerprints
        
        Returns:
            Probability array of shape (12,) with values in [0, 1]
            representing toxicity probability for each Tox21 assay
            
        Raises:
            ValueError: If input shapes are incorrect
            
        Examples:
            >>> descriptors = np.random.rand(200)
            >>> fingerprints = np.random.rand(2215)
            >>> probs = model.predict(descriptors, fingerprints)
            >>> assert probs.shape == (12,)
            >>> assert np.all((probs >= 0) & (probs <= 1))
        """
        # Validate input shapes
        descriptors = np.asarray(descriptors)
        fingerprints = np.asarray(fingerprints)
        
        if descriptors.shape != (200,):
            raise ValueError(f"Expected descriptors shape (200,), got {descriptors.shape}")
        
        if fingerprints.shape != (2215,):
            raise ValueError(f"Expected fingerprints shape (2215,), got {fingerprints.shape}")
        
        # Concatenate features
        features = np.concatenate([descriptors, fingerprints])
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict probabilities for each assay
        probs = np.array([
            model.predict(features_scaled)[0]
            for model in self.models
        ])
        
        return probs
