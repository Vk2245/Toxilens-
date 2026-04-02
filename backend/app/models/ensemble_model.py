"""
Ensemble model combining ChemBERTa-2, GNN, and LightGBM predictions.

This module implements weighted logit fusion to combine predictions from
three complementary models for improved toxicity prediction.
"""

import json
from pathlib import Path
from typing import Dict, Any, Union

import numpy as np
import torch
from torch_geometric.data import Data

from backend.app.models.descriptor_model import DescriptorModel
from backend.app.models.gnn_model import GNNModelWrapper
from backend.app.models.transformer_model import ChemBERTaModel


class EnsembleModel:
    """
    Ensemble model combining three toxicity prediction models.
    
    Uses weighted logit fusion to combine:
    - ChemBERTa-2 (transformer on SMILES)
    - ToxGNN (graph neural network)
    - LightGBM (gradient boosting on descriptors)
    
    The ensemble:
    1. Gets predictions from all three models
    2. Converts probabilities to logits
    3. Applies learned weights to logits
    4. Averages weighted logits
    5. Converts back to probabilities
    
    Examples:
        >>> ensemble = EnsembleModel(
        ...     chemberta_path="ml/artifacts/chemberta_finetuned",
        ...     gnn_path="ml/artifacts/gnn_best.pt",
        ...     lgbm_path="ml/artifacts",
        ...     weights_path="ml/artifacts/ensemble_weights.json",
        ...     device='cpu'
        ... )
        >>> result = ensemble.predict(
        ...     smiles="CCO",
        ...     graph=graph_data,
        ...     features=features
        ... )
        >>> result['probabilities'].shape
        (12,)
    """
    
    def __init__(
        self,
        chemberta_path: Union[str, Path],
        gnn_path: Union[str, Path],
        lgbm_path: Union[str, Path],
        weights_path: Union[str, Path],
        device: str = 'cpu'
    ):
        """
        Initialize ensemble model by loading all three models and weights.
        
        Args:
            chemberta_path: Path to fine-tuned ChemBERTa model directory
            gnn_path: Path to GNN checkpoint (.pt file)
            lgbm_path: Path to LightGBM artifacts directory
            weights_path: Path to ensemble weights JSON file
            device: Device for PyTorch models ('cpu' or 'cuda')
            
        Raises:
            FileNotFoundError: If any model files or weights file not found
            ValueError: If model files are invalid or incompatible
        """
        self.device = torch.device(device)
        
        # Load ensemble weights
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Ensemble weights file not found: {weights_path}")
        
        with open(weights_path, 'r') as f:
            weights_data = json.load(f)
            self.weights = np.array(weights_data['weights'])  # [w_chemberta, w_gnn, w_lgbm]
        
        # Validate weights
        if len(self.weights) != 3:
            raise ValueError(f"Expected 3 ensemble weights, got {len(self.weights)}")
        
        # Load ChemBERTa model
        self.chemberta_model = ChemBERTaModel(chemberta_path, device=device)
        
        # Load GNN model
        self.gnn_model = GNNModelWrapper(gnn_path, device=device)
        
        # Load LightGBM model
        lgbm_path = Path(lgbm_path)
        scaler_path = lgbm_path / "lgbm_scaler.pkl"
        self.lgbm_model = DescriptorModel(lgbm_path, scaler_path)
        
        self.num_tasks = 12
    
    def predict_chemberta(self, smiles: str) -> np.ndarray:
        """
        Predict using ChemBERTa model.
        
        Args:
            smiles: SMILES string
        
        Returns:
            Probability array of shape (12,)
        """
        return self.chemberta_model.predict(smiles)
    
    def predict_gnn(self, graph: Data) -> np.ndarray:
        """
        Predict using GNN model.
        
        Args:
            graph: PyTorch Geometric Data object
        
        Returns:
            Logits array of shape (12,)
        """
        return self.gnn_model.predict(graph)
    
    def predict_lgbm(self, descriptors: np.ndarray, fingerprints: np.ndarray) -> np.ndarray:
        """
        Predict using LightGBM model.
        
        Args:
            descriptors: Molecular descriptors array of shape (200,)
            fingerprints: Concatenated fingerprints array of shape (2215,)
        
        Returns:
            Probability array of shape (12,)
        """
        return self.lgbm_model.predict(descriptors, fingerprints)
    
    def predict(
        self,
        smiles: str,
        graph: Data,
        descriptors: np.ndarray,
        fingerprints: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Predict toxicity using ensemble of all three models.
        
        This method:
        1. Gets predictions from ChemBERTa (probabilities)
        2. Gets predictions from GNN (logits)
        3. Gets predictions from LightGBM (probabilities)
        4. Converts all to logits
        5. Applies weighted logit fusion
        6. Converts back to probabilities
        
        Args:
            smiles: SMILES string
            graph: PyTorch Geometric Data object
            descriptors: Molecular descriptors array of shape (200,)
            fingerprints: Concatenated fingerprints array of shape (2215,)
        
        Returns:
            Dictionary containing:
                - 'probabilities': Ensemble probabilities (12,)
                - 'logits': Ensemble logits (12,)
                - 'individual_probs': Individual model probabilities (3, 12)
                - 'individual_logits': Individual model logits (3, 12)
                
        Examples:
            >>> result = ensemble.predict(smiles, graph, descriptors, fingerprints)
            >>> result['probabilities'].shape
            (12,)
            >>> result['individual_probs'].shape
            (3, 12)
        """
        # Get predictions from each model
        chemberta_probs = self.predict_chemberta(smiles)
        gnn_logits = self.predict_gnn(graph)
        lgbm_probs = self.predict_lgbm(descriptors, fingerprints)
        
        # Convert probabilities to logits
        chemberta_logits = probs_to_logits(chemberta_probs)
        lgbm_logits = probs_to_logits(lgbm_probs)
        
        # Convert GNN logits to probabilities for individual_probs
        gnn_probs = logits_to_probs(gnn_logits)
        
        # Stack individual logits
        individual_logits = np.stack([chemberta_logits, gnn_logits, lgbm_logits], axis=0)
        
        # Weighted logit fusion
        ensemble_logits = np.average(individual_logits, axis=0, weights=self.weights)
        
        # Convert to probabilities
        ensemble_probs = logits_to_probs(ensemble_logits)
        individual_probs = np.stack([chemberta_probs, gnn_probs, lgbm_probs], axis=0)
        
        return {
            'probabilities': ensemble_probs,
            'logits': ensemble_logits,
            'individual_probs': individual_probs,
            'individual_logits': individual_logits
        }


def logit_fusion(logits_list: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Perform weighted logit fusion.
    
    Args:
        logits_list: Array of logits from multiple models (num_models, num_tasks)
        weights: Weights for each model (num_models,)
    
    Returns:
        Fused logits (num_tasks,)
    """
    # Ensure weights sum to 1
    weights = weights / weights.sum()
    
    # Weighted average of logits
    fused_logits = np.average(logits_list, axis=0, weights=weights)
    
    return fused_logits


def probs_to_logits(probs: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Convert probabilities to logits.
    
    Args:
        probs: Probabilities in [0, 1]
        eps: Small constant to avoid log(0)
    
    Returns:
        Logits
    """
    probs = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    return logits


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """
    Convert logits to probabilities using sigmoid.
    
    Args:
        logits: Logits
    
    Returns:
        Probabilities in [0, 1]
    """
    probs = 1 / (1 + np.exp(-logits))
    return probs
