"""
Ensemble model combining ChemBERTa-2, GNN, and LightGBM predictions.

This module implements weighted logit fusion to combine predictions from
three complementary models for improved toxicity prediction.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data

from ml.models.gnn import ToxGNN


class EnsembleModel:
    """
    Ensemble model combining three toxicity prediction models.
    
    Uses weighted logit fusion to combine:
    - ChemBERTa-2 (transformer on SMILES)
    - ToxGNN (graph neural network)
    - LightGBM (gradient boosting on descriptors)
    """
    
    def __init__(
        self,
        chemberta_path: str,
        gnn_path: str,
        lgbm_path: str,
        weights_path: str,
        device: str = 'cpu'
    ):
        """
        Initialize ensemble model.
        
        Args:
            chemberta_path: Path to fine-tuned ChemBERTa model directory
            gnn_path: Path to GNN checkpoint (.pt file)
            lgbm_path: Path to LightGBM artifacts directory
            weights_path: Path to ensemble weights JSON file
            device: Device for PyTorch models ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        
        # Load ensemble weights
        with open(weights_path, 'r') as f:
            weights_data = json.load(f)
            self.weights = np.array(weights_data['weights'])  # [w_chemberta, w_gnn, w_lgbm]
        
        # Load ChemBERTa model
        self._load_chemberta(chemberta_path)
        
        # Load GNN model
        self._load_gnn(gnn_path)
        
        # Load LightGBM models
        self._load_lgbm(lgbm_path)
        
        self.num_tasks = 12
    
    def _load_chemberta(self, model_path: str):
        """Load fine-tuned ChemBERTa model."""
        self.chemberta_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.chemberta_encoder = AutoModel.from_pretrained(model_path).to(self.device)
        
        # Load classifier head
        checkpoint = torch.load(f"{model_path}/classifier.pt", map_location=self.device)
        hidden_size = self.chemberta_encoder.config.hidden_size
        self.chemberta_classifier = torch.nn.Linear(hidden_size, self.num_tasks).to(self.device)
        self.chemberta_classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        self.chemberta_encoder.eval()
        self.chemberta_classifier.eval()
    
    def _load_gnn(self, checkpoint_path: str):
        """Load trained GNN model."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model with saved hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.gnn_model = ToxGNN(
            node_feat_dim=hyperparams.get('node_feat_dim', 137),  # Use saved or default
            edge_feat_dim=hyperparams.get('edge_feat_dim', 7),    # Use saved or default
            hidden_dim=hyperparams['hidden_dim'],
            num_layers=hyperparams['num_layers'],
            num_tasks=hyperparams['num_tasks'],
            dropout=hyperparams['dropout']
        ).to(self.device)
        
        self.gnn_model.load_state_dict(checkpoint['model_state_dict'])
        self.gnn_model.eval()
    
    def _load_lgbm(self, artifacts_dir: str):
        """Load LightGBM models and scaler."""
        # Load metadata to get assay names
        with open(f"{artifacts_dir}/lgbm_metadata.json", 'r') as f:
            metadata = json.load(f)
            assay_names = metadata['assay_names']
        
        # Load individual models
        self.lgbm_models = []
        for assay_name in assay_names:
            model_path = f"{artifacts_dir}/lgbm_{assay_name}.txt"
            model = lgb.Booster(model_file=model_path)
            self.lgbm_models.append(model)
        
        # Load scaler
        with open(f"{artifacts_dir}/lgbm_scaler.pkl", 'rb') as f:
            self.lgbm_scaler = pickle.load(f)
    
    @torch.no_grad()
    def predict_chemberta(self, smiles: str) -> np.ndarray:
        """
        Predict using ChemBERTa model.
        
        Args:
            smiles: SMILES string
        
        Returns:
            Logits array of shape (num_tasks,)
        """
        # Tokenize
        inputs = self.chemberta_tokenizer(
            smiles,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.chemberta_encoder(**inputs)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.chemberta_classifier(pooled_output)
        
        return logits.cpu().numpy()[0]
    
    @torch.no_grad()
    def predict_gnn(self, graph: Data) -> np.ndarray:
        """
        Predict using GNN model.
        
        Args:
            graph: PyTorch Geometric Data object
        
        Returns:
            Logits array of shape (num_tasks,)
        """
        graph = graph.to(self.device)
        logits = self.gnn_model(graph)
        return logits.cpu().numpy()[0]
    
    def predict_lgbm(self, features: np.ndarray) -> np.ndarray:
        """
        Predict using LightGBM models.
        
        Args:
            features: Feature vector (descriptors + fingerprints)
        
        Returns:
            Logits array of shape (num_tasks,)
        """
        # Scale features
        features_scaled = self.lgbm_scaler.transform(features.reshape(1, -1))
        
        # Predict probabilities
        probs = np.array([
            model.predict(features_scaled)[0]
            for model in self.lgbm_models
        ])
        
        # Convert probabilities to logits
        probs = np.clip(probs, 1e-7, 1 - 1e-7)  # Avoid log(0)
        logits = np.log(probs / (1 - probs))
        
        return logits
    
    def predict(
        self,
        smiles: str,
        graph: Data,
        features: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Predict toxicity using ensemble of all three models.
        
        Args:
            smiles: SMILES string
            graph: PyTorch Geometric Data object
            features: Feature vector (descriptors + fingerprints)
        
        Returns:
            Dictionary containing:
                - 'probabilities': Ensemble probabilities (num_tasks,)
                - 'logits': Ensemble logits (num_tasks,)
                - 'individual_probs': Individual model probabilities (3, num_tasks)
                - 'individual_logits': Individual model logits (3, num_tasks)
        """
        # Get predictions from each model
        chemberta_logits = self.predict_chemberta(smiles)
        gnn_logits = self.predict_gnn(graph)
        lgbm_logits = self.predict_lgbm(features)
        
        # Stack individual logits
        individual_logits = np.stack([chemberta_logits, gnn_logits, lgbm_logits], axis=0)
        
        # Weighted logit fusion
        ensemble_logits = np.average(individual_logits, axis=0, weights=self.weights)
        
        # Convert to probabilities
        ensemble_probs = 1 / (1 + np.exp(-ensemble_logits))
        individual_probs = 1 / (1 + np.exp(-individual_logits))
        
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
    Convert logits to probabilities.
    
    Args:
        logits: Logits
    
    Returns:
        Probabilities in [0, 1]
    """
    probs = 1 / (1 + np.exp(-logits))
    return probs


if __name__ == "__main__":
    # Test ensemble model loading
    print("Testing ensemble model initialization...")
    
    # Note: This will fail if models haven't been trained yet
    try:
        ensemble = EnsembleModel(
            chemberta_path="ml/artifacts/chemberta_finetuned",
            gnn_path="ml/artifacts/gnn_best.pt",
            lgbm_path="ml/artifacts",
            weights_path="ml/artifacts/ensemble_weights.json",
            device='cpu'
        )
        print("✓ Ensemble model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load ensemble model: {e}")
        print("  (This is expected if models haven't been trained yet)")
