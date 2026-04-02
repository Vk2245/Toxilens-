"""
ChemBERTa-2 transformer model for Tox21 toxicity prediction.

This module provides a wrapper for the fine-tuned ChemBERTa-2 model that predicts
toxicity across 12 Tox21 assays from SMILES strings.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union

import numpy as np
from transformers import AutoTokenizer, AutoModel


class ChemBERTaModel:
    """
    Fine-tuned ChemBERTa-2 model for toxicity prediction.
    
    This class loads a fine-tuned ChemBERTa-2 transformer model and provides
    a simple interface for predicting toxicity from SMILES strings.
    
    The model:
    1. Tokenizes SMILES strings using the ChemBERTa tokenizer
    2. Encodes SMILES into 768-dimensional embeddings
    3. Passes embeddings through a classification head
    4. Returns probabilities for 12 Tox21 assays
    
    Examples:
        >>> model = ChemBERTaModel("ml/artifacts/chemberta_finetuned", device="cuda")
        >>> probs = model.predict("CCO")  # Ethanol
        >>> probs.shape
        (12,)
    """
    
    def __init__(self, model_path: Union[str, Path], device: str = 'cpu'):
        """
        Initialize ChemBERTa model by loading fine-tuned model and tokenizer.
        
        Args:
            model_path: Path to fine-tuned ChemBERTa model directory containing:
                       - tokenizer files (tokenizer_config.json, vocab.txt, etc.)
                       - model files (config.json, pytorch_model.bin)
                       - classifier head (classifier.pt)
            device: Device for inference ('cpu' or 'cuda')
            
        Raises:
            FileNotFoundError: If model files not found
            ValueError: If model files are invalid or incompatible
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Set device
        self.device = torch.device(device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        
        # Load encoder (ChemBERTa base model)
        self.encoder = AutoModel.from_pretrained(str(model_path))
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        
        # Load classifier head
        classifier_path = model_path / "classifier.pt"
        if not classifier_path.exists():
            raise FileNotFoundError(f"Classifier head not found: {classifier_path}")
        
        checkpoint = torch.load(classifier_path, map_location=self.device)
        
        # Initialize classifier head
        hidden_size = self.encoder.config.hidden_size
        self.num_tasks = 12
        self.classifier = nn.Linear(hidden_size, self.num_tasks).to(self.device)
        
        # Load classifier weights
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.classifier.eval()
    
    @torch.no_grad()
    def predict(self, smiles: str) -> np.ndarray:
        """
        Predict toxicity probabilities from SMILES string.
        
        This method:
        1. Tokenizes the SMILES string with padding and truncation
        2. Encodes the SMILES using the ChemBERTa encoder
        3. Extracts the CLS token embedding (first token)
        4. Passes through the classification head
        5. Applies sigmoid to get probabilities
        
        Args:
            smiles: SMILES string representing the molecule
        
        Returns:
            Probability array of shape (12,) with values in [0, 1]
            representing toxicity probability for each Tox21 assay
            
        Examples:
            >>> probs = model.predict("CCO")
            >>> assert probs.shape == (12,)
            >>> assert np.all((probs >= 0) & (probs <= 1))
        """
        # Tokenize SMILES
        inputs = self.tokenizer(
            smiles,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Encode SMILES
        outputs = self.encoder(**inputs)
        
        # Extract CLS token embedding (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (1, hidden_size)
        
        # Classification
        logits = self.classifier(pooled_output)  # (1, 12)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Return as numpy array
        return probs.cpu().numpy()[0]
