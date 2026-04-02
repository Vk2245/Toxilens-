"""
Conformal prediction wrapper for uncertainty quantification.

This module wraps the ensemble model with MAPIE to provide prediction sets
with guaranteed coverage guarantees.
"""

import numpy as np
from typing import Dict, List, Tuple
from mapie.classification import MapieClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from ml.models.ensemble import EnsembleModel


class EnsembleWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for EnsembleModel.
    
    This wrapper allows MAPIE to calibrate the ensemble model.
    """
    
    def __init__(self, ensemble_model: EnsembleModel):
        """
        Initialize wrapper.
        
        Args:
            ensemble_model: Trained EnsembleModel instance
        """
        self.ensemble_model = ensemble_model
        self.classes_ = np.array([0, 1])  # Binary classification
    
    def fit(self, X, y):
        """Dummy fit method (model is already trained)."""
        return self
    
    def predict_proba(self, X):
        """
        Predict probabilities for calibration set.
        
        Args:
            X: List of tuples (smiles, graph, features)
        
        Returns:
            Array of shape (n_samples, 2) with [P(class=0), P(class=1)]
        """
        n_samples = len(X)
        probs = np.zeros((n_samples, 2))
        
        for i, (smiles, graph, features) in enumerate(X):
            result = self.ensemble_model.predict(smiles, graph, features)
            # For binary classification, we need [P(0), P(1)]
            # We'll use the first task as an example (can be extended for multi-task)
            prob_toxic = result['probabilities'][0]
            probs[i, 0] = 1 - prob_toxic
            probs[i, 1] = prob_toxic
        
        return probs


class ConformalEnsemble:
    """
    Conformal prediction wrapper for ensemble model.
    
    Provides prediction sets with guaranteed coverage using MAPIE.
    """
    
    def __init__(
        self,
        ensemble_model: EnsembleModel,
        alpha: float = 0.15
    ):
        """
        Initialize conformal ensemble.
        
        Args:
            ensemble_model: Trained EnsembleModel instance
            alpha: Significance level (1 - alpha = coverage target)
                   alpha=0.15 gives 85% coverage target
        """
        self.ensemble_model = ensemble_model
        self.alpha = alpha
        self.num_tasks = 12
        
        # Create MAPIE classifiers for each task
        self.mapie_classifiers = []
        for _ in range(self.num_tasks):
            wrapper = EnsembleWrapper(ensemble_model)
            mapie = MapieClassifier(
                estimator=wrapper,
                method="score",
                cv="prefit"  # Model is already trained
            )
            self.mapie_classifiers.append(mapie)
        
        self.is_calibrated = False
    
    def calibrate(
        self,
        calibration_data: List[Tuple],
        calibration_labels: np.ndarray
    ):
        """
        Calibrate conformal predictors on held-out calibration set.
        
        Args:
            calibration_data: List of (smiles, graph, features) tuples
            calibration_labels: Ground truth labels (n_samples, num_tasks)
        """
        n_samples = len(calibration_data)
        
        # Calibrate each task separately
        for task_idx in range(self.num_tasks):
            # Filter out missing labels
            mask = ~np.isnan(calibration_labels[:, task_idx])
            
            if mask.sum() > 0:
                # Prepare data for this task
                X_cal = [calibration_data[i] for i in range(n_samples) if mask[i]]
                y_cal = calibration_labels[mask, task_idx].astype(int)
                
                # Calibrate MAPIE
                self.mapie_classifiers[task_idx].fit(X_cal, y_cal)
        
        self.is_calibrated = True
    
    def predict(
        self,
        smiles: str,
        graph,
        features: np.ndarray
    ) -> Dict:
        """
        Predict with conformal prediction sets.
        
        Args:
            smiles: SMILES string
            graph: PyTorch Geometric Data object
            features: Feature vector
        
        Returns:
            Dictionary containing:
                - 'probabilities': Point predictions (num_tasks,)
                - 'prediction_sets': List of prediction sets for each task
                    Each set is one of: {0}, {1}, or {0, 1}
                - 'set_sizes': Size of each prediction set (num_tasks,)
        """
        if not self.is_calibrated:
            raise ValueError("Model must be calibrated before prediction")
        
        # Get ensemble predictions
        ensemble_result = self.ensemble_model.predict(smiles, graph, features)
        probabilities = ensemble_result['probabilities']
        
        # Get conformal prediction sets for each task
        prediction_sets = []
        set_sizes = []
        
        X_test = [(smiles, graph, features)]
        
        for task_idx in range(self.num_tasks):
            # Get prediction set from MAPIE
            _, y_ps = self.mapie_classifiers[task_idx].predict(
                X_test,
                alpha=self.alpha
            )
            
            # y_ps shape: (1, 2) - binary indicators for each class
            pred_set = set(np.where(y_ps[0])[0].tolist())
            
            prediction_sets.append(pred_set)
            set_sizes.append(len(pred_set))
        
        return {
            'probabilities': probabilities,
            'prediction_sets': prediction_sets,
            'set_sizes': np.array(set_sizes),
            'coverage_target': 1 - self.alpha
        }
    
    def predict_with_labels(
        self,
        smiles: str,
        graph,
        features: np.ndarray,
        assay_names: List[str] = None
    ) -> Dict:
        """
        Predict with human-readable labels.
        
        Args:
            smiles: SMILES string
            graph: PyTorch Geometric Data object
            features: Feature vector
            assay_names: Optional list of assay names
        
        Returns:
            Dictionary with labeled prediction sets
        """
        result = self.predict(smiles, graph, features)
        
        if assay_names is None:
            assay_names = [f'Assay_{i}' for i in range(self.num_tasks)]
        
        # Convert prediction sets to labels
        labeled_sets = []
        for task_idx, pred_set in enumerate(result['prediction_sets']):
            if pred_set == {0}:
                label = "SAFE"
            elif pred_set == {1}:
                label = "TOXIC"
            elif pred_set == {0, 1}:
                label = "UNCERTAIN"
            else:
                label = "EMPTY"
            
            labeled_sets.append({
                'assay': assay_names[task_idx],
                'probability': float(result['probabilities'][task_idx]),
                'prediction_set': label,
                'set_size': int(result['set_sizes'][task_idx])
            })
        
        return {
            'predictions': labeled_sets,
            'coverage_target': result['coverage_target'],
            'mean_set_size': float(result['set_sizes'].mean())
        }


def evaluate_coverage(
    conformal_model: ConformalEnsemble,
    test_data: List[Tuple],
    test_labels: np.ndarray
) -> Dict:
    """
    Evaluate empirical coverage on test set.
    
    Args:
        conformal_model: Calibrated ConformalEnsemble
        test_data: List of (smiles, graph, features) tuples
        test_labels: Ground truth labels (n_samples, num_tasks)
    
    Returns:
        Dictionary with coverage statistics
    """
    n_samples = len(test_data)
    num_tasks = test_labels.shape[1]
    
    # Track coverage for each task
    task_coverage = []
    task_set_sizes = []
    
    for task_idx in range(num_tasks):
        # Filter out missing labels
        mask = ~np.isnan(test_labels[:, task_idx])
        
        if mask.sum() == 0:
            continue
        
        covered = 0
        set_sizes = []
        
        for i in range(n_samples):
            if not mask[i]:
                continue
            
            smiles, graph, features = test_data[i]
            result = conformal_model.predict(smiles, graph, features)
            
            true_label = int(test_labels[i, task_idx])
            pred_set = result['prediction_sets'][task_idx]
            
            # Check if true label is in prediction set
            if true_label in pred_set:
                covered += 1
            
            set_sizes.append(len(pred_set))
        
        coverage = covered / mask.sum()
        mean_set_size = np.mean(set_sizes)
        
        task_coverage.append(coverage)
        task_set_sizes.append(mean_set_size)
    
    return {
        'mean_coverage': np.mean(task_coverage),
        'min_coverage': np.min(task_coverage),
        'max_coverage': np.max(task_coverage),
        'mean_set_size': np.mean(task_set_sizes),
        'per_task_coverage': task_coverage,
        'per_task_set_size': task_set_sizes
    }


if __name__ == "__main__":
    print("Conformal prediction module loaded successfully")
    print("Note: Requires calibrated ensemble model for testing")
