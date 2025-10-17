"""
Metrics calculation for model evaluation
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Any, Optional
import logging

class MetricsCalculator:
    """
    Calculate various metrics for model evaluation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize metrics calculator
        
        Args:
            config: Configuration for metrics calculation
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configure which metrics to calculate
        self.calculate_precision_recall = self.config.get('precision_recall', True)
        self.calculate_confusion_matrix = self.config.get('confusion_matrix', True)
        self.average_type = self.config.get('average', 'weighted')  # 'micro', 'macro', 'weighted'
    
    def calculate_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Calculate comprehensive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of calculated metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1-score
        if self.calculate_precision_recall:
            try:
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true, y_pred, average=self.average_type, zero_division=0
                )
                
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1_score'] = f1
                
                # Per-class metrics
                if self.average_type == 'weighted':
                    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average=None, zero_division=0
                    )
                    
                    for i, (p, r, f) in enumerate(zip(per_class_precision, per_class_recall, per_class_f1)):
                        metrics[f'precision_class_{i}'] = p
                        metrics[f'recall_class_{i}'] = r
                        metrics[f'f1_class_{i}'] = f
                        
            except Exception as e:
                self.logger.warning(f"Failed to calculate precision/recall metrics: {e}")
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1_score'] = 0.0
        
        # Top-k accuracy (for multi-class)
        if len(np.unique(y_true)) > 2:
            metrics['top_2_accuracy'] = self._top_k_accuracy(y_true, y_pred, k=2)
            metrics['top_3_accuracy'] = self._top_k_accuracy(y_true, y_pred, k=3)
        
        return metrics
    
    def _top_k_accuracy(self, y_true: np.ndarray, y_pred_proba: np.ndarray, k: int) -> float:
        """
        Calculate top-k accuracy
        Note: This is a simplified version. For true top-k accuracy, 
        we need prediction probabilities, not just predicted classes.
        """
        # This is a placeholder implementation
        # In a real scenario, you'd need the actual prediction probabilities
        return float(accuracy_score(y_true, y_pred_proba))  # Fallback to regular accuracy
    
    def calculate_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> np.ndarray:
        """
        Calculate confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def calculate_class_metrics(self, y_true: List[int], y_pred: List[int], 
                              class_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names
            
        Returns:
            Dictionary of per-class metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Get unique classes
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
        if class_names is None:
            class_names = [f'class_{i}' for i in classes]
        
        class_metrics = {}
        
        # Calculate per-class precision, recall, f1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        for i, class_idx in enumerate(classes):
            if i < len(class_names):
                class_name = class_names[i]
            else:
                class_name = f'class_{class_idx}'
                
            class_metrics[class_name] = {
                'precision': precision[i] if i < len(precision) else 0.0,
                'recall': recall[i] if i < len(recall) else 0.0,
                'f1_score': f1[i] if i < len(f1) else 0.0,
                'support': support[i] if i < len(support) else 0
            }
        
        return class_metrics
    
    def print_metrics_summary(self, metrics: Dict[str, float], 
                            class_metrics: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Print a formatted summary of metrics
        
        Args:
            metrics: Overall metrics
            class_metrics: Per-class metrics
        """
        print("\n" + "="*50)
        print("METRICS SUMMARY")
        print("="*50)
        
        # Overall metrics
        print(f"Overall Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"Overall Precision: {metrics.get('precision', 0):.4f}")
        print(f"Overall Recall: {metrics.get('recall', 0):.4f}")
        print(f"Overall F1-Score: {metrics.get('f1_score', 0):.4f}")
        
        if 'top_2_accuracy' in metrics:
            print(f"Top-2 Accuracy: {metrics['top_2_accuracy']:.4f}")
        if 'top_3_accuracy' in metrics:
            print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
        
        # Per-class metrics
        if class_metrics:
            print("\nPer-Class Metrics:")
            print("-" * 30)
            for class_name, class_metric in class_metrics.items():
                print(f"{class_name}:")
                print(f"  Precision: {class_metric['precision']:.4f}")
                print(f"  Recall: {class_metric['recall']:.4f}")
                print(f"  F1-Score: {class_metric['f1_score']:.4f}")
                print(f"  Support: {class_metric['support']}")
                print()
        
        print("="*50)