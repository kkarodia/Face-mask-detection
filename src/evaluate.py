"""
Model Evaluation Script for Face Mask Detection

This script evaluates the trained model and generates comprehensive metrics
including confusion matrix, classification report, and ROC curve.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from data_preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles model evaluation and metrics generation."""
    
    def __init__(self, model_path: str, class_names: list):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model
            class_names: List of class names
        """
        self.model_path = Path(model_path)
        self.class_names = class_names
        self.model = None
        
    def load_model(self):
        """Load the trained model."""
        logger.info(f"Loading model from {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        logger.info("Model loaded successfully!")
        
    def evaluate_on_dataset(
        self,
        dataset: tf.data.Dataset,
        dataset_name: str = "Test"
    ) -> dict:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset: TensorFlow dataset
            dataset_name: Name of the dataset (for logging)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating on {dataset_name} set...")
        
        # Evaluate
        results = self.model.evaluate(dataset, verbose=1)
        
        # Get metric names and values
        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = float(value)
            logger.info(f"{dataset_name} {name}: {value:.4f}")
        
        return metrics
    
    def get_predictions(self, dataset: tf.data.Dataset) -> tuple:
        """
        Get predictions and true labels from dataset.
        
        Args:
            dataset: TensorFlow dataset
            
        Returns:
            Tuple of (y_true, y_pred, y_pred_proba)
        """
        y_true = []
        y_pred_proba = []
        
        for images, labels in dataset:
            y_true.extend(labels.numpy())
            predictions = self.model.predict(images, verbose=0)
            y_pred_proba.extend(predictions.flatten())
        
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        return y_true, y_pred, y_pred_proba
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = "models/confusion_matrix.png"
    ):
        """
        Plot and save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
        plt.close()
        
        return cm
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str = "models/roc_curve.png"
    ):
        """
        Plot and save ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve',
                 fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
        plt.close()
        
        return roc_auc
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str = "models/precision_recall_curve.png"
    ):
        """
        Plot and save precision-recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curve saved to {save_path}")
        plt.close()
        
        return avg_precision
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = "models/classification_report.txt"
    ) -> str:
        """
        Generate and save classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the report
            
        Returns:
            Classification report as string
        """
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        )
        
        logger.info(f"\nClassification Report:\n{report}")
        
        # Save report
        with open(save_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
        
        logger.info(f"Classification report saved to {save_path}")
        
        return report
    
    def plot_sample_predictions(
        self,
        dataset: tf.data.Dataset,
        num_samples: int = 16,
        save_path: str = "models/sample_predictions.png"
    ):
        """
        Plot sample predictions with true and predicted labels.
        
        Args:
            dataset: TensorFlow dataset
            num_samples: Number of samples to plot
            save_path: Path to save the plot
        """
        images_to_plot = []
        labels_to_plot = []
        predictions_to_plot = []
        
        for images, labels in dataset.take(1):
            images_to_plot = images[:num_samples].numpy()
            labels_to_plot = labels[:num_samples].numpy()
            predictions = self.model.predict(images[:num_samples], verbose=0)
            predictions_to_plot = (predictions.flatten() > 0.5).astype(int)
        
        # Create grid
        rows = 4
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
        axes = axes.flatten()
        
        for idx in range(min(num_samples, len(images_to_plot))):
            img = images_to_plot[idx]
            true_label = self.class_names[labels_to_plot[idx]]
            pred_label = self.class_names[predictions_to_plot[idx]]
            
            # Determine color based on correctness
            color = 'green' if true_label == pred_label else 'red'
            
            axes[idx].imshow(img)
            axes[idx].set_title(
                f'True: {true_label}\nPred: {pred_label}',
                color=color,
                fontsize=10,
                fontweight='bold'
            )
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sample predictions saved to {save_path}")
        plt.close()
    
    def generate_evaluation_report(
        self,
        test_dataset: tf.data.Dataset,
        output_dir: str = "models"
    ) -> dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            test_dataset: Test dataset
            output_dir: Directory to save evaluation outputs
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.model is None:
            self.load_model()
        
        # Evaluate model
        metrics = self.evaluate_on_dataset(test_dataset, "Test")
        
        # Get predictions
        y_true, y_pred, y_pred_proba = self.get_predictions(test_dataset)
        
        # Generate confusion matrix
        cm = self.plot_confusion_matrix(
            y_true, y_pred,
            save_path=str(output_path / "confusion_matrix.png")
        )
        
        # Generate ROC curve
        roc_auc = self.plot_roc_curve(
            y_true, y_pred_proba,
            save_path=str(output_path / "roc_curve.png")
        )
        
        # Generate Precision-Recall curve
        avg_precision = self.plot_precision_recall_curve(
            y_true, y_pred_proba,
            save_path=str(output_path / "precision_recall_curve.png")
        )
        
        # Generate classification report
        report = self.generate_classification_report(
            y_true, y_pred,
            save_path=str(output_path / "classification_report.txt")
        )
        
        # Plot sample predictions
        self.plot_sample_predictions(
            test_dataset,
            save_path=str(output_path / "sample_predictions.png")
        )
        
        # Compile all metrics
        evaluation_results = {
            'test_metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'roc_auc': float(roc_auc),
            'average_precision': float(avg_precision),
            'classification_report': report
        }
        
        # Save evaluation results
        with open(output_path / "evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
        
        return evaluation_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate face mask detection model')
    parser.add_argument('--model-path', type=str, default='models/mask_detection_model.keras',
                       help='Path to the trained model')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing preprocessed data')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save evaluation outputs')
    
    args = parser.parse_args()
    
    # Load preprocessed data
    preprocessor = DataPreprocessor(data_dir="data/raw")
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = \
        preprocessor.load_preprocessed_data(args.data_dir)
    
    class_names = metadata['class_names']
    
    # Create test dataset
    _, _, test_dataset = preprocessor.create_tf_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, class_names)
    
    # Generate evaluation report
    results = evaluator.generate_evaluation_report(
        test_dataset,
        output_dir=args.output_dir
    )
    
    logger.info("Evaluation completed successfully!")
    
    # Print summary
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    # Debug: print all available metric keys
    print("\nAvailable metrics:")
    for key in results['test_metrics'].keys():
        print(f"  {key}: {results['test_metrics'][key]:.4f}")

    print(f"\nROC-AUC: {results['roc_auc']:.4f}")
    print(f"Average Precision: {results['average_precision']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
