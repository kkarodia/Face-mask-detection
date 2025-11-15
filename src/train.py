"""
Model Training Script for Face Mask Detection

This script handles model training with support for:
- Transfer learning with MobileNetV2
- MLflow experiment tracking
- Model checkpointing
- Early stopping
- Learning rate scheduling
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
import mlflow
import mlflow.keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from data_preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


class MaskDetectionModel:
    """Face mask detection model wrapper."""
    
    def __init__(
        self,
        img_height: int = 224,
        img_width: int = 224,
        learning_rate: float = 0.0001,
        dropout_rate: float = 0.5
    ):
        """
        Initialize the model.
        
        Args:
            img_height: Input image height
            img_width: Input image width
            learning_rate: Learning rate for optimizer
            dropout_rate: Dropout rate for regularization
        """
        self.img_height = img_height
        self.img_width = img_width
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self) -> keras.Model:
        """
        Build the model architecture using transfer learning.
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building model architecture...")
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        
       # Unfreeze base model from the start
        base_model.trainable = True

        # But freeze the first 100 layers
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        # Build model
        inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
        
        # Pre-processing for MobileNetV2
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        self.model = model
        
        logger.info("Model built successfully!")
        logger.info(f"Total parameters: {model.count_params():,}")
        logger.info(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        
        return model
    
    def get_callbacks(
        self,
        checkpoint_path: str,
        use_mlflow: bool = False
    ) -> list:
        """
        Get training callbacks.
        
        Args:
            checkpoint_path: Path to save model checkpoints
            use_mlflow: Whether to use MLflow autologging
            
        Returns:
            List of callbacks
        """
        callback_list = []
        
        # Model checkpoint
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        callback_list.append(checkpoint_callback)
        
        # Early stopping
        early_stop_callback = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop_callback)
        
        # Learning rate reduction
        reduce_lr_callback = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr_callback)
        
        # TensorBoard
        log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
        callback_list.append(tensorboard_callback)
        
        return callback_list
    
    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int = 25,
        checkpoint_path: str = "models/checkpoint.keras",
        use_mlflow: bool = False
    ):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            checkpoint_path: Path to save checkpoints
            use_mlflow: Whether to use MLflow tracking
        """
        logger.info("Starting training...")
        
        if self.model is None:
            self.build_model()
        
        # Get callbacks
        callback_list = self.get_callbacks(checkpoint_path, use_mlflow)
        
        # Train model
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info("Training completed!")
        
        return self.history
    
    def fine_tune(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int = 10,
        fine_tune_at: int = 100,
        checkpoint_path: str = "models/checkpoint_finetuned.keras"
    ):
        """
        Fine-tune the model by unfreezing some base layers.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of fine-tuning epochs
            fine_tune_at: Layer index to start fine-tuning from
            checkpoint_path: Path to save checkpoints
        """
        logger.info("Starting fine-tuning...")
        
        if self.model is None:
            raise ValueError("Model must be trained first before fine-tuning")
        
        # Unfreeze base model layers
        base_model = self.model.layers[2]  # MobileNetV2 is the 3rd layer
        base_model.trainable = True
        
        # Freeze all layers except the last fine_tune_at layers
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate / 10),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        logger.info(f"Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        
        # Get callbacks
        callback_list = self.get_callbacks(checkpoint_path)
        
        # Fine-tune
        history_fine = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info("Fine-tuning completed!")
        
        return history_fine
    
    def save_model(self, save_path: str):
        """
        Save the trained model.
        
        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def plot_training_history(self, save_path: str = "models/training_history.png"):
        """
        Plot and save training history.
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        history_dict = self.history.history
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history_dict['accuracy'], label='Train')
        axes[0, 0].plot(history_dict['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history_dict['loss'], label='Train')
        axes[0, 1].plot(history_dict['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC
        axes[1, 0].plot(history_dict['auc'], label='Train')
        axes[1, 0].plot(history_dict['val_auc'], label='Validation')
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Precision & Recall
        axes[1, 1].plot(history_dict['precision'], label='Train Precision')
        axes[1, 1].plot(history_dict['val_precision'], label='Val Precision')
        axes[1, 1].plot(history_dict['recall'], label='Train Recall')
        axes[1, 1].plot(history_dict['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
        plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train face mask detection model')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directory containing raw data')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--use-mlflow', action='store_true',
                       help='Use MLflow for experiment tracking')
    parser.add_argument('--fine-tune', action='store_true',
                       help='Enable fine-tuning after initial training')
    
    args = parser.parse_args()
    
    # Create model directory
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        data_dir=args.data_dir,
        img_height=224,
        img_width=224,
        batch_size=args.batch_size
    )
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    images, labels, class_names = preprocessor.load_data_from_directory()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(images, labels)
    
    # Create TensorFlow datasets
    train_dataset, val_dataset, test_dataset = preprocessor.create_tf_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data(
        X_train, y_train, X_val, y_val, X_test, y_test, class_names
    )
    
    # Initialize model
    model_wrapper = MaskDetectionModel(
        img_height=224,
        img_width=224,
        learning_rate=args.learning_rate
    )
    
    # Build model
    model = model_wrapper.build_model()
    model.summary()
    
    # MLflow tracking
    if args.use_mlflow:
        mlflow.set_experiment("face-mask-detection")
        mlflow.tensorflow.autolog()
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'img_height': 224,
                'img_width': 224,
                'model_architecture': 'MobileNetV2'
            })
            
            # Train model
            model_wrapper.train(
                train_dataset,
                val_dataset,
                epochs=args.epochs,
                checkpoint_path=str(model_dir / "checkpoint.keras"),
                use_mlflow=True
            )
            
            # Fine-tuning (optional)
            if args.fine_tune:
                model_wrapper.fine_tune(
                    train_dataset,
                    val_dataset,
                    epochs=10,
                    checkpoint_path=str(model_dir / "checkpoint_finetuned.keras")
                )
            
            # Save final model
            final_model_path = model_dir / "mask_detection_model.keras"
            model_wrapper.save_model(str(final_model_path))
            
            # Log model to MLflow
            mlflow.keras.log_model(model_wrapper.model, "model")
            
            # Plot and log training history
            history_plot_path = model_dir / "training_history.png"
            model_wrapper.plot_training_history(str(history_plot_path))
            mlflow.log_artifact(str(history_plot_path))
    else:
        # Train without MLflow
        model_wrapper.train(
            train_dataset,
            val_dataset,
            epochs=args.epochs,
            checkpoint_path=str(model_dir / "checkpoint.keras")
        )
        
        # Fine-tuning (optional)
        if args.fine_tune:
            model_wrapper.fine_tune(
                train_dataset,
                val_dataset,
                epochs=10,
                checkpoint_path=str(model_dir / "checkpoint_finetuned.keras")
            )
        
        # Save final model
        final_model_path = model_dir / "mask_detection_model.keras"
        model_wrapper.save_model(str(final_model_path))
        
        # Plot training history
        model_wrapper.plot_training_history(str(model_dir / "training_history.png"))
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Model saved to {final_model_path}")


if __name__ == "__main__":
    main()
