"""
Data Preprocessing Module for Face Mask Detection

This module handles data loading, preprocessing, augmentation, and splitting
for the face mask detection model.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
import shutil
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing operations."""
    
    def __init__(
        self,
        data_dir: str,
        img_height: int = 224,
        img_width: int = 224,
        batch_size: int = 32,
        validation_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize the data preprocessor.
        
        Args:
            data_dir: Path to the raw data directory
            img_height: Target image height
            img_width: Target image width
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.seed = seed
        
        # Data augmentation parameters
        self.augmentation_config = {
            'rotation_range': 15,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'horizontal_flip': True,
            'zoom_range': 0.1,
            'brightness_range': [0.8, 1.2],
            'fill_mode': 'nearest'
        }
        
    def load_data_from_directory(self) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Load images and labels from directory structure.
        
        Expected structure:
        data_dir/
            with_mask/
                img1.jpg
                img2.jpg
            without_mask/
                img1.jpg
                img2.jpg
        
        Returns:
            Tuple of (images, labels, class_names)
        """
        logger.info(f"Loading data from {self.data_dir}")
        
        images = []
        labels = []
        class_names = []
        
        # Get class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        class_names = sorted([d.name for d in class_dirs])
        
        logger.info(f"Found classes: {class_names}")
        
        # Create label mapping
        label_map = {name: idx for idx, name in enumerate(class_names)}
        
        # Load images
        for class_dir in class_dirs:
            class_label = label_map[class_dir.name]
            image_files = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.png")) + \
                         list(class_dir.glob("*.jpeg"))
            
            logger.info(f"Loading {len(image_files)} images from {class_dir.name}")
            
            for img_path in image_files:
                try:
                    # Load and preprocess image
                    img = keras.preprocessing.image.load_img(
                        img_path,
                        target_size=(self.img_height, self.img_width)
                    )
                    img_array = keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(class_label)
                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")
                    continue
        
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        logger.info(f"Loaded {len(images)} images total")
        logger.info(f"Image shape: {images[0].shape}")
        logger.info(f"Label distribution: {np.bincount(labels)}")
        
        return images, labels, class_names
    
    def normalize_images(self, images: np.ndarray) -> np.ndarray:
        """
        Normalize images to [0, 1] range.
        
        Args:
            images: Array of images
            
        Returns:
            Normalized images
        """
        return images / 255.0
    
    def split_data(
        self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            images: Array of images
            labels: Array of labels
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=self.test_split,
            random_state=self.seed,
            stratify=labels
        )
        
        # Second split: separate validation from train
        val_size_adjusted = self.validation_split / (1 - self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.seed,
            stratify=y_temp
        )
        
        logger.info(f"Train set: {len(X_train)} images")
        logger.info(f"Validation set: {len(X_val)} images")
        logger.info(f"Test set: {len(X_test)} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_generators(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """
        Create data generators with augmentation for training.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            
        Returns:
            Tuple of (train_generator, val_generator)
        """
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            **self.augmentation_config
        )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Fit generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.batch_size,
            seed=self.seed
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=self.batch_size,
            seed=self.seed
        )
        
        return train_generator, val_generator
    
    def create_tf_datasets(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create TensorFlow datasets with preprocessing and augmentation.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Normalize data
        X_train_norm = self.normalize_images(X_train)
        X_val_norm = self.normalize_images(X_val)
        X_test_norm = self.normalize_images(X_test)
        
        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_norm, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val_norm, y_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test_norm, y_test))
        
        # Data augmentation function
        def augment(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            return image, label
        
        # Apply augmentation to training data
        train_dataset = train_dataset.map(
            augment,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Batch and prefetch all datasets
        train_dataset = train_dataset.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset, test_dataset
    
    def save_preprocessed_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: list,
        output_dir: str = "data/processed"
    ):
        """
        Save preprocessed data to disk.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            class_names: List of class names
            output_dir: Directory to save processed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving preprocessed data to {output_dir}")
        
        # Save numpy arrays
        np.save(output_path / "X_train.npy", X_train)
        np.save(output_path / "y_train.npy", y_train)
        np.save(output_path / "X_val.npy", X_val)
        np.save(output_path / "y_val.npy", y_val)
        np.save(output_path / "X_test.npy", X_test)
        np.save(output_path / "y_test.npy", y_test)
        
        # Save metadata
        metadata = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'class_names': class_names,
            'num_train': len(X_train),
            'num_val': len(X_val),
            'num_test': len(X_test),
            'train_distribution': np.bincount(y_train).tolist(),
            'val_distribution': np.bincount(y_val).tolist(),
            'test_distribution': np.bincount(y_test).tolist()
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Preprocessing complete!")
        logger.info(f"Metadata: {json.dumps(metadata, indent=2)}")
    
    def load_preprocessed_data(
        self,
        data_dir: str = "data/processed"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Load preprocessed data from disk.
        
        Args:
            data_dir: Directory containing preprocessed data
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, metadata)
        """
        data_path = Path(data_dir)
        
        logger.info(f"Loading preprocessed data from {data_dir}")
        
        X_train = np.load(data_path / "X_train.npy")
        y_train = np.load(data_path / "y_train.npy")
        X_val = np.load(data_path / "X_val.npy")
        y_val = np.load(data_path / "y_val.npy")
        X_test = np.load(data_path / "X_test.npy")
        y_test = np.load(data_path / "y_test.npy")
        
        with open(data_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        logger.info("Data loaded successfully!")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, metadata


def main():
    """Main function to run preprocessing pipeline."""
    # Configuration
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed"
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        data_dir=RAW_DATA_DIR,
        img_height=224,
        img_width=224,
        batch_size=32,
        validation_split=0.1,
        test_split=0.1,
        seed=42
    )
    
    # Load raw data
    images, labels, class_names = preprocessor.load_data_from_directory()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        images, labels
    )
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        class_names,
        output_dir=PROCESSED_DATA_DIR
    )
    
    logger.info("Preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()
