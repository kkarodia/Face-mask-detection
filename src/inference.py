"""
Inference Script for Face Mask Detection

This script provides inference capabilities for the trained model,
supporting both Keras and ONNX models for optimized performance.
"""

import argparse
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ONNX runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available. Install with: pip install onnxruntime")


class MaskDetector:
    """Face mask detection inference class."""
    
    def __init__(
        self,
        model_path: str,
        class_names: list = None,
        use_onnx: bool = False,
        img_height: int = 224,
        img_width: int = 224
    ):
        """
        Initialize the mask detector.
        
        Args:
            model_path: Path to the model file
            class_names: List of class names
            use_onnx: Whether to use ONNX model
            img_height: Input image height
            img_width: Input image width
        """
        self.model_path = Path(model_path)
        self.class_names = class_names or ['with_mask', 'without_mask']
        self.use_onnx = use_onnx
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.session = None
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load the model (Keras or ONNX)."""
        if self.use_onnx:
            if not ONNX_AVAILABLE:
                raise ImportError("ONNX Runtime not installed")
            
            logger.info(f"Loading ONNX model from {self.model_path}")
            self.session = ort.InferenceSession(str(self.model_path))
            logger.info("ONNX model loaded successfully!")
        else:
            logger.info(f"Loading Keras model from {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            logger.info("Keras model loaded successfully!")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess an image for inference.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize
        img = img.resize((self.img_width, self.img_height))
        
        # Convert to array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_keras(self, image_array: np.ndarray) -> tuple:
        start_time = time.time()
        
        # Predict
        prediction = self.model.predict(image_array, verbose=0)
        raw_confidence = float(prediction[0][0])
        
        inference_time = (time.time() - start_time) * 1000
        
        # Model outputs probability for class 1 (without_mask)
        # If probability > 0.5, it predicts without_mask (class 1)
        # If probability < 0.5, it predicts with_mask (class 0)
        if raw_confidence > 0.5:
            predicted_class = 1  # without_mask
            confidence = raw_confidence
        else:
            predicted_class = 0  # with_mask
            confidence = 1 - raw_confidence
        
        return predicted_class, confidence, inference_time
    
    def predict_onnx(self, image_array: np.ndarray) -> tuple:
        """Make prediction using ONNX model."""
        start_time = time.time()
        
        # Get input name
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        outputs = self.session.run(None, {input_name: image_array})
        raw_confidence = float(outputs[0][0][0])
        
        inference_time = (time.time() - start_time) * 1000
        
        # Model outputs probability for class 1 (without_mask)
        if raw_confidence > 0.5:
            predicted_class = 1  # without_mask
            confidence = raw_confidence
        else:
            predicted_class = 0  # with_mask
            confidence = 1 - raw_confidence
        
        return predicted_class, confidence, inference_time
    
    def predict(self, image_path: str) -> dict:
        """
        Make prediction on an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess image
        image_array = self.preprocess_image(image_path)
        
        # Make prediction
        if self.use_onnx:
            predicted_class, confidence, inference_time = self.predict_onnx(image_array)
        else:
            predicted_class, confidence, inference_time = self.predict_keras(image_array)
        
        # Get class name
        class_name = self.class_names[predicted_class]
        
        # Prepare result
        result = {
            'prediction': class_name,
            'confidence': float(confidence),
            'inference_time_ms': float(inference_time),
            'model_type': 'ONNX' if self.use_onnx else 'Keras'
        }
        
        return result
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Make predictions on multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = str(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        
        return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Face mask detection inference')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--model', type=str, default='models/mask_detection_model.keras',
                       help='Path to the model file')
    parser.add_argument('--use-onnx', action='store_true',
                       help='Use ONNX model for inference')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results (JSON)')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple images from directory')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = MaskDetector(
        model_path=args.model,
        use_onnx=args.use_onnx
    )
    
    # Process image(s)
    if args.batch:
        # Process directory
        image_dir = Path(args.image)
        image_paths = list(image_dir.glob("*.jpg")) + \
                     list(image_dir.glob("*.png")) + \
                     list(image_dir.glob("*.jpeg"))
        
        logger.info(f"Processing {len(image_paths)} images...")
        results = detector.predict_batch(image_paths)
        
        # Print results
        for result in results:
            if 'error' in result:
                print(f"\n{result['image_path']}: ERROR - {result['error']}")
            else:
                print(f"\n{result['image_path']}:")
                print(f"  Prediction: {result['prediction']}")
                print(f"  Confidence: {result['confidence']:.4f}")
                print(f"  Inference Time: {result['inference_time_ms']:.2f} ms")
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
    else:
        # Process single image
        result = detector.predict(args.image)
        
        # Print result
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Image: {args.image}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Inference Time: {result['inference_time_ms']:.2f} ms")
        print(f"Model Type: {result['model_type']}")
        print("="*50)
        
        # Save result
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Result saved to {args.output}")


if __name__ == "__main__":
    main()
