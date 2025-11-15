"""
FastAPI Application for Face Mask Detection

This module provides a REST API for face mask detection with endpoints
for health checks and mask detection on uploaded images.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import time
import logging
import os
from pathlib import Path
from typing import Optional

# Import our inference module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference import MaskDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Mask Detection API",
    description="API for detecting face masks in images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
DETECTOR = None
MODEL_PATH = Path(__file__).parent.parent / "models" / "mask_detection_model.keras"


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: str
    confidence: float
    message: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def startup_event():
    """Load model on FastAPI startup."""
    global DETECTOR
    
    logger.info("="*60)
    logger.info("FASTAPI STARTUP - LOADING MODEL")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Model path exists: {MODEL_PATH.exists()}")
    logger.info(f"Absolute model path: {MODEL_PATH.absolute()}")
    
    try:
        logger.info("Attempting to create MaskDetector instance...")
        DETECTOR = MaskDetector(model_path=str(MODEL_PATH))
        logger.info("✓ DETECTOR CREATED")
        logger.info(f"DETECTOR is None: {DETECTOR is None}")
        logger.info(f"DETECTOR.model is None: {DETECTOR.model is None}")
        logger.info(f"DETECTOR.model: {DETECTOR.model}")
        logger.info("✓ MODEL LOADED SUCCESSFULLY")
    except Exception as e:
        logger.error(f"✗ FAILED TO LOAD DETECTOR: {e}")
        logger.exception("Full traceback:")
        DETECTOR = None
    
    logger.info(f"Final DETECTOR state: {DETECTOR}")
    logger.info("="*60)


@app.get("/", response_model=HealthResponse)
async def root():
    """
    Health check endpoint.
    
    Returns:
        Health status of the API
    """
    return HealthResponse(
        status="healthy",
        model_loaded=DETECTOR is not None and DETECTOR.model is not None,
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status of the API
    """
    logger.info("="*60)
    logger.info("HEALTH CHECK ENDPOINT CALLED")
    logger.info(f"DETECTOR is None: {DETECTOR is None}")
    
    if DETECTOR is not None:
        logger.info(f"DETECTOR exists!")
        logger.info(f"DETECTOR.model is None: {DETECTOR.model is None}")
        logger.info(f"DETECTOR.model: {DETECTOR.model}")
    else:
        logger.info("DETECTOR is None - model not loaded")
    
    model_loaded = DETECTOR is not None and DETECTOR.model is not None
    logger.info(f"Returning model_loaded={model_loaded}")
    logger.info("="*60)
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        version="1.0.0"
    )


@app.post("/detect", response_model=PredictionResponse)
async def detect_mask(file: UploadFile = File(...)):
    """
    Detect face mask in uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction result with confidence score
    """
    # Check if detector is loaded
    if DETECTOR is None or DETECTOR.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please try again later."
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read image
        image_bytes = await file.read()
        
        # Preprocess image using detector's method
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((DETECTOR.img_width, DETECTOR.img_height))
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction using detector
        start_time = time.time()
        predicted_class, confidence, _ = DETECTOR.predict_keras(img_array)
        inference_time = (time.time() - start_time) * 1000
        
        # Get class name and message
        class_name = DETECTOR.class_names[predicted_class]
        
        if predicted_class == 1:  # without_mask
            message = "Person is not wearing a mask"
        else:  # with_mask
            message = "Person is wearing a mask"
        
        # Prepare response
        response = PredictionResponse(
            prediction=class_name,
            confidence=float(confidence),
            message=message,
            inference_time_ms=float(inference_time)
        )
        
        logger.info(f"Prediction: {class_name}, Confidence: {confidence:.4f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/detect-batch")
async def detect_mask_batch(files: list[UploadFile] = File(...)):
    """
    Detect face masks in multiple uploaded images.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        List of prediction results
    """
    # Check if detector is loaded
    if DETECTOR is None or DETECTOR.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please try again later."
        )
    
    # Limit number of files
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch request"
        )
    
    results = []
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "error": "File must be an image"
                })
                continue
            
            # Read and process image
            image_bytes = await file.read()
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img = img.resize((DETECTOR.img_width, DETECTOR.img_height))
            img_array = np.array(img, dtype=np.float32)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            start_time = time.time()
            predicted_class, confidence, _ = DETECTOR.predict_keras(img_array)
            inference_time = (time.time() - start_time) * 1000
            
            # Get class name and message
            class_name = DETECTOR.class_names[predicted_class]
            
            if predicted_class == 1:  # without_mask
                message = "Person is not wearing a mask"
            else:  # with_mask
                message = "Person is wearing a mask"
            
            results.append({
                "filename": file.filename,
                "prediction": class_name,
                "confidence": float(confidence),
                "message": message,
                "inference_time_ms": float(inference_time)
            })
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results})


@app.get("/model-info")
async def model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Model information
    """
    if DETECTOR is None or DETECTOR.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    # Get model information
    info = {
        "model_path": str(MODEL_PATH),
        "input_shape": [DETECTOR.img_height, DETECTOR.img_width, 3],
        "output_classes": DETECTOR.class_names,
        "model_architecture": "MobileNetV2",
        "total_parameters": int(DETECTOR.model.count_params())
    }
    
    return JSONResponse(content=info)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)