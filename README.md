# Face-mask-detection
An ML CNN model to determine if someone is wearing a mask or not. 
🎯 Overview
This project implements a face mask detection system designed to classify images as either "with_mask" or "without_mask". Built with TensorFlow/Keras and FastAPI, it provides a REST API for real-time mask detection.

Key Technologies:

Deep Learning: TensorFlow 2.16, Keras
API Framework: FastAPI
Image Processing: PIL, NumPy
Deployment: Docker, Uvicorn


✨ Features
Core Features
✅ Binary Classification: Detects mask presence with >90% accuracy
✅ REST API: FastAPI-based endpoints for easy integration
✅ Batch Processing: Process multiple images in a single request
✅ Fast Inference: ~30-50ms per image on CPU
✅ Interactive Documentation: Auto-generated Swagger UI
✅ Production Ready: Docker containerization included
Additional Features
Model evaluation with comprehensive metrics
Data preprocessing and augmentation
Training visualization and history
Automated testing suite
Detailed logging and error handling


System Requirements
Minimum Requirements
OS: Windows 10/11, Linux, macOS
Python: 3.10 or 3.11 (recommended)
RAM: 8 GB
Storage: 5 GB free space
Internet: Required for initial setup
Recommended Requirements
OS: Windows 11 or Ubuntu 22.04
Python: 3.11
RAM: 16 GB
Storage: 10 GB free space
GPU: NVIDIA GPU with CUDA support (optional, for faster training)


Installation
Step 1: Clone or Extract the Project

bash
cd /path/to/face-mask-detection
Step 2: Create Virtual Environment
Windows (PowerShell):


powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
Windows (Command Prompt):


cmd
python -m venv venv
venv\Scripts\activate.bat
Linux/macOS:


bash
python3 -m venv venv
source venv/bin/activate
Step 3: Upgrade pip

bash
python -m pip install --upgrade pip
Step 4: Install Dependencies

bash
pip install -r requirements.txt
Note for Python 3.12 users: Use the compatible requirements:


bash
pip install tensorflow numpy pillow fastapi uvicorn[standard] python-multipart scikit-learn pandas matplotlib seaborn
Step 5: Verify Installation

bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"

Dataset Setup
Download the Dataset
Option 1: Using Kaggle API (Recommended)

Install Kaggle CLI:

bash
pip install kaggle
Get your Kaggle API token:
Go to https://www.kaggle.com/account
Scroll to "API" section
Click "Create New API Token"
Save kaggle.json to ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\.kaggle\ (Windows)
Download dataset:

bash
kaggle datasets download -d omkargurav/face-mask-dataset
Extract to data/raw/:
Windows:


powershell
Expand-Archive -Path face-mask-dataset.zip -DestinationPath data\raw\
Linux/macOS:


bash
unzip face-mask-dataset.zip -d data/raw/
Option 2: Manual Download

Visit: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
Click "Download" (requires Kaggle account)
Extract the ZIP file
Move folders to data/raw/:
data/raw/with_mask/ (~3,700 images)
data/raw/without_mask/ (~3,800 images)
Verify Dataset Structure

bash
# Windows
dir data\raw\with_mask
dir data\raw\without_mask

# Linux/macOS
ls data/raw/with_mask
ls data/raw/without_mask
Expected: ~7,500 total images


Training the Model
Step 1: Preprocess Data
The preprocessing happens automatically during training, but you can run it separately:


bash
python src/data_preprocessing.py
This will:

Load images from data/raw/
Resize to 224x224 pixels
Normalize pixel values
Apply data augmentation
Split into train/val/test sets (80/10/10)
Save to data/processed/
Step 2: Train the Model
Basic Training:


bash
python src/train.py --epochs 30 --batch-size 32
With MLflow Tracking:


bash
python src/train.py --epochs 30 --batch-size 32 --use-mlflow
Training Parameters:

--epochs: Number of training epochs (default: 25)
--batch-size: Batch size for training (default: 32)
--learning-rate: Learning rate (default: 0.0001)
--use-mlflow: Enable MLflow experiment tracking
--fine-tune: Enable fine-tuning after initial training
Expected Training Time:

CPU: 30-60 minutes
GPU: 10-15 minutes
Step 3: Monitor Training
During training, you'll see:


Epoch 1/30
189/189 ━━━━━━━━━━━━━━━━━━━━ 70s - loss: 0.3234 - accuracy: 0.8567
Epoch 10/30
189/189 ━━━━━━━━━━━━━━━━━━━━ 68s - loss: 0.0567 - accuracy: 0.9823
Good signs:

✅ Accuracy increasing (should reach >90% by epoch 10-15)
✅ Loss decreasing
✅ Validation accuracy close to training accuracy
Step 4: Evaluate the Model

bash
python src/evaluate.py
This generates:

models/confusion_matrix.png - Confusion matrix visualization
models/roc_curve.png - ROC curve
models/precision_recall_curve.png - Precision-Recall curve
models/classification_report.txt - Detailed metrics
models/evaluation_results.json - JSON metrics
Expected Results:

Accuracy: >90%
Precision: >90%
Recall: >90%
F1-Score: >90%



Running the API
Step 1: Start the API Server
Development Mode (with auto-reload):


bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Production Mode:


bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
You should see:


INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Model loaded successfully!
INFO:     Application startup complete.
Step 2: Access the API
Interactive Docs (Swagger UI): http://localhost:8000/docs
Alternative Docs (ReDoc): http://localhost:8000/redoc
Health Check: http://localhost:8000/health
Step 3: Test Inference
Using Python Script:


bash
python src/inference.py --image data/raw/with_mask/with_mask_1.jpg
Expected Output:


==================================================
PREDICTION RESULT
==================================================
Image: data/raw/with_mask/with_mask_1.jpg
Prediction: with_mask
Confidence: 0.9234
Inference Time: 45.23 ms
Model Type: Keras
==================================================
Testing
Automated API Testing
Test All Endpoints:


bash
python test_api.py --url http://localhost:8000 --all
Test Single Image:


bash
python test_api.py --url http://localhost:8000 --image "data/raw/with_mask/with_mask_1.jpg"
Test Batch Processing:


bash
python test_api.py --url http://localhost:8000 --image-dir "data/raw/with_mask"
Expected Output:


============================================================
FACE MASK DETECTION API TEST SUITE
============================================================
✅ Health check passed!
✅ Detection test passed!
  - Prediction: with_mask
  - Confidence: 0.9234
============================================================
Manual Testing with cURL
Single Image Detection:


bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
Health Check:


bash
curl http://localhost:8000/health
Testing in Browser
Navigate to http://localhost:8000/docs
Click on POST /detect endpoint
Click "Try it out"
Upload an image file
Click "Execute"
View the response below

API Documentation
Endpoints
GET / or GET /health
Health check endpoint.

Response:


json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
POST /detect
Detect face mask in a single image.

Request:

Content-Type: multipart/form-data
Body: file (image file - jpg, png, jpeg)
Response:


json
{
  "prediction": "with_mask",
  "confidence": 0.9234,
  "message": "Person is wearing a mask",
  "inference_time_ms": 45.23
}
Status Codes:

200: Success
400: Invalid file type
503: Model not loaded
500: Server error
POST /detect-batch
Detect face masks in multiple images (max 10).

Request:

Content-Type: multipart/form-data
Body: files (array of image files)
Response:


json
{
  "results": [
    {
      "filename": "image1.jpg",
      "prediction": "with_mask",
      "confidence": 0.9234,
      "message": "Person is wearing a mask",
      "inference_time_ms": 45.23
    },
    {
      "filename": "image2.jpg",
      "prediction": "without_mask",
      "confidence": 0.8876,
      "message": "Person is not wearing a mask",
      "inference_time_ms": 42.15
    }
  ]
}
GET /model-info
Get information about the loaded model.

Response:


json
{
  "model_path": "models/mask_detection_model.keras",
  "input_shape": [224, 224, 3],
  "output_classes": ["with_mask", "without_mask"],
  "model_architecture": "CNN",
  "total_parameters": 1234567
}


Model Performance
Architecture
Type: Convolutional Neural Network (CNN)
Input: 224x224x3 RGB images
Output: Binary classification (with_mask / without_mask)
Parameters: ~1.2M trainable parameters
Performance Metrics
Accuracy: 92-95%
Precision: 92-95%
Recall: 92-95%
F1-Score: 92-95%
Inference Time: 30-50ms per image (CPU)
Dataset Statistics
Total Images: 7,553
With Mask: 3,725 images
Without Mask: 3,828 images
Training Set: 6,041 images (80%)
Validation Set: 756 images (10%)
Test Set: 756 images (10%)

 Docker Deployment
Build Docker Image

bash
docker build -t face-mask-detection .
Run Container
Basic:


bash
docker run -p 8000:8000 face-mask-detection
With detached mode:


bash
docker run -d -p 8000:8000 --name mask-api face-mask-detection
With volume mounting (for logs):


bash
docker run -d -p 8000:8000 -v $(pwd)/logs:/app/logs --name mask-api face-mask-detection
Using Docker Compose

bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
Container Management

bash
# View running containers
docker ps

# View logs
docker logs mask-api

# Stop container
docker stop mask-api

# Remove container
docker rm mask-api

# Remove image
docker rmi face-mask-detection


📁 Project Structure

face-mask-detection/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker configuration
├── docker-compose.yml                 # Docker Compose configuration
├── setup-windows.ps1                  # Setup script for Windows (PowerShell)

│
├── src/                               # Source code
│   ├── data_preprocessing.py          # Data loading and preprocessing
│   ├── train.py                       # Model training script
│   ├── evaluate.py                    # Model evaluation
│   ├── inference.py                   # Inference on new images
│   └── convert_to_onnx.py             # ONNX conversion (optional)
│
├── api/                               # FastAPI application
│   ├── main.py                        # API endpoints
│   └── requirements.txt               # API-specific dependencies
│
├── data/                              # Data directory
│   ├── raw/                           # Original dataset (user downloads)
│   │   ├── with_mask/                 # Images with masks
│   │   └── without_mask/              # Images without masks
│   └── processed/                     # Preprocessed data (generated)
│
├── models/                            # Model artifacts (generated)
│   ├── mask_detection_model.keras     # Trained model
│   ├── confusion_matrix.png           # Evaluation visualizations
│   ├── roc_curve.png
│   └── classification_report.txt
│
├── logs/                              # Training logs
├── test_api.py                        # API testing script
└── .gitignore                         # Git ignore file
