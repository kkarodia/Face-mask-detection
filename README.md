# Face Mask Detection

An ML CNN model to determine if someone is wearing a mask or not.

## Overview

This project implements a face mask detection system designed to classify images as either "with_mask" or "without_mask". Built with TensorFlow/Keras and FastAPI, it provides a REST API for real-time mask detection.

### Key Technologies

- **Deep Learning**: TensorFlow 2.16, Keras
- **API Framework**: FastAPI
- **Image Processing**: PIL, NumPy
- **Deployment**: Docker, Uvicorn

## ✨ Features

### Core Features

- ✅ **Binary Classification**: Detects mask presence with >90% accuracy
- ✅ **REST API**: FastAPI-based endpoints for easy integration
- ✅ **Batch Processing**: Process multiple images in a single request
- ✅ **Fast Inference**: ~30-50ms per image on CPU
- ✅ **Interactive Documentation**: Auto-generated Swagger UI
- ✅ **Production Ready**: Docker containerization included

### Additional Features

- Model evaluation with comprehensive metrics
- Data preprocessing and augmentation
- Training visualization and history
- Automated testing suite
- Detailed logging and error handling

## System Requirements

### Minimum Requirements

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.10 or 3.11 (recommended)
- **RAM**: 8 GB
- **Storage**: 5 GB free space
- **Internet**: Required for initial setup

### Recommended Requirements

- **OS**: Windows 11 or Ubuntu 22.04
- **Python**: 3.11
- **RAM**: 16 GB
- **Storage**: 10 GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster training)

## Installation

### Step 1: Clone or Extract the Project

```bash
cd /path/to/face-mask-detection
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python train.py
```

### Running the API Server

```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Docker Deployment

### Build the Docker Image

```bash
docker build -t face-mask-detection .
```

### Run the Container

```bash
docker run -p 8000:8000 face-mask-detection
```

## Project Structure

```
face-mask-detection/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker configuration
├── docker-compose.yml                 # Docker Compose configuration
├── setup.sh                           # Setup automation script (Linux/Mac)
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
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- TensorFlow and Keras teams
- FastAPI framework
- Open-source community
