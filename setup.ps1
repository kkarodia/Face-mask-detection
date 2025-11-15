# Face Mask Detection Project Setup Script

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Face Mask Detection Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "ℹ Checking Python version..." -ForegroundColor Yellow
$pythonVersion = (python --version 2>&1).ToString().Split()[1]
Write-Host "✓ Python $pythonVersion detected" -ForegroundColor Green

# Create directory structure
Write-Host "ℹ Creating directory structure..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path data\raw, data\processed, models, logs, mlruns | Out-Null
Write-Host "✓ Directory structure created" -ForegroundColor Green

# Create .gitkeep files
New-Item -ItemType File -Force -Path data\raw\.gitkeep, data\processed\.gitkeep, models\.gitkeep, logs\.gitkeep | Out-Null

# Create virtual environment
Write-Host "ℹ Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv
Write-Host "✓ Virtual environment created" -ForegroundColor Green

# Activate virtual environment
Write-Host "ℹ Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host "ℹ Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "✓ Pip upgraded" -ForegroundColor Green

# Install dependencies
Write-Host "ℹ Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
pip install -r requirements.txt
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Dataset instructions
Write-Host ""
Write-Host "ℹ Dataset Setup Required:" -ForegroundColor Yellow
Write-Host "  1. Visit: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset"
Write-Host "  2. Download the dataset"
Write-Host "  3. Extract to: data\raw\"
Write-Host "  Expected structure:"
Write-Host "     data\raw\"
Write-Host "       ├── with_mask\"
Write-Host "       └── without_mask\"
Write-Host ""

# Check Docker
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "✓ Docker is installed" -ForegroundColor Green
} else {
    Write-Host "ℹ Docker is not installed. Install Docker for containerization." -ForegroundColor Yellow
}

# Create .env file
if (-not (Test-Path .env)) {
    Write-Host "ℹ Creating .env file..." -ForegroundColor Yellow
    @"
# Environment Variables
PYTHONUNBUFFERED=1
TF_CPP_MIN_LOG_LEVEL=2
MODEL_PATH=models/mask_detection_model.keras
LOG_LEVEL=INFO
"@ | Out-File -FilePath .env -Encoding UTF8
    Write-Host "✓ .env file created" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✓ Setup completed successfully!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Activate virtual environment: .\venv\Scripts\Activate.ps1"
Write-Host "  2. Download and extract dataset to data\raw\"
Write-Host "  3. Run preprocessing: python src\data_preprocessing.py"
Write-Host "  4. Train model: python src\train.py"
Write-Host "  5. Evaluate model: python src\evaluate.py"
Write-Host "  6. Start API: uvicorn api.main:app --reload"
Write-Host "  7. Build Docker image: docker build -t face-mask-detection ."
Write-Host "  8. Run container: docker run -p 8000:8000 face-mask-detection"
Write-Host ""
Write-Host "For detailed instructions, see README.md"
Write-Host ""