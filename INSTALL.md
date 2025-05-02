# Installation Guide

This guide provides instructions for installing and setting up the Traffic Light & Camera Perception System.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for real-time performance)
- CUDA and cuDNN installed (for GPU acceleration)
- OpenCV dependencies

## Environment Setup

### Option 1: Using pip

1. Clone the repository:
```bash
git clone https://github.com/yourusername/traffic-light-cv.git
cd traffic-light-cv
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

### Option 2: Using Docker

1. Clone the repository:
```bash
git clone https://github.com/yourusername/traffic-light-cv.git
cd traffic-light-cv
```

2. Build the Docker image:
```bash
docker build -t traffic-light-cv .
```

3. Run the Docker container:
```bash
docker run -it --gpus all -v $(pwd):/app traffic-light-cv
```

## Installing System Dependencies

### Ubuntu/Debian

Install required system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-opencv
```

### macOS

Install dependencies using Homebrew:

```bash
brew update
brew install opencv
brew install pkg-config
```

### Windows

For Windows, the required dependencies should be automatically installed with the Python packages. However, you might need to install Visual C++ Build Tools if you encounter issues during installation.

## CUDA Setup (for GPU acceleration)

For optimal performance, install CUDA and cuDNN:

1. Install CUDA Toolkit (version 11.3 or compatible with your PyTorch version):
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
   - Follow the installation instructions for your platform

2. Install cuDNN:
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - Follow the installation instructions for your platform

3. Verify CUDA installation:
```bash
nvcc --version
```

4. Verify PyTorch CUDA support:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Downloading Pretrained Models

The system uses YOLO-based models for traffic light and camera detection. To download a pretrained model:

```bash
# Create the models directory if it doesn't exist
mkdir -p models

# Download YOLOv8 base model (will be fine-tuned during training)
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt').save('models/yolov8s.pt')"
```

## Dataset Setup

To prepare the dataset for training:

1. Create the dataset directory structure:
```bash
mkdir -p data/train/images data/train/labels \
         data/val/images data/val/labels \
         data/test/images data/test/labels
```

2. Download and extract datasets:
   - LISA Traffic Light Dataset: [LISA](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset)
   - DTLD: [DriveU Traffic Light Dataset](https://www.uni-ulm.de/en/in/driveu/projects/driveu-traffic-light-dataset/)
   - BSTLD: [Bosch Small Traffic Lights Dataset](https://hci.iwr.uni-heidelberg.de/node/6132)

3. Convert annotations to YOLO format:
```bash
python -m src.utils.data_utils convert_annotations \
  --input path/to/annotations.json \
  --output data/train/labels \
  --format yolo \
  --image-dir path/to/images
```

## Troubleshooting

### Common Issues

1. **ImportError: libGL.so.1: cannot open shared object file**:
   - Install OpenGL libraries: `sudo apt-get install libgl1-mesa-glx`

2. **ImportError: libglib-2.0.so.0: cannot open shared object file**:
   - Install GLib: `sudo apt-get install libglib2.0-0`

3. **CUDA out of memory**:
   - Reduce batch size in `configs/detection_config.yaml`
   - Try using a smaller model (YOLOv8n instead of YOLOv8s)

4. **ModuleNotFoundError**:
   - Ensure you have activated the virtual environment
   - Check that all dependencies are installed: `pip install -r requirements.txt`

### Verifying Installation

Run the system tests to verify your installation:

```bash
python run_tests.py
```

All tests should pass if the installation is successful.