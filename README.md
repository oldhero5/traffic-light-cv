# Traffic Light & Camera Perception System

A computer vision system for detecting and analyzing traffic lights and cameras from dashcam footage.

[![Build and Publish](https://github.com/oldhero5/traffic-light-cv/actions/workflows/publish.yml/badge.svg)](https://github.com/oldhero5/traffic-light-cv/actions/workflows/publish.yml)
[![Tests](https://github.com/oldhero5/traffic-light-cv/actions/workflows/tests.yml/badge.svg)](https://github.com/oldhero5/traffic-light-cv/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/traffic-light-cv.svg)](https://badge.fury.io/py/traffic-light-cv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/oldhero5/traffic-light-cv/blob/main/LICENSE)

## Features

- Traffic light detection with state classification (red, yellow, green)
- Traffic camera detection
- GPS and IMU data integration
- 3D localization of detected objects
- Relevance estimation for traffic signals
- Mapping visualization of detected objects
- Test mode for dashcam footage simulation

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- CUDA and cuDNN installed (for GPU acceleration)

### Installation from PyPI

The simplest way to install:

```bash
pip install traffic-light-cv
```

### Setup from Source

1. Clone the repository:
```bash
git clone https://github.com/oldhero5/traffic-light-cv.git
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

### Docker

Alternatively, you can use Docker:

```bash
docker pull oldhero5/traffic-light-cv:latest
docker run -it --gpus all oldhero5/traffic-light-cv
```

Or build the image yourself:

```bash
docker build -t traffic-light-cv .
docker run -it --gpus all traffic-light-cv
```

## Usage

### Detection Demo

Run the detection demo with a video file:

```bash
python run_demo.py --input path/to/video.mp4 --output path/to/output --display
```

### Test Mode

Test the system with dashcam footage without driving around:

```bash
python run_test_mode.py --input dashcam.mp4 --camera-type gopro --display --simulate-gps
```

See [TEST_MODE.md](TEST_MODE.md) for detailed instructions.

### Training a Custom Model

Train a custom traffic light detection model:

```bash
python -m src.detection.train --config configs/detection_config.yaml
```

## Documentation

For detailed usage instructions, see:
- [INSTALL.md](INSTALL.md) - Installation instructions
- [USAGE.md](USAGE.md) - Detailed usage guide
- [TEST_MODE.md](TEST_MODE.md) - Test mode documentation

## Project Structure

```
/traffic-light-cv/
  /data/                   # Dataset storage
  /models/                 # Trained models
  /src/
    /detection/            # Traffic light/camera detection
    /localization/         # 3D positioning
    /mapping/              # Map visualization
    /relevance/            # Traffic light relevance
    /utils/                # Helper functions
  /tests/                  # Unit and integration tests
  /notebooks/              # Exploratory analysis
  /configs/                # Configuration files
  /docs/                   # Documentation
```

## License

This project is licensed under the MIT License with Attribution Requirement - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- The project builds upon the [YOLOv8](https://github.com/ultralytics/ultralytics) object detection framework
- Thanks to the open-source traffic light datasets: LISA, DTLD, and BSTLD