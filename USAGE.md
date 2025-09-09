# Usage Guide

This guide provides instructions for using the Traffic Light & Camera Perception System.

## Command-Line Interface

The system provides a command-line interface for training, testing, and running the perception system.

### Running the Demo

To run the demo with default settings:

```bash
python run_demo.py --input path/to/video.mp4 --display
```

#### Demo Options

- `--input`: Path to input video file (required)
- `--output`: Path to output directory (default: "output")
- `--config`: Path to configuration file (default: "configs/config.yaml")
- `--gps`: Path to GPS data file (optional)
- `--display`: Display output in real-time (flag)
- `--record`: Record output video (flag)
- `--skip-frames`: Number of frames to skip (default: 0)
- `--debug`: Enable debug mode (flag)

Example:

```bash
python run_demo.py \
  --input data/samples/traffic_video.mp4 \
  --output results \
  --config configs/config.yaml \
  --gps data/samples/gps_data.csv \
  --display \
  --record
```

### Training the Detection Model

To train the traffic light and camera detection model:

```bash
python -m src.detection.train --config configs/detection_config.yaml
```

#### Training Options

- `--config`: Path to training configuration file (default: "configs/detection_config.yaml")
- `--data`: Path to dataset configuration (default: "data/dataset.yaml")
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--img-size`: Image size (default: 640)
- `--weights`: Initial weights path (default: "yolov8s.pt")
- `--device`: Device to use (default: auto-select)

Example:

```bash
python -m src.detection.train \
  --config configs/detection_config.yaml \
  --data data/dataset.yaml \
  --epochs 50 \
  --batch-size 8 \
  --weights models/yolov8s.pt
```

### Running Tests

To run the system tests:

```bash
python run_tests.py
```

#### Test Options

- `--test-dir`: Directory containing tests (default: "tests")
- `--pattern`: Test file pattern (default: "test_*.py")
- `--verbose`: Enable verbose output (flag)
- `--module`: Run tests for specific module (e.g., 'detector')

Example:

```bash
python run_tests.py --verbose --module detector
```

## Configuration

The system uses YAML configuration files for various components.

### Main Configuration (config.yaml)

The main configuration file controls the behavior of the entire system:

```yaml
# System settings
system:
  debug: false
  display: true
  skip_frames: 2
  use_gpu: true
  device: "cuda"
  output_dir: "output"

# Detection settings
detection:
  model_path: "models/traffic_detector.pt"
  confidence_threshold: 0.25
  nms_threshold: 0.45
  input_size: 640

# Camera settings
camera:
  calibration_file: "configs/camera_calibration.json"
  fov_horizontal: 60.0
  fov_vertical: 35.0

# Localization settings
localization:
  use_kalman: true
  distance_estimation_method: "size"
```

### Detection Configuration (detection_config.yaml)

Controls the training of the detection model:

```yaml
# Training settings
data: data/dataset.yaml
epochs: 100
batch_size: 16
img_size: 640
weights: yolov8s.pt

# Model settings
model:
  type: yolov8
  size: s  # n, s, m, l, x
  pretrained: true

# Augmentation settings
augmentation:
  mosaic: true
  mixup: true
```

## Camera Calibration

Before using the system with a specific camera, it's recommended to calibrate the camera for better distance estimation:

1. Collect calibration images:
   - Print a checkerboard pattern and take multiple images of it from different angles
   - Place the images in a folder

2. Run the calibration script:
```bash
python -m src.utils.calibration.calibrate_from_images \
  --input calibration_images \
  --output configs/camera_calibration.json \
  --width 9 \
  --height 6 \
  --square-size 0.025
```

## Processing Your Own Data

To process your own data:

1. Prepare your video file:
   - The system works best with 1080p or higher resolution videos
   - Dashcam footage works well

2. (Optional) Prepare GPS data:
   - Create a CSV file with timestamp, latitude, longitude, heading, etc.
   - Format: `timestamp,latitude,longitude,heading,speed`

3. Run the system:
```bash
python run_demo.py --input your_video.mp4 --gps your_gps.csv --display
```

4. Output:
   - Processed video frames will be saved to the output directory
   - Map visualizations of detected objects will be saved
   - If recording is enabled, an output video will be created

## Object Tracking

The system includes M1-optimized DeepSORT tracking for maintaining object identity across frames.

### Tracking Configuration

Configure tracking in your config file:

```yaml
# Tracking settings
tracking:
  enabled: true
  tracker_type: "deepsort"  # deepsort, sort, deep_sort_realtime
  max_distance: 0.2
  min_confidence: 0.3
  max_age: 70
  n_init: 3
  nms_max_overlap: 1.0
  max_iou_distance: 0.7
  
  # M1 Optimizations
  use_mps: true              # Metal Performance Shaders
  use_neural_engine: true    # CoreML Neural Engine
  batch_similarity: true     # Batch processing for efficiency
  
  # Performance targets
  target_fps: 60             # Minimum FPS requirement
  max_objects: 100           # Maximum tracked objects
```

### API Usage

#### Basic Tracking

```python
from src.tracking.deepsort_tracker import DeepSORTTracker
from src.detection.detector import Detection

# Initialize tracker
tracker = DeepSORTTracker(
    model_path="models/deep_sort.pb",
    max_distance=0.2,
    min_confidence=0.3,
    max_age=70,
    n_init=3
)

# Process detections for a frame
detections = [
    Detection(
        class_id=0,
        class_name="traffic_light",
        confidence=0.9,
        bbox=(100, 100, 200, 200)
    )
]

# Update tracker and get tracks
tracks = tracker.update(detections)

for track in tracks:
    print(f"Track ID: {track.track_id}")
    print(f"Class: {track.class_name}")
    print(f"Confidence: {track.confidence:.2f}")
    print(f"Position: {track.bbox}")
```

#### M1 Performance Optimizations

```python
# Enable M1 optimizations
tracker.enable_m1_optimizations()

# Check Metal Performance Shaders availability
if tracker.metal_utils.mps_available:
    print("MPS acceleration enabled")

# Validate performance requirements
validation = tracker.validate_30_frame_tracking()
if validation['meets_requirement']:
    print(f"✅ 30+ frame tracking validated")
    print(f"Max persistence: {validation['max_persistence']} frames")

# Get FPS estimate
fps = tracker.get_fps_estimate()
if fps >= 60:
    print(f"✅ 60+ FPS requirement met: {fps:.1f} FPS")
```

#### Deep-Sort-Realtime Compatibility

```python
# Enable compatibility mode
tracker.enable_deep_sort_realtime_mode()

# Use realtime-style detections
realtime_detections = [
    ([100, 100, 200, 200], 0.9, "traffic_light"),
    ([300, 150, 400, 250], 0.85, "speed_camera")
]

tracks = tracker.update_with_realtime(realtime_detections)
```

#### Advanced Features

```python
# Calculate tracking confidence
confidence = tracker._calculate_tracking_confidence(track, detection)

# Memory usage monitoring
memory_stats = tracker.metal_utils.get_memory_usage()
print(f"MPS Memory: {memory_stats.get('mps_allocated_mb', 0):.1f} MB")

# Benchmark operations
benchmark_result = tracker.metal_utils.benchmark_operation(
    tracker.metal_utils.batch_cosine_similarity,
    features1, features2,
    num_iterations=100
)
print(f"Mean time: {benchmark_result['mean_time_ms']:.2f} ms")
print(f"FPS: {benchmark_result['ops_per_second']:.1f}")
```

### Performance Testing

Test tracking performance with M1 optimizations:

```bash
# Run performance tests
python run_tests.py --pytest --performance --m1-only --verbose

# Validate 60+ FPS requirement
python test_fps_validation.py

# Run memory efficiency tests
python run_tests.py --pytest -m performance -k memory

# Benchmark tracking operations
python -m pytest tests/benchmarks/test_deepsort_performance.py --benchmark-only
```

### M1 Hardware Requirements

The tracking system is optimized for M1 MacBook Pro and requires:

- **Hardware**: M1, M1 Pro, M1 Max, or M1 Ultra
- **Memory**: 8GB+ unified memory (16GB+ recommended for 4K)
- **Performance Targets**:
  - 4K: ≥60 FPS
  - 1080p: ≥120 FPS
  - Latency: <100ms end-to-end
  - Memory: <2GB for 4K processing
  - Neural Engine: >80% utilization

### Troubleshooting Tracking

- **Low FPS**: Check MPS availability with `torch.backends.mps.is_available()`
- **Memory issues**: Reduce `max_objects` or enable batch processing
- **Poor tracking**: Adjust `max_distance` and `min_confidence` thresholds
- **Track fragmentation**: Increase `max_age` and decrease `n_init`
- **M1 not detected**: Ensure PyTorch ≥2.1.0 with MPS support

## Advanced Usage

### Creating a Custom Detection Model

To create a custom detection model:

1. Prepare your dataset according to the YOLO format
2. Update the class names in `data/dataset.yaml`
3. Train the model:
```bash
python -m src.detection.train \
  --data data/dataset.yaml \
  --epochs 100 \
  --weights yolov8s.pt \
  --name custom_model
```

### Real-time Usage

For real-time usage with a camera:

```bash
python run_demo.py --display
```

This will use the default camera (index 0). To use a different camera, specify the index:

```bash
python run_demo.py --input 1 --display
```

### Using Docker

If you're using Docker:

```bash
docker run -it --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  traffic-light-cv python run_demo.py --input /app/data/video.mp4
```

## Troubleshooting

- **Low FPS**: Try skipping frames with `--skip-frames 2` or reduce the resolution
- **Poor detection**: Ensure proper lighting conditions in the video
- **Error loading model**: Check that the model path is correct and the model exists
- **GPS alignment issues**: Ensure your GPS data timestamps align with video frames