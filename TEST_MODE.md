# Test Mode Usage Guide

The Traffic Light & Camera Perception System includes a special test mode designed for experimenting with dashcam footage without driving around. This mode supports GoPro Hero 11, Tiger dashcam, and other common dashcam formats.

## Features

- Work with existing dashcam footage from GoPro Hero 11, Tiger dashcam, and others
- Simulate GPS data if none is available
- Inject simulated traffic lights and cameras into videos
- Real-time visualization with adjustable playback speed
- Map view display with detected objects

## Command-Line Interface

To run the test mode:

```bash
python run_test_mode.py --input path/to/video.mp4 --display --simulate-gps
```

### Command-Line Options

- `--input`: Path to input video file (required)
- `--output`: Path to output directory (default: "output")
- `--config`: Path to configuration file (default: "configs/config.yaml")
- `--gps`: Path to GPS data file (optional)
- `--simulate-gps`: Simulate GPS data if not provided (flag)
- `--start-lat`: Starting latitude for GPS simulation (default: 37.7749)
- `--start-lon`: Starting longitude for GPS simulation (default: -122.4194)
- `--camera-type`: Type of dashcam used (choices: "generic", "gopro", "tiger", "nextbase", "70mai")
- `--display`: Display output in real-time (flag)
- `--speed`: Playback speed multiplier (default: 1.0)
- `--inject-objects`: Inject simulated traffic lights/cameras (flag)
- `--debug`: Enable debug mode (flag)

## Supported Camera Types

The test mode includes optimized settings for several popular dashcam models:

- **GoPro Hero 11**: Wide-angle footage with 118° horizontal FOV
- **Tiger Dashcam**: Ultra-wide footage with 140° horizontal FOV
- **Nextbase**: Standard dashcam format
- **70mai**: Common budget dashcam
- **Generic**: Default camera settings if you're unsure

To specify your camera type:

```bash
python run_test_mode.py --input video.mp4 --camera-type gopro
```

## GPS Simulation

If your dashcam footage doesn't include GPS data, the test mode can simulate a realistic driving route:

```bash
python run_test_mode.py --input video.mp4 --simulate-gps --start-lat 37.7749 --start-lon -122.4194
```

The simulated GPS data will be saved to the output directory for future use.

## Injecting Simulated Objects

To test the detection system without actual traffic lights or cameras in your footage:

```bash
python run_test_mode.py --input video.mp4 --inject-objects
```

This will add simulated traffic lights and cameras to your video that the system can detect.

## Interactive Controls

When running with `--display` option, the following keyboard controls are available:

- **q**: Quit the application
- **p**: Pause/resume playback
- **s**: Save the current frame and map view as snapshots

## Examples

### Basic Test with GoPro Footage

```bash
python run_test_mode.py --input gopro_drive.mp4 --camera-type gopro --display
```

### Comprehensive Test with Simulated Data

```bash
python run_test_mode.py \
  --input dashcam.mp4 \
  --output test_results \
  --camera-type tiger \
  --simulate-gps \
  --inject-objects \
  --display \
  --speed 0.5
```

This will:
1. Use a Tiger dashcam video
2. Save results to "test_results" directory
3. Simulate GPS data for the route
4. Add simulated traffic lights and cameras
5. Display real-time output
6. Play at half speed for easier analysis

### Processing Existing Files with GPS Data

If your dashcam provides separate GPS data files:

```bash
python run_test_mode.py --input video.mp4 --gps gps_data.csv --display
```

## Output

The test mode generates the following outputs:

- Processed video with detections and telemetry overlay
- Individual frames saved periodically
- Map view images showing detected objects
- Simulated GPS data (if requested)
- Object database of detected traffic lights and cameras

## Troubleshooting

- **Slow performance**: Use `--speed` with a value < 1.0 to slow down playback
- **No detections**: Enable `--inject-objects` to verify system operation
- **GPS errors**: Verify CSV format or use `--simulate-gps` instead
- **Incorrect camera FOV**: Specify correct `--camera-type` or adjust FOV in config file