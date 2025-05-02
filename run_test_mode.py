#!/usr/bin/env python3
"""
Test mode for the Traffic Light & Camera Perception System.

This script allows testing with GoPro Hero 11 or Tiger dashcam footage without driving around.
It includes options for simulating GPS data and handling various dashcam-specific formats.
"""
import argparse
import json
import logging
import os
import sys
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml
import math

from src.detection.detector import TrafficObjectDetector
from src.localization.localizer import ObjectLocalizer
from src.mapping.mapper import Mapper
from src.relevance.relevance_estimator import RelevanceEstimator
from src.utils.video_reader import VideoReader
from src.utils.gps_reader import GPSReader
from src.utils.visualizer import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Light Perception System Test Mode")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="output", help="Path to output directory")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--gps", type=str, help="Path to GPS data file (optional)")
    parser.add_argument("--simulate-gps", action="store_true", help="Simulate GPS data if not provided")
    parser.add_argument("--start-lat", type=float, default=37.7749, help="Starting latitude for GPS simulation")
    parser.add_argument("--start-lon", type=float, default=-122.4194, help="Starting longitude for GPS simulation")
    parser.add_argument("--camera-type", type=str, default="generic", 
                        choices=["generic", "gopro", "tiger", "nextbase", "70mai"], 
                        help="Type of dashcam used")
    parser.add_argument("--display", action="store_true", help="Display output in real-time")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--inject-objects", action="store_true", help="Inject simulated traffic lights/cameras")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    return parser.parse_args()


def setup_logging(debug=False):
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("traffic_perception_test")


def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return {}


def get_camera_parameters(camera_type):
    """Get camera parameters based on camera type."""
    camera_params = {
        "generic": {
            "fov_horizontal": 60.0,
            "fov_vertical": 35.0,
            "rotate": False,
            "flip": False
        },
        "gopro": {
            "fov_horizontal": 118.0,  # Hero 11 wide mode
            "fov_vertical": 67.0,
            "rotate": False,
            "flip": False
        },
        "tiger": {
            "fov_horizontal": 140.0,
            "fov_vertical": 85.0,
            "rotate": False,
            "flip": False
        },
        "nextbase": {
            "fov_horizontal": 140.0,
            "fov_vertical": 80.0,
            "rotate": False,
            "flip": False
        },
        "70mai": {
            "fov_horizontal": 130.0,
            "fov_vertical": 75.0,
            "rotate": False,
            "flip": False
        }
    }
    
    return camera_params.get(camera_type, camera_params["generic"])


def generate_simulated_gps(
    start_lat, 
    start_lon, 
    frame_count, 
    fps, 
    speed_kph=30.0,
    route_turns=None
):
    """
    Generate simulated GPS data for testing.
    
    Args:
        start_lat: Starting latitude
        start_lon: Starting longitude
        frame_count: Number of frames in the video
        fps: Frames per second
        speed_kph: Vehicle speed in km/h
        route_turns: List of (frame, bearing_change) tuples for route turns
    
    Returns:
        List of GPS data points
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating simulated GPS data")
    
    # Convert speed to meters per second
    speed_mps = speed_kph / 3.6
    
    # Calculate distance traveled per frame
    meters_per_frame = speed_mps / fps
    
    # Initial bearing (degrees, 0 = North, 90 = East)
    bearing = 90.0  # Start going East
    
    # If no turns specified, generate some random ones
    if route_turns is None:
        # Generate random turns every ~10-20 seconds
        turn_intervals = np.random.randint(10 * fps, 20 * fps, size=int(frame_count / (15 * fps)))
        turn_frames = np.cumsum(turn_intervals)
        turn_frames = turn_frames[turn_frames < frame_count]
        
        # Generate random bearing changes (-30 to +30 degrees)
        bearing_changes = np.random.uniform(-30, 30, size=len(turn_frames))
        
        route_turns = list(zip(turn_frames, bearing_changes))
        
    # Generate GPS points
    gps_data = []
    current_lat = start_lat
    current_lon = start_lon
    timestamp = datetime.now()
    
    for frame in range(frame_count):
        # Check if we need to turn
        for turn_frame, bearing_change in route_turns:
            if frame == turn_frame:
                bearing = (bearing + bearing_change) % 360
                logger.debug(f"Turn at frame {frame}: new bearing {bearing:.1f}°")
        
        # Add some noise to make it more realistic
        speed_noise = 1.0 + np.random.normal(0, 0.05)  # 5% speed variation
        bearing_noise = np.random.normal(0, 0.5)  # 0.5° bearing variation
        
        # Calculate movement
        distance = meters_per_frame * speed_noise
        actual_bearing = bearing + bearing_noise
        
        # Convert distance and bearing to lat/lon change
        # Earth radius in meters
        earth_radius = 6371000
        
        # Convert to radians
        lat1 = math.radians(current_lat)
        lon1 = math.radians(current_lon)
        brng = math.radians(actual_bearing)
        
        # Calculate new position
        lat2 = math.asin(
            math.sin(lat1) * math.cos(distance / earth_radius) +
            math.cos(lat1) * math.sin(distance / earth_radius) * math.cos(brng)
        )
        
        lon2 = lon1 + math.atan2(
            math.sin(brng) * math.sin(distance / earth_radius) * math.cos(lat1),
            math.cos(distance / earth_radius) - math.sin(lat1) * math.sin(lat2)
        )
        
        # Convert back to degrees
        current_lat = math.degrees(lat2)
        current_lon = math.degrees(lon2)
        
        # Create GPS data point
        gps_point = {
            "frame": frame,
            "timestamp": timestamp.timestamp(),
            "latitude": current_lat,
            "longitude": current_lon,
            "altitude": 10.0 + np.random.normal(0, 0.5),  # Random altitude around 10m
            "speed": speed_kph * speed_noise / 3.6,  # m/s
            "heading": actual_bearing,
        }
        
        gps_data.append(gps_point)
        
        # Increment timestamp
        timestamp += timedelta(seconds=1.0/fps)
    
    logger.info(f"Generated {len(gps_data)} GPS data points")
    return gps_data


def save_simulated_gps(gps_data, output_file):
    """
    Save simulated GPS data to a file.
    
    Args:
        gps_data: List of GPS data points
        output_file: Path to output file
    """
    logger = logging.getLogger(__name__)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Determine file extension
    _, ext = os.path.splitext(output_file)
    
    if ext.lower() == '.csv':
        # Save as CSV
        with open(output_file, 'w') as f:
            # Write header
            f.write("frame,timestamp,latitude,longitude,altitude,speed,heading\n")
            
            # Write data
            for point in gps_data:
                f.write(f"{point['frame']},{point['timestamp']},{point['latitude']},"
                        f"{point['longitude']},{point['altitude']},{point['speed']},"
                        f"{point['heading']}\n")
        
        logger.info(f"Saved GPS data to {output_file}")
    
    elif ext.lower() == '.json':
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(gps_data, f, indent=2)
        
        logger.info(f"Saved GPS data to {output_file}")
    
    else:
        logger.warning(f"Unsupported file extension for GPS data: {ext}")


class SimulatedGPSReader:
    """GPS data reader for simulated GPS data."""
    
    def __init__(self, gps_data):
        """
        Initialize the GPS reader.
        
        Args:
            gps_data: List of GPS data points
        """
        self.logger = logging.getLogger(__name__)
        self.data = gps_data
        self.loaded = True
        
        self.logger.info(f"Initialized simulated GPS reader with {len(self.data)} points")
    
    def get_data(self, frame_number):
        """
        Get GPS data for a specific frame.
        
        Args:
            frame_number: Frame number
            
        Returns:
            GPS data dictionary or None if not found
        """
        if not self.data:
            return None
        
        # Find exact frame match
        for point in self.data:
            if point.get("frame") == frame_number:
                return point
        
        # If no exact match and we have frame numbers, interpolate
        if "frame" in self.data[0]:
            # Find surrounding frames
            prev_point = None
            next_point = None
            
            for point in self.data:
                if point["frame"] < frame_number:
                    if prev_point is None or point["frame"] > prev_point["frame"]:
                        prev_point = point
                if point["frame"] > frame_number:
                    if next_point is None or point["frame"] < next_point["frame"]:
                        next_point = point
            
            # If we have both surrounding points, interpolate
            if prev_point and next_point:
                frame_range = next_point["frame"] - prev_point["frame"]
                if frame_range == 0:
                    return prev_point
                
                # Calculate interpolation factor
                factor = (frame_number - prev_point["frame"]) / frame_range
                
                # Interpolate each value
                interpolated = {}
                for key in prev_point:
                    if key != "frame" and key in next_point:
                        if isinstance(prev_point[key], (int, float)):
                            interpolated[key] = prev_point[key] + factor * (next_point[key] - prev_point[key])
                        else:
                            interpolated[key] = prev_point[key]
                
                interpolated["frame"] = frame_number
                return interpolated
            
            # If we only have one surrounding point, use it
            if prev_point:
                return prev_point
            if next_point:
                return next_point
        
        # Fallback: use frame number as index (modulo data length)
        idx = frame_number % len(self.data)
        return self.data[idx]


def inject_traffic_objects(frame, frame_number, total_frames, objects_config=None):
    """
    Inject simulated traffic lights and cameras into the frame.
    
    Args:
        frame: Input frame
        frame_number: Current frame number
        total_frames: Total number of frames
        objects_config: Configuration for injected objects
    
    Returns:
        Frame with injected objects
    """
    # Default config if none provided
    if objects_config is None:
        objects_config = {
            "traffic_lights": [
                # Red light appearing in the middle of the video
                {
                    "start_frame": int(total_frames * 0.3),
                    "end_frame": int(total_frames * 0.4),
                    "position": (0.5, 0.4),  # x, y as fraction of frame size
                    "size": (60, 150),  # Width, height in pixels
                    "state": "red",
                    "state_change": None  # No state change
                },
                # Light changing from red to green
                {
                    "start_frame": int(total_frames * 0.6),
                    "end_frame": int(total_frames * 0.7),
                    "position": (0.6, 0.35),
                    "size": (50, 120),
                    "state": "red",
                    "state_change": {
                        "frame": int(total_frames * 0.65),
                        "new_state": "green"
                    }
                }
            ],
            "cameras": [
                # Traffic camera appearing near the end
                {
                    "start_frame": int(total_frames * 0.8),
                    "end_frame": int(total_frames * 0.9),
                    "position": (0.3, 0.3),
                    "size": (80, 80)
                }
            ]
        }
    
    # Make a copy of the frame
    output = frame.copy()
    
    # Inject traffic lights
    for tl in objects_config.get("traffic_lights", []):
        if tl["start_frame"] <= frame_number <= tl["end_frame"]:
            # Determine state
            state = tl["state"]
            if tl.get("state_change") and frame_number >= tl["state_change"]["frame"]:
                state = tl["state_change"]["new_state"]
            
            # Calculate position
            h, w = frame.shape[:2]
            x = int(tl["position"][0] * w)
            y = int(tl["position"][1] * h)
            width, height = tl["size"]
            
            # Draw traffic light housing
            cv2.rectangle(output, (x, y), (x + width, y + height), (100, 100, 100), -1)
            cv2.rectangle(output, (x, y), (x + width, y + height), (50, 50, 50), 2)
            
            # Draw lights
            light_radius = width // 3
            light_spacing = height // 4
            
            # Red light
            red_y = y + light_spacing
            cv2.circle(output, (x + width//2, red_y), light_radius, 
                      (0, 0, 255) if state == "red" else (30, 30, 100), -1)
            
            # Yellow light
            yellow_y = y + 2 * light_spacing
            cv2.circle(output, (x + width//2, yellow_y), light_radius, 
                      (0, 255, 255) if state == "yellow" else (30, 100, 100), -1)
            
            # Green light
            green_y = y + 3 * light_spacing
            cv2.circle(output, (x + width//2, green_y), light_radius, 
                      (0, 255, 0) if state == "green" else (30, 100, 30), -1)
    
    # Inject cameras
    for cam in objects_config.get("cameras", []):
        if cam["start_frame"] <= frame_number <= cam["end_frame"]:
            # Calculate position
            h, w = frame.shape[:2]
            x = int(cam["position"][0] * w)
            y = int(cam["position"][1] * h)
            width, height = cam["size"]
            
            # Draw camera housing
            cv2.rectangle(output, (x, y), (x + width, y + height), (50, 50, 200), -1)
            cv2.rectangle(output, (x, y), (x + width, y + height), (0, 0, 0), 2)
            
            # Draw lens
            lens_radius = min(width, height) // 3
            cv2.circle(output, (x + width//2, y + height//2), lens_radius, (0, 0, 0), -1)
            cv2.circle(output, (x + width//2, y + height//2), lens_radius - 2, (100, 100, 255), 2)
    
    return output


class TestModeGPSReader(GPSReader):
    """Extended GPS reader with test mode features."""
    
    def __init__(self, gps_file, simulated_data=None):
        """
        Initialize the GPS reader.
        
        Args:
            gps_file: Path to GPS data file or None
            simulated_data: Simulated GPS data if file is None
        """
        self.logger = logging.getLogger(__name__)
        
        if gps_file and os.path.exists(gps_file):
            super().__init__(gps_file)
        elif simulated_data:
            self.data = simulated_data
            self.loaded = True
            self.logger.info(f"Using simulated GPS data with {len(self.data)} points")
        else:
            self.data = []
            self.loaded = False
            self.logger.warning("No GPS data available")
    
    def get_data(self, frame_number):
        """
        Get GPS data for a specific frame.
        
        Args:
            frame_number: Frame number
            
        Returns:
            GPS data dictionary or None if not found
        """
        if not self.loaded or not self.data:
            return None
        
        # Find exact frame match first
        for point in self.data:
            if "frame" in point and point["frame"] == frame_number:
                return point
        
        # If we have frame numbers in the data, interpolate
        frame_points = [p for p in self.data if "frame" in p]
        if frame_points:
            # Find surrounding frames
            prev_points = [p for p in frame_points if p["frame"] <= frame_number]
            next_points = [p for p in frame_points if p["frame"] >= frame_number]
            
            prev_point = max(prev_points, key=lambda p: p["frame"]) if prev_points else None
            next_point = min(next_points, key=lambda p: p["frame"]) if next_points else None
            
            # If we have both surrounding points, interpolate
            if prev_point and next_point and prev_point != next_point:
                frame_range = next_point["frame"] - prev_point["frame"]
                factor = (frame_number - prev_point["frame"]) / frame_range
                
                # Interpolate each value
                interpolated = {}
                for key in prev_point:
                    if key != "frame" and key in next_point:
                        if isinstance(prev_point[key], (int, float)):
                            interpolated[key] = prev_point[key] + factor * (next_point[key] - prev_point[key])
                        else:
                            interpolated[key] = prev_point[key]
                
                interpolated["frame"] = frame_number
                return interpolated
            
            # If we only have one surrounding point, use it
            if prev_point:
                return prev_point
            if next_point:
                return next_point
        
        # Fall back to parent class behavior
        return super().get_data(frame_number)


def main():
    args = parse_args()
    logger = setup_logging(args.debug)
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        config = {}
        logger.warning(f"Configuration file not found: {config_path}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get camera parameters for the selected camera type
    camera_params = get_camera_parameters(args.camera_type)
    logger.info(f"Using camera parameters for {args.camera_type}: {camera_params}")
    
    # Update config with camera parameters
    if "camera" not in config:
        config["camera"] = {}
    config["camera"]["fov_horizontal"] = camera_params["fov_horizontal"]
    config["camera"]["fov_vertical"] = camera_params["fov_vertical"]
    
    # Initialize video reader
    if args.input:
        video_reader = VideoReader(args.input)
    else:
        logger.error("Input video required")
        return 1
    
    # Get video info
    width = video_reader.width
    height = video_reader.height
    fps = video_reader.fps
    total_frames = video_reader.total_frames
    
    logger.info(f"Video info: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
    
    # Generate or load GPS data
    gps_reader = None
    if args.gps:
        # Load GPS from file
        gps_reader = TestModeGPSReader(args.gps)
    elif args.simulate_gps:
        # Simulate GPS data
        simulated_gps = generate_simulated_gps(
            args.start_lat, args.start_lon, total_frames, fps
        )
        
        # Save simulated GPS data
        gps_file = os.path.join(args.output, "simulated_gps.csv")
        save_simulated_gps(simulated_gps, gps_file)
        
        # Create GPS reader with simulated data
        gps_reader = TestModeGPSReader(None, simulated_gps)
    
    # Initialize components
    detector = TrafficObjectDetector(
        model_path=config.get("detection", {}).get("model_path", "models/traffic_detector.pt"),
        confidence_threshold=config.get("detection", {}).get("confidence_threshold", 0.25)
    )
    
    localizer = ObjectLocalizer(
        use_kalman=config.get("localization", {}).get("use_kalman", True)
    )
    
    mapper = Mapper(
        distance_threshold=config.get("mapping", {}).get("distance_threshold", 10.0),
        persistence_file=os.path.join(args.output, "objects.json") if args.output else None
    )
    
    relevance_estimator = RelevanceEstimator()
    
    visualizer = Visualizer(
        output_dir=str(output_dir),
        show_distance=config.get("visualization", {}).get("show_distance", True),
        show_relevance=config.get("visualization", {}).get("show_relevance", True),
        show_state=config.get("visualization", {}).get("show_state", True)
    )
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(args.output, "output.mp4")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Processing loop
    try:
        frame_count = 0
        processing_times = []
        
        # Create map visualization window if display enabled
        if args.display:
            cv2.namedWindow("Traffic Perception", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Map View", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Map View", 600, 400)
        
        start_time = time.time()
        next_frame_time = start_time
        
        while True:
            # Control playback speed
            if args.display and args.speed != 0:
                current_time = time.time()
                if current_time < next_frame_time:
                    time.sleep(max(0, next_frame_time - current_time))
                next_frame_time = time.time() + (1.0 / fps) / args.speed
            
            # Read frame
            ret, frame = video_reader.read()
            if not ret:
                logger.info("End of video")
                break
            
            # Apply camera-specific transformations
            if camera_params["rotate"]:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            if camera_params["flip"]:
                frame = cv2.flip(frame, 1)  # Horizontal flip
            
            # Inject simulated traffic objects if requested
            if args.inject_objects:
                frame = inject_traffic_objects(frame, frame_count, total_frames)
            
            # Get GPS data
            gps_data = None
            if gps_reader:
                gps_data = gps_reader.get_data(frame_count)
            
            # Process frame
            process_start = time.time()
            
            # Detect traffic objects
            detections = detector.detect(frame)
            
            # Localize objects
            locations = localizer.localize(detections, frame, gps_data)
            
            # Determine relevance
            relevance = relevance_estimator.estimate(detections, frame, gps_data)
            
            # Update map
            mapped_objects = mapper.update(locations, frame_count)
            
            # Visualize results
            output_frame = visualizer.visualize(
                frame, detections, locations, relevance, mapped_objects
            )
            
            # Create map visualization
            map_image = visualizer.create_map_visualization(mapped_objects)
            
            process_end = time.time()
            processing_time = process_end - process_start
            processing_times.append(processing_time)
            
            # Add telemetry overlay
            cv2.putText(
                output_frame,
                f"Frame: {frame_count}/{total_frames} | FPS: {1.0/processing_time:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            
            if gps_data:
                cv2.putText(
                    output_frame,
                    f"GPS: {gps_data.get('latitude', 0):.6f}, {gps_data.get('longitude', 0):.6f} | "
                    f"Speed: {gps_data.get('speed', 0)*3.6:.1f} km/h | "
                    f"Heading: {gps_data.get('heading', 0):.1f}°",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            
            # Log processing time
            if frame_count % 100 == 0:
                avg_fps = 1.0 / (sum(processing_times) / len(processing_times))
                logger.info(f"Frame {frame_count}/{total_frames}, Avg FPS: {avg_fps:.2f}")
                processing_times = []
            
            # Display output if requested
            if args.display:
                cv2.imshow("Traffic Perception", output_frame)
                cv2.imshow("Map View", map_image)
                key = cv2.waitKey(1) & 0xFF
                
                # Handle key presses
                if key == ord('q'):
                    break
                elif key == ord('p'):  # Pause
                    cv2.waitKey(0)
                elif key == ord('s'):  # Save current frame
                    snapshot_path = os.path.join(args.output, f"snapshot_{frame_count:06d}.jpg")
                    cv2.imwrite(snapshot_path, output_frame)
                    map_path = os.path.join(args.output, f"map_{frame_count:06d}.jpg")
                    cv2.imwrite(map_path, map_image)
                    logger.info(f"Saved snapshot to {snapshot_path}")
            
            # Write output frame
            video_writer.write(output_frame)
            
            # Save frames and map periodically
            if frame_count % 30 == 0:
                frame_path = os.path.join(args.output, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, output_frame)
                
                map_path = os.path.join(args.output, f"map_{frame_count:06d}.jpg")
                cv2.imwrite(map_path, map_image)
            
            frame_count += 1
        
        logger.info(f"Processed {frame_count} frames")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Error occurred: {e}")
    finally:
        # Clean up
        video_reader.release()
        video_writer.release()
        if args.display:
            cv2.destroyAllWindows()
        logger.info("Processing complete")


if __name__ == "__main__":
    sys.exit(main())