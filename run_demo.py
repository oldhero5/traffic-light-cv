#!/usr/bin/env python3
"""
Demo script for the traffic perception system.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.detection.detector import TrafficObjectDetector
from src.localization.localizer import ObjectLocalizer
from src.mapping.mapper import Mapper
from src.relevance.relevance_estimator import RelevanceEstimator
from src.utils.video_reader import VideoReader
from src.utils.gps_reader import GPSReader
from src.utils.visualizer import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Light Perception System Demo")
    parser.add_argument("--input", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default="output", help="Path to output directory")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--gps", type=str, help="Path to GPS data file")
    parser.add_argument("--display", action="store_true", help="Display output in real-time")
    parser.add_argument("--record", action="store_true", help="Record output video")
    parser.add_argument("--skip-frames", type=int, default=0, help="Number of frames to skip")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    return parser.parse_args()


def setup_logging(debug=False):
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("traffic_perception")


def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return {}


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
    
    # Initialize video reader
    if args.input:
        video_reader = VideoReader(args.input)
    else:
        logger.info("No input video specified, using camera")
        video_reader = VideoReader(0)  # Use default camera
    
    # Get video info
    width = video_reader.width
    height = video_reader.height
    fps = video_reader.fps
    
    logger.info(f"Video info: {width}x{height} @ {fps:.2f} fps")
    
    # Initialize GPS reader if provided
    gps_reader = None
    if args.gps:
        gps_reader = GPSReader(args.gps)
    
    # Initialize components
    detector = TrafficObjectDetector(
        model_path=config.get('detection', {}).get('model_path', 'models/traffic_detector.pt'),
        confidence_threshold=config.get('detection', {}).get('confidence_threshold', 0.25)
    )
    
    localizer = ObjectLocalizer(
        use_kalman=config.get('localization', {}).get('use_kalman', True)
    )
    
    mapper = Mapper(
        distance_threshold=config.get('mapping', {}).get('distance_threshold', 10.0),
        persistence_file=os.path.join(args.output, 'objects.json') if args.output else None
    )
    
    relevance_estimator = RelevanceEstimator()
    
    visualizer = Visualizer(
        output_dir=str(output_dir),
        show_distance=config.get('visualization', {}).get('show_distance', True),
        show_relevance=config.get('visualization', {}).get('show_relevance', True),
        show_state=config.get('visualization', {}).get('show_state', True)
    )
    
    # Initialize video writer if recording
    video_writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(args.output, 'output.mp4')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Processing loop
    try:
        frame_count = 0
        processing_times = []
        
        while True:
            # Read frame
            ret, frame = video_reader.read()
            if not ret:
                logger.info("End of video")
                break
            
            # Skip frames if requested
            if args.skip_frames > 0 and frame_count % (args.skip_frames + 1) != 0:
                frame_count += 1
                continue
            
            # Get GPS data
            gps_data = None
            if gps_reader:
                gps_data = gps_reader.get_data(frame_count)
            
            # Process frame
            start_time = time.time()
            
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
            if frame_count % 30 == 0:  # Every 30 frames
                map_image = visualizer.create_map_visualization(mapped_objects)
                map_path = os.path.join(args.output, f"map_{frame_count:06d}.jpg")
                cv2.imwrite(map_path, map_image)
            
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            # Log processing time
            if frame_count % 100 == 0:
                avg_fps = 1.0 / (sum(processing_times) / len(processing_times))
                logger.info(f"Frame {frame_count}, Avg FPS: {avg_fps:.2f}")
                processing_times = []
            
            # Display output if requested
            if args.display:
                cv2.imshow("Traffic Perception", output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write output frame if recording
            if video_writer:
                video_writer.write(output_frame)
            
            # Save output frame
            if frame_count % 10 == 0:  # Every 10 frames
                output_path = os.path.join(args.output, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(output_path, output_frame)
            
            frame_count += 1
        
        logger.info(f"Processed {frame_count} frames")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Error occurred: {e}")
    finally:
        # Clean up
        video_reader.release()
        if video_writer:
            video_writer.release()
        if args.display:
            cv2.destroyAllWindows()
        logger.info("Processing complete")


if __name__ == "__main__":
    main()