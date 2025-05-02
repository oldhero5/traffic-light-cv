#!/usr/bin/env python3
"""
Main entry point for the traffic light & camera perception system.
"""
import argparse
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.detection.detector import TrafficObjectDetector
from src.localization.localizer import ObjectLocalizer
from src.mapping.mapper import Mapper
from src.relevance.relevance_estimator import RelevanceEstimator
from src.utils.gps_reader import GPSReader
from src.utils.video_reader import VideoReader
from src.utils.visualizer import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Light & Camera Perception System")
    parser.add_argument("--input", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default="output", help="Path to output directory")
    parser.add_argument("--model", type=str, default="models/traffic_detector.pt", help="Path to detection model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--gps", type=str, help="Path to GPS data file")
    parser.add_argument("--display", action="store_true", help="Display output in real-time")
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


def main():
    args = parse_args()
    logger = setup_logging(args.debug)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize components
    logger.info("Initializing system components...")
    
    # Initialize video reader
    if args.input:
        video_reader = VideoReader(args.input)
    else:
        logger.info("No input video specified, using camera")
        video_reader = VideoReader(0)  # Use default camera
    
    # Initialize GPS reader if GPS data is provided
    gps_reader = None
    if args.gps:
        gps_reader = GPSReader(args.gps)
    
    # Initialize detector
    detector = TrafficObjectDetector(model_path=args.model)
    
    # Initialize localizer
    localizer = ObjectLocalizer()
    
    # Initialize mapper
    mapper = Mapper()
    
    # Initialize relevance estimator
    relevance_estimator = RelevanceEstimator()
    
    # Initialize visualizer
    visualizer = Visualizer(output_dir=str(output_dir))
    
    logger.info("System initialized successfully")
    
    try:
        frame_count = 0
        processing_times = []
        
        while True:
            # Read frame
            ret, frame = video_reader.read()
            if not ret:
                logger.info("End of video")
                break
            
            # Skip frames if specified
            if args.skip_frames > 0 and frame_count % (args.skip_frames + 1) != 0:
                frame_count += 1
                continue
            
            # Read GPS data if available
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
            
            # Map objects
            mapped_objects = mapper.update(locations, frame_count)
            
            # Visualize results
            output_frame = visualizer.visualize(
                frame, detections, locations, relevance, mapped_objects
            )
            
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
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
            # Save output frame
            output_path = output_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(output_path), output_frame)
            
            frame_count += 1
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Error occurred: {e}")
    finally:
        # Clean up
        video_reader.release()
        if args.display:
            cv2.destroyAllWindows()
        logger.info("Processing complete")


if __name__ == "__main__":
    main()