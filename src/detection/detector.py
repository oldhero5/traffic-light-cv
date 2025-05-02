"""
Traffic light and camera detector module.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.utils.constants import CONFIDENCE_THRESHOLD, TRAFFIC_LIGHT_CLASSES

logger = logging.getLogger(__name__)


class Detection:
    """Class representing a detection."""
    
    def __init__(
        self,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox: Tuple[float, float, float, float],
        state: Optional[str] = None,
    ):
        """
        Initialize a detection.
        
        Args:
            class_id: ID of the detected class
            class_name: Name of the detected class
            confidence: Detection confidence score
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            state: State of traffic light (red, yellow, green) if applicable
        """
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.state = state
        
        # These will be populated by subsequent processing steps
        self.distance = None
        self.world_position = None
        self.relevance_score = None
        self.tracking_id = None


class TrafficObjectDetector:
    """Detector for traffic lights and cameras."""
    
    def __init__(
        self,
        model_path: str = "models/traffic_detector.pt",
        device: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize the traffic object detector.
        
        Args:
            model_path: Path to the YOLOv8 model
            device: Device to run inference on ('cuda', 'cpu')
            confidence_threshold: Minimum confidence threshold for detections
        """
        self.logger = logging.getLogger(__name__)
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        self._load_model()
        self.logger.info(f"Detector initialized on {self.device}")
    
    def _load_model(self):
        """Load the detection model."""
        try:
            # Check if model exists
            if not os.path.exists(self.model_path):
                self.logger.warning(f"Model not found at {self.model_path}, using default YOLO model")
                self.model = YOLO("yolov8s.pt")
            else:
                self.model = YOLO(self.model_path)
            
            # Move model to device
            self.model.to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect traffic lights and cameras in a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detections
        """
        try:
            # Perform inference
            results = self.model(frame, verbose=False)[0]  # Unpack the first (and only) result
            
            # Process detections
            detections = []
            for det in results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls_id = det
                
                # Skip if confidence is below threshold
                if conf < self.confidence_threshold:
                    continue
                
                # Get class name
                class_name = results.names[int(cls_id)]
                
                # Create detection object
                detection = Detection(
                    class_id=int(cls_id),
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
                
                # If it's a traffic light, determine its state
                if "traffic_light" in class_name:
                    detection.state = self._determine_traffic_light_state(frame, detection)
                
                detections.append(detection)
            
            return detections
        
        except Exception as e:
            self.logger.error(f"Error in detection: {e}")
            return []
    
    def _determine_traffic_light_state(self, frame: np.ndarray, detection: Detection) -> str:
        """
        Determine the state of a traffic light.
        
        Args:
            frame: Input image frame
            detection: Traffic light detection
            
        Returns:
            State of the traffic light (red, yellow, green, or unknown)
        """
        # Extract traffic light region
        x1, y1, x2, y2 = [int(c) for c in detection.bbox]
        light_roi = frame[y1:y2, x1:x2]
        
        if light_roi.size == 0:
            return "unknown"
        
        # Convert to HSV for better color discrimination
        hsv = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)
        
        # Define color masks
        # Red has two ranges in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        
        # Create masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Count pixels
        red_count = np.sum(mask_red > 0)
        yellow_count = np.sum(mask_yellow > 0)
        green_count = np.sum(mask_green > 0)
        
        # Determine state based on color with most pixels
        max_count = max(red_count, yellow_count, green_count)
        
        if max_count < 10:  # Not enough color pixels detected
            return "unknown"
        elif red_count == max_count:
            return "red"
        elif yellow_count == max_count:
            return "yellow"
        elif green_count == max_count:
            return "green"
        else:
            return "unknown"