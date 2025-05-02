"""
Unit tests for the detector module.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.detector import TrafficObjectDetector, Detection


class TestDetector(unittest.TestCase):
    """Tests for the TrafficObjectDetector class."""
    
    def setUp(self):
        """Set up test case."""
        # Create a mock model
        self.mock_model = MagicMock()
        
        # Use patch to mock the YOLO model
        self.model_patcher = patch('src.detection.detector.YOLO')
        self.mock_yolo = self.model_patcher.start()
        self.mock_yolo.return_value = self.mock_model
        
        # Create dummy image
        self.test_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Create a mock detection
        boxes = MagicMock()
        boxes.data = torch.tensor([
            [100, 200, 300, 400, 0.95, 0],  # x1, y1, x2, y2, conf, cls_id
            [500, 600, 700, 800, 0.85, 1],
        ])
        
        names = {0: 'traffic_light_red', 1: 'traffic_camera'}
        
        result = MagicMock()
        result.boxes = boxes
        result.names = names
        
        # Set up mock model to return the mock result
        self.mock_model.return_value = [result]
        
        # Create detector instance with mocked model
        self.detector = TrafficObjectDetector()
    
    def tearDown(self):
        """Clean up after test."""
        self.model_patcher.stop()
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.confidence_threshold, 0.25)
        self.mock_yolo.assert_called_once()
        self.mock_model.to.assert_called_once()
    
    def test_detect(self):
        """Test detection functionality."""
        detections = self.detector.detect(self.test_image)
        
        # Check that the model was called with the test image
        self.mock_model.assert_called_with(self.test_image, verbose=False)
        
        # Check that the correct number of detections was returned
        self.assertEqual(len(detections), 2)
        
        # Check the properties of the first detection
        self.assertEqual(detections[0].class_id, 0)
        self.assertEqual(detections[0].class_name, 'traffic_light_red')
        self.assertEqual(detections[0].confidence, 0.95)
        self.assertEqual(detections[0].bbox, (100.0, 200.0, 300.0, 400.0))
        
        # Check that state was determined for traffic light
        self.assertIsNotNone(detections[0].state)
    
    def test_determine_traffic_light_state(self):
        """Test traffic light state determination."""
        # Create a red traffic light image
        red_light = np.zeros((200, 100, 3), dtype=np.uint8)
        red_light[50:100, 25:75, 2] = 255  # Red color (BGR format)
        
        # Create a detection
        detection = Detection(
            class_id=0,
            class_name='traffic_light_red',
            confidence=0.95,
            bbox=(0, 0, 100, 200)
        )
        
        # Test state determination
        state = self.detector._determine_traffic_light_state(red_light, detection)
        self.assertEqual(state, 'red')
        
        # Create a green traffic light image
        green_light = np.zeros((200, 100, 3), dtype=np.uint8)
        green_light[100:150, 25:75, 1] = 255  # Green color (BGR format)
        
        # Test state determination
        state = self.detector._determine_traffic_light_state(green_light, detection)
        self.assertEqual(state, 'green')


if __name__ == '__main__':
    unittest.main()