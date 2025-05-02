"""
Unit tests for the localizer module.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.detector import Detection
from src.localization.localizer import ObjectLocalizer, Position, CameraCalibration


class TestLocalizer(unittest.TestCase):
    """Tests for the ObjectLocalizer class."""
    
    def setUp(self):
        """Set up test case."""
        # Create a camera calibration with known parameters
        self.calibration = CameraCalibration(
            image_width=1920,
            image_height=1080,
            fov_horizontal=60.0,
            fov_vertical=35.0
        )
        
        # Create a localizer
        self.localizer = ObjectLocalizer(
            calibration=self.calibration,
            use_kalman=False  # Disable Kalman filtering for testing
        )
        
        # Create a test image
        self.test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Create a list of test detections
        self.detections = [
            Detection(
                class_id=0,
                class_name='traffic_light_red',
                confidence=0.95,
                bbox=(900, 400, 1020, 600),
                state='red'
            ),
            Detection(
                class_id=4,
                class_name='traffic_camera',
                confidence=0.85,
                bbox=(500, 300, 600, 400)
            ),
        ]
        
        # Create GPS data
        self.gps_data = {
            'latitude': 37.7749,
            'longitude': -122.4194,
            'heading': 90.0,  # East
            'altitude': 10.0,
            'speed': 10.0,
        }
    
    def test_initialization(self):
        """Test localizer initialization."""
        self.assertIsNotNone(self.localizer.calibration)
        self.assertFalse(self.localizer.use_kalman)
    
    def test_localize(self):
        """Test localization functionality."""
        localized = self.localizer.localize(self.detections, self.test_image, self.gps_data)
        
        # Check that we got the right number of results
        self.assertEqual(len(localized), len(self.detections))
        
        # Check that each result has a detection and position
        for result in localized:
            self.assertIn('detection', result)
            self.assertIn('position', result)
            
            # Check that position has distance and bearing
            position = result['position']
            self.assertIsNotNone(position.distance)
            self.assertIsNotNone(position.relative_bearing)
    
    def test_estimate_distance_by_size(self):
        """Test distance estimation by object size."""
        # Get the traffic light detection
        traffic_light = self.detections[0]
        
        # Known height of traffic light
        known_height = 0.8  # meters
        
        # Calculate distance
        distance = self.localizer._estimate_distance_by_size(
            traffic_light, known_height, self.test_image
        )
        
        # Distance should be positive
        self.assertGreater(distance, 0.0)
        
        # Test with different pixel heights
        # Higher pixel height should give smaller distance
        
        # Create a larger traffic light (closer)
        close_tl = Detection(
            class_id=0,
            class_name='traffic_light_red',
            confidence=0.95,
            bbox=(900, 300, 1020, 700),  # Taller
            state='red'
        )
        
        # Create a smaller traffic light (farther)
        far_tl = Detection(
            class_id=0,
            class_name='traffic_light_red',
            confidence=0.95,
            bbox=(900, 450, 1020, 550),  # Shorter
            state='red'
        )
        
        close_distance = self.localizer._estimate_distance_by_size(
            close_tl, known_height, self.test_image
        )
        
        far_distance = self.localizer._estimate_distance_by_size(
            far_tl, known_height, self.test_image
        )
        
        # The closer object should have a smaller distance
        self.assertLess(close_distance, far_distance)
    
    def test_calculate_world_position(self):
        """Test calculation of world positions."""
        # Test data
        distance = 100.0  # meters
        bearing = 45.0  # degrees
        
        # Calculate world position
        position = self.localizer._calculate_world_position(
            distance, bearing, self.gps_data
        )
        
        # Check that latitude and longitude are calculated
        self.assertIn('latitude', position)
        self.assertIn('longitude', position)
        
        # Check that the position is different from the original
        self.assertNotEqual(position['latitude'], self.gps_data['latitude'])
        self.assertNotEqual(position['longitude'], self.gps_data['longitude'])
        
        # Test with zero distance (should return original position)
        position_zero = self.localizer._calculate_world_position(
            0.0, bearing, self.gps_data
        )
        
        # Should be very close to original position
        self.assertAlmostEqual(
            position_zero['latitude'], self.gps_data['latitude'], places=6
        )
        self.assertAlmostEqual(
            position_zero['longitude'], self.gps_data['longitude'], places=6
        )


if __name__ == '__main__':
    unittest.main()