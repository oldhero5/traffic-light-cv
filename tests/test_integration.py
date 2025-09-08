"""
Integration tests for the traffic perception system.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import yaml

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.detector import TrafficObjectDetector
from src.localization.localizer import ObjectLocalizer
from src.mapping.mapper import Mapper
from src.relevance.relevance_estimator import RelevanceEstimator
from src.utils.video_reader import VideoReader
from src.utils.visualizer import Visualizer


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the traffic perception system."""

    def setUp(self):
        """Set up test case."""
        # Create a test image
        self.test_image = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Draw a red traffic light in the image
        cv2.rectangle(self.test_image, (600, 300), (680, 450), (0, 0, 255), -1)
        cv2.rectangle(self.test_image, (600, 300), (680, 450), (255, 255, 255), 2)

        # Create a temporary folder for output
        self.output_dir = Path("/tmp/traffic_test")
        self.output_dir.mkdir(exist_ok=True)

        # Mock video reader
        self.video_reader_patcher = patch("src.utils.video_reader.cv2.VideoCapture")
        mock_video_capture = self.video_reader_patcher.start()
        mock_video_capture.return_value.read.return_value = (True, self.test_image)
        mock_video_capture.return_value.isOpened.return_value = True
        mock_video_capture.return_value.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 100,
        }.get(prop, 0)

        # Load config
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                "detection": {"confidence_threshold": 0.25},
                "camera": {"fov_horizontal": 60.0, "fov_vertical": 35.0},
                "localization": {"use_kalman": False},
                "mapping": {"distance_threshold": 10.0},
                "visualization": {
                    "show_distance": True,
                    "show_relevance": True,
                    "show_state": True,
                },
            }

        # Create components
        self.detector_patcher = patch("src.detection.detector.YOLO")
        mock_yolo = self.detector_patcher.start()

        # Mock YOLO model to return detections
        boxes = MagicMock()
        boxes.data = np.array(
            [
                [600, 300, 680, 450, 0.95, 0]  # x1, y1, x2, y2, conf, cls_id
            ]
        )

        result = MagicMock()
        result.boxes = boxes
        result.names = {0: "traffic_light_red"}

        mock_yolo.return_value.return_value = [result]

        # Initialize system components
        self.video_reader = VideoReader("test.mp4")
        self.detector = TrafficObjectDetector()
        self.localizer = ObjectLocalizer()
        self.mapper = Mapper()
        self.relevance_estimator = RelevanceEstimator()
        self.visualizer = Visualizer(str(self.output_dir))

    def tearDown(self):
        """Clean up after test."""
        self.video_reader_patcher.stop()
        self.detector_patcher.stop()

    def test_system_pipeline(self):
        """Test the full system pipeline."""
        # Read frame
        ret, frame = self.video_reader.read()
        self.assertTrue(ret)

        # Detect objects
        detections = self.detector.detect(frame)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "traffic_light_red")

        # Localize objects
        locations = self.localizer.localize(detections, frame, None)
        self.assertEqual(len(locations), 1)

        # Determine relevance
        relevance = self.relevance_estimator.estimate(detections, frame, None)
        self.assertEqual(len(relevance), 1)

        # Update map
        mapped_objects = self.mapper.update(locations, 1)
        self.assertEqual(len(mapped_objects), 1)

        # Visualize results
        output_frame = self.visualizer.visualize(
            frame, detections, locations, relevance, mapped_objects
        )
        self.assertEqual(output_frame.shape, frame.shape)

        # Save frame
        success = self.visualizer.save_frame(output_frame, "test_output.jpg")
        self.assertTrue(success)

        # Check that file was created
        output_path = self.output_dir / "test_output.jpg"
        self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
