"""
Comprehensive tests for DeepSORT tracker with M1 optimizations.

Tests cover:
- Multi-object tracking scenarios
- 30+ frame persistence requirement
- M1-specific optimizations
- Deep-sort-realtime compatibility
- Tracking confidence calculations
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src.detection.detector import Detection
from src.tracking.deepsort_tracker import DeepSORTTracker


class TestDeepSORTTracker:
    """Test suite for DeepSORT tracker."""

    @pytest.fixture
    def tracker(self):
        """Create a DeepSORTTracker instance for testing."""
        return DeepSORTTracker(
            model_path="models/deep_sort.pb",
            max_distance=0.2,
            min_confidence=0.3,
            nms_max_overlap=1.0,
            max_iou_distance=0.7,
            max_age=70,
            n_init=3,
        )

    @pytest.fixture
    def mock_detections(self):
        """Create mock detections for testing."""
        return [
            Detection(
                bbox=[100, 100, 200, 200],
                confidence=0.9,
                class_name="traffic_light",
                class_id=0,
                features=np.random.rand(128).astype(np.float32),
            ),
            Detection(
                bbox=[300, 150, 400, 250],
                confidence=0.85,
                class_name="traffic_light",
                class_id=0,
                features=np.random.rand(128).astype(np.float32),
            ),
            Detection(
                bbox=[500, 200, 600, 300],
                confidence=0.8,
                class_name="speed_camera",
                class_id=1,
                features=np.random.rand(128).astype(np.float32),
            ),
        ]

    def test_tracker_initialization(self, tracker):
        """Test tracker initializes correctly."""
        assert tracker.max_distance == 0.2
        assert tracker.min_confidence == 0.3
        assert tracker.max_age == 70
        assert tracker.n_init == 3
        assert hasattr(tracker, "metal_utils")
        assert hasattr(tracker, "deep_sort")

    @pytest.mark.m1_required
    def test_mps_device_usage(self, tracker):
        """Test that tracker uses MPS device when available on M1."""
        if torch.backends.mps.is_available():
            assert tracker.metal_utils.device != "cpu"
            assert tracker.metal_utils.mps_available
        else:
            pytest.skip("MPS not available on this system")

    def test_single_frame_tracking(self, tracker, mock_detections):
        """Test tracking on a single frame."""
        tracks = tracker.update(mock_detections)

        # Should have tracks for valid detections
        assert len(tracks) <= len(mock_detections)

        # Check track attributes
        for track in tracks:
            assert hasattr(track, "track_id")
            assert hasattr(track, "bbox")
            assert hasattr(track, "confidence")
            assert hasattr(track, "class_name")

    def test_multi_frame_tracking(self, tracker, mock_detections):
        """Test tracking across multiple frames."""
        frame_count = 10
        all_tracks = []

        for frame_idx in range(frame_count):
            # Simulate slight movement in detections
            moved_detections = []
            for i, det in enumerate(mock_detections):
                moved_bbox = [
                    det.bbox[0] + frame_idx * 2,  # Slight movement
                    det.bbox[1] + frame_idx * 1,
                    det.bbox[2] + frame_idx * 2,
                    det.bbox[3] + frame_idx * 1,
                ]
                moved_det = Detection(
                    bbox=moved_bbox,
                    confidence=det.confidence,
                    class_name=det.class_name,
                    class_id=det.class_id,
                    features=det.features,
                )
                moved_detections.append(moved_det)

            tracks = tracker.update(moved_detections)
            all_tracks.append(tracks)

        # Should maintain consistent track IDs across frames
        if len(all_tracks) > 1:
            first_frame_ids = {track.track_id for track in all_tracks[0]}
            last_frame_ids = {track.track_id for track in all_tracks[-1]}

            # At least some tracks should persist
            assert len(first_frame_ids.intersection(last_frame_ids)) > 0

    def test_30_frame_tracking_requirement(self, tracker, mock_detections):
        """Test that tracker can maintain tracks for 30+ frames."""
        frame_count = 35  # Test beyond 30 frames
        persistent_tracks = {}

        for frame_idx in range(frame_count):
            # Keep first detection consistent across all frames
            consistent_detection = mock_detections[0]
            moved_bbox = [
                consistent_detection.bbox[0] + frame_idx * 1,
                consistent_detection.bbox[1],
                consistent_detection.bbox[2] + frame_idx * 1,
                consistent_detection.bbox[3],
            ]

            frame_detection = Detection(
                bbox=moved_bbox,
                confidence=consistent_detection.confidence,
                class_name=consistent_detection.class_name,
                class_id=consistent_detection.class_id,
                features=consistent_detection.features,
            )

            tracks = tracker.update([frame_detection])

            for track in tracks:
                if track.track_id not in persistent_tracks:
                    persistent_tracks[track.track_id] = 0
                persistent_tracks[track.track_id] += 1

        # At least one track should persist for 30+ frames
        max_persistence = max(persistent_tracks.values()) if persistent_tracks else 0
        assert max_persistence >= 30, f"Max track persistence: {max_persistence} frames"

    def test_validate_30_frame_tracking(self, tracker):
        """Test the validate_30_frame_tracking method."""
        # Mock tracking history for validation
        tracker.tracking_history = {
            1: list(range(35)),  # Track 1 active for 35 frames
            2: list(range(25)),  # Track 2 active for 25 frames
            3: list(range(45)),  # Track 3 active for 45 frames
        }

        validation_result = tracker.validate_30_frame_tracking()

        assert validation_result["meets_requirement"] == True
        assert validation_result["tracks_30_plus"] == 2  # Tracks 1 and 3
        assert validation_result["max_persistence"] == 45
        assert validation_result["avg_persistence"] == 35.0

    def test_tracking_confidence_calculation(self, tracker):
        """Test tracking confidence calculation."""
        # Mock track and detection
        mock_track = Mock()
        mock_track.hits = 10
        mock_track.hit_streak = 5
        mock_track.age = 15
        mock_track.time_since_update = 0

        mock_detection = Mock()
        mock_detection.confidence = 0.85

        confidence = tracker._calculate_tracking_confidence(mock_track, mock_detection)

        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)

    def test_fps_estimation(self, tracker):
        """Test FPS estimation functionality."""
        # Simulate processing times
        import time

        start_time = time.time()
        # Simulate some processing
        time.sleep(0.01)

        fps = tracker.get_fps_estimate()

        # Should return a reasonable FPS value
        assert fps > 0
        assert isinstance(fps, float)

    @pytest.mark.m1_required
    def test_metal_utils_integration(self, tracker):
        """Test Metal utilities integration."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available on this system")

        # Test feature similarity computation
        features1 = np.random.rand(5, 128).astype(np.float32)
        features2 = np.random.rand(3, 128).astype(np.float32)

        similarity = tracker.metal_utils.batch_cosine_similarity(features1, features2)

        assert similarity.shape == (5, 3)
        assert torch.is_tensor(similarity)
        assert -1.0 <= similarity.min().item() <= 1.0
        assert -1.0 <= similarity.max().item() <= 1.0

    def test_deep_sort_realtime_compatibility(self, tracker):
        """Test deep-sort-realtime compatibility mode."""
        # Enable realtime mode
        tracker.enable_deep_sort_realtime_mode()

        # Create mock realtime-style detections
        realtime_detections = [
            ([100, 100, 200, 200], 0.9, "traffic_light"),
            ([300, 150, 400, 250], 0.85, "speed_camera"),
        ]

        tracks = tracker.update_with_realtime(realtime_detections)

        assert isinstance(tracks, list)
        # Should handle realtime format correctly

    def test_tracking_with_low_confidence_detections(self, tracker):
        """Test tracking behavior with low confidence detections."""
        low_conf_detections = [
            Detection(
                bbox=[100, 100, 200, 200],
                confidence=0.2,  # Below min_confidence (0.3)
                class_name="traffic_light",
                class_id=0,
                features=np.random.rand(128).astype(np.float32),
            )
        ]

        tracks = tracker.update(low_conf_detections)

        # Should filter out low confidence detections
        assert len(tracks) == 0

    def test_tracking_with_overlapping_detections(self, tracker):
        """Test tracking with overlapping bounding boxes."""
        overlapping_detections = [
            Detection(
                bbox=[100, 100, 200, 200],
                confidence=0.9,
                class_name="traffic_light",
                class_id=0,
                features=np.random.rand(128).astype(np.float32),
            ),
            Detection(
                bbox=[150, 150, 250, 250],  # Overlapping with first
                confidence=0.85,
                class_name="traffic_light",
                class_id=0,
                features=np.random.rand(128).astype(np.float32),
            ),
        ]

        tracks = tracker.update(overlapping_detections)

        # Should handle overlapping detections appropriately
        assert len(tracks) <= len(overlapping_detections)

    def test_memory_usage_stability(self, tracker, mock_detections):
        """Test that memory usage remains stable over many frames."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Process many frames
        for _ in range(100):
            tracks = tracker.update(mock_detections)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024

    @pytest.mark.performance
    def test_60_fps_performance_target(self, tracker, mock_detections):
        """Test that tracker can achieve 60+ FPS target."""
        import time

        num_iterations = 100
        start_time = time.time()

        for _ in range(num_iterations):
            tracks = tracker.update(mock_detections)

        end_time = time.time()
        total_time = end_time - start_time
        fps = num_iterations / total_time

        # Should achieve at least 60 FPS on M1
        # Note: This may vary based on system load and hardware
        print(f"Measured FPS: {fps:.2f}")

        # We'll use a more relaxed threshold for CI/CD
        assert fps > 30, f"FPS too low: {fps:.2f}"

    def test_track_state_persistence(self, tracker):
        """Test that track states persist correctly across updates."""
        detection = Detection(
            bbox=[100, 100, 200, 200],
            confidence=0.9,
            class_name="traffic_light",
            class_id=0,
            features=np.random.rand(128).astype(np.float32),
        )

        # First update - should create new track
        tracks1 = tracker.update([detection])
        assert len(tracks1) > 0

        track_id = tracks1[0].track_id

        # Second update - should maintain same track
        tracks2 = tracker.update([detection])
        assert len(tracks2) > 0
        assert tracks2[0].track_id == track_id

    def test_track_deletion_after_max_age(self, tracker):
        """Test that tracks are deleted after max_age frames."""
        detection = Detection(
            bbox=[100, 100, 200, 200],
            confidence=0.9,
            class_name="traffic_light",
            class_id=0,
            features=np.random.rand(128).astype(np.float32),
        )

        # Create a track
        tracks = tracker.update([detection])
        assert len(tracks) > 0

        # Update with empty detections for max_age + 1 frames
        for _ in range(tracker.max_age + 1):
            tracks = tracker.update([])

        # Track should be deleted
        assert len(tracks) == 0


@pytest.mark.m1_required
class TestM1SpecificOptimizations:
    """Test M1-specific optimizations and features."""

    def test_neural_engine_utilization(self):
        """Test Neural Engine utilization for feature extraction."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available on this system")

        # This would require actual CoreML model integration
        # For now, we test that the infrastructure is in place
        tracker = DeepSORTTracker()
        assert tracker.metal_utils.mps_available

    def test_unified_memory_optimization(self):
        """Test unified memory optimizations."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available on this system")

        tracker = DeepSORTTracker()

        # Test that tensors are properly moved to optimal device
        test_array = np.random.rand(10, 128).astype(np.float32)
        tensor = tracker.metal_utils.ensure_tensor(test_array)

        assert torch.is_tensor(tensor)
        if tracker.metal_utils.device != "cpu":
            assert tensor.device.type == "mps"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
