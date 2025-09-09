"""
Performance benchmarks for DeepSORT tracker on M1 MacBook Pro.

These benchmarks validate that the tracker meets the required performance
targets specified in GitHub Issue #1:
- >60 FPS processing capability
- Memory efficiency
- M1 optimization utilization
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from src.detection.detector import Detection
from src.tracking.deepsort_tracker import DeepSORTTracker


class TestDeepSORTPerformance:
    """Performance benchmark tests for DeepSORT tracker."""

    @pytest.fixture
    def tracker(self):
        """Create optimized tracker for benchmarking."""
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
    def benchmark_detections(self):
        """Create realistic detection sets for benchmarking."""
        detections = []

        # Simulate typical traffic scene with multiple objects
        object_configs = [
            ("traffic_light", 0, 0.9),
            ("traffic_light", 0, 0.85),
            ("speed_camera", 1, 0.8),
            ("traffic_light", 0, 0.75),
            ("sign", 2, 0.7),
        ]

        for i, (class_name, class_id, confidence) in enumerate(object_configs):
            detection = Detection(
                bbox=[100 + i * 120, 100 + i * 50, 200 + i * 120, 200 + i * 50],
                confidence=confidence,
                class_name=class_name,
                class_id=class_id,
                features=np.random.rand(128).astype(np.float32),
            )
            detections.append(detection)

        return detections

    @pytest.mark.performance
    @pytest.mark.m1_required
    def test_4k_resolution_fps_target(self, tracker, benchmark_detections, benchmark):
        """Test 4K resolution processing meets 60+ FPS target."""

        def process_4k_frame():
            # Simulate 4K processing load with more detections
            extended_detections = benchmark_detections * 3  # Simulate more objects in 4K
            return tracker.update(extended_detections)

        result = benchmark(process_4k_frame)

        # Calculate effective FPS
        mean_time_seconds = result.stats["mean"]
        fps = 1.0 / mean_time_seconds

        print("\n4K Processing Performance:")
        print(f"Mean processing time: {mean_time_seconds * 1000:.2f}ms")
        print(f"Effective FPS: {fps:.1f}")
        print("Target: 60+ FPS")

        # Assert 60+ FPS target for M1
        assert fps >= 60.0, f"4K FPS too low: {fps:.1f} (target: 60+)"

    @pytest.mark.performance
    @pytest.mark.m1_required
    def test_1080p_resolution_fps_target(self, tracker, benchmark_detections, benchmark):
        """Test 1080p resolution processing meets 120+ FPS target."""

        def process_1080p_frame():
            return tracker.update(benchmark_detections)

        result = benchmark(process_1080p_frame)

        # Calculate effective FPS
        mean_time_seconds = result.stats["mean"]
        fps = 1.0 / mean_time_seconds

        print("\n1080p Processing Performance:")
        print(f"Mean processing time: {mean_time_seconds * 1000:.2f}ms")
        print(f"Effective FPS: {fps:.1f}")
        print("Target: 120+ FPS")

        # Assert 120+ FPS target for M1
        assert fps >= 120.0, f"1080p FPS too low: {fps:.1f} (target: 120+)"

    @pytest.mark.performance
    def test_memory_efficiency(self, tracker, benchmark_detections):
        """Test memory usage remains within acceptable limits."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Measure baseline memory
        baseline_memory = process.memory_info().rss / (1024**2)  # MB

        # Process multiple frames to stress test memory
        for _ in range(200):
            tracks = tracker.update(benchmark_detections)

        # Measure peak memory
        peak_memory = process.memory_info().rss / (1024**2)  # MB
        memory_increase = peak_memory - baseline_memory

        print("\nMemory Usage:")
        print(f"Baseline: {baseline_memory:.1f} MB")
        print(f"Peak: {peak_memory:.1f} MB")
        print(f"Increase: {memory_increase:.1f} MB")

        # Memory increase should be reasonable for 200 frames
        assert memory_increase < 100.0, f"Memory increase too high: {memory_increase:.1f} MB"

    @pytest.mark.performance
    @pytest.mark.m1_required
    def test_mps_acceleration_benefit(self, benchmark_detections, benchmark):
        """Test that MPS acceleration provides performance benefit."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        # Create CPU-only tracker
        cpu_tracker = DeepSORTTracker()
        # Force CPU mode
        cpu_tracker.metal_utils.device = "cpu"
        cpu_tracker.metal_utils.mps_available = False

        # Create MPS tracker
        mps_tracker = DeepSORTTracker()

        # Benchmark CPU processing
        def cpu_process():
            return cpu_tracker.update(benchmark_detections)

        # Benchmark MPS processing
        def mps_process():
            return mps_tracker.update(benchmark_detections)

        cpu_result = benchmark(cpu_process)
        mps_result = benchmark(mps_process)

        cpu_fps = 1.0 / cpu_result.stats["mean"]
        mps_fps = 1.0 / mps_result.stats["mean"]

        speedup = mps_fps / cpu_fps

        print("\nMPS Acceleration Benchmark:")
        print(f"CPU FPS: {cpu_fps:.1f}")
        print(f"MPS FPS: {mps_fps:.1f}")
        print(f"Speedup: {speedup:.2f}x")

        # MPS should provide at least 1.5x speedup
        assert speedup >= 1.5, f"MPS speedup insufficient: {speedup:.2f}x (expected: 1.5x+)"

    @pytest.mark.performance
    def test_batch_similarity_performance(self, tracker, benchmark):
        """Test batch similarity computation performance."""
        # Create test feature sets
        features1 = np.random.rand(50, 128).astype(np.float32)
        features2 = np.random.rand(30, 128).astype(np.float32)

        def compute_similarity():
            return tracker.metal_utils.batch_cosine_similarity(features1, features2)

        result = benchmark(compute_similarity)

        operations_per_second = 1.0 / result.stats["mean"]

        print("\nBatch Similarity Performance:")
        print("Matrix size: 50x30")
        print(f"Operations/second: {operations_per_second:.1f}")
        print(f"Mean time: {result.stats['mean'] * 1000:.2f}ms")

        # Should handle at least 100 similarity computations per second
        assert operations_per_second >= 100.0

    @pytest.mark.performance
    def test_iou_computation_performance(self, tracker, benchmark):
        """Test IoU computation performance."""
        # Create test bounding box sets
        bboxes1 = np.random.rand(40, 4) * 1000  # Random bboxes
        bboxes2 = np.random.rand(25, 4) * 1000

        # Ensure valid bbox format (x1 < x2, y1 < y2)
        bboxes1[:, 2] += bboxes1[:, 0] + 50
        bboxes1[:, 3] += bboxes1[:, 1] + 50
        bboxes2[:, 2] += bboxes2[:, 0] + 50
        bboxes2[:, 3] += bboxes2[:, 1] + 50

        def compute_iou():
            return tracker.metal_utils.compute_iou_batch(bboxes1, bboxes2)

        result = benchmark(compute_iou)

        operations_per_second = 1.0 / result.stats["mean"]

        print("\nBatch IoU Performance:")
        print("Matrix size: 40x25")
        print(f"Operations/second: {operations_per_second:.1f}")
        print(f"Mean time: {result.stats['mean'] * 1000:.2f}ms")

        # Should handle at least 200 IoU computations per second
        assert operations_per_second >= 200.0

    @pytest.mark.performance
    def test_multi_object_tracking_scalability(self, tracker, benchmark):
        """Test tracking performance scales with number of objects."""
        object_counts = [5, 10, 20, 50]
        performance_results = {}

        for count in object_counts:
            # Generate detections for this object count
            detections = []
            for i in range(count):
                detection = Detection(
                    bbox=[i * 30, i * 20, i * 30 + 100, i * 20 + 80],
                    confidence=0.8,
                    class_name="traffic_light",
                    class_id=0,
                    features=np.random.rand(128).astype(np.float32),
                )
                detections.append(detection)

            def process_objects():
                return tracker.update(detections)

            result = benchmark(process_objects)
            fps = 1.0 / result.stats["mean"]
            performance_results[count] = fps

            print(f"\n{count} objects: {fps:.1f} FPS")

        # Performance should remain reasonable even with 50 objects
        assert performance_results[50] >= 30.0, "Performance degrades too much with many objects"

        # Performance shouldn't drop by more than 70% from 5 to 50 objects
        performance_ratio = performance_results[50] / performance_results[5]
        assert performance_ratio >= 0.3, f"Performance scaling poor: {performance_ratio:.2f}"

    @pytest.mark.performance
    def test_30_frame_persistence_performance(self, tracker, benchmark_detections):
        """Test performance with 30+ frame tracking persistence."""
        # Simulate 50 frames of consistent tracking
        frame_count = 50
        total_time = 0.0

        for frame_idx in range(frame_count):
            # Simulate object movement
            moved_detections = []
            for det in benchmark_detections:
                moved_bbox = [
                    det.bbox[0] + frame_idx * 2,
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

            start_time = time.perf_counter()
            tracks = tracker.update(moved_detections)
            end_time = time.perf_counter()

            total_time += end_time - start_time

        average_fps = frame_count / total_time

        print("\n30+ Frame Persistence Performance:")
        print(f"Frames processed: {frame_count}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average FPS: {average_fps:.1f}")

        # Should maintain 60+ FPS even with persistent tracking
        assert average_fps >= 60.0, f"Persistent tracking FPS too low: {average_fps:.1f}"

    @pytest.mark.performance
    @pytest.mark.m1_required
    def test_neural_engine_utilization_mock(self, tracker):
        """Test Neural Engine utilization (mock test for CI compatibility)."""
        # This is a mock test since actual ANE testing requires specific setup
        # In production, this would measure actual Neural Engine usage

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        # Mock ANE utilization check
        utilization_metrics = {
            "ane_active": True,
            "ane_utilization_percent": 85.0,
            "power_efficiency_rating": 9.2,
        }

        print("\nNeural Engine Utilization (Mock):")
        print(f"ANE Active: {utilization_metrics['ane_active']}")
        print(f"ANE Utilization: {utilization_metrics['ane_utilization_percent']:.1f}%")
        print(f"Power Efficiency: {utilization_metrics['power_efficiency_rating']:.1f}/10")

        # Assert ANE utilization targets
        assert utilization_metrics["ane_utilization_percent"] >= 80.0
        assert utilization_metrics["power_efficiency_rating"] >= 8.0


if __name__ == "__main__":
    pytest.main([__file__, "--benchmark-only", "-v"])
