"""
PyTest configuration and fixtures for TrafficVision AI tests.

Provides common fixtures, M1 compatibility checks, and performance
benchmarking configuration.
"""

from __future__ import annotations

import platform

import numpy as np
import pytest
import torch

from src.detection.detector import Detection


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "m1_required: mark test as requiring M1/MPS hardware support"
    )
    config.addinivalue_line("markers", "performance: mark test as a performance benchmark")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on platform and availability."""
    skip_m1 = pytest.mark.skip(reason="MPS not available - skipping M1 tests")

    # Check if MPS is available
    mps_available = torch.backends.mps.is_available()

    for item in items:
        if "m1_required" in item.keywords and not mps_available:
            item.add_marker(skip_m1)


@pytest.fixture(scope="session")
def platform_info():
    """Provide platform information for tests."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "is_apple_silicon": platform.processor() == "arm",
        "mps_available": torch.backends.mps.is_available(),
        "pytorch_version": torch.__version__,
    }


@pytest.fixture
def sample_detection():
    """Create a single sample detection for testing."""
    return Detection(
        bbox=[100, 100, 200, 200],
        confidence=0.9,
        class_name="traffic_light",
        class_id=0,
        features=np.random.rand(128).astype(np.float32),
    )


@pytest.fixture
def sample_detections():
    """Create multiple sample detections for testing."""
    detections = []

    configs = [
        ([100, 100, 200, 200], 0.9, "traffic_light", 0),
        ([300, 150, 400, 250], 0.85, "speed_camera", 1),
        ([500, 200, 600, 300], 0.8, "traffic_light", 0),
        ([700, 250, 800, 350], 0.75, "sign", 2),
    ]

    for bbox, confidence, class_name, class_id in configs:
        detection = Detection(
            bbox=bbox,
            confidence=confidence,
            class_name=class_name,
            class_id=class_id,
            features=np.random.rand(128).astype(np.float32),
        )
        detections.append(detection)

    return detections


@pytest.fixture
def moving_detections():
    """Create detections that simulate movement across frames."""

    def _create_moving_detections(frame_count: int = 10, movement_step: int = 5):
        frame_detections = []

        base_detection = Detection(
            bbox=[100, 100, 200, 200],
            confidence=0.9,
            class_name="traffic_light",
            class_id=0,
            features=np.random.rand(128).astype(np.float32),
        )

        for frame_idx in range(frame_count):
            moved_bbox = [
                base_detection.bbox[0] + frame_idx * movement_step,
                base_detection.bbox[1] + frame_idx * movement_step // 2,
                base_detection.bbox[2] + frame_idx * movement_step,
                base_detection.bbox[3] + frame_idx * movement_step // 2,
            ]

            moved_detection = Detection(
                bbox=moved_bbox,
                confidence=base_detection.confidence,
                class_name=base_detection.class_name,
                class_id=base_detection.class_id,
                features=base_detection.features,
            )

            frame_detections.append([moved_detection])

        return frame_detections

    return _create_moving_detections


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "min_rounds": 10,
        "max_time": 30.0,
        "warmup_rounds": 3,
        "fps_targets": {"4k": 60.0, "1080p": 120.0, "general": 30.0},
        "memory_limits": {"max_increase_mb": 100.0, "baseline_tolerance_mb": 50.0},
    }


@pytest.fixture
def mps_device():
    """Get MPS device if available."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        pytest.skip("MPS device not available")


@pytest.fixture
def mock_coreml_model():
    """Mock CoreML model for testing without actual model files."""

    class MockCoreMLModel:
        def predict(self, data):
            # Return mock predictions
            batch_size = data.shape[0] if hasattr(data, "shape") else 1
            return {
                "features": np.random.rand(batch_size, 128).astype(np.float32),
                "confidence": np.random.rand(batch_size).astype(np.float32) * 0.3 + 0.7,
            }

    return MockCoreMLModel()


# Benchmark-specific configuration
@pytest.fixture(scope="function")
def benchmark(benchmark):
    """Configure benchmark fixture with M1-optimized settings."""
    # Configure benchmark parameters for M1 testing
    benchmark.pedantic(rounds=10, iterations=1, warmup_rounds=3)
    return benchmark


# Skip markers for missing dependencies
def pytest_runtest_setup(item):
    """Setup test requirements and skip if dependencies missing."""
    # Check for pytest-benchmark if performance tests
    if "performance" in item.keywords:
        pytest.importorskip(
            "pytest_benchmark", reason="pytest-benchmark required for performance tests"
        )

    # Check for specific M1 requirements
    if "m1_required" in item.keywords:
        if not torch.backends.mps.is_available():
            pytest.skip("M1/MPS support required but not available")


# Custom assertions for performance testing
class PerformanceAssertions:
    """Custom assertions for performance testing."""

    @staticmethod
    def assert_fps_target(actual_fps: float, target_fps: float, tolerance: float = 0.1):
        """Assert FPS meets target with tolerance."""
        min_acceptable = target_fps * (1 - tolerance)
        assert actual_fps >= min_acceptable, (
            f"FPS {actual_fps:.1f} below target {target_fps:.1f} "
            f"(min acceptable: {min_acceptable:.1f})"
        )

    @staticmethod
    def assert_memory_limit(memory_mb: float, limit_mb: float):
        """Assert memory usage is within limits."""
        assert memory_mb <= limit_mb, (
            f"Memory usage {memory_mb:.1f} MB exceeds limit {limit_mb:.1f} MB"
        )

    @staticmethod
    def assert_latency_target(latency_ms: float, target_ms: float):
        """Assert latency meets target."""
        assert latency_ms <= target_ms, (
            f"Latency {latency_ms:.1f} ms exceeds target {target_ms:.1f} ms"
        )


@pytest.fixture
def performance_assertions():
    """Provide performance assertion utilities."""
    return PerformanceAssertions()
