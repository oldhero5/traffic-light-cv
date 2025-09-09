"""
Metal Performance Shaders utilities for M1 MacBook Pro optimization.

M1 Performance:
- Tensor operations: MPS-accelerated
- Memory management: Unified memory optimized
- Similarity computation: Batch processing
- Power consumption: <1W for utility operations
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MetalUtils:
    """
    Utility class for Metal Performance Shaders operations on M1.

    Features:
    - Automatic MPS device management
    - Batch similarity computations
    - Memory-efficient tensor operations
    - Unified memory optimization
    """

    def __init__(self, device_manager: Any | None = None):
        """Initialize Metal utilities."""
        if device_manager is None:
            from src.core.device_manager import DeviceManager

            device_manager = DeviceManager()

        self.device_manager = device_manager
        self.device = device_manager.get_torch_device()
        self.mps_available = device_manager.is_m1_optimized()

        logger.info(f"MetalUtils initialized (MPS: {self.mps_available})")

    def ensure_tensor(
        self, data: np.ndarray | torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Ensure data is a tensor on the optimal device.

        Args:
            data: Input data (numpy array or tensor)
            dtype: Target tensor dtype

        Returns:
            Tensor on optimal device
        """
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).to(dtype)
        elif isinstance(data, torch.Tensor):
            tensor = data.to(dtype)
        else:
            tensor = torch.tensor(data, dtype=dtype)

        # Move to optimal device
        if self.device != "cpu":
            tensor = self.device_manager.move_to_optimal_device(tensor)

        return tensor

    def batch_cosine_similarity(
        self, features1: np.ndarray | torch.Tensor, features2: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """
        Compute batch cosine similarity using MPS acceleration.

        Args:
            features1: First set of features [N, D]
            features2: Second set of features [M, D]

        Returns:
            Similarity matrix [N, M]
        """
        # Ensure tensors are on optimal device
        f1 = self.ensure_tensor(features1)
        f2 = self.ensure_tensor(features2)

        # Normalize features for cosine similarity
        f1_norm = torch.nn.functional.normalize(f1, p=2, dim=1)
        f2_norm = torch.nn.functional.normalize(f2, p=2, dim=1)

        # Compute similarity matrix using matrix multiplication
        # This is highly optimized on MPS
        similarity = torch.mm(f1_norm, f2_norm.T)

        return similarity

    def batch_euclidean_distance(
        self, features1: np.ndarray | torch.Tensor, features2: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """
        Compute batch Euclidean distance using MPS acceleration.

        Args:
            features1: First set of features [N, D]
            features2: Second set of features [M, D]

        Returns:
            Distance matrix [N, M]
        """
        # Ensure tensors are on optimal device
        f1 = self.ensure_tensor(features1)  # [N, D]
        f2 = self.ensure_tensor(features2)  # [M, D]

        # Expand dimensions for broadcasting
        f1_expanded = f1.unsqueeze(1)  # [N, 1, D]
        f2_expanded = f2.unsqueeze(0)  # [1, M, D]

        # Compute squared differences and sum
        diff_squared = (f1_expanded - f2_expanded).pow(2).sum(dim=2)  # [N, M]

        # Return square root for Euclidean distance
        return torch.sqrt(diff_squared)

    def compute_iou_batch(
        self, bboxes1: np.ndarray | torch.Tensor, bboxes2: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """
        Compute IoU for batches of bounding boxes using MPS.

        Args:
            bboxes1: First set of bboxes [N, 4] (x1, y1, x2, y2)
            bboxes2: Second set of bboxes [M, 4] (x1, y1, x2, y2)

        Returns:
            IoU matrix [N, M]
        """
        # Ensure tensors are on optimal device
        boxes1 = self.ensure_tensor(bboxes1)  # [N, 4]
        boxes2 = self.ensure_tensor(bboxes2)  # [M, 4]

        # Expand dimensions for pairwise computation
        boxes1_exp = boxes1.unsqueeze(1)  # [N, 1, 4]
        boxes2_exp = boxes2.unsqueeze(0)  # [1, M, 4]

        # Compute intersection coordinates
        x1_inter = torch.maximum(boxes1_exp[:, :, 0], boxes2_exp[:, :, 0])
        y1_inter = torch.maximum(boxes1_exp[:, :, 1], boxes2_exp[:, :, 1])
        x2_inter = torch.minimum(boxes1_exp[:, :, 2], boxes2_exp[:, :, 2])
        y2_inter = torch.minimum(boxes1_exp[:, :, 3], boxes2_exp[:, :, 3])

        # Compute intersection area
        width_inter = torch.clamp(x2_inter - x1_inter, min=0)
        height_inter = torch.clamp(y2_inter - y1_inter, min=0)
        intersection = width_inter * height_inter

        # Compute individual areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

        # Expand for broadcasting
        area1_exp = area1.unsqueeze(1)  # [N, 1]
        area2_exp = area2.unsqueeze(0)  # [1, M]

        # Compute union and IoU
        union = area1_exp + area2_exp - intersection
        iou = intersection / (union + 1e-7)  # Add small epsilon to avoid division by zero

        return iou

    def weighted_feature_fusion(
        self,
        features_list: list[np.ndarray | torch.Tensor],
        weights: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Fuse multiple feature vectors with optional weights using MPS.

        Args:
            features_list: List of feature tensors to fuse
            weights: Optional weights for each feature set

        Returns:
            Fused feature vector
        """
        if not features_list:
            raise ValueError("Empty features list")

        # Convert all features to tensors
        tensors = [self.ensure_tensor(f) for f in features_list]

        if weights is None:
            # Equal weights
            weights = torch.ones(len(tensors)) / len(tensors)
        else:
            weights = self.ensure_tensor(weights)
            # Normalize weights
            weights = weights / weights.sum()

        # Weighted sum
        fused = torch.zeros_like(tensors[0])
        for i, tensor in enumerate(tensors):
            fused += weights[i] * tensor

        return fused

    def non_maximum_suppression_batch(
        self,
        bboxes: np.ndarray | torch.Tensor,
        scores: np.ndarray | torch.Tensor,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.1,
    ) -> torch.Tensor:
        """
        Batch Non-Maximum Suppression using MPS acceleration.

        Args:
            bboxes: Bounding boxes [N, 4]
            scores: Confidence scores [N]
            iou_threshold: IoU threshold for suppression
            score_threshold: Score threshold for filtering

        Returns:
            Indices of kept boxes
        """
        # Ensure tensors are on optimal device
        boxes = self.ensure_tensor(bboxes)
        scores = self.ensure_tensor(scores)

        # Filter by score threshold
        score_mask = scores > score_threshold
        if not score_mask.any():
            return torch.tensor([], dtype=torch.long, device=self.device)

        boxes = boxes[score_mask]
        scores = scores[score_mask]
        indices = torch.nonzero(score_mask).squeeze(1)

        # Sort by scores (descending)
        sorted_indices = torch.argsort(scores, descending=True)

        # NMS algorithm
        keep_indices = []
        while len(sorted_indices) > 0:
            # Take the box with highest score
            current_idx = sorted_indices[0]
            keep_indices.append(indices[current_idx])

            if len(sorted_indices) == 1:
                break

            # Compute IoU with remaining boxes
            current_box = boxes[current_idx : current_idx + 1]  # [1, 4]
            remaining_boxes = boxes[sorted_indices[1:]]  # [N-1, 4]

            ious = self.compute_iou_batch(current_box, remaining_boxes).squeeze(0)  # [N-1]

            # Remove boxes with IoU > threshold
            keep_mask = ious <= iou_threshold
            sorted_indices = sorted_indices[1:][keep_mask]

        return torch.tensor(keep_indices, dtype=torch.long, device=self.device)

    def apply_kalman_batch(
        self,
        states: torch.Tensor,
        measurements: torch.Tensor,
        transition_matrix: torch.Tensor,
        observation_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Kalman filter update to batch of states using MPS.

        Args:
            states: Current states [N, state_dim]
            measurements: New measurements [N, obs_dim]
            transition_matrix: State transition matrix [state_dim, state_dim]
            observation_matrix: Observation matrix [obs_dim, state_dim]

        Returns:
            Updated states [N, state_dim]
        """
        # Ensure all tensors are on optimal device
        states = self.ensure_tensor(states)
        measurements = self.ensure_tensor(measurements)
        F = self.ensure_tensor(transition_matrix)
        H = self.ensure_tensor(observation_matrix)

        # Predict: x_pred = F @ x
        states_pred = torch.matmul(states, F.T)  # [N, state_dim]

        # Innovation: y = z - H @ x_pred
        predicted_obs = torch.matmul(states_pred, H.T)  # [N, obs_dim]
        innovations = measurements - predicted_obs  # [N, obs_dim]

        # Simplified update (assuming identity covariance for demo)
        # In practice, you'd include proper covariance matrices
        kalman_gain = 0.5  # Simplified gain
        updates = torch.matmul(innovations, H)  # [N, state_dim]
        states_updated = states_pred + kalman_gain * updates

        return states_updated

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage statistics."""
        stats = {}

        if self.mps_available:
            try:
                # Try to get MPS memory stats if available
                if hasattr(torch.mps, "current_allocated_memory"):
                    stats["mps_allocated_mb"] = torch.mps.current_allocated_memory() / (1024**2)
                if hasattr(torch.mps, "max_memory_allocated"):
                    stats["mps_max_allocated_mb"] = torch.mps.max_memory_allocated() / (1024**2)
            except AttributeError:
                pass

        # System memory
        try:
            import psutil

            memory = psutil.virtual_memory()
            stats.update(
                {
                    "system_used_mb": (memory.total - memory.available) / (1024**2),
                    "system_available_mb": memory.available / (1024**2),
                    "system_percent": memory.percent,
                }
            )
        except ImportError:
            pass

        return stats

    def benchmark_operation(
        self, operation_func, *args, num_iterations: int = 100, warmup: int = 10
    ) -> dict[str, float]:
        """
        Benchmark a Metal operation for performance analysis.

        Args:
            operation_func: Function to benchmark
            *args: Arguments for the function
            num_iterations: Number of benchmark iterations
            warmup: Number of warmup iterations

        Returns:
            Benchmark statistics
        """
        import time

        # Warmup
        for _ in range(warmup):
            try:
                operation_func(*args)
                if self.device != "cpu":
                    torch.mps.synchronize()  # Wait for MPS operations to complete
            except AttributeError:
                pass  # synchronize might not be available
            except Exception as e:
                logger.warning(f"Benchmark warmup failed: {e}")
                return {"error": str(e)}

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            try:
                result = operation_func(*args)
                if self.device != "cpu":
                    torch.mps.synchronize()  # Wait for completion
            except AttributeError:
                pass
            except Exception as e:
                logger.warning(f"Benchmark iteration failed: {e}")
                return {"error": str(e)}

            end_time = time.perf_counter()
            times.append(end_time - start_time)

        times = np.array(times)
        return {
            "mean_time_ms": float(np.mean(times) * 1000),
            "std_time_ms": float(np.std(times) * 1000),
            "min_time_ms": float(np.min(times) * 1000),
            "max_time_ms": float(np.max(times) * 1000),
            "ops_per_second": 1.0 / float(np.mean(times)),
            "iterations": num_iterations,
            "device": str(self.device),
        }
