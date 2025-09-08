"""
M1-optimized object tracking for traffic lights and cameras.
"""

from __future__ import annotations

from .deepsort_tracker import DeepSORTTracker
from .kalman_filter import M1KalmanFilter

__all__ = ["DeepSORTTracker", "M1KalmanFilter"]
