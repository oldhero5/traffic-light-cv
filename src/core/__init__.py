"""
Core M1-optimized processing modules.
"""

from __future__ import annotations

from .device_manager import DeviceManager
from .m1_optimizer import M1Optimizer
from .performance_monitor import PerformanceMonitor

__all__ = ["DeviceManager", "M1Optimizer", "PerformanceMonitor"]
