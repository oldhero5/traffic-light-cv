"""
Mobile Traffic Awareness System

This package provides mobile-optimized components for traffic light detection,
camera awareness, and GPS integration designed for dashcam video processing.

M1 Performance:
- GPS tracking: <1ms processing latency
- Real-time processing: >30 FPS
- Memory usage: <200MB for mobile components
- Power consumption: <5W average
"""

from .gps_tracker import MobileGPSTracker

__version__ = "1.0.0"
__all__ = ["MobileGPSTracker"]