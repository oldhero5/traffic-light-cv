"""
Constants used throughout the system.
"""

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

# Traffic light classes
TRAFFIC_LIGHT_CLASSES = {
    0: "traffic_light_unknown",
    1: "traffic_light_red",
    2: "traffic_light_yellow",
    3: "traffic_light_green",
}

# Traffic camera classes
TRAFFIC_CAMERA_CLASSES = {
    4: "traffic_camera",
    5: "speed_camera",
    6: "red_light_camera",
}

# Physical dimensions of objects in meters (width, height)
OBJECT_DIMENSIONS = {
    "traffic_light_unknown": (0.3, 0.8),
    "traffic_light_red": (0.3, 0.8),
    "traffic_light_yellow": (0.3, 0.8),
    "traffic_light_green": (0.3, 0.8),
    "traffic_camera": (0.4, 0.4),
    "speed_camera": (0.4, 0.4),
    "red_light_camera": (0.4, 0.4),
}

# Colors for visualization (BGR format)
COLORS = {
    "traffic_light_unknown": (128, 128, 128),  # Gray
    "traffic_light_red": (0, 0, 255),  # Red
    "traffic_light_yellow": (0, 255, 255),  # Yellow
    "traffic_light_green": (0, 255, 0),  # Green
    "traffic_camera": (255, 0, 0),  # Blue
    "speed_camera": (255, 0, 255),  # Magenta
    "red_light_camera": (255, 128, 0),  # Purple
}

# Default camera parameters
DEFAULT_CAMERA_MATRIX = [
    [1000, 0, 960],
    [0, 1000, 540],
    [0, 0, 1]
]

DEFAULT_DISTORTION_COEFFICIENTS = [0, 0, 0, 0, 0]

# Default field of view in degrees
DEFAULT_FOV_HORIZONTAL = 60.0
DEFAULT_FOV_VERTICAL = 35.0