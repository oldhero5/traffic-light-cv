"""
Object localization module for traffic lights and cameras.
"""
import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from pykalman import KalmanFilter

from src.detection.detector import Detection


class Position:
    """Class representing a 3D position."""
    
    def __init__(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        altitude: Optional[float] = None,
        distance: Optional[float] = None,
        heading: Optional[float] = None,
        relative_bearing: Optional[float] = None,
    ):
        """
        Initialize a position.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            altitude: Altitude in meters
            distance: Distance from observer in meters
            heading: Heading in degrees (0-360, 0 = North)
            relative_bearing: Bearing relative to observer's heading in degrees
        """
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.distance = distance
        self.heading = heading
        self.relative_bearing = relative_bearing


class CameraCalibration:
    """Camera calibration parameters."""
    
    def __init__(
        self,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        image_width: int = 1920,
        image_height: int = 1080,
        fov_horizontal: float = 60.0,
        fov_vertical: float = 35.0,
    ):
        """
        Initialize camera calibration.
        
        Args:
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients
            image_width: Image width in pixels
            image_height: Image height in pixels
            fov_horizontal: Horizontal field of view in degrees
            fov_vertical: Vertical field of view in degrees
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.image_width = image_width
        self.image_height = image_height
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
        
        # Calculate focal length if camera matrix is not provided
        if self.camera_matrix is None:
            # Estimate focal length from FoV
            self.focal_length_x = (self.image_width / 2) / np.tan(np.radians(self.fov_horizontal / 2))
            self.focal_length_y = (self.image_height / 2) / np.tan(np.radians(self.fov_vertical / 2))
            
            # Create estimated camera matrix
            self.camera_matrix = np.array([
                [self.focal_length_x, 0, self.image_width / 2],
                [0, self.focal_length_y, self.image_height / 2],
                [0, 0, 1]
            ])
        
        # Use zeros for distortion if not provided
        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros(5)


class ObjectLocalizer:
    """Localizes detected objects in 3D space."""
    
    # Known physical dimensions of objects in meters (width, height)
    OBJECT_DIMENSIONS = {
        "traffic_light": (0.3, 0.8),
        "traffic_camera": (0.4, 0.4),
    }
    
    def __init__(
        self,
        calibration: Optional[CameraCalibration] = None,
        use_kalman: bool = True,
    ):
        """
        Initialize the object localizer.
        
        Args:
            calibration: Camera calibration parameters
            use_kalman: Whether to use Kalman filtering for smoothing
        """
        self.logger = logging.getLogger(__name__)
        
        # Use default calibration if not provided
        self.calibration = calibration if calibration else CameraCalibration()
        
        # Kalman filter setup
        self.use_kalman = use_kalman
        self.kalman_filters = {}  # tracking_id -> KalmanFilter
        
        # Last known GPS position
        self.last_gps = None
        
        self.logger.info("Object localizer initialized")
    
    def localize(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        gps_data: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Localize detected objects.
        
        Args:
            detections: List of detections
            frame: Input image frame
            gps_data: GPS data for the current frame
            
        Returns:
            List of dictionaries with localization information
        """
        # Update last known GPS position if available
        if gps_data and "latitude" in gps_data and "longitude" in gps_data:
            self.last_gps = gps_data
        
        localized_objects = []
        
        for detection in detections:
            # Get object dimensions
            obj_type = detection.class_name
            if obj_type in self.OBJECT_DIMENSIONS:
                obj_width, obj_height = self.OBJECT_DIMENSIONS[obj_type]
            else:
                # Use default dimensions for unknown objects
                obj_width, obj_height = 0.3, 0.3
            
            # Estimate distance using size-based method
            distance = self._estimate_distance_by_size(detection, obj_height, frame)
            
            # Add distance to detection
            detection.distance = distance
            
            # Create position object
            position = Position(distance=distance)
            
            # Calculate relative bearing if we know the image center
            img_center_x = frame.shape[1] / 2
            x1, y1, x2, y2 = detection.bbox
            obj_center_x = (x1 + x2) / 2
            pixels_from_center = obj_center_x - img_center_x
            
            # Convert pixels to degrees using horizontal FOV
            angle_per_pixel = self.calibration.fov_horizontal / frame.shape[1]
            relative_bearing = pixels_from_center * angle_per_pixel
            position.relative_bearing = relative_bearing
            
            # Calculate world position if GPS data is available
            if self.last_gps:
                world_position = self._calculate_world_position(
                    distance, relative_bearing, self.last_gps
                )
                position.latitude = world_position["latitude"]
                position.longitude = world_position["longitude"]
                position.heading = self.last_gps.get("heading")
            
            # Apply Kalman filtering if tracking ID is available and Kalman is enabled
            if self.use_kalman and detection.tracking_id is not None:
                position = self._apply_kalman_filter(detection.tracking_id, position)
            
            # Add to results
            localized_obj = {
                "detection": detection,
                "position": position,
            }
            localized_objects.append(localized_obj)
            
            # Update detection object with position data
            detection.world_position = position
        
        return localized_objects
    
    def _estimate_distance_by_size(
        self,
        detection: Detection,
        known_height: float,
        frame: np.ndarray,
    ) -> float:
        """
        Estimate distance using the known size of an object.
        
        Args:
            detection: Detection object
            known_height: Known physical height of the object in meters
            frame: Input image frame
            
        Returns:
            Estimated distance in meters
        """
        # Get bounding box
        x1, y1, x2, y2 = detection.bbox
        
        # Calculate pixel height of the object
        pixel_height = y2 - y1
        
        # Avoid division by zero
        if pixel_height <= 0:
            return 100.0  # Default large distance
        
        # Calculate distance using the formula:
        # distance = (known_height * focal_length) / pixel_height
        focal_length = self.calibration.camera_matrix[1, 1]  # fy
        distance = (known_height * focal_length) / pixel_height
        
        return distance
    
    def _calculate_world_position(
        self,
        distance: float,
        bearing: float,
        gps_data: Dict,
    ) -> Dict:
        """
        Calculate world position from distance and bearing.
        
        Args:
            distance: Distance in meters
            bearing: Relative bearing in degrees
            gps_data: GPS data for the current frame
            
        Returns:
            Dictionary with latitude and longitude
        """
        # Get vehicle GPS position
        vehicle_lat = gps_data["latitude"]
        vehicle_lon = gps_data["longitude"]
        
        # Get vehicle heading (0 = North, 90 = East)
        vehicle_heading = gps_data.get("heading", 0)
        
        # Calculate absolute bearing
        absolute_bearing = (vehicle_heading + bearing) % 360
        
        # Convert to radians
        lat1 = math.radians(vehicle_lat)
        lon1 = math.radians(vehicle_lon)
        brng = math.radians(absolute_bearing)
        
        # Earth radius in meters
        earth_radius = 6371000
        
        # Calculate new position
        distance_ratio = distance / earth_radius
        lat2 = math.asin(
            math.sin(lat1) * math.cos(distance_ratio) +
            math.cos(lat1) * math.sin(distance_ratio) * math.cos(brng)
        )
        lon2 = lon1 + math.atan2(
            math.sin(brng) * math.sin(distance_ratio) * math.cos(lat1),
            math.cos(distance_ratio) - math.sin(lat1) * math.sin(lat2)
        )
        
        # Convert back to degrees
        lat2 = math.degrees(lat2)
        lon2 = math.degrees(lon2)
        
        return {
            "latitude": lat2,
            "longitude": lon2,
        }
    
    def _apply_kalman_filter(self, tracking_id: str, position: Position) -> Position:
        """
        Apply Kalman filter to smooth position estimates.
        
        Args:
            tracking_id: Tracking ID for the object
            position: Estimated position
            
        Returns:
            Filtered position
        """
        # Skip if no latitude or longitude
        if position.latitude is None or position.longitude is None:
            return position
        
        # Create state vector [lat, lon, alt, distance]
        state = np.array([
            position.latitude or 0,
            position.longitude or 0,
            position.altitude or 0,
            position.distance or 0,
        ])
        
        # Initialize new Kalman filter if not exists for this ID
        if tracking_id not in self.kalman_filters:
            # Simple constant velocity model
            transition_matrix = np.eye(4)
            observation_matrix = np.eye(4)
            
            initial_state_mean = state
            initial_state_cov = np.eye(4) * 1.0
            
            transition_cov = np.eye(4) * 0.01
            observation_cov = np.eye(4) * 0.1
            
            self.kalman_filters[tracking_id] = KalmanFilter(
                transition_matrices=transition_matrix,
                observation_matrices=observation_matrix,
                initial_state_mean=initial_state_mean,
                initial_state_covariance=initial_state_cov,
                transition_covariance=transition_cov,
                observation_covariance=observation_cov,
            )
        
        # Get Kalman filter for this object
        kf = self.kalman_filters[tracking_id]
        
        # Apply filter
        filtered_state, _ = kf.filter_update(
            filtered_state_mean=kf.initial_state_mean,
            filtered_state_covariance=kf.initial_state_covariance,
            observation=state,
        )
        
        # Update Kalman filter initial state for next update
        kf.initial_state_mean = filtered_state
        
        # Create new position with filtered values
        filtered_position = Position(
            latitude=filtered_state[0],
            longitude=filtered_state[1],
            altitude=filtered_state[2],
            distance=filtered_state[3],
            heading=position.heading,
            relative_bearing=position.relative_bearing,
        )
        
        return filtered_position