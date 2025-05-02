"""
Traffic light relevance estimation module.
"""
import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.detection.detector import Detection


class LaneInfo:
    """Class representing lane information."""
    
    def __init__(
        self,
        lane_id: int,
        is_ego: bool = False,
        direction: str = "forward",
        confidence: float = 1.0,
    ):
        """
        Initialize lane information.
        
        Args:
            lane_id: Lane identifier
            is_ego: Whether this is the ego vehicle's lane
            direction: Lane direction ("forward", "left", "right", "uturn")
            confidence: Confidence in lane detection
        """
        self.lane_id = lane_id
        self.is_ego = is_ego
        self.direction = direction
        self.confidence = confidence


class RelevanceEstimator:
    """Estimates the relevance of detected traffic lights."""
    
    def __init__(self):
        """Initialize the relevance estimator."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Relevance estimator initialized")
    
    def estimate(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        gps_data: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Estimate relevance of traffic lights.
        
        Args:
            detections: List of detections
            frame: Input image frame
            gps_data: GPS data for the current frame
            
        Returns:
            List of relevance scores for each detection
        """
        # Extract traffic light detections
        traffic_lights = [
            d for d in detections
            if "traffic_light" in d.class_name
        ]
        
        # Detect lanes
        lanes = self._detect_lanes(frame)
        
        # Calculate relevance scores
        relevance_scores = []
        
        for tl in traffic_lights:
            # Skip if no bounding box
            if not hasattr(tl, "bbox") or not tl.bbox:
                continue
            
            # Calculate basic geometric relevance
            geo_score = self._geometric_relevance(tl, frame)
            
            # Calculate lane-based relevance
            lane_score = self._lane_relevance(tl, lanes, frame)
            
            # Calculate distance-based relevance (closer is more relevant)
            dist_score = 1.0
            if hasattr(tl, "distance") and tl.distance is not None:
                # Normalize distance (0-100m) to relevance score (1.0-0.0)
                dist_score = max(0.0, 1.0 - tl.distance / 100.0)
            
            # Apply state-based modifier (red lights are more important)
            state_mod = 1.0
            if hasattr(tl, "state") and tl.state is not None:
                if tl.state == "red":
                    state_mod = 1.5  # Red lights are more important
                elif tl.state == "yellow":
                    state_mod = 1.2  # Yellow lights are somewhat important
            
            # Compute final relevance score (weighted average)
            final_score = (
                geo_score * 0.3 +
                lane_score * 0.4 +
                dist_score * 0.3
            ) * state_mod
            
            # Clip to range [0, 1]
            final_score = min(1.0, max(0.0, final_score))
            
            # Add to detection
            tl.relevance_score = final_score
            
            # Add to results
            relevance_scores.append({
                "detection": tl,
                "relevance": final_score,
                "components": {
                    "geometric": geo_score,
                    "lane": lane_score,
                    "distance": dist_score,
                    "state_mod": state_mod,
                },
            })
        
        return relevance_scores
    
    def _detect_lanes(self, frame: np.ndarray) -> List[LaneInfo]:
        """
        Detect lanes in the image.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detected lanes
        """
        # This is a simplified implementation
        # In a real system, you would use a more sophisticated lane detection algorithm
        
        # For now, just assume one lane centered in the image
        ego_lane = LaneInfo(lane_id=0, is_ego=True, direction="forward")
        
        # Add some dummy adjacent lanes
        left_lane = LaneInfo(lane_id=1, is_ego=False, direction="forward", confidence=0.7)
        right_lane = LaneInfo(lane_id=2, is_ego=False, direction="forward", confidence=0.7)
        
        return [left_lane, ego_lane, right_lane]
    
    def _geometric_relevance(
        self,
        detection: Detection,
        frame: np.ndarray,
    ) -> float:
        """
        Calculate geometric relevance based on position in image.
        
        Args:
            detection: Detection object
            frame: Input image frame
            
        Returns:
            Geometric relevance score (0-1)
        """
        # Get image dimensions
        img_height, img_width = frame.shape[:2]
        img_center_x = img_width / 2
        
        # Get bounding box
        x1, y1, x2, y2 = detection.bbox
        
        # Calculate center of bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate horizontal distance from center
        # (normalize to [0, 1], where 0 is at center and 1 is at edge)
        h_dist = abs(center_x - img_center_x) / (img_width / 2)
        h_score = 1.0 - h_dist  # Higher score for objects near center
        
        # Calculate vertical position
        # (normalize to [0, 1], where 0 is at bottom and 1 is at top)
        v_pos = 1.0 - (center_y / img_height)
        
        # Traffic lights higher in the image are generally more relevant
        # but not too high (e.g., not in the sky)
        v_score = 1.0 - abs(v_pos - 0.6) * 2.0  # Peak at 0.6 from top
        v_score = max(0.0, min(1.0, v_score))  # Clip to [0, 1]
        
        # Combined score (weighted)
        geo_score = 0.7 * h_score + 0.3 * v_score
        
        return geo_score
    
    def _lane_relevance(
        self,
        detection: Detection,
        lanes: List[LaneInfo],
        frame: np.ndarray,
    ) -> float:
        """
        Calculate lane-based relevance.
        
        Args:
            detection: Detection object
            lanes: List of detected lanes
            frame: Input image frame
            
        Returns:
            Lane relevance score (0-1)
        """
        # Get image dimensions
        img_height, img_width = frame.shape[:2]
        
        # Get bounding box
        x1, y1, x2, y2 = detection.bbox
        
        # Calculate center of bounding box
        center_x = (x1 + x2) / 2
        
        # Calculate horizontal position (0 to 1, left to right)
        h_pos = center_x / img_width
        
        # Calculate which lane the traffic light is most aligned with
        lane_positions = [0.16, 0.5, 0.84]  # Approximate positions for 3 lanes
        lane_distances = [abs(h_pos - pos) for pos in lane_positions]
        closest_lane_idx = lane_distances.index(min(lane_distances))
        closest_lane = lanes[closest_lane_idx]
        
        # Calculate relevance based on lane
        if closest_lane.is_ego:
            # Traffic light in ego lane is highly relevant
            lane_score = 1.0
        else:
            # Traffic light in adjacent lane is less relevant
            # but still somewhat relevant (especially if close)
            lane_score = 0.5
        
        # Adjust by lane detection confidence
        lane_score *= closest_lane.confidence
        
        return lane_score