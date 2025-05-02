"""
Visualization module for traffic perception system.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.detection.detector import Detection
from src.mapping.mapper import MappedObject
from src.utils.constants import COLORS


class Visualizer:
    """Visualization utilities for traffic perception."""
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        show_distance: bool = True,
        show_relevance: bool = True,
        show_state: bool = True,
        font_scale: float = 0.5,
        line_thickness: int = 2,
    ):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
            show_distance: Whether to show distance information
            show_relevance: Whether to show relevance scores
            show_state: Whether to show traffic light state
            font_scale: Font scale for text
            line_thickness: Thickness of lines
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.show_distance = show_distance
        self.show_relevance = show_relevance
        self.show_state = show_state
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info("Visualizer initialized")
    
    def visualize(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        locations: List[Dict],
        relevance: List[Dict],
        mapped_objects: List[MappedObject],
    ) -> np.ndarray:
        """
        Visualize detections and other information on the frame.
        
        Args:
            frame: Input image frame
            detections: List of detections
            locations: List of localized objects
            relevance: List of relevance scores
            mapped_objects: List of mapped objects
            
        Returns:
            Annotated frame
        """
        # Create a copy of the frame
        vis_frame = frame.copy()
        
        # Draw detections
        for detection in detections:
            # Get bounding box
            x1, y1, x2, y2 = [int(c) for c in detection.bbox]
            
            # Get color based on class
            color = COLORS.get(detection.class_name, (255, 255, 255))
            
            # Modify color based on state for traffic lights
            if "traffic_light" in detection.class_name and hasattr(detection, "state"):
                if detection.state == "red":
                    color = (0, 0, 255)  # Red
                elif detection.state == "yellow":
                    color = (0, 255, 255)  # Yellow
                elif detection.state == "green":
                    color = (0, 255, 0)  # Green
            
            # Modify thickness based on relevance
            thickness = self.line_thickness
            if hasattr(detection, "relevance_score") and detection.relevance_score is not None:
                # Scale thickness by relevance (1.0 to 3.0 * base thickness)
                relevance_factor = 1.0 + 2.0 * detection.relevance_score
                thickness = int(self.line_thickness * relevance_factor)
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare text
            texts = []
            
            # Class name (shortened)
            class_short = detection.class_name.replace("traffic_", "").replace("_", " ")
            texts.append(f"{class_short}")
            
            # Confidence
            conf_text = f"{detection.confidence:.2f}"
            texts.append(conf_text)
            
            # Distance
            if self.show_distance and hasattr(detection, "distance") and detection.distance is not None:
                dist_text = f"{detection.distance:.1f}m"
                texts.append(dist_text)
            
            # State
            if self.show_state and hasattr(detection, "state") and detection.state:
                texts.append(detection.state)
            
            # Relevance
            if self.show_relevance and hasattr(detection, "relevance_score") and detection.relevance_score is not None:
                rel_text = f"R:{detection.relevance_score:.2f}"
                texts.append(rel_text)
            
            # Draw text background
            text = " | ".join(texts)
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )
            cv2.rectangle(
                vis_frame,
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1),
                color,
                -1,  # Filled
            )
            
            # Draw text
            cv2.putText(
                vis_frame,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (0, 0, 0),  # Black text
                1,
                cv2.LINE_AA,
            )
        
        # Draw global information
        self._draw_info_panel(vis_frame, len(detections), mapped_objects)
        
        return vis_frame
    
    def _draw_info_panel(
        self,
        frame: np.ndarray,
        num_detections: int,
        mapped_objects: List[MappedObject],
    ) -> None:
        """
        Draw information panel on the frame.
        
        Args:
            frame: Frame to draw on
            num_detections: Number of detections
            mapped_objects: List of mapped objects
        """
        # Count objects by class
        class_counts = {}
        for obj in mapped_objects:
            if obj.class_name not in class_counts:
                class_counts[obj.class_name] = 0
            class_counts[obj.class_name] += 1
        
        # Draw panel background
        panel_height = 30
        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], panel_height),
            (0, 0, 0),
            -1,  # Filled
        )
        
        # Draw text
        text = f"Detections: {num_detections} | Mapped: {len(mapped_objects)}"
        
        # Add class counts
        class_texts = []
        for cls, count in class_counts.items():
            short_name = cls.replace("traffic_", "").replace("_", " ")
            class_texts.append(f"{short_name}: {count}")
        
        if class_texts:
            text += " | " + ", ".join(class_texts)
        
        cv2.putText(
            frame,
            text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White text
            1,
            cv2.LINE_AA,
        )
    
    def save_frame(self, frame: np.ndarray, filename: str) -> bool:
        """
        Save a frame to disk.
        
        Args:
            frame: Frame to save
            filename: Filename to save as
            
        Returns:
            Success flag
        """
        if not self.output_dir:
            self.logger.warning("No output directory specified")
            return False
        
        try:
            output_path = os.path.join(self.output_dir, filename)
            cv2.imwrite(output_path, frame)
            return True
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")
            return False
    
    def create_map_visualization(
        self,
        mapped_objects: List[MappedObject],
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        zoom: int = 17,
        width: int = 800,
        height: int = 600,
    ) -> np.ndarray:
        """
        Create a map visualization of mapped objects.
        
        This is a placeholder implementation - in a real system,
        you would use a mapping library like Mapbox or Folium.
        
        Args:
            mapped_objects: List of mapped objects
            center_lat: Center latitude (if None, use mean of objects)
            center_lon: Center longitude (if None, use mean of objects)
            zoom: Map zoom level
            width: Image width
            height: Image height
            
        Returns:
            Map visualization image
        """
        # Create a blank image
        map_image = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # If no objects, return blank map
        if not mapped_objects:
            cv2.putText(
                map_image,
                "No mapped objects",
                (width // 2 - 100, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            return map_image
        
        # Calculate center if not provided
        if center_lat is None or center_lon is None:
            lats = [obj.latitude for obj in mapped_objects if obj.latitude is not None]
            lons = [obj.longitude for obj in mapped_objects if obj.longitude is not None]
            
            if lats and lons:
                center_lat = sum(lats) / len(lats)
                center_lon = sum(lons) / len(lons)
            else:
                center_lat = 0.0
                center_lon = 0.0
        
        # This is where you would render a real map
        # For now, just draw a simple visualization
        
        # Draw a grid
        for i in range(0, width, 50):
            cv2.line(map_image, (i, 0), (i, height), (200, 200, 200), 1)
        for i in range(0, height, 50):
            cv2.line(map_image, (0, i), (width, i), (200, 200, 200), 1)
        
        # Calculate scale (pixels per degree)
        # This is a very simplified approach
        scale_factor = 2 ** zoom * 100
        
        # Calculate pixel coordinates
        for obj in mapped_objects:
            if obj.latitude is None or obj.longitude is None:
                continue
            
            # Convert to pixels
            x = int(width / 2 + (obj.longitude - center_lon) * scale_factor)
            y = int(height / 2 - (obj.latitude - center_lat) * scale_factor)
            
            # Skip if outside image
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            
            # Get color by class
            color = COLORS.get(obj.class_name, (255, 255, 255))
            
            # Modify color based on state for traffic lights
            if "traffic_light" in obj.class_name and obj.state:
                if obj.state == "red":
                    color = (0, 0, 255)  # Red
                elif obj.state == "yellow":
                    color = (0, 255, 255)  # Yellow
                elif obj.state == "green":
                    color = (0, 255, 0)  # Green
            
            # Draw object
            cv2.circle(map_image, (x, y), 8, color, -1)
            cv2.circle(map_image, (x, y), 8, (0, 0, 0), 1)
            
            # Draw label
            label = obj.class_name.replace("traffic_", "").replace("_", " ")
            cv2.putText(
                map_image,
                label,
                (x + 10, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        
        # Draw center point (vehicle position)
        cv2.drawMarker(
            map_image,
            (width // 2, height // 2),
            (0, 0, 255),
            cv2.MARKER_STAR,
            20,
            2,
        )
        
        # Draw coordinates
        cv2.putText(
            map_image,
            f"Center: {center_lat:.6f}, {center_lon:.6f}",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        
        return map_image