"""
Mapping module for visualizing detected objects on a map.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import rtree


class MappedObject:
    """Class representing an object on the map."""

    def __init__(
        self,
        object_id: str,
        class_name: str,
        latitude: float,
        longitude: float,
        state: str | None = None,
        confidence: float = 1.0,
        altitude: float | None = None,
        first_seen: int = 0,
        last_seen: int = 0,
        count: int = 1,
    ):
        """
        Initialize a mapped object.

        Args:
            object_id: Unique identifier for the object
            class_name: Class name of the object
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            state: State of the object (e.g., red, green, yellow for traffic lights)
            confidence: Confidence score (0-1)
            altitude: Altitude in meters
            first_seen: Frame number when first detected
            last_seen: Frame number when last detected
            count: Number of times detected
        """
        self.object_id = object_id
        self.class_name = class_name
        self.latitude = latitude
        self.longitude = longitude
        self.state = state
        self.confidence = confidence
        self.altitude = altitude
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.count = count

        # History of positions for filtering
        self.position_history = []

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.object_id,
            "class": self.class_name,
            "lat": self.latitude,
            "lon": self.longitude,
            "state": self.state,
            "confidence": self.confidence,
            "alt": self.altitude,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "count": self.count,
        }

    def update_position(self, lat: float, lon: float, alt: float | None = None) -> None:
        """
        Update position with new coordinates.

        Args:
            lat: New latitude
            lon: New longitude
            alt: New altitude (optional)
        """
        # Add current position to history (max 10 positions)
        self.position_history.append((self.latitude, self.longitude))
        if len(self.position_history) > 10:
            self.position_history.pop(0)

        # Update position with median filter if we have history
        if len(self.position_history) > 3:
            # Calculate median of recent positions
            recent_lats = [p[0] for p in self.position_history[-3:]] + [lat]
            recent_lons = [p[1] for p in self.position_history[-3:]] + [lon]

            # Use median filtering
            self.latitude = np.median(recent_lats)
            self.longitude = np.median(recent_lons)
        else:
            # Not enough history, use new position directly
            self.latitude = lat
            self.longitude = lon

        # Update altitude if provided
        if alt is not None:
            self.altitude = alt

    def update_state(self, state: str) -> None:
        """
        Update state of the object.

        Args:
            state: New state
        """
        self.state = state


class Mapper:
    """Maps detected objects to geographic coordinates."""

    def __init__(
        self,
        persistence_file: str | None = None,
        distance_threshold: float = 10.0,  # meters
    ):
        """
        Initialize the mapper.

        Args:
            persistence_file: File to save mapped objects
            distance_threshold: Maximum distance in meters to merge objects
        """
        self.logger = logging.getLogger(__name__)
        self.mapped_objects = {}  # object_id -> MappedObject
        self.persistence_file = persistence_file
        self.distance_threshold = distance_threshold

        # Initialize spatial index for fast spatial queries
        self.spatial_index = rtree.index.Index()
        self.spatial_idx_counter = 0

        # Load persisted objects if file exists
        if persistence_file and os.path.exists(persistence_file):
            self._load()

        self.logger.info("Mapper initialized")

    def update(self, localized_objects: list[dict], frame_number: int) -> list[MappedObject]:
        """
        Update the map with new localized objects.

        Args:
            localized_objects: List of localized objects
            frame_number: Current frame number

        Returns:
            List of all mapped objects
        """
        for obj in localized_objects:
            detection = obj["detection"]
            position = obj["position"]

            # Skip if no geographic coordinates
            if position.latitude is None or position.longitude is None:
                continue

            # Check if the object is already on the map
            existing_obj = self._find_nearby_object(
                position.latitude, position.longitude, detection.class_name
            )

            if existing_obj:
                # Update existing object
                existing_obj.update_position(
                    position.latitude, position.longitude, position.altitude
                )

                # Update state if it's a traffic light
                if "traffic_light" in detection.class_name and detection.state:
                    existing_obj.update_state(detection.state)

                # Update tracking information
                existing_obj.last_seen = frame_number
                existing_obj.count += 1

                # Update confidence (weighted average)
                existing_obj.confidence = existing_obj.confidence * 0.8 + detection.confidence * 0.2
            else:
                # Create new object
                object_id = f"{detection.class_name}_{len(self.mapped_objects)}"

                new_obj = MappedObject(
                    object_id=object_id,
                    class_name=detection.class_name,
                    latitude=position.latitude,
                    longitude=position.longitude,
                    state=detection.state if hasattr(detection, "state") else None,
                    confidence=detection.confidence,
                    altitude=position.altitude,
                    first_seen=frame_number,
                    last_seen=frame_number,
                )

                # Add to mapped objects
                self.mapped_objects[object_id] = new_obj

                # Add to spatial index
                self._add_to_spatial_index(new_obj)

        # Persist objects if requested
        if self.persistence_file:
            self._save()

        return list(self.mapped_objects.values())

    def get_objects_by_class(self, class_name: str) -> list[MappedObject]:
        """
        Get all mapped objects of a specific class.

        Args:
            class_name: Class name to filter by

        Returns:
            List of mapped objects of the specified class
        """
        return [obj for obj in self.mapped_objects.values() if class_name in obj.class_name]

    def get_objects_in_radius(
        self,
        latitude: float,
        longitude: float,
        radius: float,  # meters
    ) -> list[MappedObject]:
        """
        Get all mapped objects within a radius of a point.

        Args:
            latitude: Center latitude
            longitude: Center longitude
            radius: Radius in meters

        Returns:
            List of mapped objects within the radius
        """
        # Approximate degrees per meter at this latitude
        lat_meters = 111320  # meters per degree of latitude
        lon_meters = 111320 * np.cos(np.radians(latitude))  # meters per degree of longitude

        # Convert radius to degrees
        lat_radius = radius / lat_meters
        lon_radius = radius / lon_meters

        # Query spatial index
        min_lat, min_lon = latitude - lat_radius, longitude - lon_radius
        max_lat, max_lon = latitude + lat_radius, longitude + lon_radius

        nearby_indices = list(self.spatial_index.intersection((min_lon, min_lat, max_lon, max_lat)))

        # Get objects from indices
        nearby_objects = []
        for idx in nearby_indices:
            for obj in self.mapped_objects.values():
                if obj.object_id == f"idx_{idx}":
                    nearby_objects.append(obj)

        return nearby_objects

    def clear(self) -> None:
        """Clear all mapped objects."""
        self.mapped_objects = {}

        # Reset spatial index
        self.spatial_index = rtree.index.Index()
        self.spatial_idx_counter = 0

    def _find_nearby_object(
        self,
        latitude: float,
        longitude: float,
        class_name: str,
    ) -> MappedObject | None:
        """
        Find existing objects near the given coordinates of the same class.

        Args:
            latitude: Latitude to search near
            longitude: Longitude to search near
            class_name: Class name to match

        Returns:
            Nearby object if found, None otherwise
        """
        # Approximate degrees per meter at this latitude
        lat_meters = 111320  # meters per degree of latitude
        lon_meters = 111320 * np.cos(np.radians(latitude))  # meters per degree of longitude

        # Convert threshold to degrees
        lat_threshold = self.distance_threshold / lat_meters
        lon_threshold = self.distance_threshold / lon_meters

        # Query spatial index
        min_lat, min_lon = latitude - lat_threshold, longitude - lon_threshold
        max_lat, max_lon = latitude + lat_threshold, longitude + lon_threshold

        nearby_indices = list(self.spatial_index.intersection((min_lon, min_lat, max_lon, max_lat)))

        # Get objects from indices and filter by class
        for idx in nearby_indices:
            for obj in self.mapped_objects.values():
                if obj.object_id == f"idx_{idx}" and class_name in obj.class_name:
                    # Calculate exact distance
                    dist = self._haversine_distance(
                        latitude, longitude, obj.latitude, obj.longitude
                    )

                    if dist <= self.distance_threshold:
                        return obj

        return None

    def _add_to_spatial_index(self, obj: MappedObject) -> None:
        """
        Add object to spatial index.

        Args:
            obj: Object to add
        """
        # Add to spatial index
        idx = self.spatial_idx_counter
        self.spatial_index.insert(
            idx,
            (obj.longitude, obj.latitude, obj.longitude, obj.latitude),
            obj=obj.object_id,
        )

        # Link index to object
        obj.object_id = f"idx_{idx}"

        # Increment counter
        self.spatial_idx_counter += 1

    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """
        Calculate Haversine distance between two points.

        Args:
            lat1: Latitude of point 1
            lon1: Longitude of point 1
            lat2: Latitude of point 2
            lon2: Longitude of point 2

        Returns:
            Distance in meters
        """
        # Earth radius in meters
        earth_radius = 6371000

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = earth_radius * c

        return distance

    def _save(self) -> None:
        """Save mapped objects to file."""
        try:
            data = {
                "objects": [obj.to_dict() for obj in self.mapped_objects.values()],
            }

            os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)

            with open(self.persistence_file, "w") as f:
                json.dump(data, f, indent=2)

            self.logger.debug(
                f"Saved {len(self.mapped_objects)} objects to {self.persistence_file}"
            )
        except Exception as e:
            self.logger.error(f"Failed to save mapped objects: {e}")

    def _load(self) -> None:
        """Load mapped objects from file."""
        try:
            with open(self.persistence_file) as f:
                data = json.load(f)

            self.mapped_objects = {}
            for obj_data in data.get("objects", []):
                obj = MappedObject(
                    object_id=obj_data["id"],
                    class_name=obj_data["class"],
                    latitude=obj_data["lat"],
                    longitude=obj_data["lon"],
                    state=obj_data.get("state"),
                    confidence=obj_data.get("confidence", 1.0),
                    altitude=obj_data.get("alt"),
                    first_seen=obj_data.get("first_seen", 0),
                    last_seen=obj_data.get("last_seen", 0),
                    count=obj_data.get("count", 1),
                )

                self.mapped_objects[obj.object_id] = obj
                self._add_to_spatial_index(obj)

            self.logger.info(
                f"Loaded {len(self.mapped_objects)} objects from {self.persistence_file}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load mapped objects: {e}")
            self.mapped_objects = {}
