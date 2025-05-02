"""
GPS data reader module.
"""
import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


class GPSReader:
    """Class for reading GPS data from various formats."""
    
    def __init__(self, gps_file: Union[str, Path]):
        """
        Initialize the GPS reader.
        
        Args:
            gps_file: Path to GPS data file
        """
        self.logger = logging.getLogger(__name__)
        self.gps_file = str(gps_file)
        self.data = []
        self.loaded = False
        
        # Try to load the data
        self._load()
    
    def _load(self) -> None:
        """Load GPS data from file."""
        if not os.path.exists(self.gps_file):
            self.logger.error(f"GPS file not found: {self.gps_file}")
            return
        
        # Determine file format from extension
        ext = os.path.splitext(self.gps_file)[1].lower()
        
        try:
            if ext == ".csv":
                self._load_csv()
            elif ext == ".json":
                self._load_json()
            elif ext == ".nmea":
                self._load_nmea()
            elif ext == ".gpx":
                self._load_gpx()
            else:
                self.logger.error(f"Unsupported GPS file format: {ext}")
                return
            
            self.loaded = True
            self.logger.info(f"Loaded {len(self.data)} GPS points from {self.gps_file}")
        
        except Exception as e:
            self.logger.error(f"Error loading GPS data: {e}")
    
    def _load_csv(self) -> None:
        """Load GPS data from CSV file."""
        self.data = []
        
        with open(self.gps_file, "r") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Try to parse required fields
                try:
                    gps_point = {
                        "timestamp": float(row.get("timestamp", 0)),
                        "latitude": float(row.get("latitude", 0)),
                        "longitude": float(row.get("longitude", 0)),
                    }
                    
                    # Add optional fields if present
                    if "altitude" in row:
                        gps_point["altitude"] = float(row["altitude"])
                    if "heading" in row:
                        gps_point["heading"] = float(row["heading"])
                    if "speed" in row:
                        gps_point["speed"] = float(row["speed"])
                    if "frame" in row:
                        gps_point["frame"] = int(row["frame"])
                    
                    self.data.append(gps_point)
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Error parsing CSV row: {e}")
                    continue
    
    def _load_json(self) -> None:
        """Load GPS data from JSON file."""
        with open(self.gps_file, "r") as f:
            data = json.load(f)
        
        # Check if data is a list of points or has a specific structure
        if isinstance(data, list):
            # Assume list of points
            self.data = data
        elif isinstance(data, dict) and "gps_data" in data:
            # Structured with a "gps_data" key
            self.data = data["gps_data"]
        else:
            self.logger.error("Unexpected JSON structure")
            self.data = []
    
    def _load_nmea(self) -> None:
        """Load GPS data from NMEA file."""
        # This is a simplified implementation
        # In a real system, you would use a proper NMEA parser
        
        self.data = []
        current_point = {}
        
        with open(self.gps_file, "r") as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Skip lines that don't start with "$"
                if not line.startswith("$"):
                    continue
                
                # Parse NMEA sentence
                try:
                    parts = line.split(",")
                    sentence_type = parts[0]
                    
                    # Parse GGA sentence (position)
                    if sentence_type == "$GPGGA" or sentence_type == "$GNGGA":
                        if len(parts) >= 10:
                            lat = self._parse_nmea_lat(parts[2], parts[3])
                            lon = self._parse_nmea_lon(parts[4], parts[5])
                            alt = float(parts[9]) if parts[9] else 0.0
                            
                            current_point["latitude"] = lat
                            current_point["longitude"] = lon
                            current_point["altitude"] = alt
                    
                    # Parse RMC sentence (position, speed, heading)
                    elif sentence_type == "$GPRMC" or sentence_type == "$GNRMC":
                        if len(parts) >= 10:
                            lat = self._parse_nmea_lat(parts[3], parts[4])
                            lon = self._parse_nmea_lon(parts[5], parts[6])
                            speed = float(parts[7]) * 1.852 if parts[7] else 0.0  # Convert knots to km/h
                            heading = float(parts[8]) if parts[8] else 0.0
                            
                            current_point["latitude"] = lat
                            current_point["longitude"] = lon
                            current_point["speed"] = speed
                            current_point["heading"] = heading
                    
                    # If we have all required data, add point and reset
                    if all(k in current_point for k in ["latitude", "longitude"]):
                        # Add timestamp as index
                        current_point["timestamp"] = len(self.data)
                        
                        self.data.append(current_point.copy())
                        current_point = {}
                
                except Exception as e:
                    self.logger.warning(f"Error parsing NMEA sentence: {e}")
                    continue
    
    def _load_gpx(self) -> None:
        """Load GPS data from GPX file."""
        # This is a simplified implementation using string parsing
        # In a real system, you would use a proper GPX parser like gpxpy
        
        import xml.etree.ElementTree as ET
        
        self.data = []
        
        try:
            tree = ET.parse(self.gps_file)
            root = tree.getroot()
            
            # GPX namespace
            ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
            
            # Find track points
            track_points = root.findall(".//gpx:trkpt", ns)
            
            for i, point in enumerate(track_points):
                try:
                    lat = float(point.get("lat"))
                    lon = float(point.get("lon"))
                    
                    gps_point = {
                        "timestamp": i,  # Use index as timestamp
                        "latitude": lat,
                        "longitude": lon,
                    }
                    
                    # Get elevation if available
                    ele = point.find("gpx:ele", ns)
                    if ele is not None and ele.text:
                        gps_point["altitude"] = float(ele.text)
                    
                    # Get time if available
                    time_elem = point.find("gpx:time", ns)
                    if time_elem is not None and time_elem.text:
                        # Store original time string
                        gps_point["time"] = time_elem.text
                    
                    self.data.append(gps_point)
                
                except Exception as e:
                    self.logger.warning(f"Error parsing GPX point: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error parsing GPX file: {e}")
    
    def _parse_nmea_lat(self, lat_str: str, lat_dir: str) -> float:
        """
        Parse NMEA format latitude.
        
        Args:
            lat_str: Latitude string (DDMM.MMMMM)
            lat_dir: Direction (N/S)
            
        Returns:
            Latitude in decimal degrees
        """
        if not lat_str:
            return 0.0
        
        # NMEA format is DDMM.MMMMM
        degrees = float(lat_str[:2])
        minutes = float(lat_str[2:])
        
        lat = degrees + minutes / 60.0
        
        # Apply direction
        if lat_dir == "S":
            lat = -lat
        
        return lat
    
    def _parse_nmea_lon(self, lon_str: str, lon_dir: str) -> float:
        """
        Parse NMEA format longitude.
        
        Args:
            lon_str: Longitude string (DDDMM.MMMMM)
            lon_dir: Direction (E/W)
            
        Returns:
            Longitude in decimal degrees
        """
        if not lon_str:
            return 0.0
        
        # NMEA format is DDDMM.MMMMM
        degrees = float(lon_str[:3])
        minutes = float(lon_str[3:])
        
        lon = degrees + minutes / 60.0
        
        # Apply direction
        if lon_dir == "W":
            lon = -lon
        
        return lon
    
    def get_data(self, frame_number: int) -> Optional[Dict]:
        """
        Get GPS data for a specific frame.
        
        Args:
            frame_number: Frame number
            
        Returns:
            GPS data dictionary or None if not found
        """
        if not self.loaded or not self.data:
            return None
        
        # Try to find exact frame match
        for point in self.data:
            if point.get("frame") == frame_number:
                return point
        
        # If no exact match, use time-based interpolation
        if len(self.data) == 1:
            return self.data[0]
        
        # If we have a reference to video timing, interpolate
        if all("timestamp" in point for point in self.data):
            # Simplified interpolation - find closest point
            # In a real system, you would do proper interpolation
            
            # Estimate timestamp for this frame based on first and last point
            first_timestamp = self.data[0]["timestamp"]
            last_timestamp = self.data[-1]["timestamp"]
            first_frame = self.data[0].get("frame", 0)
            last_frame = self.data[-1].get("frame", len(self.data) - 1)
            
            frame_range = last_frame - first_frame
            timestamp_range = last_timestamp - first_timestamp
            
            if frame_range > 0 and timestamp_range > 0:
                # Estimate timestamp for this frame
                frame_progress = (frame_number - first_frame) / frame_range
                est_timestamp = first_timestamp + frame_progress * timestamp_range
                
                # Find closest point
                closest_idx = min(
                    range(len(self.data)),
                    key=lambda i: abs(self.data[i]["timestamp"] - est_timestamp)
                )
                
                return self.data[closest_idx]
        
        # Fallback: use frame number as index (modulo data length)
        idx = frame_number % len(self.data)
        return self.data[idx]