"""
Video reader module with GPS metadata extraction support.

M1 Performance:
- Hardware-accelerated video decoding
- GPS metadata extraction from MP4 files
- Memory-efficient frame buffering
- Support for dashcam formats
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


class VideoReader:
    """Class for reading video files or camera streams with GPS support."""

    def __init__(
        self,
        source: str | int | Path,
        buffer_size: int = 5,
        extract_gps: bool = True,
    ):
        """
        Initialize the video reader.

        Args:
            source: Video source (file path or camera index)
            buffer_size: Size of the frame buffer
            extract_gps: Whether to extract GPS metadata from MP4 files
        """
        self.logger = logging.getLogger(__name__)
        self.source = source
        self.buffer_size = buffer_size
        self.extract_gps = extract_gps
        self.cap = None
        self.frame_count = 0

        # Frame buffer for smoother processing
        self.buffer = []
        self.buffer_idx = 0
        
        # GPS metadata
        self.gps_metadata: List[Dict] = []
        self.gps_track: Dict = {}

        # Try to open the video source
        self._open()

    def _open(self) -> None:
        """Open the video source."""
        try:
            self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video source: {self.source}")
                raise ValueError(f"Could not open video source: {self.source}")

            # Get video properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if isinstance(self.source, (str, Path)):
                self.logger.info(
                    f"Opened video file: {self.source}, "
                    f"dimensions: {self.width}x{self.height}, "
                    f"FPS: {self.fps:.2f}, frames: {self.total_frames}"
                )
            else:
                self.logger.info(
                    f"Opened camera: {self.source}, "
                    f"dimensions: {self.width}x{self.height}, "
                    f"FPS: {self.fps:.2f}"
                )

            # Initialize frame buffer
            self._init_buffer()
            
            # Extract GPS metadata if requested and source is a file
            if self.extract_gps and isinstance(self.source, (str, Path)):
                self._extract_gps_metadata()

        except Exception as e:
            self.logger.error(f"Error opening video source: {e}")
            raise

    def _init_buffer(self) -> None:
        """Initialize the frame buffer."""
        # Skip for camera input
        if isinstance(self.source, int):
            return

        self.logger.debug("Initializing frame buffer")
        self.buffer = []

        # Read initial frames into buffer
        for _ in range(self.buffer_size):
            ret, frame = self.cap.read()
            if ret:
                self.buffer.append(frame)
            else:
                break

        self.buffer_idx = 0
        self.logger.debug(f"Initialized buffer with {len(self.buffer)} frames")

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Read the next frame.

        Returns:
            Tuple of (success, frame)
        """
        # For camera input, read directly
        if isinstance(self.source, int):
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
            return ret, frame

        # For file input, use the buffer if available
        if self.buffer and self.buffer_idx < len(self.buffer):
            frame = self.buffer[self.buffer_idx]
            self.buffer_idx += 1
            self.frame_count += 1

            # If we're at the end of the buffer, read a new frame and add it
            if self.buffer_idx >= len(self.buffer):
                ret, new_frame = self.cap.read()
                if ret:
                    # Remove oldest frame and add new one
                    self.buffer.pop(0)
                    self.buffer.append(new_frame)
                    self.buffer_idx -= 1

            return True, frame
        else:
            # Buffer is empty or exhausted, read directly
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
            return ret, frame

    def get_frame_count(self) -> int:
        """Get the current frame count."""
        return self.frame_count

    def get_position(self) -> float:
        """
        Get the current position in the video as a fraction.

        Returns:
            Position as a fraction (0.0 to 1.0)
        """
        if self.total_frames > 0:
            return self.frame_count / self.total_frames
        return 0.0

    def get_position_ms(self) -> int:
        """
        Get the current position in milliseconds.

        Returns:
            Position in milliseconds
        """
        if self.fps > 0:
            return int(self.frame_count * 1000 / self.fps)
        return 0

    def seek(self, frame_number: int) -> bool:
        """
        Seek to a specific frame.

        Args:
            frame_number: Frame number to seek to

        Returns:
            Success flag
        """
        if isinstance(self.source, int):
            self.logger.warning("Cannot seek in camera input")
            return False

        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.frame_count = frame_number
            self._init_buffer()
            return True
        except Exception as e:
            self.logger.error(f"Error seeking to frame {frame_number}: {e}")
            return False

    def seek_time(self, time_ms: int) -> bool:
        """
        Seek to a specific time.

        Args:
            time_ms: Time in milliseconds

        Returns:
            Success flag
        """
        if isinstance(self.source, int):
            self.logger.warning("Cannot seek in camera input")
            return False

        try:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
            new_frame_count = int(time_ms * self.fps / 1000)
            self.frame_count = new_frame_count
            self._init_buffer()
            return True
        except Exception as e:
            self.logger.error(f"Error seeking to time {time_ms}ms: {e}")
            return False

    def release(self) -> None:
        """Release the video source."""
        if self.cap:
            self.cap.release()
            self.logger.debug("Released video source")

        # Clear buffer
        self.buffer = []
    
    def _extract_gps_metadata(self):
        """Extract GPS metadata from MP4 file."""
        try:
            video_path = str(self.source)
            
            # Try multiple methods to extract GPS data
            self._extract_gps_from_metadata(video_path)
            self._extract_gps_from_subtitle_track(video_path)
            
            if self.gps_metadata:
                self.logger.info(f"Extracted {len(self.gps_metadata)} GPS points from video")
            else:
                self.logger.warning(f"No GPS metadata found in {video_path}")
                
        except Exception as e:
            self.logger.error(f"Error extracting GPS metadata: {e}")
    
    def _extract_gps_from_metadata(self, video_path: str):
        """Extract GPS data from video file metadata."""
        try:
            # Use ffprobe to extract metadata
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                metadata = json.loads(result.stdout)
                
                # Look for GPS data in format tags
                if "format" in metadata and "tags" in metadata["format"]:
                    tags = metadata["format"]["tags"]
                    
                    # Check for various GPS tag formats
                    gps_fields = ["location", "gps", "com.apple.quicktime.location.GPS"]
                    
                    for field in gps_fields:
                        if field in tags:
                            self._parse_gps_string(tags[field])
                
                # Look for GPS data in stream tags
                for stream in metadata.get("streams", []):
                    if "tags" in stream:
                        tags = stream["tags"]
                        
                        for field in gps_fields:
                            if field in tags:
                                self._parse_gps_string(tags[field])
                
        except subprocess.TimeoutExpired:
            self.logger.warning("GPS metadata extraction timed out")
        except FileNotFoundError:
            self.logger.warning("ffprobe not found - GPS extraction disabled")
        except Exception as e:
            self.logger.error(f"Error extracting GPS metadata: {e}")
    
    def _extract_gps_from_subtitle_track(self, video_path: str):
        """Extract GPS data from subtitle track (common in dashcams)."""
        try:
            # Use ffprobe to check for subtitle streams
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-select_streams", "s",
                "-show_entries", "stream=index,codec_name",
                "-of", "json",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                metadata = json.loads(result.stdout)
                
                # Check for subtitle streams
                for stream in metadata.get("streams", []):
                    if stream.get("codec_name") in ["text", "srt", "ass"]:
                        # Extract subtitle content
                        self._extract_subtitle_gps(video_path, stream["index"])
                
        except Exception as e:
            self.logger.error(f"Error extracting GPS from subtitles: {e}")
    
    def _extract_subtitle_gps(self, video_path: str, stream_index: int):
        """Extract GPS from specific subtitle stream."""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-select_streams", f"s:{stream_index}",
                "-show_packets",
                "-show_data",
                "-of", "json",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Parse subtitle packets for GPS data
                for packet in data.get("packets", []):
                    if "data" in packet:
                        # Decode subtitle data and look for GPS patterns
                        subtitle_text = self._decode_subtitle_data(packet["data"])
                        if subtitle_text:
                            self._parse_subtitle_gps(subtitle_text, packet.get("pts_time", 0))
                            
        except Exception as e:
            self.logger.error(f"Error extracting subtitle GPS: {e}")
    
    def _decode_subtitle_data(self, data_hex: str) -> Optional[str]:
        """Decode subtitle data from hex."""
        try:
            # Convert hex string to bytes and decode
            data_bytes = bytes.fromhex(data_hex)
            return data_bytes.decode('utf-8', errors='ignore')
        except Exception:
            return None
    
    def _parse_subtitle_gps(self, subtitle_text: str, timestamp: float):
        """Parse GPS data from subtitle text."""
        import re
        
        # Common dashcam GPS patterns
        patterns = [
            # Pattern: GPS: 37.7749,-122.4194,123.45,65.0,2023-01-01 12:00:00
            r'GPS:\s*([+-]?\d+\.?\d*),([+-]?\d+\.?\d*),?([+-]?\d+\.?\d*)?,?(\d+\.?\d*)?,?(.+)?',
            
            # Pattern: Lat: 37.7749, Lon: -122.4194, Alt: 123.45, Speed: 65.0
            r'Lat:\s*([+-]?\d+\.?\d*),?\s*Lon:\s*([+-]?\d+\.?\d*)',
            
            # Pattern: 37.7749°N 122.4194°W
            r'([+-]?\d+\.?\d*)°[NS]\s+([+-]?\d+\.?\d*)°[EW]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, subtitle_text)
            if match:
                try:
                    groups = match.groups()
                    
                    gps_point = {
                        "timestamp": timestamp,
                        "latitude": float(groups[0]),
                        "longitude": float(groups[1]),
                    }
                    
                    # Add optional fields if available
                    if len(groups) > 2 and groups[2]:
                        gps_point["altitude"] = float(groups[2])
                    if len(groups) > 3 and groups[3]:
                        gps_point["speed"] = float(groups[3])
                    
                    self.gps_metadata.append(gps_point)
                    
                except (ValueError, IndexError) as e:
                    self.logger.debug(f"Error parsing GPS from subtitle: {e}")
                    continue
    
    def _parse_gps_string(self, gps_string: str):
        """Parse various GPS string formats."""
        import re
        
        # Try different GPS string formats
        patterns = [
            # ISO 6709 format: +37.7749-122.4194/
            r'([+-]\d+\.?\d*)([+-]\d+\.?\d*)',
            
            # Decimal degrees: 37.7749,-122.4194
            r'(\d+\.?\d*),?\s*(-?\d+\.?\d*)',
            
            # With altitude: 37.7749,-122.4194,123.45
            r'(\d+\.?\d*),?\s*(-?\d+\.?\d*),?\s*(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, gps_string)
            if match:
                try:
                    groups = match.groups()
                    
                    gps_point = {
                        "timestamp": 0.0,  # Will be interpolated later
                        "latitude": float(groups[0]),
                        "longitude": float(groups[1]),
                    }
                    
                    if len(groups) > 2:
                        gps_point["altitude"] = float(groups[2])
                    
                    # Store as track info (single point for entire video)
                    self.gps_track = gps_point
                    
                except (ValueError, IndexError):
                    continue
    
    def get_gps_data_for_frame(self, frame_number: int) -> Optional[Dict]:
        """
        Get GPS data for a specific frame number.
        
        Args:
            frame_number: Frame number to get GPS data for
            
        Returns:
            GPS data dictionary or None if not available
        """
        if not self.gps_metadata and not self.gps_track:
            return None
        
        # If we have timestamped GPS metadata
        if self.gps_metadata:
            # Calculate time for this frame
            frame_time = frame_number / self.fps if self.fps > 0 else 0
            
            # Find closest GPS point
            closest_point = None
            min_time_diff = float('inf')
            
            for gps_point in self.gps_metadata:
                time_diff = abs(gps_point["timestamp"] - frame_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_point = gps_point
            
            if closest_point and min_time_diff < 5.0:  # Within 5 seconds
                return closest_point.copy()
        
        # Fallback to track-level GPS (single point for entire video)
        if self.gps_track:
            track_copy = self.gps_track.copy()
            track_copy["frame"] = frame_number
            return track_copy
        
        return None
    
    def has_gps_data(self) -> bool:
        """Check if GPS data is available."""
        return bool(self.gps_metadata or self.gps_track)
    
    def get_gps_metadata_summary(self) -> Dict:
        """Get summary of GPS metadata."""
        return {
            "has_gps_data": self.has_gps_data(),
            "gps_points_count": len(self.gps_metadata),
            "has_track_info": bool(self.gps_track),
            "gps_metadata": self.gps_metadata[:5] if self.gps_metadata else [],  # First 5 points
            "track_info": self.gps_track if self.gps_track else None,
        }
