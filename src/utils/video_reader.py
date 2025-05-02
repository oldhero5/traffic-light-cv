"""
Video reader module.
"""
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


class VideoReader:
    """Class for reading video files or camera streams."""
    
    def __init__(
        self,
        source: Union[str, int, Path],
        buffer_size: int = 5,
    ):
        """
        Initialize the video reader.
        
        Args:
            source: Video source (file path or camera index)
            buffer_size: Size of the frame buffer
        """
        self.logger = logging.getLogger(__name__)
        self.source = source
        self.buffer_size = buffer_size
        self.cap = None
        self.frame_count = 0
        
        # Frame buffer for smoother processing
        self.buffer = []
        self.buffer_idx = 0
        
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
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
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