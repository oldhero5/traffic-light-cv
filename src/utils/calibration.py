"""
Camera calibration utilities.
"""

from __future__ import annotations

import json
import logging
import os

import cv2
import numpy as np
from tqdm import tqdm


class CameraCalibrator:
    """Class for calibrating cameras."""

    def __init__(
        self,
        checkerboard_size: tuple[int, int] = (9, 6),
        square_size: float = 0.025,  # in meters
    ):
        """
        Initialize the camera calibrator.

        Args:
            checkerboard_size: Size of the checkerboard in (columns, rows)
            square_size: Size of each square in meters
        """
        self.logger = logging.getLogger(__name__)
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size

        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0 : checkerboard_size[0], 0 : checkerboard_size[1]].T.reshape(
            -1, 2
        )
        self.objp *= square_size  # Scale to real world size

        # Arrays to store object points and image points
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane

        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.reprojection_error = None

        self.logger.info("Camera calibrator initialized")

    def find_corners(self, image: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Find checkerboard corners in an image.

        Args:
            image: Input image

        Returns:
            Tuple of (success, corners)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        return ret, corners

    def add_calibration_image(self, image: np.ndarray) -> bool:
        """
        Add an image for calibration.

        Args:
            image: Input image

        Returns:
            Success flag
        """
        ret, corners = self.find_corners(image)

        if ret:
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners)
            return True
        else:
            return False

    def calibrate(self, image_shape: tuple[int, int] | None = None) -> bool:
        """
        Perform camera calibration.

        Args:
            image_shape: Image shape (height, width) for calibration

        Returns:
            Success flag
        """
        if not self.objpoints or not self.imgpoints:
            self.logger.error("No calibration images added")
            return False

        if image_shape is None:
            # Use last image shape
            h, w = 0, 0
            for corners in self.imgpoints:
                if corners.shape[0] > 0:
                    h, w = int(np.max(corners[:, :, 1])) + 100, int(np.max(corners[:, :, 0])) + 100
            image_shape = (h, w)

        self.logger.info(f"Calibrating camera with {len(self.objpoints)} images")

        # Perform calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_shape, None, None
        )

        if ret:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.rvecs = rvecs
            self.tvecs = tvecs

            # Calculate reprojection error
            total_error = 0
            for i in range(len(self.objpoints)):
                imgpoints2, _ = cv2.projectPoints(
                    self.objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                )
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error

            self.reprojection_error = total_error / len(self.objpoints)

            self.logger.info(
                f"Calibration successful, reprojection error: {self.reprojection_error}"
            )
            return True
        else:
            self.logger.error("Calibration failed")
            return False

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """
        Undistort an image using calibration results.

        Args:
            image: Input image

        Returns:
            Undistorted image
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.logger.warning("Camera not calibrated, returning original image")
            return image

        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)

    def save_calibration(self, filepath: str) -> bool:
        """
        Save calibration results to a file.

        Args:
            filepath: Path to save calibration

        Returns:
            Success flag
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.logger.error("Camera not calibrated")
            return False

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert to lists for JSON serialization
        calibration_data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "reprojection_error": float(self.reprojection_error),
            "image_count": len(self.objpoints),
            "checkerboard_size": self.checkerboard_size,
            "square_size": self.square_size,
        }

        try:
            with open(filepath, "w") as f:
                json.dump(calibration_data, f, indent=2)

            self.logger.info(f"Calibration saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving calibration: {e}")
            return False

    def load_calibration(self, filepath: str) -> bool:
        """
        Load calibration results from a file.

        Args:
            filepath: Path to load calibration from

        Returns:
            Success flag
        """
        if not os.path.exists(filepath):
            self.logger.error(f"Calibration file not found: {filepath}")
            return False

        try:
            with open(filepath) as f:
                calibration_data = json.load(f)

            self.camera_matrix = np.array(calibration_data["camera_matrix"])
            self.dist_coeffs = np.array(calibration_data["dist_coeffs"])
            self.reprojection_error = calibration_data["reprojection_error"]

            # Optional parameters
            if "checkerboard_size" in calibration_data:
                self.checkerboard_size = tuple(calibration_data["checkerboard_size"])
            if "square_size" in calibration_data:
                self.square_size = calibration_data["square_size"]

            self.logger.info(f"Calibration loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading calibration: {e}")
            return False


def calibrate_from_images(
    image_dir: str,
    output_file: str,
    checkerboard_size: tuple[int, int] = (9, 6),
    square_size: float = 0.025,  # in meters
) -> bool:
    """
    Calibrate camera from a directory of checkerboard images.

    Args:
        image_dir: Directory containing checkerboard images
        output_file: Path to save calibration results
        checkerboard_size: Size of the checkerboard in (columns, rows)
        square_size: Size of each square in meters

    Returns:
        Success flag
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Calibrating camera from images in {image_dir}")

    # Create calibrator
    calibrator = CameraCalibrator(checkerboard_size, square_size)

    # Get image files
    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        logger.error(f"No images found in {image_dir}")
        return False

    logger.info(f"Found {len(image_files)} images")

    # Add images for calibration
    image_shape = None
    for image_file in tqdm(image_files, desc="Processing images"):
        image = cv2.imread(image_file)

        if image is None:
            logger.warning(f"Failed to load image: {image_file}")
            continue

        if image_shape is None:
            image_shape = image.shape[:2]

        success = calibrator.add_calibration_image(image)

        if not success:
            logger.warning(f"Failed to find corners in {image_file}")

    # Perform calibration
    if not calibrator.calibrate(image_shape):
        logger.error("Calibration failed")
        return False

    # Save calibration
    if not calibrator.save_calibration(output_file):
        logger.error(f"Failed to save calibration to {output_file}")
        return False

    logger.info(f"Calibration completed and saved to {output_file}")
    return True


def undistort_images(
    image_dir: str,
    output_dir: str,
    calibration_file: str,
) -> bool:
    """
    Undistort images using a camera calibration.

    Args:
        image_dir: Directory containing images to undistort
        output_dir: Directory to save undistorted images
        calibration_file: Path to calibration file

    Returns:
        Success flag
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Undistorting images from {image_dir} to {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load calibration
    calibrator = CameraCalibrator()

    if not calibrator.load_calibration(calibration_file):
        logger.error(f"Failed to load calibration from {calibration_file}")
        return False

    # Get image files
    image_files = [
        f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        logger.error(f"No images found in {image_dir}")
        return False

    logger.info(f"Found {len(image_files)} images")

    # Undistort images
    for image_file in tqdm(image_files, desc="Undistorting images"):
        input_path = os.path.join(image_dir, image_file)
        output_path = os.path.join(output_dir, image_file)

        image = cv2.imread(input_path)

        if image is None:
            logger.warning(f"Failed to load image: {input_path}")
            continue

        undistorted = calibrator.undistort(image)

        cv2.imwrite(output_path, undistorted)

    logger.info(f"Undistorted {len(image_files)} images to {output_dir}")
    return True


def estimate_camera_params(
    image_width: int,
    image_height: int,
    fov_horizontal: float | None = None,
    fov_vertical: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate camera matrix and distortion coefficients from image size and FOV.

    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        fov_horizontal: Horizontal field of view in degrees
        fov_vertical: Vertical field of view in degrees

    Returns:
        Tuple of (camera_matrix, dist_coeffs)
    """
    # Default FOV values
    if fov_horizontal is None:
        fov_horizontal = 60.0  # degrees
    if fov_vertical is None:
        # Calculate vertical FOV from aspect ratio and horizontal FOV
        aspect_ratio = image_height / image_width
        fov_vertical = fov_horizontal * aspect_ratio

    # Calculate focal length from FOV
    fx = (image_width / 2) / np.tan(np.radians(fov_horizontal / 2))
    fy = (image_height / 2) / np.tan(np.radians(fov_vertical / 2))

    # Create camera matrix
    camera_matrix = np.array([[fx, 0, image_width / 2], [0, fy, image_height / 2], [0, 0, 1]])

    # Assume no distortion
    dist_coeffs = np.zeros(5)

    return camera_matrix, dist_coeffs
