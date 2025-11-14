import bagpy
import os
import cv2
import pandas as pd
import numpy as np
import time
from bagpy import bagreader
import matplotlib.pyplot as plt

def draw_horizontal_lines(image, num_lines=10):
    height, width = image.shape[:2]
    interval = height // num_lines
    for i in range(0, height, interval):
        cv2.line(image, (0, i), (width, i), (0, 255, 0), 1)
    return image


def split_img_and_rectify(full_img, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, baseline, rotation_vector, visualize=False):
    height, width = full_img.shape[:2]
    half_width = width // 2
    # Split into left and right images
    left_img = full_img[:, :half_width]  # Left side
    right_img = full_img[:, half_width:]  # Right side
    print(left_img.shape)
    # cv2.imshow("full img", full_img)
    # cv2.waitKey(0)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)  # Convert to rotation matrix
    translation_vector = np.array([baseline, 0, 0]) / 1000  # Convert baseline to meters

    image_size = (half_width, height)

    R1, R2, P1, P2, Q, valid_roi1, valid_roi2 = cv2.stereoRectify(
        np.array(left_camera_matrix), np.array(left_dist_coeffs),
        np.array(right_camera_matrix), np.array(right_dist_coeffs),
        image_size, rotation_matrix, translation_vector
    )

    # Compute rectification maps
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        np.array(left_camera_matrix), np.array(left_dist_coeffs), R1, P1, image_size, cv2.CV_16SC2
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        np.array(right_camera_matrix), np.array(right_dist_coeffs), R2, P2, image_size, cv2.CV_16SC2
    )

    # Apply rectification to images
    rectified_left = cv2.remap(left_img, map1_left, map2_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_img, map1_right, map2_right, cv2.INTER_LINEAR)

    if visualize:
        for row in range(0, image_size[1], 40):  # step size for lines
            cv2.line(rectified_left, (0, row), (image_size[0], row), (0, 255, 0), 1)
            cv2.line(rectified_right, (0, row), (image_size[0], row), (0, 255, 0), 1)
            cv2.imshow("Rectified Left", rectified_left)
            cv2.imshow("Rectified Right", rectified_right)
            cv2.waitKey(0)

    new_camera_matrix_left = P1[:, :3]
    new_camera_matrix_right = P2[:, :3]
    new_stereo_baseline = P2[0,3] / P2[0,0]
    P2_flipped_z = P2.copy()
    P2_flipped_z[:, 3] *= -1    #flip z-coordinate to enable later Triangulation

    # rectified_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    # rectified_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
    return rectified_left, rectified_right, left_img, right_img, new_camera_matrix_left, new_camera_matrix_right, new_stereo_baseline, P1, P2_flipped_z

def rotationMatrixToEulerAngles(R):
    """
    Converts a rotation matrix to Euler angles (in degrees) with the rotation order: Rz * Ry * Rx.
    This implementation handles the non-singular case.
    """
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        # Gimbal lock: assign zero to z.
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)
def adjust_stereo_rotation(left_img, right_img, fx, fy, cx, cy, initial_guess):
    """
    Displays a window with trackbars to manually adjust the rotation (Rx, Ry, Rz)
    for the right image and overlays it with the left image.
    
    Parameters:
      left_img: The left image as a NumPy array.
      right_img: The right image as a NumPy array.
      focal_length: (Optional) Focal length in pixels. If None, it defaults to max(width, height).
    """
    height, width = left_img.shape[:2]
    
    # Build the intrinsic camera matrix.
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    def eulerAnglesToRotationMatrix(theta):
        """
        Convert Euler angles (in degrees) to a 3x3 rotation matrix.
        Rotation order is Rz * Ry * Rx.
        """
        # Convert angles to radians.
        theta = np.radians(theta)
        rx, ry, rz = theta
        
        # Rotation matrix around x-axis.
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        
        # Rotation matrix around y-axis.
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        
        # Rotation matrix around z-axis.
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        
        # Combined rotation.
        R = Rz.dot(Ry).dot(Rx)
        return R
    
    # If an initial guess is provided in Rodrigues format, convert it to Euler angles.
    if initial_guess is not None:
        rvec = np.array(initial_guess, dtype=np.float64)
        R_init, _ = cv2.Rodrigues(rvec)
        init_euler = rotationMatrixToEulerAngles(R_init)
    else:
        init_euler = (0.0, 0.0, 0.0)  # Default Euler angles if no guess is provided.
    # Create a window with trackbars for Rx, Ry, and Rz.
    window_name = "Stereo Alignment"
    cv2.namedWindow(window_name)
    
    # Use trackbars with a range of 0-360, where 180 corresponds to 0°.
    init_rx = int(init_euler[0] + 180)
    init_ry = int(init_euler[1] + 180)
    init_rz = int(init_euler[2] + 180)
    cv2.createTrackbar("Rx", window_name, init_rx, 360, lambda x: None)
    cv2.createTrackbar("Ry", window_name, init_ry, 360, lambda x: None)
    cv2.createTrackbar("Rz", window_name, init_rz, 360, lambda x: None)

    while True:
        # Get current rotation values (offset trackbar values to be in -180..180).
        rx = cv2.getTrackbarPos("Rx", window_name) - 180
        ry = cv2.getTrackbarPos("Ry", window_name) - 180
        rz = cv2.getTrackbarPos("Rz", window_name) - 180
        
        # Compute the rotation matrix from Euler angles.
        R = eulerAnglesToRotationMatrix([rx, ry, rz])
        
        # Compute the homography: H = K * R * K_inv.
        H = K.dot(R).dot(np.linalg.inv(K))
        
        # Warp the right image using the computed homography.
        warped_right = cv2.warpPerspective(right_img, H, (width, height))
        
        # Blend the left image and the warped right image.
        blended = cv2.addWeighted(left_img, 0.5, warped_right, 0.5, 0)
        
        # Display the blended image.
        cv2.imshow(window_name, blended)
        
        # Exit on pressing the Escape key.
        if cv2.waitKey(50) & 0xFF == 27:
            break

    cv2.destroyAllWindows()