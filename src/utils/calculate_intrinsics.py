import cv2
import numpy as np
from scipy.optimize import least_squares

def optimize_intrinsics(left_img, right_img, init_params=None, num_matches=1000, reg_weight=3e-2):
    """
    Optimizes the intrinsic parameters of a camera by minimizing the vertical disparity
    between matched keypoints in a stereo pair. Assumes both cameras share the same intrinsics.
    
    Parameters:
      left_img (np.array): Grayscale left image.
      right_img (np.array): Grayscale right image.
      init_params (np.array): Optional initial guess [fx, fy, cx, cy]. If None, a default is used.
      num_matches (int): Number of top matches to use for the optimization.
    
    Returns:
      dict: Optimized intrinsic parameters as a dictionary with keys 'fx', 'fy', 'cx', 'cy'.
    """
    
    # 1. Detect and compute ORB features.
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(left_img, None)
    kp2, des2 = orb.detectAndCompute(right_img, None)
    
    # 2. Match features using a brute-force matcher.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Use only the best matches to keep the problem manageable.
    num_matches = min(num_matches, len(matches))
    pts_left = np.array([kp1[m.queryIdx].pt for m in matches[:num_matches]])
    pts_right = np.array([kp2[m.trainIdx].pt for m in matches[:num_matches]])
    
    # 3. If no initial parameters provided, use a default guess:
    H, W = left_img.shape[:2]
    if init_params is None:
        # A rough guess: focal lengths around the image width, and the principal point at the center.
        fx = fy = float(W)
        cx = W / 2.0
        cy = H / 2.0
        init_params = np.array([fx, fy, cx, cy], dtype=np.float64)
    
    # 4. Define a cost function:
    # We assume that once corrected, corresponding points (in normalized coordinates) should
    # have the same y value. We compute the normalized y coordinates as (v - cy) / fy.
    def residual(params):
        fx, fy, cx, cy = params
        # Compute normalized y-coordinates using current candidate parameters.
        # Note: the difference simplifies to (pts_left_y - pts_right_y) / fy, but we keep the form
        # to allow the regularization on cy to be meaningful.
        norm_left_y  = (pts_left[:, 1] - cy) / fy
        norm_right_y = (pts_right[:, 1] - cy) / fy
        vertical_error = norm_left_y - norm_right_y
        
        # Regularization residuals: penalize deviation from initial fy and cy.
        # (You can add regularization for fx and cx too if needed.)
        reg_fy = reg_weight * (fy - init_params[1])
        reg_cy = reg_weight * (cy - init_params[3])
        return np.concatenate([vertical_error, [reg_fy, reg_cy]])

    # 5. Run the optimization using Levenberg-Marquardt (lm).
    result = least_squares(residual, init_params, method='lm')
    optimized_params = result.x

    return {
        'fx': optimized_params[0],
        'fy': optimized_params[1],
        'cx': optimized_params[2],
        'cy': optimized_params[3]
    }


