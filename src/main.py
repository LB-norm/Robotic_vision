import bagpy
import os
import cv2
import pandas as pd
import numpy as np
import time
from bagpy import bagreader
import matplotlib.pyplot as plt
from utils import rectify
from utils import apriltag
from utils import visualization
from utils import Kalman
from utils import utils
from utils import calculate_intrinsics
from utils import Pose_Graph_Optimizer
from pyapriltags import Detector
import pickle

# Camera parameters 
left_camera_matrix = np.array([
    [350.7575, 0, 309.8225],  # fx, 0, cx
    [0, 350.7575, 195.288],  # 0, fy, cy     #695.018047141521 calculated fy
    [0, 0, 1]               # 0, 0, 1
    ])

left_dist_coeffs = np.array([-0.171785, 0.0262926,    #k1, k2, p1, p2, k3
                        -0.00134766, 5.72245e-05, 0]) 
    
right_camera_matrix = np.array([
        [349.1775, 0, 304.5125],  # fx, 0, cx
        [0, 349.1775, 193.0495],  # 0, fy, cy     #694.7714932310313 calculated fy  
        [0, 0, 1]               # 0, 0, 1
    ])

right_dist_coeffs = np.array([-0.172021, 0.026095,   #k1, k2, p1, p2, k3
                        -0.00106339, 0.000154453, 0])

baseline = 63.0015  # mm

# rotation_vector = np.array([0.000940106, -0.00694236, -0.0005005])   # RX, CV, RZ ORIGINAL
# rotation_vector = np.array([-0.0100106, +0.003236, -0.0015005])    # RX, CV, RZ TOM
# rotation_vector = np.array([-0.0175, -0.15, -0.01])/2   # RX, CV, RZ  best conf
rotation_vector = np.array([-0.000791412, 0.00154456, -0.000217368])

def extract_rvec_tvec_stacked(pose):
    R_extracted = pose[:3, :3]
    t_extracted = pose[:3, 3]

    # Convert the rotation matrix back to the Rodrigues vector
    rvec_extracted, _ = cv2.Rodrigues(R_extracted)
    rvec = rvec_extracted.reshape(3,1)
    tvec = t_extracted.reshape(3,1)
    curr_cam_pose = np.vstack((tvec, rvec))
    return curr_cam_pose
def rvec_tvec_to_homogeneous(rvec, tvec):
    """Convert (rvec, tvec) to a 4x4 homogeneous transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def homogeneous_to_rvec_tvec(T):
    """Convert a 4x4 homogeneous transformation matrix back to (rvec, tvec)."""
    R = T[:3, :3]
    tvec = T[:3, 3].reshape((3, 1))
    rvec, _ = cv2.Rodrigues(R)
    return rvec, tvec

def combine_pose_sets(pose_sets):
    """
    Combine multiple sets of poses into one list of (rvec, tvec).
    
    Parameters:
        pose_sets (list): A list of pose sets, where each pose set is a list of (rvec, tvec)
                          tuples. Each (rvec, tvec) represents a pose in the set's local coordinate
                          system.
    
    Returns:
        list: A combined list of (rvec, tvec) poses in the global coordinate system.
    """
    combined = []
    T_cum = np.eye(4)  # cumulative transformation; starts as identity
    
    for pose_set in pose_sets:
        if not pose_set:
            continue
        
        # Process each pose in the current set
        for rvec, tvec in pose_set:
            T_local = rvec_tvec_to_homogeneous(rvec, tvec)
            # Map the local pose into the global coordinate system
            T_global = np.dot(T_cum, T_local)
            global_rvec, global_tvec = homogeneous_to_rvec_tvec(T_global)
            combined.append((global_rvec, global_tvec))
        
        # Update the cumulative transform with the last pose of the current set.
        last_rvec, last_tvec = pose_set[-1]
        T_last = rvec_tvec_to_homogeneous(last_rvec, last_tvec)
        T_cum = np.dot(T_cum, T_last)
    
    return combined
def extract_and_match_features(left_img, right_img, orb, y_threshold=1.0):
    """
    Extracts ORB features from the left and right images and matches them.
    
    Args:
        left_img (np.ndarray): The left rectified stereo image.
        right_img (np.ndarray): The right rectified stereo image.
        orb (object): Pre-initialized ORB
        y_threshold (float): Maximum allowed vertical difference (in pixels) between matched keypoints.
    
    Returns:
        keypoints_left (list): Keypoints detected in the left image.
        keypoints_right (list): Keypoints detected in the right image.
        good_matches (list): Filtered list of cv2.DMatch objects representing good matches.
    """

    # Detect keypoints and compute descriptors for both images
    keypoints_left, descriptors_left = orb.detectAndCompute(left_img, None)
    keypoints_right, descriptors_right = orb.detectAndCompute(right_img, None)
    print(f"Keypoints_left: {len(keypoints_left)} \n Keypoints_right: {len(keypoints_right)}")
    # # Create a Brute-Force Matcher for binary descriptors using Hamming distance
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(descriptors_left, descriptors_right)
    
    # # Sort matches based on descriptor distance (lower distance is better)
    # matches = sorted(matches, key=lambda m: m.distance)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors_left, descriptors_right, k=2)
    #test
    filtered_matches = []
    for m, n in matches:
        y_l = keypoints_left[m.queryIdx].pt[1]
        y_r = keypoints_right[m.trainIdx].pt[1]
        if abs(y_l - y_r) < 1:
            filtered_matches.append(m)

    print(f"Matches before y‑filtering: {len(matches)}")
    print(f"Matches after  y‑filtering: {len(filtered_matches)}")
    print(f"Matches: {len(matches)}")
    # Apply the ratio test
    good_matches_map = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches_map.append(m)
    # Filter matches using the epipolar constraint:
    # Since the images are rectified, corresponding points should lie on nearly the same row.
    good_matches = []
    for m in good_matches_map:
        pt_left = keypoints_left[m.queryIdx].pt
        pt_right = keypoints_right[m.trainIdx].pt
        

        if abs(pt_left[1] - pt_right[1]) < y_threshold:
            good_matches.append(m)
    print(f"Good Matches: {len(good_matches)}")
    return keypoints_left, keypoints_right, good_matches, filtered_matches, descriptors_left, descriptors_right

def triangulate_points(
    keypoints_left,
    keypoints_right,
    good_matches,
    P_left,
    P_right,
    w_threshold=0,
    z_threshold=0,
    max_reprojection_error=1.0
):
    """
    Triangulates 3D points from a stereo image pair using matched keypoints
    and applies basic filtering to remove invalid/outlier points.

    Args:
        keypoints_left (list of cv2.KeyPoint): Keypoints from the left image.
        keypoints_right (list of cv2.KeyPoint): Keypoints from the right image.
        good_matches (list of cv2.DMatch): Matches indicating which keypoints correspond.
        P_left (np.ndarray): 3x4 projection matrix for the left camera.
        P_right (np.ndarray): 3x4 projection matrix for the right camera.
        w_threshold (float): Minimum absolute value for the homogeneous w-coordinate.
        z_threshold (float): Minimum z-value in 3D space to accept a point (removes points behind camera).
        max_reprojection_error (float): Maximum allowed reprojection error in pixels for inlier points.

    Returns:
        pts_3d_inliers (np.ndarray): Nx3 array of valid 3D points after filtering.
        inlier_matches (list of cv2.DMatch): Filtered list of matches corresponding to pts_3d_inliers.
    """
    # Collect corresponding 2D points for left and right images
    pts_left = []
    pts_right = []
    for m in good_matches:
        pts_left.append(keypoints_left[m.queryIdx].pt)
        pts_right.append(keypoints_right[m.trainIdx].pt)

    # Convert to NumPy arrays and transpose (shape: 2 x N)
    pts_left = np.array(pts_left).T
    pts_right = np.array(pts_right).T


    # Triangulate in homogeneous coordinates (shape: 4 x N)
    pts_4d_hom = cv2.triangulatePoints(P_left, P_right, pts_left, pts_right)
    
    # Normalize by w to get Cartesian coordinates (shape: 3 x N after slicing)
    w = pts_4d_hom[3, :]
   
    # Avoid division by zero or extremely small w
    valid_w_mask = np.abs(w) > w_threshold
    pts_4d_hom = pts_4d_hom[:, valid_w_mask]  # Keep only valid
    w = w[valid_w_mask]

    pts_3d = pts_4d_hom[:3, :] / w  # Now shape is 3 x M
    pts_3d = pts_3d.T               # Convert to M x 3

    # Keep track of which matches survived the w-filter
    # (we removed some columns from pts_4d_hom)
    valid_indices = np.where(valid_w_mask)[0]
    filtered_matches_after_w = [good_matches[i] for i in valid_indices]

    # Filter out points with negative or too-small Z
    z_mask = pts_3d[:, 2] > z_threshold
    pts_3d = pts_3d[z_mask]
    filtered_matches_after_z = [
        filtered_matches_after_w[i] for i, valid_z in enumerate(z_mask) if valid_z
    ]

    # Reprojection error check on the left camera
    # We will reproject each 3D point back to 2D and compare with the original left keypoint.
    final_pts_3d = []
    final_matches = []

    # Prepare original 2D points that survived the w + z filters
    left_points_filtered = []
    for m in filtered_matches_after_z:
        left_points_filtered.append(keypoints_left[m.queryIdx].pt)

    # We can do a simple manual projection: x_proj = P_left @ [X, 1]
    # Then x_proj_2d = (x_proj[0]/x_proj[2], x_proj[1]/x_proj[2])
    for i, (X, match) in enumerate(zip(pts_3d, filtered_matches_after_z)):
        # Homogeneous 3D point
        X_hom = np.array([X[0], X[1], X[2], 1.0])
        # Project to 2D
        proj_2d = P_left @ X_hom
        if np.abs(proj_2d[2]) < 1e-12:
            # Avoid dividing by near-zero z
            continue
        x_proj = proj_2d[0] / proj_2d[2]
        y_proj = proj_2d[1] / proj_2d[2]

        # Original left keypoint
        x_left, y_left = left_points_filtered[i]
        # Compute reprojection error
        error = np.sqrt((x_proj - x_left)**2 + (y_proj - y_left)**2)

        if error < max_reprojection_error:
            final_pts_3d.append(X)
            final_matches.append(match)

    pts_3d_inliers = np.array(final_pts_3d)
    inlier_matches = final_matches

    return pts_3d_inliers, inlier_matches

def clamp_pose_change(T_prev, T_est, max_trans, max_rot_deg):
    """
    Clamps the relative pose from T_prev to T_est
    so that translation does not exceed max_trans (meters)
    and rotation does not exceed max_rot_deg (degrees).
    
    Returns the clamped T_new (4x4) pose.
    """
    # 1) Relative transform: Delta T
    T_rel = np.linalg.inv(T_prev) @ T_est
    
    # 2) Extract translation
    t_rel = T_rel[0:3, 3]
    dist = np.linalg.norm(t_rel)
    
    # 3) Extract rotation (angle + axis)
    R_rel = T_rel[0:3, 0:3]
    # --- compute rotation angle
    trace_val = np.trace(R_rel)
    # clamp trace to valid range for numerical stability
    trace_val = max(min(trace_val, 3.0), -1.0)
    angle_rad = np.arccos((trace_val - 1.0) / 2.0)
    angle_deg = angle_rad * 180.0 / np.pi
    
    # --- find rotation axis (Rodrigues formula)
    # axis is extracted from off-diagonal elements
    # but handle small angles carefully
    if abs(angle_rad) < 1e-12:
        axis = np.array([1.0, 0.0, 0.0])  # arbitrary
    else:
        rx = R_rel[2,1] - R_rel[1,2]
        ry = R_rel[0,2] - R_rel[2,0]
        rz = R_rel[1,0] - R_rel[0,1]
        axis = np.array([rx, ry, rz])
        axis = axis / np.linalg.norm(axis)
    
    # 4) Clamp translation distance
    if dist > max_trans:
        dist_clamped = max_trans
    else:
        dist_clamped = dist
    
    # 5) Clamp rotation angle
    if angle_deg > max_rot_deg:
        angle_deg_clamped = max_rot_deg
    else:
        angle_deg_clamped = angle_deg
    
    angle_rad_clamped = angle_deg_clamped * np.pi / 180.0
    
    # 6) Reconstruct rotation with the clamped angle
    R_clamped = rodrigues_axis_angle(axis, angle_rad_clamped)
    
    # 7) Reconstruct translation with clamped distance
    if dist < 1e-12:
        t_clamped = t_rel  # near zero distance
    else:
        t_clamped = (t_rel / dist) * dist_clamped
    
    # 8) Build the clamped relative transform T_rel_clamped
    T_rel_clamped = np.eye(4)
    T_rel_clamped[0:3, 0:3] = R_clamped
    T_rel_clamped[0:3, 3] = t_clamped
    
    # 9) Final new pose in world frame
    T_new = T_prev @ T_rel_clamped
    return T_new


def rodrigues_axis_angle(axis, angle):
    """
    Constructs a 3x3 rotation matrix from a unit rotation axis
    and a rotation angle (radians).
    """
    if angle < 1e-12:
        return np.eye(3)  # near zero rotation
    ux, uy, uz = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    
    R = np.array([
        [c + ux*ux*C,    ux*uy*C - uz*s, ux*uz*C + uy*s],
        [uy*ux*C + uz*s, c + uy*uy*C,    uy*uz*C - ux*s],
        [uz*ux*C - uy*s, uz*uy*C + ux*s, c + uz*uz*C   ]
    ])
    return R

class Landmark:
    def __init__(self, landmark_id, position, descriptor):
        """
        A simple class for storing a 3D landmark.
        
        Args:
            landmark_id (int): A unique identifier for the landmark.
            position (np.ndarray): 3D position (x, y, z) of the landmark.
            descriptor (np.ndarray): The ORB descriptor corresponding to the feature.
        """
        self.id = landmark_id
        self.position = position
        self.descriptor = descriptor
        self.observations = []  # List to store observations: (frame_index, keypoint)

def process_frame(frame_index, left_img, right_img, global_map, prev_pose, P_left, P_right, K, orb, y_threshold=1.0, z_treshold=0, w_threshold=1e-6, inlier_treshold=30 ):
    """
    Process a new stereo frame to update the camera pose and the global map.
    
    Args:
        frame_index (int): Current frame index.
        left_img (np.ndarray): Current left rectified image.
        right_img (np.ndarray): Current right rectified image.
        global_map (dict): Global map of landmarks (landmark_id -> Landmark object).
        prev_pose (np.ndarray): Previous camera pose as a 4x4 homogeneous matrix.
        P_left (np.ndarray): 3x4 projection matrix for the left camera.
        P_right (np.ndarray): 3x4 projection matrix for the right camera.
        K (np.ndarray): Camera rectification matrix 3x3.
        orb (cv2.ORB): Pre-initialized ORB detector.
        y_threshold (float): Threshold for filtering stereo matches by vertical alignment.
        
    Returns:
        curr_pose_hom (np.ndarray): Estimated 4x4 homogeneous pose for the current frame.
        global_map (dict): Updated global map.
    """
    # 1. Detect features in the left image.     #Funktioniert!
    keypoints_left, descriptors_left = orb.detectAndCompute(left_img, None)
    print(f"[Frame {frame_index}] Detected {len(keypoints_left)} keypoints in left image.")
    # 2. Match current left image descriptors with global map descriptors.
    global_descriptors = []
    landmark_ids = []
    for lm_id, landmark in global_map.items():
        global_descriptors.append(landmark.descriptor)
        landmark_ids.append(lm_id)
    
    observed_indices = set()  # will track which keypoints are associated to existing landmarks

    pts_2d_list = []     # for storing 2D pixel coords
    pts_3d_list = []     # for storing the matched 3D coords from the map
    match_indices = []   # to keep track of which BFMatcher match these came from

    if len(global_descriptors) > 0:
        # global_descriptors = np.array(global_descriptors)
        train_mat = np.vstack(global_descriptors).astype(np.uint8)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(descriptors_left, train_mat, k=2)
        # Apply the ratio test
        good_matches_map = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches_map.append(m)
        # print(f"[Frame {frame_index}] Matches against global map: {len(matches_map)}")
        # matches_map = bf.match(descriptors_left, global_descriptors)
        # match_distances = [m.distance for m in matches_map]
        # mean_distance = np.mean(match_distances)
        # print(f"Match distances:", f"mean:{np.mean(match_distances)}, min:{np.min(match_distances)}, max{np.max(match_distances)}")
        # treshold = mean_distance * 2
        # good_matches_map = [m for m in matches_map if m.distance < treshold]
        # Build correspondences
        for i, m in enumerate(good_matches_map):
            kp = keypoints_left[m.queryIdx]
            lm_id = landmark_ids[m.trainIdx]
            pts_2d_list.append(kp.pt)
            pts_3d_list.append(global_map[lm_id].position)
            match_indices.append(i)  # store index i for later reference

            observed_indices.add(m.queryIdx)

    # Convert to NumPy arrays
    pts_2d_array = np.array(pts_2d_list, dtype=np.float32)
    pts_3d_array = np.array(pts_3d_list, dtype=np.float32)
    print(f"[Frame {frame_index}] 2D–3D correspondences assembled: {len(pts_2d_list)}")

    # 3. Pose Estimation using PnP (if enough correspondences are available).
    if len(pts_2d_array) >= 6:
            # Try to solve PnP
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d_array,
            pts_2d_array,
            K,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            # You can tune these RANSAC parameters as well:
            reprojectionError=1.0,  #Evt. weniger 1 oder 0.5
            confidence=0.99,        #Probability for usefull result
            iterationsCount=1000
        )
        print(f"[Frame {frame_index}] PnP success: {retval}. Inliers: {len(inliers) if inliers is not None else 0}")
        n_inliers = len(inliers)
        print("[PnP] tvec:", tvec.ravel())
        if n_inliers > inlier_treshold:        
            # Create 4x4 homogeneous
            curr_pose_hom = np.eye(4)

            # --- Now filter the correspondences to inliers only ---
            inliers = inliers.flatten()  # Usually shape is (N,1), flatten to (N,)

            # Extract the inlier subsets
            pts_2d_inliers = pts_2d_array[inliers]
            pts_3d_inliers = pts_3d_array[inliers]

            # further refinement
            retval2, rvec_refined, tvec_refined = cv2.solvePnP(
                pts_3d_inliers,
                pts_2d_inliers,
                K,
                distCoeffs=None,
                rvec=rvec,              # <-- from the first solvePnP call
                tvec=tvec,              # <-- from the first solvePnP call
                useExtrinsicGuess=True, # crucial for refinement
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            print("[PnP Refinement] tvec:", tvec_refined.ravel())
            # Convert the rotation vector to a rotation matrix
            R_world_to_cam, _ = cv2.Rodrigues(rvec_refined)
            t_world_to_cam = tvec_refined.reshape(-1)

            # Invert the pose if you want a camera-to-world transform
            R_cam_to_world = R_world_to_cam.T
            t_cam_to_world = -R_world_to_cam.T @ t_world_to_cam
            # Invert the pose if you want a camera-to-world transform

            curr_pose_hom[:3, :3] = R_cam_to_world
            curr_pose_hom[:3, 3]  = t_cam_to_world

            curr_pose_hom = clamp_pose_change(prev_pose.copy(), curr_pose_hom, max_trans=0.1, max_rot_deg=40)
            # Then build your final curr_pose_hom from those refined rvec/tvec

        else:
            # RANSAC failed to find a consistent solution, fallback to prev pose
            curr_pose_hom = prev_pose.copy()
    else:
        # Not enough correspondences
        curr_pose_hom = prev_pose.copy()
    
    # Update existing landmarks with current good (inlier) observations
    if retval:
        print(f"[Frame {frame_index}] Updating landmarks with {len(inliers)} inlier observations.")
        inlier_mask = np.zeros(len(pts_2d_array), dtype=bool)
        inlier_mask[inliers] = True

        for i, m in enumerate(good_matches_map):
            if inlier_mask[i]:
                lm_id = landmark_ids[m.trainIdx]
                kp_idx = m.queryIdx
                global_map[lm_id].observations.append((frame_index, keypoints_left[kp_idx]))

    # 4. Stereo matching on the current pair to triangulate new landmarks.
    keypoints_right, descriptors_right = orb.detectAndCompute(right_img, None)

    bf_stereo = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf_stereo.knnMatch(descriptors_left, descriptors_right, k=2)
    # Apply the ratio test
    good_matches_map2 = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good_matches_map2.append(m)
    good_matches = []
    pts_left_stereo = []
    pts_right_stereo = []
    new_kp_indices = []
    for m in good_matches_map2:
        pt_left = keypoints_left[m.queryIdx].pt
        pt_right = keypoints_right[m.trainIdx].pt
        if abs(pt_left[1] - pt_right[1]) < y_threshold:
            good_matches.append(m)
            # Only consider this feature as a new landmark if it wasn't already matched to an existing one.
            if m.queryIdx not in observed_indices:
                pts_left_stereo.append(pt_left)
                pts_right_stereo.append(pt_right)
                new_kp_indices.append(m.queryIdx)
    print(f"[Frame {frame_index}] Good stereo matches passing y-threshold: {len(good_matches)}")
    print(f"[Frame {frame_index}] Potential new points (not in observed_indices): {len(pts_left_stereo)}")

    if pts_left_stereo:
        pts_left_stereo = np.array(pts_left_stereo).T  # shape: 2 x N
        pts_right_stereo = np.array(pts_right_stereo).T  # shape: 2 x N
        pts_4d_hom = cv2.triangulatePoints(P_left, P_right, pts_left_stereo, pts_right_stereo)

        # pts_4d_hom = pts_4d_hom[:3, :] / w  # Now shape is 3 x M
        pts_4d_hom /= pts_4d_hom[3]
        z_vals = pts_4d_hom[2]  # shape (N,)
        valid = z_vals > z_treshold  # Maybe check for overly large values? 
        
        pts_new_3d_cam = pts_4d_hom[:3].T  # in the current camera frame
        pts_new_3d_cam = pts_new_3d_cam[valid]
        new_kp_indices = [new_kp_indices[i] for i, v in enumerate(valid) if v]
        # Transform new 3D points to the global coordinate system.
        pts_new_3d_global = (curr_pose_hom[:3, :3] @ pts_new_3d_cam.T + curr_pose_hom[:3, 3:4]).T
        
        # 5. Update the global map: add these new landmarks.
        next_landmark_id = max(global_map.keys()) + 1 if global_map else 0
        for i, kp_idx in enumerate(new_kp_indices):
            descriptor = descriptors_left[kp_idx]
            # Create a new Landmark with an initial observation.
            landmark = Landmark(landmark_id=next_landmark_id,
                                position=pts_new_3d_global[i],
                                descriptor=descriptor)
            landmark.observations.append((frame_index, keypoints_left[kp_idx]))
            global_map[next_landmark_id] = landmark
            next_landmark_id += 1

    return len(keypoints_left), n_inliers, len(good_matches), curr_pose_hom, global_map, n_inliers

def main(method="Keypoints", pose_graph_opt=False, save_plot_data=False, save_pose_data=False):
    start_time = time.time()
    img_list = utils.convert_stereo_data_to_img(skiprows=(1,16), nrows=60) #skip 1,16 -> 17 start #480Ende
    img_list = img_list[:]

    # visualization.display_images_as_gif(img_list)
    # tag_poses_global, camera_poses = apriltag.apriltag_main(img_list, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, baseline, rotation_vector)
    # visualization.plot_trajectory_2d(camera_poses)

    global_map = {}                                 #All landmarks in G
    camera_pose_hom = {}                            #All camera pose in G
    camera_pose_plot = [None] * len(img_list)       #All camera pose in G as rvec, tvec (for plotting)
    inliers_count = [None] * len(img_list)          #Debugging count
    keypoints_plot = []                             #n keypoints for plotting
    stereo_matches_plot = []                        #n stereo matches
    matches_to_global_map_plot = []                 #n matches for pnp for plotting
    all_landmarks_plot = []                         #n total landmarks
    new_landmarks_in_frame_plot = []                #n new landmarks each frame
    tvec_plot = []

    if method == "Apriltags":
        tag_poses_global, camera_poses = apriltag.apriltag_main(img_list, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, baseline, rotation_vector)
        visualization.plot_trajectory_2d(camera_poses, title=f"Camera Trajectory", save=False)
        visualization.plot_trajectory_in_environment(camera_poses, tag_poses_global)

    if method == "Keypoints":
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=int(1e5),scaleFactor=1.1, nlevels=30)

        # Intialize KalmanFilter
        dt = 1
        process_noise = 0.1
        measurement_noise = 1e-10
        kf = Kalman.KalmanFilter(dt, process_noise, measurement_noise)

        for f, img in enumerate(img_list):
            rectified_left, rectified_right, left_img, right_img, new_camera_matrix_left, new_camera_matrix_right, new_stereo_baseline, P1, P2 = rectify.split_img_and_rectify(img, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, baseline, rotation_vector)

            # visualization.compare_rectified_images(left_img, right_img)
            # visualization.compare_rectified_images(rectified_left, rectified_right)
            # visualization.quick_disparity_check(rectified_left, rectified_right)
            # disparity = visualization.compute_disparity(rectified_left, rectified_right)
            # cv2.imshow('Disp', disparity); cv2.waitKey(0); cv2.destroyAllWindows()
            if f == 0:  #First Frame init
                keypoints_left, keypoints_right, good_matches, filtered_matches, descriptors_left, descriptors_right  = extract_and_match_features(rectified_left, rectified_right, orb)
                keypoints_plot.append(len(keypoints_left))
                stereo_matches_plot.append(len(good_matches))
                left_matched_kps = [keypoints_left[m.queryIdx] for m in filtered_matches]
                # left_matched_pts = [keypoints_left[m.queryIdx].pt for m in filtered_matches]
                # visualization.plot_keypoints(rectified_left, keypoints_left)
                # visualization.plot_keypoints(rectified_left, left_matched_kps)
                triangulated_3d_points, matches_obj = triangulate_points(keypoints_left, keypoints_right, good_matches, P1, P2)

                print(f"Valid_Triangulated_Pts: {len(triangulated_3d_points)}")
                for i, m in enumerate(matches_obj):
                    # Use the left image descriptor as the landmark descriptor.
                    descriptor = descriptors_left[m.queryIdx]
                    # Create a new landmark with an initial observation (frame index 0).
                    landmark = Landmark(landmark_id=i, position=triangulated_3d_points[i], descriptor=descriptor)
                    landmark.observations.append((0, keypoints_left[m.queryIdx]))
                    global_map[i] = landmark
                curr_pose_hom = np.eye(4)  #4x4 Einheitsmatrix mit translation = 0
                # visualization.plot_keypoints_from_map(rectified_left, global_map)
                matches_to_global_map_plot.append(None)
                all_landmarks_plot.append(len(global_map))
                new_landmarks_in_frame_plot.append(len(global_map))
            else:
                n_keypoints_left, n_inlier_machtes, n_stereo_matches, curr_pose_hom, global_map, n_inliers = process_frame(f, rectified_left, rectified_right, global_map, camera_pose_hom[f-1], P1, P2, new_camera_matrix_left, orb)
                keypoints_plot.append(n_keypoints_left)
                stereo_matches_plot.append(n_stereo_matches)
                matches_to_global_map_plot.append(n_inlier_machtes)   #Matches for current Pose estimation
                all_landmarks_plot.append(len(global_map))
                new_landmarks_in_frame_plot.append(len(global_map)-all_landmarks_plot[f-1])
            
            curr_pose_extracted = extract_rvec_tvec_stacked(curr_pose_hom)
            smoothed_tvec = curr_pose_extracted[0:3].flatten()  # No Kalman
            smoothed_rvec = curr_pose_extracted[3:6].flatten()  # No Kalman
            # kf.predict()
            # smoothed_cam_pose = kf.update(curr_pose_extracted)
            # smoothed_tvec = smoothed_cam_pose[0:3].flatten()
            # smoothed_rvec = smoothed_cam_pose[3:6].flatten()
            R, _ = cv2.Rodrigues(smoothed_rvec)
            curr_pose_hom = np.eye(4)
            curr_pose_hom[:3, :3] = R
            curr_pose_hom[:3, 3] = smoothed_tvec.flatten()
            camera_pose_hom[f] = curr_pose_hom
            camera_pose_plot[f] = (smoothed_rvec, smoothed_tvec)
            tvec_plot.append(smoothed_tvec)

        for i,pose in enumerate(camera_pose_plot):
            print(f"Frame{i} tvec:{pose[1]}")

        for i,count in enumerate(inliers_count):
            print(f"Frame{i} count:{count}")
        
        # Ploting lists printed for debugging
        print(f"Keypoints per frame: {keypoints_plot}")
        print(f"Stereo Matches: {stereo_matches_plot}")
        print(f"Matches to global map: {matches_to_global_map_plot}")
        print(f"All_landmarks: {all_landmarks_plot}")
        print(f"Framewise landmarks: {new_landmarks_in_frame_plot}")

    if save_plot_data == True:
        with open('camera_pose_plot.pkl', 'wb') as f:
            pickle.dump(camera_pose_plot, f)

    if save_plot_data == True:
        df = pd.DataFrame({
            "Keypoints per frame": keypoints_plot,
            "Stereo matches": stereo_matches_plot,
            "Matches to global map": matches_to_global_map_plot,
            "All landmarks": all_landmarks_plot,
            "New landmarks per frame": new_landmarks_in_frame_plot,
            "tvec": tvec_plot
        })
        df.to_excel("my_data.xlsx", index=True)

    if pose_graph_opt == True:
        refined_rvec, refined_tvec = Pose_Graph_Optimizer.refine_camera_poses(camera_pose_plot)
        camera_pose_plot_ref = []
        for i in range(len(refined_rvec)):
            camera_pose_plot_ref.append((refined_rvec[i], refined_tvec[i]))
        visualization.plot_trajectory_2d(camera_poses=camera_pose_plot_ref)

    end_time = time.time()
    print(f"Calculating time: {end_time-start_time}")

    return camera_pose_plot

if __name__ == "__main__":
    camera_pose_plot = main(method="Apriltags", save_plot_data=True)
    visualization.plot_trajectory_2d(camera_poses=camera_pose_plot, arrows=False)
    

    
