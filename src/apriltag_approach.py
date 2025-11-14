import cv2
import numpy as np
from pyapriltags import Detector
from utils import Kalman
from pyapriltags import Detector
from utils import rectify
from utils import utils
from utils import visualization

def detect_tags(image, rect_camera_matrix, dist_coeffs=None, tag_size=0.162, detector=None, decision_margin=20):
    """
    Detect AprilTags in an image and estimate their 3D pose using solvePnP.
    
    :param image: Input BGR or grayscale image from which tags will be detected.
    :param camera_matrix: Intrinsic camera matrix (3x3).
    :param dist_coeffs: Distortion coefficients (if available).
    :param tag_size: Physical side length of the AprilTag (same units as you want in translation).
    :param detector: A pre-initialized AprilTag detector (your existing 'detector.detect' object).
    :return: A dictionary of tag_id -> (rvec, tvec).
             rvec, tvec are each 3-element numpy arrays.
    """
    
    # Convert to grayscale if needed
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    detections = detector.detect(gray)
    
    # Prepare output dictionary
    tags = {}
    filtered_detections = []

    # Define 3D corner points of the tag, centered at (0, 0, 0).
    # X right, Y down, Z forward (depending on your coordinate frame convention).
    obj_pts = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0]
    ], dtype=np.float32)
    
    for det in detections:
        if det.decision_margin < decision_margin:
            print(f"Didnt use tag: {det.tag_id} with confidence: {det.decision_margin}")
            continue

        filtered_detections.append(det)

        corners = np.array(det.corners, dtype=np.float32)
        order = [3,2,1,0]
        reordered_corners = corners[order]  #Get the same order as obj_pts

        # Use solvePnP to get rotation/translation
        success, rvec, tvec = cv2.solvePnP(obj_pts,
                                           reordered_corners,
                                           rect_camera_matrix,
                                           distCoeffs=dist_coeffs,  
                                           flags=cv2.SOLVEPNP_ITERATIVE)    
        if success:
            # Flatten for convenience, but keep both R and T
            rvec = rvec.flatten()
            tvec = tvec.flatten()
            tags[det.tag_id] = (rvec, tvec)
        else:
            # Handle case if solvePnP fails for any reason
            tags[det.tag_id] = (None, None)
    return tags, filtered_detections


def rigid_transform_3D(A, B):
    """
    Computes the rotation R and translation t that best align
    two sets of 3D points (A and B) in a least-squares sense.
    
    Input:
        A: (N x 3) array of points (model points)
        B: (N x 3) array of points (measured points)
    Output:
        R: (3 x 3) rotation matrix
        t: (3 x 1) translation vector
    """
    assert A.shape == B.shape
    N = A.shape[0]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # Center the points
    AA = A - centroid_A
    BB = B - centroid_B
    
    # Compute covariance matrix
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    t = centroid_B - R @ centroid_A
    return R, t

def fuse_apriltag_detections(left_detections, right_detections,
                             left_cam_matrix, right_cam_matrix,
                             tag_size, stereo_baseline):
    """
    Fuse AprilTag detections from a stereo pair.
    
    Parameters:
      left_detections: dict mapping tag_id to detection info from left image.
                       Each detection must have a key "corners" with a (4,2) array.
      right_detections: similar dict for the right image.
      left_cam_matrix: (3x3) intrinsic matrix for left camera.
      right_cam_matrix: (3x3) intrinsic matrix for right camera.
      tag_size: side length of the tag (in your chosen world units).
      stereo_baseline: distance between the left and right cameras along the x-axis.
      
    Returns:
      tag_poses: dictionary mapping tag_id to a dict with keys "rvec" and "tvec"
                 corresponding to the tag’s rotation (Rodrigues vector) and translation.
                 
    Note:
      - For tags seen in both images, we triangulate the 3D corner positions and then
        compute the rigid transform from the known tag model points.
      - For tags seen only in one image, we fall back to solvePnP.
      - We assume that the cameras are rectified. In that case, we can form the projection
        matrices as follows.
    """
    tag_poses = {}

    # Build projection matrices.
    # Left camera: [I | 0]
    P_left = left_cam_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
    
    # Right camera: assume translation along x by -stereo_baseline
    t_right = np.array([[-stereo_baseline], [0], [0]])
    P_right = right_cam_matrix @ np.hstack([np.eye(3), t_right])
    
    # Define the tag's object points (assuming a centered square, adjust ordering as needed)
    half_size = tag_size / 2.0
    # Order: top-left, top-right, bottom-right, bottom-left
    obj_pts = np.array([[-half_size,  half_size, 0],
                        [ half_size,  half_size, 0],
                        [ half_size, -half_size, 0],
                        [-half_size, -half_size, 0]], dtype=np.float32)
    
    left_det_dict = {}
    for det in left_detections:
        corners = np.array(det.corners, dtype=np.float32)
        left_det_dict[det.tag_id] = corners[[3,2,1,0]]

    right_det_dict = {}
    for det in right_detections:
        corners = np.array(det.corners, dtype=np.float32)
        right_det_dict[det.tag_id] = corners[[3,2,1,0]]

    # Consider all tags that appear in either detection set
    all_tags = set(left_det_dict.keys()).union(set(right_det_dict.keys()))
    # print(f"All_tags: {all_tags}")
    for tag_id in all_tags:
        # Case 1: Tag detected in both cameras: use stereo triangulation
        if tag_id in left_det_dict and tag_id in right_det_dict:
            left_corners = left_det_dict[tag_id]
            right_corners = right_det_dict[tag_id]
            
            if left_corners.shape != (4, 2) or right_corners.shape != (4, 2):
                print(f"Tag {tag_id}: corner shape mismatch, skipping stereo fusion.")
                continue
            
            # Prepare points for cv2.triangulatePoints (expects 2xN arrays)
            pts_left = left_corners.T  # shape: (2, 4)
            pts_right = right_corners.T  # shape: (2, 4)
            
            # Triangulate (points will be in homogeneous coordinates in left camera frame)
            pts_4d = cv2.triangulatePoints(P_left, P_right, pts_left, pts_right)
            pts_3d = (pts_4d[:3, :] / pts_4d[3, :]).T  # shape: (4, 3)
            
            # Now solve the 3D–3D alignment between our known object points and the measured 3D points.
            R, t = rigid_transform_3D(obj_pts, pts_3d)
            # Convert the rotation matrix to a Rodrigues vector
            rvec, _ = cv2.Rodrigues(R)
            tvec = t.reshape(3, 1)
            tag_poses[tag_id] = (rvec.flatten(), tvec.flatten())
        
        # Case 2: Only in left image – use standard solvePnP with left camera
        elif tag_id in left_det_dict:
            left_corners = left_det_dict[tag_id]
            if left_corners.shape != (4, 2):
                print(f"Tag {tag_id}: left corner shape mismatch, skipping.")
                continue
            
            ret, rvec, tvec = cv2.solvePnP(obj_pts, left_corners,
                                           left_cam_matrix, None,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
            if ret:
                tag_poses[tag_id] =  (rvec.flatten(), tvec.flatten())
            else:
                print(f"Tag {tag_id}: solvePnP failed for left detection.")
                
        # Case 3: Only in right image – use solvePnP and then transform to left camera frame
        elif tag_id in right_det_dict:
            right_corners = right_det_dict[tag_id]
            if right_corners.shape != (4, 2):
                print(f"Tag {tag_id}: right corner shape mismatch, skipping.")
                continue
            
            ret, rvec, tvec = cv2.solvePnP(obj_pts, right_corners,
                                           right_cam_matrix, None,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
            if ret:
                # Transform from right camera frame to left camera frame.
                # For our assumed stereo setup, the relation is:
                #   X_left = X_right + [stereo_baseline, 0, 0]^T
                tvec_left = tvec + np.array([[stereo_baseline], [0], [0]], dtype=tvec.dtype)
                tag_poses[tag_id] = (rvec.flatten(), tvec_left.flatten())
            else:
                print(f"Tag {tag_id}: solvePnP failed for right detection.")
    
    return tag_poses

def apriltag_main(img_list, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, baseline, rotation_vector):
     # Initialize Kalman filter parameters
    dt = 1/15  # for example, 30 FPS => 1/30 sec per frame
    process_noise = 0.01
    measurement_noise = 1e-6
    kf = Kalman.KalmanFilter(dt, process_noise, measurement_noise)
    decision_margin = 40

    trajectory = []
    detector = Detector(families='tag36h11')

    prev_tags = None
    global_pose = np.eye(4)  # Initial pose at origin

    rect_left_img_list = []
    rect_right_img_list = []
    left_img_list = []
    # undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    left_img_detections = []   
    right_img_detections = []
    Apriltag_images = []
    for img in img_list:
        rect_left, rect_right, left_img, right_img, rect_matrix_left, rect_matrix_right, stereo_baseline, P1, P2 = rectify.split_img_and_rectify(img, left_camera_matrix, left_dist_coeffs, right_camera_matrix, right_dist_coeffs, baseline, rotation_vector, visualize=False)
        curr_tags_left, detections_left = detect_tags(rect_left, rect_matrix_left, dist_coeffs=None, tag_size=0.162, detector=detector, decision_margin=decision_margin) 
        left_img_detections.append(curr_tags_left)
        annotated = visualization.draw_tags(rect_left, detections_left)
        Apriltag_images.append(annotated)
        # cv2.imshow("Apriltags Detections", annotated)
        # cv2.waitKey(0)
        # curr_tags_right, detections_right = detect_tags(rect_right, rect_matrix_right, dist_coeffs=None, tag_size=0.162, detector=detector, decision_margin=decision_margin) 
        # right_img_detections.append(curr_tags_right)
        # curr_tags_fused = fuse_apriltag_detections(detections_left, detections_right, rect_matrix_left, rect_matrix_right, tag_size=0.162, stereo_baseline=stereo_baseline)
        # left_img_detections.append(curr_tags_fused)
        # print(curr_tags_fused)
    # visualization.images_to_video(Apriltag_images, "apriltag_video.mp4", 15)
    camera_poses = [None] * len(img_list)
    for i in range(len(left_img_detections)):
        print(f"{i}-frame : {left_img_detections[i]}")
    visualization.plot_apriltag_timeline(left_img_detections)

    # For frame 0, define the identity pose:
    rvec_identity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    tvec_identity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    identity_measurement = np.vstack((tvec_identity.reshape(3,1), rvec_identity.reshape(3,1)))
    kf.predict()
    pred = kf.update(identity_measurement)
    smoothed_tvec = pred[0:3].flatten()
    smoothed_rvec = pred[3:6].flatten()
    camera_poses[0] = (smoothed_rvec, smoothed_tvec)

    tag_poses_global = {}  # tag_id -> (rvec_global, tvec_global)

    for tag_id, (rvec_ct, tvec_ct) in left_img_detections[0].items():   #Passt!
        # Keine Inverse nötig, da Kamera = Identity -> Kamera zu Tag = Tag in Global
        # rvec_tag_g, tvec_tag_g = invert_pose(rvec_ct, tvec_ct)
        rvec_tag_g, tvec_tag_g = rvec_ct, tvec_ct
        tag_poses_global[tag_id] = (rvec_tag_g, tvec_tag_g)

    for f in range(1, len(left_img_detections)):
        detection_dict = left_img_detections[f]
        
        # We'll store the camera pose after we find at least one known tag
        camera_pose_candidates = []
        
        for tag_id, (rvec_ct, tvec_ct) in detection_dict.items():
            if tag_id in tag_poses_global:
                # We know this tag's global pose from a previous frame
                rvec_tag_g, tvec_tag_g = tag_poses_global[tag_id]
                
                # If we invert camera->tag, we get tag->camera.
                rvec_tc, tvec_tc = utils.invert_pose(rvec_ct, tvec_ct)

                # Then we compose tag->global with tag->camera to get camera->global.
                rvec_c_g, tvec_c_g = utils.compose_poses(rvec_tag_g, tvec_tag_g, rvec_tc, tvec_tc)

                camera_pose_candidates.append((rvec_c_g, tvec_c_g))

        if camera_pose_candidates:
            # Convert candidate lists to numpy arrays
            rvecs = np.array([pose[0] for pose in camera_pose_candidates])
            tvecs = np.array([pose[1] for pose in camera_pose_candidates])

            # Hier noch einmal reinschauen CHATGPT AprilTag Pose Vis (Rotation Caveat beachten!), evt Gewichtung über Prediction Konfidence oder so! Sprünge beheben (Falls ein Tag lange nicht erkannt wird oder so?)
            # avg_rvec = np.mean(rvecs, axis=0)
            # avg_tvec = np.mean(tvecs, axis=0)
            avg_rvec = np.mean(rvecs, axis=0).reshape(3,1)
            avg_tvec = np.mean(tvecs, axis=0).reshape(3,1)
            measuremt = np.vstack((avg_tvec, avg_rvec))
            # print(f"Avg_tvec: {avg_tvec}")
            # print(f"Avg_rvec: {avg_rvec}")
            # camera_poses[f] = (avg_rvec, avg_tvec)
        # else:
        #     print(f"No known tag found in frame {f}, skipping.")
        #     avg_rvec = None
        #     avg_tvec = None

        #     measuremt = None
        #     continue
        
        kf.predict()
        if measuremt is not None:
            smoothed_pred = kf.update(measuremt)
        
        smoothed_tvec = smoothed_pred[0:3].flatten()
        smoothed_rvec = smoothed_pred[3:6].flatten()
        camera_poses[f] = (smoothed_rvec, smoothed_tvec)

        # Now that we have camera_poses[f], we can compute global pose for the new tags
        rvec_camera_g, tvec_camera_g = camera_poses[f]
        for tag_id, (rvec_ct, tvec_ct) in detection_dict.items():
            if tag_id not in tag_poses_global:
                # Compose camera->global with tag->camera:
                #  (global->camera) ∘ (camera->tag) = global->tag
                rvec_tag_g, tvec_tag_g = utils.compose_poses(rvec_camera_g, tvec_camera_g, rvec_ct, tvec_ct)
                tag_poses_global[tag_id] = (rvec_tag_g, tvec_tag_g)

    return tag_poses_global, camera_poses