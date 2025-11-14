import numpy as np
from scipy.optimize import least_squares

def rodrigues_to_rotation_matrix(rvec):
    """
    Convert a Rodrigues rotation vector (r_x, r_y, r_z) into a 3x3 rotation matrix.
    """
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        # No rotation, return identity
        return np.eye(3)
    k = rvec / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def rotation_matrix_to_rodrigues(R):
    """
    Convert a 3x3 rotation matrix into a Rodrigues rotation vector (r_x, r_y, r_z).
    """
    # A common formula is based on log(R), or we can use a more direct approach
    # but for simplicity let's rely on the basic formula with trace.
    cos_theta = ((np.trace(R) - 1) / 2.0)
    # clamp into [-1,1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if abs(theta) < 1e-12:
        return np.zeros(3)  # no rotation
    # Otherwise
    rvec = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ]) / (2*np.sin(theta))
    return rvec * theta

def create_transform(rvec, tvec):
    """
    Create a 4x4 homogeneous transform from Rodrigues (3) + translation (3).
    """
    R = rodrigues_to_rotation_matrix(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec
    return T

def decompose_transform(T):
    """
    Decompose a 4x4 transform back into (rvec, tvec).
    """
    R = T[:3, :3]
    t = T[:3, 3]
    rvec = rotation_matrix_to_rodrigues(R)
    return rvec, t

def invert_transform(T):
    """
    Invert a 4x4 homogeneous transform.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def relative_transform(rvec1, tvec1, rvec2, tvec2):
    """
    Compute the transform that takes pose1 -> pose2 from the given rvec/tvec.
    Returns (rvec_rel, tvec_rel).
    """
    T1 = create_transform(rvec1, tvec1)
    T2 = create_transform(rvec2, tvec2)
    T_rel = invert_transform(T1) @ T2
    r_rel, t_rel = decompose_transform(T_rel)
    return r_rel, t_rel

def pack_poses(rvecs, tvecs):
    """
    Convert a list of (rvec, tvec) into a 1D parameter vector for the solver.
    """
    # Each pose has 6 parameters: [r_x, r_y, r_z, t_x, t_y, t_z]
    return np.hstack([np.hstack([rvec, tvec]) for rvec, tvec in zip(rvecs, tvecs)])

def unpack_poses(params):
    """
    Convert the 1D parameter vector back to a list of (rvec, tvec).
    """
    # Each pose has 6 parameters
    poses = []
    for i in range(0, len(params), 6):
        rvec = params[i:i+3]
        tvec = params[i+3:i+6]
        poses.append((rvec, tvec))
    return poses

def pose_graph_residuals(params, relative_constraints, loop_closure_weight=0.4):
    """
    The main cost function to minimize.
    
    Arguments:
    ----------
    params : 1D array of length 6*N (flattened rvec, tvec for N poses).
    relative_constraints : List of dictionaries of the form:
        {
          'i': index of the first frame,
          'j': index of the second frame,
          'rvec_ij': relative rotation from i to j (Rodrigues),
          'tvec_ij': relative translation from i to j
        }
      For consecutive frames, j = i+1, but you can also add loop closure or
      any other known constraints.
    loop_closure_weight : float
      Weight to apply to the loop closure or any special constraints 
      (here it's just an example).
    
    Returns:
    --------
    residuals : 1D array of errors.
    """
    # Unpack the current parameters into (rvec, tvec)
    poses = unpack_poses(params)
    
    residuals = []
    # For each relative constraint, measure the error
    for c in relative_constraints:
        i = c['i']
        j = c['j']
        r_ij_meas = c['rvec_ij']
        t_ij_meas = c['tvec_ij']
        
        # Predicted relative transform
        r_ij_pred, t_ij_pred = relative_transform(poses[i][0], poses[i][1],
                                                  poses[j][0], poses[j][1])
        
        # We compute difference in rotation vector space (naive) and translation
        rot_error = r_ij_pred - r_ij_meas
        trans_error = t_ij_pred - t_ij_meas
        
        # Add them to the residuals
        residuals.extend(rot_error)
        residuals.extend(trans_error)
    
    # Optionally you can add a loop closure constraint 
    # that the last pose ~ first pose with some weight
    # (If you want them to be exactly the same, you can do that as well.)
    # Example: difference in positions/rotations between first and last
    i_first = 0
    i_last = len(poses) - 1
    rvec_first, tvec_first = poses[i_first]
    rvec_last, tvec_last = poses[i_last]
    
    # If we expect them to be identical or "close":
    # We'll just measure the difference directly
    loop_rot_error = (rvec_last - rvec_first) * loop_closure_weight
    loop_trans_error = (tvec_last - tvec_first) * loop_closure_weight
    
    residuals.extend(loop_rot_error)
    residuals.extend(loop_trans_error)
    
    return np.array(residuals)

def refine_camera_poses(poses):
    """
    Take in a list of camera poses (Rodrigues + translation) and
    refine them by imposing that consecutive transforms match
    the input transforms, and that the first and last pose are
    consistent (loop closure).
    
    Parameters
    ----------
    initial_rvecs : list of length N, each is shape (3,)
    initial_tvecs : list of length N, each is shape (3,)
    
    Returns
    -------
    refined_rvecs, refined_tvecs : lists of the same shape, refined by optimization.
    """
    initial_rvecs = []
    initial_tvecs = []

    for pose in poses:
        initial_rvecs.append(pose[0])
        initial_tvecs.append(pose[1])

    N = len(initial_rvecs)
    assert len(initial_tvecs) == N
    
    # Build constraints for consecutive frames
    # The measured relative transforms come from the original initial guesses.
    relative_constraints = []
    for i in range(N - 1):
        r_ij, t_ij = relative_transform(initial_rvecs[i], initial_tvecs[i],
                                        initial_rvecs[i+1], initial_tvecs[i+1])
        constraint = {
            'i': i,
            'j': i+1,
            'rvec_ij': r_ij,
            'tvec_ij': t_ij
        }
        relative_constraints.append(constraint)
    
    # Pack initial guess
    x0 = pack_poses(initial_rvecs, initial_tvecs)
    
    # Run the optimization
    result = least_squares(
        fun=pose_graph_residuals,
        x0=x0,
        args=(relative_constraints, 0.5),  # 1.0 is loop_closure_weight
        method='lm'  # 'lm' or 'trf' or 'dogbox'; 'lm' can be good for small problem
    )
    
    # Unpack the refined parameters
    refined_poses = unpack_poses(result.x)
    refined_rvecs, refined_tvecs = zip(*refined_poses)
    
    return list(refined_rvecs), list(refined_tvecs)

