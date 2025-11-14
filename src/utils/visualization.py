import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from pyapriltags import Detector
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import plotly.graph_objects as go

def plot_apriltag_timeline(detections, cmap_name='tab20'):
    # 1. Extract all unique tag IDs and sort them (to fix row order)
    all_ids = sorted({tag for d in detections for tag in d})

    # 2. For each tag, collect the frame indices where it appears
    id_to_frames = {tag: [] for tag in all_ids}
    for frame_idx, det in enumerate(detections):
        for tag in det:
            id_to_frames[tag].append(frame_idx)

    # 3. Prepare data for eventplot: list of lists of frame indices
    events = [id_to_frames[tag] for tag in all_ids]

    # 4. Generate a distinct color for each tag
    cmap = plt.cm.get_cmap(cmap_name, len(all_ids))
    colors = [cmap(i) for i in range(len(all_ids))]

    # 5. Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.eventplot(
        events,
        orientation='horizontal',
        linelengths=0.8,
        colors=colors,
    )

    # 6. Tidy up axes
    ax.set_yticks(range(len(all_ids)))
    ax.set_yticklabels(all_ids)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('AprilTag ID')
    ax.set_title('AprilTag Presence Timeline')
    plt.tight_layout()
    plt.show()

def display_images_as_gif(img_list, fps=15):
    """
    Displays a list of images rapidly like a GIF using OpenCV.

    Args:
        image_paths (list of str): List of file paths to images.
        fps (int): Frames per second, default is 30.
    """
    if not img_list:
        print("No images to display.")
        return

    delay = 1 / fps  # Convert FPS to delay in seconds

    for idx, img in enumerate(img_list, start=1):
        print(f"Showing frame {idx}/{len(img_list)}")
        cv2.imshow("Rapid Image Viewer", img)
        if cv2.waitKey(int(delay * 1000)) & 0xFF == 27:  # Exit on ESC key
            break

    cv2.destroyAllWindows()

def plot_trajectory(trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', label='Estimated')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    plt.show()

def draw_tags(image, detections):
    """Draw AprilTag borders and IDs on the image."""
    for detection in detections:
        corners = detection.corners.astype(int)
        cv2.polylines(image, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(image, f"ID: {detection.tag_id}", tuple(corners[0]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

def images_to_video(
    images: np.ndarray,
    output_path: str,
    fps: float,
    codec: str = 'mp4v'
) -> None:
    """
    Convert a sequence of images to a video file.

    Parameters
    ----------
    images : list of str or numpy.ndarray
        Either a list of filesystem paths to image files, or a list of OpenCV images (numpy arrays).
    output_path : str
        Path to the output video file. Extension determines container (e.g. .mp4, .avi).
    fps : float
        Frames per second for the output video.
    codec : str, optional
        FourCC code for the codec (default 'mp4v'). Common codes:
          - 'XVID' for .avi
          - 'mp4v' for .mp4
          - 'MJPG' for .avi/.mp4
    """
    # Helper to load or verify image array
    def _load(img):
        if isinstance(img, str):
            frame = cv2.imread(img)
            if frame is None:
                raise FileNotFoundError(f"Could not load image at path: {img}")
            return frame
        elif isinstance(img, np.ndarray):
            return img
        else:
            raise TypeError("Each item in images must be a file path or an ndarray.")

    # Load first frame to get video size
    first_frame = _load(images[0])
    height, width = first_frame.shape[:2]

    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*codec)
    # Make sure output directory exists
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames
    for img in images:
        frame = _load(img)
        # If frame size differs, resize to match first frame
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    writer.release()
    print(f"Video saved to {output_path}")

def visualize_tags(image, detections, color=(0, 255, 0), thickness=2, font_scale=1.0):
    """
    Draw bounding lines around each AprilTag detection and label the tag ID.
    
    :param image: The original BGR image (will be modified in place).
    :param detections: A list of detections from pyapriltags (each has 'corners', 'center', 'tag_id').
    :param color: Color for drawing lines and text (BGR).
    :param thickness: Line thickness for the bounding box.
    :param font_scale: Scale for the text label.
    """
    print("detections type:", type(detections))
    print(detections)
    for det in detections:
        print(det)
        corners = det.corners  # shape: (4,2)
        center = det.center    # shape: (2,)
        tag_id = det.tag_id

        # Convert to int for drawing
        corners = np.round(corners).astype(int)
        center = tuple(np.round(center).astype(int))

        # Draw a polygon around the corners
        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            cv2.line(image, p1, p2, color, thickness)

        # Put the tag ID near the center
        cv2.putText(image, 
                    f"ID:{tag_id}", 
                    center, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    color, 
                    thickness)

    return image

def plot_keypoints_over_frames(all_kp, good_kp, title="Keypoints per Frame"):
    """
    Plots total vs. good keypoints for each frame.
    
    Parameters
    ----------
    all_kp : list[int]
        Total keypoints detected in each frame.
    good_kp : list[int]
        Number of good keypoints (matches) in each frame.
    title : str
        Chart title.
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    if len(all_kp) != len(good_kp):
        raise ValueError("all_kp and good_kp must be the same length")

    frames = list(range(len(all_kp)))  # 0,1,2,...

    fig = go.Figure()

    # Total keypoints line
    fig.add_trace(go.Scatter(
        x=frames,
        y=all_kp,
        mode='lines+markers',
        name='Total keypoints',
        hovertemplate="Frame %{x}<br>Total: %{y}<extra></extra>"
    ))

    # Good keypoints line
    fig.add_trace(go.Scatter(
        x=frames,
        y=good_kp,
        mode='lines+markers',
        name='Good keypoints',
        hovertemplate="Frame %{x}<br>Good: %{y}<extra></extra>"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Frame Number",
        yaxis_title="Number of Keypoints",
        xaxis=dict(tickmode='linear'),
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig
def plot_trajectory_2d(camera_poses, title="Robot Trajectory", save=False, arrows=False, arrow_length=0.05):
    """
    Plots the robot (camera) trajectory in 2D using the (x, y) components of each tvec.
    
    :param camera_poses: A list where camera_poses[f] = (rvec, tvec), each a 3-element np.array.
                         If some entries are None, they will be skipped.
    :param title: A string for the plot title.
    """
    xs, ys, frame_number = [], [], []
    # print(camera_poses)
    for i, pose in enumerate(camera_poses):
        if pose is None:
            # e.g., no valid pose for that frame
            xs.append(np.nan)
            ys.append(np.nan)
            frame_number.append(i)
            continue
        rvec, tvec = pose
        # tvec is typically [x, y, z]; 
        # you can choose which components are "horizontal plane" vs "vertical" 
        x, y, z = tvec
        xs.append(x)
        ys.append(z)
        frame_number.append(i)

    plt.figure(figsize=(10,10))
    
    # Optionally, draw the trajectory as a line
    plt.plot(xs, ys, color='gray', linestyle='-', alpha=0.5)
    
    # Plot points with colors corresponding to their frame number
    sc = plt.scatter(xs, ys, c=frame_number, cmap='viridis', marker='o', s=25)
    
    # Add a colorbar to show frame number mapping
    plt.colorbar(sc, label='Frame Number')
    if arrows == True:
            # Now add arrows indicating the camera's viewing direction
        for pose in camera_poses:
            if pose is None:
                continue
            rvec, tvec = pose
            x, y, z = tvec
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            # Define the forward direction in camera coordinates (here along positive z)
            forward = R.dot(np.array([0, 0, 1]))
            # Extract the horizontal (x, z) components
            forward_x, forward_z = forward[0], forward[2]
            # Normalize the horizontal vector to keep arrow lengths consistent
            norm = np.sqrt(forward_x**2 + forward_z**2)
            if norm > 0:
                forward_x /= norm
                forward_z /= norm
            # Draw an arrow at (x, z) with the specified arrow length
            plt.arrow(x, z, arrow_length * forward_x, arrow_length * forward_z, color='red',
                    head_width=0.005, head_length=0.02)
        
    plt.title(title)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis('equal')  # Ensures that the scales for x and y are the same
    plt.grid(True)
    if save:
        plt.savefig(title+".png", dpi=600)
    else:
        plt.show()

def draw_tag_pose(
    image: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    tag_size: float,
    axis_length: float = 0.05,
    tag_id: str = None
) -> np.ndarray:
    """
    Draw the 3D coordinate frame and (optionally) a bounding box for an AprilTag
    of known 'tag_size'. If 'tag_id' is given, label the tag in the image.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.
    rvec : np.ndarray
        (3,) rotation vector for the tag.
    tvec : np.ndarray
        (3,) translation vector for the tag.
    camera_matrix : np.ndarray
        3x3 camera intrinsic matrix.
    dist_coeffs : np.ndarray
        Camera distortion coefficients.
    tag_size : float
        Physical side length of the tag (e.g., meters).
    axis_length : float
        Length of the axes to draw (same units as 'tag_size').
    tag_id : str, optional
        ID string to label the tag.

    Returns
    -------
    annotated_image : np.ndarray
        Annotated copy of the input image.
    """
    annotated_image = image.copy()

    # 1. Draw the coordinate axes (X=red, Y=green, Z=blue in OpenCV)
    cv2.drawFrameAxes(
        annotated_image,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        axis_length
    )

    # 2. (Optional) Reproject corners to show bounding box
    # Define the tag corners in its local 3D coordinate system
    half_size = tag_size / 2.0
    tag_3d_points = np.array([
        [-half_size, -half_size, 0],
        [ half_size, -half_size, 0],
        [ half_size,  half_size, 0],
        [-half_size,  half_size, 0]
    ], dtype=np.float32)

    # Project the 3D corners into the image
    projected_points, _ = cv2.projectPoints(
        tag_3d_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2).astype(int)

    # Draw lines between consecutive corners to form the box
    color = (0, 255, 0)  # Green
    thickness = 2
    for i in range(4):
        p1 = tuple(projected_points[i])
        p2 = tuple(projected_points[(i + 1) % 4])
        cv2.line(annotated_image, p1, p2, color, thickness)

    # 3. (Optional) Label the tag ID near the first corner
    if tag_id is not None:
        text_pos = tuple(projected_points[0])
        cv2.putText(
            annotated_image,
            f"ID: {tag_id}",
            (text_pos[0], text_pos[1] - 10),  # Slightly above the corner
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # Font scale
            (0, 0, 255),  # Text color (red)
            2  # Thickness
        )

    return annotated_image

def visualize_multiple_tags(
    image: np.ndarray,
    tags_dict: dict,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    tag_size: float,
    axis_length: float = 0.05
) -> np.ndarray:
    """
    Visualize multiple AprilTags by drawing their pose axes and bounding boxes
    on a single image.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.
    tags_dict : dict
        Dictionary {tag_id: (rvec, tvec)}.
    camera_matrix : np.ndarray
        3x3 camera intrinsic matrix.
    dist_coeffs : np.ndarray
        Distortion coefficients.
    tag_size : float
        Physical side length of each AprilTag (same units as rvec/tvec).
    axis_length : float
        Length of the coordinate axes to draw.

    Returns
    -------
    annotated_image : np.ndarray
        The annotated version of 'image' with all tags drawn.
    """
    annotated_image = image.copy()

    for tag_id, (rvec, tvec) in tags_dict.items():
        annotated_image = draw_tag_pose(
            annotated_image,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
            tag_size,
            axis_length,
            tag_id=str(tag_id)  # Convert to string if numeric
        )

    return annotated_image



def plot_apriltags_top_view(poses, arrow_length=0.1):
    """
    Plots a 2D top-down view of AprilTag poses, optionally flipping the entire image 180°.

    Parameters:
        poses (dict): Dictionary of poses in the form {tag_id: (rvec, tvec)}.
                      rvec: Rotation vector (as a list or np.array).
                      tvec: Translation vector (as a list or np.array) [x, y, z].
        arrow_length (float): Length of the arrow to indicate orientation.
        flip (bool): If True, flip the entire image 180° (i.e., apply (x,y) -> (-x,-y) 
                     and adjust the orientation accordingly).
    """
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    for tag_id, (rvec, tvec) in poses.items():
        # Convert inputs to numpy arrays.
        rvec = np.array(rvec, dtype=float)
        tvec = np.array(tvec, dtype=float)
        
        # Convert the rotation vector to a rotation matrix.
        R, _ = cv2.Rodrigues(rvec)
        
        # Extract yaw angle assuming rotation about the z-axis.
        # yaw = arctan2(R[1, 0], R[0, 0])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        
        # Get x and y position from tvec.
        x, y = tvec[0], tvec[2]
        tag_size = 0.162
        # Initially, the rectangle is unrotated.
        rect = patches.Rectangle((x - tag_size/2, y - tag_size/2), tag_size, tag_size, 
                                 linewidth=1, edgecolor='k', facecolor='lightblue', zorder=2)
        
        # Create a rotation transform around the center of the rectangle.
        tform = transforms.Affine2D().rotate_around(x, y, yaw) + ax.transData
        rect.set_transform(tform)
        ax.add_patch(rect)
        
        # Draw the tag number centered inside the rectangle.
        ax.text(x, y, f'{tag_id}', ha='center', va='center', fontsize=10, color='red', zorder=3)
        # # Compute arrow components based on the yaw angle.
        # dx = arrow_length * np.cos(yaw)
        # dy = arrow_length * np.sin(yaw)
        
        # # Plot the arrow indicating orientation.
        # ax.arrow(x, y, dx, dy, head_width=0.05 * arrow_length, head_length=0.05 * arrow_length,
        #          fc='k', ec='k', length_includes_head=True)
        
        # # Plot the tag position.
        # ax.plot(x, y, 'bo')
        
        # # Annotate the tag with its id.
        # ax.text(x, y, f' {tag_id}', fontsize=12, color='red')
    
    # Set labels, title, grid and ensure equal scaling.
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Apriltags Top View')
    ax.grid(True)
    ax.axis('equal')
    
    plt.show()

def plot_trajectory_in_environment(camera_poses, apriltag_poses, 
                                     title="Apriltag positions", 
                                     arrow_length=0.1, save=False):
    """
    Plots the AprilTag environment and the robot (camera) trajectory in a single 2D top-down view.
    
    Parameters:
        camera_poses (list): List where each element is either None or a tuple (rvec, tvec).
                             tvec is expected to be a 3-element array-like [x, y, z].
        apriltag_poses (dict): Dictionary of AprilTag poses in the form {tag_id: (rvec, tvec)}.
                               tvec is expected to be [x, y, z].
        title (str): Title of the plot.
        arrow_length (float): Length of the arrow to indicate the orientation of each AprilTag.
        flip (bool): If True, flip the entire AprilTag image 180°.
        save (bool): If True, save the plot as an image file. Otherwise, display it.
    """
    # Create a single figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # -------------------------------
    # Plot the AprilTag environment
    # -------------------------------
    for tag_id, (rvec, tvec) in apriltag_poses.items():
        # Ensure inputs are numpy arrays
        rvec = np.array(rvec, dtype=float)
        tvec = np.array(tvec, dtype=float)
        
        # Convert the rotation vector to a rotation matrix.
        R, _ = cv2.Rodrigues(rvec)
        
        # Compute yaw angle assuming rotation about the z-axis.
        yaw = np.arctan2(R[1, 0], R[0, 0])
        
        # Use tvec components: x = tvec[0], and y for plotting = tvec[2]
        x, y = tvec[0], tvec[2]
        
        tag_size = 0.162
        # Initially, the rectangle is unrotated.
        rect = patches.Rectangle((x - tag_size/2, y - tag_size/2), tag_size, 0.01, 
                                 linewidth=1, edgecolor='k', facecolor='lightblue', zorder=2)
        
        # Create a rotation transform around the center of the rectangle.
        tform = transforms.Affine2D().rotate_around(x, y, yaw) + ax.transData
        rect.set_transform(tform)
        ax.add_patch(rect)
        
        # Draw the tag number centered inside the rectangle.
        ax.text(x, y, f'{tag_id}', ha='center', va='center', fontsize=10, color='red', zorder=3)
    
    # ------------------------------------
    # Plot the camera (robot) trajectory
    # ------------------------------------
    xs, ys, frame_numbers = [], [], []
    for i, pose in enumerate(camera_poses):
        if pose is None:
            xs.append(np.nan)
            ys.append(np.nan)
            frame_numbers.append(i)
            continue
        
        rvec, tvec = pose
        # tvec is expected to be [x, y, z]; we use x and z for plotting.
        x, y, z = tvec
        xs.append(x)
        ys.append(z)
        frame_numbers.append(i)
    
    # # Draw the trajectory line (with transparency)
    # ax.plot(xs, ys, color='gray', linestyle='-', alpha=0.5)
    
    # Scatter plot with colors corresponding to the frame number
    # sc = ax.scatter(xs, ys, c=frame_numbers, cmap='viridis', marker='o', s=50)
    sc = ax.scatter(0, 0,  marker='x', s=100)
    
    # Add a colorbar to show frame numbers.
    # plt.colorbar(sc, label='Frame Number', ax=ax)
    
    # -------------------------------
    # Finalize the plot
    # -------------------------------
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    ax.grid(True)
    ax.axis('equal')  # Keep the aspect ratio equal for proper scaling
    
    if save:
        plt.savefig(title + ".png", dpi=600)
    else:
        plt.show()

def plot_3D_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the landmark positions
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')
    ax.set_title('3D Landmark Map')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_keypoints_from_map(frame, keypoints):
        # Convert BGR (OpenCV default) to RGB for matplotlib
    first_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Iterate through your landmarks
    for id, keypoint in keypoints.items():
            obs = keypoint.observations[0][1]
            x, y = obs.pt

            # Draw a small circle at the keypoint location
            cv2.circle(first_frame_rgb, (int(x), int(y)), 1, (255, 0, 0), -1)

    # Display the image with landmarks
    plt.imshow(first_frame_rgb)
    plt.title(f'Image with {len(keypoints)} Landmarks')
    plt.axis('off')
    plt.show()

def compare_rectified_images(left_img, right_img, num_lines=10):
    """
    Draws horizontal epipolar lines on rectified stereo images and displays them side by side.
    
    Parameters:
    - left_img: Left rectified image as a NumPy array.
    - right_img: Right rectified image as a NumPy array.
    - num_lines: Number of horizontal lines to draw (default is 10).
    """
    if left_img is None or right_img is None:
        raise ValueError("One or both images are invalid.")

    # Create copies to draw on
    left_with_lines = left_img.copy()
    right_with_lines = right_img.copy()

    # Get the height of the images (assuming both images are rectified and have the same height)
    height = left_img.shape[0]

    # Draw horizontal lines at evenly spaced intervals
    for y in np.linspace(0, height - 1, num_lines, dtype=int):
        color = (0, 255, 0)  # Green color for the lines
        thickness = 1
        cv2.line(left_with_lines, (0, y), (left_img.shape[1], y), color, thickness)
        cv2.line(right_with_lines, (0, y), (right_img.shape[1], y), color, thickness)

    # Combine the images side by side
    combined = np.hstack((left_with_lines, right_with_lines))
    # Display the combined image
    cv2.namedWindow("Images with Epipolar Lines", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Images with Epipolar Lines", 1200, 600)
    
    cv2.imshow("Images with Epipolar Lines", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_keypoints(image, keypoints):
    """
    Plots the keypoints on the input image using small red dots.
    
    Parameters:
        image (numpy.ndarray): The image in BGR format.
        keypoints (list): List of cv2.KeyPoint objects detected in the image.
    """
    # Convert the image from BGR to RGB for correct display with matplotlib.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract the (x, y) coordinates from the keypoints.
    xs = [int(kp.pt[0]) for kp in keypoints]
    ys = [int(kp.pt[1]) for kp in keypoints]
    
    # Plot the image and overlay the keypoints using small red dots.
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.scatter(xs, ys, s=5, c='red', marker='o')  # s controls the size of the dots
    plt.title(f'Image with {len(keypoints)} Keypoints')
    plt.axis('off')
    plt.show()

def quick_disparity_check(left_img, right_img, method='SGBM', show_result=True):
    """
    Compute a quick disparity map from rectified stereo images using OpenCV.
    
    Args:
        left_img (np.ndarray):  Rectified left image (grayscale or color).
        right_img (np.ndarray): Rectified right image (grayscale or color).
        method (str): 'BM' or 'SGBM' to choose the block matching method.
        show_result (bool): If True, display the disparity map in a window.
    
    Returns:
        disparity (np.ndarray): The raw disparity map, same size as the input images.
    """
    # Convert to grayscale if needed
    if len(left_img.shape) == 3 and left_img.shape[2] == 3:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_img
    
    if len(right_img.shape) == 3 and right_img.shape[2] == 3:
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    else:
        right_gray = right_img
    
    # Choose stereo matcher
    if method.upper() == 'BM':
        # StereoBM typically works with 8-bit images only
        stereo = cv2.StereoBM_create(
            numDisparities=64,  # must be multiple of 16
            blockSize=20       # larger blockSize -> smoother disparity
        )
    else:
        # StereoSGBM (more robust, but slower)
        # Some typical parameter choices:
        min_disp = 0
        num_disp = 128  # must be multiple of 16
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=5,
            P1=8 * 3 * 5**2,      # 8 * number_of_image_channels * blockSize^2
            P2=32 * 3 * 5**2,     # 32 * number_of_image_channels * blockSize^2
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=1
        )

    # Compute disparity (the result is often a 16-bit signed single-channel image)
    disparity_16S = stereo.compute(left_gray, right_gray)
    
    # For visualization, you typically convert disparity to float and then normalize
    disparity = disparity_16S.astype(np.float32) / 16.0
    
    if show_result:
        # Normalize for display
        disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, 
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_vis = cv2.applyColorMap(disp_vis, 4)
        cv2.imshow('Disp', disp_vis); cv2.waitKey(0); cv2.destroyAllWindows()
    
    return disparity

def compute_disparity(
    left_img: np.ndarray,
    right_img: np.ndarray,
    method: str = 'SGBM',
    num_disparities: int = 16*5,
    block_size: int = 5,
    pre_filter_cap: int = 31,
    uniqueness_ratio: int = 15,
    speckle_window_size: int = 100,
    speckle_range: int = 2,
    disp12_max_diff: int = 1,
    gauss_kernel: int = 5,
    gauss_sigma: float = 1
) -> np.ndarray:
    """
    Compute disparity map from two rectified stereo images.

    Parameters
    ----------
    left_img : np.ndarray
        Left rectified image (color or grayscale).
    right_img : np.ndarray
        Right rectified image (same size & type as left_img).
    method : {'BM', 'SGBM'}
        Algorithm: 'BM' for StereoBM, 'SGBM' for StereoSGBM.
    num_disparities : int
        Max disparity minus min disparity (>0 and divisible by 16).
    block_size : int
        Matched block size. It must be odd and >=1.
    pre_filter_cap : int
        Truncation value for the prefiltered image pixels (StereoBM only).
    uniqueness_ratio : int
        Margin in percentage by which the best cost function value should "win" the second best to accept the match.
    speckle_window_size : int
        Maximum size of smooth disparity regions to consider speckles and invalidate.
    speckle_range : int
        Maximum disparity variation within each connected component.
    disp12_max_diff : int
        Maximum allowed difference in the left-right disparity check.

    Returns
    -------
    disp_norm : np.ndarray
        8-bit normalized disparity map (0..255).
    """

    # 1) Convert to grayscale if needed
    if len(left_img.shape) == 3:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    else:
        left_gray, right_gray = left_img, right_img

    # 2) Optional Gaussian smoothing
    if gauss_kernel > 0 and gauss_kernel % 2 == 1:
        left_gray  = cv2.GaussianBlur(left_gray,  (gauss_kernel, gauss_kernel), gauss_sigma)
        right_gray = cv2.GaussianBlur(right_gray, (gauss_kernel, gauss_kernel), gauss_sigma)

    # 3) Create matcher
    if method.upper() == 'BM':
        # StereoBM parameters
        matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
        matcher.setPreFilterCap(pre_filter_cap)
        matcher.setUniquenessRatio(uniqueness_ratio)
        matcher.setSpeckleWindowSize(speckle_window_size)
        matcher.setSpeckleRange(speckle_range)
        matcher.setDisp12MaxDiff(disp12_max_diff)
    elif method.upper() == 'SGBM':
        # StereoSGBM parameters
        matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 1 * block_size**2,
            P2=32 * 1 * block_size**2,
            disp12MaxDiff=disp12_max_diff,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size,
            speckleRange=speckle_range,
            preFilterCap=pre_filter_cap,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    else:
        raise ValueError("Unsupported method: choose 'BM' or 'SGBM'")

    # 3) Compute raw disparity (16-bit signed)
    disparity = matcher.compute(left_gray, right_gray).astype(np.float32)
    print(disparity)
    disparity = disparity / 16
    disparity[disparity <= 0] = np.nan                                      # mask out invalid
    vmin, vmax = np.nanpercentile(disparity, [5, 95])                   # ignore outliers
    disp_norm = np.uint8(255 * (np.clip(disparity, vmin, vmax) - vmin) / (vmax - vmin))

    return disp_norm

def adjust_yaw(image, angle):
    """
    Adjusts the yaw of an image by applying a small rotation around its center.

    Parameters:
        image (numpy.ndarray): The input right image.
        angle (float): The yaw adjustment angle in degrees (positive for counterclockwise).

    Returns:
        numpy.ndarray: The transformed image with yaw adjusted.
    """
    # Get image dimensions
    h, w = image.shape[:2]

    # Define the rotation matrix
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

    return rotated_image

def overlay_stereo_images(left_img, right_img, K_left, K_right, R, baseline, plane_depth=1.0, blend=0.5):
    """
    Projects the right image onto the left camera image plane using the provided camera parameters.
    
    Parameters:
      left_img (np.ndarray): Left image.
      right_img (np.ndarray): Right image.
      K_left (np.ndarray): Intrinsic matrix for the left camera (3x3).
      K_right (np.ndarray): Intrinsic matrix for the right camera (3x3).
      R (np.ndarray): Rotation matrix (3x3) from the right camera to the left camera.
      T (np.ndarray): Translation vector (3x1 or 3, ) from the right camera to the left camera.
      plane_depth (float): Assumed depth of the scene plane in the right camera coordinate system.
      blend (float): Blending factor for overlaying the images.
    
    Returns:
      overlay (np.ndarray): An image overlaying the left image with the warped right image.
    
    Note:
      This function approximates a full pixel-to-pixel mapping by computing a homography from the four
      corners of the right image. It works well when the scene is nearly planar at the chosen depth.
    """
    T = np.array([baseline, 0, 0]) / 1000
    # Get the size of the right image
    h_r, w_r = right_img.shape[:2]
    
    # Define corner points in the right image (in pixel coordinates)
    pts_right = np.array([[0, 0],
                          [w_r - 1, 0],
                          [w_r - 1, h_r - 1],
                          [0, h_r - 1]], dtype=np.float32)
    
    # Function to project a point from the right image to the left image,
    # assuming the point lies on a plane at Z = plane_depth in the right camera coordinate system.
    def project_point(pt):
        u, v = pt
        # Convert pixel coordinate to normalized coordinates in the right camera
        x_norm = np.linalg.inv(K_right) @ np.array([u, v, 1.0])
        # Scale so that the Z-coordinate becomes plane_depth
        scale = plane_depth / x_norm[2]
        X_right = x_norm * scale  # 3D point in the right camera frame
        
        # Transform the point to the left camera coordinate system
        X_left = R @ X_right + T.reshape(3)
        # Project into the left image
        x_proj = K_left @ X_left
        x_proj /= x_proj[2]
        return [x_proj[0], x_proj[1]]
    
    # Compute the corresponding points in the left image for the corners of the right image
    pts_left = np.array([project_point(pt) for pt in pts_right], dtype=np.float32)
    
    # Compute the homography from the right image to the left image using the four corresponding points
    H, status = cv2.findHomography(pts_right, pts_left)
    
    # Warp the right image using the computed homography
    h_l, w_l = left_img.shape[:2]
    warped_right = cv2.warpPerspective(right_img, H, (w_l, h_l))
    
    # Overlay the warped right image and the left image
    overlay = cv2.addWeighted(left_img, blend, warped_right, 1 - blend, 0)
    
    return overlay