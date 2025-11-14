import pandas as pd
import numpy as np
import math
from utils import visualization

# --- User Configuration ---
CSV_PATH   = r"C:\Python Projects\Adv. Robotic Vision\rosbag_mid_tower_1.csv"
COL_NAMES  = ['Frame','Sub Frame','RX','RY','RZ','TX','TY','TZ']

fps_vicon = 400
fps_camera = 10


step = int(round(fps_vicon / fps_camera))
# 1) Load & clean
df_full = pd.read_csv(
    CSV_PATH,
    skiprows=6,
    header=None,
    names=COL_NAMES
)

start_trim = 700
end_trim = 8900
df_cut = df_full.iloc[start_trim : len(df_full)-end_trim].reset_index(drop=True)
df = df_cut.iloc[::step].reset_index(drop=True)
camera_pose_plot = []
for _, row in df.iterrows():
    # rotation vector as a 3×1 array
    rvec = np.array([row['RX'],
                     row['RZ'],
                     row['RY']])
    # translation vector as a 3×1 array
    tvec = np.array([row['TX']/1000,
                     row['TZ']/1000,
                    row['TY']/1000])
    camera_pose_plot.append((rvec, tvec))

visualization.plot_trajectory_2d(camera_pose_plot, "Robot Trajectory", arrows=False)


