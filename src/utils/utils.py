import bagpy
import os
import cv2
from bagpy import bagreader
import pandas as pd
import numpy as np

def extract_data():
    b = bagreader(r"C:\Python_Projects\Adv. Robotic Vision\2025-05-08-16-30-09.bag")
    print(b.topic_table)
    for topic in b.topics:
        test = b.message_by_topic(topic)
        print(test)

# def convert_stereo_data_to_img(skiprows, nrows):
#     raw_stereo_data_df = pd.read_csv(r"C:\Python Projects\Adv. Robotic Vision\2025-05-08-16-30-09\zed-zed_node-stereo_raw-image_raw_color.csv", nrows=nrows)
#     converted_images = []
#     for i in range(len(raw_stereo_data_df)):
#         raw_string = raw_stereo_data_df["data"].iloc[i]
#         if raw_string.startswith("b'") and raw_string.endswith("'"):
#             raw_string = raw_string[2:-1]
#         test_img = bytes(raw_string.encode('utf-8').decode('unicode_escape'), 'latin-1')
#         comb_width = 1280
#         height = 360
#         channels = 4 #BGRA
#         image_array = np.frombuffer(test_img, dtype=np.uint8)
#         image_bgra = image_array.reshape((height, comb_width, channels))
#         image_bgr = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR)
#         converted_images.append(image_bgr)
    
#     return converted_images

def convert_stereo_data_to_img(skiprows=None, nrows=None):
    # build a skip‐list if we were passed a tuple (start, end)…
    if isinstance(skiprows, tuple) and len(skiprows) == 2:
        start, end = skiprows
        skip_list = list(range(start, end+1))
    else:
        skip_list = skiprows

    # explicitly say header=0 so pandas still reads the first row as column names
    raw = pd.read_csv(
        r"C:\Python_Projects\Adv._Robotic_Vision\2025-05-08-16-30-09\zed-zed_node-stereo_raw-image_raw_color.csv",
        header=0,
        skiprows=skip_list,
        nrows=nrows
    )

    images = []
    for raw_string in raw["data"]:
        # strip the b'…' wrapper
        if raw_string.startswith("b'") and raw_string.endswith("'"):
            raw_string = raw_string[2:-1]

        # decode back to bytes
        blob = bytes(raw_string.encode('utf-8').decode('unicode_escape'), 'latin-1')

        # reshape & convert
        comb_width, height, channels = 1280, 360, 4
        arr = np.frombuffer(blob, dtype=np.uint8)
        img_bgra = arr.reshape((height, comb_width, channels))
        img_bgr  = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
        images.append(img_bgr)

    return images

def invert_pose(rvec, tvec):    #Funktioniert
    """
    Invert a transform given by Rodrigues rotation vector (rvec) and translation vector (tvec).
    Returns (rvec_inv, tvec_inv).
    """
    R, _ = cv2.Rodrigues(rvec)    # 3x3 rotation from camera to tag
    R_inv = R.T
    t_inv = -R_inv @ tvec
    rvec_inv, _ = cv2.Rodrigues(R_inv)
    return rvec_inv.flatten(), t_inv.flatten()
    # return rvec, tvec

def compose_poses(rvecA, tvecA, rvecB, tvecB):  #Funktioniert
    """
    Return the transform for A->B = A->G ∘ B->A?
    Actually we want: T_AB = T_A ∘ T_B
    If T_A transforms from some frame A to global,
    and T_B transforms from some frame B to A,
    the result is from B to global.

    So (rvec_out, tvec_out) = Compose(rvecA, tvecA, rvecB, tvecB).
    """
    # print(f"TEST INPUT: {rvecA, tvecA, rvecB, tvecB}")
    RA, _ = cv2.Rodrigues(rvecA)
    RB, _ = cv2.Rodrigues(rvecB)
    RC = RA @ RB
    tC = RA @ tvecB + tvecA
    rvecC, _ = cv2.Rodrigues(RC)
    # print(f"TEST OUTPUT: {rvecC.flatten(), tC.flatten()}")
    return rvecC.flatten(), tC.flatten()

