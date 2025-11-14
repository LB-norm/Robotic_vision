from PIL import Image
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

def fuze_env():
    # 1) List your files
    paths = [
        "Enviroment_measures/Screenshot1.png",
        "Enviroment_measures/Screenshot2.png",
        "Enviroment_measures/Screenshot3.png",
        "Enviroment_measures/Screenshot4.png",
        "Enviroment_measures/Screenshot5.png"
    ]

    # 2) Load and convert to RGBA
    layers = [Image.open(p).convert("RGBA") for p in paths]

    # 3) Give each layer an equal alpha so they all “show through”
    alpha_value = int(255 / len(layers))   # for 5 images → 51

    for img in layers:
        img.putalpha(alpha_value)

    # 4) Composite them one by one onto a transparent background
    base = Image.new("RGBA", layers[0].size, (0, 0, 0, 0))
    for img in layers:
        base = Image.alpha_composite(base, img)

    arr = np.array(base)   # shape (H, W, 4)

    # 2) Split channels
    r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]

    # 3) Build a boolean mask for “orange” pixels
    #    (tweak these thresholds to match your exact hue/saturation)
    orange_mask = (
        (r > 10) &            # red channel fairly high
        (g >  10) &            # green moderate
        (b < 150)              # blue low
    )


    # 4) Choose a boost factor
    boost = 20.0  # 2× brighter (you can try 1.5, 3.0, …)

    # 5) Multiply only the orange pixels (and clip back into [0,255])
    arr[orange_mask, 0] = np.clip(r[orange_mask] * boost, 0, 255)  # R
    arr[orange_mask, 1] = np.clip(g[orange_mask] * boost, 0, 255)  # G
    arr[orange_mask, 2] = np.clip(b[orange_mask] * boost, 0, 255)  # B
    # (optionally boost alpha too)
    arr[orange_mask, 3] = np.clip(a[orange_mask] * boost, 0, 255)

    # 6) Convert back to uint8 and save
    out = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    out.save("highlight_boosted.png")
    out.show()

    
def overlay_with_vid(vid_path, map_path):
    # Re-introduce the old names MoviePy expects:
    Image.ANTIALIAS = Image.Resampling.LANCZOS
    Image.BICUBIC  = Image.Resampling.BICUBIC
    Image.BILINEAR = Image.Resampling.BILINEAR
    # 1) Load your robot video
    video = VideoFileClip(vid_path)

    # 2) Load your map as a clip, match duration & size, set opacity
    map_clip = (
        ImageClip(map_path)
        .set_duration(video.duration)
        .resize(video.size)        # only if needed
        .set_opacity(0.4)          # fade background a bit
    )

    # 3) Composite and write out
    final = CompositeVideoClip([video, map_clip])
    final.write_videofile("fused2.mp4", codec="libx264", audio_codec="aac")

#fuze_env()
vid_path = "Vicon_Tracker_crop.mp4"
map_path = "highlight_boosted3.png"
overlay_with_vid(vid_path, map_path)