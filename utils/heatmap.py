import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


def extract_image(video_path):
    # load video
    video_npz = np.load(video_path)

    # extract a frame from the video
    vid_frame = video_npz["4"]

    # setting image configuration
    vid_frame = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2RGB)
    vid_frame = cv2.resize(vid_frame, dsize=(224, 224))

    return vid_frame


def overlay(img, data):
    # convert heatmap data into ndarray
    data = np.asarray(data) * 255
    data = np.array(data, dtype=np.uint8)

    # convert heatmap according to cv2.COLORMAP_HOT
    # reference link: https://docs.opencv.org/2.4/modules/contrib/doc/facerec/colormaps.html
    heatmap = cv2.applyColorMap(data, cv2.COLORMAP_HOT)

    # resize heatmap to be same with img
    heatmapx16 = cv2.resize(heatmap, None, fx=16, fy=16, interpolation=cv2.INTER_AREA)

    # overlay img and heatmap
    dst = cv2.addWeighted(img, 0.5, heatmapx16, 0.5, 0)

    return dst


if __name__ == "__main__":
    # set sample video path
    video_path = "data/video/5k4ajti-d-c.npz"

    # extract sample image from the video
    img = extract_image(video_path)
    cv2.imwrite("guitar.png", img)

    # example heatmap array
    data = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.6, 0.6, 0.8, 0.8, 0.8, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.8, 0.8, 0.81, 1, 0.8, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.8, 1, 1, 1, 1, 1, 0.8, 0, 0, 0, 0],
        [0, 0, 0, 0.8, 1, 1, 1, 1, 1, 0.8, 0.6, 0, 0, 0],
        [0, 0, 0, 0.8, 1, 1, 1, 1, 1, 0.8, 0.6, 0, 0, 0],
        [0, 0, 0, 0.8, 0.8, 1, 1, 1, 0.6, 0.8, 0.6, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.6, 0.8, 0.8, 0.8, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    # overlay headtmap and image
    dst = overlay(img, data)
    cv2.imwrite("heatmap.png", dst)
