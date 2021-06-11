from utils.heatmap import extract_image, overlay
from model.avolnet import AVOLNet
from torch.utils.data import DataLoader, Dataset
from utils.dataset import AudioSet
from utils.util import reverseTransform, bgr2rgb
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def localize_sound(model_path):
    dataset = AudioSet("test", "./data/test/video", "./data/test/audio")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print("Loading data.")

    model = AVOLNet()
    # Load from before
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loading from previous checkpoint.")

    model.eval()
    # 0: True, 1: False correspondence
    # Audio-visual localization map on true positive pairs
    for i, data in enumerate(dataloader):    
        img, aud, res = data
        if res.item() != 0:  # Exclude false pairs
            continue
        with torch.no_grad():
            res, loc = model(img, aud)
            if res.item() != 0:  # Exclude false negative pairs
                continue
            img, aud = reverseTransform(img, aud)
            img = bgr2rgb(img)
            img = np.transpose(img[0], (1, 2, 0))
            img = img.numpy()
            img = img / img.max()
            img = 255 * img
            img = img.astype(np.uint8)
            cv2.imwrite("localization/origin_img_{}.png".format(i), img)
            result = overlay(img, loc[0][0])
            cv2.imwrite("localization/heatmap_result_{}.png".format(i), result)
            res = input("Continue?(y/n) : ")

            if res == "n":
                break
            else:
                print("Save origin and heatmap images successfully.")


if __name__ == "__main__":
    model_path = "./save/AVOL-Net_inst.pt"

    localize_sound(model_path)
