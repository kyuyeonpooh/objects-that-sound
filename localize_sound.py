from utils.heatmap import extract_image, overlay
from model.avolnet import AVOLNet
from torch.utils.data import DataLoader, Dataset
from utils.dataset import AudioSet
from utils.util import reverseTransform
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
    for i, data in enumerate(dataloader):
        img, aud, res = data
        if res.item() == 0:
            continue
        with torch.no_grad():
            res, loc = model(img, aud)
            if res.item() == 0:
                continue
            img, aud = reverseTransform(img, aud)
            img = np.transpose(img[0], (1,2,0))
            img = img.numpy()
            img = img / img.max()
            img = 255 * img
            img = img.astype(np.uint8)            
            cv2.imwrite("origin_img.png", img)
            result = overlay(img, loc[0][0])
            cv2.imwrite("heatmap_result.png", result)
            res = input("Continue?(y/n) : ")
            
            if res == "n":
                break
            else:
                print("Save origin and heatmap images successfully.")

if __name__ == "__main__":
    model_path = "./save/avol_model/AVOL_train_model_120.pt"
    
    localize_sound(model_path)