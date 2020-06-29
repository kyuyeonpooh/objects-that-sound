import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.L3 import L3Net
from model.avenet import AVENet
from model.avolnet import AVOLNet
from utils.dataset import AudioSet


def test(name_of_run, model_epoch, model_name):
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        if use_cuda:
            print("Failed to find GPU, using CPU instead.")
        device = torch.device("cpu")

    if model_name == "AVE":
        model = AVENet()
    elif model_name == "L3":
        model = L3Net()
    elif model_name == "AVOL":
        model = AVOLNet()
    else:
        raise ValueError("Unexpected model name.")
    model.load_state_dict(torch.load("/hdd/save/{}/{}_{}.pt".format(name_of_run, name_of_run, model_epoch)))
    model.to(device)
    model.eval()

    test_correct = 0
    test_total = 0

    test = AudioSet("test", "./data/test/video", "./data/test/audio")
    test_loader = DataLoader(test, batch_size=64, num_workers=6, pin_memory=True)

    for i, (img, aud, label) in enumerate(test_loader):
        img, aud, label = img.to(device), aud.to(device), label.to(device)
        with torch.no_grad():
            if model_name == "AVE" or model_name == "L3":
                out, _, _ = model(img, aud)
            else:
                out, _ = model(img, aud)
                label = label.float()
            if model_name == "AVE" or model_name == "L3":
                prediction = torch.argmax(out, dim=1)
            else:
                prediction = torch.round(out)
            test_correct += (label == prediction).sum().item()
            test_total += label.size(0)

    return test_correct / test_total


if __name__ == "__main__":
    name_of_run = "L3_train_augment"
    model_name = "L3"  # AVE / AVOL / L3
    range_min, range_max = 80, 200
    max_test_acc = 0
    max_test_acc_i = 0
    for i in range(range_min, range_max):
        print("{}_{}.pt".format(name_of_run, i))
        test_acc = test(name_of_run, i, model_name)
        print("Test accuracy:", test_acc)
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            max_test_acc_i = i
    print("Max test accuracy:", max_test_acc)
    print("Max test accuracy in", max_test_acc_i)
