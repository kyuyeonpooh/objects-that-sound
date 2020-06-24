import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.avenet import AVENet
from model.avolnet import AVOLNet
from utils.dataset import AudioSet


def test(model_epoch):
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        if use_cuda:
            print("Failed to find GPU, using CPU instead.")
        device = torch.device("cpu")

    model_name = "AVE"

    if model_name == "AVE":
        model = AVENet()
    else:
        model = AVOLNet()
    model.load_state_dict(torch.load("/hdd/save/AVE_train_no_init/AVE_train_no_init_{}.pt".format(model_epoch)))
    model.to(device)
    model.eval()

    test_correct = 0
    test_total = 0

    test = AudioSet("test", "./data/test/video", "./data/test/audio")
    test_loader = DataLoader(test, batch_size=64, num_workers=8, pin_memory=True)

    for i, (img, aud, label) in enumerate(tqdm(test_loader)):
        img, aud, label = img.to(device), aud.to(device), label.to(device)
        with torch.no_grad():
            if model_name == "AVE":
                out, _, _ = model(img, aud)
            else:
                out, _ = model(img, aud)
                label = label.float()
            if model_name == "AVE":
                prediction = torch.argmax(out, dim=1)
            else:
                prediction = torch.round(out)
            test_correct += (label == prediction).sum().item()
            test_total += label.size(0)

    return test_correct / test_total


if __name__ == "__main__":
    max_test_acc = 0
    max_test_acc_i = 0
    for i in range(40, 250):
        print("AVE_train_no_init_{}.pt".format(i))
        test_acc = test(i)
        print("Test accuracy:", test_acc)
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            max_test_acc_i = i
    print("Max test accuracy:", max_test_acc)
    print("Max test accuracy in", max_test_acc_i)
