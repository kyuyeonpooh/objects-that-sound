# ======================================================================
# [Note] Please run this file in the root directory.
# [Example] ~/objects-that-sound$ python train.py           (O)
#           ~/objects-that-sound/utils$ python ../train.py  (X)
# ======================================================================

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
from model.L3 import L3Net
from utils.dataset import AudioSet


def train(
    name_of_run,
    train_vid_dir,
    train_aud_dir,
    val_vid_dir,
    val_aud_dir,
    use_cuda=True,
    epoch=500,
    batch_size=64,
    ncpu=8,
    lr=5e-5,
    weight_decay=1e-5,
    use_lr_scheduler=True,
    csv_log_dir="log/",
    model_save_dir="/hdd/save/L3_train_augment",
    model_name="L3",
    **kwargs
):
    # gpu settings
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for training:", torch.cuda.get_device_name())
    else:
        if use_cuda:
            print("Failed to find GPU, using CPU instead.")
        device = torch.device("cpu")
    print("Current device:", device)

    # model, loss, and optimizer settings
    if model_name == "AVE":
        model = AVENet()
    elif model_name == "AVOL":
        model = AVOLNet()
    elif model_name == "L3":
        model = L3Net()
    else:
        raise ValueError("Unkown model name.")
    model.to(device)

    if model_name == "AVE" or model_name == "L3":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if use_lr_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=16, gamma=0.94)  # lr decreases by 6% every 16 epochs

    # dataset, dataloader settings
    train = AudioSet("train", train_vid_dir, train_aud_dir, **kwargs)
    val = AudioSet("val", val_vid_dir, val_aud_dir, **kwargs)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=ncpu, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=ncpu, pin_memory=True)

    # log file and tensorboard settings
    log_file = open(os.path.join(csv_log_dir, name_of_run + ".csv"), "w")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    tensorboard = SummaryWriter(os.path.join("runs", name_of_run))
    img_rand, aud_rand = torch.rand(batch_size, 3, 224, 224).to(device), torch.rand(batch_size, 1, 257, 199).to(device)
    tensorboard.add_graph(model, input_to_model=(img_rand, aud_rand))

    # train with validation
    for e in range(epoch):
        print("Epoch", e + 1)

        # train
        train_loss = 0
        train_correct = 0
        train_total = 0
        model.train()
        for i, (img, aud, label) in enumerate(train_loader):
            optimizer.zero_grad()
            img, aud, label = img.to(device), aud.to(device), label.to(device)
            if model_name == "AVE" or model_name == "L3":
                out, _, _ = model(img, aud)
            else:
                out, _ = model(img, aud)
                label = label.float()
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if model_name == "AVE" or model_name == "L3":
                    prediction = torch.argmax(out, dim=1)
                else:
                    prediction = torch.round(out)
                train_loss += loss.item()
                train_correct += (label == prediction).sum().item()
                train_total += label.size(0)

            if (i + 1) % 100 == 0:
                train_loss /= 100
                train_acc = train_correct / train_total
                # print("train_loss: {:.4f}, train_acc: {:.4f}".format(train_loss, train_acc))

                val_loss = 0
                val_correct = 0
                val_total = 0
                for j, (img, aud, label) in enumerate(val_loader):
                    img, aud, label = img.to(device), aud.to(device), label.to(device)
                    with torch.no_grad():
                        if model_name == "AVE" or model_name == "L3":
                            out, _, _ = model(img, aud)
                        else:
                            out, _ = model(img, aud)
                            label = label.float()
                        loss = criterion(out, label)
                        if model_name == "AVE" or model_name == "L3":
                            prediction = torch.argmax(out, dim=1)
                        else:
                            prediction = torch.round(out)
                        val_loss += loss.item()
                        val_correct += (label == prediction).sum().item()
                        val_total += label.size(0)
                    if j == 9:
                        break
                val_loss /= 10
                val_acc = val_correct / val_total
                csv_writer.writerow([e + 1, train_loss, train_acc, val_loss, val_acc])
                tensorboard.add_scalar("train_loss", train_loss, global_step=e * len(train_loader) + i + 1)
                tensorboard.add_scalar("train_acc", train_acc, global_step=e * len(train_loader) + i + 1)
                tensorboard.add_scalar("val_loss", val_loss, global_step=e * len(train_loader) + i + 1)
                tensorboard.add_scalar("val_acc", val_acc, global_step=e * len(train_loader) + i + 1)
                print(
                    "train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                        train_loss, train_acc, val_loss, val_acc
                    )
                )

                train_loss = 0
                train_correct = 0
                train_total = 0
        """
        # validation
        val_loss = 0
        val_correct = 0
        val_total = 0
        model.eval()
        for img, aud, label in tqdm(val_loader, desc="Val"):
            img, aud, label = img.to(device), aud.to(device), label.to(device)
            with torch.no_grad():
                out, _, _ = model(img, aud)
                loss = criterion(out, label)
                prediction = torch.argmax(out, dim=1)
                val_loss += loss.item()
                val_correct += (label == prediction).sum().item()
                val_total += label.size(0)
        """

        # update lr_scheduler
        scheduler.step()

        # write log
        """
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        csv_writer.writerow([e + 1, train_loss, train_acc, val_loss, val_acc])
        tensorboard.add_scalar("train_loss", train_loss, global_step=e + 1)
        tensorboard.add_scalar("train_acc", train_acc, global_step=e + 1)
        tensorboard.add_scalar("val_loss", val_loss, global_step=e + 1)
        tensorboard.add_scalar("val_acc", val_acc, global_step=e + 1)
        print(
            "train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                train_loss, train_acc, val_loss, val_acc
            )
        )
        """

        # save model weight
        torch.save(model.state_dict(), os.path.join(model_save_dir, name_of_run + "_{}.pt".format(e + 1)))

    log_file.close()


if __name__ == "__main__":
    train(
        "L3_train_augment",
        "./data/train/video",
        "./data/train/audio",
        "./data/val/video",
        "./data/val/audio",
        model_name="L3",
    )
