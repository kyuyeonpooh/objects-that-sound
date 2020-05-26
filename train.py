import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.avenet import AVENet
from utils.dataset import AudioSet


def train(use_cuda=True, epoch=500, lr=1e-4, weight_decay=1e-5):
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for training:", torch.cuda.get_device_name())
    else:
        device = torch.device("cpu")
    print("Current device:", device)

    avenet = AVENet()
    avenet = avenet.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(avenet.parameters(), lr=lr, weight_decay=weight_decay)

    audioset = AudioSet("./data/video", "./data/audio")
    audioset_val = AudioSet("./data/video", "./data/audio", val=True)
    dataloader = DataLoader(audioset, batch_size=64, shuffle=True, num_workers=6, pin_memory=True)
    dataloader_val = DataLoader(audioset_val, batch_size=64, shuffle=True, num_workers=6)

    logfile = open("log.txt", "w")

    for e in range(epoch):
        print("Epoch", e + 1)
        train_loss = 0
        train_correct = 0
        for i, (img, aud, label) in enumerate(dataloader):
            optimizer.zero_grad()
            img, aud, label = img.to(device), aud.to(device), label.to(device)
            out = avenet(img, aud)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                out_indices = torch.argmax(out, dim=1)
                train_loss += loss.item()
                train_correct += (label == out_indices).sum().item()

            # log per 100 subepochs
            if (i + 1) % 100 == 0:
                print(out_indices)
                with torch.no_grad():
                    # test on 10 val batches
                    val_loss = 0
                    val_correct = 0
                    for j, (imgval, audval, labelval) in enumerate(dataloader_val):
                        if j == 10:
                            break
                        imgval, audval, labelval = imgval.to(device), audval.to(device), labelval.to(device)
                        outval = avenet(imgval, audval)
                        val_loss += criterion(outval, labelval).item()
                        val_correct += (labelval == torch.argmax(outval, dim=1)).sum().item()
                    log = "Epoch: {}, Subepoch: {}, Loss: {:.4f}, Acc: {:.4f}, Val_loss:{:.4f}, Val_acc: {:.4f}\n".format(
                        e + 1,
                        i + 1,
                        train_loss / 100,
                        train_correct / (64 * 100),
                        val_loss / 10,
                        val_correct / (64 * 10),
                    )
                    print(log, end="")
                    logfile.write(log)
                    train_loss = 0
                    train_correct = 0
            # save per 1000 subepochs
            if (i + 1) % 1000 == 0:
                torch.save(avenet.state_dict(), "./save/ave_{}_{}.pt".format(e + 1, i + 1))


if __name__ == "__main__":
    train()
