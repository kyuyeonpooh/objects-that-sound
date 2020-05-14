import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.avenet import AVENet
from util.dataset import AudioSet


def train(use_cuda=True, epoch=1000, lr=0.25e-4):
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for training:", torch.cuda.get_device_name())
    else:
        device = torch.device("cpu")
    print("Current device:", device)

    avenet = AVENet()
    avenet.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(avenet.parameters(), lr=lr)

    audioset = AudioSet("./data/video", "./data/audio")
    dataloader = DataLoader(audioset, batch_size=64, shuffle=True, num_workers=6, pin_memory=True)

    logfile = open("log.txt", "w")

    for e in range(epoch):
        print("Epoch", e + 1)
        for i, (img, aud, label) in enumerate(dataloader):
            optimizer.zero_grad()
            img, aud, label = img.to(device), aud.to(device), label.to(device)
            out = avenet(img, aud)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                with torch.no_grad():
                    print(torch.nn.functional.softmax(out)[:10])
                    print(torch.argmax(torch.nn.functional.softmax(out), dim=1))
                    print(label)
                    print(avenet.fc3.weight, avenet.fc3.bias)
                    out_indices = torch.argmax(out, dim=1)
                    acc = (label == out_indices).sum().item() / out_indices.size(0)
                    log = "Epoch: {}, Subepoch: {}, Loss: {:.4f}, Acc: {:.4f}\n".format(e + 1, i + 1, loss.item(), acc)
                    print(log, end="")
                    logfile.write(log)
        torch.save(avenet.state_dict(), "./save/ave_" + str(e + 1) + ".pt")


if __name__ == "__main__":
    train()
