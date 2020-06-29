import os

import torch
from torch.utils.data import DataLoader, Dataset

from model.L3 import L3Net
from model.avenet import AVENet
from utils.dataset import AudioSet
from utils.util import reverseTransform


def generateEmbeddingsForVideoAudio(model_name, model_path, emb_path, use_cuda=True, use_tags=True):
    # Get video embeddings on the test set
    dataset = AudioSet("embedding", "./data/test/video", "./data/test/audio")
    dataloader = DataLoader(dataset, batch_size=1)
    print("Loading data.")

    if model_name == "AVE":
        model = getAVENet(use_cuda)
    elif model_name == "L3":
        model = getL3Net(use_cuda)
    else:
        raise ValueError("Unknown model name.")

    # Load from before
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loading from previous checkpoint.")

    imgList, audList, resList, vidTagList, audTagList = [], [], [], [], []
    imgEmbedList, audEmbedList = [], []
    audioSampleList = []

    model.eval()
    for i, data in enumerate(dataloader):
        # Filter the bad ones first
        if use_tags:
            img, aud, res, vidTag, audTag, audSamples = data
        else:
            img, aud, res = data

        # res = res.squeeze(1)
        idx = (res != 2).numpy().astype(bool)
        if idx.sum() == 0:
            continue

        # Find the new variables
        img = torch.Tensor(img.numpy()[idx, :, :, :])
        aud = torch.Tensor(aud.numpy()[idx, :, :, :])
        res = torch.LongTensor(res.numpy()[idx])

        if use_tags:
            vidTag = torch.cat(tuple(x for x in vidTag)).numpy()
            audTag = torch.cat(tuple(x for x in audTag)).numpy()
            audSamples = audSamples.numpy()[idx]

        # with torch.no_grad():
        img = img.clone().detach()
        aud = aud.clone().detach()
        res = res.clone().detach()

        # M = img.shape[0]
        if use_cuda:
            img = img.cuda()
            aud = aud.cuda()
            res = res.cuda()

        o, imgEmbed, audEmbed = model(img, aud)
        _, ind = o.max(1)

        # Grab the correct indices
        idx = ((res == 0)).data.cpu().numpy().astype(bool)

        if idx[0]:
            # img, aud = reverseTransform(img, aud)
            # imgList.append(img.data.cpu().numpy()[idx, :])
            # audList.append(aud.data.cpu().numpy()[idx, :])
            imgEmbedList.append(imgEmbed.data.cpu().numpy())
            audEmbedList.append(audEmbed.data.cpu().numpy())
            if use_tags:
                vidTagList.append(vidTag)
                audTagList.append(audTag)
                audioSampleList.append(audSamples)

    if use_tags:
        torch.save([imgList, audList, imgEmbedList, audEmbedList, vidTagList, audTagList, audioSampleList], emb_path)
    else:
        torch.save([imgList, audList, imgEmbedList, audEmbedList], emb_path)


def getAVENet(use_cuda=True):
    model = AVENet()
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        if use_cuda:
            print("Failed to find GPU, using CPU instead.")
        device = torch.device("cpu")
    if use_cuda:
        model.to(device)
    return model


def getL3Net(use_cuda=True):
    model = L3Net()
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name())
    else:
        if use_cuda:
            print("Failed to find GPU, using CPU instead.")
        device = torch.device("cpu")
    if use_cuda:
        model.to(device)
    return model


if __name__ == "__main__":
    model_name = "L3"
    model_path = "./save/L3-Net_augment_inst.pt"
    emb_path = "./embedding/L3_aug_inst.pt"
    generateEmbeddingsForVideoAudio(model_name, model_path=model_path, emb_path=emb_path, use_cuda=True, use_tags=True)
