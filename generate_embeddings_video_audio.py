from model.avenet import AVENet
from torch.utils.data import DataLoader, Dataset
from utils.util import reverseTransform
from utils.dataset import AudioSet
import os
import torch


def generateEmbeddingsForVideoAudio(model_name, use_cuda, use_tags):
    # Get video embeddings on the test set
    dataset = AudioSet("embedding", "./data/test/video", "./data/test/audio")
    dataloader = DataLoader(dataset, batch_size=1)
    print("Loading data.")
    # for img, aud, res, vidTags, audTags, audioSample in dataloader:
    # 	break

    model = getAVENet(use_cuda)

    # Load from before
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name))
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

        if i == 6000:
            break

    if use_tags:
        torch.save(
            [imgList, audList, imgEmbedList, audEmbedList, vidTagList, audTagList, audioSampleList],
            "savedEmbeddings.pt",
        )
    else:
        torch.save([imgList, audList, imgEmbedList, audEmbedList], "savedEmbeddings.pt")


def getAVENet(use_cuda=True):
    model = AVENet()
    if use_cuda:
        model = model.cuda()

    return model


if __name__ == "__main__":
    model_path = "/hdd/save/AVE_train_augment_80.pt"
    generateEmbeddingsForVideoAudio(model_name=model_path, use_cuda=True, use_tags=True)
