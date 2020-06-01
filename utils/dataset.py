# ======================================================================
# [Note] Running this file is not mandatory.
#        If you want to run this file for testing batch generation,
#        please run this file in the root directory.
# [Example] ~/objects-that-sound$ python utils/dataset.py       (O)
#           ~/objects-that-sound/utils$ python dataset.py       (X)
# ======================================================================


import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class AudioSet(Dataset):
    def __init__(self, mode, src_vid_npz_dir, src_aud_npz_dir, nseg=9, csv="./csv/label.csv", **kwargs):
        if mode == "train":
            self.train = True
        elif mode == "val" or mode == "test":
            self.train = False
        else:
            raise ValueError("Argument mode should be one among train, val, and test.")

        # organize video and audio npz files
        self.src_vid_npz_dir = src_vid_npz_dir
        self.src_aud_npz_dir = src_aud_npz_dir
        self.vid_list = os.listdir(self.src_vid_npz_dir)
        self.aud_list = os.listdir(self.src_aud_npz_dir)
        self.vid_list.sort()
        self.aud_list.sort()
        self.length = len(self.vid_list)  # this is also same to len(self.aud_list)
        self.nseg = nseg

        # image augmentation on train set
        if self.train:
            self.vid_transforms = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(224),
                    transforms.ColorJitter(brightness=0.1, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        # no augmentation is applied on validation or test set
        else:
            self.vid_transforms = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        custom_spec_mean = kwargs.custom_spec_mean if hasattr(kwargs, "custom_spec_mean") else -24.2
        custom_spec_std = kwargs.custom_spec_std if hasattr(kwargs, "custom_spec_std") else 26.68

        # normalize a spectrogram into zero mean, unit variance
        # [Note] ToTensor() in this case does not scale the spectrogram because its np.dtype != np.uint8
        if not custom_spec_mean or not custom_spec_std:  # spectrogram mean and std is already given
            mean, std = get_spectrogram_mean_std(src_aud_npz_dir, nseg, mode)  # compute mean and std for given set
            self.aud_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[mean], std=[std])]
            )
        else:
            self.aud_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0], std=[custom_spec_std])]
            )
        with open(csv) as f:
            segments = f.readlines()
            segments = [s[:-1].split(",") for s in segments]
            segments = [s[:1] + [s[3:]] for s in segments]
            self.label_dict = dict(segments)

    def get_label_set(self, vid_id):
        return set(self.label_dict[vid_id])

    """
    def check_shape(self, vid_tensor, aud_tensor):
        if tuple(vid_tensor.shape) != self.expected_vid_shape:
            raise ValueError(
                "Invalid video frame shape, expected {} but {} given.".format(self.expected_vid_shape, vid_tensor.shape)
            )
        if tuple(aud_tensor.shape) != self.expected_aud_shape:
            raise ValueError(
                "Invalid spectrogram shape, expected {} but {} given.".format(self.expected_aud_shape, aud_tensor.shape)
            )
    """

    def __len__(self):
        # [Note] If index is in [0, self.length), make positive sample.
        #        If index is in [self.length, self.length * 2), make negative sample of video for each [0, self.length) index.
        return self.length * 2

    def __getitem__(self, idx):
        # positive sample
        if idx < self.length:
            vid_path = os.path.join(self.src_vid_npz_dir, self.vid_list[idx])
            aud_path = os.path.join(self.src_aud_npz_dir, self.aud_list[idx])

            time_idx = str(random.randrange(0, self.nseg))
            vid_arr = np.load(vid_path)[time_idx]
            aud_arr = np.load(aud_path)[time_idx]

            vid_tensor = self.vid_transforms(vid_arr)
            aud_tensor = self.aud_transforms(aud_arr)
            assert vid_tensor.shape == (3, 224, 224)
            assert aud_tensor.shape == (1, 257, 199)

            return vid_tensor, aud_tensor, torch.tensor(0)

        # negative sample
        elif idx >= self.length:
            vid_idx = idx - self.length

            # get random audio spectrogram
            while True:
                aud_idx = random.randrange(0, self.length)
                # vid_id = self.vid_list[vid_idx][:-4]
                # aud_id = self.aud_list[aud_idx][:-4]
                # if not (self.get_label_set(vid_id) & self.get_label_set(aud_id)):
                if vid_idx != aud_idx:
                    break
            vid_path = os.path.join(self.src_vid_npz_dir, self.vid_list[vid_idx])
            aud_path = os.path.join(self.src_aud_npz_dir, self.aud_list[aud_idx])

            vid_time_idx = str(random.randrange(0, self.nseg))
            aud_time_idx = str(random.randrange(0, self.nseg))
            vid_arr = np.load(vid_path)[vid_time_idx]
            aud_arr = np.load(aud_path)[aud_time_idx]

            vid_tensor = self.vid_transforms(vid_arr)
            aud_tensor = self.aud_transforms(aud_arr)
            assert vid_tensor.shape == (3, 224, 224)
            assert aud_tensor.shape == (1, 257, 199)

            return vid_tensor, aud_tensor, torch.tensor(1)

        else:
            raise IndexError("Index {} out of range.".format(idx))


# get mean and standard deviation of all spectrograms
def get_spectrogram_mean_std(src_aud_npz_dir, nseg, mode):
    aud_list = os.listdir(src_aud_npz_dir)
    spec_sum = np.zeros((257, 199))

    for i, aud_file in enumerate(aud_list):
        specs = np.load(os.path.join(src_aud_npz_dir, aud_file))
        for j in range(nseg):
            spec_sum += specs[str(j)]
        print("Getting mean and standard deviation of {} set: {} / {}\r".format(mode, i + 1, len(aud_list)), end="")
    print()
    spec_sum /= len(aud_list) * nseg

    mean = np.mean(spec_sum)
    std = np.std(spec_sum)

    return mean, std


# batch generation test
if __name__ == "__main__":
    # get train set
    train = AudioSet("train", "./data/train/video", "./data/train/audio")
    train_loader = DataLoader(train, batch_size=1, shuffle=True)

    # generate 10 batches
    for i, (img, aud, label) in enumerate(train_loader):
        print("Batch #{}:".format(i + 1), img.shape, aud.shape, label.shape)
        if i == 9:
            break
