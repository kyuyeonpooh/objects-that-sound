import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import matplotlib.pyplot as plt


class AudioSet(Dataset):
    def __init__(
        self,
        out_vid_dir,
        out_aud_dir,
        vid_shape=(3, 224, 224),
        aud_shape=(1, 257, 199),
        vid_transforms=None,
        aud_transforms=None,
        max_extract=9,
        val=False,
        debug=False,
    ):
        # organize video and audio npz files
        self.src_vid_dir = out_vid_dir
        self.src_aud_dir = out_aud_dir
        self.vid_list = os.listdir(self.src_vid_dir)
        self.aud_list = os.listdir(self.src_aud_dir)
        self.vid_list.sort()
        self.aud_list.sort()
        self.length = len(self.vid_list)  # this must be equal to len(self.aud_list)

        # expected output shape
        self.vid_shape = vid_shape
        self.aud_shape = aud_shape

        # set transforms
        if vid_transforms is None:
            self.vid_transforms = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomCrop(224, pad_if_needed=True),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.vid_transforms = vid_transforms
        if aud_transforms is None:
            """
            normalize a spectrogram by scaling it,
            from 10 * log(spectrogram) to log(spectrogram) (e.g. [-50, 50] to [-5, 5])
            ToTensor() in this case does not scale the spectrogram because spectrogram np.dtype != np.uint8
            """
            self.aud_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.0], std=[80.0])]
            )
        else:
            self.aud_transforms = aud_transforms

        self.max_extract = max_extract
        self.val = val
        self.debug = debug

    def check_shape(self, vid_tensor, aud_tensor):
        if tuple(vid_tensor.shape) != self.vid_shape:
            raise ValueError(
                "Invalid video frame shape, expected {} but {} given.".format(self.vid_shape, vid_arr.shape)
            )
        if tuple(aud_tensor.shape) != self.aud_shape:
            raise ValueError(
                "Invalid spectrogram shape, expected {} but {} given.".format(self.aud_shape, aud_arr.shape)
            )

    def __len__(self):
        """
        [0, self.length): positive sample
        [self.length, self.length * 2): negative sample of video from [0, self.length) index
        """
        return self.length * 2 * 9

    def __getitem__(self, idx):
        if self.debug:
            vid_tensor = torch.rand(self.vid_shape)
            aud_tensor = torch.rand(self.aud_shape)
            label = torch.randint(0, 2, size=(1,))
            return vid_tensor, aud_tensor, label

        time_idx = str(idx % 9)
        idx = idx // 9

        # positive sample
        if idx < self.length:
            vid_path = os.path.join(self.src_vid_dir, self.vid_list[idx])
            aud_path = os.path.join(self.src_aud_dir, self.aud_list[idx])

            # time_idx = str(random.randrange(0, self.max_extract))
            vid_arr = np.load(vid_path)[time_idx]
            aud_arr = np.load(aud_path)[time_idx]

            vid_tensor = self.vid_transforms(vid_arr)
            aud_tensor = self.aud_transforms(aud_arr)
            self.check_shape(vid_tensor, aud_tensor)

            return vid_tensor, aud_tensor, torch.tensor(1)

        # negative sample
        elif idx >= self.length:
            vid_idx = idx - self.length
            aud_idx = vid_idx
            # get another audio spectrogram randomly
            while aud_idx != vid_idx:
                aud_idx = random.randrange(0, len(self.length))
            vid_path = os.path.join(self.src_vid_dir, self.vid_list[vid_idx])
            aud_path = os.path.join(self.src_aud_dir, self.aud_list[aud_idx])

            # vid_time_idx = str(random.randrange(0, self.max_extract))
            vid_time_idx = time_idx
            aud_time_idx = str(random.randrange(0, self.max_extract))
            vid_arr = np.load(vid_path)[vid_time_idx]
            aud_arr = np.load(aud_path)[aud_time_idx]

            vid_tensor = self.vid_transforms(vid_arr)
            aud_tensor = self.aud_transforms(aud_arr)
            self.check_shape(vid_tensor, aud_tensor)

            return vid_tensor, aud_tensor, torch.tensor(0)

        else:
            raise IndexError("Index out of range, idx: {}".format(idx))


# batch generation test
if __name__ == "__main__":
    audioset = AudioSet("./data/video", "./data/audio")
    dataloader = DataLoader(audioset, batch_size=1, shuffle=True)
    for i, (img, aud, label) in enumerate(dataloader):
        image = aud[0].squeeze()
        fig = plt.gcf()
        plt.imshow(image)
        fig.savefig("img.pdf")
        break
