import torch
import numpy as np


class ImgSampleDataset(torch.utils.data.Dataset):
    """Dataset class for NeRF with image point sampling.

    Args:
        img (torch.tensor/numpy.array): array of images.
        num_sample (int): sample size for one datapoint.
    """

    def __init__(self, img, num_sample=1024):
        self.img = torch.tensor(img, dtype=torch.float32)
        self.num_sample = num_sample

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        pixel_ids = \
            torch.tensor(
                np.random.choice(
                    np.arange(len(img)),
                    size=self.num_sample,
                    replace=False
                ), dtype=torch.long)

        img = img[pixel_ids]
        img_ids = torch.tensor([idx]*self.num_sample, dtype=torch.long)
        return img, pixel_ids, img_ids


def collate_fn_sample(batch):
    imgs = torch.cat([d[0] for d in batch], dim=0)
    pixel_ids = torch.cat([d[1] for d in batch], dim=0)
    imgs_ids = torch.cat([d[2] for d in batch], dim=0)
    return imgs, pixel_ids, imgs_ids


class PosedDataset(torch.utils.data.Dataset):
    """Dataset class for NeRF with pre-computed rays.

    Args:
        imgs (torch.tensor/numpy.array): array of images.
        os (torch.tensor/numpy.array): translation of camera.
        ds (torch.tensor/numpy.array): ray direction of camera.
    """

    def __init__(self, imgs, os_, ds):
        self.imgs = torch.tensor(imgs, dtype=torch.float32)
        self.os = torch.tensor(os_, dtype=torch.float32)
        self.ds = torch.tensor(ds, dtype=torch.float32)

    def __len__(self):
        return len(self.os)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        o = self.os[idx]
        d = self.ds[idx]
        return img, o, d


def collate_fn_posed(batch):
    imgs = torch.stack([d[0] for d in batch], dim=0)
    os_ = torch.stack([d[1] for d in batch], dim=0)
    ds = torch.stack([d[2] for d in batch], dim=0)
    return imgs, os_, ds
