import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class TeethDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.split == "train":
            img_path = os.path.join(self.root, "img", self.imgs[idx])
            mask_path = os.path.join(self.root, "masks", self.masks[idx])
            
            #image, label = data['image'], data['label']
        else:
            img_path = os.path.join(self.root, "val\img", self.imgs[idx])
            mask_path = os.path.join(self.root, "val\masks", self.masks[idx])
            
        image = np.array( Image.open(img_path))
        label = np.array( Image.open(mask_path))
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
 
        return sample