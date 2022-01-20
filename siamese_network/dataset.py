import torchvision
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import torch
import PIL.ImageOps
import glob


def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("figure.png")


class SiameseTripletDataset(Dataset):

    def __init__(self, folder_dataset, transform=None, should_invert=True):
        self.folder_dataset = folder_dataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        # Get a random image which will be used as an anchor
        anchor_image = random.choice(self.folder_dataset.imgs)
        # anchor_image = (img_path, class_id)

        while True:
            # keep looping till a different class image is found. Negative image.
            neg_image = random.choice(self.folder_dataset.imgs)
            if anchor_image[1] != neg_image[1]:
                break

        # Getting anchor image file and class name
        anchor_class_name = anchor_image[0].split('/')[-2]

        # Getting all the images which belong to the same class as anchor image.
        class_files = glob.glob(
            self.folder_dataset.root+anchor_class_name+'/*')

        class_files = [x for x in class_files if x != anchor_image[0]]

        if len(class_files) == 0:
            # If there is no image (other than anchor image) belonging to the anchor image class, anchor
            # image will be taken as positive sample
            pos_image = anchor_image[0]
        else:
            # Choose random image (of same class as anchor image) as positive sample
            pos_image = random.choice(class_files)

        assert anchor_class_name == pos_image.split('/')[-2], \
            "Anchor image and positive image are not belong to same class"

        anchor = Image.open(anchor_image[0])
        negative = Image.open(neg_image[0])
        positive = Image.open(pos_image)

        anchor = anchor.convert("RGB")
        negative = negative.convert("RGB")
        positive = positive.convert("RGB")

        if self.should_invert:
            anchor = PIL.ImageOps.invert(anchor)
            positive = PIL.ImageOps.invert(positive)
            negative = PIL.ImageOps.invert(negative)

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return len(self.folder_dataset.imgs)


if __name__ == '__main__':
    training_dir = "./data/"
    folder_dataset = torchvision.datasets.ImageFolder(root=training_dir)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
        torchvision.transforms.ToTensor()
    ])

    siamese_dataset = SiameseTripletDataset(
        folder_dataset=folder_dataset,
        transform=transforms,
        should_invert=False)

    loader = DataLoader(
        siamese_dataset, shuffle=True, batch_size=1)
    batch = next(iter(loader))
    concatenated = torch.cat(
        (batch[0], batch[1], batch[2]), 0)
    imshow(torchvision.utils.make_grid(concatenated))
