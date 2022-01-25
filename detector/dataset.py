import os
import torch
import torchvision
from PIL import Image
from natsort import natsorted
import glob
import xml.etree.ElementTree as ET

import config as cfg


class CityCamDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        self.imgs = natsorted(glob.glob(self.root + "/*.jpg"))

    def __getitem__(self, idx):

        img_path = self.imgs[idx]
        ann_path = os.path.splitext(img_path)[0] + ".xml"
        
        image = Image.open(img_path).convert("RGB")

        img_w, img_h = image.size

        x_scale = img_w / cfg.train_input_size[0]
        y_scale = img_h / cfg.train_input_size[1]

        image = image.resize(cfg.train_input_size)

        tree = ET.parse(ann_path)
        root = tree.getroot()

        labels = []
        boxes = []

        for obj in root.iter('vehicle'):
            class_type = obj.find('type').text
            class_type = self.get_class_type(class_type)

            x_min = int(obj.find('bndbox/xmin').text) / x_scale
            y_min = int(obj.find('bndbox/ymin').text) / y_scale
            x_max = int(obj.find('bndbox/xmax').text) / x_scale
            y_max = int(obj.find('bndbox/ymax').text) / y_scale

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append([class_type])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        image = torchvision.transforms.ToTensor()(image)

        target = {}

        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
    
        if self.transforms is not None:
            img, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.imgs)

    def get_class_type(self, class_type):  # merging some of classes
        if class_type in ["1", "2", "3"]:
            return 1  # car
        elif class_type in ["4", "5", "6"]:
            return 2  # truck
        elif class_type == "7":
            return 3  # van
        elif class_type in ["8", "9"]:
            return 4  # bus


if __name__ == "__main__":
    train_dir = "/home/muco/git-repos/vehicle-tracker/detector/data/train/"

    dataset = CityCamDataset(train_dir, transforms=None)
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1)
    batch = next(iter(loader))
    print(batch)
