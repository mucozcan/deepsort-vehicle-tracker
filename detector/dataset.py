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

        for obj in root.findall('vehicle'):
            class_type = obj.find('type').text
            class_type = self.get_class_type(class_type)
            if class_type is None:
                continue
            x_min = int(obj.find('bndbox/xmin').text) / x_scale
            y_min = int(obj.find('bndbox/ymin').text) / y_scale
            x_max = int(obj.find('bndbox/xmax').text) / x_scale
            y_max = int(obj.find('bndbox/ymax').text) / y_scale

            if x_min == x_max or y_min == y_max:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_type)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        image = torchvision.transforms.ToTensor()(image)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = image_id

        if self.transforms is not None:
            image, target = self.transforms(image, target)

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
        else:
            return None


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    import cv2
    import numpy as np

    dataset = CityCamDataset(cfg.train_data_dir, transforms=None)

    for i in range(10):
        image, targets = dataset[i]
        print(image)
        image = image.permute(1, 2, 0).numpy() * 255
        image = np.ascontiguousarray(image, dtype=np.uint8)
        print(image)
        for box, label in zip(targets['boxes'], targets['labels']):
            label = cfg.class_dict[label.item()]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 1
            )
            cv2.putText(
                image, label, (int(box[0]), int(box[1]-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            cv2.imshow('Image', image)
            cv2.waitKey(0)
