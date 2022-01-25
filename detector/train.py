import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch

import config as cfg
from dataset import CityCamDataset

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    pretrained=True)

num_classes = cfg.num_classes

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

train_dataset = CityCamDataset(cfg.train_data_dir, transforms=None)
test_dataset = CityCamDataset(cfg.test_data_dir, transforms=None)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    num_workers=2,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=cfg.batch_size,
    num_workers=2,
)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, cfg.lr, cfg.momentum, cfg.weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=2,
    factor=0.05,
    verbose=True
)  # Reduces learning sate when a metric has stopped improving.

criterion =  # TODO add loss function for object detection
