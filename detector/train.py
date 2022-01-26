import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from tqdm import tqdm
import math
import sys

import config as cfg
from dataset import CityCamDataset, collate_fn

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    pretrained=True)

num_classes = cfg.num_classes

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features, num_classes)  # TODO implement def build_model(num_classes)

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

train_dataset = CityCamDataset(cfg.train_data_dir, transforms=None)
test_dataset = CityCamDataset(cfg.test_data_dir, transforms=None)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    # shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)

model.to(device)
model.train()

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, cfg.lr, cfg.momentum, cfg.weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=2,
    factor=0.05,
    verbose=True
)  # Reduces learning sate when a metric has stopped improving.

mean_losses = []

for epoch in range(cfg.epochs):

    running_loss = []
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (images, targets) in loop:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        mean_loss = sum(running_loss) / len(running_loss)
        loop.set_description(f"Epoch: [{epoch + 1}/{cfg.epochs}]")
        loop.set_postfix(box_loss=loss_dict['loss_rpn_box_reg'].item(),
                         class_loss=loss_dict['loss_classifier'].item(),
                         batch_loss=loss.item(), mean_loss=mean_loss,
                         lr=optimizer.param_groups[0]["lr"])
    if len(mean_losses) > 1:
        if mean_loss < min(mean_losses):
            torch.save(model.state_dict(), cfg.model_save_path)

    mean_losses.append(mean_loss)

    scheduler.step(mean_loss)
    # TODO add evaluation
