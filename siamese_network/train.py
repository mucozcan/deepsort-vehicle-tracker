import torchvision
from torch.utils.data import DataLoader
import torch
import PIL
from tqdm import tqdm

from utils import get_gaussian_mask
from dataset import SiameseTripletDataset
from models import SiameseNetwork, TripletLoss
import config as cfg

folder_dataset = torchvision.datasets.ImageFolder(root=cfg.train_dir)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((cfg.input_size)),
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    torchvision.transforms.ToTensor()
])

siamese_dataset = SiameseTripletDataset(folder_dataset=folder_dataset,
                                 transform=transforms, should_invert=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

model = SiameseNetwork().to(device) 

criterion = TripletLoss(margin=1)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience = 2,
        factor = 0.05,
        verbose = True
        )# Reduces learning rate when a metric has stopped improving.

counter = []
loss_history = []
iteration_number = 0

train_loader = DataLoader(siamese_dataset, shuffle=True, num_workers=4,
                              batch_size=cfg.batch_size)

# Multiply each image with mask to give attention to center of the image.
gaussian_mask = get_gaussian_mask(height=cfg.input_size[0], width=cfg.input_size[1]).to(device)

mean_losses = []

for epoch in range(cfg.epochs):
    
    running_loss = []
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, data in loop:
      
        # Get anchor, positive and negative samples
        anchor, positive, negative = data
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # Multiple image patches with gaussian mask. It will act as an attention mechanism which will focus on the center of the patch
        anchor, positive, negative = anchor * gaussian_mask, positive * \
            gaussian_mask, negative * gaussian_mask

        optimizer.zero_grad()  # Reset the optimizer gradients to zero

        anchor_out, positive_out, negative_out = model(
            anchor, positive, negative)  

        # Compute triplet loss (based on cosine simality) on the output feature maps
        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()  
        optimizer.step()  
        
        running_loss.append(loss.item())
        mean_loss = sum(running_loss) / len(running_loss)

        loop.set_description(f"Epoch: [{epoch + 1}/{cfg.epochs}]")
        loop.set_postfix(batch_loss = loss.item(), mean_loss = mean_loss, 
                                lr = optimizer.param_groups[0]["lr"])
    
    if len(mean_losses) > 1:
        if mean_loss < min(mean_losses):
            torch.save(model.state_dict(), cfg.model_save_path)

    mean_losses.append(mean_loss)
    
    scheduler.step(mean_loss)

