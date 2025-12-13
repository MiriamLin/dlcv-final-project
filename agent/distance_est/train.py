import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from data_loader import DistanceDataset
from model import ResNetDistanceRegressor
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--data_dir", type=Path, required=True)
parser.add_argument("--json_file", type=Path, required=True)
args = parser.parse_args()

Path('ckpt').mkdir(exist_ok=True)

dataset = DistanceDataset(args.data_dir, args.json_file, rgb=True, depth=True)
loader = DataLoader(dataset, batch_size=18, shuffle=True, num_workers=4)

model = ResNetDistanceRegressor(input_channels=6, backbone='resnet50', pretrained=True).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
criterion = nn.MSELoss()

print(f"Dataset size: {len(dataset)}")
print(f"Number of batches: {len(loader)}")

# Training loop
num_epochs = 5
for epoch in tqdm(range(num_epochs), position=0, leave=True):
    tqdm.write(f"Starting epoch {epoch + 1}/{num_epochs}")
    model.train()
    epoch_loss = 0
    num_batches = len(loader)
    log_interval = 50

    for i, (inputs, targets) in tqdm(enumerate(loader), total=num_batches, position=1, leave=False):
        inputs, targets = inputs.cuda(), targets.cuda()
        preds = model(inputs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss

        if (i + 1) % log_interval == 0 or (i + 1) == num_batches:
            tqdm.write(f"[Epoch {epoch+1}/{num_epochs}] "
                  f"Iter {i+1}/{num_batches} "
                  f"Loss: {batch_loss:.4f}")

    avg_loss = epoch_loss / num_batches
    print(f"[Epoch {epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), f"ckpt/epoch_{epoch+1}.pth")
    print(f"Model saved at epoch {epoch+1}")