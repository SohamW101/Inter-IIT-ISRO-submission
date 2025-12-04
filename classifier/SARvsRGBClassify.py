import os
import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from tqdm import tqdm

# Speed optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

scaler = torch.cuda.amp.GradScaler()


class RGBvsSAR_Dataset(Dataset):
    def __init__(self, rgb_dir, sar_dir, size=64):
        self.rgb = glob.glob(os.path.join(rgb_dir, "*.png"))
        self.sar = glob.glob(os.path.join(sar_dir, "*.png"))
        self.data = [(p, 0) for p in self.rgb] + [(p, 1) for p in self.sar]

        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        try:
            img = Image.open(path)

            if img.mode != "RGB":
                img = img.convert("RGB")

            img = self.transform(img)

        except Exception as e:
            print(f"[WARN] Skipping corrupted/broken image: {path} ({e})")

            img = torch.zeros(3, 64, 64)

        return img, torch.tensor([label], dtype=torch.float32)

class RGBvsSAR_CNN(nn.Module):
    def __init__(self):
        super(RGBvsSAR_CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))   # output: 256×4×4

        x = x.view(x.size(0), -1)  # flatten 4096
        x = F.relu(self.fc1(x))
        x = self.fc2(x)            # logits
        return x



def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            total_loss += loss.item()

    model.train()
    return total_loss / len(loader)


def train_model(rgb_dir, sar_dir, epochs=10, batch_size=4, lr=1e-3, size=64):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RGBvsSAR_Dataset(rgb_dir, sar_dir, size=size)

    # train/val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # fast dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, prefetch_factor=2
    )

    model = RGBvsSAR_CNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-6)

    best_loss = float("inf")
    best_path = "best_model.pth"

    for epoch in range(epochs):
        print(f"\n Epoch {epoch+1}/{epochs}")

        # training loop with progress bar
        progress = tqdm(train_loader, desc="Training", leave=False)
        for imgs, labels in progress:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress.set_postfix({"loss": loss.item()})

        # validation
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

        # save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"Saved Best Model → {best_path} (loss={best_loss:.4f})")

    return best_path


def load_best_model(path="best_model_red_param.pth"):
    model = RGBvsSAR_CNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def predict_image(model, image_path, size=64):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor()
    ])

    x = transform(img).unsqueeze(0)
    x = x.to(next(model.parameters()).device)

    with torch.no_grad():
        logits = model(x)           # raw output
        prob = torch.sigmoid(logits)

    return "SAR" if prob.item() > 0.5 else "RGB"



if __name__ == "__main__":
    rgb_dir = "/home/raid/ameya/classifier/dataset/rgb"
    sar_dir = "/home/raid/ameya/classifier/dataset/sar"

    best_model_path = train_model(
        rgb_dir,
        sar_dir,
        epochs=40,
        batch_size=128,
        lr=1e-3
    )

    model = load_best_model(best_model_path)
    print(predict_image(model, "/home/raid/ameya/deoldify/libcong.jpg"))
