# classifier_inference.py
import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


def load_model(weights_path="classifier/classifier_rgb_vs_sar.pth", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RGBvsSAR_CNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def classify_image(model, image_path, size=64):
    device = next(model.parameters()).device
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor()
    ])

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    SARres = "Greyscale"
    RGBres = "RGB"

    return SARres if prob > 0.5 else RGBres
