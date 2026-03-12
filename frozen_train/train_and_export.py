# train_and_export.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
import os
from tqdm import tqdm

DATA_DIR = Path("dataset/train")
VAL_DIR = Path("dataset/val")
IMG_SIZE = (320, 240)
BATCH = 32
EPOCHS = 12
LR = 1e-3
MODEL_OUT = "freeze_detector_scripted.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PairsDataset(Dataset):
    def __init__(self, root_dir, img_size=(320, 240), transforms=None):
        self.root = Path(root_dir)
        self.pairs = []
        for cls in ("frozen", "normal"):
            p = self.root / cls
            for f in sorted(p.glob("*_a.png")):
                b = f.with_name(f.name.replace("_a", "_b"))
                if b.exists():
                    self.pairs.append((str(f), str(b), 1 if cls == "frozen" else 0))
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a_path, b_path, label = self.pairs[idx]

        A = Image.open(a_path).convert("RGB").resize(self.img_size)
        B = Image.open(b_path).convert("RGB").resize(self.img_size)

        import torchvision.transforms.functional as F
        A_t = F.to_tensor(A)  # C,H,W
        B_t = F.to_tensor(B)

        pair = torch.cat([A_t, B_t], dim=0)  # → 6×H×W
        return pair, torch.tensor([label], dtype=torch.float32)

class FreezeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # allows arbitrary HxW -> 1x1
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def make_loader(root, img_size, batch):
    ds = PairsDataset(root, img_size=img_size)
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4, drop_last=True)

def train():
    model = FreezeNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()

    train_loader = make_loader(DATA_DIR, IMG_SIZE, BATCH)
    val_loader = make_loader(VAL_DIR, IMG_SIZE, BATCH)

    best_val = 1e9
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total = 0
        for xb,yb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb).view(-1)
            loss = crit(logits, yb.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            total += xb.size(0)
        print(f"Train loss: {total_loss/total:.6f}")

        model.eval()
        vloss = 0.0
        vcount = 0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb).view(-1)
                loss = crit(logits, yb.view(-1))
                vloss += float(loss.item()) * xb.size(0)
                vcount += xb.size(0)
        vavg = vloss / max(1, vcount)
        print(f"Val loss: {vavg:.6f}")
        if vavg < best_val:
            best_val = vavg
            print("Saving best model.")
            example = torch.randn(1, 6, IMG_SIZE[1], IMG_SIZE[0]).to("cpu")
            model_cpu = model.to("cpu")
            model_cpu.eval()
            traced = torch.jit.trace(model_cpu, example)
            traced.save(MODEL_OUT)

if __name__ == "__main__":
    train()

