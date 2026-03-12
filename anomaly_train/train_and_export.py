import os
from glob import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision

IMAGE_SIZE = 320
BATCH = 8
EPOCHS = 20
DEVICE = "cpu"


class TripletDataset(Dataset):
    def __init__(self, anchor_dir, pos_dir, neg_dir):
        self.anchors = glob(anchor_dir + "/*")
        self.pos = glob(pos_dir + "/*")
        self.neg = glob(neg_dir + "/*")

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        anchor_path = random.choice(self.anchors)
        pos_path = self.pos[idx]
        neg_path = random.choice(self.neg)

        anchor = Image.open(anchor_path).convert("RGB")
        pos = Image.open(pos_path).convert("RGB")
        neg = Image.open(neg_path).convert("RGB")

        return (
            self.transform(anchor),
            self.transform(pos),
            self.transform(neg),
        )


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.efficientnet_b0(weights='DEFAULT')
        self.encoder.classifier = nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x


def main():
    dataset = TripletDataset("dataset/anchor", "dataset/positive", "dataset/negative")
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)

    model = SiameseNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.TripletMarginLoss(margin=1.0)

    for epoch in range(EPOCHS):
        for anchor, pos, neg in loader:
            anchor = anchor.to(DEVICE)
            pos = pos.to(DEVICE)
            neg = neg.to(DEVICE)

            a = model(anchor)
            p = model(pos)
            n = model(neg)

            loss = criterion(a, p, n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

    # TorchScript экспорт (CPU)
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    traced = torch.jit.trace(model, dummy)
    traced.save("siamese_model_cpu.pt")
    print("TorchScript модель сохранена: siamese_model_cpu.pt")


if __name__ == "__main__":
    main()

