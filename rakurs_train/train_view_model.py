# train_view_model.py
import os, random
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import numpy as np
from tqdm import tqdm

IMAGE_SIZE = 320   # training size (you can change)
BATCH = 8
EPOCHS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANCHOR_DIR = "dataset_view/anchor"
POS_DIR = "dataset_view/positive"
NEG_DIR = "dataset_view/negative"

class TripletViewDataset(Dataset):
    def __init__(self, anchors, pos_list, neg_list, transform):
        self.anchors = anchors
        self.pos = pos_list
        self.neg = neg_list
        self.transform = transform

    def __len__(self):
        return max(len(self.pos), len(self.neg))

    def __getitem__(self, idx):
        anchor_path = random.choice(self.anchors)
        pos_path = self.pos[idx % len(self.pos)]
        neg_path = random.choice(self.neg)
        a = Image.open(anchor_path).convert("RGB")
        p = Image.open(pos_path).convert("RGB")
        n = Image.open(neg_path).convert("RGB")
        return self.transform(a), self.transform(p), self.transform(n)

def build_model():
    model = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    # remove classifier head, get feature vector
    model.classifier = nn.Identity()
    return model

def compute_embeddings(model, loader, device):
    model.eval()
    embs = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            out = model(x)
            out = nn.functional.normalize(out, p=2, dim=1)
            embs.append(out.cpu())
    if embs:
        return torch.cat(embs, dim=0)
    return torch.empty(0)

def main():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    anchors = glob(os.path.join(ANCHOR_DIR, "*"))
    poses = glob(os.path.join(POS_DIR, "*"))
    negs = glob(os.path.join(NEG_DIR, "*"))
    if not (anchors and poses and negs):
        print("Dataset incomplete. Make sure anchor/pos/neg exist."); return

    dataset = TripletViewDataset(anchors, poses, negs, transform)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)

    model = build_model().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    best_val_score = 1e9

    # small val loader by sampling some pos/neg and anchors
    val_anchor = [random.choice(anchors) for _ in range(50)]
    val_pos = [random.choice(poses) for _ in range(50)]
    val_neg = [random.choice(negs) for _ in range(50)]
    val_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for a,p,n in pbar:
            a = a.to(DEVICE); p = p.to(DEVICE); n = n.to(DEVICE)
            a_emb = model(a)
            p_emb = model(p)
            n_emb = model(n)
            a_emb = nn.functional.normalize(a_emb, p=2, dim=1)
            p_emb = nn.functional.normalize(p_emb, p=2, dim=1)
            n_emb = nn.functional.normalize(n_emb, p=2, dim=1)
            loss = criterion(a_emb, p_emb, n_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.item()))

        # validation: compute distance distributions
        model.eval()
        with torch.no_grad():
            # anchor embedding (average of anchors)
            anchor_tensors = torch.stack([val_transform(Image.open(x).convert("RGB")) for x in val_anchor]).to(DEVICE)
            anchor_embs = model(anchor_tensors); anchor_embs = nn.functional.normalize(anchor_embs, p=2, dim=1)
            anchor_mean = anchor_embs.mean(dim=0, keepdim=True)  # 1 x D

            pos_tensors = torch.stack([val_transform(Image.open(x).convert("RGB")) for x in val_pos]).to(DEVICE)
            neg_tensors = torch.stack([val_transform(Image.open(x).convert("RGB")) for x in val_neg]).to(DEVICE)
            pos_embs = model(pos_tensors); pos_embs = nn.functional.normalize(pos_embs, p=2, dim=1)
            neg_embs = model(neg_tensors); neg_embs = nn.functional.normalize(neg_embs, p=2, dim=1)

            pos_dists = torch.norm(pos_embs - anchor_mean, dim=1).cpu().numpy()
            neg_dists = torch.norm(neg_embs - anchor_mean, dim=1).cpu().numpy()

            pos_mean = pos_dists.mean(); neg_mean = neg_dists.mean()
            print(f"Val: pos_mean={pos_mean:.4f} neg_mean={neg_mean:.4f}")

            # choose threshold between pos and neg means (simple)
            thr = float((pos_mean + neg_mean) / 2.0)
            val_score = (pos_mean, neg_mean, thr)

            # save model if separation improved (neg_mean - pos_mean larger)
            sep = neg_mean - pos_mean
            if sep > best_val_score:
                pass
            # here we just always save last model for simplicity
            print("Saving model script...")
            example = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to("cpu")
            model_cpu = model.to("cpu").eval()
            traced = torch.jit.trace(model_cpu, example)
            traced.save("view_model.pt")
            # also save threshold
            with open("view_threshold.txt", "w") as f:
                f.write(str(thr))
        # move model back to device for next epoch
        model.to(DEVICE)
    print("Training finished. model saved: view_model.pt, threshold in view_threshold.txt")

if __name__ == "__main__":
    main()
