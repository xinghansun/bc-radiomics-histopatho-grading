import torch
import re

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk

from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm


class MriDataset(Dataset):
    def __init__(self, sample_list):
        """
        sample_list: list of dict with keys:
            - 'phase_paths': list of file paths
            - 'label': int
        """
        self.samples = sample_list

    def extract_time(self, sitk_img):
        if sitk_img.HasMetaDataKey("AcquisitionTime"):
            t = sitk_img.GetMetaData("AcquisitionTime")
            h, m, s = int(t[:2]), int(t[2:4]), float(t[4:])
            return h * 3600 + m * 60 + s
        else:
            # shouldn't happen in our data. 
            # otherwise infer using surrounding phase times (TODO)
            return 0 
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        phase_volumes = []
        phase_times = []

        for path in sorted(sample["phase_paths"]):
            # I think we don't need to normalize as we did it in preprocessing
            img_sitk = sitk.ReadImage(str(path))
            arr = sitk.GetArrayFromImage(img_sitk)
            phase_volumes.append(torch.tensor(arr, dtype=torch.float32))
            phase_times.append(self.extract_time(img_sitk))

        T = len(phase_volumes)
        volume_tensor = torch.stack(phase_volumes)  # [T, D, H, W]
        time_tensor = torch.tensor(phase_times, dtype=torch.float32)
        mask = torch.ones(T, dtype=torch.bool)

        return volume_tensor, time_tensor, mask, torch.tensor(sample["label"], dtype=torch.long)

def collate_fn(batch):
    """
    batch: list of (volume_tensor [T, D, H, W], times [T], mask [T], label)
    Returns:
        imgs: [B, max_T, D, H, W]
        times: [B, max_T]
        masks: [B, max_T]
        labels: [B]
    """
    max_T = max(x[0].shape[0] for x in batch)
    B = len(batch)
    D, H, W = batch[0][0].shape[1:]

    imgs = torch.zeros(B, max_T, D, H, W)
    times = torch.zeros(B, max_T)
    masks = torch.zeros(B, max_T, dtype=torch.bool)
    labels = []

    for i, (vol, _, _, label) in enumerate(batch):
        T = vol.shape[0]
        imgs[i, :T] = vol
        times[i, :T] = torch.arange(T).float()
        masks[i, :T] = True
        labels.append(label)

    return imgs, times, masks, torch.stack(labels)

class Time2Vec(nn.Module):
    """ Convert timestamps to vector for training """
    def __init__(self, out_dim):
        super(Time2Vec, self).__init__()
        self.w0 = nn.Linear(1, 1)
        self.wi = nn.Linear(1, out_dim - 1)

    def forward(self, t): # t: [B, T]
        t = t.unsqueeze(-1)  # [B, T, 1]
        v0 = self.w0(t) # [B, T, 1]
        v1 = torch.sin(self.wi(t)) # [B, T, D-1]
        return torch.cat([v0, v1], dim=-1) # [B, T, D]

class Simple3DCNN(nn.Module):
    def __init__(self, out_dim=256):
        super(Simple3DCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)  # [B, 32, 1, 1, 1]
        )
        self.fc =  nn.Sequential(
            nn.Linear(32, out_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):  # x: [B*T, 1, D, H, W]
        feat = self.conv(x).flatten(1)
        return self.fc(feat)  # [B*T, out_dim]

class TumorGradePredictor(nn.Module):
    def __init__(self, cnn_out_dim=256, time_emb_dim=32, nhead=4, num_classes=3):
        super().__init__()
        self.encoder = Simple3DCNN(out_dim=cnn_out_dim)
        self.time2vec = Time2Vec(out_dim=time_emb_dim)
        self.embed_dim = cnn_out_dim + time_emb_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=nhead,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, imgs, times, mask):
        """
        imgs: [B, T, D, H, W]
        times: [B, T]
        mask: [B, T] (bool)
        """
        B, T, D, H, W = imgs.shape
        x = imgs.reshape(B * T, 1, D, H, W)
        x_feat = self.encoder(x).view(B, T, -1) # [B, T, cnn_out_dim]

        t_feat = self.time2vec(times) # [B, T, time_emb_dim]
        combined = torch.cat([x_feat, t_feat], dim=-1) # [B, T, embed_dim]

        cls_tok = self.cls_token.repeat(B, 1, 1) # [B, 1, embed_dim]
        x_seq = torch.cat([cls_tok, combined], dim=1) # [B, T+1, embed_dim]

        if mask is not None:
            extended_mask = torch.cat([torch.ones(B, 1, device=mask.device), mask], dim=1)  # [B, T+1]
            src_key_padding_mask = ~extended_mask
        else:
            src_key_padding_mask = None

        trans_out = self.transformer(x_seq, src_key_padding_mask=src_key_padding_mask)[:, 0]  # [B, embed_dim]
        return self.fc(self.dropout(trans_out))  # [B, num_classes]

def train_model(model, train_loader, val_loader=None, 
                num_epochs=20, lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for imgs, times, masks, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs = imgs.to(device) # [B, T, D, H, W]
            times = times.to(device) # [B, T]
            masks = masks.to(device) # [B, T]
            labels = labels.to(device) # [B]

            logits = model(imgs, times, masks) # [B, num_classes]
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples
        print(f"Train Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

        if val_loader:
            validate(model, val_loader, device)

def validate(model, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, times, masks, labels in loader:
            imgs = imgs.to(device)
            times = times.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            logits = model(imgs, times, masks)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)

    acc = total_correct / total_samples
    print(f"Val Accuracy: {acc:.4f}")

model = TumorGradePredictor()
tumor_dir = "./tumour_extracted"
sample_list = [
    {
        "phase_paths": sorted([os.path.join(tumor_dir, f) for f in os.listdir(tumor_dir) if re.match("BC_100_.+gz", f)]),
        "label": 0
    }
]
dataset = MriDataset(sample_list)
print(len(dataset))

train_set, val_set = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
print(f"Train: {len(train_set)}; Val: {len(val_set)}")

train_loader = DataLoader(train_set, batch_size=4, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_set, batch_size=4, collate_fn=collate_fn)