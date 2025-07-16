#!/usr/bin/env python3
"""
Step 4: Training & validation loop for RouteLSTM on h5 sequences
"""
import argparse
import os
import math
import logging

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- logging setup ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class H5SequenceDataset(Dataset):
    def __init__(self, h5_path, idx_list, win):
        self.h5_path = h5_path
        self.ids = idx_list
        self.win = win
        self.dX = None
        self.dy = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        # lazy open
        if self.dX is None or self.dy is None:
            f = h5py.File(self.h5_path, 'r')
            self.dX = f['X']
            self.dy = f['y']
        idx = self.ids[i]
        row = self.dX[idx]
        # split history cells vs weather features
        X_cells = torch.from_numpy(row[:self.win].astype(np.int64))
        raw_wx = row[self.win:].astype(np.float32)
        raw_wx = np.nan_to_num(raw_wx, nan=0.0, posinf=0.0, neginf=0.0)
        # tile weather to match history length
        wx = torch.from_numpy(raw_wx).unsqueeze(0).expand(self.win, -1)
        y = int(self.dy[idx])
        return X_cells, wx, y


class RouteLSTM(nn.Module):
    def __init__(self, n_cells, wx_dim, emb_dim=64, hidden=128, layers=1):
        super().__init__()
        self.embed = nn.Embedding(n_cells, emb_dim)
        self.lstm = nn.LSTM(emb_dim + wx_dim, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, n_cells)

    def forward(self, X_cells, X_wx):
        # X_cells: (B, WIN), X_wx: (B, WIN, wx_dim)
        emb = self.embed(X_cells)             # (B, WIN, emb_dim)
        inp = torch.cat([emb, X_wx], dim=2)   # (B, WIN, emb_dim+wx_dim)
        out, _ = self.lstm(inp)
        logits = self.fc(out[:, -1])          # predict next cell
        return logits

def train(args):
    # prepare checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    start_epoch = 1

    # load train & val indices (sub-sample validation)
    train_idx = np.load(args.train_idx)
    if args.subsample_train > 0:
        train_idx = train_idx[: args.subsample_train]
    val_idx = np.load(args.val_idx)
    val_count = int(len(train_idx) * 0.2)
    val_idx = val_idx[:val_count]

    # detect weather feature dimension
    with h5py.File(args.h5, 'r') as hf:
        total_dim = hf['X'].shape[1]
    args.wx_dim = total_dim - args.win

    # datasets & loaders
    train_ds = H5SequenceDataset(args.h5, train_idx, args.win)
    val_ds   = H5SequenceDataset(args.h5, val_idx,   args.win)

    device = torch.device('mps' if args.device=='mps' and torch.backends.mps.is_available() else args.device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # model, optimizer, loss
    model = RouteLSTM(args.n_cells, args.wx_dim, emb_dim=args.emb_dim,
                      hidden=args.hidden, layers=args.layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    total_train_batches = math.ceil(len(train_idx)/args.batch_size)
    total_val_batches   = math.ceil(len(val_idx)/args.batch_size)

    for epoch in range(start_epoch, args.epochs+1):
        # 1) Training pass
        model.train()
        train_loss = 0.0
        total_samples = 0
        log_interval = 100
        for batch_i, (X_cells, wx, y_true) in enumerate(train_loader, 1):
            X_cells = X_cells.to(device)
            wx       = wx.to(device)
            y_true   = y_true.to(device)

            optimizer.zero_grad()
            logits = model(X_cells, wx)
            loss = criterion(logits, y_true)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y_true.size(0)
            total_samples += y_true.size(0)

            if batch_i % log_interval == 0:
                remaining = total_train_batches - batch_i
                logger.info(f"Epoch {epoch} Batch {batch_i}/{total_train_batches}: loss {loss.item():.4f} ({remaining} batches left)")

        avg_loss = train_loss / total_samples
        logger.info(f"Epoch {epoch} — train loss: {avg_loss:.4f} — processed {total_train_batches} batches")

        # 2) Validation pass
        model.eval()
        correct1 = correct5 = total = 0
        logger.info(f"Epoch {epoch} — starting validation")
        with torch.no_grad():
            for batch_i, (X_cells, wx, y_true) in enumerate(val_loader, 1):
                X_cells = X_cells.to(device)
                wx       = wx.to(device)
                y_true   = y_true.to(device)

                logits = model(X_cells, wx)
                topk = logits.topk(5, dim=1).indices.cpu().numpy()
                y_np = y_true.cpu().numpy()
                correct1 += (topk[:,0] == y_np).sum()
                correct5 += (topk == y_np.reshape(-1,1)).sum()
                total += y_np.shape[0]
                if batch_i % log_interval == 0:
                    rem_val = total_val_batches - batch_i
                    logger.info(f"Epoch {epoch} Val Batch {batch_i}/{total_val_batches}: Top-1 so far {correct1}, Top-5 so far {correct5} ({rem_val} batches left)")

        logger.info(f"Epoch {epoch} — val Top-1: {correct1/total:.4f}, Top-5: {correct5/total:.4f}")

        # 3) Save checkpoint
        ckpt_path = f"checkpoints/route_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5",       default="data/processed/sequences.h5")
    p.add_argument("--train-idx", required=True)
    p.add_argument("--val-idx",   required=True)
    p.add_argument("--subsample-train", type=int, default=5000000,
                   help="Number of training sequences to use (default 5M)")
    p.add_argument("--n-cells",   type=int, required=True)
    p.add_argument("--win",       type=int, default=8)
    p.add_argument("--batch-size",type=int, default=1024)
    p.add_argument("--epochs",    type=int, default=5)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--emb-dim",   type=int, default=64)
    p.add_argument("--hidden",    type=int, default=128)
    p.add_argument("--layers",    type=int, default=1)
    p.add_argument("--device",    default="cpu", choices=["cpu","cuda","mps"])
    args = p.parse_args()
    train(args)