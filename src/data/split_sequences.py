#!/usr/bin/env python3
"""
Step 2: Split sequences.h5 → train_idx.npy, val_idx.npy
 - 80/20 split by index
 - 20% ของ unique users เป็น “cold-user” test set
"""
import argparse
import os

import h5py
import numpy as np


def main(args):
    # เปิดไฟล์ HDF5
    with h5py.File(args.h5, "r") as f:
        U = f["U"][:]   # load all user_id (int32 array)
        N = U.shape[0]

    # กำหนด seed เพื่อ reproducibility
    np.random.seed(args.seed)

    # แยก cold-users (20% ของ unique user)
    users = np.unique(U)
    n_cold = int(len(users) * args.cold_ratio)
    cold_users = np.random.choice(users, size=n_cold, replace=False)
    is_cold = np.isin(U, cold_users)

    # สร้าง indices ทั้งหมด
    idx = np.arange(N)

    # 80/20 train/val split by index with user-holdout
    cutoff = int(N * args.train_ratio)
    train_mask = (~is_cold) & (idx < cutoff)
    val_mask   = (~is_cold) & (idx >= cutoff) | is_cold

    train_idx = idx[train_mask]
    val_idx   = idx[val_mask]

    # บันทึกไฟล์
    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train_idx.npy")
    val_path   = os.path.join(args.out_dir, "val_idx.npy")
    np.save(train_path, train_idx)
    np.save(val_path,   val_idx)

    print(f"Total seqs: {N:,}")
    print(f" Train    : {train_idx.size:,} sequences → {train_path}")
    print(f" Val (incl. cold-user): {val_idx.size:,} sequences → {val_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5",         required=True,
                   help="path to data/processed/sequences.h5")
    p.add_argument("--out-dir",    default="data/splits",
                   help="where to save train_idx.npy & val_idx.npy")
    p.add_argument("--train-ratio",type=float, default=0.8,
                   help="fraction of data-by-index for training")
    p.add_argument("--cold-ratio", type=float, default=0.2,
                   help="fraction of users held out as cold-user test")
    p.add_argument("--seed",       type=int, default=42,
                   help="random seed for user sampling")
    args = p.parse_args()
    main(args)