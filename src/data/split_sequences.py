#!/usr/bin/env python3
"""
Step 2: Split sequences.h5 → train_idx.npy, val_idx.npy, warm_test_idx.npy, cold_idx.npy
 - 80/10/10 split by index for warm users (train/val/test)
 - 20% ของ unique users เป็น “cold-user” test set (all cold-user rows)
 - test_idx.npy is the concatenation of warm_test_idx and cold_idx for backwards compatibility
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

    val_ratio = 1 - args.train_ratio - args.test_ratio
    assert 0 <= val_ratio <= 1, "train_ratio + test_ratio must be <= 1 and >= 0"

    warm_idx = idx[~is_cold]          # all non‑cold rows, sorted
    n_warm = warm_idx.size
    n_train = int(n_warm * args.train_ratio)
    n_val   = int(n_warm * val_ratio)
    train_idx = warm_idx[:n_train]
    val_idx   = warm_idx[n_train:n_train+n_val]
    warm_test_idx = warm_idx[n_train+n_val:]
    cold_idx = idx[is_cold]
    test_idx = np.concatenate([warm_test_idx, cold_idx])

    # บันทึกไฟล์
    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train_idx.npy")
    val_path   = os.path.join(args.out_dir, "val_idx.npy")
    warm_test_path = os.path.join(args.out_dir, "warm_test_idx.npy")
    cold_path = os.path.join(args.out_dir, "cold_idx.npy")
    test_path = os.path.join(args.out_dir, "test_idx.npy")

    np.save(train_path, train_idx)
    np.save(val_path, val_idx)
    np.save(warm_test_path, warm_test_idx)
    np.save(cold_path, cold_idx)
    np.save(test_path, test_idx)

    print(f"Total sequences : {N:,}")
    print(f" Warm  train    : {train_idx.size:,}")
    print(f" Warm  val      : {val_idx.size:,}")
    print(f" Warm  test     : {warm_test_idx.size:,}")
    print(f" Cold‑user test : {cold_idx.size:,}")
    print(f" ==> test_idx.npy (warm+cold) : {test_idx.size:,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5",         required=True,
                   help="path to data/processed/sequences.h5")
    p.add_argument("--out-dir",    default="data/splits",
                   help="where to save train_idx.npy & val_idx.npy")
    p.add_argument("--train-ratio",type=float, default=0.8,
                   help="fraction of warm data-by-index for training")
    p.add_argument("--test-ratio", type=float, default=0.1,
                   help="fraction of warm data-by-index for testing")
    p.add_argument("--cold-ratio", type=float, default=0.2,
                   help="fraction of users held out as cold-user test")
    p.add_argument("--seed",       type=int, default=42,
                   help="random seed for user sampling")
    args = p.parse_args()
    main(args)