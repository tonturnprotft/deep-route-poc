import os
import sys
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt

# ── 1) Make sure we can import your model ───────────────────────────────────
script_dir = os.path.dirname(__file__)
project_root = script_dir
sys.path.insert(0, project_root)
from src.models.train_route_lstm import RouteLSTM

# ── 2) Configuration ─────────────────────────────────────────────────────────
H5_PATH        = os.path.join(project_root, "data", "processed", "sequences_remap.h5")
TRAIN_IDX      = os.path.join(project_root, "data", "splits", "train_idx.npy")
VAL_IDX        = os.path.join(project_root, "data", "splits", "val_idx.npy")
CHECKPOINT_DIR = os.path.join(project_root, "checkpoints")
SUBSAMPLE_VAL  = 5_000_000         # only eval on first 5M samples
BATCH_SIZE     = 1024
WIN            = 8
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 3) Picklable Dataset (opens HDF5 per worker) ────────────────────────────
class H5EvalDataset(Dataset):
    def __init__(self, h5_path, idx_list, win):
        self.h5_path = h5_path
        self.ids      = idx_list
        self.win      = win
        self.h5       = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.h5 is None:
            print(f"[Dataset] Opening HDF5 file: {self.h5_path}")
            self.h5 = h5py.File(self.h5_path, "r")
            self.X  = self.h5["X"]
            self.y  = self.h5["y"]
            self.U  = self.h5["U"]

        idx   = self.ids[i]
        row   = self.X[idx]
        cells = torch.from_numpy(row[:self.win].astype(np.int64))
        wx    = np.nan_to_num(row[self.win:].astype(np.float32))
        wx    = torch.from_numpy(wx).unsqueeze(0).expand(self.win, -1)
        label = int(self.y[idx])
        user  = int(self.U[idx])
        return cells, wx, label, user

# ── 4) Evaluation function ───────────────────────────────────────────────────
def eval_checkpoint(ckpt_path, zero_wx=False):
    print(f"[Eval] Starting evaluation: checkpoint={ckpt_path}, zero_wx={zero_wx}")

    # a) infer wx_dim
    with h5py.File(H5_PATH, "r") as f:
        total_dim = f["X"].shape[1]
    wx_dim = total_dim - WIN
    print(f"[Eval] Inferred wx_dim={wx_dim}")

    # b) load model
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = RouteLSTM(
        n_cells=ckpt["model"]["embed.weight"].shape[0],
        wx_dim=wx_dim
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[Eval] Loaded model state dict, ready for inference")

    # c) prepare loader
    val_idx = np.load(VAL_IDX)[:SUBSAMPLE_VAL]
    print(f"[Eval] Subsampled validation to {len(val_idx)} samples")
    ds      = H5EvalDataset(H5_PATH, val_idx, WIN)
    loader  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # d) run inference
    total, c1, c5 = 0, 0, 0
    per_user      = defaultdict(lambda: [0,0,0])
    num_batches   = len(loader)
    print(f"[Eval] Beginning inference loop ({num_batches} batches)")
    with torch.no_grad():
        for batch_idx, (cells, wx, labels, users) in enumerate(loader, 1):
            cells = cells.to(DEVICE)
            wx    = wx.to(DEVICE)
            if zero_wx:
                wx = torch.zeros_like(wx)

            logits = model(cells, wx)
            top5   = logits.topk(5, dim=1).indices.cpu().numpy()
            y_np   = labels.numpy()

            total += y_np.size
            c1    += (top5[:,0] == y_np).sum()
            c5    += (top5 == y_np.reshape(-1,1)).sum()

            for u, true, pred5 in zip(users.numpy(), y_np, top5):
                cnts = per_user[u]
                cnts[0] += 1
                cnts[1] += int(pred5[0] == true)
                cnts[2] += int(true in pred5)

            if batch_idx % 100 == 0:
                print(f"[Eval] Batch {batch_idx}/{num_batches} processed")

    top1 = c1/total if total>0 else 0.0
    top5 = c5/total if total>0 else 0.0
    print(f"[Eval] Completed inference: total={total}, Top-1={top1:.4f}, Top-5={top5:.4f}\n")
    return {"total": total, "top1": top1, "top5": top5, "per_user": per_user}

# ── 5) Main: pick last checkpoint & run eval ─────────────────────────────────
def main():
    print("[Main] Locating checkpoints...")
    pt_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
    last_ckpt = sorted(
        pt_files,
        key=lambda f: int(f.split("epoch")[1].split(".")[0])
    )[-1]
    ckpt_path = os.path.join(CHECKPOINT_DIR, last_ckpt)
    print(f"[Main] Using latest checkpoint: {last_ckpt}\n")

    # full-model eval
    print("[Main] >>> Full-model evaluation")
    full = eval_checkpoint(ckpt_path, zero_wx=False)

    # ablation eval
    print("[Main] >>> Ablation (no-weather) evaluation")
    abla = eval_checkpoint(ckpt_path, zero_wx=True)

    # derive cold users
    train_idx = np.load(TRAIN_IDX)
    val_idx   = np.load(VAL_IDX)[:SUBSAMPLE_VAL]
    with h5py.File(H5_PATH, "r") as f:
        all_users = f["U"][:]
    train_u_set = set(all_users[train_idx])
    val_u_set   = set(all_users[val_idx])
    cold_users  = val_u_set - train_u_set
    print(f"[Main] Computed cold_users size: {len(cold_users)}\n")

    # warm vs cold breakdown
    per_user = full["per_user"]
    warm_tot = warm_c1 = warm_c5 = 0
    cold_tot = cold_c1 = cold_c5 = 0
    for u, (cnt, c1u, c5u) in per_user.items():
        if u in cold_users:
            cold_tot += cnt; cold_c1 += c1u; cold_c5 += c5u
        else:
            warm_tot += cnt; warm_c1 += c1u; warm_c5 += c5u

    print(f"[Main] Overall → Top-1: {full['top1']:.4f}, Top-5: {full['top5']:.4f}")
    print(f"[Main] Ablation → Top-1: {abla['top1']:.4f}, Top-5: {abla['top5']:.4f}")

    if warm_tot > 0:
        print(f"[Main] Warm users → Top-1: {warm_c1/warm_tot:.4f}, Top-5: {warm_c5/warm_tot:.4f}")
    else:
        print(f"[Main] Warm users → no data (warm_tot={warm_tot})")
    if cold_tot > 0:
        print(f"[Main] Cold users → Top-1: {cold_c1/cold_tot:.4f}, Top-5: {cold_c5/cold_tot:.4f}\n")
    else:
        print(f"[Main] Cold users → no data (cold_tot={cold_tot})\n")

    # f) plots
    print("[Main] Plotting results...")
    if warm_tot > 0 and cold_tot > 0:
        plt.figure()
        plt.bar(["Warm-Top1", "Cold-Top1"], [warm_c1/warm_tot, cold_c1/cold_tot])
        plt.ylabel("Accuracy"); plt.title("Top-1: Warm vs Cold Users")
        plt.show()
    else:
        print("[Main] Skipping Warm vs Cold plot due to insufficient data.")

    plt.figure()
    plt.bar(["Full", "No-Weather"], [full["top1"], abla["top1"]])
    plt.ylabel("Top-1 Accuracy"); plt.title("Ablation: Weather Feature")
    plt.show()

if __name__ == "__main__":
    main()