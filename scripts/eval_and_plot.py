#!/usr/bin/env python
"""
Evaluate a Route-LSTM checkpoint on several *.npy index files and
produce a simple bar-chart of Top-1 / Top-5 accuracy.

Example
-------
python scripts/eval_and_plot.py \
    --h5 data/processed/porto_sequences_win10.h5 \
    --ckpt checkpoints/route_epoch7.pt \
    --idx-files data/splits/porto_win10/test_idx.npy \
               data/splits/porto_win10/warm_test_idx.npy \
               data/splits/porto_win10/cold_idx.npy \
    --out plot_eval.png \
    --device mps
"""
import argparse, json, logging, subprocess, sys, tempfile, textwrap
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR   = Path(__file__).resolve().parents[1] / "src" / "models"
EVAL_SCRIPT  = SCRIPT_DIR / "eval_route_lstm.py"

def run_eval(h5, ckpt, idx_file, win, batch, device):
    """Invoke eval_route_lstm.py as a subprocess and parse the final line."""
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--h5", h5,
        "--idx", idx_file,
        "--ckpt", ckpt,
        "--win", str(win),
        "--batch", str(batch),
        "--device", device,
        "--workers", "0",  # avoid h5py pickling issues
    ]
    logger.info("▶ %s", " ".join(cmd))

    # Stream logs to the console *and* keep a copy so we can parse at the end.
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    out_lines = []
    for line in proc.stdout:
        print(line, end="")          # real‑time output
        out_lines.append(line)

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"eval_route_lstm.py failed with exit code {proc.returncode}")

    out = "".join(out_lines)

    # Match “Top‑1:” / “Top-1:” with *any* single separator character between “Top” and the digit.
    pattern = re.compile(r"Top.\s*1:\s*([0-9.]+)\s*Top.\s*5:\s*([0-9.]+)")
    for line in reversed(out_lines):
        m = pattern.search(line)
        if m:
            return tuple(map(float, m.groups()))

    raise RuntimeError(f"Could not parse Top-1/Top-5 from eval output:\n{out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--idx-files", nargs="+", required=True,
                   help="One or more *.npy index files")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Display names; default = file stem")
    p.add_argument("--win", type=int, default=10)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--device", default="mps", choices=["cpu","cuda","mps"])
    p.add_argument("--out", default="eval_bar.png",
                   help="PNG file to save the plot")
    args = p.parse_args()

    labels = args.labels or [Path(f).stem for f in args.idx_files]
    if len(labels) != len(args.idx_files):
        p.error("labels and idx-files must be same length")

    top1s, top5s = [], []
    for label, idxf in zip(labels, args.idx_files):
        t1, t5 = run_eval(args.h5, args.ckpt, idxf,
                          args.win, args.batch, args.device)
        logger.info("✔ %s  Top-1 %.4f  Top-5 %.4f", label, t1, t5)
        top1s.append(t1)
        top5s.append(t5)

    # ---------- plotting ----------
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, top1s, width, label="Top-1")
    ax.bar(x + width/2, top5s, width, label="Top-5")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f"Route-LSTM ({Path(args.ckpt).name})")

    plt.tight_layout()
    fig.savefig(args.out, dpi=150)
    logger.info("📈  Plot saved → %s", args.out)

if __name__ == "__main__":
    main()