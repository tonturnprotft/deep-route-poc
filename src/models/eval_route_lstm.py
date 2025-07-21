import argparse, math, logging, h5py, numpy as np, torch
from train_route_lstm import H5SequenceDataset, RouteLSTM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="data/processed/porto_sequences_win10.h5")
    p.add_argument("--idx", required=True, help=".npy file with row indices to score")
    p.add_argument("--ckpt", required=True, help="model checkpoint (.pt)")
    p.add_argument("--win", type=int, default=10)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--device", default="mps", choices=["cpu","cuda","mps"])
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 avoids h5py pickling issue)")
    args = p.parse_args()

    idx = np.load(args.idx)
    ds  = H5SequenceDataset(args.h5, idx, args.win)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=(args.device!="mps"))
    n_batches = len(loader)
    logger.info(f"Evaluating {len(ds):,d} sequences in {n_batches} batches (batch_size={args.batch})")

    device = torch.device('mps' if args.device=='mps' and torch.backends.mps.is_available() else args.device)

    ckpt = torch.load(args.ckpt, map_location=device)

    # Infer architecture parameters from checkpoint so shapes always match
    n_cells     = ckpt['model']['fc.weight'].shape[0]
    hidden_dim  = ckpt['model']['fc.weight'].shape[1]               # e.g. 256
    emb_dim     = ckpt['model']['embed.weight'].shape[1]            # e.g. 128

    model = RouteLSTM(
        n_cells=n_cells,
        wx_dim=ds[0][1].shape[-1],
        emb_dim=emb_dim,
        hidden=hidden_dim,
    )
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    correct1 = correct5 = total = 0
    with torch.no_grad():
        for i, (X_cells, wx, y) in enumerate(loader, 1):
            X_cells, wx, y = X_cells.to(device), wx.to(device), y.to(device)
            logits = model(X_cells, wx)
            topk   = logits.topk(5, dim=1).indices
            correct1 += (topk[:,0] == y).sum().item()
            correct5 += (topk == y[:,None]).sum().item()
            total    += y.size(0)
            if i % 100 == 0 or i == n_batches:
                logger.info(
                    "Batch %d/%d (%.1f%%) – running Top‑1: %.4f  Top‑5: %.4f",
                    i, n_batches, 100 * i / n_batches,
                    correct1 / total, correct5 / total
                )
    logger.info("Hold‑out Top‑1: %.4f  Top‑5: %.4f", correct1/total, correct5/total)

if __name__ == "__main__":
    main()