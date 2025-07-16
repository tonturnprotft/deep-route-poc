# src/data/test_generator.py
import sys
from pathlib import Path
# add project src folder to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from data.split_sequences import main as _  # just to show structure
from models.train_route_lstm import SequenceGenerator  # or wherever you put it

# parameters for sequence generator
WIN = 8         # sliding-window length used when building sequences
WX_DIM = 3      # number of weather feature dimensions in each sequence (total_features - WIN)

if __name__ == "__main__":
    h5  = "data/processed/sequences.h5"
    tix = np.load("data/splits/train_idx.npy")[: 5*1024]
    gen = SequenceGenerator(h5_path=h5, idx_list=tix, batch_size=1024, win=WIN, wx_dim=WX_DIM)
    for i, (X, y) in enumerate(gen):
        print(f"batch {i:2d} → X.shape={X.shape}, y.shape={y.shape}")
        if i >= 4: break