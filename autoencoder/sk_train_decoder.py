# Running this file directly puts its own folder on sys.path, not the repository
# root, so the shared package would not resolve. Prepend the root before any
# first-party import. Harmless when the module is imported as part of a package.
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = str(_Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
import os
import glob
import yaml
import h5py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from shared.sk_utils import Utils as u
from autoencoder.sk_decoder import RetinaDecoder

class RGCDataset(Dataset):
    """
    Dataset that pairs RGC activations with their corresponding video frames.

    HDF5 file handles are opened lazily per worker process (keyed by PID) so
    the dataset can be safely used with num_workers > 0 in DataLoader.
    """
    def __init__(self, activations_path, training_params_path):

        train_cfg = u.read_params(training_params_path)
        self.input_window = train_cfg.get("input_window_size", 1)

        if os.path.isdir(activations_path):
            file_paths = glob.glob(os.path.join(activations_path, "*.h5"))
        else:
            file_paths = [activations_path]

        if not file_paths:
            raise ValueError(f"No .h5 files found for path: {activations_path}")

        # Store paths only — handles are opened lazily in __getitem__ per worker.
        self.file_paths = file_paths
        self.index_map  = []
        # pid -> [h5py.File, ...]
        self._file_handles: dict = {}

        for file_idx, fp in enumerate(file_paths):
            with h5py.File(fp, 'r') as f:
                if "trials" not in f:
                    print(f"Skipping {fp}: not in trials format.")
                    continue

                for local_idx in f["trials"].keys():
                    trial_grp = f["trials"][local_idx]
                    fr_shape  = trial_grp["firing_rates"]["0"].shape
                    vf_shape  = trial_grp["target_video"].shape
                    
                    # Assuming transposed format (T_windows, n_cells)
                    n_windows = fr_shape[0]
                    retina_win_size = vf_shape[0] - n_windows + 1
                    for t in range(self.input_window - 1, n_windows):
                        self.index_map.append((file_idx, local_idx, t, retina_win_size))

        print(f"Dataset compiled: {len(self.index_map)} spatiotemporal samples from {len(file_paths)} files.")
        if len(self.index_map) == 0:
            raise ValueError("No valid trials found in the provided activations.")

        # Save dimensions for model initialisation
        with h5py.File(file_paths[0], 'r') as f0:
            first_trial = list(f0["trials"].keys())[0]
            fr_grp = f0["trials"][first_trial]["firing_rates"]
            self.n_cells_per_mosaic = [fr_grp[str(i)].shape[1] for i in range(len(fr_grp))]
            self.frame_shape = f0["trials"][first_trial]["target_video"].shape[1:]

    def _get_handles(self):
        """Return this process's open HDF5 handles, opening them if necessary."""
        import h5py
        pid = os.getpid()
        if pid not in self._file_handles:
            self._file_handles[pid] = [h5py.File(fp, 'r') for fp in self.file_paths]
        return self._file_handles[pid]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, local_idx, t, retina_win_size = self.index_map[idx]
        h5_file = self._get_handles()[file_idx]

        fr_grp = h5_file[f"trials/{local_idx}/firing_rates"]
        vf_ds  = h5_file[f"trials/{local_idx}/target_video"]

        x_list = []
        for m_idx in range(len(fr_grp)):
            # Indexing (T_windows, n_cells) -> slice time, then transpose back to (n_cells, Win)
            x_np = fr_grp[str(m_idx)][t - (self.input_window - 1) : t + 1, :].T
            x_list.append(torch.tensor(x_np, dtype=torch.float32))

        y = torch.tensor(vf_ds[t + retina_win_size - 1], dtype=torch.float32).unsqueeze(0)
        return x_list, y

    def __del__(self):
        for handles in getattr(self, '_file_handles', {}).values():
            for f in handles:
                try:
                    f.close()
                except Exception:
                    pass

def train(args):
    # 1. Load Configurations
    model_params = u.read_params(args.params)
    train_cfg = u.read_params(args.training_params)

    # CLI overrides
    device = args.device if args.device else train_cfg.get("device", "cpu")
    epochs = args.epochs if args.epochs else train_cfg.get("epochs", 50)
    batch_size = args.batch_size if args.batch_size else train_cfg.get("batch_size", 16)
    lr = args.lr if args.lr else train_cfg.get("learning_rate", 1e-4)

    print(f"Training on device: {device}")

    # 2. Data Preparation
    print(f"Loading data from {args.activations}...")
    dataset = RGCDataset(args.activations, args.training_params)

    dataset_prop = args.dataset_prop if args.dataset_prop is not None else train_cfg.get("dataset_prop", 1.0)
    num_workers  = args.num_workers  if args.num_workers  is not None else train_cfg.get("num_workers",  4)

    # Validation split
    val_split  = train_cfg.get("validation_split", 0.1)
    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Draw one fixed random subset per split and keep it for the entire run.
    # DataLoader shuffle=True still randomises *order* within the subset each epoch.
    def _fixed_subset(base_ds, n_total, prop):
        n = max(1, int(n_total * prop))
        idx = torch.randperm(n_total)[:n].tolist()
        return torch.utils.data.Subset(base_ds, idx)

    train_subset = _fixed_subset(train_ds, train_size, dataset_prop)
    val_subset = _fixed_subset(val_ds, val_size, dataset_prop)

    if dataset_prop < 1.0:
        print(f"Using {dataset_prop:.1%} of data: "
              f"{len(train_subset):,} train / {len(val_subset):,} val samples (fixed for this run).")

    # Optional Preloading into RAM
    preload = args.preload if args.preload is not None else train_cfg.get("preload", False)
    if preload:
        print("Preloading data into RAM...")
        def _preload_subset(subset):
            # Using a simple loop; since num_workers=0 here it's slow but happens once.
            x_samples = [] # list of lists of tensors
            y_samples = []
            for i in tqdm(range(len(subset)), desc="Preloading"):
                x, y = subset[i]

                if x[2].shape[0] != 1008:
                    continue
                x_samples.append(x)
                y_samples.append(y)
            
            # x_samples: N samples x M mosaics x (Cells, Win)
            # We want M mosaics x N samples x Cells x Win
            num_mosaics = len(x_samples[0])
            x_preloaded = []
            for m in range(num_mosaics):
                x_preloaded.append(torch.stack([s[m] for s in x_samples]))
            
            y_preloaded = torch.stack(y_samples)
            return x_preloaded, y_preloaded

        train_x, train_y = _preload_subset(train_subset)
        val_x, val_y = _preload_subset(val_subset)

        # Custom TensorDataset-like structure for the list of mosaic tensors
        class PreloadedDataset(Dataset):
            def __init__(self, x_list, y):
                self.x_list = x_list
                self.y = y
            def __len__(self): return len(self.y)
            def __getitem__(self, i):
                return [x[i] for x in self.x_list], self.y[i]

        train_ds_final = PreloadedDataset(train_x, train_y)
        val_ds_final = PreloadedDataset(val_x, val_y)
        
        # Once preloaded, we don't need many workers or complex contexts
        loader_kwargs = {"batch_size": batch_size, "num_workers": 0, "pin_memory": True}
        train_loader = DataLoader(train_ds_final, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds_final, shuffle=False, **loader_kwargs)
    else:
        # Standard Lazy Loading
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": (num_workers > 0),
            "multiprocessing_context": "spawn" if num_workers > 0 else None
        }
        train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)

    # 3. Model Initialization
    decoder_params = {
        "n_cells_per_mosaic": [n * dataset.input_window for n in dataset.n_cells_per_mosaic],
        "frame_shape": dataset.frame_shape,
        "num_blocks": train_cfg.get("num_blocks", 4),
        "num_kernels": train_cfg.get("num_kernels", 64),
        "bias": train_cfg.get("bias", False)
    }
    
    model = RetinaDecoder(decoder_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    print(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized. Total trainable parameters: {num_params:,}\n")

    # 4. Training Loop
    best_epoch = 0
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    
    epoch_pbar = tqdm(range(epochs), desc="Training Progress", dynamic_ncols=False, leave=True)
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        
        # iterate through the batches
        for x_list, y in train_loader:
            x_list = [x.to(device) for x in x_list]
            y = y.to(device)
            
            optimizer.zero_grad()
            output = model(x_list)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_list, y in val_loader:
                x_list = [x.to(device) for x in x_list]
                y = y.to(device)
                output = model(x_list)
                val_loss += criterion(output, y).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        train_loss_history.append(avg_train)
        val_loss_history.append(avg_val)
        
        epoch_pbar.set_description(f"Epoch {epoch+1}/{epochs}")
        epoch_pbar.set_postfix({"Train MSE": f"{avg_train:.6f}", "Val MSE": f"{avg_val:.6f}", "Best Val MSE": f"{best_val_loss:.6f} @ Epoch {best_epoch}"})
        
        if avg_val < best_val_loss:
            best_epoch = epoch + 1
            best_val_loss = avg_val
            save_dict = {
                'model_state_dict': model.state_dict(),
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history
            }
            torch.save(save_dict, args.output)

    print("\nTraining Complete.")
    
    # Save the final model and full history
    last_output = args.output.replace(".pt", "_final.pt") if ".pt" in args.output else args.output + "_final.pt"
    save_dict = {
        'model_state_dict': model.state_dict(),
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history
    }
    torch.save(save_dict, last_output)
    print(f"Final model and training history saved to {last_output}\n")

def main():
    parser = argparse.ArgumentParser(description="Train the Decoder model on RGC activations.")
    parser.add_argument("--activations", type=str, required=True, help="Path to the .h5 file or directory containing RGC activations.")
    parser.add_argument("--params", type=str, default=f"{_REPO_ROOT}/shared/params.yaml", help="Path to the retina model params.yaml.")
    parser.add_argument("--training-params", type=str, default="training_params.yaml", help="Path to training_params.yaml.")
    parser.add_argument("--output", type=str, default="best_decoder.pt", help="Path to save the best model weights.")
    parser.add_argument("--device",      type=str,   help="Device to use (cpu, cuda). Overrides training_params.yaml.")
    parser.add_argument("--epochs",      type=int,   help="Number of epochs. Overrides training_params.yaml.")
    parser.add_argument("--batch-size",  type=int,   help="Batch size. Overrides training_params.yaml.")
    parser.add_argument("--lr",          type=float, help="Learning rate. Overrides training_params.yaml.")
    parser.add_argument("--dataset-prop", type=float, default=None,
                        help="Fraction of data to use (0, 1]. A fixed random subset is drawn once at startup. "
                             "Overrides training_params.yaml.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader worker processes for parallel HDF5 reading. Overrides training_params.yaml.")
    parser.add_argument("--preload", action="store_true", default=None,
                        help="Preload the entire subset into RAM before training. Much faster but requires enough RAM.")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
