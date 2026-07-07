import argparse
import h5py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sk_retina_autoencoder import RetinaAutoencoder
from sk_utils import Utils as u

class VideoWindowDataset(Dataset):
    
    def __init__(self, h5_path, videos, target_h, target_w, crop_coords=(0, 0), window_size=16, random_crops=False, seed=0):
        """torch.utils.data.Dataset that takes an h5 file of video frames
        and produces windows for training

        Args:
            h5_path (str): path to h5 file containing videos
            target_h (int): height of output training examples
            target_w (int): width of output training examples
            crop_coords (tuple, optional): for fixed training (x,y) positions. Defaults to (0, 0).
            window_size (int, optional): window size used in training to output. Defaults to 16.
            random_crops (bool, optional): whether random (x,y) positions should be used. Defaults to False.
                                           Overrides crop_coords.
            seed (int, optional): random seed to use for reproducibility. Defaults to 0.
        """

        self.h5_path = h5_path
        self.target_h = int(target_h)
        self.target_w = int(target_w)
        self.crop_coords = tuple(crop_coords)
        self.window_size = max(1, int(window_size))
        self.random_crops = bool(random_crops)
        np.random.seed(seed)
        
        # read the h5 file
        self.data = h5py.File(h5_path, "r")
        self.videos = videos
        self.data_len = int(np.sum([self.data[ds].attrs["frame_count"] for ds in videos]) / self.window_size)

    def __len__(self):
        """returns number of windows for training

        Returns:
            int: number of windows in the train_dataset
        """

        return self.data_len

    def __getitem__(self, idx):
        """produces next item/batch

        Args:
            idx (int): idx of window to return

        Returns:
            (torch.Tensor, torch.Tensor): the window sample, the window target
        """

        # get the right video to pull from and the start frame
        video = self.videos[idx % len(self.videos)]
        start_frame_idx = max(0, int(((idx / len(self.videos) * 0.9) - (idx // len(self.videos))) * self.data[video].attrs["frame_count"]))

        # get the right window and squeeze the first dim since this is stored [W, 1, H, W]
        window = self.data[video][start_frame_idx:start_frame_idx+self.window_size]

        # crop and return
        if self.random_crops:
            self.crop_coords = (np.random.randint(0, self.data[video].attrs["original_w"] - self.target_w), np.random.randint(0, self.data[video].attrs["original_h"] - self.target_h))
        crop = torch.Tensor(window[:, self.crop_coords[1]:self.crop_coords[1]+self.target_h, self.crop_coords[0]:self.crop_coords[0]+self.target_w])
        return crop, crop[-1].unsqueeze(0)

def train_autoencoder(data_path, cell_params, training_params_path, output_path, args):
    """train the RetinaAutoencoder end-to-end

    Args:
        data_path (str): path to the h5 file containing training examples
        cell_params (dict): parameters necessary for setting up retina cell type encoders
        training_params_path (str): path to training params file
        output_path (str): path to store model weights
        args (argparse.ArgumentParser): stores other arguments for training

    Raises:
        ValueError: raised if some cell types are not present
    """
    
    # set up the data for training
    print(f"Setting up the Dataset and Loader: {data_path}")
    train_cfg = u.read_params(training_params_path)
    video_params = dict(cell_params.get("video_parameters", {}))
    window_size = train_cfg.get("input_window_size", 16)
    target_h, target_w = train_cfg.get("target_height", 128), train_cfg.get("target_width", 128)

    with h5py.File(data_path, "r") as f:
        h5_file = h5py.File(data_path)
        videos = np.array(list(h5_file.keys()))
    train_videos_idx = np.random.choice([0,1], size=videos.shape, p=[0.1,0.9])
    train_videos = [v for v,i in zip(videos,train_videos_idx) if i == 1]
    val_videos = [v for v,i in zip(videos,train_videos_idx) if i == 0]

    train_dataset = VideoWindowDataset(h5_path=data_path, videos=train_videos, target_h=target_h, target_w=target_w, crop_coords=(args.x or 0, args.y or 0), window_size=window_size, random_crops=args.random_crops, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size or 1, num_workers=args.num_workers or 0, pin_memory=args.device.startswith("cuda"), persistent_workers=(args.num_workers or 0) > 0, shuffle=True)
    val_dataset = VideoWindowDataset(h5_path=data_path, videos=val_videos, target_h=target_h, target_w=target_w, crop_coords=(args.x or 0, args.y or 0), window_size=window_size, random_crops=args.random_crops, seed=args.seed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size or 1, num_workers=args.num_workers or 0, pin_memory=args.device.startswith("cuda"), persistent_workers=(args.num_workers or 0) > 0, shuffle=True)
    print(f"{len(train_dataset.videos):,} videos loaded for training with a total of {train_dataset.data_len:,} windows.")
    print(f"{len(val_dataset.videos)} videos loaded for validation with a total of {val_dataset.data_len:,} windows.")

    # inititalize the model using only the required cell types
    requested_cell_types = train_cfg.get("cell_types", None)
    all_cell_params = {k: v for k, v in cell_params.items() if k != "video_parameters"}
    if requested_cell_types is not None:
        missing = [ct for ct in requested_cell_types if ct not in all_cell_params]
        if missing:
            raise ValueError(f"cell_types listed in params_training.yaml not found in params.yaml: {missing}")
        active_cell_params = {ct: all_cell_params[ct] for ct in requested_cell_types}
    else:
        active_cell_params = all_cell_params

    print(f"Initializing model with frame shape {target_h:,}x{target_w:,} and {list(active_cell_params.keys())} active cell types")
    model = RetinaAutoencoder(active_cell_params, video_params,
        decoder_params={
            "frame_shape": (target_h, target_w),
            "num_blocks": train_cfg.get("num_blocks", 4),
            "num_kernels": train_cfg.get("num_kernels", 64),
            "bias": train_cfg.get("bias", False),
        }).to(args.device)
    print(f"Model transferred to {next(model.parameters()).device} with {sum(p.numel() for p in model.parameters()):,} total parameters.")

    # set up optimizer, loss, learning rate
    lr = args.lr or train_cfg.get("learning_rate", 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    print(f"using {optimizer.__class__.__name__} with lr={lr} and {type(criterion).__name__} loss")

    # start training!
    n_epochs = args.epochs or train_cfg.get("epochs", 1)
    print(f"Training for {n_epochs:,} epochs with batch size {args.batch_size or 1:,}")
    progress_bar = tqdm(np.arange(n_epochs), leave=True)
    train_loss_history = []
    val_loss_history = []

    for epoch in progress_bar:
        progress_bar.set_description(f"Epoch [{epoch+1:,}/{n_epochs:,}]")
        epoch_loss = 0.0
        n_batches = 0

        model.train()
        for batch_input, batch_target in train_loader:
            # batch_input:  (B, T, H, W)
            # batch_target: (B, 1, H, W)
            batch_input = batch_input.to(args.device).float()
            batch_target = batch_target.to(args.device).float()

            optimizer.zero_grad()
            batch_loss = 0.0

            recon, _, _ = model(batch_input, encoder_grad_types=args.encoder_grad)
            batch_loss += criterion(recon, batch_target)

            batch_loss = batch_loss / batch_input.size(0)
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            n_batches += 1
        train_loss_history.append(epoch_loss)

        # validation
        model.eval()
        val_loss = 0
        val_batches = 0
        for val_input, val_target in val_loader:
            val_batches += 1
            val_input = val_input.to(args.device).float()
            val_target = val_target.to(args.device).float()
            recon, _, _ = model(val_input)
            val_loss += criterion(recon, val_target).item() / val_input.size(0)

        # store the model if it's better than previous
        val_loss_history.append(val_loss)
        if val_loss < min(val_loss_history):
            torch.save({
                "model_state_dict": model.decoder.state_dict(),
                "train_loss_history": train_loss_history,
                "val_loss_history": val_loss_history
                }, f"{output_path.split('.')[0]}_best.pt")

        progress_bar.set_postfix(train_loss=epoch_loss, val_loss=val_loss)

    torch.save({
        "model_state_dict": model.decoder.state_dict(),
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history
        }, f"{output_path.split('.')[0]}_last.pt")
    print(f"Saved decoder weights to {output_path.split('.')[0]}_last.pt")

def main():
    parser = argparse.ArgumentParser(description="Train the retina autoencoder decoder from video windows.")
    parser.add_argument("--videos", type=str, required=True, help="h5 file containing videos.")
    parser.add_argument("--cell-params", type=str, default="params.yaml")
    parser.add_argument("--training-params", type=str, default="params_training.yaml")
    parser.add_argument("--output", type=str, default="best_autoencoder_decoder.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Video windows per training step.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--encoder-grad", nargs="*", default=None, help="Cell types that retain gradients during encoding.")
    parser.add_argument("--x", type=int, default=0, help="Top-left X crop coordinate.")
    parser.add_argument("--y", type=int, default=0, help="Top-left Y crop coordinate.")
    parser.add_argument("--random-crops", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    params = u.read_params(args.cell_params)
    train_autoencoder(args.videos, params, args.training_params, args.output, args)

if __name__ == "__main__":
    main()