import argparse
import itertools
import os

import decord
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils._pytree import tree_map
from tqdm import tqdm

from sk_retina_autoencoder import RetinaAutoencoder
from sk_utils import Utils as u

def _to_gray(raw):
    return (raw @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)) / 255.0

def crop_video(frames, H_full, W_full, x_start, y_start, target_h, target_w):
    x_end, y_end = x_start + target_w, y_start + target_h
    if x_end <= W_full and y_end <= H_full:
        return frames[:, y_start:y_end, x_start:x_end]

    valid_x_end = min(x_end, W_full)
    valid_y_end = min(y_end, H_full)
    crop = frames[:, y_start:valid_y_end, x_start:valid_x_end]
    pad_w = target_w - crop.shape[2]
    pad_h = target_h - crop.shape[1]
    return np.pad(crop, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")

class VideoWindowIterableDataset(IterableDataset):
    """
    Streams fixed temporal windows sampled randomly from a list of videos.

    VideoReader objects are opened inside __iter__, after DataLoader forks
    worker processes, so each worker owns its own file handles. Video paths
    are sharded across workers to avoid redundant I/O.

    Args:
        video_paths:  List of paths to video files.
        target_h:     Crop height in pixels.
        target_w:     Crop width in pixels.
        crop_coords:  Fixed (x, y) top-left crop coordinate. Ignored when
                      random_crops=True.
        window_size:  Number of frames in the prediction target.
        overlap:      Number of preceding context frames prepended to each
                      window (typically window_size - 1 for a causal model).
        random_crops: If True, sample a uniformly random crop origin per window.
        seed:         Base RNG seed. Each worker derives an independent seed
                      from [seed, worker_id] to avoid correlated sampling.
    """

    def __init__(
        self,
        video_paths,
        target_h,
        target_w,
        crop_coords=(0, 0),
        window_size=16,
        overlap=15,
        random_crops=False,
        seed=0,
    ):
        super().__init__()
        self.video_paths = list(video_paths)
        self.target_h = int(target_h)
        self.target_w = int(target_w)
        self.crop_coords = tuple(crop_coords)
        self.window_size = max(1, int(window_size))
        self.overlap = max(0, int(overlap))
        self.random_crops = bool(random_crops)
        self.seed = int(seed)
        self._min_frames = self.window_size + self.overlap

    def _sample_crop(self, rng, h_full, w_full):
        if self.random_crops:
            x = int(rng.integers(0, max(1, w_full - self.target_w + 1)))
            y = int(rng.integers(0, max(1, h_full - self.target_h + 1)))
            return x, y
        return self.crop_coords

    def _read_window(self, rng, vr):
        total_frames = len(vr)
        if total_frames < self._min_frames:
            return None

        start = int(rng.integers(0, total_frames - self.window_size + 1))
        end = start + self.window_size

        raw = vr.get_batch(list(range(start, end))).asnumpy()
        frames = _to_gray(raw).astype(np.float32)

        h_full, w_full = frames.shape[1], frames.shape[2]
        cx, cy = self._sample_crop(rng, h_full, w_full)
        frames = crop_video(frames, h_full, w_full, cx, cy, self.target_h, self.target_w)

        input_frames = torch.from_numpy(np.ascontiguousarray(frames[:self.window_size]))
        target_frames = input_frames[-1:]
        return input_frames, target_frames

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is not None:
            paths = self.video_paths[worker_info.id :: worker_info.num_workers]
            rng = np.random.default_rng([self.seed, worker_info.id])
        else:
            paths = self.video_paths
            rng = np.random.default_rng(self.seed)

        if not paths:
            return

        # Open readers inside the worker, after fork, so handles are not shared.
        readers = []
        for p in paths:
            try:
                readers.append(decord.VideoReader(p, ctx=decord.cpu(0)))
            except Exception as e:
                wid = worker_info.id if worker_info else 0
                print(f"[worker {wid}] Could not open {p}: {e}")

        readers = [vr for vr in readers if len(vr) >= self._min_frames]

        if not readers:
            wid = worker_info.id if worker_info else 0
            print(f"[worker {wid}] No usable videos after filtering.")
            return

        while True:
            vr = readers[int(rng.integers(0, len(readers)))]
            result = self._read_window(rng, vr)
            if result is not None:
                yield result

def train_autoencoder(video_paths, params, training_params_path, output_path, args):
    """Train the decoder end-to-end across all configured videos."""
    train_cfg = u.read_params(training_params_path)
    video_params = dict(params.get("video_parameters", {}))
    fps = u.read_video(video_paths[0], stream=True, chunk_frames=1)[1]["fps"]
    video_params["frame_rate"] = fps
    target_h, target_w = video_params.get("frame_shape", [128, 128])

    print(f"Initializing model with {len(video_paths)} videos, frame shape {target_h}x{target_w}")
    model = RetinaAutoencoder(
        {k: v for k, v in params.items() if k != "video_parameters"},
        video_params,
        decoder_params={
            "frame_shape": (target_h, target_w),
            "num_blocks": train_cfg.get("num_blocks", 4),
            "num_kernels": train_cfg.get("num_kernels", 64),
            "bias": train_cfg.get("bias", False),
        },
        cell_minibatch_size=args.cell_minibatch,
        temporal_batch_size=args.temp_batch,
    ).to(args.device)

    lr = args.lr or train_cfg.get("learning_rate", 1e-4)
    print(f"Using optimizer settings: lr={lr}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    window_size = params.get("ON_Parasol").get("temporal").get("window_size", 16)
    overlap = 0

    dataset = VideoWindowIterableDataset(
        video_paths=video_paths,
        target_h=target_h,
        target_w=target_w,
        crop_coords=(args.x or 0, args.y or 0),
        window_size=window_size,
        overlap=overlap,
        random_crops=args.random_crops,
        seed=args.seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or 1,
        num_workers=args.num_workers or 0,
        pin_memory=args.device.startswith("cuda"),
        # persistent_workers keeps worker processes alive across steps,
        # avoiding the overhead of re-opening VideoReaders each epoch.
        persistent_workers=(args.num_workers or 0) > 0,
    )

    n_epochs = args.epochs or train_cfg.get("epochs", 1)
    steps_per_epoch = args.steps_per_epoch or train_cfg.get("steps_per_epoch", 1000)
    print(f"Starting training: {n_epochs} epochs x {steps_per_epoch} steps, batch size {args.batch_size or 1}")
    print(f"Training on device: {args.device}")

    model.train()
    for epoch in tqdm(range(n_epochs), desc="Autoencoder epochs"):
        epoch_loss = 0.0
        n_batches = 0

        for batch_input, batch_target in itertools.islice(loader, steps_per_epoch):
            batch_input = batch_input.to(args.device).float()
            batch_target = batch_target.to(args.device).float()

            optimizer.zero_grad()

            # Run the full batch through the encoder/decoder path in one vectorized pass.
            recon_batch = torch.vmap(
                lambda sample_input: model(sample_input, encoder_grad_types=args.encoder_grad)[0].squeeze(1)
            )(batch_input)

            batch_loss = criterion(recon_batch, batch_target)
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            n_batches += 1

        print(f"Epoch {epoch + 1:03d} | loss={epoch_loss / max(1, n_batches):.6f}")

    state_dict = tree_map(lambda x: x[0], batched_params)
    torch.save({"model_state_dict": model.decoder.state_dict()}, output_path)
    print(f"Saved decoder weights to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train the decoder in one pass from video windows.")
    parser.add_argument("--video", type=str, required=True, help="Video file or directory of videos to use.")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to the retina model parameters.")
    parser.add_argument("--training-params", type=str, default="params_training.yaml", help="Path to training settings.")
    parser.add_argument("--output", type=str, default="best_autoencoder_decoder.pt", help="Where to save the trained decoder.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Number of gradient steps per epoch.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of video windows per training step.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--encoder-grad", nargs="*", default=None, help="Cell types allowed to keep gradients during encoding.")
    parser.add_argument("--x", type=int, default=0, help="Top-left X crop coordinate.")
    parser.add_argument("--y", type=int, default=0, help="Top-left Y crop coordinate.")
    parser.add_argument("--random-crops", action="store_true", help="Use a random crop for every sampled training window.")
    parser.add_argument("--cell-minibatch", type=int, default=512, help="Cells per vectorized batch.")
    parser.add_argument("--temp-batch", type=int, default=64, help="Temporal windows per batch.")
    parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed for reproducible sampling.")
    args = parser.parse_args()

    params = u.read_params(args.params)

    if os.path.isdir(args.video):
        video_files = sorted([
            os.path.join(args.video, f)
            for f in os.listdir(args.video)
            if f.endswith((".mp4", ".avi", ".mkv"))
        ])
    else:
        video_files = [args.video]

    train_autoencoder(video_files, params, args.training_params, args.output, args)


if __name__ == "__main__":
    main()