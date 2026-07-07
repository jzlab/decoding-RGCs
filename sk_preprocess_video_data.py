"""
sk_preprocess_videos.py

Convert a folder of videos into a single HDF5 file for fast training data access.

Each video is stored as a separate dataset named by its stem:
    f["video_001"]  ->  (T, H, W)  uint8

Metadata is stored as dataset attributes:
    f["video_001"].attrs["fps"]           -> float
    f["video_001"].attrs["frame_count"]   -> int
    f["video_001"].attrs["original_h"]    -> int
    f["video_001"].attrs["original_w"]    -> int

A top-level attribute records the preprocessing config:
    f.attrs["chunk_size"]   (temporal chunk size used)
    f.attrs["created"]      (ISO timestamp)

Frames are stored as uint8 (0-255 grayscale) rather than float32 to keep
file size 4x smaller. The /255 normalisation is done in the dataset __iter__.

Chunk size is set to (1, orig_h, orig_w) so that a single
training window maps to exactly one HDF5 chunk — minimising wasted reads.

Usage
-----
python sk_preprocess_videos.py \\
    --input  ~/Downloads/CMD/training/split/ \\
    --output ~/Downloads/CMD/training/data.h5 \\
    --compression lzf 
"""

import argparse
import os
import queue
import threading
from datetime import datetime, timezone

import decord
import h5py
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Frame utilities  (mirrors sk_train_autoencoder.py)
# ---------------------------------------------------------------------------

def _to_gray(raw: np.ndarray) -> np.ndarray:
    """Convert (T, H, W, 3) uint8 RGB to (T, H, W) float32 in [0, 255]."""
    return raw @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(path, decode_batch, h5_file, comp) -> dict | None:
    """
    Decode all frames from a video, convert to grayscale uint8, centre-crop.

    Returns a dict with keys:
        name        : str   dataset name (video file stem)
        frames      : np.ndarray  (T, H, W) uint8
        fps         : float
        original_h  : int
        original_w  : int

    Returns None on failure so the caller can skip gracefully.
    """

    try:
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        fps = vr.get_avg_fps()
        T = len(vr)

        if T == 0:
            print(f"  [skip] {os.path.basename(path)} — no frames")
            return None

        # start dataset
        first = vr[0].asnumpy()
        orig_h, orig_w = first.shape[0], first.shape[1]
        name = os.path.splitext(os.path.basename(path))[0]
        ds = h5_file.create_dataset(name, shape=(T, orig_h, orig_w), dtype=np.uint8, chunks=(1, orig_h, orig_w), compression=comp)

        # Decode in batches to bound peak memory
        for batch_start in range(0, T, decode_batch):
            batch_end = min(batch_start + decode_batch, T)
            indices = list(range(batch_start, batch_end))
            raw = vr.get_batch(indices).asnumpy()          # (B, H, W, 3)
            gray = _to_gray(raw)                            # (B, H, W) float32
            ds[batch_start:batch_end] = np.clip(gray, 0, 255).astype(np.uint8)

        # store relevant attributes
        ds.attrs["fps"] = float(fps)
        ds.attrs["frame_count"] = T
        ds.attrs["original_h"] = int(orig_h)
        ds.attrs["original_w"] = int(orig_w)
        return True

    except Exception as e:
        print(f"  [error] {os.path.basename(path)}: {e}")
        return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a folder of videos into a single HDF5 file."
    )
    parser.add_argument("--input", type=str, required=True, help="Directory containing input video files.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output .h5 file.")
    parser.add_argument("--compression", type=str, default="lzf", choices=["lzf", "gzip", "none"], help="HDF5 chunk compression. lzf is fastest; none for benchmarking.")
    parser.add_argument("--decode-batch", type=int, default=256, help="Frames decoded per decord.get_batch call.")
    parser.add_argument("--extensions",  type=str, nargs="+", default=[".mp4", ".avi", ".mkv"], help="Video file extensions to include.")
    args = parser.parse_args()

    # Collect video paths
    video_paths = sorted([os.path.join(args.input, f) for f in os.listdir(args.input) if os.path.splitext(f)[1].lower() in args.extensions])
    if not video_paths:
        print(f"No videos found in {args.input} with extensions {args.extensions}")
        return

    print(f"Found {len(video_paths)} videos.")
    print(f"Output: {args.output}, compression={args.compression}")

    # write the files to output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f.attrs["compression"] = args.compression or "none"
        f.attrs["created"] = datetime.now(timezone.utc).isoformat()

        for vp in tqdm(video_paths):
            process_video(vp, args.decode_batch, f, args.compression)

    print(f"\nDone. {len(video_paths)} videos total.")

if __name__ == "__main__":
    main()