import os
import argparse
import random
import numpy as np
import torch
import h5py
import json
import scipy.ndimage
from sk_utils import Utils as u
from sk_retina_encoders import Retina

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def apply_augmentation(full_video, cx, cy, target_h, target_w, angle, spatial_scale, temporal_scale):
    """
    Applies spatial rotation, spatial scaling, and temporal scaling to a bounding box around (cx, cy).
    NOTE: Requires the full video in RAM because scipy rotation needs random access.
    """
    T_full, H_full, W_full = full_video.shape
    
    # 1. Determine bounding box for rotation to avoid black corners
    h_src = target_h * spatial_scale
    w_src = target_w * spatial_scale
    D = int(np.ceil(np.sqrt(h_src**2 + w_src**2)))
    
    # Center of the requested target crop
    center_x = cx + target_w / 2.0
    center_y = cy + target_h / 2.0
    
    half_D = int(np.ceil(D / 2.0))
    x_min = int(center_x - half_D)
    x_max = int(center_x + half_D)
    y_min = int(center_y - half_D)
    y_max = int(center_y + half_D)
    
    # Pad if out of bounds
    pad_left   = max(0, -x_min)
    pad_top    = max(0, -y_min)
    pad_right  = max(0, x_max - W_full)
    pad_bottom = max(0, y_max - H_full)
    
    valid_x_min = max(0, x_min)
    valid_y_min = max(0, y_min)
    valid_x_max = min(W_full, x_max)
    valid_y_max = min(H_full, y_max)
    
    crop_D = full_video[:, valid_y_min:valid_y_max, valid_x_min:valid_x_max]
    
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop_D = np.pad(crop_D, ((0,0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
        
    # 2. Spatial Rotation
    if angle != 0:
        rotated_D = scipy.ndimage.rotate(crop_D, angle, axes=(1, 2), reshape=False, mode='reflect')
    else:
        rotated_D = crop_D
        
    # 3. Center crop to (h_src, w_src)
    ch_src, cw_src = int(np.ceil(h_src)), int(np.ceil(w_src))
    y_start = half_D - ch_src // 2
    x_start = half_D - cw_src // 2
    y_end   = y_start + ch_src
    x_end   = x_start + cw_src
    
    final_crop = rotated_D[:, y_start:y_end, x_start:x_end]
    
    # 4. PyTorch interpolation for exact sizing and temporal scaling
    vid_tensor = torch.from_numpy(final_crop).unsqueeze(0).float()  # (1, T, H, W)
    
    # Spatial resize
    if ch_src != target_h or cw_src != target_w:
        vid_tensor = torch.nn.functional.interpolate(
            vid_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False
        )
        
    # Temporal scaling
    if temporal_scale != 1.0:
        T_new = int(np.round(T_full * temporal_scale))
        vid_tensor = vid_tensor.reshape(1, T_full, -1).permute(0, 2, 1)  # (1, H*W, T)
        vid_tensor = torch.nn.functional.interpolate(vid_tensor, size=T_new, mode='linear', align_corners=False)
        vid_tensor = vid_tensor.permute(0, 2, 1).reshape(1, T_new, target_h, target_w)
        
    return vid_tensor.squeeze(0).numpy()


def crop_video(full_video, H_full, W_full, x_start, y_start, target_h, target_w):
    """
    Applies a spatial crop to the loaded full video, using zero padding if out of bounds.
    """
    x_end, y_end = x_start + target_w, y_start + target_h

    if x_end <= W_full and y_end <= H_full:
        video_frames = full_video[:, y_start:y_end, x_start:x_end]
    else:
        valid_x_end = min(x_end, W_full)
        valid_y_end = min(y_end, H_full)
        crop = full_video[:, y_start:valid_y_end, x_start:valid_x_end]
        pad_w = target_w - crop.shape[2]
        pad_h = target_h - crop.shape[1]
        video_frames = np.pad(crop, ((0,0), (0, pad_h), (0, pad_w)), mode='constant')
        
    return video_frames


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def setup_retina(params, fps, cell_minibatch=None, temp_batch=None, device="cpu"):
    # print("Initializing Retina model...")
    video_params = params.get("video_parameters", {})
    video_params["frame_rate"] = fps
    
    retina = Retina(
        params, 
        video_params, 
        cell_minibatch_size=cell_minibatch, 
        temporal_batch_size=temp_batch
    )
    retina.to(device)
    retina.eval()
    return retina


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def get_activations_bulk(video_frames, retina, device="cpu"):
    """Run the full video through the retina in one shot (used for augmented trials)."""
    video_tensor = torch.from_numpy(np.ascontiguousarray(video_frames)).to(device).float()
    
    with torch.no_grad():
        _, firing_rates = retina(video_tensor, pad=False)

    del video_tensor
    torch.cuda.empty_cache()
    return firing_rates



# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(video_path, output_path, args, params, target_h, target_w,
                  pbar=None):
    """
    Stream activations for each crop and write results directly to HDF5.

    The video is never loaded in full.  For each streaming chunk:
      - Cropped frames are appended directly to a resizable HDF5 dataset.
      - The GPU forward pass runs on that chunk and the activation windows
        are accumulated in memory (firing rates are far smaller than raw video).

    Augmented trials still require the full video in RAM because scipy's
    spatial rotation needs random frame access.
    """

    # 1. Video properties (no frame decode)
    _, props = u.read_video(video_path, stream=True, chunk_frames=1)
    T      = props["total_frames"] - 1
    H_full = props["height"]
    W_full = props["width"]
    fps    = props["fps"]

    # 2. Setup Retina model
    retina = setup_retina(params, fps, args.cell_minibatch, args.temp_batch, args.device)
    mosaic_names = [m.m_params["cell_type"] for m in retina.mosaics]
    win_size = len(retina.mosaics[0].temporal_filter_tensor)
    overlap  = win_size - 1

    # 3. Crop coordinates
    if args.random_crops > 0:
        coords = [
            (random.randint(0, max(0, W_full - target_w)),
             random.randint(0, max(0, H_full - target_h)))
            for _ in range(args.random_crops)
        ]
    else:
        coords = [(args.x or 0, args.y or 0)]

    # 4. Load full video only when augmentation requires random frame access
    needs_full_video = args.augment and args.n_augs > 0
    full_video = u.read_video(video_path)[0] if needs_full_video else None

    # 5. Open HDF5 once and write everything incrementally
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vname     = os.path.basename(video_path)
    trial_idx = 0

    with h5py.File(output_path, "w") as hf:
        # Header
        v_info = hf.create_group("video_info")
        v_info.attrs["path"]  = video_path
        v_info.attrs["fps"]   = fps
        v_info.attrs["shape"] = (T, H_full, W_full)
        hf.attrs["mosaic_names"] = json.dumps(mosaic_names)
        trials_grp = hf.create_group("trials")

        for cx, cy in coords:
            if pbar is not None:
                pbar.set_postfix_str(f"{vname} crop({cx},{cy})")

            # ------------------------------------------------------------------
            # Base trial: stream the video, write frames and compute activations
            # chunk by chunk — no full-video load.
            # ------------------------------------------------------------------
            trial_grp = trials_grp.create_group(str(trial_idx))
            fr_grp    = trial_grp.create_group("firing_rates")

            # Resizable dataset for target video frames.
            # HDF5 chunks are set to one frame row to allow efficient appending.
            target_ds = trial_grp.create_dataset(
                "target_video",
                shape=(0, target_h, target_w),
                maxshape=(None, target_h, target_w),
                dtype="float32",
                compression="lzf",
                chunks=(1, target_h, target_w),
            )

            mosaic_rate_chunks = None
            is_first_chunk     = True

            frame_gen, _ = u.read_video(
                video_path,
                stream=True,
                chunk_frames=args.stream_chunk,
                win_overlap=overlap,
            )

            for chunk in frame_gen:
                chunk_cropped = crop_video(chunk, H_full, W_full, cx, cy, target_h, target_w)

                # Write new frames to HDF5 (skip the overlap prefix on non-first chunks
                # — those frames were already written by the previous iteration).
                frames_to_save = chunk_cropped if is_first_chunk else chunk_cropped[overlap:]
                n_existing = target_ds.shape[0]
                target_ds.resize(n_existing + len(frames_to_save), axis=0)
                target_ds[n_existing:] = frames_to_save

                # GPU forward pass
                chunk_tensor = torch.from_numpy(np.ascontiguousarray(chunk_cropped)).to(args.device).float()
                with torch.no_grad():
                    _, chunk_rates = retina(chunk_tensor, pad=False)

                if mosaic_rate_chunks is None:
                    mosaic_rate_chunks = [[] for _ in chunk_rates]
                for m_idx, rate in enumerate(chunk_rates):
                    mosaic_rate_chunks[m_idx].append(rate.cpu())

                del chunk_tensor, chunk_rates, chunk_cropped, chunk
                torch.cuda.empty_cache()
                is_first_chunk = False

            # Write accumulated firing rates (much smaller than raw video)
            for m_idx, rate_chunks in enumerate(mosaic_rate_chunks):
                # Transpose to (T_windows, n_cells) for faster reading
                full_rate = torch.cat(rate_chunks, dim=1).T
                fr_grp.create_dataset(str(m_idx), data=full_rate.numpy(), compression="lzf")
                del full_rate

            # Trial metadata
            crop_grp = trial_grp.create_group("crop")
            for k, v in {"x": cx, "y": cy, "target_h": target_h, "target_w": target_w}.items():
                crop_grp.attrs[k] = v
            aug_grp = trial_grp.create_group("augmentation")
            for k, v in {"angle": 0.0, "spatial_scale": 1.0, "temporal_scale": 1.0}.items():
                aug_grp.attrs[k] = v

            torch.cuda.empty_cache()
            trial_idx += 1

            # ------------------------------------------------------------------
            # Augmented trials: scipy rotation needs the full video in RAM.
            # This is unavoidable without a full rewrite of the augmentation
            # pipeline; it is only triggered when --augment is passed.
            # ------------------------------------------------------------------
            if args.augment and full_video is not None:
                for _ in range(args.n_augs):
                    angle   = random.uniform(0, 360)
                    s_scale = random.uniform(0.5, 2.0)
                    t_scale = random.uniform(0.5, 2.0)

                    aug_frames = apply_augmentation(
                        full_video, cx, cy, target_h, target_w, angle, s_scale, t_scale
                    )
                    aug_rates = get_activations_bulk(aug_frames, retina, args.device)

                    aug_trial = trials_grp.create_group(str(trial_idx))

                    aug_fr_grp = aug_trial.create_group("firing_rates")
                    for m_idx, rate in enumerate(aug_rates):
                        # Transpose to (T_windows, n_cells) for faster reading
                        aug_fr_grp.create_dataset(str(m_idx), data=rate.cpu().numpy().T, compression="lzf")

                    aug_trial.create_dataset("target_video", data=aug_frames, compression="lzf")

                    aug_crop = aug_trial.create_group("crop")
                    for k, v in {"x": cx, "y": cy, "target_h": target_h, "target_w": target_w}.items():
                        aug_crop.attrs[k] = v
                    aug_aug = aug_trial.create_group("augmentation")
                    for k, v in {"angle": angle, "spatial_scale": s_scale, "temporal_scale": t_scale}.items():
                        aug_aug.attrs[k] = v

                    del aug_frames, aug_rates
                    torch.cuda.empty_cache()
                    trial_idx += 1

            if pbar is not None:
                pbar.update(1)

    if full_video is not None:
        del full_video


def main():
    parser = argparse.ArgumentParser(
        description="Generate RGC activations from a video using the Retina model."
    )
    parser.add_argument("--video",   type=str, required=True,
                        help="Path to the input video file or directory of videos.")
    parser.add_argument("--params",  type=str, default="params.yaml",
                        help="Path to params.yaml.")
    parser.add_argument("--output",  type=str, required=True,
                        help="Path to save the output activations (.h5) or directory if --video is a directory.")
    parser.add_argument("--x",       type=int, default=None,
                        help="Top-left X coordinate (column) for the spatial crop.")
    parser.add_argument("--y",       type=int, default=None,
                        help="Top-left Y coordinate (row) for the spatial crop.")
    parser.add_argument("--random-crops", type=int, default=0,
                        help="If >0, ignores x/y and generates N random crops.")
    parser.add_argument("--augment", action="store_true",
                        help="Enable randomized augmentations (rotation, spatial scale, temporal scale).")
    parser.add_argument("--n-augs",  type=int, default=0,
                        help="Number of augmented variations to generate per crop.")
    parser.add_argument("--cell-minibatch", type=int, default=500,
                        help="Number of cells to process in a single vectorized batch.")
    parser.add_argument("--temp-batch",     type=int, default=50,
                        help="Number of temporal windows to process in a single vectorized batch.")
    parser.add_argument("--stream-chunk",   type=int, default=1500,
                        help="Number of raw frames per streaming chunk for the base trial. "
                             "Lower = less RAM, more disk I/O. Default 1500.")
    parser.add_argument("--device",  type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on.")

    args = parser.parse_args()

    params = u.read_params(args.params)
    video_params = params.get("video_parameters", {})
    target_h, target_w = video_params.get("frame_shape", [128, 128])

    # Crops per video is known purely from args — no need to open any files.
    crops_per_video = args.random_crops if args.random_crops > 0 else 1

    if os.path.isdir(args.video):
        valid_exts = [".mp4", ".avi", ".mkv"]
        video_files = sorted(
            f for f in os.listdir(args.video)
            if any(f.endswith(ext) for ext in valid_exts)
        )
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        work = [
            (os.path.join(args.video, v),
             os.path.join(args.output, f"{os.path.splitext(v)[0]}_acts.h5"))
            for v in video_files
        ]
    else:
        work = [(args.video, args.output)]

    # Skip videos whose output file already exists.
    pending = [(v, o) for v, o in work if not os.path.exists(o)]
    skipped = len(work) - len(pending)
    if skipped:
        print(f"Skipping {skipped} already-completed video(s).")

    total_crops = len(pending) * crops_per_video
    print(f"Processing {len(pending)} video(s) × {crops_per_video} crop(s) = {total_crops} total.")

    with tqdm(total=total_crops, unit="crop", desc="Activations") as pbar:
        for v_path, out_path in pending:
            process_video(v_path, out_path, args, params, target_h, target_w, pbar=pbar)


if __name__ == "__main__":
    main()
