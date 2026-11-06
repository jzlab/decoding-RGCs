import os
import sys
import random
import yaml

import numpy as np
import cv2
import decord
import torch
import numexpr as ne
import h5py
import glob
from tqdm import tqdm

class Utils:

    def check_gpus():

        print(f"GPUs present?\t{torch.cuda.is_available()}\t{torch.cuda.device_count()}")
        for d in range(torch.cuda.device_count()):
            print(f"Device {d}: {torch.cuda.get_device_name(d)}")

    def read_params(path:str="params.yaml"):
        """read parameters from yaml file for RGC details

        Args:
            path (str, optional): path to yaml file. Defaults to "params.yaml".
        """

        # set the default loader to handle math expressions
        def expr_contructor(loader, node):
            expr = loader.construct_scalar(node)
            return float(ne.evaluate(expr).item())
        yaml.SafeLoader.add_constructor("!expr", expr_contructor)

        with open(path, 'r') as file:
            params = yaml.safe_load(file)

        # convert params to int/float if necessary (just in case they havent been read properly)
        for key, value in params.items():
            if isinstance(value, str):
                try:
                    params[key] = float(value)
                except ValueError:
                    pass

        return params
    
    def read_video(path: str, stream: bool = False, chunk_frames: int = 1000, win_overlap: int = 0):
        """Read a video file, either fully or as a streaming generator.

        Args:
            path (str): Path to the video file.
            stream (bool): If False (default), load all frames into memory and return
                a single numpy array.  If True, return a generator that yields one
                numpy chunk at a time so that only `chunk_frames` frames are ever
                held in memory simultaneously.
            chunk_frames (int): Number of *new* frames per yielded chunk when
                streaming.  Ignored when stream=False.
            win_overlap (int): Number of frames from the end of the previous chunk
                to prepend to each new chunk.  Use this to provide the temporal
                filter's look-back window so boundary windows are computed correctly.
                Ignored when stream=False.

        Returns:
            Non-streaming: (frames_array, props_dict)
                frames_array: float32 ndarray of shape (T, H, W), values in [0, 1].
                props_dict:   {"fps", "width", "height", "total_frames"}.
            Streaming: (generator, props_dict)
                Each iteration of the generator yields a float32 ndarray of shape
                (chunk_T, H, W).  The very first chunk has no prepended overlap.
        """

        # Open the video and collect properties (fast — no frame decode yet)
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        fps = vr.get_avg_fps()
        height, width = vr[0].shape[:2]
        total_frames = len(vr)
        props = {"fps": fps, "width": width, "height": height, "total_frames": total_frames}

        def _to_gray(raw):
            """Convert a uint8 (T, H, W, 3) batch to float32 (T, H, W) grayscale."""
            return (raw @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)) / 255.0

        if not stream:
            # --- Bulk mode: original behaviour ---
            frames = vr.get_batch(range(total_frames - 1)).asnumpy()
            frames = _to_gray(frames)
            return frames, props

        # --- Streaming mode ---
        def _frame_generator():
            prev_tail = None  # holds the overlap tail from the previous chunk

            for chunk_start in range(0, total_frames, chunk_frames):
                chunk_end = min(chunk_start + chunk_frames, total_frames)
                indices = list(range(chunk_start, chunk_end))
                raw = vr.get_batch(indices).asnumpy()
                chunk = _to_gray(raw)

                if prev_tail is not None and win_overlap > 0:
                    # Prepend the overlap from the previous chunk so the caller
                    # can form complete temporal windows across the boundary.
                    chunk = np.concatenate([prev_tail, chunk], axis=0)

                # Save the tail of this chunk (before yield, so it's not GC'd)
                if win_overlap > 0:
                    prev_tail = chunk[-win_overlap:]

                yield chunk

                # Explicitly delete raw frames to free memory promptly
                del raw, chunk

        return _frame_generator(), props

    @staticmethod
    def migrate_h5_transposition(path: str):
        """Migrate HDF5 activation files to transposed format (T_windows, n_cells) in-place.
        
        Args:
            path (str): Path to a single .h5 file or a directory containing them.
        """
        if os.path.isdir(path):
            h5_files = sorted(glob.glob(os.path.join(path, "*.h5")))
        elif os.path.isfile(path):
            h5_files = [path]
        else:
            print(f"Error: {path} is not a valid file or directory.")
            return

        print(f"Found {len(h5_files)} files to migrate in-place.")
        
        for fp in tqdm(h5_files, desc="Migrating In-Place"):
            try:
                with h5py.File(fp, 'r+') as f:
                    if "trials" not in f:
                        continue

                    trials_grp = f["trials"]
                    for trial_id in trials_grp.keys():
                        trial_grp = trials_grp[trial_id]
                        if "firing_rates" not in trial_grp:
                            continue
                            
                        fr_grp = trial_grp["firing_rates"]
                        for m_id in list(fr_grp.keys()):
                            ds = fr_grp[m_id]
                            data = ds[()]
                            
                            if data.ndim == 2:
                                # Old format: (n_cells, T) -> shape[0] < shape[1]
                                if data.shape[0] > data.shape[1]:
                                    continue # Already transposed
                                
                                transposed_data = data.T
                                compression = ds.compression
                                del fr_grp[m_id]
                                fr_grp.create_dataset(m_id, data=transposed_data, compression=compression or "lzf")
            except Exception as e:
                print(f"Error migrating {os.path.basename(fp)}: {e}")

        print("\nMigration complete. Reclaim space with 'h5repack' if desired.")

    @staticmethod
    def reconstruct_batch(folder_path, weights_path, params_path="params.yaml", training_params_path="params_training.yaml", device=None):
        """
        Runs sk_reconstruct_video.py on all videos in a folder.
        For each video, it runs:
        1. Full reconstruction (all 4 cell types)
        2. Masked reconstructions (only 1 cell type active at a time)
        
        Args:
            folder_path (str): Path to folder containing .mp4 or .avi videos.
            weights_path (str): Path to the decoder weights file (.pt).
            params_path (str): Path to the params.yaml file.
            training_params_path (str): Path to the training params file.
            device (str): 'cuda' or 'cpu'. Defaults to auto-detect.
        """
        import subprocess
        import glob
        import os
        import sys

        # Find videos
        extensions = ['*.mp4', '*.avi', '*.mov']
        video_files = []
        for ext in extensions:
            video_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        if not video_files:
            print(f"No videos found in {folder_path}")
            return

        print(f"Found {len(video_files)} videos. Starting batch reconstruction...")
        
        python_exec = sys.executable
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        for video in video_files:
            basename = os.path.splitext(os.path.basename(video))[0]
            # Create a dedicated activations file for this video in its current folder
            act_path = os.path.join(folder_path, f"{basename}_activations.pt")
            
            # Setup environment to avoid MKL threading issues
            env = os.environ.copy()
            env["MKL_THREADING_LAYER"] = "GNU"

            # --- 1. Full Reconstruction ---
            out_all = os.path.join(folder_path, f"{basename}_reconstructed_all.mp4")
            print(f"\n[Batch] Processing {basename}: FULL reconstruction...")
            cmd_all = [
                python_exec, "sk_reconstruct_video.py",
                "--video", video,
                "--weights", weights_path,
                "--params", params_path,
                "--training-params", training_params_path,
                "--output-activations", act_path,
                "--output-video", out_all,
                "--device", device
            ]
            subprocess.run(cmd_all, check=True, env=env)
            
            # --- 2. Masked Reconstructions (One cell type at a time) ---
            # There are 4 cell types: 0, 1, 2, 3
            celltypes = {
                "0": "ONParasol",
                "1": "OFFParasol",
                "2": "ONMidget",
                "3": "OFFMidget"
            }
            for i in range(4):
                # zero out everything EXCEPT cell type i
                zeros = [str(j) for j in range(4) if j != i]
                out_masked = os.path.join(folder_path, f"{basename}_reconstructed_{celltypes[str(i)]}.mp4")
                
                print(f"\n[Batch] Processing {basename}: ONLY {celltypes[str(i)]}...")
                cmd_masked = [
                    python_exec, "sk_reconstruct_video.py",
                    "--video", video,
                    "--weights", weights_path,
                    "--params", params_path,
                    "--training-params", training_params_path,
                    "--activations", act_path,  # Load the activations we just generated
                    "--output-video", out_masked,
                    "--device", device,
                    "--zero-celltype"
                ] + zeros
                subprocess.run(cmd_masked, check=True, env=env)
            
            print(f"\n[Batch] Finished {basename}.")
