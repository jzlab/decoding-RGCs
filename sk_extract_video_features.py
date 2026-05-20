import sys, os
import glob 
import cv2
import numpy as np
import pandas as pd

def extract_frame_features(video_path: str) -> list[dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    features = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ensure grayscale (handles both true BW and color input)
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Normalize to [0, 1] float for consistent metrics
        gray_f = gray.astype(np.float64) / 255.0

        # ------------------------------------------------------------------
        # 1. Mean luminance
        #    Simple mean of pixel intensities across the frame.
        # ------------------------------------------------------------------
        mean_luminance = gray_f.mean()

        # ------------------------------------------------------------------
        # 2. RMS contrast
        #    Standard deviation of pixel intensities.
        #    Note: this is a global measure — it won't capture local contrast.
        # ------------------------------------------------------------------
        rms_contrast = gray_f.std()

        # ------------------------------------------------------------------
        # 3. Spatial frequency content: ratio of high- to low-freq energy
        #    We split the FFT magnitude spectrum into a low- and high-freq
        #    half using a circular mask centred on DC.
        #    A high ratio → lots of fine detail; low ratio → smooth/blurry.
        # ------------------------------------------------------------------
        fft        = np.fft.fft2(gray_f)
        fft_shift  = np.fft.fftshift(fft)          # centre DC component
        magnitude  = np.abs(fft_shift)

        rows, cols = gray_f.shape
        crow, ccol = rows // 2, cols // 2
        radius     = min(rows, cols) // 4           # boundary between low/high

        Y, X = np.ogrid[:rows, :cols]
        dist_from_centre = np.sqrt((X - ccol)**2 + (Y - crow)**2)

        low_freq_energy  = magnitude[dist_from_centre <= radius].sum()
        high_freq_energy = magnitude[dist_from_centre >  radius].sum()
        # Guard against divide-by-zero on blank frames
        sf_ratio = high_freq_energy / (low_freq_energy + 1e-8)

        # ------------------------------------------------------------------
        # 4. Edge density
        #    Canny edge detection → proportion of pixels classified as edges.
        #    Thresholds (50, 150) are standard defaults; tune if needed.
        # ------------------------------------------------------------------
        edges        = cv2.Canny(gray, threshold1=50, threshold2=150)
        edge_density = (edges > 0).sum() / edges.size

        # ------------------------------------------------------------------
        # 5. Optical flow magnitude (Farneback dense flow)
        #    Mean magnitude of the flow field between this frame and the last.
        #    First frame has no predecessor, so we record NaN.
        # ------------------------------------------------------------------
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray,
                flow=None,
                pyr_scale=0.5,  # image pyramid scale
                levels=3,       # pyramid levels
                winsize=15,     # averaging window size
                iterations=3,
                poly_n=5,       # neighbourhood size for poly expansion
                poly_sigma=1.2,
                flags=0
            )
            # flow shape: (H, W, 2) — channel 0 = dx, channel 1 = dy
            flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
        else:
            flow_magnitude = float("nan")

        features.append({
            "frame":          frame_idx,
            "mean_luminance": round(mean_luminance, 4),
            "rms_contrast":   round(rms_contrast,   4),
            "sf_ratio":       round(sf_ratio,        4),
            "edge_density":   round(edge_density,    4),
            "flow_magnitude": round(flow_magnitude,  4) if not np.isnan(flow_magnitude) else None,
        })

        prev_gray = gray
        frame_idx += 1

    cap.release()
    return features



if __name__ == "__main__":

    """if os.isdir(sys.argv[0]):
        files = glob.glob(f"{sys.argv[0]}/*.mp4")
    else:
        files = [sys.argv[0]]"""

    results = extract_frame_features("/home/rotation/Downloads/CMD/testing/waves2_reconstructed_ONParasol.mp4")
    df = pd.DataFrame(results)
    print(df.head(10))
    df.to_csv("frame_features_ONParasol.csv", index=False)