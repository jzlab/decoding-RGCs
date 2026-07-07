import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f

__all__ = [
    "Retina",
    "RetinalGanglionCellMosaic",
    "SmoothMonostratifiedCellMosaic",
    "UnnamedCellMosaic",
]


class Retina(nn.Module):
    """Stores and manages multiple RGC mosaics."""

    def __init__(self, model_parameters: dict, video_parameters: dict):
        super().__init__()

        self.m_params = model_parameters
        self.v_params = video_parameters

        self.mosaics = nn.ModuleList()
        for cell_type, params in self.m_params.items():
            if cell_type != "video_parameters":
                if "Smooth_Monostratified" in cell_type:
                    self.mosaics.append(SmoothMonostratifiedCellMosaic(params, self.v_params))
                else:
                    self.mosaics.append(RetinalGanglionCellMosaic(params, self.v_params))

        self.n_cells = max(m.n_cells for m in self.mosaics)

    def forward(self, x, pad=True):
        """Forward pass through each RGC mosaic.

        Args:
            x (torch.Tensor): Input video of shape (B, T, H, W).

        Returns:
            tuple: (linear_responses, firing_rates), each (n_mosaics, n_cells, B).
        """
        res_linear = []
        res_rate   = []
        for mosaic in self.mosaics:
            lin, rate = mosaic(x)
            res_linear.append(lin)
            res_rate.append(rate)

        if pad:
            def pad_to(t, target_n):
                deficit = target_n - t.shape[0]
                if deficit > 0:
                    t = f.pad(t, (0, 0, 0, deficit), mode="constant", value=0)
                return t

            res_linear = [pad_to(t, self.n_cells) for t in res_linear]
            res_rate   = [pad_to(t, self.n_cells) for t in res_rate]

        return torch.stack(res_linear), torch.stack(res_rate)


class RetinalGanglionCellMosaic(nn.Module):
    """LN encoding model of a single RGC type mosaic.

    Processes batched video input of shape (B, T, H, W) where T equals
    the encoder's temporal window size. Each sample in the batch is treated
    as one independent temporal window.
    """

    def __init__(self, model_parameters: dict, video_parameters: dict):
        super().__init__()

        self.m_params = model_parameters
        self.v_params = video_parameters

        self.spatial_filter = self._spatial_filter()
        self.temporal_filter = self._temporal_filter()
        self.spatiotemporal_filter = self._spatiotemporal_filter()

        self.rf_diam = self.m_params["tiling_config"]["rf_diameter"]

        # Conv2d expects:
        # (out_channels, in_channels, kH, kW)
        self.register_buffer("w_conv", self.spatiotemporal_filter.unsqueeze(0))
        self.register_buffer("temporal_filter_tensor", self.temporal_filter)

        # store the mosaic locations
        self.mosaic = self._tile_cells()
        self.register_buffer("mosaic_tensor", torch.as_tensor(self.mosaic, dtype=torch.float32))
        self.n_cells = len(self.mosaic)

        # Cell locations in response-map coordinates
        self.register_buffer("cell_x", torch.floor(self.mosaic_tensor[:, 0]).long())
        self.register_buffer("cell_y", torch.floor(self.mosaic_tensor[:, 1]).long())

        self.nonlinearity = self._nonlinearity()

    # ------------------------------------------------------------------
    # Filter construction
    # ------------------------------------------------------------------

    def _spatial_filter(self):
        def gaussian_2d(w, h, sigma):
            cy, cx = h / 2.0, w / 2.0
            y  = torch.arange(h, dtype=torch.float32)
            x  = torch.arange(w, dtype=torch.float32)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            return torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

        w, h     = self.m_params["spatial"]["width"], self.m_params["spatial"]["height"]
        center   = gaussian_2d(w, h, self.m_params["spatial"]["center_size"])
        surround = gaussian_2d(w, h, self.m_params["spatial"]["surround_size"])
        center   = center   / torch.sum(center)
        surround = surround / torch.sum(surround)
        return (
            self.m_params["spatial"]["center_strength"]    * center
            + self.m_params["spatial"]["surround_strength"] * surround
        )

    def _temporal_filter(self, smooth: bool = True):
        temporal_cfg = self.m_params["temporal"]
        memory = int(temporal_cfg.get("memory_frames", temporal_cfg.get("memory_ms", 16)))
        dt_ms  = 1000.0 / self.v_params["frame_rate"]
        t_ms   = torch.arange(memory, dtype=torch.float32) * dt_ms

        lobe1 = self.m_params["temporal"]["amp1"] * torch.exp(
            -((t_ms - self.m_params["temporal"]["peak1_ms"]) ** 2)
            / (2 * self.m_params["temporal"]["width1_ms"] ** 2)
        )
        lobe2 = self.m_params["temporal"]["amp2"] * torch.exp(
            -((t_ms - self.m_params["temporal"]["peak2_ms"]) ** 2)
            / (2 * self.m_params["temporal"]["width2_ms"] ** 2)
        )

        if smooth:
            onset_window = 1.0 - torch.exp(-t_ms / 15.0)
            lobe1 = lobe1 * onset_window
            lobe2 = lobe2 * onset_window

        filt = (-lobe1 + lobe2) if self.m_params["cell_type"].startswith("OFF") else (lobe1 - lobe2)
        return torch.flip(filt, dims=[0])

    def _spatiotemporal_filter(self):
        st = self.temporal_filter.view(-1, 1, 1) * self.spatial_filter.view(1, *self.spatial_filter.shape)
        return st / torch.norm(st)

    # ------------------------------------------------------------------
    # Mosaic tiling
    # ------------------------------------------------------------------

    def _tile_cells(self, max_cells=None):
        height, width = self.v_params["frame_shape"]
        s = self.m_params["spatial"]["center_size"] / self.m_params["tiling_config"]["coverage_factor"]

        B = torch.tensor([
            [s,    s / 2.0],
            [0.0,  s * torch.sqrt(torch.tensor(3.0)) / 2.0],
        ], dtype=torch.float32)
        B_inv = torch.inverse(B)

        margin  = self.m_params["spatial"]["center_size"] / 2.0
        corners = torch.tensor([[0,0],[width,0],[0,height],[width,height]], dtype=torch.float32)
        lc      = corners @ B_inv.T

        n_min = int(torch.floor(lc[:, 0].min())) - 1
        n_max = int(torch.ceil( lc[:, 0].max())) + 1
        m_min = int(torch.floor(lc[:, 1].min())) - 1
        m_max = int(torch.ceil( lc[:, 1].max())) + 1

        all_positions = []
        for m in range(m_min, m_max + 1):
            for n in range(n_min, n_max + 1):
                pos = torch.tensor([float(n), float(m)], dtype=torch.float32) @ B.T
                x, y = pos[0].item(), pos[1].item()
                if margin <= x <= width - margin and margin <= y <= height - margin:
                    all_positions.append((x, y))

        all_positions = torch.tensor(all_positions, dtype=torch.float32)

        if max_cells is not None and len(all_positions) > max_cells:
            method = self.m_params["tiling_config"]["selection_method"]
            if method in ("center_first", "edge_first"):
                center_pt = torch.tensor([width / 2.0, height / 2.0], dtype=torch.float32)
                distances = torch.norm(all_positions - center_pt, dim=1)
                _, sorted_idx = torch.sort(distances, descending=(method == "edge_first"))
                all_positions = all_positions[sorted_idx[:max_cells]]
            elif method == "random":
                torch.manual_seed(42)
                all_positions = all_positions[torch.randperm(len(all_positions))[:max_cells]]
            elif method == "grid_order":
                all_positions = all_positions[:max_cells]

        offset = torch.tensor(self.m_params["tiling_config"]["offset"], dtype=torch.float32)
        return all_positions + offset

    def _calculate_vectorized_indices(self):
        height, width = self.v_params["frame_shape"]
        rf_diam       = self.rf_diam
        half_diam     = rf_diam / 2.0
        padded_width  = width + 2 * rf_diam

        y_range = torch.arange(rf_diam)
        x_range = torch.arange(rf_diam)
        yy, xx  = torch.meshgrid(y_range, x_range, indexing="ij")
        local_offsets = (yy * padded_width + xx).flatten()

        all_indices = []
        for pos in self.mosaic_tensor:
            pos_x, pos_y   = pos[0].item(), pos[1].item()
            v_start_padded = math.floor(pos_y - half_diam) + rf_diam
            h_start_padded = math.floor(pos_x - half_diam) + rf_diam
            base_index     = v_start_padded * padded_width + h_start_padded
            all_indices.append(local_offsets + base_index)

        return torch.stack(all_indices)

    # ------------------------------------------------------------------
    # Nonlinearity
    # ------------------------------------------------------------------

    def _nonlinearity(self):
        alpha = self.m_params["nonlinearity"]["alpha"]
        beta  = self.m_params["nonlinearity"]["beta"]
        gamma = self.m_params["nonlinearity"]["gamma"]

        match self.m_params["nonlinearity"]["type"]:
            case "soft-rectifier":
                return lambda x: alpha * torch.log(1.0 + torch.exp(beta * (x - gamma)))
            case "sigmoid":
                return lambda x: alpha / (1.0 + torch.exp(-beta * (x - gamma)))
            case "relu":
                return lambda x: alpha * torch.clamp(beta * (x - gamma), min=0.0)
            case "exp":
                return lambda x: alpha * torch.exp(beta * (x - gamma))
            case "ppc":
                return lambda x: x ** alpha / (beta * x + 1)
            case _:
                return lambda x: x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Args:
            x: (B, T, H, W)
        Returns:
            linear:      (N_cells, B)
            firing_rate: (N_cells, B)
        """
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.to(self.w_conv.device).float()
        
        # Spatial padding
        x_padded = f.pad(x, (self.rf_diam, self.rf_diam, self.rf_diam, self.rf_diam), mode="constant", value=0)
        
        # Compute spatiotemporal filter response at every location
        response_map = f.conv2d(x_padded, self.w_conv, padding=self.rf_diam//2)
        # response_map: (B, T-15, H, W)
        
        # Sample mosaic locations
        response = response_map[:, :, self.cell_y, self.cell_x]  # (B, T-15, N_cells)
        
        # Apply 1D conv across temporal dimension (learnable sliding window)
        # Reshape to (B*N_cells, T-15, 1) for conv1d
        B, T_out, N_cells = response.shape
        response_seq = response.permute(0, 2, 1).reshape(B*N_cells, 1, T_out)
        
        # Conv1d with kernel_size=16 -> output length = T_out - 16 + 1
        temporal_response = self.temporal_conv(response_seq)  # (B*N_cells, 1, T_out-15)
        
        # Reshape back and aggregate across time
        temporal_response = temporal_response.squeeze(1).view(B, N_cells, -1)
        linear = temporal_response.mean(dim=-1).T  # (N_cells, B)
        
        nonlinear = self.nonlinearity(linear)
        firing_rate = torch.clamp(nonlinear, max=float(self.m_params["max_firing_rate"]))
        
        return linear, firing_rate

    def spikes(self, firing_rate):
        dt           = 1.0 / self.v_params["frame_rate"]
        spike_counts = torch.poisson(firing_rate * dt).to(torch.uint16)
        counts_np    = spike_counts.cpu().numpy()

        all_spike_times = []
        for cell_idx in range(counts_np.shape[0]):
            counts     = counts_np[cell_idx]
            cell_times = []
            for bin_idx, k in enumerate(counts):
                if k > 0:
                    start_t, end_t = bin_idx * dt, (bin_idx + 1) * dt
                    cell_times.extend(np.random.uniform(start_t, end_t, int(k)))
            all_spike_times.append(np.sort(np.array(cell_times, dtype=np.float32)))

        return all_spike_times, spike_counts


class SmoothMonostratifiedCellMosaic(RetinalGanglionCellMosaic):
    """LN-LN encoding model of smooth monostratified (SM) ganglion cells.

    Based on Rhoades et al. 2019. Each subunit applies its own spatiotemporal
    filter and nonlinearity before summation into the output nonlinearity.
    """

    def __init__(
        self,
        model_parameters: dict,
        video_parameters: dict,
        cell_minibatch_size: int = None,
    ):
        self.n_subunits = model_parameters["n_subunits"]
        super().__init__(
            model_parameters,
            video_parameters
        )

        for i in range(self.n_subunits):
            self.register_buffer(f"w_sub_{i}", self.spatiotemporal_filter[i].reshape(-1))

        self.subunit_nonlinearity = self._subunit_nonlinearity()

    # ------------------------------------------------------------------
    # Filter construction (overrides)
    # ------------------------------------------------------------------

    def _spatial_filter(self):
        w           = self.m_params["spatial"]["width"]
        h           = self.m_params["spatial"]["height"]
        n_hotspots  = self.m_params["n_hotspots"]
        ring_radius = self.m_params["spatial"]["hotspot_ring_radius"]
        sigma       = self.m_params["spatial"]["hotspot_sigma"]
        jitter      = self.m_params["spatial"]["hotspot_jitter"]
        grouping    = self.m_params["hotspot_grouping"]

        torch.manual_seed(42)
        np.random.seed(43)

        def SM_gaussian_2d(w, h, sigma, ring_radius, jitter):
            theta = np.random.uniform(0, 2 * np.pi)
            cy    = (h / 2) + ring_radius * np.sin(theta) + np.random.normal(0, jitter)
            cx    = (w / 2) + ring_radius * np.cos(theta) + np.random.normal(0, jitter)
            y     = torch.arange(h, dtype=torch.float32)
            x     = torch.arange(w, dtype=torch.float32)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            return torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

        hotspots = torch.stack([SM_gaussian_2d(w, h, sigma, ring_radius, jitter) for _ in range(n_hotspots)])
        hotspots = hotspots / torch.sum(hotspots)

        subunit_filters = []
        for group_indices in grouping:
            group_sum = sum(hotspots[i] for i in group_indices)
            subunit_filters.append(group_sum / torch.sum(group_sum))

        return torch.stack(subunit_filters)

    def _temporal_filter(self, smooth: bool = True):
        filters = []
        for i in range(self.n_subunits):
            t_params = self.m_params["temporal"][f"subunit_{i}"]
            memory   = int(t_params.get("memory_frames", t_params.get("memory_ms", 16)))
            dt_ms    = 1000.0 / self.v_params["frame_rate"]
            t_ms     = torch.arange(memory, dtype=torch.float32) * dt_ms

            lobe1 = t_params["amp1"] * torch.exp(
                -((t_ms - t_params["peak1_ms"]) ** 2) / (2 * t_params["width1_ms"] ** 2)
            )
            lobe2 = t_params["amp2"] * torch.exp(
                -((t_ms - t_params["peak2_ms"]) ** 2) / (2 * t_params["width2_ms"] ** 2)
            )

            if smooth:
                onset_window = 1.0 - torch.exp(-t_ms / 15.0)
                lobe1 = lobe1 * onset_window
                lobe2 = lobe2 * onset_window

            filt = (-lobe1 + lobe2) if self.m_params["cell_type"].startswith("OFF") else (lobe1 - lobe2)
            filters.append(torch.flip(filt, dims=[0]))

        return torch.stack(filters)

    def _spatiotemporal_filter(self):
        filters = []
        for i in range(self.n_subunits):
            st = self.temporal_filter[i].view(-1, 1, 1) * self.spatial_filter[i].view(1, *self.spatial_filter[i].shape)
            filters.append(st / torch.norm(st))
        return torch.stack(filters)

    def _subunit_nonlinearity(self):
        alpha = self.m_params["subunit_nonlinearity"]["alpha"]
        beta  = self.m_params["subunit_nonlinearity"]["beta"]
        gamma = self.m_params["subunit_nonlinearity"]["gamma"]

        match self.m_params["subunit_nonlinearity"]["type"]:
            case "soft-rectifier":
                return lambda x: alpha * torch.log(1.0 + torch.exp(beta * (x - gamma)))
            case "sigmoid":
                return lambda x: alpha / (1.0 + torch.exp(-beta * (x - gamma)))
            case "relu":
                return lambda x: alpha * torch.clamp(beta * (x - gamma), min=0.0)
            case "exp":
                return lambda x: alpha * torch.exp(beta * (x - gamma))
            case _:
                return lambda x: x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """LN-LN forward pass for the SM cell mosaic.

        Args:
            x (torch.Tensor): Batched video input of shape (B, T, H, W).

        Returns:
            tuple: (linear_responses, firing_rates), each (N_cells, B).
        """
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.to(self.w_sub_0.device).float()

        B, T, H, W = x.shape
        rf_diam     = self.rf_diam

        x_padded = f.pad(x, (rf_diam, rf_diam, rf_diam, rf_diam), mode="constant", value=0)
        x_flat   = x_padded.reshape(B, T, -1)

        c_batch_size = self.cell_minibatch_size or self.n_cells

        subunit_responses = []
        for sub_idx in range(self.n_subunits):
            w_sub = getattr(self, f"w_sub_{sub_idx}")

            cell_chunks = []
            for c_start in range(0, self.n_cells, c_batch_size):
                c_end   = min(c_start + c_batch_size, self.n_cells)
                indices = self.rf_indices_tensor[c_start:c_end]

                # (B, T, C_chunk, RF_pixels)
                patches = x_flat[:, :, indices]

                # (B, C_chunk, T * RF_pixels)
                patches = patches.permute(0, 2, 1, 3).reshape(B, c_end - c_start, -1)

                # (C_chunk, B)
                cell_chunks.append((patches @ w_sub).T)

            subunit_responses.append(self.subunit_nonlinearity(torch.cat(cell_chunks, dim=0)))

        # Sum out-of-place: (N_cells, B)
        summed      = torch.stack(subunit_responses, dim=0).sum(dim=0)
        firing_rate = torch.clamp(self.nonlinearity(summed), max=float(self.m_params["max_firing_rate"]))

        return summed, firing_rate


class UnnamedCellMosaic(nn.Module):
    """Simple conv-style model to optimize for a novel cell type.

    Expects batched input of shape (B, T, H, W). Applies 2D convolutions
    over the spatial dimensions independently per timestep, then pools
    across time to produce a (N_cells, B) output consistent with the
    RGC mosaic interface.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=16, stride=16)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=2,  stride=2)
        self.relu  = nn.ReLU()

    @property
    def n_cells(self):
        return 1

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, T, H, W)

        Returns:
            tuple: (responses, responses), each (N_cells, B).
        """
        B, T, H, W = x.shape

        x_flat = x.reshape(B * T, 1, H, W)
        out    = self.relu(self.conv1(x_flat))
        out    = self.relu(self.conv2(out))
        out    = out.reshape(B, T, -1).mean(dim=1).T  # (N_cells, B)

        return out, out