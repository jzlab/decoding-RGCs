import os
import sys
import time
import math
import random
import numpy as np

from tqdm import tqdm
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as f

from sk_utils import Utils as u
from skimage.transform import resize

class Retina(nn.Module):
    """
    stores and manages multiple RGC Mosaics
    """

    def __init__(self, model_parameters:dict, video_parameters:dict, cell_minibatch_size:int=None, temporal_batch_size:int=None):
        """
        stores and manages multiple RetinalGanglionCellMosaics
        """
        super().__init__()

        self.m_params = model_parameters
        self.v_params = video_parameters
        self.cell_minibatch_size = cell_minibatch_size
        self.temporal_batch_size = temporal_batch_size

        self.mosaics = nn.ModuleList()
        for cell_type, params in self.m_params.items():
            if cell_type != "video_parameters":
                if "Smooth_Monostratified" in cell_type:
                    self.mosaics.append(SmoothMonostratifiedCellMosaic(
                        params, 
                        self.v_params, 
                        cell_minibatch_size=self.cell_minibatch_size,
                        temporal_batch_size=self.temporal_batch_size
                    ))
                else:
                    self.mosaics.append(RetinalGanglionCellMosaic(
                        params, 
                        self.v_params, 
                        cell_minibatch_size=self.cell_minibatch_size,
                        temporal_batch_size=self.temporal_batch_size
                    ))

        # store the max number of cells to pad properly
        self.n_cells = max([m.n_cells for m in self.mosaics])

    def forward(self, x, pad=True):
        """perform a forward pass through each RGC mosaic

        Args:
            x (torch.Tensor): input video of shape (T, H, W)

        Returns:
            tuple: (linear_responses, firing_rates)
        """

        # pass through each mosaic
        res_linear = []
        res_rate = []
        for mosaic in self.mosaics:
            lin, rate = mosaic.forward(x)
            res_linear.append(lin)
            res_rate.append(rate)

        # pad the mosaics to the largest cell count
        # biologically, this is because different RGC mosaics have larger cell count
        # when they have smaller RFs, so we "create" dummy cells that are not activated
        # to make the total number of cells equal across mosaics to make vectorization easier
        def pad_responses(tensors, target_n):
            padded = []
            for t in tensors:
                if t.shape[0] < target_n:
                    # pad the cell dimension (dim 0)
                    t = f.pad(t, (0, 0, 0, target_n - t.shape[0]), 'constant', 0)
                padded.append(t)
            return padded

        if pad:
            padded_linear = pad_responses(res_linear, self.n_cells)
            padded_rate = pad_responses(res_rate, self.n_cells)
            return torch.stack(padded_linear), torch.stack(padded_rate)
        else:
            return res_linear, res_rate

class RetinalGanglionCellMosaic(nn.Module):
    """
    Model representing a mosaic of Retinal Ganglion Cells (RGCs) that processes 
    video input through spatiotemporal filtering and non-linear activation.
    """

    def __init__(self, model_parameters: dict, video_parameters: dict, cell_minibatch_size: int = None, temporal_batch_size: int = None):
        """
        Initialize the RGC Mosaic model.

        Args:
            model_parameters (dict): Parameters for spatial, temporal, tiling and nonlinearity configuration.
            video_parameters (dict): Parameters for video properties like frame rate and shape.
            cell_minibatch_size (int, optional): Number of cells to process in a single vectorized batch.
            temporal_batch_size (int, optional): Number of temporal windows to process in a single vectorized batch.
        """
        super().__init__()
        
        # store the params
        self.m_params = model_parameters
        self.v_params = video_parameters
        self.cell_minibatch_size = cell_minibatch_size
        self.temporal_batch_size = temporal_batch_size

        # compute the spatiotemporal filter and register components
        self.spatial_filter = self._spatial_filter()
        self.temporal_filter = self._temporal_filter()
        self.spatiotemporal_filter = self._spatiotemporal_filter()
        
        # Store filters as buffers for device management
        self.register_buffer('w', self.spatiotemporal_filter.reshape(-1))
        self.register_buffer('temporal_filter_tensor', self.temporal_filter)

        # stitch the filters to create the RGC mosaic
        self.mosaic = self._tile_cells()
        self.register_buffer('mosaic_tensor', torch.as_tensor(self.mosaic, dtype=torch.float32))
        self.n_cells = len(self.mosaic)

        # Pre-compute spatial indices for vectorized gathering
        self.rf_diam = self.m_params["tiling_config"]["rf_diameter"]
        self.rf_indices = self._calculate_vectorized_indices()
        self.register_buffer('rf_indices_tensor', self.rf_indices)

        # store the nonlinearity
        self.nonlinearity = self._nonlinearity()
        
    def _calculate_vectorized_indices(self):
        """
        Pre-compute spatial pixel indices for each cell's receptive field.
        These indices allow for the extraction of all cell patches in a single
        vectorized gathering operation.

        Returns:
            torch.Tensor: Tensor of indices (N_cells, rf_diam * rf_diam).
        """
        height, width = self.v_params["frame_shape"]
        rf_diam = self.rf_diam
        half_diam = rf_diam / 2.0
        
        # We index into a padded frame to handle boundary cells safely.
        # Padding size equals rf_diam to provide a 'dark' zone for all possible shifts.
        padded_width = width + 2 * rf_diam
        
        # Generate local patch coordinates (offsets from center)
        y_range = torch.arange(rf_diam)
        x_range = torch.arange(rf_diam)
        yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
        local_offsets = yy * padded_width + xx

        all_indices = []
        for pos in self.mosaic_tensor:
            pos_x, pos_y = pos[0].item(), pos[1].item()
            
            # Compute top-left corner of the RF in the PADDED frame.
            # pos in original: (x, y). In padded: (x + rf_diam, y + rf_diam)
            # v_start_raw in original: math.floor(pos_y - half_diam)
            # v_start_padded: math.floor(pos_y - half_diam) + rf_diam
            v_start_padded = math.floor(pos_y - half_diam) + rf_diam
            h_start_padded = math.floor(pos_x - half_diam) + rf_diam
            
            # Generate flat indices for this cell's patch in the padded frame
            base_index = v_start_padded * padded_width + h_start_padded
            cell_indices = local_offsets.flatten() + base_index
            all_indices.append(cell_indices)

        return torch.stack(all_indices)

    def _spatial_filter(self):
        """
        Compute the spatial receptive field using a Difference of Gaussians (DoG).

        Returns:
            torch.Tensor: The 2D spatial filter weights.
        """
        
        def gaussian_2d(w, h, sigma):
            # generate coords
            cy, cx = h / 2.0, w / 2.0
            y = torch.arange(h, dtype=torch.float32)
            x = torch.arange(w, dtype=torch.float32)
            yy, xx = torch.meshgrid(y, x, indexing='ij')

            # 2D gaussian and return
            return torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))

        # create the 2 gaussians and normalize
        w, h = self.m_params["spatial"]["width"], self.m_params["spatial"]["height"]
        center = gaussian_2d(w, h, self.m_params["spatial"]["center_size"])
        surround = gaussian_2d(w, h, self.m_params["spatial"]["surround_size"])

        # normalize and combine
        center = center / torch.sum(center)
        surround = surround / torch.sum(surround)
        return self.m_params["spatial"]["center_strength"] * center + self.m_params["spatial"]["surround_strength"] * surround

    def _temporal_filter(self, smooth: bool = True):
        """
        Generate the biphasic temporal filter for the model.

        Args:
            smooth (bool): Whether to apply a smooth onset window (Exponential). Defaults to True.

        Returns:
            torch.Tensor: The 1D temporal filter weights.
        """

        memory = int(self.m_params["temporal"]["memory_ms"] * (self.v_params["frame_rate"] / 1000))
        dt_ms = 1000. / self.v_params["frame_rate"]
        t_ms = torch.arange(memory, dtype=torch.float32) * dt_ms

        # create the lobes
        lobe1 = self.m_params["temporal"]["amp1"] * torch.exp(-((t_ms - self.m_params["temporal"]["peak1_ms"])**2) / (2 * self.m_params["temporal"]["width1_ms"]**2))
        lobe2 = self.m_params["temporal"]["amp2"] * torch.exp(-((t_ms - self.m_params["temporal"]["peak2_ms"])**2) / (2 * self.m_params["temporal"]["width2_ms"]**2))

        # smooth the filter
        if smooth:
            onset_tau_ms = 15.0
            onset_window = 1.0 - torch.exp(-t_ms / onset_tau_ms)
            lobe1 = lobe1 * onset_window
            lobe2 = lobe2 * onset_window

        # biphasic based on ON/OFF + cell type
        if self.m_params["cell_type"].startswith("OFF"):
            filter = -lobe1 + lobe2
        else:
            filter = lobe1 - lobe2

        return torch.flip(filter, dims=[0])

    def _spatiotemporal_filter(self):
        """
        Combine spatial and temporal filters into a 3D spatiotemporal filter.

        Returns:
            torch.Tensor: The 3D filter (frames, height, width).
        """

        # weight each spatial filter by temporal value and normalize
        spatiotemporal = self.temporal_filter.view(-1, 1, 1) * self.spatial_filter.view(1, *self.spatial_filter.shape)
        return spatiotemporal / torch.norm(spatiotemporal)

    def _tile_cells(self, max_cells=None):
        """
        Create hexagonal mosaic of receptive field positions using lattice basis formulation.

        Args:
            max_cells (int, optional): Maximum number of cells to include. Defaults to None.

        Returns:
            torch.Tensor: Tensor of (x, y) coordinates for the mosaic.
        """

        height, width = self.v_params["frame_shape"]

        # Adjust spacing
        s = self.m_params["spatial"]["center_size"] / self.m_params["tiling_config"]["coverage_factor"]

        # Hexagonal lattice basis
        B = torch.tensor([
            [s, s / 2.0],
            [0.0, s * torch.sqrt(torch.tensor(3.0)) / 2.0]
        ], dtype=torch.float32)
        B_inv = torch.inverse(B)

        # Determine margin constraints
        margin = self.m_params["spatial"]["center_size"] / 2.0

        # Define rectangle corners in Cartesian space
        corners = torch.tensor([
            [0, 0],
            [width, 0],
            [0, height],
            [width, height]
        ], dtype=torch.float32)

        # Map corners into lattice coordinates
        lattice_corners = corners @ B_inv.T

        # Compute lattice bounds
        n_min = int(torch.floor(lattice_corners[:, 0].min())) - 1
        n_max = int(torch.ceil(lattice_corners[:, 0].max())) + 1
        m_min = int(torch.floor(lattice_corners[:, 1].min())) - 1
        m_max = int(torch.ceil(lattice_corners[:, 1].max())) + 1

        all_positions = []

        # Sample lattice and map forward
        for m in range(m_min, m_max + 1):
            for n in range(n_min, n_max + 1):
                pos = torch.tensor([float(n), float(m)], dtype=torch.float32) @ B.T
                x, y = pos[0].item(), pos[1].item()

                if (margin <= x <= width - margin and
                    margin <= y <= height - margin):
                    all_positions.append((x, y))

        all_positions = torch.tensor(all_positions, dtype=torch.float32)

        # Selection strategies
        if max_cells is not None and len(all_positions) > max_cells:
            if self.m_params["tiling_config"]["selection_method"] in ['center_first', 'edge_first']:
                center_pt = torch.tensor([width / 2.0, height / 2.0], dtype=torch.float32)
                distances = torch.norm(all_positions - center_pt, dim=1)
                
                reverse = (self.m_params["tiling_config"]["selection_method"] == 'edge_first')
                values, sorted_idx = torch.sort(distances, descending=reverse)
                all_positions = all_positions[sorted_idx[:max_cells]]
            elif self.m_params["tiling_config"]["selection_method"] == 'random':
                torch.manual_seed(42)
                idx = torch.randperm(len(all_positions))[:max_cells]
                all_positions = all_positions[idx]
            elif self.m_params["tiling_config"]["selection_method"] == 'grid_order':
                all_positions = all_positions[:max_cells]

        # offset the cells if necessary
        offset = torch.tensor(self.m_params["tiling_config"]["offset"], dtype=torch.float32)
        return all_positions + offset

    def _nonlinearity(self):
        """
        Define the nonlinearity to use for the model.

        Returns:
            func: A lambda function that applies the activation.
        """

        alpha = self.m_params["nonlinearity"]["alpha"]
        beta = self.m_params["nonlinearity"]["beta"]
        gamma = self.m_params["nonlinearity"]["gamma"]

        match self.m_params["nonlinearity"]["type"]:
            case "soft-rectifier":
                return lambda x: alpha * torch.log(1.0 + torch.exp(beta * (x - gamma)))
            case "sigmoid":
                return lambda x: alpha / (1.0 + torch.exp(-beta * (x - gamma)))
            case "relu":
                return lambda x: alpha * torch.clamp(beta * (x - gamma), min=0.0)
            case _:
                return lambda x: x

    def forward(self, x):
        """
        Forward pass for the RGC Mosaic using chunked vectorization.

        Args:
            x (np.ndarray or torch.Tensor): The input video stimulus (T, H, W).

        Returns:
            torch.Tensor: The output firing rate for each RGC (N_cells, T_windows).
        """
        
        # Ensure input is a tensor on the correct device
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.to(self.w.device).float()

        # Reshape input to (N_windows, Win_size, H, W)
        win_size = len(self.temporal_filter_tensor)
        x_win = x.unfold(0, win_size, 1).permute(0, 3, 1, 2)
        
        n_windows, _, H, W = x_win.shape
        rf_diam = self.rf_diam
        
        # Pad the entire windowed video sequence once to handle boundary RFs.
        # Padding with zero (representing darkness) matches the original logic.
        # F.pad order: (left, right, top, bottom)
        x_win_padded = f.pad(x_win, (rf_diam, rf_diam, rf_diam, rf_diam), mode='constant', value=0)
        
        # Flatten the spatial dimensions of the padded frames for gathering
        # Shape: (N_windows, Win_size, Padded_Pixels)
        x_win_flat = x_win_padded.reshape(n_windows, win_size, -1)
        
        # Determine minibatch sizes (using defaults if not provided)
        t_batch_size = self.temporal_batch_size if self.temporal_batch_size else n_windows
        c_batch_size = self.cell_minibatch_size if self.cell_minibatch_size else self.n_cells

        linear = torch.zeros((self.n_cells, n_windows), device=x.device, dtype=torch.float32)

        # Vectorized hierarchical batching
        for t_start in range(0, n_windows, t_batch_size):
            t_end = min(t_start + t_batch_size, n_windows)
            
            # Select frame chunk: (T_chunk, Win_size, Padded_Pixels)
            x_chunk = x_win_flat[t_start:t_end]
            
            for c_start in range(0, self.n_cells, c_batch_size):
                c_end = min(c_start + c_batch_size, self.n_cells)
                
                # Extract indices for this minibatch: (C_chunk, RF_Pixels)
                indices = self.rf_indices_tensor[c_start:c_end]
                
                # Shape: (T_chunk, Win_size, C_chunk, RF_Pixels)
                patches = x_chunk[:, :, indices] 
                
                # Reshape to (T_chunk, C_chunk, Win_size * RF_Pixels)
                patches = patches.permute(0, 2, 1, 3).reshape(t_end - t_start, c_end - c_start, -1)
                
                # Matrix multiplication with shared filter w: (T_chunk, C_chunk)
                linear[c_start:c_end, t_start:t_end] = (patches @ self.w).T

        # Apply activation and cap firing rate
        nonlinear = self.nonlinearity(linear)
        firing_rate = torch.clamp(nonlinear, max=float(self.m_params["max_firing_rate"]))

        return linear, firing_rate

    def spikes(self, firing_rate):
        """
        Generate spikes and spike times from the firing rate.

        Args:
            firing_rate (torch.Tensor): The computed firing rate (N_cells, T).

        Returns:
            tuple: (spike_times, spike_counts)
        """

        dt = 1.0 / self.v_params["frame_rate"]
        
        # Rate-based Poisson count: (N_cells, T)
        spike_counts = torch.poisson(firing_rate * dt).to(torch.uint16)
        
        # For spike times, we follow the original logic (list of arrays)
        # This is kept as NumPy/CPU for compatibility with list-based irregular data
        counts_np = spike_counts.cpu().numpy()
        all_spike_times = []
        
        for cell_idx in range(counts_np.shape[0]):
            counts = counts_np[cell_idx]
            cell_times = []
            for bin_idx, k in enumerate(counts):
                if k > 0:
                    start_t, end_t = bin_idx * dt, (bin_idx + 1) * dt
                    times = np.random.uniform(start_t, end_t, int(k))
                    cell_times.extend(times)
            all_spike_times.append(np.sort(np.array(cell_times, dtype=np.float32)))

        return all_spike_times, spike_counts

class SmoothMonostratifiedCellMosaic(RetinalGanglionCellMosaic):
    """LN-LN encoding model of smooth monostratified (SM) ganglion cells.
    
    Based on Rhoades et al. 2019, "Unusual Physiological Properties of 
    Smooth Monostratified Ganglion Cell Types in Primate Retina."

    SM cells differ from standard parasol/midget RGCs in that their receptive
    fields contain multiple spatially-segregated nonlinear subunits ("hotspots").
    Each subunit applies its own spatiotemporal filter and nonlinearity before
    summation, producing an LN-LN architecture:

        stimulus → [subunit_i: spatial → temporal → nonlinearity] * N → sum → output nonlinearity → rate
    """

    def __init__(self, model_parameters: dict, video_parameters: dict, 
                 cell_minibatch_size: int = None, temporal_batch_size: int = None):
        """
        Initialize the SM Cell Mosaic model.

        Args:
            model_parameters (dict): Parameters including hotspot spatial config,
                per-subunit temporal filters, tiling, and dual nonlinearities.
            video_parameters (dict): Video properties (frame_rate, frame_shape).
            cell_minibatch_size (int, optional): Cells per vectorized batch.
            temporal_batch_size (int, optional): Temporal windows per vectorized batch.
        """

        self.n_subunits = model_parameters["n_subunits"]

        # Initialize the parent — this calls _spatial_filter(), _temporal_filter(),
        # _spatiotemporal_filter(), _tile_cells(), _calculate_vectorized_indices(),
        # and _nonlinearity() via super().__init__().
        # We override the filter methods below so the parent builds our subunit filters.
        super().__init__(
            model_parameters, video_parameters,
            cell_minibatch_size=cell_minibatch_size,
            temporal_batch_size=temporal_batch_size
        )

        # The parent registered self.w as the entire spatiotemporal_filter flattened
        # into a single vector, which is incorrect for the subunit model. We need
        # separate per-subunit weight buffers for the LN-LN forward pass.
        del self.w
        for i in range(self.n_subunits):
            self.register_buffer(
                f"w_sub_{i}", 
                self.spatiotemporal_filter[i].reshape(-1)
            )

        # Build the subunit nonlinearity (distinct from the output nonlinearity
        # which was built by the parent's _nonlinearity() call)
        self.subunit_nonlinearity = self._subunit_nonlinearity()

    def _spatial_filter(self):
        """
        Compute the multi-hotspot spatial receptive field for SM cells.
        
        Generates n_hotspots Gaussian hotspots arranged on a noisy ring,
        then consolidates them into n_subunits groups. Uses fixed seeds
        (torch.manual_seed(42), np.random.seed(43)) for deterministic,
        identical RFs across all cells in the mosaic.

        Returns:
            torch.Tensor: Subunit spatial filters of shape (n_subunits, H, W).
        """

        w = self.m_params["spatial"]["width"]
        h = self.m_params["spatial"]["height"]
        n_hotspots = self.m_params["n_hotspots"]
        ring_radius = self.m_params["spatial"]["hotspot_ring_radius"]
        sigma = self.m_params["spatial"]["hotspot_sigma"]
        jitter = self.m_params["spatial"]["hotspot_jitter"]
        grouping = self.m_params["hotspot_grouping"]

        # Fix seeds for deterministic, identical hotspot patterns
        torch.manual_seed(42)
        np.random.seed(43)

        def SM_gaussian_2d(w, h, sigma, ring_radius, jitter):
            """Generate a single Gaussian hotspot at a noisy position on a ring."""
            theta_random = np.random.uniform(0, 2 * np.pi, 1)
            cy = (h / 2) + (ring_radius * np.sin(theta_random)) + np.random.normal(0, jitter, 1)
            cx = (w / 2) + (ring_radius * np.cos(theta_random)) + np.random.normal(0, jitter, 1)
            y = torch.arange(h, dtype=torch.float32)
            x = torch.arange(w, dtype=torch.float32)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            return torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))

        # Generate all hotspots
        hotspots = torch.zeros(size=(n_hotspots, h, w))
        for i in range(n_hotspots):
            hotspots[i] = SM_gaussian_2d(w, h, sigma, ring_radius, jitter)

        # Normalize all hotspots together (as in the user's prototype)
        hotspots = hotspots / torch.sum(hotspots)

        # Consolidate hotspots into subunit groups and normalize each group
        subunit_filters = []
        for group_indices in grouping:
            group_sum = sum(hotspots[i] for i in group_indices)
            subunit_filters.append(group_sum / torch.sum(group_sum))

        return torch.stack(subunit_filters)

    def _temporal_filter(self, smooth: bool = True):
        """
        Generate per-subunit biphasic temporal filters.
        
        Each subunit has its own temporal dynamics read from
        m_params["temporal"]["subunit_i"], producing distinct
        spatiotemporal signatures as observed by Rhoades et al.

        Args:
            smooth (bool): Whether to apply smooth onset window. Defaults to True.

        Returns:
            torch.Tensor: Per-subunit temporal filters of shape (n_subunits, T_memory).
        """

        temporal_filters = []
        for i in range(self.n_subunits):
            t_params = self.m_params["temporal"][f"subunit_{i}"]

            memory = int(t_params["memory_ms"] * (self.v_params["frame_rate"] / 1000))
            dt_ms = 1000. / self.v_params["frame_rate"]
            t_ms = torch.arange(memory, dtype=torch.float32) * dt_ms

            # Biphasic lobes
            lobe1 = t_params["amp1"] * torch.exp(
                -((t_ms - t_params["peak1_ms"])**2) / (2 * t_params["width1_ms"]**2)
            )
            lobe2 = t_params["amp2"] * torch.exp(
                -((t_ms - t_params["peak2_ms"])**2) / (2 * t_params["width2_ms"]**2)
            )

            # Smooth onset window
            if smooth:
                onset_tau_ms = 15.0
                onset_window = 1.0 - torch.exp(-t_ms / onset_tau_ms)
                lobe1 = lobe1 * onset_window
                lobe2 = lobe2 * onset_window

            # ON/OFF polarity
            if self.m_params["cell_type"].startswith("OFF"):
                filt = -lobe1 + lobe2
            else:
                filt = lobe1 - lobe2

            temporal_filters.append(torch.flip(filt, dims=[0]))

        return torch.stack(temporal_filters)

    def _spatiotemporal_filter(self):
        """
        Combine per-subunit spatial and temporal filters into per-subunit
        3D spatiotemporal filters.

        Returns:
            torch.Tensor: Shape (n_subunits, T_memory, H, W).
        """

        # self.spatial_filter: (n_subunits, H, W)
        # self.temporal_filter: (n_subunits, T_memory)
        filters = []
        for i in range(self.n_subunits):
            st = self.temporal_filter[i].view(-1, 1, 1) * self.spatial_filter[i].view(1, *self.spatial_filter[i].shape)
            st = st / torch.norm(st)
            filters.append(st)

        return torch.stack(filters)

    def _subunit_nonlinearity(self):
        """
        Define the nonlinearity applied at each subunit before summation.
        Read from m_params["subunit_nonlinearity"].

        Returns:
            func: A lambda function that applies the subunit activation.
        """

        alpha = self.m_params["subunit_nonlinearity"]["alpha"]
        beta = self.m_params["subunit_nonlinearity"]["beta"]
        gamma = self.m_params["subunit_nonlinearity"]["gamma"]

        match self.m_params["subunit_nonlinearity"]["type"]:
            case "soft-rectifier":
                return lambda x: alpha * torch.log(1.0 + torch.exp(beta * (x - gamma)))
            case "sigmoid":
                return lambda x: alpha / (1.0 + torch.exp(-beta * (x - gamma)))
            case "relu":
                return lambda x: alpha * torch.clamp(beta * (x - gamma), min=0.0)
            case _:
                return lambda x: x

    def forward(self, x):
        """
        LN-LN forward pass for the SM Cell Mosaic.

        For each subunit:
            1. Extract RF patches from the stimulus
            2. Apply the subunit's spatiotemporal filter (linear)
            3. Apply the subunit nonlinearity
        Then sum across subunits, apply output nonlinearity, and clamp.

        Args:
            x (np.ndarray or torch.Tensor): Input video stimulus (T, H, W).

        Returns:
            tuple: (linear_responses, firing_rates), each of shape (N_cells, T_windows).
        """

        # Ensure input is a tensor on the correct device
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        # Use the first subunit's weight buffer to determine device
        x = x.to(self.w_sub_0.device).float()

        # All subunits share the same temporal window size (same memory_ms)
        win_size = self.spatiotemporal_filter.shape[1]
        x_win = x.unfold(0, win_size, 1).permute(0, 3, 1, 2)

        n_windows, _, H, W = x_win.shape
        rf_diam = self.rf_diam

        # Pad frames for boundary handling
        x_win_padded = f.pad(x_win, (rf_diam, rf_diam, rf_diam, rf_diam), mode='constant', value=0)
        x_win_flat = x_win_padded.reshape(n_windows, win_size, -1)

        # Determine batch sizes
        t_batch_size = self.temporal_batch_size if self.temporal_batch_size else n_windows
        c_batch_size = self.cell_minibatch_size if self.cell_minibatch_size else self.n_cells

        # Accumulate the summed subunit responses
        summed_subunit_response = torch.zeros(
            (self.n_cells, n_windows), device=x.device, dtype=torch.float32
        )

        # Process each subunit independently through the LN-LN pipeline
        for sub_idx in range(self.n_subunits):
            # Get this subunit's flattened spatiotemporal filter
            w_sub = getattr(self, f"w_sub_{sub_idx}")

            # Compute linear response for this subunit across all cells
            subunit_linear = torch.zeros(
                (self.n_cells, n_windows), device=x.device, dtype=torch.float32
            )

            for t_start in range(0, n_windows, t_batch_size):
                t_end = min(t_start + t_batch_size, n_windows)
                x_chunk = x_win_flat[t_start:t_end]

                for c_start in range(0, self.n_cells, c_batch_size):
                    c_end = min(c_start + c_batch_size, self.n_cells)

                    indices = self.rf_indices_tensor[c_start:c_end]

                    # (T_chunk, Win_size, C_chunk, RF_Pixels)
                    patches = x_chunk[:, :, indices]

                    # (T_chunk, C_chunk, Win_size * RF_Pixels)
                    patches = patches.permute(0, 2, 1, 3).reshape(
                        t_end - t_start, c_end - c_start, -1
                    )

                    # Dot product with this subunit's filter
                    subunit_linear[c_start:c_end, t_start:t_end] = (patches @ w_sub).T

            # Apply subunit nonlinearity BEFORE summation (the key LN-LN step)
            summed_subunit_response += self.subunit_nonlinearity(subunit_linear)

        # Apply output nonlinearity and clamp firing rate
        firing_rate = self.nonlinearity(summed_subunit_response)
        firing_rate = torch.clamp(firing_rate, max=float(self.m_params["max_firing_rate"]))

        return summed_subunit_response, firing_rate