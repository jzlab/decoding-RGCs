from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from sk_retina_encoders import Retina, SmoothMonostratifiedCellMosaic

# RF constraint families for ``LearnableSmoothMonostratifiedMosaic``.  Only
# ``nonnegative_sum1`` forbids opponency; it is kept to reproduce older runs.
KERNEL_PARAMETERIZATIONS = ("zero_mean_unit_norm", "center_surround", "nonnegative_sum1")


def _select_cell_types(model_params: dict, include_cell_types: Sequence[str] | None) -> list[str]:
    """Resolve the cell types included in the static diffusion encoder.

    Args:
        model_params (dict): Retina parameters loaded from YAML.
        include_cell_types (Sequence[str] | None): CLI-selected cell-type tags, if supplied.

    Returns:
        list[str]: Selected cell-type tags in their stable YAML order.
    """
    available = [
        value["cell_type"]
        for value in model_params.values()
        if isinstance(value, dict) and "cell_type" in value
    ]
    config_selection = model_params.get("diffusion", {}).get("include_cell_types")
    requested = include_cell_types if include_cell_types is not None else config_selection
    if requested is None:
        return available

    requested = list(requested)
    unknown = sorted(set(requested).difference(available))
    if unknown:
        raise ValueError(f"Unknown cell types: {unknown}. Available cell types: {available}.")
    return [cell_type for cell_type in available if cell_type in requested]


def _apply_nonlinearity(x: torch.Tensor, params: dict) -> torch.Tensor:
    """Apply an existing RGC nonlinearity to a response tensor.

    Args:
        x (torch.Tensor): Linear response tensor.
        params (dict): Nonlinearity configuration from the retina parameters.

    Returns:
        torch.Tensor: Nonlinear response tensor.
    """

    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    kind = params["type"]

    if kind == "soft-rectifier":
        # softplus is numerically stable where log(1 + exp(.)) is not.
        return alpha * F.softplus(beta * (x - gamma))
    if kind == "sigmoid":
        return alpha * torch.sigmoid(beta * (x - gamma))
    if kind == "relu":
        return alpha * torch.clamp(beta * (x - gamma), min=0.0)
    if kind == "exp":
        return alpha * torch.exp(beta * (x - gamma))
    if kind == "ppc":
        return x.pow(alpha) / (beta * x + 1)
    return x

def scale_sm_spatial_params(cell_params: dict, scale: float) -> dict:
    """Rescale a smooth-monostratified cell type's spatial geometry.

    Every length in the cell's configuration is expressed in pixels of a
    256-pixel frame.  Reconstructing at a smaller resolution without rescaling
    leaves the RF larger than the image and collapses the hexagonal mosaic to
    a handful of whole-frame cells, so this scales the learned type alone and
    leaves the configured fixed types untouched.

    Args:
        cell_params: Configuration for one smooth-monostratified cell type.
        scale: Multiplier applied to every spatial length, such as 0.25 for 64 px.

    Returns:
        Deep-copied configuration with rescaled spatial lengths.
    """
    if scale <= 0:
        raise ValueError(f"rf_scale must be positive, got {scale}.")
    if scale == 1.0:
        return copy.deepcopy(cell_params)

    scaled = copy.deepcopy(cell_params)
    spatial = scaled["spatial"]
    # Filter extent and tiling diameter must stay odd-free integers so the
    # meshgrid and index arithmetic in the source mosaic stay well formed.
    for key in ("width", "height"):
        spatial[key] = max(3, int(round(spatial[key] * scale)))
    for key in ("hotspot_ring_radius", "hotspot_sigma", "hotspot_jitter", "center_size"):
        if key in spatial:
            spatial[key] = float(spatial[key]) * scale
    tiling = scaled.get("tiling_config", {})
    if "rf_diameter" in tiling:
        tiling["rf_diameter"] = max(3, int(round(tiling["rf_diameter"] * scale)))
    return scaled


class StaticRGCMosaic(nn.Module):
    """Evaluate one RGC mosaic with convolution and hex-centre sampling.

    Args:
        source_mosaic (nn.Module): Existing retina mosaic that defines filters and centres.
    """

    def __init__(self, source_mosaic: nn.Module):
        super().__init__()
        self.params = source_mosaic.m_params
        self.max_firing_rate = float(self.params["max_firing_rate"])
        self.is_lnl = isinstance(source_mosaic, SmoothMonostratifiedCellMosaic)

        # ``spatial_filter`` is shared by cells in a mosaic.  Convolution
        # evaluates every possible RF placement; gather below retains only the
        # hexagonal placements produced by the original tiling code.
        spatial = source_mosaic.spatial_filter.detach().clone()
        if self.is_lnl:
            self.register_buffer("kernels", spatial.unsqueeze(1))
        else:
            self.register_buffer("kernels", spatial.unsqueeze(0).unsqueeze(1))

        self.register_buffer("centres", source_mosaic.mosaic_tensor.detach().clone())
        self.register_buffer("rf_shape", torch.tensor(spatial.shape[-2:]))
        if self.is_lnl:
            self.subunit_nonlinearity_params = self.params["subunit_nonlinearity"]

    def _gather_hex_centres(self, convolution_map: torch.Tensor) -> torch.Tensor:
        """Sample a convolution map at the hexagonal RGC centre coordinates.

        Args:
            convolution_map (torch.Tensor): Dense map shaped (batch, channels, H, W).

        Returns:
            torch.Tensor: Samples shaped (batch, channels, n_cells).
        """
        kernel_h, kernel_w = (int(v) for v in self.rf_shape.tolist())
        pad_h, pad_w = kernel_h, kernel_w
        centre_x, centre_y = self.centres.unbind(dim=1)
        y = torch.floor(centre_y - kernel_h / 2.0).long() + pad_h
        x = torch.floor(centre_x - kernel_w / 2.0).long() + pad_w
        height, width = convolution_map.shape[-2:]
        if (y < 0).any() or (x < 0).any() or (y >= height).any() or (x >= width).any():
            raise ValueError("RGC centres fall outside the convolution map.")
        flat_index = y * width + x
        flat = convolution_map.flatten(start_dim=2)
        return flat.index_select(2, flat_index)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a batch of grayscale images into firing rates.

        Args:
            image (torch.Tensor): Grayscale images shaped (batch, 1, height, width).

        Returns:
            torch.Tensor: Firing rates shaped (batch, n_cells).
        """
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("StaticRGCMosaic expects images shaped (batch, 1, H, W).")

        kernel_h, kernel_w = (int(v) for v in self.rf_shape.tolist())
        conv = F.conv2d(image, self.kernels, padding=(kernel_h, kernel_w))
        sampled = self._gather_hex_centres(conv)

        if self.is_lnl:
            # Static approximation of the LN-LN cell: temporal filtering is
            # intentionally omitted.  Add temporal dynamics here in a future
            # video-window version before applying this subunit nonlinearity.
            linear = _apply_nonlinearity(sampled, self.subunit_nonlinearity_params).sum(dim=1)
        else:
            linear = sampled.squeeze(1)
        rates = _apply_nonlinearity(linear, self.params["nonlinearity"])
        return rates.clamp(max=self.max_firing_rate)


class StaticRGCEncoder(nn.Module):
    """Encode grayscale images with a differentiable collection of static mosaics.

    Args:
        mosaics (Sequence[StaticRGCMosaic]): Static RGC mosaic modules.
        cell_type_names (Sequence[str]): Ordered cell-type tag for each mosaic.
    """

    def __init__(self, mosaics: Sequence[StaticRGCMosaic], cell_type_names: Sequence[str]):
        super().__init__()
        if len(mosaics) != len(cell_type_names):
            raise ValueError("Each static mosaic must have one corresponding cell-type name.")
        self.mosaics = nn.ModuleList(mosaics)
        self.cell_type_names = list(cell_type_names)

    @classmethod
    def from_params(cls, model_params: dict, frame_shape: tuple[int, int] | None = None, include_cell_types: Sequence[str] | None = None) -> "StaticRGCEncoder":
        """Build static mosaic geometry from existing retina parameters.

        Args:
            model_params (dict): Retina model parameters loaded from YAML.
            frame_shape (tuple[int, int] | None): Optional height and width for retiled mosaics.
            include_cell_types (Sequence[str] | None): Optional cell-type tags selected by the CLI.

        Returns:
            StaticRGCEncoder: Encoder containing one static module per mosaic.
        """
        
        selected_types = _select_cell_types(model_params, include_cell_types)
        source_params_by_type = {
            value["cell_type"]: value
            for value in model_params.values()
            if isinstance(value, dict) and "cell_type" in value
        }
        video_params = dict(model_params["video_parameters"])
        if frame_shape is not None:
            video_params["frame_shape"] = list(frame_shape)
        retina_params = {
            **{cell_type: source_params_by_type[cell_type] for cell_type in selected_types},
            "video_parameters": video_params,
        }
        # Retina construction supplies the existing DoG/LN-LN filters and
        # hexagonal tilings.  Its temporal filters are not used in this module.
        source = Retina(retina_params, video_params)
        return cls([StaticRGCMosaic(mosaic) for mosaic in source.mosaics], selected_types)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Encode grayscale images with every configured RGC mosaic.

        Args:
            image (torch.Tensor): Grayscale images shaped (batch, 1, height, width).

        Returns:
            list[torch.Tensor]: One firing-rate tensor shaped (batch, n_cells) per mosaic.
        """

        return [mosaic(image) for mosaic in self.mosaics]


class LearnableSmoothMonostratifiedMosaic(nn.Module):
    """A constrained, static LN-LN mosaic used as a task-optimized RGC type.

    This module deliberately models only spatial processing: training images do
    not identify temporal filters.  It shares a fixed hexagonal mosaic with a
    configured SM cell type, while its non-negative subunit RFs and gain/offset
    nonlinearities are trainable.
    """

    def __init__(self, source_mosaic: SmoothMonostratifiedCellMosaic, init_seed: int = 0, n_cells: int | None = None, kernel_parameterization: str = "zero_mean_unit_norm", subunit_nonlinearity: str | None = None, output_nonlinearity: str | None = None):
        """Initialize a random constrained SM mosaic using fixed source geometry.

        Args:
            source_mosaic: Configured SM mosaic that supplies RF shape and cell centres.
            init_seed: Random seed for the trainable RF initialization.
            n_cells: Optional number of leading hexagonal cells to retain.
            kernel_parameterization: RF constraint family; see ``KERNEL_PARAMETERIZATIONS``.
            subunit_nonlinearity: Optional override of the configured subunit response type.
            output_nonlinearity: Optional override of the configured cell output response type.

        Returns:
            None.
        """
        super().__init__()

        if kernel_parameterization not in KERNEL_PARAMETERIZATIONS:
            raise ValueError(
                f"kernel_parameterization must be one of {KERNEL_PARAMETERIZATIONS}, got {kernel_parameterization!r}."
            )
        self.kernel_parameterization = kernel_parameterization

        # Preserve the configured response conventions while randomizing RFs.
        self.params = copy.deepcopy(source_mosaic.m_params)
        self.max_firing_rate = float(self.params["max_firing_rate"])
        # ``exp`` subunits were chosen for the hand-parameterized hotspot RFs,
        # whose responses are bounded and positive.  Signed, contrast-sensitive
        # kernels make that choice saturate, so both stages are overridable.
        self.subunit_kind = subunit_nonlinearity or self.params["subunit_nonlinearity"]["type"]
        self.output_kind = output_nonlinearity or self.params["nonlinearity"]["type"]

        # Fix the mosaic geometry so optimization changes measurements, not layout.
        spatial_shape = source_mosaic.spatial_filter.shape
        self.n_subunits = spatial_shape[0]
        centres = source_mosaic.mosaic_tensor.detach().clone()
        if n_cells is not None:
            if n_cells < 1 or n_cells > len(centres):
                raise ValueError(f"--learned-n-cells must be in [1, {len(centres)}], got {n_cells}.")
            centres = centres[:n_cells]
        self.register_buffer("centres", centres)
        self.register_buffer("rf_shape", torch.tensor(spatial_shape[-2:]))

        # A local RF parameterization keeps this type a retinal filter rather
        # than an unrestricted image-to-code network.  The constraint family is
        # applied in ``kernels``; only the unconstrained tensors live here.
        generator = torch.Generator(device="cpu").manual_seed(init_seed)
        self.raw_kernels = nn.Parameter(
            0.05 * torch.randn((self.n_subunits, 1, *spatial_shape[-2:]), generator=generator)
        )
        if self.kernel_parameterization == "center_surround":
            # A second non-negative field and a bounded opponency weight give an
            # explicitly antagonistic RF without allowing arbitrary sign patterns.
            self.raw_surround_kernels = nn.Parameter(
                0.05 * torch.randn((self.n_subunits, 1, *spatial_shape[-2:]), generator=generator)
            )
            self.raw_surround_strength = nn.Parameter(torch.zeros(self.n_subunits))

        def log_positive(value: float) -> torch.Tensor:
            """Convert a positive scalar to an unconstrained log parameter.

            Args:
                value: Positive scalar used to initialize a gain parameter.

            Returns:
                Scalar tensor containing the log-transformed value.
            """
            return torch.tensor(max(float(value), 1e-4)).log()

        # Initialize trainable gains around the configured response scale.
        subunit = self.params["subunit_nonlinearity"]
        output = self.params["nonlinearity"]
        self.log_subunit_alpha = nn.Parameter(log_positive(subunit["alpha"]))
        self.log_subunit_beta = nn.Parameter(log_positive(subunit["beta"]))
        self.subunit_gamma = nn.Parameter(torch.tensor(float(subunit["gamma"])))
        self.log_output_alpha = nn.Parameter(log_positive(output["alpha"]))
        self.log_output_beta = nn.Parameter(log_positive(output["beta"]))
        self.output_gamma = nn.Parameter(torch.tensor(float(output["gamma"])))

    @classmethod
    def from_params(cls, model_params: dict, frame_shape: tuple[int, int], cell_type: str, init_seed: int = 0, n_cells: int | None = None, kernel_parameterization: str = "zero_mean_unit_norm", subunit_nonlinearity: str | None = None, output_nonlinearity: str | None = None, rf_scale: float = 1.0) -> "LearnableSmoothMonostratifiedMosaic":
        """Build a learnable SM mosaic with configured geometry but random RFs.

        Args:
            model_params: Full retina parameter dictionary.
            frame_shape: Height and width of images passed to the mosaic.
            cell_type: Configured smooth-monostratified cell-type tag.
            init_seed: Random seed for learned RF initialization.
            n_cells: Optional number of fixed hexagonal cells to retain.
            kernel_parameterization: RF constraint family; see ``KERNEL_PARAMETERIZATIONS``.
            subunit_nonlinearity: Optional override of the configured subunit response type.
            output_nonlinearity: Optional override of the configured cell output response type.
            rf_scale: Multiplier applied to this cell type's spatial lengths only.

        Returns:
            Randomly initialized learnable smooth-monostratified mosaic.
        """
        # Locate the requested configured cell type without mutating YAML values.
        params_by_type = {
            value["cell_type"]: copy.deepcopy(value)
            for value in model_params.values()
            if isinstance(value, dict) and "cell_type" in value
        }
        if cell_type not in params_by_type:
            raise ValueError(f"Unknown learned cell type {cell_type!r}.")
        cell_params = params_by_type[cell_type]
        if "Smooth_Monostratified" not in cell_type:
            raise NotImplementedError(
                "Only Smooth_Monostratified is implemented as a learnable type currently."
            )
        cell_params = scale_sm_spatial_params(cell_params, rf_scale)
        # Reuse the established tiling code only to construct fixed geometry.
        video_params = dict(model_params["video_parameters"])
        video_params["frame_shape"] = list(frame_shape)
        source = SmoothMonostratifiedCellMosaic(cell_params, video_params)
        return cls(
            source,
            init_seed=init_seed,
            n_cells=n_cells,
            kernel_parameterization=kernel_parameterization,
            subunit_nonlinearity=subunit_nonlinearity,
            output_nonlinearity=output_nonlinearity,
        )

    @staticmethod
    def _unit_norm(kernels: torch.Tensor) -> torch.Tensor:
        """Scale each subunit kernel to unit L2 norm.

        Args:
            kernels: Kernel tensor shaped ``(n_subunits, 1, RF_height, RF_width)``.

        Returns:
            Kernel tensor of the same shape with unit-norm subunits.
        """
        # Matches the norm convention the configured mosaics use in
        # ``sk_retina_encoders._spatiotemporal_filter``.
        norms = kernels.flatten(start_dim=1).norm(dim=1).clamp_min(1e-8)
        return kernels / norms.view(-1, 1, 1, 1)

    @property
    def kernels(self) -> torch.Tensor:
        """Return constrained trainable subunit spatial kernels.

        Args:
            None.

        Returns:
            Tensor shaped ``(n_subunits, 1, RF_height, RF_width)``.
        """
        if self.kernel_parameterization == "nonnegative_sum1":
            # Legacy family.  Non-negative weights summing to one make every
            # subunit a weighted pixel average, so the cell can only ever
            # report local mean luminance.  Retained to reproduce older runs.
            kernels = F.softplus(self.raw_kernels)
            return kernels / kernels.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

        if self.kernel_parameterization == "center_surround":
            # Non-negative centre and surround fields combined with a bounded
            # opponency weight, mirroring the configured DoG cell types.
            centre = F.softplus(self.raw_kernels)
            centre = centre / centre.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            surround = F.softplus(self.raw_surround_kernels)
            surround = surround / surround.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            opponency = torch.sigmoid(self.raw_surround_strength).view(-1, 1, 1, 1)
            return self._unit_norm(centre - opponency * surround)

        # Zero-mean, unit-norm: the smallest change that admits opponency.
        # Removing the DC component makes each subunit report local contrast
        # instead of local luminance, which is what the fixed DoG types do.
        kernels = self.raw_kernels - self.raw_kernels.mean(dim=(-2, -1), keepdim=True)
        return self._unit_norm(kernels)

    def _gather_hex_centres(self, convolution_map: torch.Tensor) -> torch.Tensor:
        """Sample dense convolution features at the fixed hexagonal cell centres.

        Args:
            convolution_map: Tensor shaped ``(B, n_subunits, H, W)``.

        Returns:
            Tensor shaped ``(B, n_subunits, n_cells)``.
        """
        # Convert original-image centres to coordinates in the padded map.
        kernel_h, kernel_w = (int(v) for v in self.rf_shape.tolist())
        pad_h, pad_w = kernel_h, kernel_w
        centre_x, centre_y = self.centres.unbind(dim=1)
        y = torch.floor(centre_y - kernel_h / 2.0).long() + pad_h
        x = torch.floor(centre_x - kernel_w / 2.0).long() + pad_w
        height, width = convolution_map.shape[-2:]
        if (y < 0).any() or (x < 0).any() or (y >= height).any() or (x >= width).any():
            raise ValueError("RGC centres fall outside the convolution map.")
        indices = y * width + x
        return convolution_map.flatten(start_dim=2).index_select(2, indices)

    @staticmethod
    def _nonlinearity(x: torch.Tensor, kind: str, log_alpha: torch.Tensor, log_beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Apply a trainable positive-gain nonlinearity.

        Args:
            x: Input response tensor.
            kind: Configured nonlinearity name.
            log_alpha: Unconstrained log output-gain parameter.
            log_beta: Unconstrained log input-gain parameter.
            gamma: Trainable response offset.

        Returns:
            Transformed response tensor with the same shape as ``x``.
        """
        alpha, beta = log_alpha.exp(), log_beta.exp()
        z = beta * (x - gamma)
        if kind == "soft-rectifier":
            return alpha * F.softplus(z)
        if kind == "sigmoid":
            return alpha * torch.sigmoid(z)
        if kind == "relu":
            return alpha * z.clamp_min(0)
        if kind == "exp":
            return alpha * torch.exp(z.clamp(max=20))
        if kind == "ppc":
            # PPC is not defined for negative values; smooth rectification
            # preserves finite gradients for random initializations.
            positive = F.softplus(x)
            return positive.pow(alpha) / (beta * positive + 1)
        return x

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode grayscale batches as task-optimized RGC firing rates.

        Args:
            image: Grayscale image tensor shaped ``(B, 1, H, W)``.

        Returns:
            Firing-rate tensor shaped ``(B, n_cells)``.
        """
        return self.activations(image)["firing_rates"]

    def activations(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute all static LN-LN response stages for a grayscale image batch.

        Args:
            image: Grayscale image tensor shaped ``(B, 1, H, W)``.

        Returns:
            Dictionary containing subunit linear and nonlinear responses,
            summed response, and final firing rates.
        """
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("LearnableSmoothMonostratifiedMosaic expects (B, 1, H, W) images.")

        # Filter each image with every learnable subunit and sample cells.
        kernel_h, kernel_w = (int(v) for v in self.rf_shape.tolist())
        convolved = F.conv2d(image, self.kernels, padding=(kernel_h, kernel_w))
        sampled = self._gather_hex_centres(convolved)
        subunit = self._nonlinearity(
            sampled, self.subunit_kind, self.log_subunit_alpha, self.log_subunit_beta, self.subunit_gamma
        )

        # Aggregate nonlinear subunits before applying the cell output response.
        linear = subunit.sum(dim=1)
        rates = self._nonlinearity(
            linear, self.output_kind, self.log_output_alpha, self.log_output_beta, self.output_gamma
        )
        return {
            "subunit_linear": sampled,
            "subunit_nonlinear": subunit,
            "summed_response": linear,
            "firing_rates": rates.clamp(max=self.max_firing_rate),
        }

    def regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RF smoothness and subunit-diversity penalties.

        Args:
            None.

        Returns:
            Smoothness and diversity scalar penalty tensors.
        """
        # Penalize spatially rough kernels and redundant subunit filters.
        kernels = self.kernels.squeeze(1)
        smoothness = (kernels[..., 1:, :] - kernels[..., :-1, :]).square().mean()
        smoothness = smoothness + (kernels[..., :, 1:] - kernels[..., :, :-1]).square().mean()
        flat = F.normalize(kernels.flatten(start_dim=1), dim=1)
        similarity = flat @ flat.T
        diversity = (similarity - torch.eye(self.n_subunits, device=kernels.device)).square().mean()
        return smoothness, diversity
