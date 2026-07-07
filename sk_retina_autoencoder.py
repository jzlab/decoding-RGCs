import torch
import torch.nn as nn

from sk_decoder import RetinaDecoder
from sk_retina_encoders import (
    RetinalGanglionCellMosaic,
    SmoothMonostratifiedCellMosaic,
    UnnamedCellMosaic,
)


class RetinaAutoencoder(nn.Module):
    """Compose the retina encoder bank with the UNet-like decoder.

    The encoder side is built from the models defined in sk_retina_encoders.py,
    while the decoder side is the existing RetinaDecoder from sk_decoder.py.
    This keeps the model assembly in one place and makes it easy to add more
    encoding modules later without changing the decoder wiring.
    """

    def __init__(self, model_parameters: dict, video_parameters: dict, decoder_params: dict | None = None):
        super().__init__()

        self.model_parameters = dict(model_parameters)
        self.video_parameters = dict(video_parameters)

        # Build an explicit encoder bank from the centralized encoder definitions.
        # This keeps the encoder setup easy to extend later while leaving the
        # decoder wiring to the standard RGC mosaics only.
        self.encoder_modules = nn.ModuleList()
        for cell_type, params in self.model_parameters.items():
            if cell_type == "video_parameters":
                continue

            if "Smooth_Monostratified" in cell_type:
                self.encoder_modules.append( SmoothMonostratifiedCellMosaic(params, self.video_parameters) )
            elif "Unnamed" in cell_type:
                self.encoder_modules.append( UnnamedCellMosaic() )
            else:
                self.encoder_modules.append( RetinalGanglionCellMosaic(params, self.video_parameters))

        self.n_cells_per_mosaic = [module.n_cells for module in self.encoder_modules if not isinstance(module, UnnamedCellMosaic)]

        # Default decoder configuration uses the encoder bank's actual mosaic sizes.
        if decoder_params is None:
            decoder_params = {
                "n_cells_per_mosaic": self.n_cells_per_mosaic,
                "frame_shape": tuple(self.video_parameters.get("frame_shape", (128, 128))),
                "num_blocks": 3,
                "num_kernels": 64,
                "bias": False,
            }
        else:
            decoder_params = dict(decoder_params)
            decoder_params.setdefault("n_cells_per_mosaic", self.n_cells_per_mosaic)
            decoder_params.setdefault("frame_shape", tuple(self.video_parameters.get("frame_shape", (128, 128))))

        self.decoder = RetinaDecoder(decoder_params)

    @property
    def encoder(self):
        """Backwards-compatible alias for the encoder bank."""
        return self.encoder_modules

    def _prepare_decoder_inputs(self, firing_rates):
        """Convert encoder outputs to the decoder's expected format.

        The retina encoder returns tensors shaped as (n_cells, n_windows).
        The decoder expects a list of tensors shaped as (n_windows, n_cells),
        so we transpose each mosaic response before reconstruction.
        """
        prepared = []
        for response in firing_rates:
            if response.dim() == 1:
                response = response.unsqueeze(0)
            prepared.append(response.transpose(0, 1).contiguous())
        return prepared

    def forward(self, x, encoder_grad_types=None):
        """Run the full encoder-decoder path.

        Returns
        -------
        tuple
            (reconstruction, linear_responses, firing_rates)
        """
        linear_responses = []
        firing_rates = []
        allow_grad = {name.lower() for name in (encoder_grad_types or [])}

        for module in self.encoder_modules:
            module_input = x
            if isinstance(module, UnnamedCellMosaic):
                if module_input.dim() == 3:
                    module_input = module_input.unsqueeze(1)
                elif module_input.dim() == 4 and module_input.shape[1] != 1:
                    module_input = module_input[:, :1]

            cell_type = getattr(getattr(module, "m_params", None), "get", lambda *_: "")("cell_type", "")
            if not cell_type and hasattr(module, "__class__"):
                cell_type = module.__class__.__name__

            context = torch.enable_grad() if cell_type.lower() in allow_grad else torch.no_grad()
            with context:
                lin, rate = module(module_input)

            linear_responses.append(lin)
            firing_rates.append(rate)

        # Only feed the standard RGC mosaics into the decoder.
        decoder_inputs = self._prepare_decoder_inputs(
            [rate for module, rate in zip(self.encoder_modules, firing_rates)
             if not isinstance(module, UnnamedCellMosaic)]
        )
        reconstruction = self.decoder(decoder_inputs)

        return reconstruction, linear_responses, firing_rates


def build_retina_autoencoder(model_parameters: dict, video_parameters: dict, **kwargs) -> RetinaAutoencoder:
    """Convenience constructor for the autoencoder shell."""
    return RetinaAutoencoder(model_parameters, video_parameters, **kwargs)
