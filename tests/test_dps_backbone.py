"""Tests for the vendored DPS (ADM / guided-diffusion) backbone.

The published checkpoints are multi-gigabyte downloads, so these tests never
touch them. They instead pin the things that silently corrupt a reconstruction
if they drift: the learn_sigma channel split, the state_dict layout the
checkpoints are keyed on, scheduler parameterisation, and the fact that the
likelihood gradient reaches x_t through the network.
"""

from __future__ import annotations

import pytest
import torch

from diffusion.dps_backbone import (
    MODEL_PRESETS,
    DPSUNet,
    build_dps_scheduler,
    build_unet,
    epsilon_from_model_output,
    load_dps_unet,
)

# A deliberately small ADM network. GroupNorm32 fixes 32 groups, so every
# channel width must stay divisible by 32.
TINY = {
    "image_size": 64,
    "num_channels": 32,
    "num_res_blocks": 1,
    "channel_mult": "",
    "learn_sigma": True,
    "class_cond": False,
    "use_checkpoint": False,
    "attention_resolutions": "16",
    "num_heads": 1,
    "num_head_channels": 32,
    "num_heads_upsample": -1,
    "use_scale_shift_norm": True,
    "dropout": 0.0,
    "resblock_updown": True,
    "use_fp16": False,
    "use_new_attention_order": False,
}


# ------------------------------------------------------------------ epsilon split


def test_epsilon_split_takes_leading_half():
    """learn_sigma output is [epsilon, variance]; only epsilon may be used."""
    sample = torch.zeros(2, 3, 8, 8)
    epsilon = torch.randn(2, 3, 8, 8)
    variance = torch.randn(2, 3, 8, 8)
    model_output = torch.cat([epsilon, variance], dim=1)

    recovered = epsilon_from_model_output(model_output, sample)

    assert recovered.shape == epsilon.shape
    torch.testing.assert_close(recovered, epsilon)


def test_epsilon_split_passes_through_three_channel_output():
    """The diffusers backbone emits epsilon alone, which must be untouched."""
    sample = torch.zeros(1, 3, 8, 8)
    model_output = torch.randn(1, 3, 8, 8)
    torch.testing.assert_close(
        epsilon_from_model_output(model_output, sample), model_output
    )


def test_epsilon_split_rejects_unexpected_width():
    sample = torch.zeros(1, 3, 8, 8)
    with pytest.raises(ValueError, match="5-channel"):
        epsilon_from_model_output(torch.randn(1, 5, 8, 8), sample)


# ------------------------------------------------------------ architecture layout


@pytest.mark.parametrize(
    "preset, expected_params_millions",
    [("imagenet256", 553.0), ("ffhq256", 93.6)],
)
def test_preset_parameter_counts(preset, expected_params_millions):
    """Guards the published checkpoints' shapes.

    ADM ImageNet-256 unconditional is a ~553M parameter network. A transcription
    error in channel_mult, num_res_blocks or the attention schedule moves this
    number, and would otherwise only surface as a failed strict load once the
    multi-gigabyte checkpoint is in hand.
    """
    model, config = build_unet(preset)
    millions = sum(p.numel() for p in model.parameters()) / 1e6
    assert millions == pytest.approx(expected_params_millions, rel=0.01)
    assert config.out_channels == 6, "both presets set learn_sigma"
    assert config.sample_size == 256


def test_state_dict_keys_match_adm_naming():
    """Checkpoint keys are derived from attribute names; pin the top-level ones."""
    model, _ = build_unet(TINY)
    keys = list(model.state_dict())

    assert keys[0] == "time_embed.0.weight"
    assert keys[-1] == "out.2.bias"
    prefixes = {key.split(".")[0] for key in keys}
    assert prefixes == {"time_embed", "input_blocks", "middle_block", "output_blocks", "out"}
    # Unconditional models must not carry a class-embedding table.
    assert not any(key.startswith("label_emb") for key in keys)


def test_resblock_internal_naming():
    """ResBlock sub-module names are part of every checkpoint key."""
    model, _ = build_unet(TINY)
    keys = set(model.state_dict())
    assert any(".in_layers." in key for key in keys)
    assert any(".emb_layers." in key for key in keys)
    assert any(".out_layers." in key for key in keys)
    assert any(".skip_connection." in key for key in keys)
    assert any(".qkv." in key for key in keys)
    assert any(".proj_out." in key for key in keys)


def test_fp16_is_rejected_rather_than_silently_wrong():
    with pytest.raises(NotImplementedError, match="use_fp16"):
        build_unet({**TINY, "use_fp16": True})


# ------------------------------------------------------------------- forward pass


def test_forward_emits_double_channels_and_keeps_spatial_shape():
    model, _ = build_unet(TINY)
    unet = DPSUNet(model, build_unet(TINY)[1]).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = unet(x, torch.tensor(7)).sample
    assert out.shape == (1, 6, 64, 64)


def test_scalar_timestep_is_broadcast_over_the_batch():
    """The sampler passes one scalar timestep for the whole batch."""
    model, config = build_unet(TINY)
    unet = DPSUNet(model, config).eval()
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        out = unet(x, torch.tensor(11)).sample
    assert out.shape[0] == 3


def _break_zero_init(model: torch.nn.Module) -> None:
    """Give every zero-initialised tensor a small random value.

    ADM applies ``zero_module`` to the output convolution, each ResBlock's
    output tail and each attention ``proj_out``. An untrained network therefore
    emits exactly zero and has a genuinely zero input-gradient, which would make
    a gradient-flow test pass vacuously (and fail for the wrong reason). Trained
    checkpoints have no such structure.
    """
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.abs().sum() == 0:
                parameter.normal_(0.0, 0.05)


def test_gradient_reaches_the_input():
    """Likelihood guidance needs dL/dx_t, so the graph must survive the U-Net.

    AttentionBlock always runs under gradient checkpointing, which is where a
    reentrant implementation would silently drop the input gradient. Breaking the
    zero-init first is what forces the signal through attention and the residual
    tails rather than through skip connections alone.
    """
    model, config = build_unet(TINY)
    _break_zero_init(model)
    unet = DPSUNet(model, config).eval()
    x = torch.randn(1, 3, 64, 64, requires_grad=True)

    epsilon = epsilon_from_model_output(unet(x, torch.tensor(3)).sample, x)
    (gradient,) = torch.autograd.grad(epsilon.square().mean(), x)

    assert gradient.shape == x.shape
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0, "no gradient reached x_t"


def test_untrained_network_emits_zero_by_construction():
    """Documents the zero_module behaviour the gradient test has to work around."""
    model, config = build_unet(TINY)
    unet = DPSUNet(model, config).eval()
    with torch.no_grad():
        out = unet(torch.randn(1, 3, 64, 64), torch.tensor(3)).sample
    assert out.abs().sum() == 0


# --------------------------------------------------------------------- scheduler


def test_scheduler_matches_dps_diffusion_config():
    """Mirrors configs/diffusion_config.yaml in the DPS repository."""
    scheduler = build_dps_scheduler()
    config = scheduler.config
    assert config.num_train_timesteps == 1000
    assert config.beta_schedule == "linear"
    assert config.beta_start == pytest.approx(0.0001)
    assert config.beta_end == pytest.approx(0.02)
    assert config.variance_type == "learned_range"
    assert config.prediction_type == "epsilon"
    assert config.clip_sample is True


def test_scheduler_consumes_the_six_channel_output():
    """step() must split the learned variance itself, not choke on it."""
    scheduler = build_dps_scheduler()
    scheduler.set_timesteps(10)
    sample = torch.randn(1, 3, 16, 16)
    model_output = torch.randn(1, 6, 16, 16)
    result = scheduler.step(model_output, scheduler.timesteps[0], sample)
    assert result.prev_sample.shape == sample.shape


# ------------------------------------------------------------------ strict load


def test_missing_checkpoint_names_the_download():
    with pytest.raises(FileNotFoundError, match="drive.google.com"):
        load_dps_unet("definitely/not/here.pt", "imagenet256")


def test_mismatched_checkpoint_raises_instead_of_random_init(tmp_path):
    """The reference create_model silently random-initialises here; we must not.

    A randomly initialised prior still produces images, so this failure mode is
    invisible in the output and would quietly invalidate a whole sweep.
    """
    model, _ = build_unet(TINY)
    path = tmp_path / "wrong_arch.pt"
    torch.save(model.state_dict(), path)

    with pytest.raises(RuntimeError, match="does not match"):
        load_dps_unet(path, "ffhq256")


def test_round_trip_load_succeeds_for_matching_architecture(tmp_path):
    model, _ = build_unet(TINY)
    path = tmp_path / "tiny.pt"
    torch.save(model.state_dict(), path)

    loaded = load_dps_unet(path, TINY)

    assert loaded.config.out_channels == 6
    assert not any(p.requires_grad for p in loaded.parameters())
    for (name, a), (_, b) in zip(
        model.state_dict().items(), loaded.model.state_dict().items()
    ):
        torch.testing.assert_close(a, b, msg=f"weight {name} changed on load")


def test_presets_cover_both_published_checkpoints():
    assert set(MODEL_PRESETS) == {"imagenet256", "ffhq256"}
