"""Tiny random SD components and engine builders (no weights are downloaded).

Shared by the integration tests; the builders are also imported by the engine
child process (`ProcessEngine(device, builder="tiny_engine:build")`).
"""

import torch


def tiny_components(cross_dim=32):
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=32, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 64), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=cross_dim,
        attention_head_dim=(2, 4), use_linear_projection=True, norm_num_groups=32,
    )
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(32, 64),
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"), norm_num_groups=32,
    )
    scheduler = EulerDiscreteScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012,
                                       steps_offset=1, timestep_spacing="trailing")
    embeds = torch.randn(1, 77, cross_dim)
    return unet, vae, scheduler, embeds


def build(device):
    """SDTurboEngine whose load() assembles the tiny model instead of SD-Turbo."""
    from core.pipeline import SDTurboEngine

    engine = SDTurboEngine(device)

    def load(status):
        status("tiny model loading")
        engine.assemble(*tiny_components())

    engine.load = load
    return engine


def build_oom(device):
    """Loads fine, then every img2img fails like DirectML out of memory."""
    engine = build(device)

    def img2img(*_args, **_kwargs):
        raise RuntimeError("Could not allocate tensor with 2221312 bytes. "
                           "There is not enough GPU video memory available!")

    engine.img2img = img2img
    return engine
