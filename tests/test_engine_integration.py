"""SDTurboEngine against the real diffusers code with tiny random models.

No weights are downloaded: the UNet / VAE are randomly initialised and a few
MB in size, so the output is noise. What this checks is the plumbing: FP32 VAE
encode -> latents into the img2img pipeline with prompt_embeds and
guidance 0 -> step callback -> output_type="latent" -> FP32 decode, on the CPU
and (when torch-directml is installed) on the DirectML GPU.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from conftest import CPU  # noqa: E402
from core.device import DeviceInfo, select_device  # noqa: E402
from core.misread import plan_for  # noqa: E402
from core.pipeline import PromptEmbeds, Reconstructor, SDTurboEngine  # noqa: E402
from tiny_engine import tiny_components  # noqa: E402


def devices():
    found = [pytest.param(CPU, id="cpu")]
    dml = select_device("dml")
    if dml.kind == "dml":
        found.append(pytest.param(dml, id="directml"))
    return found


def build(device):
    engine = SDTurboEngine(device)
    engine.assemble(*tiny_components())
    return engine


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("w", [0.0, 0.5, 1.0])
def test_img2img_runs_planned_steps(device, w):
    plan = plan_for(w)
    engine = build(device)
    image = np.random.default_rng(0).integers(0, 255, (64, 96, 3), dtype=np.uint8)
    ticks = []
    out = engine.img2img(image, plan.num_inference_steps, plan.strength, 7, lambda: ticks.append(1))
    assert out.shape == image.shape and out.dtype == np.uint8
    assert len(ticks) == plan.denoise_steps + 2  # encode + steps + decode
    assert engine.vae.dtype == torch.float32


def test_same_seed_same_result():
    engine = build(CPU)
    image = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    a = engine.img2img(image, 4, 0.5, 3, lambda: None)
    b = engine.img2img(image, 4, 0.5, 3, lambda: None)
    assert np.array_equal(a, b)


def test_fp16_failure_falls_back_to_fp32(monkeypatch):
    engine = build(CPU)
    engine.pipe.unet.to(torch.float16)
    calls = []
    original = engine._img2img

    def flaky(*args):
        calls.append(engine.pipe.unet.dtype)
        if engine.pipe.unet.dtype == torch.float16:
            raise RuntimeError('"LayerNormKernelImpl" not implemented for \'Half\'')
        return original(*args)

    monkeypatch.setattr(engine, "_img2img", flaky)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    engine.img2img(image, 2, 0.5, 0, lambda: None)
    assert calls == [torch.float16, torch.float32]


def test_full_reconstruction_with_tiny_model(picture):
    def factory(device):
        engine = SDTurboEngine(device)
        engine.load = lambda status: engine.assemble(*tiny_components())
        return engine

    rec = Reconstructor(engine_factory=factory, device_selector=lambda pref: CPU, cpu_selector=lambda: CPU,
                        sides=(128, 96, 64))
    progress = []
    result = rec.run(picture, 0.8, 5, on_progress=progress.append)
    assert max(result.size) == 128 and progress[-1] == 100


def test_prompt_embeds_disk_cache(tmp_path):
    cache = PromptEmbeds(model_id="test/model", prompt="hello", cache_dir=tmp_path)
    tensor = torch.randn(1, 77, 8)
    cache.save(tensor)
    assert cache.path.exists()
    assert torch.equal(cache.get(), tensor)
    other = PromptEmbeds(model_id="test/model", prompt="another prompt", cache_dir=tmp_path)
    assert other.path != cache.path


def test_device_info_for_dml_is_usable():
    info = select_device("auto")
    assert isinstance(info, DeviceInfo)
    if info.kind == "dml":
        x = torch.ones(4).to(info.torch_device)
        assert float((x * 2).sum().cpu()) == 8.0
