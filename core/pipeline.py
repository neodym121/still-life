"""Model loading and the reconstruction pipeline.

    prepare (448 px) -> misread -> SD-Turbo img2img [-> misread -> img2img]... -> anchor

`Reconstructor` owns the stage order, progress, the out-of-memory cascade
(448 -> 384 -> 320 px, then CPU) and the CPU fallback. The neural part sits
behind a small engine interface so the cascade can be tested with fake engines.
In the app the engine is a `ProcessEngine` (core/engine_process.py): SD-Turbo
runs in a child process so that unloading really returns the GPU memory.

    engine.device         DeviceInfo
    engine.low_memory     bool
    engine.load(status)
    engine.enable_low_memory() -> bool
    engine.img2img(image, num_inference_steps, strength, seed, tick) -> image
    engine.free_memory()
    engine.release()

`tick()` is called once after VAE encode, once per denoising step and once
after VAE decode, and raises `Cancelled` when the user stopped the run.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

import config
from core.device import DeviceInfo, cpu_device, peak_memory_gb, reset_peak_memory
from core.engine_process import ProcessEngine, probe_device
from core.imaging import prepare_array
from core.misread import apply_anchor, misread_detailed, plan_for

log = logging.getLogger(__name__)

StatusFn = Callable[[str], None]
TickFn = Callable[[], None]


class Cancelled(Exception):
    """The user stopped the run."""


class ReconstructionError(RuntimeError):
    pass


class NonFiniteOutput(RuntimeError):
    """The UNet produced NaN/Inf (typically FP16 trouble)."""


_OOM_MARKERS = (
    "out of memory",
    "not enough gpu video memory",
    "not enough memory",
    "could not allocate",
    "failed to allocate",
    "allocation failed",
    "insufficient memory",
    "e_outofmemory",
    "0x8007000e",
)


def is_oom_error(exc: BaseException) -> bool:
    """True for CUDA / DirectML / CPU allocation failures (checks the cause chain)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
            return True
        text = str(exc).lower()
        if any(marker in text for marker in _OOM_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


# --------------------------------------------------------------------------
# Out-of-memory cascade
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Attempt:
    device: DeviceInfo
    long_side: int
    low_memory: bool = False


def plan_attempts(device: DeviceInfo, cpu: DeviceInfo, sides=config.FALLBACK_SIDES,
                  start_low_memory: bool = False) -> list[Attempt]:
    """Order of tries: GPU (CUDA: then with CPU offload) at shrinking sizes, then CPU."""
    sides = tuple(sides)
    attempts: list[Attempt] = []
    if device.kind == "cuda":
        if not start_low_memory:
            attempts.append(Attempt(device, sides[0], False))
        attempts += [Attempt(device, side, True) for side in sides]
    elif device.kind == "dml":
        attempts += [Attempt(device, side) for side in sides]
    attempts += [Attempt(cpu, side) for side in sides]
    return attempts


class _Progress:
    def __init__(self, total: int, on_progress: Callable[[int], None], is_cancelled: Callable[[], bool]):
        self.total = max(1, total)
        self.done = 0
        self.last = -1
        self.on_progress = on_progress
        self.is_cancelled = is_cancelled
        self.emit(0)

    def emit(self, value: int) -> None:
        if value != self.last:
            self.last = value
            self.on_progress(value)

    def tick(self) -> None:
        if self.is_cancelled():
            raise Cancelled()
        self.done = min(self.total, self.done + 1)
        self.emit(min(99, int(100 * self.done / self.total)))


def _noop(*_args) -> None:
    pass


class Reconstructor:
    """Runs the whole reconstruction. Not thread-safe: one run at a time."""

    def __init__(self, device_pref: str = "auto", engine_factory=None, device_selector=None,
                 cpu_selector=None, sides=config.FALLBACK_SIDES):
        self.device_pref = device_pref
        self._engine_factory = engine_factory or ProcessEngine
        self._select_device = device_selector or probe_device
        self._cpu_device = cpu_selector or cpu_device
        self.sides = tuple(sides)
        self.device: Optional[DeviceInfo] = None
        self.engine = None
        self._device_lock = threading.Lock()

    def ensure_device(self) -> DeviceInfo:
        """Detect the device once (the window probes it at startup in a thread)."""
        with self._device_lock:
            if self.device is None:
                self.device = self._select_device(self.device_pref)
                log.info("Device: %s%s", self.device.label, f" ({self.device.note})" if self.device.note else "")
            return self.device

    @property
    def model_loaded(self) -> bool:
        return self.engine is not None

    def unload(self) -> None:
        """Free the model (GPU memory included). The next run loads it again."""
        if self.engine is not None:
            self.engine.release()
            self.engine = None
            log.info("Model unloaded")

    def preload(self, on_status: StatusFn = _noop) -> None:
        """Load the model up front (used by tools/sweep.py to time runs fairly)."""
        device = self.ensure_device()
        self._engine_for(Attempt(device, self.sides[0]), on_status)

    def run(self, image: Image.Image, weirdness: float, seed: int, *,
            on_progress: Callable[[int], None] = _noop, on_status: StatusFn = _noop,
            is_cancelled: Callable[[], bool] = lambda: False) -> Image.Image:
        plan = plan_for(weirdness)
        device = self.ensure_device()
        cpu = device if device.kind == "cpu" else self._cpu_device()
        start_low = bool(self.engine is not None and self.engine.device.kind == "cuda" and self.engine.low_memory)
        attempts = plan_attempts(device, cpu, self.sides, start_low_memory=start_low)
        log.info("Run: weirdness=%.2f seed=%d plan=%s", plan.weirdness, seed, plan)

        index = 0
        retried = False
        last_error: Optional[BaseException] = None
        while index < len(attempts):
            attempt = attempts[index]
            if is_cancelled():
                raise Cancelled()
            try:
                engine = self._engine_for(attempt, on_status)
            except Cancelled:
                raise
            except Exception as exc:
                last_error = exc
                if attempt.device.is_gpu:
                    log.exception("Loading the model on %s failed", attempt.device.label)
                    on_status(f"Could not load the model on {attempt.device.backend}. Falling back to the CPU…")
                    index = self._first_cpu(attempts, index)
                    self._switch_to_cpu(cpu)
                    continue
                raise
            try:
                return self._render(engine, image, plan, attempt.long_side, seed,
                                    on_progress, on_status, is_cancelled)
            except Cancelled:
                raise
            except Exception as exc:
                last_error = exc
                engine.free_memory()
                if is_oom_error(exc) and not retried:
                    # Often just fragmentation or another app's peak: free and try once more.
                    retried = True
                    log.warning("Out of memory on %s at %d px, retrying once after freeing memory: %s",
                                attempt.device.label, attempt.long_side, exc)
                    on_status("Not enough memory. Freeing it and trying again…")
                    continue
                if is_oom_error(exc) and index + 1 < len(attempts):
                    nxt = attempts[index + 1]
                    log.warning("Out of memory on %s at %d px: %s", attempt.device.label, attempt.long_side, exc)
                    if nxt.device.kind != attempt.device.kind:
                        on_status("Not enough GPU memory even at the smallest size. Falling back to the CPU…")
                        self._switch_to_cpu(cpu)
                    elif nxt.low_memory and not attempt.low_memory:
                        on_status("Not enough GPU memory. Retrying with model CPU offload…")
                    else:
                        on_status(f"Not enough memory at {attempt.long_side} px. Retrying at {nxt.long_side} px…")
                    index += 1
                    continue
                if attempt.device.is_gpu and not is_oom_error(exc):
                    log.exception("Inference on %s failed", attempt.device.label)
                    on_status(f"{attempt.device.backend} inference failed. Falling back to the CPU…")
                    index = self._first_cpu(attempts, index)
                    self._switch_to_cpu(cpu)
                    continue
                raise
        raise ReconstructionError(f"Ran out of memory at every size, last error: {last_error}") from last_error

    # -- internals --------------------------------------------------------

    @staticmethod
    def _first_cpu(attempts: list[Attempt], index: int) -> int:
        for i in range(index + 1, len(attempts)):
            if attempts[i].device.kind == "cpu":
                return i
        return len(attempts)

    def _switch_to_cpu(self, cpu: DeviceInfo) -> None:
        if self.engine is not None and self.engine.device.kind != "cpu":
            self.engine.release()
            self.engine = None
        if self.device is None or self.device.kind != "cpu":
            failed = self.device.backend if self.device else "GPU"
            self.device = replace(cpu, note=f"fell back from {failed}")

    def _engine_for(self, attempt: Attempt, on_status: StatusFn):
        if self.engine is not None and self.engine.device.kind != attempt.device.kind:
            self.engine.release()
            self.engine = None
        if self.engine is None:
            engine = self._engine_factory(attempt.device)
            started = time.perf_counter()
            engine.load(on_status)
            log.info("Model loaded on %s in %.1f s", attempt.device.label, time.perf_counter() - started)
            self.engine = engine
        if attempt.low_memory and not self.engine.low_memory:
            self.engine.enable_low_memory()
        return self.engine

    def _render(self, engine, image: Image.Image, plan, long_side: int, seed: int,
                on_progress, on_status: StatusFn, is_cancelled) -> Image.Image:
        work = prepare_array(image, long_side)
        steps_per_pass = plan.denoise_steps + 2  # encode + steps + decode
        total = 1 + (plan.passes - 1) + plan.passes * steps_per_pass + 1
        progress = _Progress(total, on_progress, is_cancelled)
        h, w = work.shape[:2]
        on_status(f"Misreading the picture ({w}×{h})…")

        first = misread_detailed(work, plan, seed)
        x, touched = first.image, first.touched
        progress.tick()
        for p in range(plan.passes):
            if p:
                again = misread_detailed(x, plan, seed + 7919 * p, scale=plan.repass_scale)
                x, touched = again.image, np.maximum(touched, again.touched)
                progress.tick()
            label = f" · pass {p + 1}/{plan.passes}" if plan.passes > 1 else ""
            on_status(f"Reconstructing on {engine.device.backend}{label}…")
            x = engine.img2img(x, plan.num_inference_steps, plan.strength, seed + p, progress.tick)
        out = apply_anchor(x, first.reference, plan, touched)
        progress.tick()
        progress.emit(100)
        return Image.fromarray(out)


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------

class PassthroughEngine:
    """No neural network: returns the misread image as is.

    Used by `tools/sweep.py --no-model` to tune the procedural part, and by tests.
    """

    def __init__(self, device: DeviceInfo):
        self.device = device
        self.low_memory = False

    def load(self, status: StatusFn) -> None:
        pass

    def enable_low_memory(self) -> bool:
        return False

    def img2img(self, image: np.ndarray, num_inference_steps: int, strength: float, seed: int,
                tick: TickFn) -> np.ndarray:
        for _ in range(int(num_inference_steps * strength) + 2):
            tick()
        return image

    def free_memory(self) -> None:
        pass

    def release(self) -> None:
        pass


class PromptEmbeds:
    """`prompt_embeds` for the service prompt, computed once and cached on disk.

    The tokenizer and text encoder are only loaded when the cache is missing and
    are dropped right after encoding.
    """

    def __init__(self, model_id: str = config.MODEL_ID, prompt: str = config.SERVICE_PROMPT,
                 cache_dir: Path = config.CACHE_DIR):
        self.model_id = model_id
        self.prompt = prompt
        self.cache_dir = Path(cache_dir)

    @property
    def path(self) -> Path:
        digest = hashlib.sha1(f"{self.model_id}\n{self.prompt}".encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"prompt_embeds_{digest}.pt"

    def get(self, status: StatusFn = _noop):
        import torch

        path = self.path
        if path.exists():
            try:
                return torch.load(path, map_location="cpu", weights_only=True)
            except Exception as exc:
                log.warning("Ignoring broken prompt cache %s: %s", path, exc)
        status("Encoding the service prompt (one time)…")
        embeds = self.encode()
        self.save(embeds)
        return embeds

    def encode(self):
        import torch
        from transformers import CLIPTextModel, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        text_encoder = _load_weights(CLIPTextModel, self.model_id, "text_encoder", None).float().eval()
        ids = tokenizer(self.prompt, padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            embeds = text_encoder(ids)[0].float().contiguous()
        del text_encoder, tokenizer
        gc.collect()
        log.info("Encoded the service prompt: %s, text encoder unloaded", tuple(embeds.shape))
        return embeds

    def save(self, embeds) -> None:
        import torch

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            torch.save(embeds, tmp)
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("Could not cache prompt embeddings: %s", exc)


def _load_weights(cls, model_id: str, subfolder: str, dtype):
    """from_pretrained, preferring the fp16 weight files (half the download)."""
    kwargs = {"subfolder": subfolder}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if config.PREFER_FP16_WEIGHTS:
        try:
            return cls.from_pretrained(model_id, variant="fp16", **kwargs)
        except (OSError, ValueError) as exc:
            log.info("No fp16 weights for %s (%s), loading the default files", subfolder, exc)
    return cls.from_pretrained(model_id, **kwargs)


class SDTurboEngine:
    """SD-Turbo img2img: UNet FP16 on CUDA (FP32 on CPU/DirectML), VAE always FP32.

    The VAE runs outside the diffusers pipeline: we encode to latents ourselves,
    let the pipeline denoise (`output_type="latent"`) and decode in FP32, which
    avoids the black images FP16 VAEs give on GTX 16xx cards.
    """

    def __init__(self, device: DeviceInfo, model_id: str = config.MODEL_ID,
                 prompt_embeds: Optional[PromptEmbeds] = None):
        self.device = device
        self.model_id = model_id
        self.prompt_embeds = prompt_embeds or PromptEmbeds(model_id)
        self.low_memory = False
        self.pipe = None
        self.vae = None
        self._embeds = None
        self._status: StatusFn = _noop

    # -- loading ------------------------------------------------------------

    def load(self, status: StatusFn = _noop) -> None:
        import torch
        from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

        self._status = status
        status(f"Loading SD-Turbo on {self.device.label}… (the first run downloads ~2.6 GB)")
        embeds = self.prompt_embeds.get(status)
        unet_dtype = torch.float16 if self.device.unet_dtype == "float16" else torch.float32
        status(f"Loading SD-Turbo weights ({'FP16' if unet_dtype == torch.float16 else 'FP32'} UNet)…")
        unet = _load_weights(UNet2DConditionModel, self.model_id, "unet", unet_dtype)
        vae = _load_weights(AutoencoderKL, self.model_id, "vae", torch.float32)
        scheduler = EulerDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler",
                                                           timestep_spacing="trailing")
        status(f"Moving the model to {self.device.backend}…")
        self.assemble(unet, vae, scheduler, embeds)

    def assemble(self, unet, vae, scheduler, embeds) -> None:
        """Build the pipeline from ready components (tests pass tiny models here)."""
        from diffusers import StableDiffusionImg2ImgPipeline

        pipe = StableDiffusionImg2ImgPipeline(
            vae=vae, text_encoder=None, tokenizer=None, unet=unet, scheduler=scheduler,
            safety_checker=None, feature_extractor=None, requires_safety_checker=False,
        )
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_attention_slicing()
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        self.pipe, self.vae, self._embeds = pipe, vae, embeds.float().cpu()

        vram = self.device.vram_gb
        if self.device.kind == "cuda" and vram is not None and vram < config.CUDA_OFFLOAD_BELOW_GB:
            log.info("%.1f GB of VRAM: starting with model CPU offload", vram)
            self.enable_low_memory()
        else:
            pipe.to(self.device.torch_device)

    def enable_low_memory(self) -> bool:
        """CUDA only: model CPU offload + VAE tiling."""
        if self.device.kind != "cuda" or self.pipe is None or self.low_memory:
            return False
        self.free_memory()
        self.pipe.enable_model_cpu_offload(device=self.device.torch_device)
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
        self.low_memory = True
        log.info("Model CPU offload enabled")
        return True

    # -- inference ----------------------------------------------------------

    def img2img(self, image: np.ndarray, num_inference_steps: int, strength: float, seed: int,
                tick: TickFn) -> np.ndarray:
        try:
            return self._img2img(image, num_inference_steps, strength, seed, tick)
        except Cancelled:
            raise
        except Exception as exc:
            if not self._unet_is_half() or is_oom_error(exc):
                raise
            log.warning("FP16 UNet failed (%s), switching to FP32", exc)
            self._status("FP16 failed on this device, switching the UNet to FP32…")
            self._upcast_unet()
            return self._img2img(image, num_inference_steps, strength, seed, tick)

    def _img2img(self, image, num_inference_steps, strength, seed, tick):
        import torch

        pipe = self.pipe
        device = pipe._execution_device
        unet_dtype = pipe.unet.dtype
        scaling = self.vae.config.scaling_factor
        with torch.no_grad():
            pixels = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
            pixels = (pixels.float() / 127.5 - 1.0).to(device)
            latents = self.vae.encode(pixels).latent_dist.mode() * scaling
            tick()

            def on_step(_pipe, _i, _t, kwargs):
                tick()
                return kwargs

            out = pipe(
                prompt_embeds=self._embeds.to(device=device, dtype=unet_dtype),
                image=latents.to(dtype=unet_dtype),
                strength=float(strength),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=0.0,
                generator=torch.Generator("cpu").manual_seed(int(seed)),
                output_type="latent",
                callback_on_step_end=on_step,
            ).images
            if not bool(torch.isfinite(out).all()):
                raise NonFiniteOutput("the UNet produced NaN/Inf values")
            decoded = self.vae.decode(out.to(device=device, dtype=torch.float32) / scaling).sample
            tick()
        decoded = decoded[0].float().cpu().clamp(-1.0, 1.0)
        result = ((decoded + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
        del pixels, latents, out, decoded
        gc.collect()  # DirectML keeps device buffers alive until their tensors are collected
        return result

    def _unet_is_half(self) -> bool:
        import torch

        return self.pipe is not None and self.pipe.unet.dtype == torch.float16

    def _upcast_unet(self) -> None:
        import torch

        self.pipe.unet.to(dtype=torch.float32)
        self.free_memory()

    # -- memory -------------------------------------------------------------

    def free_memory(self) -> None:
        gc.collect()
        if self.device.kind == "cuda":
            import torch

            torch.cuda.empty_cache()

    def reset_peak_memory(self) -> None:
        reset_peak_memory(self.device)

    def peak_memory_gb(self) -> Optional[float]:
        return peak_memory_gb(self.device)

    def release(self) -> None:
        """Drops the tensors. The backend may keep the memory reserved until the
        process ends, which is why the app runs this engine in a ProcessEngine."""
        self.pipe = None
        self.vae = None
        self.free_memory()
