"""Constants shared by the app, the pipeline and the tools."""

from pathlib import Path

APP_NAME = "Still Life AI"
APP_ID = "StillLifeAI"

APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
LOG_DIR = APP_DIR / "logs"
LOG_FILE = LOG_DIR / "stilllife.log"
CACHE_DIR = APP_DIR / "cache"

# --- Model -------------------------------------------------------------------

MODEL_ID = "stabilityai/sd-turbo"

# SD-Turbo needs some text conditioning. The prompt is encoded once, cached on
# disk (CACHE_DIR) and the text encoder is unloaded right after. Keep it neutral:
# with "mannequin" / "waxy" the model paints mannequins and wax faces, and
# "backrooms" / "fluorescent" push it towards a yellow cast.
SERVICE_PROMPT = "the same picture redrawn from memory, slightly wrong details"

# Try the fp16 weight files first: half the download, upcast where needed.
PREFER_FP16_WEIGHTS = True

# DirectML runs the UNet in FP32 by default. Set to True to try FP16 there;
# on errors or NaNs the engine falls back to FP32 by itself.
DML_TRY_FP16 = False

# CUDA cards with less VRAM than this start with model CPU offload enabled.
CUDA_OFFLOAD_BELOW_GB = 3.5

# --- Working resolution ------------------------------------------------------

# Long side of the working image. It does NOT depend on the weirdness.
WORK_LONG_SIDE = 448
# Out-of-memory cascade: retry smaller before falling back to the CPU.
FALLBACK_SIDES = (448, 384, 320)
SIZE_MULTIPLE = 8
MIN_SIDE = 64

# Transparent pixels are flattened onto this colour.
ALPHA_BACKGROUND = (255, 255, 255)

# --- Generation --------------------------------------------------------------

DEFAULT_SEED = 1984

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
