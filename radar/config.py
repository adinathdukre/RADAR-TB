from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("RADAR_DATA", "data/Task2/Data"))
TRAIN_IMG_DIR = Path(os.environ.get("RADAR_TRAIN_IMG", str(DATA_ROOT / "train")))
TEST_IMG_DIR = Path(os.environ.get("RADAR_TEST_IMG", str(DATA_ROOT / "test")))
TRAIN_CSV = Path(os.environ.get("RADAR_TRAIN_CSV", str(DATA_ROOT / "train.csv")))
TEST_CSV = Path(os.environ.get("RADAR_TEST_CSV", str(DATA_ROOT / "test.csv")))

WORK_DIR = Path(os.environ.get("RADAR_WORK", "work"))
BBOX_DIR = WORK_DIR / "bboxes"
TOKEN_DIR = Path(os.environ.get("RADAR_TOKENS", str(WORK_DIR / "tokens")))
EXTERNAL_DIR = Path(os.environ.get("RADAR_EXTERNAL", str(WORK_DIR / "external")))
WEIGHTS_DIR = Path(os.environ.get("RADAR_WEIGHTS", "weights"))
CACHE_DIR = Path(os.environ.get("RADAR_CACHE", "checkpoints"))

HF_ID = "microsoft/rad-dino"
LAYERS = [3, 6, 9, 12]
EMBED = 768
RES = 518
HIRES_RES = 700

LABELS = {"normal": 0, "tb": 1}
INV_LABELS = {0: "Normal", 1: "TB"}
MODALITIES = ["CR", "DX", "XC", "XA"]
SUPERVISED_MODALITY = "CR"
STYLE_MODALITIES = ["XC", "DX", "XA"]

GROUPS = {
    "dep": ["dep_plain_seed*.pt", "dep_aug_seed*.pt"],
    "adv": ["adv_seed*.pt"],
    "cdan": ["cdan_seed*.pt"],
    "hires": ["hires_seed*.pt"],
}
GROUP_WEIGHTS = {"adv": 0.26, "cdan": 0.35, "dep": 0.39}
W_CROP = 0.90
W_HIRES = 0.15
PLATT_A = 1.425
PLATT_B = 0.057
THRESHOLD = 0.94
POLARITY_REF = "polarity_ref.npy"
