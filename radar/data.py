from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def load_df(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["new_id"] = df["new_id"].astype(str).str.strip()
    if "TB/Normal" in df.columns:
        df["label"] = df["TB/Normal"].astype(str).str.strip().str.lower().map(C.LABELS)
    else:
        df["label"] = np.nan
    mod = df["Modality_DICOM"] if "Modality_DICOM" in df.columns else pd.Series(["?"] * len(df))
    df["modality"] = mod.astype(str).str.strip().str.upper()
    return df


def image_path(new_id: str, split: str) -> Path:
    return (C.TRAIN_IMG_DIR if split == "train" else C.TEST_IMG_DIR) / f"{new_id}.png"


def cr_train_split(df: pd.DataFrame, val_frac: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    pool = df[(df["modality"] == C.SUPERVISED_MODALITY) & (df["label"].notna())].copy()
    val_idx = []
    for _, grp in pool.groupby("label"):
        n_val = max(1, int(round(len(grp) * val_frac)))
        val_idx.extend(rng.choice(grp.index.to_numpy(), size=n_val, replace=False).tolist())
    return pool.drop(index=val_idx).copy(), pool.loc[val_idx].copy()


def style_reservoir(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["modality"].isin(C.STYLE_MODALITIES)].copy()
