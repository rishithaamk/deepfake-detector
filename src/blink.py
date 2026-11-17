"""Blink detection and blink-based biometric features from an EAR time-series."""
import numpy as np


def detect_blinks(ear, thr=None, min_frames=2):
    ear = np.asarray(ear, dtype=float)
    if thr is None:
        thr = 0.75 * np.percentile(ear, 90)  # adaptive: per-person open-eye baseline
    closed = ear < thr
    blinks, start = [], None
    for i, c in enumerate(closed):
        if c and start is None:
            start = i
        elif not c and start is not None:
            if i - start >= min_frames:
                blinks.append((start, i))
            start = None
    return blinks, float(thr)


BLINK_COLS = ["blink_rate", "blink_dur_mean", "blink_dur_std", "ibi_mean", "ibi_std",
              "ear_mean", "ear_std", "ear_min", "closed_ratio"]


def blink_features(ear, fps):
    ear = np.asarray(ear, dtype=float)
    total = len(ear) / fps
    blinks, thr = detect_blinks(ear)
    durs = np.array([(e - s) / fps for s, e in blinks])
    starts = np.array([s / fps for s, _ in blinks])
    ibi = np.diff(starts) if len(starts) > 1 else np.array([total])
    feats = {
        "blink_rate": len(blinks) / total * 60.0 if total > 0 else 0.0,  # blinks per minute
        "blink_dur_mean": float(durs.mean()) if len(durs) else 0.0,
        "blink_dur_std": float(durs.std()) if len(durs) else 0.0,
        "ibi_mean": float(ibi.mean()),
        "ibi_std": float(ibi.std()),
        "ear_mean": float(ear.mean()),
        "ear_std": float(ear.std()),
        "ear_min": float(ear.min()),
        "closed_ratio": float((ear < thr).mean()),
    }
    return feats, blinks, thr
