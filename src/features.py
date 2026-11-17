"""Video / image -> feature dictionaries (frequency + blink biometrics)."""
import cv2
import numpy as np

from .face import FaceTracker, mean_ear, crop_face
from .frequency import frequency_features, N_BINS
from .blink import blink_features, BLINK_COLS

FREQ_COLS = [f"freq_{i}" for i in range(N_BINS)] + ["hf_ratio", "spec_slope"]
ALL_COLS = FREQ_COLS + BLINK_COLS


def vec_to_dict(vec):
    return {c: float(v) for c, v in zip(FREQ_COLS, vec)}


def _shrink(frame, max_side=640):
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(frame, (int(w * s), int(h * s))) if s < 1 else frame


def analyze_video(path, max_frames=300, freq_every=5):
    """Returns dict(features, ear, fps, blinks, thr, mean_spectrum, face_crop) or None if no usable face."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 5:
        fps = 30.0
    tracker = FaceTracker(static=False)
    ears, vecs, last_crop, n = [], [], None, 0
    try:
        while n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            frame = _shrink(frame)
            pts = tracker.process(frame)
            if pts is None:
                ears.append(np.nan)
                continue
            ears.append(mean_ear(pts))
            if len(ears) % freq_every == 0:
                crop = crop_face(frame, pts)
                if crop is not None:
                    last_crop = crop
                    vecs.append(frequency_features(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
    finally:
        cap.release()
        tracker.close()
    ears = np.array(ears, dtype=float)
    valid = ~np.isnan(ears)
    if len(ears) == 0 or valid.sum() < 30 or valid.mean() < 0.3 or not vecs:
        return None
    idx = np.arange(len(ears))
    ears = np.interp(idx, idx[valid], ears[valid])  # fill frames where the face was lost
    bf, blinks, thr = blink_features(ears, fps)
    mean_vec = np.mean(vecs, axis=0)
    feats = {**vec_to_dict(mean_vec), **bf}
    return {"features": feats, "ear": ears, "fps": fps, "blinks": blinks, "thr": thr,
            "mean_spectrum": mean_vec[:N_BINS], "face_crop": cv2.cvtColor(last_crop, cv2.COLOR_BGR2RGB)}


def analyze_image(path, tracker):
    """Frequency-only features for a still image. Falls back to the whole image if no face is found."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = _shrink(img)
    pts = tracker.process(img)
    crop = crop_face(img, pts)
    if crop is None:
        return None
    return vec_to_dict(frequency_features(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
