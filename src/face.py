"""Facial landmarks (MediaPipe FaceMesh), Eye Aspect Ratio and face cropping."""
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

# FaceMesh landmark indices for the 6 EAR points of each eye: p1(corner), p2, p3 (upper lid), p4(corner), p5, p6 (lower lid)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Downloading face_landmarker.task (~4 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class FaceTracker:
    """Works with both MediaPipe APIs: legacy mp.solutions (<=0.10.21) and the new Tasks API (>=0.10.30)."""

    def __init__(self, static: bool = False):
        self.legacy = hasattr(mp, "solutions")
        if self.legacy:
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=static, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=_ensure_model()),
                running_mode=vision.RunningMode.IMAGE, num_faces=1)
            self.mesh = vision.FaceLandmarker.create_from_options(opts)

    def process(self, bgr):
        """Return (N, 2) pixel landmarks or None if no face."""
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        h, w = bgr.shape[:2]
        if self.legacy:
            res = self.mesh.process(rgb)
            if not res.multi_face_landmarks:
                return None
            lm = res.multi_face_landmarks[0].landmark
        else:
            res = self.mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.face_landmarks:
                return None
            lm = res.face_landmarks[0]
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

    def close(self):
        self.mesh.close()


def eye_aspect_ratio(pts, idx):
    p1, p2, p3, p4, p5, p6 = pts[idx]
    vert = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horiz = 2.0 * np.linalg.norm(p1 - p4) + 1e-6
    return float(vert / horiz)


def mean_ear(pts):
    return (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2.0


def crop_face(bgr, pts=None, margin=0.15, size=128):
    """Crop the face using landmark bounds (or return the whole image resized if pts is None)."""
    h, w = bgr.shape[:2]
    if pts is None:
        return cv2.resize(bgr, (size, size))
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = int(max(0, x1 - margin * bw)), int(max(0, y1 - margin * bh))
    x2, y2 = int(min(w, x2 + margin * bw)), int(min(h, y2 + margin * bh))
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size))
