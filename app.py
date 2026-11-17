"""Step 3: Gradio app - video analysis + live webcam analysis.   Run:  python app.py"""
import collections, os, threading, time

import cv2
import gradio as gr
import joblib
import numpy as np
from matplotlib.figure import Figure

from src.face import FaceTracker, mean_ear, crop_face, LEFT_EYE, RIGHT_EYE
from src.features import analyze_video, vec_to_dict
from src.frequency import frequency_features, N_BINS


def _load(path):
    return joblib.load(path) if os.path.exists(path) else None


VIDEO_MODEL = _load("models/detector_video.joblib")
IMAGE_MODEL = _load("models/detector_image.joblib")


def predict_fake_prob(bundle, feats):
    x = np.array([[feats[c] for c in bundle["cols"]]])
    return float(bundle["model"].predict_proba(x)[0, 1])


# ---------------------------------------------------------------- video tab
def plot_ear(res):
    fig = Figure(figsize=(7, 3))
    ax = fig.subplots()
    t = np.arange(len(res["ear"])) / res["fps"]
    ax.plot(t, res["ear"], lw=1, label="EAR")
    ax.axhline(res["thr"], color="r", ls="--", label="blink threshold")
    for s, e in res["blinks"]:
        ax.axvspan(s / res["fps"], e / res["fps"], color="orange", alpha=0.35)
    ax.set_xlabel("time (s)"); ax.set_ylabel("Eye Aspect Ratio"); ax.set_title("Blink analysis"); ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_spectrum(res):
    fig = Figure(figsize=(7, 3))
    ax = fig.subplots()
    ax.plot(np.log(res["mean_spectrum"] + 1e-8))
    ax.set_xlabel("spatial frequency (low -> high)"); ax.set_ylabel("log power"); ax.set_title("Azimuthally averaged power spectrum")
    fig.tight_layout()
    return fig


def analyze(video):
    if not video:
        return None, "Upload a video first.", None, None, None
    res = analyze_video(video)
    if res is None:
        return None, "No stable face found in this video (need a visible face for >= 1 s).", None, None, None
    f = res["features"]
    label, notes = None, []
    if VIDEO_MODEL:
        p = predict_fake_prob(VIDEO_MODEL, f)
        label = {"Fake": p, "Real": 1 - p}
    else:
        notes.append("_No trained model found in `models/` - showing biometrics only._")
    if f["blink_rate"] < 8:
        notes.append("Blink rate is unusually LOW (early deepfakes under-blink).")
    elif f["blink_rate"] > 40:
        notes.append("Blink rate is unusually HIGH.")
    md = (f"**Blink rate:** {f['blink_rate']:.1f} /min (typical 15-20)  \n"
          f"**Mean blink duration:** {f['blink_dur_mean'] * 1000:.0f} ms (typical 100-400)  \n"
          f"**Inter-blink interval:** {f['ibi_mean']:.2f} s  \n"
          f"**High-frequency energy ratio:** {f['hf_ratio']:.4f}  \n"
          f"**Spectral slope:** {f['spec_slope']:.3f}  \n\n" + "  \n".join(notes))
    return label, md, plot_ear(res), plot_spectrum(res), res["face_crop"]


# ---------------------------------------------------------------- live tab
LIVE_TRACKER = FaceTracker(static=False)
LOCK = threading.Lock()


def live(frame, state):
    if frame is None:
        return None, state, "Waiting for webcam..."
    now = time.time()
    if state is None:
        state = {"t0": now, "ear": collections.deque(maxlen=240), "closed": False, "blinks": 0,
                 "n": 0, "prob": collections.deque(maxlen=10), "hf": collections.deque(maxlen=10)}
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    with LOCK:
        pts = LIVE_TRACKER.process(bgr)
    if pts is None:
        return frame, state, "No face detected"
    e = mean_ear(pts)
    state["ear"].append(e)
    buf = np.array(state["ear"])
    thr = 0.75 * np.percentile(buf, 90) if len(buf) >= 40 else 0.2
    if e < thr and not state["closed"]:
        state["closed"] = True
    elif e >= thr and state["closed"]:
        state["closed"] = False
        state["blinks"] += 1
    state["n"] += 1
    if state["n"] % 5 == 0:
        crop = crop_face(bgr, pts)
        if crop is not None:
            vec = frequency_features(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            state["hf"].append(float(vec[N_BINS]))
            if IMAGE_MODEL:
                state["prob"].append(predict_fake_prob(IMAGE_MODEL, vec_to_dict(vec)))
    out = frame.copy()
    for i in LEFT_EYE + RIGHT_EYE:
        cv2.circle(out, (int(pts[i][0]), int(pts[i][1])), 2, (0, 255, 0), -1)
    elapsed = now - state["t0"]
    rate = state["blinks"] / (elapsed / 60) if elapsed > 15 else float("nan")
    lines = [f"**EAR:** {e:.3f} {'(eyes closed)' if state['closed'] else ''}",
             f"**Blinks:** {state['blinks']} in {elapsed:.0f}s" + (f" -> {rate:.1f}/min" if rate == rate else " (calibrating...)")]
    if state["prob"]:
        p = float(np.mean(state["prob"]))
        lines.append(f"**Frequency-model fake score:** {p:.2f}")
    elif state["hf"]:
        lines.append(f"**High-frequency energy ratio:** {np.mean(state['hf']):.4f}")
    if rate == rate and (rate < 8 or rate > 40):
        lines.append("Warning: abnormal blink rate")
    return out, state, "  \n".join(lines)


# ---------------------------------------------------------------- UI
with gr.Blocks(title="Deepfake Artifact Detector & Blink Analyzer") as demo:
    gr.Markdown("# Real-Time Facial Deepfake Artifact Detector & Blinking Analyzer\n"
                "MediaPipe landmarks -> Eye-Aspect-Ratio blink biometrics + FFT frequency artifacts -> scikit-learn classifier.")
    with gr.Tab("Analyze video"):
        with gr.Row():
            vid = gr.Video(label="Input video (face visible, >= 5 s recommended)")
            with gr.Column():
                lab = gr.Label(label="Verdict")
                info = gr.Markdown()
        btn = gr.Button("Analyze", variant="primary")
        with gr.Row():
            p1, p2, crop = gr.Plot(label="EAR / blinks"), gr.Plot(label="Spectrum"), gr.Image(label="Analyzed face crop")
        btn.click(analyze, vid, [lab, info, p1, p2, crop])
    with gr.Tab("Live webcam"):
        st = gr.State(None)
        with gr.Row():
            cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam")
            out_img = gr.Image(label="Landmarks")
        live_info = gr.Markdown()
        cam.stream(live, [cam, st], [out_img, st, live_info])
    gr.Markdown("Research/educational demo - not a forensic tool. Accuracy depends on training data and video quality.")

if __name__ == "__main__":
    demo.launch()
