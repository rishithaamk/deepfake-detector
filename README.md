---
title: Deepfake Detector
emoji: 👁️
sdk: gradio
app_file: app.py
python_version: "3.11"
---

# Real-Time Facial Deepfake Artifact Detector & Blinking Analyzer# Real-Time Facial Deepfake Artifact Detector & Blinking Analyzer
## Live Demo

[Open the app on Hugging Face Spaces](https://huggingface.co/spaces/rishithamk14/deepfake-detector)

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/rishithamk14/deepfake-detector)

**Concepts (Unit IV):** Facial landmarks · Frequency-artifact analysis · Biometrics  
**Stack:** MediaPipe FaceMesh, OpenCV, NumPy FFT, scikit-learn, Gradio (Dlib is an optional alternative landmark detector; MediaPipe is used here because it is faster and needs no model download).

## How it works
```
video/webcam -> MediaPipe FaceMesh (468 landmarks)
   |-> Eye Aspect Ratio per frame -> blink detection -> biometric features
   |     (blink rate, blink duration, inter-blink interval, EAR stats)
   '-> face crop -> grayscale -> 2-D FFT -> radial power spectrum -> frequency features
                      (GAN up-sampling leaves abnormal high-frequency energy)
features -> scikit-learn classifier (LogReg / SVM / RandomForest / GradBoost) -> Real vs Fake
-> Gradio UI (video upload + live webcam)
```
EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|); it drops towards 0 when the eye closes.

## Project files
| File | Purpose |
|---|---|
| `src/face.py` | FaceMesh wrapper, EAR, face crop |
| `src/blink.py` | blink detection + biometric features |
| `src/frequency.py` | FFT radial spectrum features |
| `src/features.py` | video/image -> feature dict |
| `extract_features.py` | dataset -> `features_*.csv` (resumable) |
| `train.py` | model selection, test metrics, ROC/confusion plots, **ablation** |
| `app.py` | Gradio app (also the file you deploy) |
| `notebooks/Deepfake_Detector_Colab.ipynb` | same pipeline for Google Colab |

## Step 1 - VS Code setup (use Python 3.10 or 3.11)
```bash
cd deepfake-detector
python -m venv .venv
.venv\Scripts\activate            # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
In VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" -> pick `.venv`.

## Step 2 - Get a real dataset
Pick ONE (A is fastest, B is best for your project because only video has blinking):

**A. 140k Real and Fake Faces (images, StyleGAN fakes)** - Kaggle, search "140k Real and Fake Faces" (xhlulu). Frequency features only. Good for a first end-to-end run.

**B. Celeb-DF v2 (videos, face-swap deepfakes)** - request via the official GitHub page `yuezunli/celeb-deepfakeforensics`. Folders: `Celeb-real`, `YouTube-real`, `Celeb-synthesis`.

**C. FaceForensics++ (videos)** - request access via GitHub `ondyari/FaceForensics`; or search Kaggle for "FaceForensics++ c23" mirrors. Use the c23 videos: `original` = real, `Deepfakes`, `FaceSwap`, `Face2Face`, `NeuralTextures` = fake.

I could not verify download links from here, so confirm the names above on Kaggle/GitHub. You do not need the whole dataset: **100-200 videos per class is enough** for a project.

## Step 3 - Extract features
```bash
# videos (Celeb-DF example)
python extract_features.py --mode video --real_dir data/Celeb-real data/YouTube-real --fake_dir data/Celeb-synthesis --limit 150 --out features_video.csv
# images (140k example - check actual folders)
python extract_features.py --mode image --real_dir data/real_vs_fake/real-vs-fake/train/real --fake_dir data/real_vs_fake/real-vs-fake/train/fake --limit 2000 --out features_image.csv
```
Video extraction takes ~5-15 s per video on a laptop; it saves progress, so you can stop and re-run.

## Step 4 - Train and evaluate
```bash
python train.py --mode video --features features_video.csv
python train.py --mode image --features features_image.csv     # optional; powers the live-webcam fake score
```
Outputs: `models/detector_*.joblib`, `outputs/confusion_*.png`, `roc_*.png`, `ablation.png/csv` (Frequency only vs Blink only vs Both), `metrics_*.json`.
Train/test split is grouped by source identity (prevents leakage) and model choice uses CV on the train split only.

## Step 5 - Run the app
```bash
python app.py        # open http://127.0.0.1:7860
```
Tab 1: upload a video -> verdict, blink stats, EAR plot with blinks, spectrum plot. Tab 2: webcam -> live EAR, blink count/rate, frequency score. (Gradio streams ~10 fps, so very short blinks can be missed live; the video tab is more accurate.)

## Step 6 - Deploy (Hugging Face Spaces, free)
1. Create an account, then New Space -> SDK **Gradio** -> CPU basic.
2. Upload: `app.py`, `requirements.txt`, `packages.txt`, `src/`, `models/` (the trained `.joblib` files).
3. Put this block at the very top of the Space's `README.md`:
```
---
title: Deepfake Artifact Detector
sdk: gradio
app_file: app.py
python_version: "3.10"
---
```
4. Wait for the build; you get a public URL. Webcam needs HTTPS, which Spaces provides.
Quick alternative for the demo: run the Colab notebook and use `demo.launch(share=True)` (link lasts 72 h).

## Step 7 - Submissions checklist
- [ ] Source code (this folder) + Colab notebook with outputs visible
- [ ] Screenshots: dataset folders, extraction log, training log, ROC, confusion matrix, ablation, app (real video, fake video, webcam), deployed URL
- [ ] Demo video (screen-record: upload real clip, fake clip, webcam)
- [ ] PPT (outline below)

## PPT outline (10-12 slides)
1 Title · 2 Problem (deepfake threat) · 3 Objectives · 4 GenAI concepts: how deepfakes are made (GAN/autoencoder) and why they leave artifacts · 5 Pipeline diagram · 6 Facial landmarks + EAR formula · 7 Blink biometrics (rate 15-20/min, 100-400 ms) · 8 Frequency analysis (FFT, radial spectrum, example plots) · 9 Dataset + split · 10 Results (ROC, confusion, ablation) · 11 App demo screenshots + deployment · 12 Limitations & future work (CNN/Xception, rPPG, temporal models) · References

## Honest limitations (say these in the viva)
- Modern deepfakes blink normally, so blink features alone are weak; the ablation shows this.
- Frequency cues are sensitive to compression and resolution; real and fake sets may differ in compression, which can inflate accuracy.
- Hand-crafted features + classical ML are a baseline, not state of the art. Not a forensic tool.
- Code was smoke-tested on synthetic data only; run it end to end on your data and tell me any error you see.
