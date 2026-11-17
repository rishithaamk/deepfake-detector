"""Step 2: train + evaluate classifiers, run feature ablation, save model and plots.

  python train.py --mode video --features features_video.csv
  python train.py --mode image --features features_image.csv
"""
import argparse, json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.features import FREQ_COLS, ALL_COLS
from src.blink import BLINK_COLS


def models():
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(probability=True, C=3)),
        "RandomForest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)),
        "GradBoost": make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["video", "image"], required=True)
    ap.add_argument("--features", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.features).dropna()
    cols = FREQ_COLS if a.mode == "image" else ALL_COLS
    X, y, g = df[cols].values, df["label"].values, df["group"].values
    print(f"{len(df)} samples | real={int((y == 0).sum())} fake={int((y == 1).sum())} | {len(set(g))} groups")

    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, g))
    Xtr, Xte, ytr, yte, gtr = X[tr], X[te], y[tr], y[te], g[tr]

    # model selection with grouped CV on the TRAIN split only
    cv = GroupKFold(n_splits=min(5, len(set(gtr))))
    cv_scores = {}
    for name, m in models().items():
        cv_scores[name] = cross_val_score(m, Xtr, ytr, groups=gtr, cv=cv, scoring="roc_auc").mean()
        print(f"CV AUC  {name:13s} {cv_scores[name]:.3f}")
    best = max(cv_scores, key=cv_scores.get)
    model = models()[best].fit(Xtr, ytr)

    prob = model.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    acc, auc = accuracy_score(yte, pred), roc_auc_score(yte, prob)
    print(f"\nBEST = {best} | TEST accuracy={acc:.3f} AUC={auc:.3f}")
    print(classification_report(yte, pred, target_names=["real", "fake"]))

    ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=["real", "fake"]).plot(cmap="Blues")
    plt.title(f"Confusion matrix ({best})"); plt.savefig(f"outputs/confusion_{a.mode}.png", dpi=150, bbox_inches="tight"); plt.close()
    fpr, tpr, _ = roc_curve(yte, prob)
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}"); plt.plot([0, 1], [0, 1], "--", c="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC"); plt.legend()
    plt.savefig(f"outputs/roc_{a.mode}.png", dpi=150, bbox_inches="tight"); plt.close()

    # ablation (video mode): which feature family matters?
    if a.mode == "video":
        res = {}
        for tag, c in {"Frequency only": FREQ_COLS, "Blink only": BLINK_COLS, "Frequency + Blink": ALL_COLS}.items():
            m = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1))
            m.fit(df.iloc[tr][c].values, ytr)
            p = m.predict_proba(df.iloc[te][c].values)[:, 1]
            res[tag] = {"accuracy": accuracy_score(yte, p >= .5), "auc": roc_auc_score(yte, p)}
        ab = pd.DataFrame(res).T
        print("\nABLATION\n", ab.round(3))
        ab.to_csv("outputs/ablation.csv")
        ab["auc"].plot(kind="bar", ylim=(0, 1), rot=0, title="Ablation (test AUC)")
        plt.savefig("outputs/ablation.png", dpi=150, bbox_inches="tight"); plt.close()

    joblib.dump({"model": model, "cols": cols, "mode": a.mode, "name": best}, f"models/detector_{a.mode}.joblib")
    json.dump({"best": best, "cv_auc": cv_scores, "test_acc": acc, "test_auc": auc, "n_train": len(tr), "n_test": len(te)},
              open(f"outputs/metrics_{a.mode}.json", "w"), indent=2)
    print(f"Saved models/detector_{a.mode}.joblib and plots in outputs/")


if __name__ == "__main__":
    main()
