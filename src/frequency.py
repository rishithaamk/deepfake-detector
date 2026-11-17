"""Frequency-domain artifact analysis: 2-D FFT -> azimuthally averaged power spectrum."""
import numpy as np

N_BINS = 32


def radial_profile(gray, n_bins=N_BINS):
    gray = gray.astype(np.float32)
    gray = gray - gray.mean()
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)  # reduce edge leakage
    f = np.fft.fftshift(np.fft.fft2(gray * win))
    power = np.abs(f) ** 2
    yy, xx = np.indices(power.shape)
    r = np.hypot(yy - h // 2, xx - w // 2)
    bins = np.linspace(0, min(h, w) // 2, n_bins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < n_bins)
    sums = np.bincount(idx[ok], weights=power.ravel()[ok], minlength=n_bins)
    cnt = np.bincount(idx[ok], minlength=n_bins)
    return sums / np.maximum(cnt, 1)  # linear power per radial bin


def frequency_features(gray):
    """Vector of length N_BINS + 2: normalised log radial spectrum, high-freq energy ratio, spectral slope."""
    p = radial_profile(gray)
    logp = np.log(p + 1e-8)
    norm = logp - logp[1]  # illumination / contrast invariant
    third = N_BINS // 3
    hf_ratio = float(p[-third:].sum() / (p.sum() + 1e-8))
    slope = float(np.polyfit(np.arange(N_BINS), logp, 1)[0])
    return np.concatenate([norm, [hf_ratio, slope]]).astype(np.float32)
