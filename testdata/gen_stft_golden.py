#!/usr/bin/env python3
"""Generate a librosa STFT power-spectrogram golden vector for the simd parity
tests (f64/stft_test.go, f32/stft_test.go).

Run from the repo root with the project venv:

    .venv-librosa/bin/python testdata/gen_stft_golden.py

The script emits the exact signal and periodic-Hann window it used, so the Go
tests feed byte-identical inputs and there is no convention drift between the two
sides. The power spectrogram is |librosa.stft(...)|**2 over nfft/2+1 bins, which
is what librosa.feature.melspectrogram consumes before the mel projection.
"""
import json

import librosa
import numpy as np

NFFT, HOP, N = 1024, 768, 4096

# Deterministic real signal (mirrors the Go testSignal recipe).
t = np.arange(N, dtype=np.float64)
sig = np.sin(0.3 * t) + 0.5 * np.cos(0.11 * t + 1) - 0.25 * np.sin(0.027 * t)

# Periodic Hann (fftbins=True): w[n] = 0.5 - 0.5*cos(2*pi*n/nfft), n in [0, nfft).
win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT)

cases = []
for go_pad, pad_mode in [("PadZero", "constant"), ("PadReflect", "reflect")]:
    spec = librosa.stft(
        sig, n_fft=NFFT, hop_length=HOP, window=win, center=True, pad_mode=pad_mode
    )
    power = (np.abs(spec) ** 2).T  # frames x bins
    cases.append(
        {
            "go_pad": go_pad,
            "pad_mode": pad_mode,
            "frames": int(power.shape[0]),
            "bins": int(power.shape[1]),
            "power": power.reshape(-1).tolist(),
        }
    )

out = {
    "librosa_version": librosa.__version__,
    "nfft": NFFT,
    "hop": HOP,
    "signal": sig.tolist(),
    "window": win.tolist(),
    "cases": cases,
}
with open("testdata/stft_librosa_golden.json", "w") as f:
    json.dump(out, f)

print(
    "librosa",
    librosa.__version__,
    "cases",
    [(c["go_pad"], c["frames"], c["bins"]) for c in cases],
)
