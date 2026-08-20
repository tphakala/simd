#!/usr/bin/env python3
"""Generate a librosa STFT power-spectrogram golden vector for the simd parity
tests (f64/stft_test.go, f32/stft_test.go), plus an istft golden
(istft_librosa_golden.json) for the ISTFT parity tests.

Run from the repo root with the project venv. By default it writes the f64
package copy; pass one or more output directories to target others:

    .venv-librosa/bin/python testdata/gen_stft_golden.py              # -> f64/testdata
    .venv-librosa/bin/python testdata/gen_stft_golden.py f32/testdata # -> f32/testdata

The golden is embedded into the test binary with //go:embed, so it must live
under the package directory (go:embed cannot reach "../testdata"). The signal and
periodic-Hann window are NOT stored: the Go tests regenerate them with the same
deterministic formulas (testSignal / hann), so only librosa's power output needs
pinning. Power is rounded to 10 significant figures (quantization ~1e-10, far
under the test's 1e-6 parity tolerance) to keep the committed file small.
"""
import json
import os
import sys

import librosa
import numpy as np

NFFT, HOP, N = 1024, 768, 4096

# Deterministic real signal; MUST match f64 testSignal / f32 testSignal in Go.
t = np.arange(N, dtype=np.float64)
sig = np.sin(0.3 * t) + 0.5 * np.cos(0.11 * t + 1) - 0.25 * np.sin(0.027 * t)

# Periodic Hann (fftbins=True); MUST match the Go hann helper.
win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT)


def round_sig(x, sig=10):
    return float(f"{x:.{sig}g}")


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
            "power": [round_sig(v) for v in power.reshape(-1)],
        }
    )

out = {
    "librosa_version": librosa.__version__,
    "nfft": NFFT,
    "hop": HOP,
    "n": N,
    "cases": cases,
}

out_dirs = sys.argv[1:] or ["f64/testdata"]
for d in out_dirs:
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "stft_librosa_golden.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote", path)

# ISTFT golden: a per-bin gain is applied to the forward spectrum before the
# inverse, so the test pins normalization and centering rather than only the
# identity round trip. hop NFFT/4 keeps every sample fully covered by the Hann
# squared-window overlap, so there are no unnormalizable edge samples.
ISTFT_HOP = NFFT // 4
GAIN_PERIOD = 7.0
k = np.arange(NFFT // 2 + 1, dtype=np.float64)
gain = 0.5 + 0.5 * np.cos(k / GAIN_PERIOD)
icases = []
for go_pad, pad_mode in [("PadZero", "constant"), ("PadReflect", "reflect")]:
    spec = librosa.stft(
        sig, n_fft=NFFT, hop_length=ISTFT_HOP, window=win, center=True, pad_mode=pad_mode
    )
    y = librosa.istft(
        spec * gain[:, None], hop_length=ISTFT_HOP, window=win, center=True, length=N
    )
    icases.append(
        {
            "go_pad": go_pad,
            "frames": int(spec.shape[1]),
            "y": [round_sig(v) for v in y],
        }
    )
iout = {
    "librosa_version": librosa.__version__,
    "nfft": NFFT,
    "hop": ISTFT_HOP,
    "n": N,
    "gain_period": GAIN_PERIOD,
    "cases": icases,
}
for d in out_dirs:
    path = os.path.join(d, "istft_librosa_golden.json")
    with open(path, "w") as f:
        json.dump(iout, f)
    print("wrote", path)

print(
    "librosa",
    librosa.__version__,
    "cases",
    [(c["go_pad"], c["frames"], c["bins"]) for c in cases],
)
