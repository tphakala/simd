# simd

[![Go Reference](https://pkg.go.dev/badge/github.com/tphakala/simd.svg)](https://pkg.go.dev/github.com/tphakala/simd)
[![Go Report Card](https://goreportcard.com/badge/github.com/tphakala/simd)](https://goreportcard.com/report/github.com/tphakala/simd)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance SIMD (Single Instruction, Multiple Data) library for Go providing vectorized operations on float64, float32, float16, int32, int16, complex128, and complex64 slices.

## Features

- **Pure Go assembly** - Native Go assembler, simple cross-compilation
- **Runtime CPU detection** - Automatically selects optimal implementation (AVX-512, AVX+FMA, AVX without FMA, SSE2, NEON, NEON+FP16, or pure Go); the minimum amd64 SIMD tier is per-package (see [Architecture Support](#architecture-support))
- **Zero allocations** - All operations work on pre-allocated slices
- **80+ operations** - Arithmetic, reduction, statistical, vector, signal processing, activation functions, and complex number operations
- **Multi-architecture** - AMD64 (AVX-512/AVX+FMA/SSE2, c64 needs SSE4.1) and ARM64 (NEON/NEON+FP16) with pure Go fallback
- **Half-precision support** - Native FP16 SIMD on ARM64 with FP16 extension (Apple Silicon, Cortex-A55+); F16C-accelerated conversions on AMD64
- **Tunable dispatch** - `SIMD_DISABLE` env var masks feature tiers at startup (avoid AVX-512 downclocking, exercise lower tiers, benchmark tier-vs-tier)
- **Thread-safe** - All functions are safe for concurrent use

## Installation

```bash
go get github.com/tphakala/simd
```

Requires Go 1.25+

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/tphakala/simd/cpu"
    "github.com/tphakala/simd/f64"
)

func main() {
    fmt.Println("SIMD:", cpu.Info())

    // Vector operations
    a := []float64{1, 2, 3, 4, 5, 6, 7, 8}
    b := []float64{8, 7, 6, 5, 4, 3, 2, 1}

    // Dot product
    dot := f64.DotProduct(a, b)
    fmt.Println("Dot product:", dot) // 120

    // Element-wise operations
    dst := make([]float64, len(a))
    f64.Add(dst, a, b)
    fmt.Println("Sum:", dst) // [9, 9, 9, 9, 9, 9, 9, 9]

    // Statistical operations
    mean := f64.Mean(a)
    stddev := f64.StdDev(a)
    fmt.Printf("Mean: %.2f, StdDev: %.2f\n", mean, stddev)

    // Vector operations
    f64.Normalize(dst, a)
    fmt.Println("Normalized:", dst)

    // Distance calculation
    dist := f64.EuclideanDistance(a, b)
    fmt.Println("Distance:", dist)
}
```

## Packages

### `cpu` - CPU Feature Detection

```go
import "github.com/tphakala/simd/cpu"

fmt.Println(cpu.Info())        // "AMD64 AVX-512", "AMD64 AVX+FMA", "AMD64 AVX", "AMD64 SSE2", "AMD64 (scalar)", "ARM64 NEON+FP16", or "ARM64 NEON"
                               // SVE-capable ARM64 hosts append " (SVE detected, unused)" - the library runs the NEON path
fmt.Println(cpu.HasAVX())      // true/false
fmt.Println(cpu.HasAVX2())     // true/false
fmt.Println(cpu.HasFMA())      // true/false
fmt.Println(cpu.HasAVX512VL()) // true/false (AVX-512 F+VL)
fmt.Println(cpu.HasNEON())     // true/false
fmt.Println(cpu.HasFP16())     // true/false (ARM64 half-precision SIMD)
fmt.Println(cpu.HasPCLMULQDQ()) // true/false (x86 carry-less multiply)
fmt.Println(cpu.HasF16C())     // true/false (x86 half<->single conversion)
fmt.Println(cpu.HasPMULL())    // true/false (ARM64 polynomial multiply)
```

#### Disabling feature tiers with `SIMD_DISABLE`

Set the `SIMD_DISABLE` environment variable before the process starts to mask
detected CPU features. This is useful for forcing a lower tier on parts where
heavy AVX-512 use causes frequency downclocking, exercising the SSE2/NEON/pure-Go
paths locally, and benchmarking tiers against each other on one machine.

The value is a comma-separated, case-insensitive list of tokens, read once at
program start. Each token clears its own flag plus everything that depends on it:

| Token       | Clears                                    |
| ----------- | ----------------------------------------- |
| `avx512`    | AVX512F, AVX512VL                         |
| `avx2`      | AVX2 (and the `avx512` set)               |
| `avx`       | AVX, FMA, F16C (and the `avx2` set)       |
| `fma`       | FMA only                                  |
| `sse42`     | SSE42 (and the `avx` set)                 |
| `sse41`     | SSE41 (and the `sse42` set)               |
| `ssse3`     | SSSE3 (and the `sse41` set)               |
| `sse3`      | SSE3 (and the `ssse3` set)                |
| `pclmulqdq` | PCLMULQDQ only                            |
| `neon`      | NEON, FP16, SVE, SVE2, PMULL              |
| `fp16`      | FP16 only                                 |
| `sve`       | SVE, SVE2                                 |
| `pmull`     | PMULL only                                |
| `all`       | every flag (forces the pure-Go path)      |

F16C is VEX-encoded and only detected alongside AVX, so it clears with the `avx`
cascade (and therefore with every `sse*` token and `all`); `avx2`, `fma`, and
`avx512` sit above AVX and leave F16C set.

Unknown tokens are ignored (the library never panics or writes to stderr on env
input). `cpu.Info()` reflects the cleared flags.

```sh
SIMD_DISABLE=avx512 go test ./...   # run as if the CPU had no AVX-512
SIMD_DISABLE=all go test ./...      # force the pure-Go path everywhere
```

The variable must be set before the process starts; it cannot be toggled at
runtime, because the SIMD packages cache their selected kernels during package
init based on the features visible at that moment (function pointers on amd64,
capability flags on arm64).

### `crc` - Cyclic Redundancy Checks

```go
import "github.com/tphakala/simd/crc"

// CRC-16 over poly 0x8005 (init 0, MSB-first, no reflection), the unreflected
// 0x8005 parameterization FLAC uses; folded 16 bytes at a time with PCLMULQDQ
// (amd64) / PMULL (arm64), scalar slice-by-16 fallback.
sum := crc.Checksum16(p) // bit-identical to the scalar reference, zero-alloc
```

| Function          | Description                              | Acceleration                       |
| ----------------- | ---------------------------------------- | ---------------------------------- |
| `Checksum16(p)`   | CRC-16 (poly 0x8005, MSB-first; used by FLAC) | PCLMULQDQ / PMULL carry-less fold |

### `f64` - float64 Operations

**Scope:** `f64` carries the FLAC/LPC and scientific double-precision surface,
including `Autocorrelate` (lag-vectorized LPC autocorrelation). Audio/ML-specific
helpers (PCM conversions, split-format complex ops, indexed/strided dot products)
live in `f32` instead, so the two float surfaces are intentionally asymmetric.

| Category        | Function                            | Description                   | SIMD Width                          |
| --------------- | ----------------------------------- | ----------------------------- | ----------------------------------- |
| **Arithmetic**  | `Add(dst, a, b)`                    | Element-wise addition         | 8x (AVX-512) / 4x (AVX) / 2x (NEON) |
|                 | `Sub(dst, a, b)`                    | Element-wise subtraction      | 8x / 4x / 2x                        |
|                 | `Mul(dst, a, b)`                    | Element-wise multiplication   | 8x / 4x / 2x                        |
|                 | `Div(dst, a, b)`                    | Element-wise division         | 8x / 4x / 2x                        |
|                 | `Scale(dst, a, s)`                  | Multiply by scalar            | 8x / 4x / 2x                        |
|                 | `AddScalar(dst, a, s)`              | Add scalar                    | 8x / 4x / 2x                        |
|                 | `SubFromScalar(dst, a, s)`          | Scalar minus vector           | 8x / 4x / 2x (composed SIMD)        |
|                 | `FMA(dst, a, b, c)`                 | Fused multiply-add: a\*b+c    | 8x / 4x / 2x                        |
|                 | `AddScaled(dst, alpha, s)`          | dst += alpha\*s (axpy)        | 8x / 4x / 2x                        |
| **Unary**       | `Abs(dst, a)`                       | Absolute value                | 8x / 4x / 2x                        |
|                 | `Neg(dst, a)`                       | Negation                      | 8x / 4x / 2x                        |
|                 | `Sqrt(dst, a)`                      | Square root                   | 8x / 4x / 2x                        |
|                 | `Reciprocal(dst, a)`                | Reciprocal (1/x)              | 8x / 4x / 2x                        |
|                 | `Round(dst, src)`                   | Round half away from zero     | 4x (AVX) / 2x (NEON) / Go fallback  |
| **Reduction**   | `DotProduct(a, b)`                  | Dot product                   | 8x / 4x / 2x                        |
|                 | `WeightedSum(w, src)`               | Weighted sum Σ(wᵢ·srcᵢ)       | 8x / 4x / 2x                        |
|                 | `SumOfSquares(src)`                 | Sum of squares Σ(srcᵢ²)       | 8x / 4x / 2x                        |
|                 | `Sum(a)`                            | Sum of elements               | 8x / 4x / 2x                        |
|                 | `Min(a)`                            | Minimum value                 | 8x / 4x / 2x                        |
|                 | `Max(a)`                            | Maximum value                 | 8x / 4x / 2x                        |
|                 | `MinIdx(a)`                         | Index of minimum value        | Pure Go                             |
|                 | `MaxIdx(a)`                         | Index of maximum value        | Pure Go                             |
| **Statistical** | `Mean(a)`                           | Arithmetic mean               | 8x / 4x / 2x                        |
|                 | `Variance(a)`                       | Population variance           | 8x / 4x / 2x                        |
|                 | `StdDev(a)`                         | Standard deviation            | 8x / 4x / 2x                        |
| **Vector**      | `EuclideanDistance(a, b)`           | L2 distance                   | 8x / 4x / 2x                        |
|                 | `Normalize(dst, a)`                 | Unit vector normalization     | 8x / 4x / 2x                        |
|                 | `CumulativeSum(dst, a)`             | Running sum                   | Sequential                          |
| **Range**       | `Clamp(dst, a, min, max)`           | Clamp to range                | 8x / 4x / 2x                        |
| **Activation**  | `Sigmoid(dst, src)`                 | Sigmoid: 1/(1+e^-x)           | 4x (AVX2) / 2x (NEON)               |
|                 | `ReLU(dst, src)`                    | Rectified Linear Unit         | 4x (AVX) / 2x (NEON)                |
|                 | `Tanh(dst, src)`                    | Hyperbolic tangent            | 4x (AVX2) / 2x (NEON)               |
|                 | `Exp(dst, src)`                     | Exponential e^x               | 4x (AVX2) / 2x (NEON)               |
|                 | `ClampScale(dst, src, min, max, s)` | Fused clamp and scale         | 4x (AVX) / 2x (NEON)                |
| **Transcendental** | `Log(dst, src)`                  | Natural log ln(x)             | 4x (AVX2+FMA) / 2x (NEON)           |
|                 | `Log2(dst, src)` / `Log10(dst, src)`| Base-2 / base-10 log          | 4x (AVX2+FMA) / 2x (NEON)           |
|                 | `Pow(dst, src, exp)`                | Scalar power x^exp (PCEN, dB) | 4x (AVX2+FMA) / 2x (NEON)           |
|                 | `PowElem(dst, base, exp)`           | Elementwise base^exp          | 4x (AVX2+FMA) / 2x (NEON)           |
| **Batch**       | `DotProductBatch(r, rows, v)`       | Multiple dot products         | 8x / 4x / 2x                        |
| **Signal**      | `ConvolveValid(dst, sig, k)`        | FIR filter / convolution      | 8x / 4x / 2x                        |
|                 | `ConvolveValidMulti(dsts, sig, ks)` | Multi-kernel convolution      | 8x / 4x / 2x                        |
|                 | `ConvolveDecimate(dst,sig,k,f,p)`   | Strided FIR downsample (decimate) | 8x / 4x / 2x                    |
|                 | `AccumulateAdd(dst, src, off)`      | Overlap-add: dst[off:] += src | 8x / 4x / 2x                        |
|                 | `Autocorrelate(autoc, x, maxLag)`   | LPC autocorrelation Σ x[i]·x[i-lag] (bit-exact) | 4x (AVX2) / 2x (NEON)     |
| **Audio**       | `Interleave2(dst, a, b)`            | Pack stereo: [L,R,L,R,...]    | 4x / 2x                             |
|                 | `Deinterleave2(a, b, src)`          | Unpack stereo to channels     | 4x / 2x                             |
|                 | `InterleaveN(dst, srcs)`            | Pack N planar streams (any N; N-stream Interleave2) | N=2,4,8 AVX, N=3,6 AVX2 / N=2,3,4 NEON; else Go |
|                 | `DeinterleaveN(dsts, src)`          | Unpack N interleaved streams (any N) | N=2,4,8 AVX, N=3,6 AVX2 / N=2,3,4 NEON; else Go |
|                 | `CubicInterpDot(hist,a,b,c,d,x)`    | Fused cubic interp dot product| 4x / 2x                             |

`DotProductBatch` scores its `[][]float64` rows in groups of four, keeping the
query vector resident in registers across each group via a fused 4-row kernel on
AMD64 (AVX-512 and AVX+FMA) and ARM64 NEON instead of re-loading it per row.
Short, ragged, or sub-SIMD-width rows fall back to the per-row dot product, with
identical results.

`Autocorrelate` computes the LPC autocorrelation `autoc[lag] = Σ x[i]·x[i-lag]`
used by FLAC-style encoders. It vectorizes across lags (one accumulator lane per
lag, never fusing the multiply-add), so each lag's sum keeps the exact left-to-right
order of the scalar loop and every build emits byte-identical results to the pure-Go
reference. The AVX2 path accumulates four lags per YMM, NEON two lags per V register;
non-AVX2/NEON CPUs and short blocks use the scalar reference.

#### STFT (fused real-input short-time Fourier transform)

`STFTPlan` is the spectral front-end's missing middle: the library already covers
the post-FFT power spectrum (`c128.AbsSq`), mel projection (`DotProductBatch`),
and PCEN / log-mel normalization (`Exp`, `Mul`, `Log`), but not the transform.

Both `f64` and `f32` provide it (with `complex64` output for `f32`).

```go
plan, _ := f64.NewSTFTPlan(1024)               // power-of-two nfft; reuse across calls
bins := plan.NumBins()                         // nfft/2 + 1 (Hermitian half-spectrum)
nFrames := plan.NumFrames(len(signal), hop, f64.PadZero)

spec := make([][]complex128, nFrames)          // caller-owned output, one row per frame
for i := range spec { spec[i] = make([]complex128, bins) }
plan.STFT(spec, signal, hann, hop, f64.PadZero) // fills spec; returns frames written

// Flat, frame-contiguous power (stride NumBins) feeds DotProductBatch directly
// as a mel-filterbank projection, with no per-frame allocation:
power := make([]float64, nFrames*bins)
plan.STFTPowerInto(power, signal, hann, hop, f64.PadZero)
for f := range nFrames {
    f64.DotProductBatch(mel[f], filterbank, power[f*bins:(f+1)*bins])
}
```

The transform uses a half-length complex FFT (rfft, ~2x cheaper than a full
complex FFT), keeps the twiddle/bit-reversal plan resident, and fuses the window
multiply into the frame pack (and the `|.|^2` power step in `STFTPower` /
`STFTPowerInto`). The `PadMode` argument selects the framing convention: `NoPad`
is the no-padding case (frame `f` is `signal[f*hop : f*hop+nfft]`, matching
librosa `stft(..., center=False)`), while `PadZero` and `PadReflect` center each
frame with `nfft/2` of zero or reflect padding per side, matching librosa
`center=True` (`pad_mode="constant"` / `"reflect"`). `NumFrames` reports the frame
count for a given pad mode so you can size buffers. The centered output is pinned
against a librosa golden vector in the tests. The plan is allocation-free across
calls; a plan holds transform scratch, so use one plan per goroutine. This first
cut is a correct scalar radix-2 transform (power-of-two `nfft`); vectorizing the
inner butterfly is a profile-gated follow-up.

### `f32` - float32 Operations

Same API as `f64` but for `float32` with wider SIMD.

**Scope:** `f32` carries the audio/FFT/ML surface on top of the shared arithmetic
API: PCM sample-format conversions, split-format complex operations, and the
indexed/strided dot products (`DotProductIndexed`, `DotProductStrided`) used by
streaming DSP. These are f32-specific and have no f64 equivalent by design.

| Architecture    | SIMD Width  |
| --------------- | ----------- |
| AMD64 (AVX-512) | 16x float32 |
| AMD64 (AVX+FMA) | 8x float32  |
| AMD64 (SSE2)    | 4x float32  |
| ARM64 (NEON)    | 4x float32  |

**PCM conversion** (audio sample-format conversion, f32-specific; no f64 equivalent):

| Function | Description | SIMD Width |
| --- | --- | --- |
| `Int32ToFloat32Scale(dst, src, s)` | PCM int32 to normalized float | 8x (AVX2) / 4x (NEON) |
| `Int16ToFloat32Scale(dst, src, s)` | PCM int16 to normalized float | 8x (AVX2) / 4x (NEON) |
| `Float32ToInt16Scale(dst, src, s)` | Normalized float to PCM int16 | 8x (AVX2) / 4x (NEON) |

Each has an `Unsafe` variant that skips bounds reconciliation.

`InterleaveN`/`DeinterleaveN` add an 8-stream AVX path (8x8 register transpose) and
a 3-stream AVX2 path (per-stream `VPERMPS` gathers merged with `VPBLENDD`, since 3
streams do not map onto a clean register transpose) on top of the shared N=2/4 (AVX)
and N=2/3/4/6/8 (NEON) kernels; all other stream counts use the allocation-free
generic path. The N=3 case is the 16k -> 48k upsample hot path: the AVX2
gather/blend kernel runs roughly 2.8x (interleave) and 3.2x (deinterleave) over the
generic loop on AVX2. The ARM64 N=6 (5.1 audio) and N=8 (7.1 audio) NEON kernels zip
adjacent channel pairs at `.4S` so each 64-bit lane holds a frame pair, then store
with `ST3`/`ST4` at `.2D` (the inverse via `LD3`/`LD4` plus `UZP1`/`UZP2`); they run
roughly 4.4x (N=6) and 3.4x-4.6x (N=8) over the generic loop on the Raspberry Pi 5.
The 6-stream AVX2 path (the 8k -> 48k upsample) zips stream pairs into three
double-wide pair streams, then reuses the f64 N=3 interleave on those pairs, so it
needs no index tables; it runs roughly 2x (interleave and deinterleave) on AVX2.
`f64` adds N=3 and N=6 (AVX2) plus N=8 (AVX), processing 4 frames per block (a YMM holds
4 doubles): N=3 uses immediate `VPERMPD` gathers merged with `VBLENDPD`, N=6 zips pairs
at 128-bit-lane granularity with `VPERM2F128` (roughly 4x interleave, 1.5x deinterleave),
and N=8 runs two stacked 4x4 transposes (streams 0-3 fill each frame's low YMM, streams
4-7 the high YMM).

**Row-major batch dot products** (for flat vector stores):

| Function | Description |
| --- | --- |
| `DotProductIndexed(dst, base, query, rowIDs, dims) bool` | Scores selected row-major rows by `uint32` row ID without building `[][]float32`; returns whether an optimized SIMD batch kernel handled at least one batch. |
| `DotProductStrided(dst, base, query, rowCount, dims, stride) bool` | Scores contiguous or fixed-stride row-major rows; returns whether an optimized SIMD batch kernel handled at least one batch. |

Both APIs are allocation-free. The batched SIMD kernel covers AMD64 (AVX-512 / AVX+FMA) and ARM64 (NEON); unsupported CPUs, tiny shapes, tails, and ragged inputs use the per-row fallback.

`DotProductBatch` scores its `[][]float32` rows in groups of four, keeping the
query vector resident in registers across each group instead of re-loading it
for every row. The fused 4-row kernel runs on AVX-512, AVX+FMA, and ARM64 NEON;
short, ragged, or sub-SIMD-width rows fall back to the per-row dot product.
Results are identical to the per-row path either way.

**Additional split-format complex operations** (for FFT pipelines with separate real/imag arrays):

| Category   | Function                              | Description                        | SIMD Width       |
| ---------- | ------------------------------------- | ---------------------------------- | ---------------- |
| **Complex**| `MulComplex(dstRe,dstIm,aRe,aIm,bRe,bIm)` | Split-format complex multiply  | 8x (AVX) / 4x (NEON) |
|            | `MulConjComplex(dstRe,dstIm,aRe,aIm,bRe,bIm)` | Multiply by conjugate      | 8x / 4x          |
|            | `AbsSqComplex(dst,aRe,aIm)`           | Magnitude squared                  | 8x / 4x          |
|            | `ButterflyComplex(uRe,uIm,lRe,lIm,twRe,twIm)` | FFT butterfly with twiddle | 8x / 4x          |
|            | `RealFFTUnpack(outRe,outIm,zRe,zIm,twRe,twIm)` | Real FFT unpack step     | 8x / 4x          |
| **Utility**| `Reverse(dst, src)`                   | Reverse slice order                | 8x / 4x          |
|            | `AddSub(sum, diff, a, b)`             | Fused sum and difference           | 8x / 4x          |

### `f16` - float16 (Half-Precision) Operations

IEEE 754 half-precision floating-point operations, optimized for ML inference, audio DSP, and memory-bandwidth-bound workloads.

`Float16` is a storage type. On ARM64 the full operation set runs on NEON; on AMD64 the `ToFloat32Slice`/`FromFloat32Slice` conversions use F16C hardware instructions (`VCVTPH2PS`/`VCVTPS2PH`, available on every AVX2-capable x86 since 2012) while the other ops use the pure-Go reference (x86 has no half-precision arithmetic outside AVX512-FP16).

```go
import "github.com/tphakala/simd/f16"

// Convert between float32 and float16
h := f16.FromFloat32(3.14)
f := f16.ToFloat32(h)

// Vector operations (same API as f32/f64)
a := make([]f16.Float16, 1024)
b := make([]f16.Float16, 1024)
dst := make([]f16.Float16, 1024)

f16.Add(dst, a, b)           // Element-wise addition
dot := f16.DotProduct(a, b)  // Dot product (returns float32)
f16.ReLU(dst, a)             // Activation functions
```

| Category        | Function                            | Description                   | SIMD Width       |
| --------------- | ----------------------------------- | ----------------------------- | ---------------- |
| **Conversion**  | `ToFloat32(h)`                      | FP16 → float32                | Scalar           |
|                 | `FromFloat32(f)`                    | float32 → FP16                | Scalar           |
|                 | `ToFloat32Slice(dst, src)`          | Batch FP16 → float32          | 8x (F16C) / 8x (NEON+FP16) |
|                 | `FromFloat32Slice(dst, src)`        | Batch float32 → FP16          | 8x (F16C) / 8x (NEON+FP16) |
| **Arithmetic**  | `Add(dst, a, b)`                    | Element-wise addition         | 8x (NEON+FP16)   |
|                 | `Sub(dst, a, b)`                    | Element-wise subtraction      | 8x (NEON+FP16)   |
|                 | `Mul(dst, a, b)`                    | Element-wise multiplication   | 8x (NEON+FP16)   |
|                 | `Div(dst, a, b)`                    | Element-wise division         | 8x (NEON+FP16)   |
|                 | `Scale(dst, a, s)`                  | Multiply by scalar            | 8x (NEON+FP16)   |
|                 | `AddScalar(dst, a, s)`              | Add scalar                    | 8x (NEON+FP16)   |
|                 | `FMA(dst, a, b, c)`                 | Fused multiply-add: a*b+c     | 8x (NEON+FP16)   |
|                 | `AddScaled(dst, alpha, s)`          | dst += alpha*s (AXPY)         | 8x (NEON+FP16)   |
| **Unary**       | `Abs(dst, a)`                       | Absolute value                | 8x (NEON+FP16)   |
|                 | `Neg(dst, a)`                       | Negation                      | 8x (NEON+FP16)   |
|                 | `Sqrt(dst, a)`                      | Square root                   | 8x (NEON+FP16)   |
|                 | `Reciprocal(dst, a)`                | Reciprocal (1/x)              | 8x (NEON+FP16)   |
| **Reduction**   | `DotProduct(a, b)` → float32        | Dot product                   | 8x (NEON+FP16)   |
|                 | `DotProductF32(a, b)` → float32     | Dot product (FP32 widen)      | 8x (NEON)        |
|                 | `Sum(a)` → float32                  | Sum of elements               | 8x (NEON+FP16)   |
|                 | `Min(a)`                            | Minimum value                 | 8x (NEON+FP16)   |
|                 | `Max(a)`                            | Maximum value                 | 8x (NEON+FP16)   |
|                 | `MinIdx(a)`                         | Index of minimum              | Pure Go          |
|                 | `MaxIdx(a)`                         | Index of maximum              | Pure Go          |
| **Statistical** | `Mean(a)` → float32                 | Arithmetic mean               | 8x (NEON+FP16)   |
|                 | `Variance(a)` → float32             | Population variance           | 8x (NEON)        |
|                 | `StdDev(a)` → float32               | Standard deviation            | 8x (NEON)        |
| **Vector**      | `EuclideanDistance(a, b)` → float32 | L2 distance                   | 8x (NEON)        |
|                 | `Normalize(dst, a)`                 | Unit vector normalization     | 8x (NEON+FP16)   |
|                 | `CumulativeSum(dst, a)`             | Running sum                   | Sequential       |
| **Range**       | `Clamp(dst, a, min, max)`           | Clamp to range                | 8x (NEON+FP16)   |
|                 | `ClampScale(dst, src, min, max, s)` | Fused clamp and scale         | 8x (NEON)        |
| **Activation**  | `ReLU(dst, src)`                    | Rectified Linear Unit         | 8x (NEON+FP16)   |
|                 | `Sigmoid(dst, src)`                 | Sigmoid: 1/(1+e^-x)           | Pure Go          |
|                 | `Tanh(dst, src)`                    | Hyperbolic tangent            | Pure Go          |
|                 | `Exp(dst, src)`                     | Exponential e^x               | Pure Go          |
| **Batch**       | `DotProductBatch(r, rows, v)`       | Multiple dot products         | 8x (NEON+FP16)   |
| **Signal**      | `ConvolveValid(dst, sig, k)`        | FIR filter / convolution      | Pure Go          |
|                 | `AccumulateAdd(dst, src, off)`      | Overlap-add: dst[off:] += src | 8x (NEON+FP16)   |
| **Audio**       | `Interleave2(dst, a, b)`            | Pack stereo: [L,R,L,R,...]    | 8x (NEON)        |
|                 | `Deinterleave2(a, b, src)`          | Unpack stereo to channels     | 8x (NEON)        |

**Key characteristics:**

- **Storage**: IEEE 754 half-precision (1 sign, 5 exponent, 10 mantissa bits)
- **Precision**: ~3.3 decimal digits, range ~6×10⁻⁸ to 65504
- **Reductions**: Accumulate in float32 for numerical stability
- **Memory efficiency**: 2x bandwidth vs float32 (8 elements per 128-bit NEON vector)
- **DotProduct saturation**: On ARM64 with FP16 SIMD, `DotProduct` computes per-element products in FP16 and saturates to ±Inf when `|a[i] * b[i]| > 65504`. Use `DotProductF32` (FP32 widening before multiply, ~1.5-2x slower) for audio DSP or raw-signal inputs that can produce out-of-range products.
- **FP32-widened ops**: `DotProductF32`, `EuclideanDistance`, `Variance`, `StdDev`, and `ClampScale` widen each FP16 lane to FP32 before arithmetic, so they match the pure-Go reference and never saturate. They use only base-NEON instructions (the `FCVTL`/`FCVTN` conversions are ARMv8.0-A, not the FEAT_FP16 extension), so they run on any ARM64 NEON core, including non-FP16 parts (Cortex-A72/A53). `Interleave2`/`Deinterleave2` are likewise bit-exact 16-bit lane permutes (`ZIP`/`UZP`) that run on any ARM64 NEON core.

**Benchmark (1024 elements, Raspberry Pi 5 / Cortex-A76, zero allocations):**

| Operation         | SIMD   | Pure Go  | Speedup   |
| ----------------- | ------ | -------- | --------- |
| EuclideanDistance | 481 ns | 5996 ns  | **12.5x** |
| Variance          | 506 ns | 8971 ns  | **17.7x** |
| Interleave2       | 177 ns | 2159 ns  | **12.2x** |
| Deinterleave2     | 177 ns | 2166 ns  | **12.2x** |
| ClampScale        | 531 ns | 12788 ns | **24.1x** |

**Hardware requirements:**

- **Native FP16 SIMD**: ARM64 with FEAT_FP16 (ARMv8.2-A+)
  - Apple Silicon (M1/M2/M3/M4) ✅
  - Cortex-A55, A75, A76, A77, A78, X1, X2, X3 ✅
  - Raspberry Pi 5 (Cortex-A76) ✅
- **Pure Go fallback**: All other platforms
  - Raspberry Pi 3/4 (Cortex-A53/A72 - ARMv8.0) - works but no SIMD acceleration
  - AMD64 - works but no SIMD acceleration

### `c128` - complex128 Operations

SIMD-accelerated complex number operations for FFT-based signal processing.

**Scope:** `c64`/`c128` are deliberately small, FFT-pipeline helper sets (multiply,
conjugate-multiply, dot/Hermitian products, scale, add/sub, abs/absSq, conj). They
are not a general complex-arithmetic surface; operations outside the FFT pipeline
are intentionally absent.

| Category       | Function             | Description                        | SIMD Width              |
| -------------- | -------------------- | ---------------------------------- | ----------------------- |
| **Arithmetic** | `Mul(dst, a, b)`     | Complex multiplication             | 4x (AVX-512) / 2x (AVX) |
|                | `MulConj(dst, a, b)` | Multiply by conjugate: a × conj(b) | 4x / 2x                 |
|                | `Scale(dst, a, s)`   | Scale by complex scalar            | 4x / 2x                 |
|                | `Add(dst, a, b)`     | Complex addition                   | 4x / 2x                 |
|                | `Sub(dst, a, b)`     | Complex subtraction                | 4x / 2x                 |
| **Reduction**  | `DotProduct(a, b)`     | Complex dot product sum(a·b)       | 2x (AVX) / 1x (SSE2, NEON) |
|                | `DotProductConj(a, b)` | Hermitian inner product sum(a·conj(b)) | 2x (AVX) / 1x (SSE2, NEON) |
| **Unary**      | `Abs(dst, a)`        | Complex magnitude \|a + bi\|       | 4x (AVX-512) / 2x (AVX) |
|                | `AbsSq(dst, a)`      | Magnitude squared \|a + bi\|²      | 4x / 2x                 |
|                | `Conj(dst, a)`       | Complex conjugate: a - bi          | 4x / 2x                 |
| **Conversion** | `FromReal(dst, src)` | Real to complex: src → src+0i      | 2x (AVX-512/AVX) / 2x (NEON) |

These operations are designed for FFT-based signal processing pipelines:

```go
import "github.com/tphakala/simd/c128"

// Frequency-domain multiplication (FFT convolution)
signalFFT := make([]complex128, n)
kernelFFT := make([]complex128, n)
result := make([]complex128, n)
magnitude := make([]float64, n)

// Frequency-domain filtering
c128.Mul(result, signalFFT, kernelFFT)          // Complex multiply
c128.MulConj(result, signalFFT, kernelFFT)      // Cross-correlation

// Spectrogram and magnitude analysis
c128.Abs(magnitude, signalFFT)                  // Extract magnitude for display
```

**Use Cases:**

- **Abs/AbsSq**: Spectrograms, power spectral density, frequency analysis
- **Conj**: Cross-correlation, frequency-domain filtering
- **Mul/MulConj**: FFT-based convolution, filtering, correlation

**Benchmark (1024 elements, Intel Core i7-1260P, AVX+FMA):**

| Operation | SIMD    | Pure Go | Speedup   |
| --------- | ------- | ------- | --------- |
| Mul       | 252 ns  | 679 ns  | **2.7x**  |
| MulConj   | 260 ns  | 723 ns  | **2.8x**  |
| Scale     | 193 ns  | 643 ns  | **3.3x**  |
| Add       | 165 ns  | 461 ns  | **2.8x**  |
| Abs       | 661 ns  | 2252 ns | **3.4x**  |
| AbsSq     | 228 ns  | 430 ns  | **1.9x**  |
| Conj      | 125 ns  | 405 ns  | **3.2x**  |

### `c64` - complex64 Operations

SIMD-accelerated single-precision complex number operations. Like `c128`, this is
a deliberately small FFT-pipeline helper set (see the `c128` scope note). On amd64
the SIMD floor is SSE4.1 (the "SSE2" routines use `BLENDPS`), one tier above the
other float packages.

| Category       | Function             | Description                        | SIMD Width                        |
| -------------- | -------------------- | ---------------------------------- | --------------------------------- |
| **Arithmetic** | `Mul(dst, a, b)`     | Complex multiplication             | 8x (AVX-512) / 4x (AVX) / 2x (NEON) |
|                | `MulConj(dst, a, b)` | Multiply by conjugate: a × conj(b) | 8x / 4x / 2x                      |
|                | `Scale(dst, a, s)`   | Scale by complex scalar            | 8x / 4x / 2x                      |
|                | `Add(dst, a, b)`     | Complex addition                   | 8x / 4x / 2x                      |
|                | `Sub(dst, a, b)`     | Complex subtraction                | 8x / 4x / 2x                      |
| **Reduction**  | `DotProduct(a, b)`     | Complex dot product sum(a·b)       | 4x (AVX) / 2x (SSE, NEON) |
|                | `DotProductConj(a, b)` | Hermitian inner product sum(a·conj(b)) | 4x (AVX) / 2x (SSE, NEON) |
| **Unary**      | `Abs(dst, a)`        | Complex magnitude \|a + bi\|       | 8x / 4x / 2x                      |
|                | `AbsSq(dst, a)`      | Magnitude squared \|a + bi\|²      | 8x / 4x / 2x                      |
|                | `Conj(dst, a)`       | Complex conjugate: a - bi          | 8x / 4x / 2x                      |
| **Conversion** | `FromReal(dst, src)` | Real to complex: src → src+0i      | 8x / 4x / 2x                      |

Same API as `c128` but for `complex64` with 2x wider SIMD (8 bytes vs 16 bytes per element):

```go
import "github.com/tphakala/simd/c64"

// Single-precision FFT processing
signalFFT := make([]complex64, n)
kernelFFT := make([]complex64, n)
result := make([]complex64, n)
magnitude := make([]float32, n)

c64.Mul(result, signalFFT, kernelFFT)     // Complex multiply
c64.Abs(magnitude, signalFFT)              // Extract magnitude
```

### `i32` - int32 Operations

SIMD-accelerated integer-domain operations for integer-DSP hot loops, where the per-sample work is integer arithmetic and channel (de)interleaving rather than floating-point math:

| Category       | Function                   | Description                                            | SIMD Width            |
| -------------- | -------------------------- | ------------------------------------------------------ | --------------------- |
| **Interleave** | `Interleave2(dst, a, b)`   | Pack two channels into interleaved stereo              | 8x (AVX) / 4x (NEON)  |
|                | `Deinterleave2(a, b, src)` | Split interleaved stereo into two channels             | 8x (AVX) / 4x (NEON)  |
| **Arithmetic** | `Add(dst, a, b)`           | Element-wise add `dst = a + b`                         | 8x (AVX2) / 4x (NEON) |
|                | `Sub(dst, a, b)`           | Element-wise subtract `dst = a - b`                    | 8x (AVX2) / 4x (NEON) |
| **Reduction**  | `MinMax(res) (min, max)`   | Signed int32 per-slice minimum and maximum in one pass | 8x (AVX2) / 4x (NEON) |

```go
import "github.com/tphakala/simd/i32"

left := make([]int32, n)
right := make([]int32, n)
stereo := make([]int32, n*2)

i32.Interleave2(stereo, left, right)   // [l0, r0, l1, r1, ...]
i32.Deinterleave2(left, right, stereo) // inverse: split back to channels

dst := make([]int32, n)
i32.Add(dst, left, right) // element-wise dst = left + right
i32.Sub(dst, left, right) // element-wise dst = left - right

mn, mx := i32.MinMax(left) // smallest and largest value in one signed pass
```

Interleaving is pure 32-bit-lane movement, so those kernels reuse the proven `f32` shuffle/permute encodings (AVX `VUNPCKLPS`/`VPERM2F128`, NEON `ZIP`/`UZP` on `.4S`); the bit pattern of each lane is irrelevant, so negative values and the type extremes round-trip exactly. `Add` and `Sub` do element-wise integer-ALU work on 256-bit (AVX2) / 128-bit (NEON) lanes with two's-complement wraparound, so they are bit-identical to the pure-Go reference across the full int32 range. `MinMax` returns the smallest and largest int32 in one signed pass (`VPMINSD`/`VPMAXSD` on AVX2, `SMIN`/`SMAX` with single-instruction `SMINV`/`SMAXV` folds on NEON); since min/max of int32 has no accumulation order, the SIMD paths are bit-identical to the pure-Go reference by construction (~10x AVX2, ~5x NEON). All zero-allocation.

> The FLAC-specific integer kernels (fixed predictors, quantized-LPC residual/restore, mid/side decorrelation, and the Rice cost search) that previously lived here now live in the codec that owns them ([go-flac](https://github.com/tphakala/go-flac)); this package keeps only the generic integer ops above.

### `i16` - int16 Operations

The 16-bit integer counterpart to `i32`, for raw-PCM hot loops where the source samples are 16-bit and the cheapest place to vectorize is the channel (de)interleaving that happens before samples are widened to int32. Inter-channel decorrelation can exceed the source bit depth by one bit, so arithmetic is done after widening to `i32`; this package carries only the operations that provably help at 16-bit width:

**Scope:** `i16` is deliberately movement-only (interleave/deinterleave). There are
no int16 arithmetic primitives on purpose: widen to `i32` and use its arithmetic
surface, because 16-bit arithmetic overflows as soon as channels are decorrelated.

| Category       | Function                   | Description                                | SIMD Width                         |
| -------------- | -------------------------- | ------------------------------------------ | ---------------------------------- |
| **Interleave** | `Interleave2(dst, a, b)`   | Pack two channels into interleaved stereo  | 16x (AVX2) / 8x (SSE2) / 8x (NEON) |
|                | `Deinterleave2(a, b, src)` | Split interleaved stereo into two channels | 16x (AVX2) / 8x (SSE2) / 8x (NEON) |

```go
import "github.com/tphakala/simd/i16"

left := make([]int16, n)
right := make([]int16, n)
stereo := make([]int16, n*2)

i16.Interleave2(stereo, left, right)   // [l0, r0, l1, r1, ...]
i16.Deinterleave2(left, right, stereo) // inverse: split back to channels
```

Like the `i32` interleave kernels, these are pure 16-bit-lane movement (AVX2/SSE2 word unpacks plus a lane permute, NEON `ZIP`/`UZP` on `.8H`), so the bit pattern of each lane is irrelevant and every value round-trips exactly: negative values and the int16 extremes are preserved. Both kernels are zero-allocation.

### `i8` - int8 Operations

SIMD-accelerated int8 operations for quantized numeric pipelines. The narrow `-128..127` range makes element-wise arithmetic overflow almost immediately, so this package does not mirror the wrapping arithmetic of `i16`/`i32`. It ships the operations that are genuinely high-impact and well-defined at 8-bit width: saturating arithmetic, element-wise min/max/clamp and saturating abs/neg/abs-diff, int32-accumulated reductions, signed min/max, the per-tensor abs-max for dynamic quantization, and sign-extending widening.

| Category       | Function                   | Description                                                    | SIMD Width             |
| -------------- | -------------------------- | -------------------------------------------------------------- | ---------------------- |
| **Arithmetic** | `AddSaturate(dst, a, b)`   | Element-wise add, clamped to `[-128, 127]`                     | 32x (AVX2) / 16x (NEON)|
|                | `SubSaturate(dst, a, b)`   | Element-wise subtract, clamped to `[-128, 127]`                | 32x (AVX2) / 16x (NEON)|
|                | `AddScalarSaturate(dst, a, s)`| Add a scalar, clamped to `[-128, 127]`                      | 32x (AVX2) / 16x (NEON)|
|                | `SubScalarSaturate(dst, a, s)`| Subtract a scalar, clamped to `[-128, 127]`                 | 32x (AVX2) / 16x (NEON)|
| **Element-wise** | `Min(dst, a, b)`         | Element-wise signed minimum of two slices                      | 32x (AVX2) / 16x (NEON)|
|                | `Max(dst, a, b)`           | Element-wise signed maximum of two slices                      | 32x (AVX2) / 16x (NEON)|
|                | `Clamp(dst, src, lo, hi)`  | Clamp each element to `[lo, hi]` (activation clipping)         | 32x (AVX2) / 16x (NEON)|
|                | `Abs(dst, a)`              | Saturating absolute value (`abs(-128) = 127`)                  | 32x (AVX2) / 16x (NEON)|
|                | `Neg(dst, a)`              | Saturating negation (`neg(-128) = 127`)                        | 32x (AVX2) / 16x (NEON)|
|                | `AbsDiff(dst, a, b)`       | Saturating `\|a - b\|`, clamped to `[0, 127]`                  | 32x (AVX2) / 16x (NEON)|
| **Widening**   | `ToInt16(dst, src)`        | Sign-extend `int8` to `int16`                                  | 16x (AVX2) / 16x (NEON)|
|                | `ToInt32(dst, src)`        | Sign-extend `int8` to `int32`                                  | 8x (AVX2) / 8x (NEON)  |
| **Reduction**  | `Sum(a) int32`             | int32-accumulated sum                                          | 16x (AVX2) / 16x (NEON)|
|                | `DotProduct(a, b) int32`   | int32-accumulated dot product (quantized matmul inner loop)    | 16x (AVX2) / 16x (NEON, SDOT)|
|                | `MinMax(a) (min, max)`     | Signed int8 per-slice minimum and maximum in one pass          | 32x (AVX2) / 16x (NEON)|
|                | `MaxAbs(a) int`            | Per-tensor abs-max (dynamic-quantization scale), range `[0,128]`| 32x (AVX2) / 16x (NEON)|

```go
import "github.com/tphakala/simd/i8"

a := []int8{ /* ... */ }
b := []int8{ /* ... */ }

dst := make([]int8, len(a))
i8.AddSaturate(dst, a, b)      // saturating dst = clamp(a + b, -128, 127)
i8.SubSaturate(dst, a, b)      // saturating dst = clamp(a - b, -128, 127)
i8.AddScalarSaturate(dst, a, 8) // saturating dst = clamp(a + 8, -128, 127)
i8.SubScalarSaturate(dst, a, 8) // saturating dst = clamp(a - 8, -128, 127)

i8.Min(dst, a, b)         // element-wise signed min
i8.Max(dst, a, b)         // element-wise signed max
i8.Clamp(dst, a, -64, 64) // clamp each element to [-64, 64]
i8.Abs(dst, a)            // saturating |a|, abs(-128) = 127
i8.Neg(dst, a)            // saturating -a, neg(-128) = 127
i8.AbsDiff(dst, a, b)     // saturating |a - b|, clamped to [0, 127]

dot := i8.DotProduct(a, b) // int32-accumulated sum(a[i]*b[i])
sum := i8.Sum(a)           // int32-accumulated sum
mn, mx := i8.MinMax(a)     // smallest and largest value in one signed pass
scale := i8.MaxAbs(a)      // per-tensor abs-max for dynamic quantization

w16 := make([]int16, len(a))
i8.ToInt16(w16, a) // sign-extend to int16 (exact)
```

`AddSaturate`/`SubSaturate` (and the scalar-broadcast `AddScalarSaturate`/`SubScalarSaturate`) use single saturating instructions (`VPADDSB`/`VPSUBSB` on AVX2, `SQADD`/`SQSUB` on NEON) and clamp instead of wrapping, which is what 8-bit arithmetic almost always wants. The element-wise group is single-instruction too: `Min`/`Max` map to `VPMINSB`/`VPMAXSB` (`SMIN`/`SMAX` on NEON), `Clamp` broadcasts the bounds and applies max-then-min, and `Abs`/`Neg` saturate so `-128` maps to `127` (`SQABS`/`SQNEG` on NEON; `max(a, saturating(0-a))` and `saturating(0-a)` on AVX2). `AbsDiff` saturates `|a - b|` to `[0, 127]` (`SABD` then an unsigned min with 127 on NEON; `max(saturating(a-b), saturating(b-a))` on AVX2), and `MaxAbs` returns the per-tensor abs-max as `int` (range `[0, 128]`, since `|-128| = 128` does not fit `int8`) via `PABSB`+unsigned `PMAXUB` on AVX2 and `ABS`+`UMAXV` on NEON, which is the scale a dynamic quantizer needs. `Sum` and `DotProduct` accumulate in int32 with two's-complement wraparound; since int32 wrapping addition is associative, the lane-parallel SIMD reductions are bit-identical to the scalar reference regardless of summation order, and the int8 products never overflow their lane (`|int8 * int8| <= 16384`). `DotProduct` is the inner loop of quantized matmul/convolution: on AVX2 it widens with `VPMOVSXBW` and reduces with `VPMADDWD`; on ARM64 with `FEAT_DotProd` it uses `SDOT` (16 multiply-accumulates per instruction), falling back to a `SMULL`/`SADALP` base-NEON path on cores without it. All operations are zero-allocation and bit-exact against the pure-Go reference.

> **Planned follow-ups:** `float32 <-> int8` affine `Quantize`/`Dequantize` (scale + zero-point), an AVX-512 VNNI (`VPDPBUSD`) `DotProduct` fast path, and 8-bit channel `Interleave2`/`Deinterleave2`.

## Performance

### AMD64 (Intel Core i7-1260P, AVX+FMA)

#### float64 Operations - SIMD vs Pure Go (1024 elements)

| Category        | Operation         | SIMD (ns) | Go (ns) | Speedup  |
| --------------- | ----------------- | --------- | ------- | -------- |
| **Arithmetic**  | Add               | 88        | 210     | **2.4x**  |
|                 | Sub               | 87        | 211     | **2.4x**  |
|                 | Mul               | 87        | 210     | **2.4x**  |
|                 | Div               | 459       | 899     | **2.0x**  |
|                 | Scale             | 86        | 237     | **2.8x**  |
|                 | AddScalar         | 76        | 235     | **3.1x**  |
|                 | FMA               | 120       | 470     | **3.9x**  |
| **Unary**       | Abs               | 71        | 246     | **3.5x**  |
|                 | Neg               | 74        | 235     | **3.2x**  |
|                 | Sqrt              | 690       | 1388    | **2.0x**  |
|                 | Reciprocal        | 513       | 938     | **1.8x**  |
| **Reduction**   | DotProduct        | 54        | 887     | **16.5x** |
|                 | Sum               | 35        | 427     | **12.1x** |
|                 | Min               | 148       | 350     | **2.4x**  |
|                 | Max               | 151       | 370     | **2.5x**  |
| **Statistical** | Mean              | 33        | 419     | **12.7x** |
|                 | Variance\*        | 552       | 3893    | **7.1x**  |
|                 | StdDev\*          | 556       | 3900    | **7.0x**  |
| **Vector**      | EuclideanDistance | 76        | 1173    | **15.4x** |
|                 | Normalize         | 536       | 692     | **1.3x**  |
|                 | CumulativeSum     | 472       | 457     | 1.0x      |
| **Range**       | Clamp             | 83        | 880     | **10.6x** |

\*Variance/StdDev benchmarked at 4096 elements (SIMD benefits at larger sizes)

#### float32 Operations - SIMD vs Pure Go (1024 elements)

| Category       | Operation  | SIMD (ns) | Go (ns) | Speedup   |
| -------------- | ---------- | --------- | ------- | --------- |
| **Arithmetic** | Add        | 61        | 287     | **4.7x**  |
|                | Sub        | 48        | 205     | **4.3x**  |
|                | Mul        | 49        | 206     | **4.2x**  |
|                | Div        | 137       | 664     | **4.8x**  |
|                | Scale      | 43        | 229     | **5.3x**  |
|                | AddScalar  | 36        | 228     | **6.3x**  |
|                | FMA        | 60        | 290     | **4.9x**  |
| **Unary**      | Abs        | 40        | 250     | **6.2x**  |
|                | Neg        | 82        | 471     | **5.8x**  |
| **Reduction**  | DotProduct | 32        | 426     | **13.3x** |
|                | Sum        | 18        | 416     | **22.6x** |
|                | Min        | 66        | 347     | **5.2x**  |
|                | Max        | 120       | 382     | **3.2x**  |
| **Statistical**| Variance\*  | 164       | 921     | **5.6x**  |
|                | StdDev\*    | 164       | 903     | **5.5x**  |
| **Vector**     | EuclideanDistance\* | 35 | 434     | **12.4x** |
| **Range**      | Clamp      | 45        | 753     | **16.6x** |

\*Variance/StdDev/EuclideanDistance use their own fixed 1000-element benchmark
(the other rows are at 1024 elements); all numbers come from one run on this host.

#### Activation Functions - SIMD vs Pure Go

**float32 (1024 elements):**

| Function   | SIMD (ns) | Go (ns)  | Speedup    | SIMD Throughput |
| ---------- | --------- | -------- | ---------- | --------------- |
| Sigmoid    | 348       | 5826     | **17x**    | 23.5 GB/s       |
| ReLU       | 36        | 480      | **13x**    | 226 GB/s        |
| Tanh       | 385       | 28219    | **73x**    | 21.3 GB/s       |
| Exp        | 264       | 5123     | **19x**    | 31.0 GB/s       |

**float64 (1024 elements):**

| Function   | SIMD (ns) | Go (ns)  | Speedup    | SIMD Throughput |
| ---------- | --------- | -------- | ---------- | --------------- |
| Sigmoid    | 745       | 5367     | **7.2x**   | 22.0 GB/s       |
| ReLU       | 79        | 537      | **6.8x**   | 240 GB/s        |
| Tanh       | 894       | 6600     | **7.4x**   | 18.3 GB/s       |
| Exp        | 622       | 4848     | **7.8x**   | 26.4 GB/s       |

**Key Characteristics:**

- **Tanh**: 73x speedup for f32 - fast approximation with saturation vs the slow math.Tanh
- **ReLU**: Highest throughput (226-240 GB/s) - simple max(0, x) operation
- **Sigmoid**: 17x speedup for f32 - fast approximation with exponential
- **Exp**: 19x speedup for f32 (12x on ARM64 NEON) via range reduction plus a degree-5 polynomial; max relative error ~7e-6 (f32), ~3e-6 (f64)

#### Batch & Signal Processing (varied sizes)

| Operation                | Config                | SIMD    | Go      | Speedup  |
| ------------------------ | --------------------- | ------- | ------- | -------- |
| DotProductBatch (f64)    | 256 vec × 100 rows    | 1.3 µs  | 22.0 µs | **16.4x**|
| DotProductBatch (f32)    | 256 vec × 100 rows    | 0.73 µs | 9.6 µs  | **13.2x**|
| ConvolveValid (f64)      | 4096 sig × 64 ker     | 25.3 µs | 198 µs  | **7.8x** |
| ConvolveValid (f32)      | 4096 sig × 64 ker     | 17.6 µs | 79 µs   | **4.5x** |
| ConvolveValidMulti (f64) | 1000 sig × 64 ker × 2 | 10.5 µs | -       | -        |
| CubicInterpDot (f64)     | 241 taps              | 35 ns   | 300 ns  | **8.6x** |
| CubicInterpDot (f32)     | 241 taps              | 20 ns   | 201 ns  | **10.2x**|
| Int32ToFloat32Scale      | 1024 elements         | 45 ns   | 366 ns  | **8.2x** |
| Int32ToFloat32Scale      | 4096 elements         | 148 ns  | 1448 ns | **9.8x** |
| Int16ToFloat32Scale      | 1024 elements         | 51 ns   | 473 ns  | **9.2x** |
| Int16ToFloat32Scale      | 4096 elements         | 173 ns  | 1897 ns | **11.0x**|
| Float32ToInt16Scale      | 1024 elements         | 88 ns   | 1262 ns | **14.4x**|
| Float32ToInt16Scale      | 4096 elements         | 347 ns  | 5434 ns | **15.7x**|
| Interleave2 (f64)        | 1000 pairs            | 218 ns  | -       | -        |
| Deinterleave2 (f64)      | 1000 pairs            | 228 ns  | -       | -        |
| Interleave2 (f32)        | 1000 pairs            | 108 ns  | -       | -        |
| Deinterleave2 (f32)      | 1000 pairs            | 218 ns  | -       | -        |

#### ConvolveDecimate (fused strided convolution)

`ConvolveDecimate` fuses an FIR downsample loop into one call. The relevant
baseline is what a consumer writes today: a Go loop calling `DotProductUnsafe`
at each strided window (the inner dot is already SIMD). Both compute identical
results; the fused kernel removes the per-output call, dispatch and slice-header
overhead and keeps the kernel pointer resident, so the win is largest for short
kernels. Signal length 4096, allocation-free. Measured (AVX2 on x86-64, NEON on
a Raspberry Pi 5):

| Config              | f32 x86 | f64 x86 | f32 NEON | f64 NEON |
| ------------------- | ------- | ------- | -------- | -------- |
| 20 taps, 2x decimate  | **2.0x** | **2.2x** | **1.7x** | **2.0x** |
| 32 taps, 2x decimate  | **2.3x** | **2.2x** | **1.9x** | **1.7x** |
| 64 taps, 2x decimate  | **2.0x** | **1.9x** | **1.7x** | **1.3x** |
| 241 taps, 2x decimate | **1.6x** | **1.2x** | **1.2x** | **1.1x** |
| 241 taps, 4x decimate | **1.3x** | **1.2x** | **1.2x** | **1.1x** |

#### Autocorrelate (lag-vectorized LPC autocorrelation, f64)

`Autocorrelate` is the LPC autocorrelation step in a FLAC-style encoder, the
largest remaining single-core hotspot there. Vectorizing across lags keeps the
result byte-identical to the scalar reference while still beating it. Block size
4096, allocation-free, speedup over the pure-Go fallback (AVX2 on x86-64, NEON on
a Raspberry Pi 5):

| Config (n=4096)       | amd64 (AVX2) | arm64 (NEON) |
| --------------------- | ------------ | ------------ |
| maxLag 8              | **3.0x**     | **2.4x**     |
| maxLag 12             | **3.2x**     | **2.5x**     |
| maxLag 32             | **3.4x**     | **2.6x**     |

#### Performance Summary

| Package  | Average Speedup | Best         | Operations   |
| -------- | --------------- | ------------ | ------------ |
| **f32**  | **6.6x**        | 22.6x (Sum)        | 62 functions |
| **f64**  | **4.1x**        | 16.5x (DotProduct) | 51 functions |
| **c128** | **2.8x**        | 3.4x (Abs)         | 11 functions |
| **c64**  | **6.0x**        | 22.0x (Scale)      | 11 functions |

### ARM64 (Raspberry Pi 5, NEON)

#### float64 Operations

| Operation  | Size | Time   | Throughput |
| ---------- | ---- | ------ | ---------- |
| DotProduct | 128  | 47 ns  | 44 GB/s    |
| DotProduct | 1024 | 327 ns | 50 GB/s    |
| Add        | 1024 | 495 ns | 50 GB/s    |
| Mul        | 1024 | 495 ns | 50 GB/s    |
| FMA        | 1024 | 604 ns | 54 GB/s    |
| Sum        | 1024 | 435 ns | 19 GB/s    |
| Mean       | 1024 | 431 ns | 19 GB/s    |

#### float32 Operations

| Operation  | Size  | Time    | Throughput |
| ---------- | ----- | ------- | ---------- |
| DotProduct | 128   | 27 ns   | 38 GB/s    |
| DotProduct | 1024  | 167 ns  | 49 GB/s    |
| DotProduct | 16384 | 2.86 µs | 46 GB/s    |
| Add        | 1024  | 248 ns  | 50 GB/s    |
| Mul        | 1024  | 248 ns  | 50 GB/s    |
| FMA        | 1024  | 303 ns  | 54 GB/s    |

#### Comparison vs Pure Go

| Operation        | Size | SIMD   | Pure Go | Speedup  |
| ---------------- | ---- | ------ | ------- | -------- |
| DotProduct (f32) | 128  | 27 ns  | 112 ns  | **4.1x** |
| DotProduct (f32) | 1024 | 167 ns | 861 ns  | **5.2x** |
| DotProduct (f64) | 128  | 47 ns  | 111 ns  | **2.4x** |
| DotProduct (f64) | 1024 | 327 ns | 861 ns  | **2.6x** |
| Add (f32)        | 1024 | 248 ns | 863 ns  | **3.5x** |
| Sum (f32)        | 1024 | 220 ns | 862 ns  | **3.9x** |

### int32 (i32) - SIMD vs Pure Go (1000 elements)

| Operation     | AMD64 (AVX/AVX2)             | ARM64 (NEON, Pi 5)            |
| ------------- | ---------------------------- | ----------------------------- |
| Interleave2   | 110 ns vs 440 ns (**4.0x**)  | 321 ns vs 1682 ns (**5.2x**)  |
| Deinterleave2 | 217 ns vs 443 ns (**2.0x**)  | 322 ns vs 1684 ns (**5.2x**)  |
| MinMax        | 40 ns vs 431 ns (**10.7x**)  | 211 ns vs 1102 ns (**5.2x**)  |

### int16 (i16) - SIMD vs Pure Go (1000 elements)

| Operation     | AMD64 (AVX2/SSE2)           | ARM64 (NEON, Pi 5)            |
| ------------- | --------------------------- | ----------------------------- |
| Interleave2   | 53 ns vs 560 ns (**10.6x**) | 165 ns vs 2105 ns (**12.8x**) |
| Deinterleave2 | 54 ns vs 607 ns (**11.3x**) | 165 ns vs 2120 ns (**12.9x**) |

Both i16 kernels are zero-allocation and bit-exact against the pure-Go reference (verified with negative values and the int16 extremes); they move whole 16-bit lanes, so the bit pattern of each sample is irrelevant to correctness.

All int32 kernels are zero-allocation and bit-exact against the pure-Go reference (verified across the sign and high bits with negative values and the type extremes). The interleave kernels move whole 32-bit lanes, so the bit pattern of each sample is irrelevant to correctness. `Add` and `Sub` are element-wise integer-ALU ops with two's-complement wraparound, matching the scalar reference across the full int32 range. `MinMax` is exact by construction (signed min/max has no accumulation order or wrapping); its parity tests plant `MinInt32`/`MaxInt32` in both a mid-block lane and the scalar tail, in both orderings, to catch a dropped vector lane or a skipped tail.

### Performance Notes

- **AMD64**: Explicit SIMD ranges from roughly **2-6x** on memory-bound elementwise
  operations up to **10-16x** on reductions and fused kernels (DotProduct, Sum,
  EuclideanDistance, Clamp). The elementwise multiples are more modest than on older
  Go toolchains because Go 1.26 generates tighter code for the scalar reference loops,
  which speeds up the pure-Go baseline the SIMD path is measured against.

- **ARM64**: NEON SIMD provides substantial speedups over pure Go across all operations:
  - float32: **3.5x - 5.2x** faster (4 elements per 128-bit vector)
  - float64: **2.4x - 2.6x** faster (2 elements per 128-bit vector)

- **CumulativeSum** is inherently sequential (each element depends on the previous) and uses pure Go on all platforms.

- **Methodology**: amd64 numbers are from the Intel Core i7-1260P (AVX+FMA) and arm64
  numbers from a Raspberry Pi 5 (Cortex-A76, NEON), both pinned to the `performance`
  CPU governor, built with the Go 1.26 toolchain (the module itself still targets the
  Go 1.25 minimum in `go.mod`; 1.26 is only what these benchmarks were measured on).
  Pure-Go baselines use the same binary via `SIMD_DISABLE=all` or each operation's
  `*Go` reference; each pair reports the best of repeated runs. Displayed nanoseconds
  are rounded to whole ns, so the speedup column (computed from the raw timings) may
  differ from a recomputation using the rounded ns shown.

## Known Limitations

### Small Slice Fallback for Min/Max (AMD64)

On AMD64, the `Min` and `Max` functions fall back to pure Go for small slices:

- **float64**: slices with fewer than 4 elements
- **float32**: slices with fewer than 8 elements

This is because AVX assembly loads multiple elements at once (4 float64s or 8 float32s), which would cause out-of-bounds memory access on smaller slices.

The Go fallback for small slices is intentional and likely optimal - SIMD setup overhead (register loading, masking, horizontal reduction) would exceed the cost of a simple 2-3 element comparison loop.

## Architecture Support

The library selects the best available kernel at runtime and falls back to pure
Go when no SIMD path applies. The amd64 baseline is not uniform across packages:
each package only ships the kernels its workload needs, so the **minimum amd64
instruction-set tier that activates SIMD differs per package** (verified against
each package's `*_amd64.go` dispatch):

| Package | amd64 minimum SIMD tier | Higher amd64 tiers used | Below the minimum |
| ------- | ----------------------- | ----------------------- | ----------------- |
| `f32`   | SSE2                    | AVX+FMA, AVX-512        | pure Go (baseline guarantees SSE2 on amd64) |
| `f64`   | SSE2                    | AVX (no FMA), AVX+FMA, AVX-512 | pure Go (baseline guarantees SSE2) |
| `c128`  | SSE2                    | AVX (no FMA), AVX+FMA, AVX-512 | pure Go (baseline guarantees SSE2) |
| `c64`   | SSE4.1 (BLENDPS)        | AVX+FMA, AVX-512        | pure Go |
| `i16`   | SSE2                    | AVX2                    | pure Go (baseline guarantees SSE2) |
| `i32`   | AVX (interleave), AVX2 (arithmetic) | -           | pure Go |
| `i8`    | AVX2                    | -                       | pure Go |
| `f16`   | F16C (slice conversions only) | -                 | pure Go (all f16 compute is pure Go on amd64) |
| `crc`   | PCLMULQDQ               | -                       | scalar slice-by-16 |

SSE2 is part of the amd64 baseline, so `f32`/`f64`/`c128`/`i16` always run SIMD on
amd64 (their pure-Go path is effectively a non-amd64 safety net). AVX-512 uses the
`AVX512F && AVX512VL` gate. `cpu.Info()` reports the host-wide tier (AVX-512 /
AVX+FMA / AVX / SSE2 / scalar); a package whose minimum is above that tier (e.g.
`i32` on an SSE-only host) runs pure Go even though `Info()` shows SSE2.

ARM64 runs NEON kernels throughout, with an FP16 (FEAT_FP16) fast path in `f16`
and FP16-widened variants elsewhere, plus an SDOT (FEAT_DotProd) fast path for
`i8.DotProduct` (base-NEON `SMULL`/`SADALP` on cores without it). **SVE/SVE2 is detected but unused:** there are
no SVE kernels yet, so an SVE-capable host (Graviton 3, Neoverse V1) still runs the
NEON path, and `cpu.Info()` annotates this as `ARM64 NEON+FP16 (SVE detected, unused)`.

The f16 per-architecture summary:

| Architecture | Instruction Set | f64/f32/c128/c64  | f16                    |
| ------------ | --------------- | ----------------- | ---------------------- |
| AMD64        | AVX-512         | Full SIMD support | F16C conversions       |
| AMD64        | AVX + FMA       | Full SIMD support | F16C conversions       |
| AMD64        | SSE2/SSE4.1     | Full SIMD support | Pure Go fallback       |
| ARM64        | NEON + FP16     | Full SIMD support | Full SIMD support      |
| ARM64        | NEON only       | Full SIMD support | Pure Go fallback       |
| Other        | -               | Pure Go fallback  | Pure Go fallback       |

(AMD64 f16 "F16C conversions" = hardware `ToFloat32Slice`/`FromFloat32Slice`; all
other f16 ops run the pure-Go reference. F16C is VEX-encoded and needs AVX, so
amd64 parts without AVX use pure Go for conversions too.)

**ARM64 FP16 support by device:**

| Device / SoC              | Core(s)       | Architecture | FP16 SIMD |
| ------------------------- | ------------- | ------------ | --------- |
| Apple Silicon (M1-M4)     | Firestorm+    | ARMv8.4-A    | ✅ Yes    |
| Raspberry Pi 5            | Cortex-A76    | ARMv8.2-A    | ✅ Yes    |
| Raspberry Pi 4            | Cortex-A72    | ARMv8.0-A    | ❌ No     |
| Raspberry Pi 3            | Cortex-A53    | ARMv8.0-A    | ❌ No     |
| AWS Graviton 2/3          | Neoverse N1/V1| ARMv8.2-A+   | ✅ Yes    |
| Ampere Altra              | Neoverse N1   | ARMv8.2-A    | ✅ Yes    |

## Design Principles

1. **Pure Go assembly** - Native Go assembler for maximum portability and easy cross-compilation
2. **Runtime dispatch** - CPU features detected once at init time, zero runtime overhead
3. **Zero allocations** - No heap allocations in hot paths
4. **Safe defaults** - Gracefully falls back to pure Go on unsupported CPUs
5. **Boundary safe** - Handles any slice length, not just SIMD-aligned sizes

## Testing

The library includes comprehensive tests with pure Go reference implementations for validation:

```bash
# Run all tests
go test ./...

# Run tests with verbose output
task test

# Run benchmarks
task bench

# Compare SIMD vs pure Go performance
task bench:compare

# Show CPU SIMD capabilities
task cpu
```

See [Taskfile.yml](Taskfile.yml) for all available tasks.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
