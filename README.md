# simd

[![Go Reference](https://pkg.go.dev/badge/github.com/tphakala/simd.svg)](https://pkg.go.dev/github.com/tphakala/simd)
[![Go Report Card](https://goreportcard.com/badge/github.com/tphakala/simd)](https://goreportcard.com/report/github.com/tphakala/simd)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance SIMD (Single Instruction, Multiple Data) library for Go providing vectorized operations on float64, float32, float16, int32, int16, complex128, and complex64 slices.

## Features

- **Pure Go assembly** - Native Go assembler, simple cross-compilation
- **Runtime CPU detection** - Automatically selects optimal implementation (AVX-512, AVX+FMA, AVX without FMA, SSE4.1, NEON, NEON+FP16, or pure Go)
- **Zero allocations** - All operations work on pre-allocated slices
- **80+ operations** - Arithmetic, reduction, statistical, vector, signal processing, activation functions, and complex number operations
- **Multi-architecture** - AMD64 (AVX-512/AVX+FMA/SSE4.1) and ARM64 (NEON/NEON+FP16) with pure Go fallback
- **Half-precision support** - Native FP16 SIMD on ARM64 with FP16 extension (Apple Silicon, Cortex-A55+)
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
fmt.Println(cpu.HasAVX())      // true/false
fmt.Println(cpu.HasAVX2())     // true/false
fmt.Println(cpu.HasFMA())      // true/false
fmt.Println(cpu.HasAVX512VL()) // true/false (AVX-512 F+VL)
fmt.Println(cpu.HasNEON())     // true/false
fmt.Println(cpu.HasFP16())     // true/false (ARM64 half-precision SIMD)
```

### `f64` - float64 Operations

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
AMD64 (AVX-512 and AVX+FMA) instead of re-loading it per row. Short, ragged, or
sub-SIMD-width rows fall back to the per-row dot product, with identical results.

`Autocorrelate` computes the LPC autocorrelation `autoc[lag] = Σ x[i]·x[i-lag]`
used by FLAC-style encoders. It vectorizes across lags (one accumulator lane per
lag, never fusing the multiply-add), so each lag's sum keeps the exact left-to-right
order of the scalar loop and every build emits byte-identical results to the pure-Go
reference. The AVX2 path accumulates four lags per YMM, NEON two lags per V register;
non-AVX2/NEON CPUs and short blocks use the scalar reference.

### `f32` - float32 Operations

Same API as `f64` but for `float32` with wider SIMD:

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
and N=2/3/4 (NEON) kernels; all other stream counts use the allocation-free generic
path. The N=3 case is the 16k -> 48k upsample hot path: the AVX2 gather/blend kernel
runs roughly 2.8x (interleave) and 3.2x (deinterleave) over the generic loop on AVX2.
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
|                 | `ToFloat32Slice(dst, src)`          | Batch FP16 → float32          | 8x (NEON+FP16)   |
|                 | `FromFloat32Slice(dst, src)`        | Batch float32 → FP16          | 8x (NEON+FP16)   |
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
| EuclideanDistance | 481 ns | 5995 ns  | **12.5x** |
| Variance          | 506 ns | 8901 ns  | **17.6x** |
| Interleave2       | 178 ns | 2163 ns  | **12.2x** |
| Deinterleave2     | 178 ns | 2167 ns  | **12.2x** |
| ClampScale        | 531 ns | 12211 ns | **23.0x** |

**Hardware requirements:**

- **Native FP16 SIMD**: ARM64 with FEAT_FP16 (ARMv8.2-A+)
  - Apple Silicon (M1/M2/M3/M4) ✅
  - Cortex-A55, A75, A76, A77, A78, X1, X2, X3 ✅
  - Raspberry Pi 5 (Cortex-A76) ✅
- **Pure Go fallback**: All other platforms
  - Raspberry Pi 3/4 (Cortex-A53/A72 - ARMv8.0) - works but no SIMD acceleration
  - AMD64 - works but no SIMD acceleration

### `c128` - complex128 Operations

SIMD-accelerated complex number operations for FFT-based signal processing:

| Category       | Function             | Description                        | SIMD Width              |
| -------------- | -------------------- | ---------------------------------- | ----------------------- |
| **Arithmetic** | `Mul(dst, a, b)`     | Complex multiplication             | 4x (AVX-512) / 2x (AVX) |
|                | `MulConj(dst, a, b)` | Multiply by conjugate: a × conj(b) | 4x / 2x                 |
|                | `Scale(dst, a, s)`   | Scale by complex scalar            | 4x / 2x                 |
|                | `Add(dst, a, b)`     | Complex addition                   | 4x / 2x                 |
|                | `Sub(dst, a, b)`     | Complex subtraction                | 4x / 2x                 |
| **Unary**      | `Abs(dst, a)`        | Complex magnitude \|a + bi\|       | 4x (AVX-512) / 2x (AVX) |
|                | `AbsSq(dst, a)`      | Magnitude squared \|a + bi\|²      | 4x / 2x                 |
|                | `Conj(dst, a)`       | Complex conjugate: a - bi          | 4x / 2x                 |

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

**Benchmark (1024 elements, Intel Xeon Platinum 8362 AVX-512):**

| Operation | SIMD    | Pure Go | Speedup   |
| --------- | ------- | ------- | --------- |
| Mul       | 239 ns  | 1021 ns | **4.3x**  |
| MulConj   | 314 ns  | 1434 ns | **4.6x**  |
| Scale     | 180 ns  | 959 ns  | **5.3x**  |
| Add       | 255 ns  | 733 ns  | **2.9x**  |
| Abs       | 918 ns  | 3453 ns | **3.8x**  |
| AbsSq     | 237 ns  | 594 ns  | **2.5x**  |
| Conj      | 163 ns  | 594 ns  | **3.7x**  |

### `c64` - complex64 Operations

SIMD-accelerated single-precision complex number operations:

| Category       | Function             | Description                        | SIMD Width                        |
| -------------- | -------------------- | ---------------------------------- | --------------------------------- |
| **Arithmetic** | `Mul(dst, a, b)`     | Complex multiplication             | 8x (AVX-512) / 4x (AVX) / 2x (NEON) |
|                | `MulConj(dst, a, b)` | Multiply by conjugate: a × conj(b) | 8x / 4x / 2x                      |
|                | `Scale(dst, a, s)`   | Scale by complex scalar            | 8x / 4x / 2x                      |
|                | `Add(dst, a, b)`     | Complex addition                   | 8x / 4x / 2x                      |
|                | `Sub(dst, a, b)`     | Complex subtraction                | 8x / 4x / 2x                      |
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

SIMD-accelerated integer-domain operations for codec and integer-DSP hot loops (for example a pure-Go FLAC encoder/decoder), where the per-sample work is integer arithmetic and channel (de)interleaving rather than floating-point math:

| Category            | Function                                | Description                                              | SIMD Width            |
| ------------------- | --------------------------------------- | -------------------------------------------------------- | --------------------- |
| **Interleave**      | `Interleave2(dst, a, b)`                | Pack two channels into interleaved stereo                | 8x (AVX) / 4x (NEON)  |
|                     | `Deinterleave2(a, b, src)`              | Split interleaved stereo into two channels               | 8x (AVX) / 4x (NEON)  |
| **Decorrelation**   | `Add(dst, a, b)`                        | Element-wise add (RIGHT_SIDE decode)                     | 8x (AVX2) / 4x (NEON) |
|                     | `Sub(dst, a, b)`                        | Element-wise subtract (side channel / LEFT_SIDE decode)  | 8x (AVX2) / 4x (NEON) |
|                     | `MidSideEncode(mid, side, left, right)` | Mid/side forward decorrelation                           | 8x (AVX2) / 4x (NEON) |
|                     | `MidSideDecode(left, right, mid, side)` | Mid/side inverse (parity-bit reconstruction)             | 8x (AVX2) / 4x (NEON) |
| **Fixed predictor** | `Diff1`..`Diff4(dst, src)`              | Order 1-4 fixed-predictor encode residual (forward diff) | 8x (AVX2) / 4x (NEON) |
|                     | `Restore1`..`Restore4(dst, src)`        | Order 1-4 fixed-predictor decode (inverse; prefix sum)   | 8x (AVX2) / 4x (NEON) |
|                     | `FixedAbsSums(src, sums)`               | Order 0-4 residual abs-sums in one pass (predictor select) | 8x (AVX2) / 4x (NEON) |
| **LPC**             | `LPCResidualEncode(res, samples, coeffs, shift)` | Quantized-LPC encode residual FIR (parallel across outputs) | 8x (AVX2) / 4x (NEON) |
|                     | `LPCRestore(out, residual, coeffs, shift)`       | Quantized-LPC decode (serial recurrence; vectorized taps)   | 8x (AVX2) / 4x (NEON) |
| **Rice**            | `RiceSums(sums, res)`                   | Per-parameter zigzag unary-bit sums `Σ (zigzag(res)>>k)`, all FLAC params k=0..30 | 8x (AVX2) / 4x (NEON) |
|                     | `RiceBestParam(res, maxParam)`          | Cost-minimizing Rice parameter + its bit count           | 8x (AVX2) / 4x (NEON) |
|                     | `ZigzagSum(res)`                        | Total zigzag fold `Σ zigzag(res)` (estimate fast path)   | 8x (AVX2) / 4x (NEON) |
| **Reduction**       | `MinMax(res) (min, max)`                | Signed int32 per-slice minimum and maximum in one pass   | 8x (AVX2) / 4x (NEON) |

```go
import "github.com/tphakala/simd/i32"

left := make([]int32, n)
right := make([]int32, n)
stereo := make([]int32, n*2)

i32.Interleave2(stereo, left, right)   // [l0, r0, l1, r1, ...]
i32.Deinterleave2(left, right, stereo) // inverse: split back to channels

res := make([]int32, n)
i32.Diff2(res, left)     // order-2 fixed-predictor encode residual
i32.Restore2(left, res)  // exact inverse: reconstruct the samples

var absSums [5]uint64
i32.FixedAbsSums(left, &absSums) // order 0-4 residual abs-sums for predictor selection

coeffs := []int32{...}             // quantized LPC coefficients (order = len)
i32.LPCResidualEncode(res, left, coeffs, shift) // encode FIR residual
i32.LPCRestore(left, res, coeffs, shift)        // exact inverse: reconstruct

param, bits := i32.RiceBestParam(res, 14) // best Rice parameter and its bit cost
total := i32.ZigzagSum(res)               // Σ zigzag(res): the k=0 Rice sum on its own
mn, mx := i32.MinMax(res)                 // smallest and largest residual in one signed pass
```

Interleaving is pure 32-bit-lane movement, so those kernels reuse the proven `f32` shuffle/permute encodings (AVX `VUNPCKLPS`/`VPERM2F128`, NEON `ZIP`/`UZP` on `.4S`); the bit pattern of each lane is irrelevant, so negative values and the type extremes round-trip exactly. The decorrelation and fixed-predictor kernels do integer-ALU work on 256-bit (AVX2) / 128-bit (NEON) lanes. `RestoreK` is the exact inverse of `DiffK`: since the order-`K` fixed predictor is the `K`-th forward difference, restoration is `K` cumulative-sum passes built on one SIMD prefix-sum kernel (11x at order 1 down to ~5x at order 4 on AVX2; 6x to 3x on NEON, all zero-allocation). The LPC kernels accumulate the prediction sum in int64 (matching libFLAC), so they stay bit-exact for the full coefficient precision and order: `LPCResidualEncode` is a FIR that vectorizes across output samples (widening `VPMULDQ` / `SMLAL`, ~7-9x on AVX2, ~4-5x on NEON), while `LPCRestore` is a serial recurrence whose per-output tap dot product is vectorized, keeping the just-written newest samples on store-forwardable scalar loads (~1.4-3.9x on AVX2, ~1.8-3.6x on NEON, order 8 to 32). `FixedAbsSums` picks the cheapest fixed predictor in one pass: it forms the order-0..4 forward finite differences in int64 (a 4th difference of int32 samples exceeds the int32 range) with a windowed sign-extending subtract cascade, sums `|e_order|` per order excluding the warm-up, and runs ~2.6x on AVX2, ~1.7x on NEON. The Rice cost search folds each residual to its unsigned zigzag symbol `(r<<1)^(r>>31)`, widens it to int64, and accumulates `Σ (zigzag(res)>>k)` for every Rice parameter at once (the symbols are halved progressively, exact for the logical shift); the kernel now covers FLAC's full range `k` in 0..30 (the 5-bit method, previously a scalar tail above 14), so a 31-column `RiceSums` runs ~13x on AVX2 and ~4x on NEON instead of falling back to scalar. `RiceBestParam` adds the `n*(k+1)` code overhead and picks the cheapest `k`; this is the exact bit cost, not the `sum>>k` approximation. `ZigzagSum` exposes just the `k=0` total `Σ zigzag(res)` for the estimate path (~3.7x AVX2, ~2.4x NEON). `MinMax` returns the smallest and largest int32 in one signed pass (`VPMINSD`/`VPMAXSD` on AVX2, `SMIN`/`SMAX` with single-instruction `SMINV`/`SMAXV` folds on NEON); the Rice planner uses it per partition because the largest zigzag fold is reached at the most-negative or most-positive sample. Since min/max of int32 has no accumulation order, the SIMD paths are bit-identical to the pure-Go reference by construction (~10x AVX2, ~5x NEON). All zero-allocation.

### `i16` - int16 Operations

The 16-bit integer counterpart to `i32`, for raw-PCM hot loops (such as a pure-Go FLAC codec) where the source samples are 16-bit and the cheapest place to vectorize is the channel (de)interleaving that happens before samples are widened to int32. FLAC decorrelation widens samples (the side and mid channels can exceed the source bit depth by one bit), so the arithmetic primitives live in `i32`; this package carries only the operations that provably help at 16-bit width:

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

## Performance

### AMD64 (Intel Core i7-1260P, AVX+FMA)

#### float64 Operations - SIMD vs Pure Go (1024 elements)

| Category        | Operation         | SIMD (ns) | Go (ns) | Speedup  |
| --------------- | ----------------- | --------- | ------- | -------- |
| **Arithmetic**  | Add               | 84        | 446     | **5.3x** |
|                 | Sub               | 84        | 335     | **4.0x** |
|                 | Mul               | 86        | 436     | **5.1x** |
|                 | Div               | 441       | 941     | **2.1x** |
|                 | Scale             | 68        | 272     | **4.0x** |
|                 | AddScalar         | 68        | 286     | **4.2x** |
|                 | FMA               | 110       | 557     | **5.0x** |
| **Unary**       | Abs               | 66        | 365     | **5.6x** |
|                 | Neg               | 66        | 306     | **4.6x** |
|                 | Sqrt              | 658       | 1323    | **2.0x** |
|                 | Reciprocal        | 447       | 920     | **2.1x** |
| **Reduction**   | DotProduct        | 162       | 859     | **5.3x** |
|                 | Sum               | 82        | 184     | **2.3x** |
|                 | Min               | 157       | 340     | **2.2x** |
|                 | Max               | 154       | 352     | **2.3x** |
| **Statistical** | Mean              | 82        | 184     | **2.3x** |
|                 | Variance\*        | 820       | 902     | **1.1x** |
|                 | StdDev\*          | 825       | 905     | **1.1x** |
| **Vector**      | EuclideanDistance | 216       | 1071    | **5.0x** |
|                 | Normalize         | 220       | 1080    | **4.9x** |
|                 | CumulativeSum     | 428       | 425     | 1.0x     |
| **Range**       | Clamp             | 81        | 640     | **7.9x** |

\*Variance/StdDev benchmarked at 4096 elements (SIMD benefits at larger sizes)

#### float32 Operations - SIMD vs Pure Go (1024 elements)

| Category       | Operation  | SIMD (ns) | Go (ns) | Speedup   |
| -------------- | ---------- | --------- | ------- | --------- |
| **Arithmetic** | Add        | 47        | 441     | **9.4x**  |
|                | Sub        | 49        | 339     | **6.9x**  |
|                | Mul        | 49        | 436     | **8.9x**  |
|                | Div        | 138       | 655     | **4.8x**  |
|                | Scale      | 40        | 299     | **7.4x**  |
|                | AddScalar  | 39        | 272     | **7.0x**  |
|                | FMA        | 64        | 444     | **6.9x**  |
| **Unary**      | Abs        | 37        | 656     | **17.6x** |
|                | Neg        | 40        | 273     | **6.9x**  |
| **Reduction**  | DotProduct | 71        | 424     | **5.9x**  |
|                | Sum        | 41        | 123     | **3.0x**  |
|                | Min        | 65        | 340     | **5.2x**  |
|                | Max        | 66        | 352     | **5.3x**  |
| **Range**      | Clamp      | 47        | 701     | **14.8x** |

#### Activation Functions - SIMD vs Pure Go

**float32 (1024 elements):**

| Function   | SIMD (ns) | Go (ns)  | Speedup    | SIMD Throughput |
| ---------- | --------- | -------- | ---------- | --------------- |
| Sigmoid    | 138       | 5906     | **43x**    | 59.3 GB/s       |
| ReLU       | 39        | 662      | **17x**    | 211 GB/s        |
| Tanh       | 138       | 28116    | **204x**   | 59.5 GB/s       |
| Exp        | 312       | 5555     | **18x**    | 26.3 GB/s       |

**float64 (1024 elements):**

| Function   | SIMD (ns) | Go (ns)  | Speedup    | SIMD Throughput |
| ---------- | --------- | -------- | ---------- | --------------- |
| Sigmoid    | 745       | 5640     | **7.6x**   | 22.0 GB/s       |
| ReLU       | 68        | 646      | **9.5x**   | 240 GB/s        |
| Tanh       | 836       | 6529     | **7.8x**   | 19.6 GB/s       |
| Exp        | 606       | 4698     | **7.8x**   | 27.0 GB/s       |

**Key Characteristics:**

- **Tanh**: 200x+ speedup for f32 - fast approximation with saturation vs math.Tanh
- **ReLU**: Highest throughput (211-240 GB/s) - simple max(0, x) operation
- **Sigmoid**: 43x speedup for f32 - fast approximation with exponential
- **Exp**: 18x speedup for f32 (12x on ARM64 NEON) via range reduction plus a degree-5 polynomial; max relative error ~7e-6 (f32), ~3e-6 (f64)

#### Batch & Signal Processing (varied sizes)

| Operation                | Config                | SIMD    | Go      | Speedup  |
| ------------------------ | --------------------- | ------- | ------- | -------- |
| DotProductBatch (f64)    | 256 vec × 100 rows    | 3.2 µs  | 20.5 µs | **6.4x** |
| DotProductBatch (f32)    | 256 vec × 100 rows    | 1.5 µs  | 9.8 µs  | **6.7x** |
| ConvolveValid (f64)      | 4096 sig × 64 ker     | 26.6 µs | 169 µs  | **6.3x** |
| ConvolveValid (f32)      | 4096 sig × 64 ker     | 17.9 µs | 80 µs   | **4.5x** |
| ConvolveValidMulti (f64) | 1000 sig × 64 ker × 2 | 13.4 µs | -       | -        |
| CubicInterpDot (f64)     | 241 taps              | 47 ns   | 88 ns   | **1.9x** |
| CubicInterpDot (f32)     | 241 taps              | 21 ns   | 66 ns   | **3.1x** |
| Int32ToFloat32Scale      | 1024 elements         | 50 ns   | 405 ns  | **8.1x** |
| Int32ToFloat32Scale      | 4096 elements         | 157 ns  | 1586 ns | **10.1x**|
| Int16ToFloat32Scale      | 1024 elements         | 58 ns   | 483 ns  | **8.3x** |
| Int16ToFloat32Scale      | 4096 elements         | 200 ns  | 1914 ns | **9.6x** |
| Float32ToInt16Scale      | 1024 elements         | 92 ns   | 1360 ns | **14.8x**|
| Float32ToInt16Scale      | 4096 elements         | 365 ns  | 5420 ns | **14.8x**|
| Interleave2 (f64)        | 1000 pairs            | 216 ns  | -       | -        |
| Deinterleave2 (f64)      | 1000 pairs            | 216 ns  | -       | -        |
| Interleave2 (f32)        | 1000 pairs            | 109 ns  | -       | -        |
| Deinterleave2 (f32)      | 1000 pairs            | 216 ns  | -       | -        |

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
| 20 taps, 2x decimate  | **2.0x** | **2.3x** | **1.7x** | **1.9x** |
| 32 taps, 2x decimate  | **2.3x** | **1.7x** | **1.9x** | **1.7x** |
| 64 taps, 2x decimate  | **2.0x** | **1.8x** | **1.7x** | **1.3x** |
| 241 taps, 2x decimate | **1.5x** | **1.3x** | **1.2x** | **1.1x** |
| 241 taps, 4x decimate | **1.4x** | **1.2x** | **1.2x** | **1.1x** |

#### Autocorrelate (lag-vectorized LPC autocorrelation, f64)

`Autocorrelate` is the LPC autocorrelation step in a FLAC-style encoder, the
largest remaining single-core hotspot there. Vectorizing across lags keeps the
result byte-identical to the scalar reference while still beating it. Block size
4096, allocation-free, speedup over the pure-Go fallback (AVX2 on x86-64, NEON on
a Raspberry Pi 5):

| Config (n=4096)       | amd64 (AVX2) | arm64 (NEON) |
| --------------------- | ------------ | ------------ |
| maxLag 8              | **2.9x**     | **2.4x**     |
| maxLag 12             | **3.0x**     | **2.4x**     |
| maxLag 32             | **3.5x**     | **2.7x**     |

#### Performance Summary

| Package  | Average Speedup | Best         | Operations   |
| -------- | --------------- | ------------ | ------------ |
| **f32**  | **6.5x**        | 21.8x (Abs)  | 35 functions |
| **f64**  | **3.2x**        | 7.9x (Clamp) | 32 functions |
| **c128** | **2.7x**        | 3.4x (Abs)   | 8 functions  |
| **c64**  | **~2x**         | ~3x (Mul)    | 9 functions  |

### ARM64 (Raspberry Pi 5, NEON)

#### float64 Operations

| Operation  | Size | Time   | Throughput |
| ---------- | ---- | ------ | ---------- |
| DotProduct | 277  | 151 ns | 29 GB/s    |
| DotProduct | 1000 | 513 ns | 31 GB/s    |
| Add        | 1000 | 775 ns | 31 GB/s    |
| Mul        | 1000 | 727 ns | 33 GB/s    |
| FMA        | 1000 | 890 ns | 36 GB/s    |
| Sum        | 1000 | 635 ns | 13 GB/s    |
| Mean       | 1000 | 677 ns | 12 GB/s    |

#### float32 Operations

| Operation  | Size  | Time    | Throughput |
| ---------- | ----- | ------- | ---------- |
| DotProduct | 100   | 37 ns   | 21 GB/s    |
| DotProduct | 1000  | 263 ns  | 30 GB/s    |
| DotProduct | 10000 | 2.78 µs | 29 GB/s    |
| Add        | 1000  | 389 ns  | 31 GB/s    |
| Mul        | 1000  | 390 ns  | 31 GB/s    |
| FMA        | 1000  | 479 ns  | 33 GB/s    |

#### Comparison vs Pure Go

| Operation        | Size | SIMD   | Pure Go | Speedup  |
| ---------------- | ---- | ------ | ------- | -------- |
| DotProduct (f32) | 100  | 37 ns  | 137 ns  | **3.7x** |
| DotProduct (f32) | 1000 | 262 ns | 1350 ns | **5.2x** |
| DotProduct (f64) | 100  | 62 ns  | 138 ns  | **2.2x** |
| DotProduct (f64) | 1000 | 513 ns | 1353 ns | **2.6x** |
| Add (f32)        | 1000 | 389 ns | 2015 ns | **5.2x** |
| Sum (f32)        | 1000 | 343 ns | 1327 ns | **3.9x** |

### int32 (i32) - SIMD vs Pure Go (1000 elements)

| Operation     | AMD64 (AVX/AVX2)             | ARM64 (NEON, Pi 5)            |
| ------------- | ---------------------------- | ----------------------------- |
| Interleave2   | 121 ns vs 487 ns (**4.0x**)  | 322 ns vs 1679 ns (**5.2x**)  |
| Deinterleave2 | 228 ns vs 482 ns (**2.1x**)  | 322 ns vs 1682 ns (**5.2x**)  |
| Restore1      | 181 ns vs 2019 ns (**11.2x**)| 605 ns vs 3763 ns (**6.2x**)  |
| Restore2      | 320 ns vs 2296 ns (**7.2x**) | 1101 ns vs 4592 ns (**4.2x**) |
| Restore3      | 462 ns vs 2493 ns (**5.4x**) | 1608 ns vs 5421 ns (**3.4x**) |
| Restore4      | 575 ns vs 2778 ns (**4.8x**) | 2093 ns vs 6245 ns (**3.0x**) |
| LPCResidualEncode (order 8)  | 743 ns vs 5333 ns (**7.2x**)  | 2495 ns vs 10707 ns (**4.3x**) |
| LPCResidualEncode (order 32) | 2397 ns vs 22046 ns (**9.2x**)| 8148 ns vs 40811 ns (**5.0x**) |
| LPCRestore (order 8)         | 3583 ns vs 4944 ns (**1.4x**) | 6128 ns vs 10889 ns (**1.8x**) |
| LPCRestore (order 32)        | 4443 ns vs 17529 ns (**3.9x**)| 11303 ns vs 41026 ns (**3.6x**) |
| RiceSums (k=0..14)           | 810 ns vs 10490 ns (**13.0x**)| 4090 ns vs 17163 ns (**4.2x**) |
| RiceBestParam (k=0..14)      | 810 ns vs 10490 ns (**13.0x**)| 4081 ns vs 17163 ns (**4.2x**) |
| RiceSums (k=0..30, 5-bit)    | 1685 ns vs 21401 ns (**12.7x**)| 8503 ns vs 34111 ns (**4.0x**) |
| ZigzagSum                    | 88 ns vs 328 ns (**3.7x**)    | 476 ns vs 1124 ns (**2.4x**)  |
| FixedAbsSums (orders 0..4)   | 752 ns vs 1952 ns (**2.6x**)  | 2881 ns vs 4777 ns (**1.7x**) |
| MinMax                       | 47 ns vs 480 ns (**10.2x**)   | 216 ns vs 1103 ns (**5.1x**)  |

### int16 (i16) - SIMD vs Pure Go (1000 elements)

| Operation     | AMD64 (AVX2/SSE2)           | ARM64 (NEON, Pi 5)            |
| ------------- | --------------------------- | ----------------------------- |
| Interleave2   | 60 ns vs 585 ns (**9.8x**)  | 165 ns vs 2106 ns (**12.8x**) |
| Deinterleave2 | 59 ns vs 640 ns (**10.9x**) | 165 ns vs 2120 ns (**12.9x**) |

Both i16 kernels are zero-allocation and bit-exact against the pure-Go reference (verified with negative values and the int16 extremes); they move whole 16-bit lanes, so the bit pattern of each sample is irrelevant to correctness.

All int32 kernels are zero-allocation and bit-exact against the pure-Go reference (verified across the sign and high bits with negative values and the type extremes). The Restore baseline is the pure-Go decode recurrence; `RestoreK` reconstructs samples as `K` SIMD prefix-sum passes (the inverse of the order-`K` forward difference). The LPC kernels accumulate in int64 and are checked against an arbitrary-precision `math.big` oracle and the encode->decode round-trip; `LPCRestore` is the serial decode recurrence (vectorized per-output tap dot product), dispatched to SIMD only at order >= 8 where it beats the scalar path. The Rice kernels compute the exact per-parameter unary-bit sums `Σ (zigzag(res)>>k)` for FLAC's full parameter range in one sweep, checked against an independent oracle and brute-force parameter scan: the 4-bit method (k=0..14) and the 5-bit method (k=0..30, a low-15 kernel plus a pre-shifted high-16 kernel) are both fully vectorized, replacing the scalar tail that previously handled k>14. `ZigzagSum` is that sweep's k=0 column on its own (Σ zigzag), and `FixedAbsSums` computes the order-0..4 fixed-predictor residual abs-sums in one windowed int64 cascade; both are checked against independent oracles across the sign bit and the int32 extremes (where the 4th difference exceeds int32, exercising the int64 width). `MinMax` is exact by construction (signed min/max has no accumulation order or wrapping); its parity tests plant `MinInt32`/`MaxInt32` in both a mid-block lane and the scalar tail, in both orderings, to catch a dropped vector lane or a skipped tail.

### Performance Notes

- **AMD64**: Explicit SIMD provides **5x** speedups for most operations compared to pure Go, with consistent high throughput across all vector sizes.

- **ARM64**: NEON SIMD provides substantial speedups over pure Go across all operations:
  - float32: **3.7x - 5.2x** faster (4 elements per 128-bit vector)
  - float64: **2.2x - 2.6x** faster (2 elements per 128-bit vector)

- **CumulativeSum** is inherently sequential (each element depends on the previous) and uses pure Go on all platforms.

## Known Limitations

### Small Slice Fallback for Min/Max (AMD64)

On AMD64, the `Min` and `Max` functions fall back to pure Go for small slices:

- **float64**: slices with fewer than 4 elements
- **float32**: slices with fewer than 8 elements

This is because AVX assembly loads multiple elements at once (4 float64s or 8 float32s), which would cause out-of-bounds memory access on smaller slices.

The Go fallback for small slices is intentional and likely optimal - SIMD setup overhead (register loading, masking, horizontal reduction) would exceed the cost of a simple 2-3 element comparison loop.

## Architecture Support

| Architecture | Instruction Set | f64/f32/c128/c64  | f16               |
| ------------ | --------------- | ----------------- | ----------------- |
| AMD64        | AVX-512         | Full SIMD support | Pure Go fallback  |
| AMD64        | AVX + FMA       | Full SIMD support | Pure Go fallback  |
| AMD64        | SSE4.1          | Full SIMD support | Pure Go fallback  |
| ARM64        | NEON + FP16     | Full SIMD support | Full SIMD support |
| ARM64        | NEON only       | Full SIMD support | Pure Go fallback  |
| Other        | -               | Pure Go fallback  | Pure Go fallback  |

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
