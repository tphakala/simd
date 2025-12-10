# simd

[![Go Reference](https://pkg.go.dev/badge/github.com/tphakala/simd.svg)](https://pkg.go.dev/github.com/tphakala/simd)
[![Go Report Card](https://goreportcard.com/badge/github.com/tphakala/simd)](https://goreportcard.com/report/github.com/tphakala/simd)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance SIMD (Single Instruction, Multiple Data) library for Go providing vectorized operations on float64, float32, float16, and complex128 slices.

## Features

- **Pure Go assembly** - Native Go assembler, simple cross-compilation
- **Runtime CPU detection** - Automatically selects optimal implementation (AVX-512, AVX+FMA, SSE2, NEON, NEON+FP16, or pure Go)
- **Zero allocations** - All operations work on pre-allocated slices
- **80+ operations** - Arithmetic, reduction, statistical, vector, signal processing, activation functions, and complex number operations
- **Multi-architecture** - AMD64 (AVX-512/AVX+FMA/SSE2) and ARM64 (NEON/NEON+FP16) with pure Go fallback
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

fmt.Println(cpu.Info())      // "AMD64 AVX-512", "AMD64 AVX+FMA", "AMD64 SSE2", or "ARM64 NEON"
fmt.Println(cpu.HasAVX())    // true/false
fmt.Println(cpu.HasAVX512()) // true/false
fmt.Println(cpu.HasNEON())   // true/false
fmt.Println(cpu.HasFP16())   // true/false (ARM64 half-precision SIMD)
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
|                 | `FMA(dst, a, b, c)`                 | Fused multiply-add: a\*b+c    | 8x / 4x / 2x                        |
|                 | `AddScaled(dst, alpha, s)`          | dst += alpha\*s (axpy)        | 8x / 4x / 2x                        |
| **Unary**       | `Abs(dst, a)`                       | Absolute value                | 8x / 4x / 2x                        |
|                 | `Neg(dst, a)`                       | Negation                      | 8x / 4x / 2x                        |
|                 | `Sqrt(dst, a)`                      | Square root                   | 8x / 4x / 2x                        |
|                 | `Reciprocal(dst, a)`                | Reciprocal (1/x)              | 8x / 4x / 2x                        |
| **Reduction**   | `DotProduct(a, b)`                  | Dot product                   | 8x / 4x / 2x                        |
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
| **Activation**  | `Sigmoid(dst, src)`                 | Sigmoid: 1/(1+e^-x)           | 4x (AVX) / 2x (NEON)                |
|                 | `ReLU(dst, src)`                    | Rectified Linear Unit         | 8x / 4x / 2x                        |
|                 | `Tanh(dst, src)`                    | Hyperbolic tangent            | 8x / 4x / 2x                        |
|                 | `Exp(dst, src)`                     | Exponential e^x               | Pure Go                             |
|                 | `ClampScale(dst, src, min, max, s)` | Fused clamp and scale         | 8x / 4x / 2x                        |
| **Batch**       | `DotProductBatch(r, rows, v)`       | Multiple dot products         | 8x / 4x / 2x                        |
| **Signal**      | `ConvolveValid(dst, sig, k)`        | FIR filter / convolution      | 8x / 4x / 2x                        |
|                 | `ConvolveValidMulti(dsts, sig, ks)` | Multi-kernel convolution      | 8x / 4x / 2x                        |
|                 | `AccumulateAdd(dst, src, off)`      | Overlap-add: dst[off:] += src | 8x / 4x / 2x                        |
| **Audio**       | `Interleave2(dst, a, b)`            | Pack stereo: [L,R,L,R,...]    | 4x / 2x                             |
|                 | `Deinterleave2(a, b, src)`          | Unpack stereo to channels     | 4x / 2x                             |
|                 | `CubicInterpDot(hist,a,b,c,d,x)`    | Fused cubic interp dot product| 4x / 2x                             |
|                 | `Int32ToFloat32Scale(dst,src,s)`    | PCM int32 to normalized float | 8x / 4x                             |

### `f32` - float32 Operations

Same API as `f64` but for `float32` with wider SIMD:

| Architecture    | SIMD Width  |
| --------------- | ----------- |
| AMD64 (AVX-512) | 16x float32 |
| AMD64 (AVX+FMA) | 8x float32  |
| AMD64 (SSE2)    | 4x float32  |
| ARM64 (NEON)    | 4x float32  |

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
|                 | `Sum(a)` → float32                  | Sum of elements               | 8x (NEON+FP16)   |
|                 | `Min(a)`                            | Minimum value                 | 8x (NEON+FP16)   |
|                 | `Max(a)`                            | Maximum value                 | 8x (NEON+FP16)   |
|                 | `MinIdx(a)`                         | Index of minimum              | Pure Go          |
|                 | `MaxIdx(a)`                         | Index of maximum              | Pure Go          |
| **Statistical** | `Mean(a)` → float32                 | Arithmetic mean               | 8x (NEON+FP16)   |
|                 | `Variance(a)` → float32             | Population variance           | Pure Go          |
|                 | `StdDev(a)` → float32               | Standard deviation            | Pure Go          |
| **Vector**      | `EuclideanDistance(a, b)` → float32 | L2 distance                   | Pure Go          |
|                 | `Normalize(dst, a)`                 | Unit vector normalization     | 8x (NEON+FP16)   |
|                 | `CumulativeSum(dst, a)`             | Running sum                   | Sequential       |
| **Range**       | `Clamp(dst, a, min, max)`           | Clamp to range                | 8x (NEON+FP16)   |
|                 | `ClampScale(dst, src, min, max, s)` | Fused clamp and scale         | Pure Go          |
| **Activation**  | `ReLU(dst, src)`                    | Rectified Linear Unit         | 8x (NEON+FP16)   |
|                 | `Sigmoid(dst, src)`                 | Sigmoid: 1/(1+e^-x)           | Pure Go          |
|                 | `Tanh(dst, src)`                    | Hyperbolic tangent            | Pure Go          |
|                 | `Exp(dst, src)`                     | Exponential e^x               | Pure Go          |
| **Batch**       | `DotProductBatch(r, rows, v)`       | Multiple dot products         | 8x (NEON+FP16)   |
| **Signal**      | `ConvolveValid(dst, sig, k)`        | FIR filter / convolution      | Pure Go          |
|                 | `AccumulateAdd(dst, src, off)`      | Overlap-add: dst[off:] += src | 8x (NEON+FP16)   |
| **Audio**       | `Interleave2(dst, a, b)`            | Pack stereo: [L,R,L,R,...]    | Pure Go          |
|                 | `Deinterleave2(a, b, src)`          | Unpack stereo to channels     | Pure Go          |

**Key characteristics:**

- **Storage**: IEEE 754 half-precision (1 sign, 5 exponent, 10 mantissa bits)
- **Precision**: ~3.3 decimal digits, range ~6×10⁻⁸ to 65504
- **Reductions**: Accumulate in float32 for numerical stability
- **Memory efficiency**: 2x bandwidth vs float32 (8 elements per 128-bit NEON vector)

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

**Benchmark (1024 elements, Intel i7-1260P AVX+FMA):**

| Operation | SIMD    | Pure Go | Speedup   |
| --------- | ------- | ------- | --------- |
| Mul       | 341 ns  | 757 ns  | **2.2x**  |
| MulConj   | 340 ns  | 749 ns  | **2.2x**  |
| Scale     | 253 ns  | 551 ns  | **2.2x**  |
| Add       | 86 ns   | 189 ns  | **2.2x**  |
| Abs       | 1326 ns | 2260 ns | **1.7x**  |
| AbsSq     | 367 ns  | 504 ns  | **1.37x** |
| Conj      | 304 ns  | 474 ns  | **1.56x** |

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

**float64 (1024 elements):**

| Function   | SIMD (ns) | Go (ns)  | Speedup    | SIMD Throughput |
| ---------- | --------- | -------- | ---------- | --------------- |
| ReLU       | 68        | 646      | **9.5x**   | 240 GB/s        |
| Tanh       | 445       | 6230     | **14x**    | 36.8 GB/s       |

**Key Characteristics:**

- **Tanh**: 200x+ speedup for f32 - fast approximation with saturation vs math.Tanh
- **ReLU**: Highest throughput (211-240 GB/s) - simple max(0, x) operation
- **Sigmoid**: 43x speedup for f32 - fast approximation with exponential

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
| Int32ToFloat32Scale      | 1024 elements         | 40 ns   | 364 ns  | **9.0x** |
| Int32ToFloat32Scale      | 4096 elements         | 153 ns  | 1439 ns | **9.4x** |
| Interleave2 (f64)        | 1000 pairs            | 216 ns  | -       | -        |
| Deinterleave2 (f64)      | 1000 pairs            | 216 ns  | -       | -        |
| Interleave2 (f32)        | 1000 pairs            | 109 ns  | -       | -        |
| Deinterleave2 (f32)      | 1000 pairs            | 216 ns  | -       | -        |

#### Performance Summary

| Package  | Average Speedup | Best         | Operations   |
| -------- | --------------- | ------------ | ------------ |
| **f32**  | **6.5x**        | 21.8x (Abs)  | 32 functions |
| **f64**  | **3.2x**        | 7.9x (Clamp) | 32 functions |
| **c128** | **1.77x**       | 2.2x (Mul)   | 8 functions  |

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

| Architecture | Instruction Set | f64/f32/c128      | f16               |
| ------------ | --------------- | ----------------- | ----------------- |
| AMD64        | AVX-512         | Full SIMD support | Pure Go fallback  |
| AMD64        | AVX + FMA       | Full SIMD support | Pure Go fallback  |
| AMD64        | SSE2            | Full SIMD support | Pure Go fallback  |
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
