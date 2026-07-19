// Package simd provides high-performance SIMD-accelerated operations for Go.
//
// This module contains subpackages for CPU feature detection and vectorized
// numeric operations:
//
//   - [github.com/tphakala/simd/cpu] - CPU feature detection
//   - [github.com/tphakala/simd/f64] - float64 SIMD operations (FLAC/LPC and scientific surface)
//   - [github.com/tphakala/simd/f32] - float32 SIMD operations (audio/FFT/ML surface)
//   - [github.com/tphakala/simd/f16] - float16 storage type (ARM64 NEON+FP16 compute; amd64 F16C slice conversions)
//   - [github.com/tphakala/simd/i32] - int32 SIMD operations (integer DSP)
//   - [github.com/tphakala/simd/i16] - int16 SIMD operations (PCM movement, and widening int16 x int16 -> int32 reductions)
//   - [github.com/tphakala/simd/i8] - int8 SIMD operations (saturating arithmetic, int32-accumulated reductions, quantized DSP)
//   - [github.com/tphakala/simd/c64] - complex64 SIMD operations (FFT-pipeline helpers)
//   - [github.com/tphakala/simd/c128] - complex128 SIMD operations (FFT-pipeline helpers)
//   - [github.com/tphakala/simd/cint] - fixed-point complex SIMD operations (int32 data x int16 Q15 twiddle; integer FFT butterflies)
//   - [github.com/tphakala/simd/crc] - CRC checksums (carry-less-multiply folding)
//
// # Architecture Support
//
// The library automatically selects the optimal implementation at runtime. The
// minimum amd64 instruction-set tier that activates SIMD differs per package
// (each package only ships the kernels its workload needs):
//
//   - AMD64: AVX-512 (8x float64, 16x float32) > AVX+FMA (4x float64, 8x float32) >
//     AVX (no FMA, f64/c128) > SSE2 (f32/f64/c128, i16 interleave/dot/xcorr)
//     or SSE4.1 (c64) > pure Go.
//     i32 needs AVX/AVX2, cint and i8 need AVX2, crc needs PCLMULQDQ, and f16 uses F16C
//     for its slice conversions only (every other f16 op is pure Go on amd64).
//     SSE2 is part of the amd64 baseline, so f32/f64/c128 always get SIMD on
//     amd64, as do i16's interleave/dot/xcorr kernels; i16's element-wise ops
//     (Abs, MulQ15) and its MaxAbs reduction are AVX2-or-Go, like i8 and the
//     i32 arithmetic.
//   - ARM64: NEON/ASIMD throughout (2x float64, 4x float32), with an FP16
//     (FEAT_FP16) fast path in the f16 package and an SDOT (FEAT_DotProd) fast
//     path for i8.DotProduct, and SMLAL/SMLAL2 widening multiply-accumulate
//     for i16.DotProduct. SVE/SVE2 is detected by cpu.Info() but no SVE
//     kernels exist yet, so SVE hosts run the NEON path.
//   - Other: Pure Go fallback
//
// f16 is a storage type: SIMD acceleration is ARM64-only for compute (NEON+FP16),
// plus F16C-accelerated ToFloat32Slice/FromFloat32Slice conversions on amd64.
//
// # Disabling SIMD tiers
//
// Set the SIMD_DISABLE environment variable before the process starts to mask
// detected CPU features (a comma-separated, case-insensitive token list such as
// "avx512", "avx", "neon", or "all"). It is useful for avoiding AVX-512
// downclocking, exercising the lower tiers locally, and benchmarking tiers
// against each other. Unknown tokens are ignored. The variable cannot be toggled
// at runtime, because the SIMD packages cache their selected kernels during
// package init (function pointers on amd64, capability flags on arm64). See the
// cpu package for the full token table.
//
// # Quick Start
//
//	import (
//	    "github.com/tphakala/simd/cpu"
//	    "github.com/tphakala/simd/f64"
//	)
//
//	func main() {
//	    fmt.Println("SIMD:", cpu.Info())
//
//	    a := []float64{1, 2, 3, 4, 5, 6, 7, 8}
//	    b := []float64{8, 7, 6, 5, 4, 3, 2, 1}
//
//	    // Dot product
//	    dot := f64.DotProduct(a, b)
//
//	    // Element-wise operations
//	    dst := make([]float64, len(a))
//	    f64.Add(dst, a, b)
//	    f64.Mul(dst, a, b)
//	}
//
// # Available Operations
//
// Core arithmetic: Add, Sub, Mul, Div, Scale, AddScalar, AddScaled, FMA
//
// Reductions: Sum, DotProduct, DotProductBatch, DotProductIndexed, DotProductStrided, Min, Max, MaxAbs, MinIdx, MaxIdx
//
// Statistics: Mean, Variance, StdDev, EuclideanDistance, Normalize
//
// Element-wise: Abs, Neg, Sqrt, Reciprocal, Clamp
//
// Activation functions: Sigmoid, ReLU, Tanh, Exp, ClampScale
//
// Transcendental (f32/f64): Log, Log2, Log10, Pow, PowElem (plus LogInPlace, PowInPlace),
// SIMD-accelerated on AVX2+FMA and NEON
//
// Audio DSP: Interleave2, Deinterleave2, ConvolveValid, ConvolveValidMulti, ConvolveValidMaxAbs, ConvolveValidMaxAbsMulti, ConvolveDecimate, AccumulateAdd, CumulativeSum, CubicInterpDot, Int32ToFloat32Scale, Int16ToFloat32Scale, Float32ToInt16Scale
//
// Sliding-window argmin (f32): MinIdxOfSum, MinIdxOfSumRows (batched sliding-window argmin of a[i]+k[base+r*slide+i], first-index-wins ties, bit-exact across all paths)
//
// Spectral (f64, f32): STFTPlan (NewSTFTPlan, STFT, STFTPower, STFTPowerInto, NumFrames) - fused real-input short-time Fourier transform with optional librosa-style center=true framing (PadMode: NoPad/PadZero/PadReflect)
//
// Integer DSP (i16): Interleave2, Deinterleave2, DotProduct, DotProductUnsafe, XCorr (widening int16 x int16 -> wrapping int32; ARM64 SMLAL/SMLAL2, amd64 PMADDWD/VPMADDWD; XCorr evaluates 4 correlation lags per kernel call), Abs, MaxAbs, MulQ15 (wrapping 16-bit absolute value, widened abs-max, rounding Q15 multiply)
//
// Integer DSP (i32): Interleave2, Deinterleave2, Add, Sub, Abs, Sum, MinMax, MaxAbs, NegWhereNeg, ScaleQ31, ScaleQ15, Butterfly, FIRValidQ15
//
// Integer DSP (i8): AddSaturate, SubSaturate, AddScalarSaturate, SubScalarSaturate, Min, Max, Clamp, Abs, Neg, AbsDiff, MaxAbs, SumAbs, SAD, ToInt16, ToInt32, Sum, MinMax, DotProduct (int32-accumulated; ARM64 SDOT / amd64 VPMADDWD)
//
// Complex (c64/c128): Add, Sub, Mul, MulConj, DotProduct, DotProductConj, Conj, Abs, AbsSq, Scale
//
// Fixed-point complex (cint): Add, Sub, Mul, MulConj, MulByScalar (int32 data x int16 Q15 twiddle, truncating C_MUL; for integer FFT butterflies)
//
// CRC (crc): Checksum16 (CRC-16, poly 0x8005, MSB-first, no reflection; used by FLAC among others, PCLMULQDQ/PMULL carry-less-multiply fold)
//
// # Design Principles
//
//   - No CGO: Pure Go assembly for easy cross-compilation
//   - Zero allocations: All operations work on pre-allocated slices
//   - Thread-safe: All functions are safe for concurrent use
//   - Safe defaults: Graceful fallback to pure Go on unsupported CPUs
package simd
