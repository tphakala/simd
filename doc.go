// Package simd provides high-performance SIMD-accelerated operations for Go.
//
// This module contains subpackages for CPU feature detection and vectorized
// numeric operations:
//
//   - [github.com/tphakala/simd/cpu] - CPU feature detection
//   - [github.com/tphakala/simd/f64] - float64 SIMD operations
//   - [github.com/tphakala/simd/f32] - float32 SIMD operations
//   - [github.com/tphakala/simd/i32] - int32 SIMD operations (integer DSP)
//   - [github.com/tphakala/simd/c128] - complex128 SIMD operations
//   - [github.com/tphakala/simd/crc] - CRC checksums (carry-less-multiply folding)
//
// # Architecture Support
//
// The library automatically selects the optimal implementation at runtime:
//
//   - AMD64: AVX-512 (8x float64, 16x float32), AVX+FMA (4x float64, 8x float32), or SSE2 fallback
//   - ARM64: NEON/ASIMD instructions (2x float64, 4x float32)
//   - Other: Pure Go fallback
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
// Reductions: Sum, DotProduct, DotProductBatch, DotProductIndexed, DotProductStrided, Min, Max, MinIdx, MaxIdx
//
// Statistics: Mean, Variance, StdDev, EuclideanDistance, Normalize
//
// Element-wise: Abs, Neg, Sqrt, Reciprocal, Clamp
//
// Activation functions: Sigmoid, ReLU, Tanh, Exp, ClampScale
//
// Transcendental (f32/f64): Log, Log2, Log10, Pow, PowElem (plus LogInPlace, PowInPlace)
//
// Audio DSP: Interleave2, Deinterleave2, ConvolveValid, ConvolveValidMulti, ConvolveDecimate, AccumulateAdd, CumulativeSum, CubicInterpDot, Int32ToFloat32Scale, Int16ToFloat32Scale, Float32ToInt16Scale
//
// Integer DSP (i32): Interleave2, Deinterleave2, Add, Sub, MidSideEncode, MidSideDecode, Diff1..Diff4, Restore1..Restore4, LPCResidualEncode, LPCRestore, RiceSums, RiceBestParam
//
// Complex (c64/c128): Add, Sub, Mul, MulConj, DotProduct, DotProductConj, Conj, Abs, AbsSq, Scale
//
// CRC (crc): Checksum16 (FLAC CRC-16, PCLMULQDQ/PMULL carry-less-multiply fold)
//
// # Design Principles
//
//   - No CGO: Pure Go assembly for easy cross-compilation
//   - Zero allocations: All operations work on pre-allocated slices
//   - Thread-safe: All functions are safe for concurrent use
//   - Safe defaults: Graceful fallback to pure Go on unsupported CPUs
package simd
