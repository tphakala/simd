// Package f64 provides SIMD-accelerated operations on float64 slices.
//
// All functions automatically select the optimal implementation based on
// runtime CPU feature detection. Functions gracefully fall back to pure Go
// implementations on unsupported architectures.
//
// Thread Safety: All functions are safe for concurrent use.
// Memory: All functions are zero-allocation (no heap allocations).
//
// # Aliasing
//
// The element-wise maps may be used fully in place: the destination may alias an
// input exactly, element for element. This holds for the unary maps (Abs, Neg,
// Round, Sqrt, Reciprocal, Exp, Log, Log2, Log10, ReLU, Sigmoid, Tanh, Scale,
// AddScalar, SubFromScalar, Clamp, ClampScale, Pow), for the two-pass
// CumulativeSum and Normalize (dst==a), for the binary maps (Add, Sub, Mul, Div,
// PowElem, where dst may alias any input, or both at once), and for the fused
// multiply-add FMA (dst may alias a, b or c). The guarantee is mechanical: each
// SIMD block reads its whole block of inputs into registers before storing any
// output lane, and the scalar tail reads each lane before it writes that lane, so
// an exact overlay is well defined lane by lane on every dispatch path (amd64 SSE,
// AVX with and without FMA, AVX-512, arm64 NEON, and the pure-Go fallback).
//
// A destination must not overlap an input at a shifted offset. A SIMD load pulls
// a whole block of an input ahead of the stores, so a shifted overlay clobbers
// input lanes a later iteration has not yet read; the resulting corruption
// pattern is undefined and varies with kernel width and length.
//
// The departures from this default are documented on the functions themselves:
//   - AddScaled and AccumulateAdd read and rewrite dst in place (they are
//     accumulators); their source slice must be disjoint from the destination
//     window, except that AddScaled also permits s==dst exactly.
//   - ButterflyComplex and ButterflyComplexStage update their data slices in
//     place; those slices must be distinct from one another, and the twiddles
//     must not overlap them.
//   - The mirror, window, stride, interleave and batch operations (Interleave2/N,
//     Deinterleave2/N, ConvolveValid and ConvolveValidMulti, ConvolveDecimate,
//     DotProductBatch, Autocorrelate, RealFFTUnpack and RealFFTPower) index inputs
//     and outputs at different positions, so their outputs must not overlap any
//     input; RealFFTUnpack and RealFFTPower carry their own detailed notes.
//
// The reductions (DotProduct, WeightedSum, SumOfSquares, Sum, Mean, Max, Min,
// MaxAbs, MaxIdx, MinIdx, Variance, StdDev, EuclideanDistance, ConvolveValidMaxAbs
// and ConvolveValidMaxAbsMulti, CubicInterpDot) write no output slice, so aliasing
// does not apply to them. The in-place transcendental variants (ExpInPlace,
// LogInPlace, PowInPlace, ReLUInPlace, SigmoidInPlace, TanhInPlace) operate on a
// single slice, so aliasing does not arise, and the Unsafe variants follow the
// aliasing rules of the checked form they mirror.
package f64

import "math"

// DotProduct computes the dot product of two float64 slices.
// Returns sum(a[i] * b[i]) for i in 0..min(len(a), len(b)).
//
// Uses AVX+FMA on AMD64, NEON on ARM64, with pure Go fallback.
func DotProduct(a, b []float64) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	return dotProduct(a, b)
}

// DotProductUnsafe computes the dot product without empty-slice checks.
// It skips the len==0 guard in [DotProduct] but is otherwise identical:
// the underlying SIMD kernels and Go fallback clamp to min(len(a), len(b)) internally,
// so mismatched lengths do not cause out-of-bounds access.
//
// PRECONDITIONS (caller must ensure):
//   - len(a) > 0 && len(b) > 0
func DotProductUnsafe(a, b []float64) float64 {
	return dotProduct(a, b)
}

// WeightedSum returns Σ(weights[i] * src[i]) for i in 0..min(len(weights), len(src)).
// This is equivalent to [DotProduct]; the alternate name documents intent at the call site
// when the operands have asymmetric roles (signal vs. weights).
func WeightedSum(weights, src []float64) float64 {
	if len(weights) == 0 || len(src) == 0 {
		return 0
	}
	return dotProduct(weights, src)
}

// SumOfSquares returns Σ(src[i]²).
func SumOfSquares(src []float64) float64 {
	if len(src) == 0 {
		return 0
	}
	return dotProduct(src, src)
}

// Add computes element-wise addition: dst[i] = a[i] + b[i].
// Processes min(len(dst), len(a), len(b)) elements.
func Add(dst, a, b []float64) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	add(dst[:n], a[:n], b[:n])
}

// Sub computes element-wise subtraction: dst[i] = a[i] - b[i].
// Processes min(len(dst), len(a), len(b)) elements.
func Sub(dst, a, b []float64) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	sub(dst[:n], a[:n], b[:n])
}

// Mul computes element-wise multiplication: dst[i] = a[i] * b[i].
// Processes min(len(dst), len(a), len(b)) elements.
func Mul(dst, a, b []float64) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	mul(dst[:n], a[:n], b[:n])
}

// Div computes element-wise division: dst[i] = a[i] / b[i].
// Processes min(len(dst), len(a), len(b)) elements.
// Division by zero produces +Inf, -Inf, or NaN per IEEE 754.
func Div(dst, a, b []float64) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	div(dst[:n], a[:n], b[:n])
}

// Scale multiplies each element by a scalar: dst[i] = a[i] * s.
// Processes min(len(dst), len(a)) elements.
func Scale(dst, a []float64, s float64) {
	n := min(len(a), len(dst))
	if n == 0 {
		return
	}
	scale(dst[:n], a[:n], s)
}

// AddScalar adds a scalar to each element: dst[i] = a[i] + s.
// Processes min(len(dst), len(a)) elements.
func AddScalar(dst, a []float64, s float64) {
	n := min(len(a), len(dst))
	if n == 0 {
		return
	}
	addScalar(dst[:n], a[:n], s)
}

// SubFromScalar subtracts each element from a scalar: dst[i] = s - a[i].
// Processes min(len(dst), len(a)) elements.
func SubFromScalar(dst, a []float64, s float64) {
	n := min(len(a), len(dst))
	if n == 0 {
		return
	}
	subFromScalar64(dst[:n], a[:n], s)
}

// Sum returns the sum of all elements in the slice.
func Sum(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	return sum(a)
}

// Min returns the minimum value in the slice.
// Returns +Inf for empty slices.
//
// NaN handling: unlike [math.Min], this function does not propagate NaN.
// If the input contains NaN values, the result is architecture-dependent.
// Callers that require strict NaN semantics should filter NaN values first.
func Min(a []float64) float64 {
	if len(a) == 0 {
		return posInf
	}
	return min64(a)
}

// Max returns the maximum value in the slice.
// Returns -Inf for empty slices.
//
// NaN handling: unlike [math.Max], this function does not propagate NaN.
// If the input contains NaN values, the result is architecture-dependent.
// Callers that require strict NaN semantics should filter NaN values first.
func Max(a []float64) float64 {
	if len(a) == 0 {
		return negInf
	}
	return max64(a)
}

// MaxAbs returns the maximum absolute value in the slice (the infinity norm),
// max_i |a[i]|. Returns 0 for an empty slice.
//
// Uses AVX2/SSE2 on AMD64 (AVX-512 CPUs reuse the AVX2 kernel), NEON on ARM64,
// with a pure Go fallback. a is read-only; the call allocates nothing.
//
// NaN handling: |NaN| is NaN and compares false, so the Go path skips NaN. On the
// SIMD paths NaN handling is architecture-dependent, matching [Min] and [Max].
// Callers needing strict NaN semantics should filter NaN first.
func MaxAbs(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	return maxAbs64(a)
}

// Abs computes element-wise absolute value: dst[i] = |a[i]|.
// Processes min(len(dst), len(a)) elements.
func Abs(dst, a []float64) {
	n := min(len(a), len(dst))
	if n == 0 {
		return
	}
	abs64(dst[:n], a[:n])
}

// Neg computes element-wise negation: dst[i] = -a[i].
// Processes min(len(dst), len(a)) elements.
func Neg(dst, a []float64) {
	n := min(len(a), len(dst))
	if n == 0 {
		return
	}
	neg64(dst[:n], a[:n])
}

// FMA computes fused multiply-add: dst[i] = a[i] * b[i] + c[i].
// Uses hardware FMA when available for better precision and performance.
func FMA(dst, a, b, c []float64) {
	n := min(len(c), minLen(len(dst), len(a), len(b)))
	if n == 0 {
		return
	}
	fma64(dst[:n], a[:n], b[:n], c[:n])
}

// Clamp clamps each element to [min, max]: dst[i] = clamp(a[i], min, max).
func Clamp(dst, a []float64, minVal, maxVal float64) {
	n := min(len(a), len(dst))
	if n == 0 {
		return
	}
	clamp64(dst[:n], a[:n], minVal, maxVal)
}

// Round rounds each element to the nearest integer, half away from zero:
// dst[i] = round(src[i]). Processes min(len(dst), len(src)) elements.
func Round(dst, src []float64) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	round64(dst[:n], src[:n])
}

func minLen(a, b, c int) int {
	if b < a {
		a = b
	}
	if c < a {
		a = c
	}
	return a
}

// Sqrt computes element-wise square root: dst[i] = sqrt(a[i]).
// Processes min(len(dst), len(a)) elements.
func Sqrt(dst, a []float64) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	sqrt64(dst[:n], a[:n])
}

// Reciprocal computes element-wise reciprocal: dst[i] = 1/a[i].
// Processes min(len(dst), len(a)) elements.
func Reciprocal(dst, a []float64) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	reciprocal64(dst[:n], a[:n])
}

// Mean computes the arithmetic mean of a slice.
// Returns 0 for empty slices.
func Mean(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	return Sum(a) / float64(len(a))
}

// Variance computes the population variance of a slice.
// Returns 0 for empty slices.
func Variance(a []float64) float64 {
	n := len(a)
	if n == 0 {
		return 0
	}
	mean := Mean(a)
	return variance64(a, mean)
}

// StdDev computes the population standard deviation of a slice.
// Returns 0 for empty slices.
func StdDev(a []float64) float64 {
	return math.Sqrt(Variance(a))
}

// EuclideanDistance computes the Euclidean distance between two vectors.
// Returns sqrt(sum((a[i] - b[i])^2)) for i in 0..min(len(a), len(b)).
func EuclideanDistance(a, b []float64) float64 {
	n := min(len(a), len(b))
	if n == 0 {
		return 0
	}
	return euclideanDistance64(a[:n], b[:n])
}

// Normalize normalizes a vector to unit length: dst = a / ||a||.
// If the magnitude is zero or very small (< 1e-10), copies the input unchanged.
// Processes min(len(dst), len(a)) elements.
func Normalize(dst, a []float64) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}

	// Compute magnitude
	mag := 0.0
	for i := range n {
		mag += a[i] * a[i]
	}
	mag = math.Sqrt(mag)

	// Avoid division by zero
	if mag < normalizeMagnitudeThreshold {
		copy(dst[:n], a[:n])
		return
	}

	// Scale by 1/magnitude
	Scale(dst[:n], a[:n], 1.0/mag)
}

// CumulativeSum computes the cumulative sum: dst[i] = sum(a[0:i+1]).
// Processes min(len(dst), len(a)) elements.
func CumulativeSum(dst, a []float64) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	cumulativeSum64(dst[:n], a[:n])
}

// DotProductBatch computes multiple dot products against the same vector.
// results[i] = DotProduct(rows[i], vec) for each row.
// This is more cache-efficient than calling DotProduct in a loop because
// vec stays hot in L1 cache across all dot products.
//
// Processes min(len(results), len(rows)) rows.
// Each row is processed up to min(len(row), len(vec)) elements.
func DotProductBatch(results []float64, rows [][]float64, vec []float64) {
	n := min(len(results), len(rows))
	if n == 0 || len(vec) == 0 {
		return
	}
	dotProductBatch64(results[:n], rows[:n], vec)
}

// Autocorrelate computes the autocorrelation of x at lags 0..maxLag:
//
//	autoc[lag] = Σ x[i]*x[i-lag]  for i in lag..len(x)-1
//
// summed left to right with separate multiply and add. The AVX2 and NEON
// kernels vectorize ACROSS lags (one accumulator lane per lag, never fusing the
// multiply-add), so every build produces byte-identical results to the pure-Go
// reference. This is the LPC autocorrelation step used by FLAC-style encoders,
// where the quantized predictor coefficients (hence the output bytes) depend on
// the exact rounding of this reduction.
//
// Preconditions: maxLag >= 0 and len(autoc) >= maxLag+1. Lags beyond len(x)-1
// (whose sums are empty) are written as 0. Processes lags 0..min(maxLag,
// len(autoc)-1).
func Autocorrelate(autoc, x []float64, maxLag int) {
	if maxLag < 0 || len(x) == 0 {
		return
	}
	if maxLag > len(autoc)-1 {
		maxLag = len(autoc) - 1
	}
	if maxLag < 0 {
		return
	}
	autocorrelate64(autoc, x, maxLag)
}

// ConvolveValid computes valid convolution of signal with kernel.
// dst[i] = sum(signal[i+j] * kernel[j]) for j in 0..len(kernel)-1.
// Output length is len(signal) - len(kernel) + 1.
//
// Processes min(len(dst), len(signal)-len(kernel)+1) output elements.
// This is equivalent to applying a FIR filter without zero-padding.
func ConvolveValid(dst, signal, kernel []float64) {
	if len(kernel) == 0 || len(signal) < len(kernel) {
		return
	}
	validLen := len(signal) - len(kernel) + 1
	n := min(len(dst), validLen)
	if n == 0 {
		return
	}
	convolveValid64(dst[:n], signal, kernel)
}

// ConvolveDecimate computes a decimating (strided) valid convolution: it keeps
// only every factor-th valid-convolution output, starting at phase.
//
//	dst[k] = sum_{i=0}^{len(kernel)-1} signal[phase + k*factor + i] * kernel[i]
//
// The kernel is applied as a plain dot product; pre-reverse it for true
// convolution (matching DotProductUnsafe and ConvolveValid usage). factor must
// be >= 1 (factor == 1 is valid convolution at every position) and phase must be
// in [0, factor); factor < 1, phase < 0, or phase >= factor are treated as
// no-ops. With factor == 1 and phase == 0 this is exactly ConvolveValid.
//
// The number of outputs is the count of strided positions whose full kernel
// window fits in signal; ConvolveDecimate writes min(len(dst), that count) and
// leaves the remainder of dst untouched. It allocates nothing and operates on
// the caller-provided buffers.
func ConvolveDecimate(dst, signal, kernel []float64, factor, phase int) {
	kLen := len(kernel)
	if kLen == 0 || factor < 1 || phase < 0 || phase >= factor {
		return
	}
	span := len(signal) - kLen - phase
	if span < 0 {
		return
	}
	n := min(len(dst), span/factor+1)
	if n == 0 {
		return
	}
	convolveDecimate64(dst[:n], signal, kernel, factor, phase)
}

// AccumulateAdd adds src to dst starting at offset: dst[offset:offset+len(src)] += src.
// This is a key primitive for overlap-add in FFT-based convolution.
//
// The destination window dst[offset:offset+len(src)] is a read-modify-write
// accumulator; src must not overlap that window. An overlap-add caller passes a
// src region disjoint from the destination window, so this is the natural usage.
//
// Panics if offset+len(src) > len(dst) or if offset < 0.
func AccumulateAdd(dst, src []float64, offset int) {
	if offset < 0 {
		panic("simd: negative offset")
	}
	n := len(src)
	if n == 0 {
		return
	}
	if offset+n > len(dst) {
		panic("simd: offset+len(src) exceeds len(dst)")
	}
	accumulateAdd64(dst[offset:offset+n], src)
}

const (
	// normalizeMagnitudeThreshold is the minimum magnitude for normalization.
	// Vectors with magnitude below this are left unchanged to avoid division by zero.
	normalizeMagnitudeThreshold = 1e-10
)

var (
	posInf = math.Inf(1)
	negInf = math.Inf(-1)
)

// MinIdx returns the index of the minimum value in the slice.
// Returns -1 for empty slices.
func MinIdx(a []float64) int {
	if len(a) == 0 {
		return -1
	}
	return minIdx64(a)
}

// MaxIdx returns the index of the maximum value in the slice.
// Returns -1 for empty slices.
func MaxIdx(a []float64) int {
	if len(a) == 0 {
		return -1
	}
	return maxIdx64(a)
}

// AddScaled adds scaled values to dst: dst[i] += alpha * s[i].
// This is the AXPY operation from BLAS Level 1.
// Processes min(len(dst), len(s)) elements.
//
// dst is a read-modify-write accumulator. s may overlay dst exactly (the
// self-scaling dst[i] += alpha*dst[i], since each lane is read before it is
// rewritten), but s must not overlap dst at a shifted offset.
func AddScaled(dst []float64, alpha float64, s []float64) {
	n := min(len(dst), len(s))
	if n == 0 {
		return
	}
	addScaled64(dst[:n], alpha, s[:n])
}

// Interleave2 interleaves two slices: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], dst[3]=b[1], ...
// Processes min(len(a), len(b), len(dst)/2) pairs.
// This is useful for converting separate channels to interleaved stereo audio.
func Interleave2(dst, a, b []float64) {
	n := min(len(dst)/interleave2Channels, min(len(a), len(b)))
	if n == 0 {
		return
	}
	interleave2_64(dst[:n*interleave2Channels], a[:n], b[:n])
}

// Deinterleave2 deinterleaves a slice: a[0]=src[0], b[0]=src[1], a[1]=src[2], b[1]=src[3], ...
// Processes min(len(a), len(b), len(src)/2) pairs.
// This is the inverse of Interleave2, useful for splitting stereo audio to channels.
func Deinterleave2(a, b, src []float64) {
	n := min(len(src)/interleave2Channels, min(len(a), len(b)))
	if n == 0 {
		return
	}
	deinterleave2_64(a[:n], b[:n], src[:n*interleave2Channels])
}

const interleave2Channels = 2

// InterleaveN interleaves N planar streams into a single interleaved buffer:
//
//	dst[i*N + c] = srcs[c][i],   N = len(srcs)
//
// It is the N-stream generalization of Interleave2. N == 1 copies srcs[0] into
// dst; N == 2 produces the same result as Interleave2. The number of frames
// written is min(len(dst)/N, min over c of len(srcs[c])); dst beyond n*N and any
// ragged source tails are left untouched. An empty srcs is a no-op. It allocates
// nothing and operates on the caller-provided buffers.
func InterleaveN(dst []float64, srcs [][]float64) {
	nc := len(srcs)
	if nc == 0 {
		return
	}
	n := len(dst) / nc
	for _, s := range srcs {
		if len(s) < n {
			n = len(s)
		}
	}
	if n == 0 {
		return
	}
	interleaveN64(dst, srcs, n)
}

// DeinterleaveN splits one interleaved buffer into N planar streams:
//
//	dsts[c][i] = src[i*N + c],   N = len(dsts)
//
// It is the inverse of InterleaveN and the N-stream generalization of
// Deinterleave2. N == 1 copies src into dsts[0]; N == 2 produces the same result
// as Deinterleave2. The number of frames written is min(len(src)/N, min over c
// of len(dsts[c])); any ragged destination tails are left untouched. An empty
// dsts is a no-op. It allocates nothing and operates on the caller-provided
// buffers.
func DeinterleaveN(dsts [][]float64, src []float64) {
	nc := len(dsts)
	if nc == 0 {
		return
	}
	n := len(src) / nc
	for _, d := range dsts {
		if len(d) < n {
			n = len(d)
		}
	}
	if n == 0 {
		return
	}
	deinterleaveN64(dsts, src, n)
}

// CubicInterpDot computes the fused cubic interpolation dot product:
//
//	Σ hist[i] * (a[i] + x*(b[i] + x*(c[i] + x*d[i])))
//
// This is the hot inner loop for polyphase resampling with cubic coefficient
// interpolation. The polynomial a + x*(b + x*(c + x*d)) is evaluated using
// Horner's method for numerical stability, then multiplied by hist and summed.
//
// Parameters:
//   - hist: history buffer (signal samples)
//   - a, b, c, d: cubic polynomial coefficient arrays
//   - x: fractional phase, typically in [0, 1)
//
// All slices must have equal length. Returns 0 for empty slices.
//
// This fused operation is more efficient than 4 separate DotProduct calls
// because it reads the hist array only once (37% less memory bandwidth).
//
// Uses AVX+FMA on AMD64, NEON on ARM64, with pure Go fallback.
func CubicInterpDot(hist, a, b, c, d []float64, x float64) float64 {
	n := minLen5(len(hist), len(a), len(b), len(c), len(d))
	if n == 0 {
		return 0
	}
	return cubicInterpDot64(hist[:n], a[:n], b[:n], c[:n], d[:n], x)
}

// CubicInterpDotUnsafe computes the fused cubic interpolation dot product
// without length validation.
//
// PRECONDITIONS (caller must ensure):
//   - len(hist) == len(a) == len(b) == len(c) == len(d)
//   - len(hist) > 0
//
// Violating these preconditions results in undefined behavior.
// Use CubicInterpDot for safe operation with automatic length handling.
func CubicInterpDotUnsafe(hist, a, b, c, d []float64, x float64) float64 {
	return cubicInterpDot64(hist, a, b, c, d, x)
}

func minLen5(a, b, c, d, e int) int {
	return min(a, b, c, d, e)
}

// polyphaseMaxFracBits64 is the largest sub-phase fractional width for which
// float64(frac) is exact for every frac in [0, 1<<fracBits). float64 has a 53-bit
// significand, so every integer below 2^53 round-trips exactly; at fracBits == 53
// the largest frac is 2^53-1, still below 2^53 and so still exact. Above 53 the
// frac-to-float conversion would round and x = float64(frac) * fracScale would
// stop matching the consumer bit-for-bit, so PolyphaseResampleCubic rejects it.
// This mirrors f32's polyphaseMaxFracBits32 == 24: both use the full significand
// width p (24 for float32, 53 for float64).
const polyphaseMaxFracBits64 = 53

// PolyphaseResampleCubic resamples hist into out with soxr-style polyphase FIR
// filtering and cubic sub-phase coefficient interpolation, running the whole
// output block in one fused pass. It is the fused form of a per-output loop that
// calls [CubicInterpDot]: for each output it derives the input window and the
// interpolation phase from a fixed-point accumulator, evaluates
//
//	out[k] = Σ hist[div+i] * (a[phase][i] + x*(b[phase][i] + x*(c[phase][i] + x*d[phase][i])))
//
// over tapsPerPhase taps, then advances the accumulator by step. The coefficient
// banks a, b, c, d are indexed [phase][tap] and hold the Catmull-Rom cubic
// interpolation coefficients (base, linear, quadratic, cubic); they match
// go-audio-resampler's bank layout directly. Only the first numPhases rows are
// read, and only the first tapsPerPhase taps of each; longer rows are fine.
//
// The accumulator is fixed-point: at = (inputIndex*numPhases + phase) <<
// fracBits + frac, and step is the per-output increment in the same units. fracBits
// is the sub-phase fractional width (soxr uses 16). x = float64(frac) * 2^-fracBits
// is the sub-phase position in [0, 1). numPhases is soxr's L. Output k is produced
// only while k < len(out) and div+tapsPerPhase <= len(hist); the first output not
// satisfying the window bound ends the block.
//
// It returns n, the number of outputs written to out[:n], and atOut, the
// accumulator after n outputs. atOut == at + int64(n)*step exactly, so streaming
// callers carry atOut into the next call as the new at (rebasing at by
// consumed*numPhases<<fracBits when they drop consumed input samples from hist, as
// go-audio-resampler does). The internal (div, phase, frac) are re-derived from
// the single int64 accumulator on the next call and are never exposed.
//
// PolyphaseResampleCubic validates its inputs and is a no-op returning (0, at)
// (never an error, never a panic) if any of these do not hold: numPhases >= 1,
// tapsPerPhase >= 1, step >= 1, at >= 0, 0 <= fracBits <= 53, len(a|b|c|d) >=
// numPhases, every one of the first numPhases rows of each bank has length >=
// tapsPerPhase, and at + int64(len(out))*step does not overflow int64.
// Degenerate but valid inputs (len(out) == 0, len(hist) <
// tapsPerPhase, or an initial window already past the end of hist) also yield
// (0, at) because no output satisfies the window bound.
//
// Uses AVX+FMA on AMD64 and NEON on ARM64 for tapsPerPhase past the vector width,
// gated on the same CPU features and tap threshold as [CubicInterpDot] so the
// fused result is bit-identical to the per-output form on every CPU; below the
// threshold it uses the pure-Go path. It allocates nothing.
func PolyphaseResampleCubic(out, hist []float64, a, b, c, d [][]float64, at, step int64, numPhases, tapsPerPhase, fracBits int) (n int, atOut int64) {
	if numPhases < 1 || tapsPerPhase < 1 || step < 1 || at < 0 ||
		fracBits < 0 || fracBits > polyphaseMaxFracBits64 {
		return 0, at
	}
	if len(a) < numPhases || len(b) < numPhases || len(c) < numPhases || len(d) < numPhases {
		return 0, at
	}
	for p := range numPhases {
		if len(a[p]) < tapsPerPhase || len(b[p]) < tapsPerPhase ||
			len(c[p]) < tapsPerPhase || len(d[p]) < tapsPerPhase {
			return 0, at
		}
	}
	// Reject inputs whose block accumulator could overflow int64. The loop runs
	// at most len(out) times, so at+len(out)*step bounds every accumulator value;
	// if that product would overflow, the internal div wraps negative and defeats
	// the window guard (a panic on the Go path, an out-of-range read on the SIMD
	// paths), and atOut would no longer equal at+n*step. Real resamplers keep step
	// and at far below this, so no legitimate input is rejected.
	if outLen := int64(len(out)); outLen > 0 && step > (math.MaxInt64-at)/outLen {
		return 0, at
	}
	n = polyphaseResampleCubic64(out, hist, a, b, c, d, at, step, numPhases, tapsPerPhase, fracBits)
	return n, at + int64(n)*step
}

// PolyphaseResampleCubicUnsafe is [PolyphaseResampleCubic] without input
// validation, for performance-critical callers that have already checked their
// arguments.
//
// PRECONDITIONS (caller must ensure):
//   - numPhases >= 1, tapsPerPhase >= 1, step >= 1, at >= 0
//   - 0 <= fracBits <= 53
//   - len(a), len(b), len(c), len(d) each >= numPhases
//   - each of the first numPhases rows of a, b, c, d has length >= tapsPerPhase
//   - at + int64(len(out))*step does not overflow int64
//
// Violating these preconditions results in undefined behavior. Unlike the safe
// form, this variant does not reject an overflowing at/step, so a caller that
// breaks the last precondition can drive the internal accumulator to wrap and
// read out of range. When the preconditions hold it returns the same (n, atOut)
// as the safe form.
func PolyphaseResampleCubicUnsafe(out, hist []float64, a, b, c, d [][]float64, at, step int64, numPhases, tapsPerPhase, fracBits int) (n int, atOut int64) {
	n = polyphaseResampleCubic64(out, hist, a, b, c, d, at, step, numPhases, tapsPerPhase, fracBits)
	return n, at + int64(n)*step
}

// ConvolveValidMulti applies multiple kernels to the same signal.
// dsts[k][i] = sum(signal[i+j] * kernels[k][j]) for each kernel k.
// All kernels must have the same length.
//
// This is a convenience wrapper that calls ConvolveValid for each kernel.
// For polyphase resampling with multiple filter phases, this provides
// a clean API without additional overhead.
//
// Panics if kernels have different lengths or if dsts/kernels lengths don't match.
func ConvolveValidMulti(dsts [][]float64, signal []float64, kernels [][]float64) {
	numKernels := len(kernels)
	if numKernels == 0 || len(dsts) < numKernels {
		return
	}

	// Validate all kernels have the same length
	kLen := len(kernels[0])
	if kLen == 0 || len(signal) < kLen {
		return
	}
	for i := 1; i < numKernels; i++ {
		if len(kernels[i]) != kLen {
			panic("simd: all kernels must have the same length")
		}
	}

	validLen := len(signal) - kLen + 1

	// Determine actual output length based on smallest dst
	n := validLen
	for i := range numKernels {
		if len(dsts[i]) < n {
			n = len(dsts[i])
		}
	}
	if n <= 0 {
		return
	}

	convolveValidMulti64(dsts, signal, kernels, n, kLen)
}

// ConvolveValidMaxAbs returns max(|valid-convolution output|) without
// materializing the output slice: the peak (infinity norm) of the FIR applied to
// signal with no zero-padding. Returns 0 when len(kernel) == 0 or
// len(signal) < len(kernel).
//
// Each output element is a SIMD dot product; the abs-max is fused into the pass,
// so there is no scratch buffer and no second scan over an output array. This is
// the peak-detection / true-peak primitive. a is read-only; the call allocates
// nothing.
func ConvolveValidMaxAbs(signal, kernel []float64) float64 {
	if len(kernel) == 0 || len(signal) < len(kernel) {
		return 0
	}
	return convolveValidMaxAbs64(signal, kernel)
}

// ConvolveValidMaxAbsMulti returns the single maximum of |valid-convolution
// output| across every kernel applied to signal, without materializing any
// output. This is the polyphase true-peak primitive: pass the N phase kernels and
// get back the peak of the reconstructed signal in one call. Returns 0 when
// kernels is empty, the first kernel is empty, or len(signal) is shorter than the
// kernel length. The call allocates nothing.
//
// Panics if the kernels do not all share one length, matching [ConvolveValidMulti].
func ConvolveValidMaxAbsMulti(signal []float64, kernels [][]float64) float64 {
	numKernels := len(kernels)
	if numKernels == 0 {
		return 0
	}
	kLen := len(kernels[0])
	if kLen == 0 || len(signal) < kLen {
		return 0
	}
	for i := 1; i < numKernels; i++ {
		if len(kernels[i]) != kLen {
			panic("simd: all kernels must have the same length")
		}
	}
	var m float64
	for _, kernel := range kernels {
		if km := convolveValidMaxAbs64(signal, kernel); km > m {
			m = km
		}
	}
	return m
}

// Sigmoid computes the sigmoid activation function: dst[i] = 1 / (1 + e^(-src[i])).
// This is commonly used as an activation function in neural networks.
// Processes min(len(dst), len(src)) elements.
//
// Uses AVX2 on AMD64 (4x float64), NEON on ARM64 (2x float64). FMA is not
// used: the kernel reconstructs 2^k with 256-bit integer ops instead.
func Sigmoid(dst, src []float64) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	sigmoid64(dst[:n], src[:n])
}

// SigmoidInPlace computes the sigmoid activation function in-place: a[i] = 1 / (1 + e^(-a[i])).
// This is commonly used as an activation function in neural networks.
//
// Uses AVX2 on AMD64 (4x float64), NEON on ARM64 (2x float64). FMA is not
// used: the kernel reconstructs 2^k with 256-bit integer ops instead.
func SigmoidInPlace(a []float64) {
	if len(a) == 0 {
		return
	}
	sigmoid64(a, a)
}

// ReLU computes the Rectified Linear Unit: dst[i] = max(0, src[i]).
// This is commonly used as an activation function in neural networks.
// Processes min(len(dst), len(src)) elements.
//
// Uses AVX on AMD64 (4x float64), NEON on ARM64 (2x float64).
func ReLU(dst, src []float64) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	relu64(dst[:n], src[:n])
}

// ReLUInPlace computes ReLU in-place: a[i] = max(0, a[i]).
func ReLUInPlace(a []float64) {
	if len(a) == 0 {
		return
	}
	relu64(a, a)
}

// ClampScale performs fused clamp and scale: dst[i] = (clamp(src[i], min, max) - min) * scale.
// This is useful for normalizing data to a specific range.
// Processes min(len(dst), len(src)) elements.
//
// Uses AVX on AMD64 (4x float64), NEON on ARM64 (2x float64).
func ClampScale(dst, src []float64, minVal, maxVal, scale float64) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	clampScale64(dst[:n], src[:n], minVal, maxVal, scale)
}

// Tanh computes the hyperbolic tangent: dst[i] = tanh(src[i]).
// Uses fast approximation: tanh(x) ≈ x / (1 + |x|) for |x| < 1, sign(x) for |x| >= 2.5, polynomial otherwise.
// Processes min(len(dst), len(src)) elements.
//
// Uses AVX2 on AMD64 (4x float64), NEON on ARM64 (2x float64).
func Tanh(dst, src []float64) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	tanh64(dst[:n], src[:n])
}

// TanhInPlace computes tanh in-place: a[i] = tanh(a[i]).
func TanhInPlace(a []float64) {
	if len(a) == 0 {
		return
	}
	tanh64(a, a)
}

// Exp computes the exponential function: dst[i] = e^src[i].
// Processes min(len(dst), len(src)) elements.
//
// The SIMD paths use range reduction plus a degree-5 polynomial, giving a
// maximum relative error of about 3e-6. Inputs are clamped to [-709, 709] so
// results stay finite (exp(709) is near MaxFloat64); inputs below about -709
// underflow to 0. This matches the pure-Go fallback's clamping.
//
// Uses AVX2 on AMD64, NEON on ARM64 (2x float64), and falls back
// to math.Exp otherwise.
func Exp(dst, src []float64) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	exp64(dst[:n], src[:n])
}

// ExpInPlace computes exp in-place: a[i] = e^a[i].
func ExpInPlace(a []float64) {
	if len(a) == 0 {
		return
	}
	exp64(a, a)
}

// Log computes the natural logarithm elementwise: dst[i] = ln(src[i]).
// Processes min(len(dst), len(src)) elements. Edge cases match math.Log:
// Log(0) = -Inf, Log(x < 0) = NaN, Log(+Inf) = +Inf, Log(NaN) = NaN.
//
// On AVX2+FMA and NEON hosts a vectorized kernel is used (atanh-form minimax
// polynomial; worst-case relative error is a few float64 ulps, including
// subnormal inputs). Elsewhere it falls back to the math.Log reference.
// Allocation-free and safe for concurrent use on disjoint buffers.
func Log(dst, src []float64) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	log64(dst[:n], src[:n])
}

// LogInPlace computes the natural logarithm in place: a[i] = ln(a[i]).
func LogInPlace(a []float64) {
	if len(a) == 0 {
		return
	}
	log64(a, a)
}

// Log2 computes the base-2 logarithm elementwise: dst[i] = log2(src[i]).
// Useful for log-frequency and octave math. Processes min(len(dst), len(src))
// elements; edge cases match math.Log2.
func Log2(dst, src []float64) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	log2_64(dst[:n], src[:n])
}

// Log10 computes the base-10 logarithm elementwise: dst[i] = log10(src[i]).
// This is the building block for dB conversion (20*log10 for amplitude,
// 10*log10 for power) and log-mel spectrograms. Processes
// min(len(dst), len(src)) elements; edge cases match math.Log10.
func Log10(dst, src []float64) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	log10_64(dst[:n], src[:n])
}

// Pow raises each element to a scalar power: dst[i] = src[i]**exp. The scalar
// exponent is the common DSP case (for example the ^0.35 power-law compression
// in PCEN). Processes min(len(dst), len(src)) elements; edge cases match
// math.Pow (Pow(x, 0) = 1, Pow(negative, non-integer) = NaN, Pow(0, negative)
// = +Inf).
//
// On AVX2+FMA and NEON hosts, slices whose elements are all positive and
// finite are computed with a fused exp(exp*ln(x)) kernel whose relative error
// is bounded by the Exp core (~3e-6); overflow yields +Inf and underflow 0,
// matching math.Pow. Slices containing non-positive, infinite, or NaN bases,
// and calls with a zero or non-finite exponent, take the exact math.Pow path.
// Allocation-free and safe for concurrent use on disjoint buffers.
func Pow(dst, src []float64, exp float64) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	pow64(dst[:n], src[:n], exp)
}

// PowInPlace raises each element to a scalar power in place: a[i] = a[i]**exp.
func PowInPlace(a []float64, exp float64) {
	if len(a) == 0 {
		return
	}
	pow64(a, a, exp)
}

// PowElem raises each base to its own exponent: dst[i] = base[i]**exp[i].
// Processes min(len(dst), len(base), len(exp)) elements; edge cases match
// math.Pow. The SIMD fast path and its fallback rules match Pow, with the
// additional requirement that every exponent is finite.
func PowElem(dst, base, exp []float64) {
	n := min(len(dst), len(base), len(exp))
	if n == 0 {
		return
	}
	powElem64(dst[:n], base[:n], exp[:n])
}

func minLen6(a, b, c, d, e, f int) int {
	return min(a, b, c, d, e, f)
}

// ButterflyComplex performs the FFT butterfly operation with twiddle factor multiply:
//
//	temp_re = lower_re[i]*tw_re[i] - lower_im[i]*tw_im[i]
//	temp_im = lower_re[i]*tw_im[i] + lower_im[i]*tw_re[i]
//	upper_re[i], lower_re[i] = upper_re[i]+temp_re, upper_re[i]-temp_re
//	upper_im[i], lower_im[i] = upper_im[i]+temp_im, upper_im[i]-temp_im
//
// This fused operation avoids intermediate memory writes, keeping temp values
// in registers for significant speedup in FFT inner loops. It is the float64
// counterpart of f32.ButterflyComplex.
//
// Processes min(len(upperRe), len(upperIm), len(lowerRe), len(lowerIm), len(twRe), len(twIm)) elements.
// All slices are modified in-place: upper receives upper+temp, lower receives upper-temp.
//
// Aliasing: each butterfly reads its inputs before writing either output, so the
// four data slices are updated in place. They must be four distinct slices, and
// the twiddles twRe/twIm must not overlap any of them.
//
// Uses AVX+FMA on AMD64, NEON on ARM64, with a pure Go fallback.
func ButterflyComplex(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm []float64) {
	n := minLen6(len(upperRe), len(upperIm), len(lowerRe), len(lowerIm), len(twRe), len(twIm))
	if n == 0 {
		return
	}
	butterflyComplex64(upperRe[:n], upperIm[:n], lowerRe[:n], lowerIm[:n], twRe[:n], twIm[:n])
}

// butterflyStageRadix is the radix of a decimation-in-time stage: every block
// holds exactly one upper half and one lower half, each span elements long, so
// a block is butterflyStageRadix*span elements wide.
const butterflyStageRadix = 2

// ButterflyComplexStage applies one complete radix-2 decimation-in-time stage
// in place over split-complex data. For every block k in steps of 2*span, and
// every j in [0, span), it performs the ButterflyComplex operation on the pair
// (k+j, k+span+j) with twiddle (twRe[j], twIm[j]):
//
//	temp_re = re[k+span+j]*twRe[j] - im[k+span+j]*twIm[j]
//	temp_im = re[k+span+j]*twIm[j] + im[k+span+j]*twRe[j]
//	re[k+j], re[k+span+j] = re[k+j]+temp_re, re[k+j]-temp_re
//	im[k+j], im[k+span+j] = im[k+j]+temp_im, im[k+j]-temp_im
//
// It is the stage-level form of [ButterflyComplex]. Driving a stage through
// ButterflyComplex costs one call per block, so the call count grows as the
// runs get short and per-call overhead dominates the small-span stages. Taking
// the whole stage lets the implementation pick its own vectorization axis,
// which is not expressible through the per-block API: across j when span is
// large enough to fill a vector, and across blocks when span is short enough
// that a whole vector of them fits. A span between the two fills neither axis
// (span 3 on the 4-wide AVX path) and runs the scalar tail.
//
// The stage is a no-op unless span > 0, len(twRe) >= span and len(twIm) >= span,
// and it is also a no-op when no complete block fits. Blocks are processed while
// k+2*span <= n, where n = min(len(re), len(im)); a trailing partial block is
// left untouched.
//
// Aliasing: the stage updates re and im in place; they must be distinct buffers,
// and the twiddles twRe/twIm must not overlap either.
//
// Uses AVX+FMA on AMD64, NEON on ARM64, with a pure Go fallback. As with
// [ButterflyComplex], results are not guaranteed bit-identical between the
// vector and fallback paths, so do not depend on exact equality across build
// targets or across spans.
func ButterflyComplexStage(re, im []float64, span int, twRe, twIm []float64) {
	if span <= 0 || len(twRe) < span || len(twIm) < span {
		return
	}
	n := min(len(re), len(im))
	blockLen := butterflyStageRadix * span
	blocks := n / blockLen
	if blocks == 0 {
		return
	}
	used := blocks * blockLen
	butterflyComplexStage64(re[:used], im[:used], span, blocks, twRe[:span], twIm[:span])
}

// butterflyStage4Radix is the radix of the radix-4 decimation-in-time stage:
// every block holds four span-length sub-vectors x0..x3, so a block is
// butterflyStage4Radix*span elements wide.
const butterflyStage4Radix = 4

// ButterflyComplexStage4 applies one complete radix-4 decimation-in-time stage in
// place over split-complex data. For every block k in steps of 4*span, and every
// j in [0, span), it combines the four sub-vectors x0..x3 at (k+j, k+span+j,
// k+2*span+j, k+3*span+j) using the three twiddles (tw1[j], tw2[j], tw3[j]):
//
//	t1 = x1*tw1[j]; t2 = x2*tw2[j]; t3 = x3*tw3[j]   (full complex multiplies)
//	a = x0 + t1; b = x0 - t1; c = t2 + t3; d = t2 - t3
//	re[k+j]        = a.re + c.re; im[k+j]        = a.im + c.im
//	re[k+span+j]   = b.re + d.im; im[k+span+j]   = b.im - d.re
//	re[k+2*span+j] = a.re - c.re; im[k+2*span+j] = a.im - c.im
//	re[k+3*span+j] = b.re - d.im; im[k+3*span+j] = b.im + d.re
//
// It is the radix-4 counterpart of [ButterflyComplexStage]: one radix-4 stage at
// span s does the work of two radix-2 stages, first at span s then at 2*s, in a
// single pass over the data. Folding two stages into one halves the passes over
// the array and lets the implementation combine all four points while they are
// still in registers, which is where a radix-4 core wins over a radix-2 one on a
// memory-bound transform.
//
// # Twiddle convention
//
// The three twiddle slices are the powers of w = exp(-2*pi*i/(4*span)):
//
//	tw1[j] = w^(2j)   (applied to x1)
//	tw2[j] = w^j      (applied to x2)
//	tw3[j] = w^(3j)   (applied to x3)
//
// The kernel itself is value-agnostic; those are the values a Cooley-Tukey
// transform must supply. Two properties are load-bearing and intentional, not
// mistakes to "fix":
//
//   - x1 carries w^(2j), not w^j. The input to a decimation-in-time stage is
//     radix-2 bit-reversed, which swaps sub-vectors 1 and 2 relative to a plain
//     radix-4 layout, so x1 takes the square of the base twiddle. With this
//     convention tw1 is exactly the radix-2 span-s twiddle table (exp(-i*pi*j/s))
//     and tw2 is the first s entries of the radix-2 span-2s table, so a caller
//     already holding the radix-2 tables reuses them verbatim.
//   - The -i and +i cross-adds on the (k+span+j) and (k+3*span+j) outputs
//     hard-code the FORWARD (negative-exponent) DFT direction. An inverse
//     transform is a forward pass wrapped in conjugation of the input and output.
//
// The stage is a no-op unless span > 0 and all six twiddle slices have length at
// least span, and it is also a no-op when no complete block fits. Blocks are
// processed while k+4*span <= n, where n = min(len(re), len(im)); a trailing
// partial block is left untouched. The span-2 case is never reached by a radix-4
// Cooley-Tukey schedule (spans run 1, 4, 16, ...), so this stage does not carry a
// dedicated span-2 vector path; a span-2 caller still gets a correct result from
// the general j-axis path (a 2-wide NEON iteration, or the scalar tail of the
// 4-wide AVX path).
//
// Uses AVX+FMA on AMD64, NEON on ARM64, with a pure Go fallback. As with
// [ButterflyComplexStage], results are not guaranteed bit-identical between the
// vector and fallback paths, so do not depend on exact equality across build
// targets or across spans.
func ButterflyComplexStage4(re, im []float64, span int,
	tw1Re, tw1Im, tw2Re, tw2Im, tw3Re, tw3Im []float64) {
	if span <= 0 ||
		len(tw1Re) < span || len(tw1Im) < span ||
		len(tw2Re) < span || len(tw2Im) < span ||
		len(tw3Re) < span || len(tw3Im) < span {
		return
	}
	n := min(len(re), len(im))
	blockLen := butterflyStage4Radix * span
	blocks := n / blockLen
	if blocks == 0 {
		return
	}
	used := blocks * blockLen
	butterflyComplexStage4x64(re[:used], im[:used], span, blocks,
		tw1Re[:span], tw1Im[:span], tw2Re[:span], tw2Im[:span], tw3Re[:span], tw3Im[:span])
}

// realFFTUnpackMinN is the minimum n required for RealFFTUnpack (need at least 2 bins).
const realFFTUnpackMinN = 2

// RealFFTUnpack performs the unpacking step of a real-valued FFT.
//
// Given Z = FFT(packed real data of size 2n), this computes the real FFT output X
// for bins k = 1 to n-1. The formula for each bin is:
//
//	conj_z = conj(Z[n-k])
//	even = 0.5 * (Z[k] + conj_z)
//	diff = Z[k] - conj_z
//	odd  = W[k] * (-0.5i) * diff
//	X[k] = even + odd
//
// Expanding the complex arithmetic:
//
//	evenRe = 0.5 * (zRe[k] + zRe[n-k])
//	evenIm = 0.5 * (zIm[k] - zIm[n-k])
//	diffRe = zRe[k] - zRe[n-k]
//	diffIm = zIm[k] + zIm[n-k]
//	oddRe  = 0.5 * (twRe[k-1]*diffIm + twIm[k-1]*diffRe)
//	oddIm  = 0.5 * (twIm[k-1]*diffIm - twRe[k-1]*diffRe)
//	outRe[k] = evenRe + oddRe
//	outIm[k] = evenIm + oddIm
//
// Parameters:
//   - outRe, outIm: Output arrays, X[k] written to index k for k in [1, n-1]
//   - zRe, zIm: Input Z array of length n
//   - twRe, twIm: Twiddle factors W[k] at index k-1 (length n-1)
//
// The DC bin (k=0) and Nyquist bin (k=n) must be handled separately by the caller.
// Typical real FFT post-processing:
//
//	X[0] = Z[0].real + Z[0].imag  (DC component)
//	X[n] = Z[0].real - Z[0].imag  (Nyquist component)
//
// # Aliasing
//
// outRe and outIm must not overlap each other, nor any of zRe, zIm, twRe or
// twIm, in any way, not even as an exact element-for-element overlay. Bin k
// reads z at both k and the mirror n-k, plus the twiddles at k-1, before it
// writes out[k], so any overlay lets a store land on an input that a later bin
// has not read yet. Measured on the pure Go path, an output overlaid on z first
// corrupts bin floor(n/2)+1 and every bin above it. The vector kernels survive a
// few more sizes, because a SIMD block loads its whole input block before
// storing any output lane, but that is block scheduling rather than a guarantee
// and it varies with kernel width and n. Passing the same slice as outRe and
// outIm makes every bin wrong on every path. Pass outputs distinct from the
// inputs and from each other.
//
// It is the float64 counterpart of f32.RealFFTUnpack.
//
// Uses AVX2+FMA on AMD64, NEON on ARM64, with a pure Go fallback.
func RealFFTUnpack(outRe, outIm, zRe, zIm, twRe, twIm []float64) {
	n := len(zRe)
	if n < realFFTUnpackMinN {
		return
	}
	// Validate slice lengths
	if len(zIm) < n || len(outRe) < n || len(outIm) < n || len(twRe) < n-1 || len(twIm) < n-1 {
		return
	}
	realFFTUnpack64(outRe, outIm, zRe, zIm, twRe, twIm, n)
}

// RealFFTPower writes the power spectrum dst[k] = |X[k]|^2 for k in [1, n-1] of a
// packed real-input FFT, given the half-size complex spectrum Z (length n) and the
// unpack twiddles, without materialising the complex bins X.
//
// It is the fused, power-writing counterpart of RealFFTUnpack: identical inputs
// and identical bin range, but it emits |X_k|^2 directly. Each bin unpacks exactly
// as RealFFTUnpack does and is then squared and summed in registers:
//
//	evenRe = 0.5*(zRe[k] + zRe[n-k]); evenIm = 0.5*(zIm[k] - zIm[n-k])
//	diffRe = zRe[k] - zRe[n-k];       diffIm = zIm[k] + zIm[n-k]
//	oddRe  = 0.5*(twRe[k-1]*diffIm + twIm[k-1]*diffRe)
//	oddIm  = 0.5*(twIm[k-1]*diffIm - twRe[k-1]*diffRe)
//	X[k].re = evenRe + oddRe; X[k].im = evenIm + oddIm
//	dst[k]  = X[k].re^2 + X[k].im^2
//
// so it makes a single pass over the spectrum with no intermediate complex bins. A
// spectrogram, mel front end, or PSD consumer wants |X_k|^2, and computing it as
// RealFFTUnpack + Mul + FMA is three passes over the bins that write and re-read
// the complex half-spectrum; folding the magnitude-squared into the unpack drops
// the two extra passes. See issue #198.
//
// Parameters:
//   - dst: output power, dst[k] = |X[k]|^2 written for k in [1, n-1] (length >= n)
//   - zRe, zIm: half-size complex spectrum Z, length n
//   - twRe, twIm: twiddle factors W[k] at index k-1 (length n-1), where
//     W[k] = exp(-i*pi*k/n) = cos(pi*k/n) - i*sin(pi*k/n)
//
// The DC bin (k=0) and Nyquist bin (k=n) are the caller's responsibility, exactly
// as for RealFFTUnpack. Both are real, so their powers are:
//
//	|X[0]|^2 = (Z[0].real + Z[0].imag)^2  (DC)
//	|X[n]|^2 = (Z[0].real - Z[0].imag)^2  (Nyquist)
//
// The SIMD kernels fuse the magnitude-squared with a hardware FMA on their vector
// lanes (single rounding). The pure-Go path writes a separate multiply and add,
// but the Go compiler contracts that into an FMA on some architectures (arm64) and
// not others (amd64). So the results agree only to within rounding, not
// bit-for-bit, and the exact bits can differ across architectures, exactly as the
// RealFFTUnpack odd-term FMA already does.
//
// # Aliasing
//
// dst must not overlap zRe, zIm, twRe or twIm in any way, not even as an exact
// element-for-element overlay. Bin k reads Z at both k and the mirror n-k plus the
// twiddle at k-1 before it writes dst at k, and the SIMD kernels re-read the tail
// input with an overlapping block, so any overlay lets a store land on an input a
// later bin has not read yet. Same precondition as RealFFTUnpack.
//
// Uses AVX2+FMA on AMD64, NEON on ARM64, with a pure Go fallback.
func RealFFTPower(dst, zRe, zIm, twRe, twIm []float64) {
	n := len(zRe)
	if n < realFFTUnpackMinN {
		return
	}
	// Validate slice lengths.
	if len(zIm) < n || len(dst) < n || len(twRe) < n-1 || len(twIm) < n-1 {
		return
	}
	realFFTPower64(dst, zRe, zIm, twRe, twIm, n)
}
