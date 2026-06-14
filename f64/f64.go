// Package f64 provides SIMD-accelerated operations on float64 slices.
//
// All functions automatically select the optimal implementation based on
// runtime CPU feature detection. Functions gracefully fall back to pure Go
// implementations on unsupported architectures.
//
// Thread Safety: All functions are safe for concurrent use.
// Memory: All functions are zero-allocation (no heap allocations).
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
// Uses AVX+FMA on AMD64 (4x float64), NEON on ARM64 (2x float64).
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
// Uses AVX+FMA on AMD64 (4x float64), NEON on ARM64 (2x float64).
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
// Uses AVX on AMD64 (4x float64), NEON on ARM64 (2x float64).
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
// Uses AVX on AMD64 (2x float64), NEON on ARM64 (2x float64), and falls back
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
