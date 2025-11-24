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

// DotProductUnsafe computes the dot product without length validation.
// This is a low-overhead variant for performance-critical code paths.
//
// PRECONDITIONS (caller must ensure):
//   - len(a) == len(b)
//   - len(a) > 0
//
// Violating these preconditions results in undefined behavior (panic or incorrect results).
// Use DotProduct for safe operation with automatic length handling.
func DotProductUnsafe(a, b []float64) float64 {
	return dotProduct(a, b)
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

// Sum returns the sum of all elements in the slice.
func Sum(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	return sum(a)
}

// Min returns the minimum value in the slice.
// Returns +Inf for empty slices.
func Min(a []float64) float64 {
	if len(a) == 0 {
		return posInf
	}
	return min64(a)
}

// Max returns the maximum value in the slice.
// Returns -Inf for empty slices.
func Max(a []float64) float64 {
	if len(a) == 0 {
		return negInf
	}
	return max64(a)
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
// Uses polynomial approximation for reasonable accuracy and performance.
// Processes min(len(dst), len(src)) elements.
//
// Uses AVX+FMA on AMD64 (4x float64), NEON on ARM64 (2x float64).
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
