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
	n := min(len(b), len(a))
	if n == 0 {
		return 0
	}
	return dotProduct(a[:n], b[:n])
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
