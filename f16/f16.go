// Package f16 provides SIMD-accelerated operations on float16 (half-precision) slices.
//
// Float16 is stored as uint16 in IEEE 754 half-precision format (1 sign bit,
// 5 exponent bits, 10 mantissa bits). This provides ~3.3 decimal digits of
// precision with a dynamic range of ~6×10⁻⁸ to 65504.
//
// Key benefits of float16:
//   - 2x memory bandwidth efficiency vs float32
//   - 2x elements per SIMD register (8 per NEON 128-bit vector)
//   - Sufficient precision for ML inference, audio DSP, and many signal processing tasks
//
// On ARM64 with FP16 support (Apple Silicon, Cortex-A55+), operations use native
// half-precision SIMD instructions. On other platforms, operations convert to
// float32 for computation.
//
// Reductions (Sum, DotProduct) accumulate in float32 for numerical stability.
//
// Thread Safety: All functions are safe for concurrent use.
// Memory: All functions are zero-allocation (no heap allocations).
package f16

import "math"

const (
	// normEpsilon is the threshold below which a vector magnitude is considered zero.
	normEpsilon = 1e-7
	// interleaveChannels is the number of channels for interleave/deinterleave operations.
	interleaveChannels = 2
)

// Float16 is a 16-bit IEEE 754 half-precision floating-point number.
// Stored as uint16, use ToFloat32/FromFloat32 for conversion.
type Float16 = uint16

// ToFloat32 converts a Float16 to float32.
func ToFloat32(h Float16) float32 {
	return toFloat32(h)
}

// FromFloat32 converts a float32 to Float16.
func FromFloat32(f float32) Float16 {
	return fromFloat32(f)
}

// ToFloat32Slice converts a slice of Float16 to float32.
// dst and src may overlap only if they start at the same address.
func ToFloat32Slice(dst []float32, src []Float16) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	toFloat32Slice(dst[:n], src[:n])
}

// FromFloat32Slice converts a slice of float32 to Float16.
// dst and src may overlap only if they start at the same address.
func FromFloat32Slice(dst []Float16, src []float32) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	fromFloat32Slice(dst[:n], src[:n])
}

// DotProduct computes the dot product of two Float16 slices.
// Returns sum(a[i] * b[i]) for i in 0..min(len(a), len(b)).
// Accumulates in float32 for numerical stability.
//
// Uses native FP16 SIMD on ARM64 with FP16 support. On ARM64 with FP16
// SIMD, intermediate products are computed in FP16 and may saturate to
// ±Inf when |a[i] * b[i]| > 65504. Use DotProductF32 if the dynamic range
// of your inputs can produce such products.
func DotProduct(a, b []Float16) float32 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	return dotProduct(a, b)
}

// DotProductF32 computes the dot product with FP32 widening before multiply.
//
// Unlike DotProduct, intermediate products do not saturate when |a[i] * b[i]|
// exceeds the FP16 representable maximum (~65504). The cost on ARM64 with
// native FP16 SIMD is roughly 1.5-2x more work than DotProduct.
//
// Use this for audio/signal-processing inputs whose per-element products may
// fall outside the FP16 range. For ML workloads with inputs normalized to
// [-1, 1], DotProduct is faster with equivalent accuracy on every supported
// platform.
//
// Returns sum(float32(a[i]) * float32(b[i])) for i in 0..min(len(a), len(b)),
// or 0 if either slice is empty. Accumulates in float32 for numerical
// stability.
func DotProductF32(a, b []Float16) float32 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	return dotProductF32(a, b)
}

// Add computes element-wise addition: dst[i] = a[i] + b[i].
func Add(dst, a, b []Float16) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	add(dst[:n], a[:n], b[:n])
}

// Sub computes element-wise subtraction: dst[i] = a[i] - b[i].
func Sub(dst, a, b []Float16) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	sub(dst[:n], a[:n], b[:n])
}

// Mul computes element-wise multiplication: dst[i] = a[i] * b[i].
func Mul(dst, a, b []Float16) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	mul(dst[:n], a[:n], b[:n])
}

// Scale multiplies each element by a scalar: dst[i] = a[i] * s.
func Scale(dst, a []Float16, s Float16) {
	n := min(len(a), len(dst))
	if n == 0 {
		return
	}
	scale(dst[:n], a[:n], s)
}

// FMA computes fused multiply-add: dst[i] = a[i] * b[i] + c[i].
func FMA(dst, a, b, c []Float16) {
	n := min(len(c), minLen(len(dst), len(a), len(b)))
	if n == 0 {
		return
	}
	fma16(dst[:n], a[:n], b[:n], c[:n])
}

// Sum returns the sum of all elements.
// Accumulates in float32 for numerical stability.
func Sum(a []Float16) float32 {
	if len(a) == 0 {
		return 0
	}
	return sum(a)
}

// Abs computes element-wise absolute value: dst[i] = |a[i]|.
func Abs(dst, a []Float16) {
	n := min(len(a), len(dst))
	if n == 0 {
		return
	}
	abs16(dst[:n], a[:n])
}

// Neg computes element-wise negation: dst[i] = -a[i].
func Neg(dst, a []Float16) {
	n := min(len(a), len(dst))
	if n == 0 {
		return
	}
	neg16(dst[:n], a[:n])
}

// ReLU computes the Rectified Linear Unit: dst[i] = max(0, src[i]).
func ReLU(dst, src []Float16) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	relu16(dst[:n], src[:n])
}

// Sigmoid computes the sigmoid activation function: dst[i] = 1 / (1 + e^(-src[i])).
func Sigmoid(dst, src []Float16) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	sigmoid16(dst[:n], src[:n])
}

// Min returns the minimum value.
func Min(a []Float16) Float16 {
	if len(a) == 0 {
		return fromFloat32Go(float32(math.Inf(1)))
	}
	return min16(a)
}

// Max returns the maximum value.
func Max(a []Float16) Float16 {
	if len(a) == 0 {
		return fromFloat32Go(float32(math.Inf(-1)))
	}
	return max16(a)
}

// Mean computes the arithmetic mean of a slice.
// Returns 0 for empty slices.
func Mean(a []Float16) float32 {
	if len(a) == 0 {
		return 0
	}
	return Sum(a) / float32(len(a))
}

// Div computes element-wise division: dst[i] = a[i] / b[i].
func Div(dst, a, b []Float16) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	div16(dst[:n], a[:n], b[:n])
}

// AddScalar adds a scalar to each element: dst[i] = a[i] + s.
func AddScalar(dst, a []Float16, s Float16) {
	n := min(len(a), len(dst))
	if n == 0 {
		return
	}
	addScalar16(dst[:n], a[:n], s)
}

// Clamp clamps each element to [minVal, maxVal].
func Clamp(dst, a []Float16, minVal, maxVal Float16) {
	n := min(len(a), len(dst))
	if n == 0 {
		return
	}
	clamp16(dst[:n], a[:n], minVal, maxVal)
}

// Sqrt computes element-wise square root: dst[i] = sqrt(a[i]).
func Sqrt(dst, a []Float16) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	sqrt16(dst[:n], a[:n])
}

// Reciprocal computes element-wise reciprocal: dst[i] = 1/a[i].
func Reciprocal(dst, a []Float16) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	reciprocal16(dst[:n], a[:n])
}

// Exp computes element-wise exponential: dst[i] = e^src[i].
func Exp(dst, src []Float16) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	exp16(dst[:n], src[:n])
}

// ExpInPlace computes exp in-place: a[i] = e^a[i].
func ExpInPlace(a []Float16) {
	if len(a) == 0 {
		return
	}
	exp16(a, a)
}

// Tanh computes element-wise hyperbolic tangent: dst[i] = tanh(src[i]).
func Tanh(dst, src []Float16) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	tanh16(dst[:n], src[:n])
}

// TanhInPlace computes tanh in-place: a[i] = tanh(a[i]).
func TanhInPlace(a []Float16) {
	if len(a) == 0 {
		return
	}
	tanh16(a, a)
}

// MinIdx returns the index of the minimum value in the slice.
// Returns -1 for empty slices.
func MinIdx(a []Float16) int {
	if len(a) == 0 {
		return -1
	}
	return minIdx16(a)
}

// MaxIdx returns the index of the maximum value in the slice.
// Returns -1 for empty slices.
func MaxIdx(a []Float16) int {
	if len(a) == 0 {
		return -1
	}
	return maxIdx16(a)
}

// AddScaled adds scaled values to dst: dst[i] += alpha * s[i].
// This is the AXPY operation from BLAS Level 1.
func AddScaled(dst []Float16, alpha Float16, s []Float16) {
	n := min(len(dst), len(s))
	if n == 0 {
		return
	}
	addScaled16(dst[:n], alpha, s[:n])
}

// Normalize normalizes a vector to unit length: dst = a / ||a||.
// If the magnitude is zero or very small, copies the input unchanged.
func Normalize(dst, a []Float16) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}

	// Compute magnitude using dot product with itself
	mag := float32(math.Sqrt(float64(DotProduct(a[:n], a[:n]))))

	// Avoid division by zero
	if mag < normEpsilon {
		copy(dst[:n], a[:n])
		return
	}

	// Scale by 1/magnitude
	invMag := FromFloat32(1.0 / mag)
	Scale(dst[:n], a[:n], invMag)
}

// EuclideanDistance computes the Euclidean distance between two vectors.
// Returns sqrt(sum((a[i] - b[i])^2)).
func EuclideanDistance(a, b []Float16) float32 {
	n := min(len(a), len(b))
	if n == 0 {
		return 0
	}
	return euclideanDistance16(a[:n], b[:n])
}

// Variance computes the population variance of a slice.
// Returns 0 for empty slices.
func Variance(a []Float16) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}
	mean := Mean(a)
	return variance16(a, mean)
}

// StdDev computes the population standard deviation of a slice.
// Returns 0 for empty slices.
func StdDev(a []Float16) float32 {
	return float32(math.Sqrt(float64(Variance(a))))
}

// CumulativeSum computes the cumulative sum: dst[i] = sum(a[0:i+1]).
func CumulativeSum(dst, a []Float16) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	cumulativeSum16(dst[:n], a[:n])
}

// DotProductBatch computes multiple dot products against the same vector.
// results[i] = DotProduct(rows[i], vec) for each row.
func DotProductBatch(results []float32, rows [][]Float16, vec []Float16) {
	n := min(len(results), len(rows))
	if n == 0 || len(vec) == 0 {
		return
	}
	dotProductBatch16(results[:n], rows[:n], vec)
}

// AccumulateAdd adds src to dst: dst[i] += src[i].
// This is useful for overlap-add operations.
func AccumulateAdd(dst, src []Float16, offset int) {
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
	accumulateAdd16(dst[offset:offset+n], src)
}

// ConvolveValid computes valid convolution of signal with kernel.
// dst[i] = sum(signal[i+j] * kernel[j]) for j in 0..len(kernel)-1.
func ConvolveValid(dst, signal, kernel []Float16) {
	if len(kernel) == 0 || len(signal) < len(kernel) {
		return
	}
	validLen := len(signal) - len(kernel) + 1
	n := min(len(dst), validLen)
	if n == 0 {
		return
	}
	convolveValid16(dst[:n], signal, kernel)
}

// Interleave2 interleaves two slices: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], ...
func Interleave2(dst, a, b []Float16) {
	n := min(len(dst)/interleaveChannels, min(len(a), len(b)))
	if n == 0 {
		return
	}
	interleave2_16(dst[:n*interleaveChannels], a[:n], b[:n])
}

// Deinterleave2 deinterleaves a slice: a[0]=src[0], b[0]=src[1], a[1]=src[2], ...
func Deinterleave2(a, b, src []Float16) {
	n := min(len(src)/interleaveChannels, min(len(a), len(b)))
	if n == 0 {
		return
	}
	deinterleave2_16(a[:n], b[:n], src[:n*interleaveChannels])
}

// ClampScale performs fused clamp and scale: dst[i] = (clamp(src[i], min, max) - min) * scale.
func ClampScale(dst, src []Float16, minVal, maxVal, scale Float16) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	clampScale16(dst[:n], src[:n], minVal, maxVal, scale)
}

// ReLUInPlace computes ReLU in-place: a[i] = max(0, a[i]).
func ReLUInPlace(a []Float16) {
	if len(a) == 0 {
		return
	}
	relu16(a, a)
}

// SigmoidInPlace computes sigmoid in-place: a[i] = 1/(1+e^(-a[i])).
func SigmoidInPlace(a []Float16) {
	if len(a) == 0 {
		return
	}
	sigmoid16(a, a)
}

// DotProductUnsafe computes the dot product without length validation.
// PRECONDITIONS: len(a) == len(b), len(a) > 0
func DotProductUnsafe(a, b []Float16) float32 {
	return dotProduct(a, b)
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
