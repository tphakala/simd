package f64

// Pure Go reference implementations for validating SIMD operations.
// These are intentionally simple and obviously correct.

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// Reference implementations (simple loops, obviously correct)
// =============================================================================

func dotProductRef(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func sumRef(a []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i]
	}
	return sum
}

func minRef(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	result := a[0]
	for i := 1; i < len(a); i++ {
		if a[i] < result {
			result = a[i]
		}
	}
	return result
}

func maxRef(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	result := a[0]
	for i := 1; i < len(a); i++ {
		if a[i] > result {
			result = a[i]
		}
	}
	return result
}

func addRef(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func subRef(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func mulRef(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func divRef(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

func scaleRef(dst, a []float64, s float64) {
	for i := range dst {
		dst[i] = a[i] * s
	}
}

func addScalarRef(dst, a []float64, s float64) {
	for i := range dst {
		dst[i] = a[i] + s
	}
}

func absRef(dst, a []float64) {
	for i := range dst {
		dst[i] = math.Abs(a[i])
	}
}

func negRef(dst, a []float64) {
	for i := range dst {
		dst[i] = -a[i]
	}
}

func fmaRef(dst, a, b, c []float64) {
	for i := range dst {
		dst[i] = a[i]*b[i] + c[i]
	}
}

func clampRef(dst, a []float64, minVal, maxVal float64) {
	for i := range dst {
		v := a[i]
		if v < minVal {
			v = minVal
		}
		if v > maxVal {
			v = maxVal
		}
		dst[i] = v
	}
}

func sqrtRef(dst, a []float64) {
	for i := range dst {
		dst[i] = math.Sqrt(a[i])
	}
}

func reciprocalRef(dst, a []float64) {
	for i := range dst {
		dst[i] = 1.0 / a[i]
	}
}

func meanRef(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	return sumRef(a) / float64(len(a))
}

func varianceRef(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	mean := meanRef(a)
	var sum float64
	for i := range a {
		diff := a[i] - mean
		sum += diff * diff
	}
	return sum / float64(len(a))
}

func euclideanDistanceRef(a, b []float64) float64 {
	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func normalizeRef(dst, a []float64) {
	var magnitude float64
	for i := range a {
		magnitude += a[i] * a[i]
	}
	magnitude = math.Sqrt(magnitude)
	if magnitude < 1e-10 {
		copy(dst, a)
		return
	}
	invMag := 1.0 / magnitude
	for i := range a {
		dst[i] = a[i] * invMag
	}
}

func cumulativeSumRef(dst, a []float64) {
	var sum float64
	for i := range a {
		sum += a[i]
		dst[i] = sum
	}
}

// =============================================================================
// Test helpers
// =============================================================================

const refTolerance64 = 1e-10

func refAlmostEqual64(a, b float64) bool {
	if math.IsNaN(a) && math.IsNaN(b) {
		return true
	}
	if math.IsInf(a, 0) && math.IsInf(b, 0) {
		return math.Signbit(a) == math.Signbit(b)
	}
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff <= refTolerance64
}

func refSlicesEqual64(got, want []float64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if !refAlmostEqual64(got[i], want[i]) {
			return false
		}
	}
	return true
}

// makeTestData creates test vectors with predictable values
func makeTestData64(n int) (a, b, c []float64) {
	a = make([]float64, n)
	b = make([]float64, n)
	c = make([]float64, n)
	for i := range n {
		a[i] = float64(i + 1)
		b[i] = float64(n - i)
		c[i] = 0.5
	}
	return
}

// makeMixedSigns creates vectors with negative and positive values
func makeMixedSigns64(n int) (a, b []float64) {
	a = make([]float64, n)
	b = make([]float64, n)
	half := n / 2
	for i := range n {
		if i < half {
			a[i] = -float64(half - i)
			b[i] = float64(half - i)
		} else {
			a[i] = float64(i - half + 1)
			b[i] = -float64(i - half + 1)
		}
	}
	return
}

// =============================================================================
// Reference validation tests
// =============================================================================

// TestDotProduct_Ref validates SIMD DotProduct against pure Go reference
func TestDotProduct_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 256, 1000}

	for _, n := range sizes {
		a, b, _ := makeTestData64(n)

		got := DotProduct(a, b)
		want := dotProductRef(a, b)

		assert.InDelta(t, want, got, refTolerance64, "DotProduct n=%d", n)
	}

	t.Run("mixed_signs", func(t *testing.T) {
		a, b := makeMixedSigns64(10)
		got := DotProduct(a, b)
		want := dotProductRef(a, b)
		assert.InDelta(t, want, got, refTolerance64, "DotProduct mixed signs")
	})
}

// TestSum_Ref validates SIMD Sum against pure Go reference
func TestSum_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 256, 1000}

	for _, n := range sizes {
		a, _, _ := makeTestData64(n)

		got := Sum(a)
		want := sumRef(a)

		assert.InDelta(t, want, got, refTolerance64, "Sum n=%d", n)
	}
}

// TestMinMax_Ref validates SIMD Min/Max against pure Go reference
func TestMinMax_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 256}

	for _, n := range sizes {
		a, _, _ := makeTestData64(n)

		gotMin := Min(a)
		wantMin := minRef(a)
		assert.InDelta(t, wantMin, gotMin, refTolerance64, "Min n=%d", n)

		gotMax := Max(a)
		wantMax := maxRef(a)
		assert.InDelta(t, wantMax, gotMax, refTolerance64, "Max n=%d", n)
	}
}

// TestAdd_Ref validates SIMD Add against pure Go reference
func TestAdd_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33}

	for _, n := range sizes {
		a, b, _ := makeTestData64(n)
		got := make([]float64, n)
		want := make([]float64, n)

		Add(got, a, b)
		addRef(want, a, b)

		require.Len(t, got, len(want), "Add n=%d length mismatch", n)
		assert.InDeltaSlice(t, want, got, refTolerance64, "Add n=%d", n)
	}
}

// TestSub_Ref validates SIMD Sub against pure Go reference
func TestSub_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17}

	for _, n := range sizes {
		a, b, _ := makeTestData64(n)
		got := make([]float64, n)
		want := make([]float64, n)

		Sub(got, a, b)
		subRef(want, a, b)

		assert.InDeltaSlice(t, want, got, refTolerance64, "Sub n=%d", n)
	}
}

// TestMul_Ref validates SIMD Mul against pure Go reference
func TestMul_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17}

	for _, n := range sizes {
		a, b, _ := makeTestData64(n)
		got := make([]float64, n)
		want := make([]float64, n)

		Mul(got, a, b)
		mulRef(want, a, b)

		assert.InDeltaSlice(t, want, got, refTolerance64, "Mul n=%d", n)
	}
}

// TestDiv_Ref validates SIMD Div against pure Go reference
func TestDiv_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17}

	for _, n := range sizes {
		a, b, _ := makeTestData64(n)
		got := make([]float64, n)
		want := make([]float64, n)

		Div(got, a, b)
		divRef(want, a, b)

		assert.InDeltaSlice(t, want, got, refTolerance64, "Div n=%d", n)
	}
}

// TestScale_Ref validates SIMD Scale against pure Go reference
func TestScale_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17}
	scalar := 2.5

	for _, n := range sizes {
		a, _, _ := makeTestData64(n)
		got := make([]float64, n)
		want := make([]float64, n)

		Scale(got, a, scalar)
		scaleRef(want, a, scalar)

		assert.InDeltaSlice(t, want, got, refTolerance64, "Scale n=%d", n)
	}
}

// TestAddScalar_Ref validates SIMD AddScalar against pure Go reference
func TestAddScalar_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17}
	scalar := 10.5

	for _, n := range sizes {
		a, _, _ := makeTestData64(n)
		got := make([]float64, n)
		want := make([]float64, n)

		AddScalar(got, a, scalar)
		addScalarRef(want, a, scalar)

		assert.InDeltaSlice(t, want, got, refTolerance64, "AddScalar n=%d", n)
	}
}

// TestAbs_Ref validates SIMD Abs against pure Go reference
func TestAbs_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17}

	for _, n := range sizes {
		a, _ := makeMixedSigns64(n)
		got := make([]float64, n)
		want := make([]float64, n)

		Abs(got, a)
		absRef(want, a)

		assert.InDeltaSlice(t, want, got, refTolerance64, "Abs n=%d", n)
	}
}

// TestNeg_Ref validates SIMD Neg against pure Go reference
func TestNeg_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17}

	for _, n := range sizes {
		a, _, _ := makeTestData64(n)
		got := make([]float64, n)
		want := make([]float64, n)

		Neg(got, a)
		negRef(want, a)

		assert.InDeltaSlice(t, want, got, refTolerance64, "Neg n=%d", n)
	}
}

// TestFMA_Ref validates SIMD FMA against pure Go reference
func TestFMA_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17}

	for _, n := range sizes {
		a, b, c := makeTestData64(n)
		got := make([]float64, n)
		want := make([]float64, n)

		FMA(got, a, b, c)
		fmaRef(want, a, b, c)

		assert.InDeltaSlice(t, want, got, refTolerance64, "FMA n=%d", n)
	}
}

// TestClamp_Ref validates SIMD Clamp against pure Go reference
func TestClamp_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17}
	minVal := 3.0
	maxVal := 10.0

	for _, n := range sizes {
		a, _, _ := makeTestData64(n)
		got := make([]float64, n)
		want := make([]float64, n)

		Clamp(got, a, minVal, maxVal)
		clampRef(want, a, minVal, maxVal)

		assert.InDeltaSlice(t, want, got, refTolerance64, "Clamp n=%d", n)
	}
}

// TestSqrt_Ref validates SIMD Sqrt against pure Go reference
func TestSqrt_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17}

	for _, n := range sizes {
		a, _, _ := makeTestData64(n)
		got := make([]float64, n)
		want := make([]float64, n)

		Sqrt(got, a)
		sqrtRef(want, a)

		assert.InDeltaSlice(t, want, got, refTolerance64, "Sqrt n=%d", n)
	}
}

// TestReciprocal_Ref validates SIMD Reciprocal against pure Go reference
func TestReciprocal_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17}

	for _, n := range sizes {
		a, _, _ := makeTestData64(n)
		got := make([]float64, n)
		want := make([]float64, n)

		Reciprocal(got, a)
		reciprocalRef(want, a)

		assert.InDeltaSlice(t, want, got, refTolerance64, "Reciprocal n=%d", n)
	}
}

// TestMean_Ref validates SIMD Mean against pure Go reference
func TestMean_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100, 256}

	for _, n := range sizes {
		a, _, _ := makeTestData64(n)

		got := Mean(a)
		want := meanRef(a)

		assert.InDelta(t, want, got, refTolerance64, "Mean n=%d", n)
	}
}

// TestVariance_Ref validates SIMD Variance against pure Go reference
func TestVariance_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100}

	for _, n := range sizes {
		a, _, _ := makeTestData64(n)

		got := Variance(a)
		want := varianceRef(a)

		// Variance can have larger numerical differences
		assert.InDelta(t, want, got, 1e-6, "Variance n=%d", n)
	}
}

// TestEuclideanDistance_Ref validates SIMD EuclideanDistance against pure Go reference
func TestEuclideanDistance_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100}

	for _, n := range sizes {
		a, b, _ := makeTestData64(n)

		got := EuclideanDistance(a, b)
		want := euclideanDistanceRef(a, b)

		assert.InDelta(t, want, got, refTolerance64, "EuclideanDistance n=%d", n)
	}
}

// TestNormalize_Ref validates SIMD Normalize against pure Go reference
func TestNormalize_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17}

	for _, n := range sizes {
		a, _, _ := makeTestData64(n)
		got := make([]float64, n)
		want := make([]float64, n)

		Normalize(got, a)
		normalizeRef(want, a)

		assert.InDeltaSlice(t, want, got, refTolerance64, "Normalize n=%d", n)
	}
}

// TestCumulativeSum_Ref validates CumulativeSum against pure Go reference
func TestCumulativeSum_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17}

	for _, n := range sizes {
		a, _, _ := makeTestData64(n)
		got := make([]float64, n)
		want := make([]float64, n)

		CumulativeSum(got, a)
		cumulativeSumRef(want, a)

		assert.InDeltaSlice(t, want, got, refTolerance64, "CumulativeSum n=%d", n)
	}
}

// =============================================================================
// Edge case tests for 100% coverage
// =============================================================================

// TestEdgeCases_Empty tests all functions with empty slices
func TestEdgeCases_Empty(t *testing.T) {
	empty := []float64{}
	dst := make([]float64, 10)

	t.Run("DotProduct_empty", func(t *testing.T) {
		got := DotProduct(empty, empty)
		assert.InDelta(t, 0.0, got, 1e-10, "DotProduct empty should return 0")
	})

	t.Run("Sum_empty", func(t *testing.T) {
		got := Sum(empty)
		assert.InDelta(t, 0.0, got, 1e-10, "Sum empty should return 0")
	})

	t.Run("Min_empty", func(t *testing.T) {
		got := Min(empty)
		assert.True(t, math.IsInf(got, 1), "Min empty should return +Inf")
	})

	t.Run("Max_empty", func(t *testing.T) {
		got := Max(empty)
		assert.True(t, math.IsInf(got, -1), "Max empty should return -Inf")
	})

	t.Run("Add_empty", func(t *testing.T) {
		assert.NotPanics(t, func() { Add(dst, empty, empty) })
	})

	t.Run("Sub_empty", func(t *testing.T) {
		assert.NotPanics(t, func() { Sub(dst, empty, empty) })
	})

	t.Run("Mul_empty", func(t *testing.T) {
		assert.NotPanics(t, func() { Mul(dst, empty, empty) })
	})

	t.Run("Div_empty", func(t *testing.T) {
		assert.NotPanics(t, func() { Div(dst, empty, empty) })
	})

	t.Run("Scale_empty", func(t *testing.T) {
		assert.NotPanics(t, func() { Scale(dst, empty, 2.0) })
	})

	t.Run("AddScalar_empty", func(t *testing.T) {
		assert.NotPanics(t, func() { AddScalar(dst, empty, 2.0) })
	})

	t.Run("Abs_empty", func(t *testing.T) {
		assert.NotPanics(t, func() { Abs(dst, empty) })
	})

	t.Run("Neg_empty", func(t *testing.T) {
		assert.NotPanics(t, func() { Neg(dst, empty) })
	})

	t.Run("FMA_empty", func(t *testing.T) {
		assert.NotPanics(t, func() { FMA(dst, empty, empty, empty) })
	})

	t.Run("Clamp_empty", func(t *testing.T) {
		assert.NotPanics(t, func() { Clamp(dst, empty, 0, 1) })
	})

	t.Run("Sqrt_empty", func(t *testing.T) {
		assert.NotPanics(t, func() { Sqrt(dst, empty) })
	})

	t.Run("Reciprocal_empty", func(t *testing.T) {
		assert.NotPanics(t, func() { Reciprocal(dst, empty) })
	})

	t.Run("Mean_empty", func(t *testing.T) {
		got := Mean(empty)
		assert.InDelta(t, 0.0, got, 1e-10, "Mean empty should return 0")
	})

	t.Run("Variance_empty", func(t *testing.T) {
		got := Variance(empty)
		assert.InDelta(t, 0.0, got, 1e-10, "Variance empty should return 0")
	})

	t.Run("StdDev_empty", func(t *testing.T) {
		got := StdDev(empty)
		assert.InDelta(t, 0.0, got, 1e-10, "StdDev empty should return 0")
	})

	t.Run("EuclideanDistance_empty", func(t *testing.T) {
		got := EuclideanDistance(empty, empty)
		assert.InDelta(t, 0.0, got, 1e-10, "EuclideanDistance empty should return 0")
	})

	t.Run("Normalize_empty", func(t *testing.T) {
		assert.NotPanics(t, func() { Normalize(dst, empty) })
	})

	t.Run("CumulativeSum_empty", func(t *testing.T) {
		assert.NotPanics(t, func() { CumulativeSum(dst, empty) })
	})

	t.Run("DotProductBatch_empty_results", func(t *testing.T) {
		assert.NotPanics(t, func() {
			DotProductBatch([]float64{}, [][]float64{{1, 2}}, []float64{1, 2})
		})
	})

	t.Run("DotProductBatch_empty_rows", func(t *testing.T) {
		assert.NotPanics(t, func() {
			DotProductBatch([]float64{1}, [][]float64{}, []float64{1, 2})
		})
	})

	t.Run("DotProductBatch_empty_vec", func(t *testing.T) {
		assert.NotPanics(t, func() {
			DotProductBatch([]float64{1}, [][]float64{{1, 2}}, []float64{})
		})
	})

	t.Run("ConvolveValid_empty_kernel", func(t *testing.T) {
		assert.NotPanics(t, func() {
			ConvolveValid(dst, []float64{1, 2, 3}, []float64{})
		})
	})

	t.Run("ConvolveValid_signal_smaller_than_kernel", func(t *testing.T) {
		assert.NotPanics(t, func() {
			ConvolveValid(dst, []float64{1}, []float64{1, 2, 3})
		})
	})

	t.Run("ConvolveValid_empty_dst", func(t *testing.T) {
		assert.NotPanics(t, func() {
			ConvolveValid([]float64{}, []float64{1, 2, 3, 4, 5}, []float64{1, 2})
		})
	})
}

// TestEdgeCases_MismatchedLengths tests functions with different length inputs
func TestEdgeCases_MismatchedLengths(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	b := []float64{1, 2, 3}
	c := []float64{1, 2}

	t.Run("DotProduct_mismatched", func(t *testing.T) {
		got := DotProduct(a, b)
		want := dotProductRef(a[:3], b)
		assert.InDelta(t, want, got, refTolerance64, "DotProduct mismatched lengths")
	})

	t.Run("Add_dst_smaller", func(t *testing.T) {
		dst := make([]float64, 2)
		Add(dst, a, b)
		want := []float64{a[0] + b[0], a[1] + b[1]}
		assert.InDeltaSlice(t, want, dst, refTolerance64, "Add with smaller dst")
	})

	t.Run("minLen_coverage", func(t *testing.T) {
		// Test all branches of minLen
		assert.Equal(t, 3, minLen(5, 3, 10), "minLen(5,3,10) b < a")
		assert.Equal(t, 2, minLen(5, 6, 2), "minLen(5,6,2) c < a")
		assert.Equal(t, 1, minLen(1, 5, 10), "minLen(1,5,10) a is smallest")
	})

	t.Run("FMA_mismatched", func(t *testing.T) {
		dst := make([]float64, 10)
		FMA(dst, a, b, c)
		// Should use min length of all inputs
		want := make([]float64, 2)
		for i := range 2 {
			want[i] = a[i]*b[i] + c[i]
		}
		assert.InDeltaSlice(t, want, dst[:2], refTolerance64, "FMA mismatched lengths")
	})

	t.Run("EuclideanDistance_mismatched", func(t *testing.T) {
		got := EuclideanDistance(a, b)
		want := euclideanDistanceRef(a[:3], b)
		assert.InDelta(t, want, got, refTolerance64, "EuclideanDistance mismatched lengths")
	})
}

// TestNormalize_ZeroMagnitude tests Normalize with near-zero vectors
func TestNormalize_ZeroMagnitude(t *testing.T) {
	// Vector with magnitude below threshold (1e-10)
	tiny := []float64{1e-12, 1e-12, 1e-12}
	dst := make([]float64, 3)

	Normalize(dst, tiny)

	// Should copy input unchanged
	assert.InDeltaSlice(t, tiny, dst, refTolerance64, "Normalize zero magnitude should copy input unchanged")
}

// TestDotProductBatch_Ref validates DotProductBatch against reference
func TestDotProductBatch_Ref(t *testing.T) {
	vec := []float64{1, 2, 3, 4}
	rows := [][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{1, 1, 1, 1},
		{2, 2, 2, 2},
	}
	results := make([]float64, len(rows))

	DotProductBatch(results, rows, vec)

	expected := []float64{1, 2, 10, 20}
	assert.InDeltaSlice(t, expected, results, refTolerance64, "DotProductBatch basic")

	t.Run("with_empty_row", func(t *testing.T) {
		rowsWithEmpty := [][]float64{
			{1, 2, 3, 4},
			{}, // empty row - should return 0
			{1, 1, 1, 1},
		}
		res := make([]float64, 3)
		DotProductBatch(res, rowsWithEmpty, vec)
		want := []float64{30, 0, 10}
		assert.InDeltaSlice(t, want, res, refTolerance64, "DotProductBatch with empty row")
	})

	t.Run("results_smaller", func(t *testing.T) {
		smallResults := make([]float64, 2)
		DotProductBatch(smallResults, rows, vec)
		assert.InDeltaSlice(t, expected[:2], smallResults, refTolerance64, "DotProductBatch results smaller than rows")
	})

	t.Run("varying_row_lengths", func(t *testing.T) {
		mixedRows := [][]float64{
			{1, 2, 3, 4, 5}, // longer than vec
			{1, 2},          // shorter than vec
			{1, 2, 3, 4},    // same as vec
		}
		res := make([]float64, 3)
		DotProductBatch(res, mixedRows, vec)
		want := []float64{30, 5, 30} // 1*1+2*2+3*3+4*4=30, 1*1+2*2=5, 1*1+2*2+3*3+4*4=30
		assert.InDeltaSlice(t, want, res, refTolerance64, "DotProductBatch varying row lengths")
	})
}

// TestConvolveValid_Ref validates ConvolveValid against reference
func TestConvolveValid_Ref(t *testing.T) {
	signal := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	kernel := []float64{1, 2, 1}

	// Expected output length: len(signal) - len(kernel) + 1 = 8 - 3 + 1 = 6
	dst := make([]float64, 6)
	ConvolveValid(dst, signal, kernel)

	// Compute expected using pure Go
	expected := make([]float64, 6)
	for i := range 6 {
		for j := range kernel {
			expected[i] += signal[i+j] * kernel[j]
		}
	}

	assert.InDeltaSlice(t, expected, dst, refTolerance64, "ConvolveValid basic")

	t.Run("dst_smaller", func(t *testing.T) {
		smallDst := make([]float64, 3)
		ConvolveValid(smallDst, signal, kernel)
		assert.InDeltaSlice(t, expected[:3], smallDst, refTolerance64, "ConvolveValid with smaller dst")
	})
}
