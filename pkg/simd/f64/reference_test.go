package f64

// Pure Go reference implementations for validating SIMD operations.
// These are intentionally simple and obviously correct.

import (
	"math"
	"testing"
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
	min := a[0]
	for i := 1; i < len(a); i++ {
		if a[i] < min {
			min = a[i]
		}
	}
	return min
}

func maxRef(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	max := a[0]
	for i := 1; i < len(a); i++ {
		if a[i] > max {
			max = a[i]
		}
	}
	return max
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
	for i := 0; i < n; i++ {
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
	for i := 0; i < n; i++ {
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

		if !refAlmostEqual64(got, want) {
			t.Errorf("DotProduct n=%d: got %v, want %v", n, got, want)
		}
	}

	t.Run("mixed_signs", func(t *testing.T) {
		a, b := makeMixedSigns64(10)
		got := DotProduct(a, b)
		want := dotProductRef(a, b)
		if !refAlmostEqual64(got, want) {
			t.Errorf("DotProduct mixed: got %v, want %v", got, want)
		}
	})
}

// TestSum_Ref validates SIMD Sum against pure Go reference
func TestSum_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 256, 1000}

	for _, n := range sizes {
		a, _, _ := makeTestData64(n)

		got := Sum(a)
		want := sumRef(a)

		if !refAlmostEqual64(got, want) {
			t.Errorf("Sum n=%d: got %v, want %v", n, got, want)
		}
	}
}

// TestMinMax_Ref validates SIMD Min/Max against pure Go reference
func TestMinMax_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 256}

	for _, n := range sizes {
		a, _, _ := makeTestData64(n)

		gotMin := Min(a)
		wantMin := minRef(a)
		if !refAlmostEqual64(gotMin, wantMin) {
			t.Errorf("Min n=%d: got %v, want %v", n, gotMin, wantMin)
		}

		gotMax := Max(a)
		wantMax := maxRef(a)
		if !refAlmostEqual64(gotMax, wantMax) {
			t.Errorf("Max n=%d: got %v, want %v", n, gotMax, wantMax)
		}
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

		if !refSlicesEqual64(got, want) {
			t.Errorf("Add n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual64(got, want) {
			t.Errorf("Sub n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual64(got, want) {
			t.Errorf("Mul n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual64(got, want) {
			t.Errorf("Div n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual64(got, want) {
			t.Errorf("Scale n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual64(got, want) {
			t.Errorf("AddScalar n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual64(got, want) {
			t.Errorf("Abs n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual64(got, want) {
			t.Errorf("Neg n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual64(got, want) {
			t.Errorf("FMA n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual64(got, want) {
			t.Errorf("Clamp n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual64(got, want) {
			t.Errorf("Sqrt n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual64(got, want) {
			t.Errorf("Reciprocal n=%d: got %v, want %v", n, got, want)
		}
	}
}

// TestMean_Ref validates SIMD Mean against pure Go reference
func TestMean_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100, 256}

	for _, n := range sizes {
		a, _, _ := makeTestData64(n)

		got := Mean(a)
		want := meanRef(a)

		if !refAlmostEqual64(got, want) {
			t.Errorf("Mean n=%d: got %v, want %v", n, got, want)
		}
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
		tolerance := 1e-6
		diff := got - want
		if diff < 0 {
			diff = -diff
		}
		if diff > tolerance {
			t.Errorf("Variance n=%d: got %v, want %v", n, got, want)
		}
	}
}

// TestEuclideanDistance_Ref validates SIMD EuclideanDistance against pure Go reference
func TestEuclideanDistance_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100}

	for _, n := range sizes {
		a, b, _ := makeTestData64(n)

		got := EuclideanDistance(a, b)
		want := euclideanDistanceRef(a, b)

		if !refAlmostEqual64(got, want) {
			t.Errorf("EuclideanDistance n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual64(got, want) {
			t.Errorf("Normalize n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual64(got, want) {
			t.Errorf("CumulativeSum n=%d: got %v, want %v", n, got, want)
		}
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
		if got != 0 {
			t.Errorf("DotProduct empty: got %v, want 0", got)
		}
	})

	t.Run("Sum_empty", func(t *testing.T) {
		got := Sum(empty)
		if got != 0 {
			t.Errorf("Sum empty: got %v, want 0", got)
		}
	})

	t.Run("Min_empty", func(t *testing.T) {
		got := Min(empty)
		if got != posInf {
			t.Errorf("Min empty: got %v, want +Inf", got)
		}
	})

	t.Run("Max_empty", func(t *testing.T) {
		got := Max(empty)
		if got != negInf {
			t.Errorf("Max empty: got %v, want -Inf", got)
		}
	})

	t.Run("Add_empty", func(t *testing.T) {
		Add(dst, empty, empty)
		// Should not panic, just return
	})

	t.Run("Sub_empty", func(t *testing.T) {
		Sub(dst, empty, empty)
	})

	t.Run("Mul_empty", func(t *testing.T) {
		Mul(dst, empty, empty)
	})

	t.Run("Div_empty", func(t *testing.T) {
		Div(dst, empty, empty)
	})

	t.Run("Scale_empty", func(t *testing.T) {
		Scale(dst, empty, 2.0)
	})

	t.Run("AddScalar_empty", func(t *testing.T) {
		AddScalar(dst, empty, 2.0)
	})

	t.Run("Abs_empty", func(t *testing.T) {
		Abs(dst, empty)
	})

	t.Run("Neg_empty", func(t *testing.T) {
		Neg(dst, empty)
	})

	t.Run("FMA_empty", func(t *testing.T) {
		FMA(dst, empty, empty, empty)
	})

	t.Run("Clamp_empty", func(t *testing.T) {
		Clamp(dst, empty, 0, 1)
	})

	t.Run("Sqrt_empty", func(t *testing.T) {
		Sqrt(dst, empty)
	})

	t.Run("Reciprocal_empty", func(t *testing.T) {
		Reciprocal(dst, empty)
	})

	t.Run("Mean_empty", func(t *testing.T) {
		got := Mean(empty)
		if got != 0 {
			t.Errorf("Mean empty: got %v, want 0", got)
		}
	})

	t.Run("Variance_empty", func(t *testing.T) {
		got := Variance(empty)
		if got != 0 {
			t.Errorf("Variance empty: got %v, want 0", got)
		}
	})

	t.Run("StdDev_empty", func(t *testing.T) {
		got := StdDev(empty)
		if got != 0 {
			t.Errorf("StdDev empty: got %v, want 0", got)
		}
	})

	t.Run("EuclideanDistance_empty", func(t *testing.T) {
		got := EuclideanDistance(empty, empty)
		if got != 0 {
			t.Errorf("EuclideanDistance empty: got %v, want 0", got)
		}
	})

	t.Run("Normalize_empty", func(t *testing.T) {
		Normalize(dst, empty)
	})

	t.Run("CumulativeSum_empty", func(t *testing.T) {
		CumulativeSum(dst, empty)
	})

	t.Run("DotProductBatch_empty_results", func(t *testing.T) {
		DotProductBatch([]float64{}, [][]float64{{1, 2}}, []float64{1, 2})
	})

	t.Run("DotProductBatch_empty_rows", func(t *testing.T) {
		DotProductBatch([]float64{1}, [][]float64{}, []float64{1, 2})
	})

	t.Run("DotProductBatch_empty_vec", func(t *testing.T) {
		DotProductBatch([]float64{1}, [][]float64{{1, 2}}, []float64{})
	})

	t.Run("ConvolveValid_empty_kernel", func(t *testing.T) {
		ConvolveValid(dst, []float64{1, 2, 3}, []float64{})
	})

	t.Run("ConvolveValid_signal_smaller_than_kernel", func(t *testing.T) {
		ConvolveValid(dst, []float64{1}, []float64{1, 2, 3})
	})

	t.Run("ConvolveValid_empty_dst", func(t *testing.T) {
		ConvolveValid([]float64{}, []float64{1, 2, 3, 4, 5}, []float64{1, 2})
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
		if !refAlmostEqual64(got, want) {
			t.Errorf("DotProduct mismatched: got %v, want %v", got, want)
		}
	})

	t.Run("Add_dst_smaller", func(t *testing.T) {
		dst := make([]float64, 2)
		Add(dst, a, b)
		want := []float64{a[0] + b[0], a[1] + b[1]}
		if !refSlicesEqual64(dst, want) {
			t.Errorf("Add dst_smaller: got %v, want %v", dst, want)
		}
	})

	t.Run("minLen_coverage", func(t *testing.T) {
		// Test all branches of minLen
		// b < a
		n := minLen(5, 3, 10)
		if n != 3 {
			t.Errorf("minLen(5,3,10): got %d, want 3", n)
		}
		// c < a (after b >= a)
		n = minLen(5, 6, 2)
		if n != 2 {
			t.Errorf("minLen(5,6,2): got %d, want 2", n)
		}
		// a is smallest
		n = minLen(1, 5, 10)
		if n != 1 {
			t.Errorf("minLen(1,5,10): got %d, want 1", n)
		}
	})

	t.Run("FMA_mismatched", func(t *testing.T) {
		dst := make([]float64, 10)
		FMA(dst, a, b, c)
		// Should use min length of all inputs
		want := make([]float64, 10)
		for i := 0; i < 2; i++ {
			want[i] = a[i]*b[i] + c[i]
		}
		if !refSlicesEqual64(dst[:2], want[:2]) {
			t.Errorf("FMA mismatched: got %v, want %v", dst[:2], want[:2])
		}
	})

	t.Run("EuclideanDistance_mismatched", func(t *testing.T) {
		got := EuclideanDistance(a, b)
		want := euclideanDistanceRef(a[:3], b)
		if !refAlmostEqual64(got, want) {
			t.Errorf("EuclideanDistance mismatched: got %v, want %v", got, want)
		}
	})
}

// TestNormalize_ZeroMagnitude tests Normalize with near-zero vectors
func TestNormalize_ZeroMagnitude(t *testing.T) {
	// Vector with magnitude below threshold (1e-10)
	tiny := []float64{1e-12, 1e-12, 1e-12}
	dst := make([]float64, 3)

	Normalize(dst, tiny)

	// Should copy input unchanged
	if !refSlicesEqual64(dst, tiny) {
		t.Errorf("Normalize zero mag: got %v, want %v (unchanged)", dst, tiny)
	}
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
	if !refSlicesEqual64(results, expected) {
		t.Errorf("DotProductBatch: got %v, want %v", results, expected)
	}

	// Test with empty row (covers ARM64 branch)
	t.Run("with_empty_row", func(t *testing.T) {
		rowsWithEmpty := [][]float64{
			{1, 2, 3, 4},
			{},  // empty row - should return 0
			{1, 1, 1, 1},
		}
		res := make([]float64, 3)
		DotProductBatch(res, rowsWithEmpty, vec)
		want := []float64{30, 0, 10}
		if !refSlicesEqual64(res, want) {
			t.Errorf("DotProductBatch with empty: got %v, want %v", res, want)
		}
	})

	// Test with results smaller than rows
	t.Run("results_smaller", func(t *testing.T) {
		smallResults := make([]float64, 2)
		DotProductBatch(smallResults, rows, vec)
		if !refSlicesEqual64(smallResults, expected[:2]) {
			t.Errorf("DotProductBatch results_smaller: got %v, want %v", smallResults, expected[:2])
		}
	})

	// Test with varying row lengths
	t.Run("varying_row_lengths", func(t *testing.T) {
		mixedRows := [][]float64{
			{1, 2, 3, 4, 5}, // longer than vec
			{1, 2},         // shorter than vec
			{1, 2, 3, 4},   // same as vec
		}
		res := make([]float64, 3)
		DotProductBatch(res, mixedRows, vec)
		// Each row uses min(len(row), len(vec))
		want0 := float64(1*1 + 2*2 + 3*3 + 4*4)
		want1 := float64(1*1 + 2*2)
		want2 := float64(1*1 + 2*2 + 3*3 + 4*4)
		if !refAlmostEqual64(res[0], want0) || !refAlmostEqual64(res[1], want1) || !refAlmostEqual64(res[2], want2) {
			t.Errorf("DotProductBatch varying: got %v, want [%v, %v, %v]", res, want0, want1, want2)
		}
	})
}

// TestConvolveValid_Ref validates ConvolveValid against reference
func TestConvolveValid_Ref(t *testing.T) {
	signal := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	kernel := []float64{1, 2, 1}

	// Expected output length: len(signal) - len(kernel) + 1 = 8 - 3 + 1 = 6
	dst := make([]float64, 6)
	ConvolveValid(dst, signal, kernel)

	// Compute expected
	expected := make([]float64, 6)
	for i := 0; i < 6; i++ {
		for j := 0; j < len(kernel); j++ {
			expected[i] += signal[i+j] * kernel[j]
		}
	}

	if !refSlicesEqual64(dst, expected) {
		t.Errorf("ConvolveValid: got %v, want %v", dst, expected)
	}

	// Test with dst smaller than valid output length
	t.Run("dst_smaller", func(t *testing.T) {
		smallDst := make([]float64, 3)
		ConvolveValid(smallDst, signal, kernel)
		if !refSlicesEqual64(smallDst, expected[:3]) {
			t.Errorf("ConvolveValid dst_smaller: got %v, want %v", smallDst, expected[:3])
		}
	})
}
