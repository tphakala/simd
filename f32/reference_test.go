package f32

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

func dotProductRef(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func sumRef(a []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i]
	}
	return sum
}

func minRef(a []float32) float32 {
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

func maxRef(a []float32) float32 {
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

func addRef(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func subRef(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func mulRef(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func divRef(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

func scaleRef(dst, a []float32, s float32) {
	for i := range dst {
		dst[i] = a[i] * s
	}
}

func addScalarRef(dst, a []float32, s float32) {
	for i := range dst {
		dst[i] = a[i] + s
	}
}

func absRef(dst, a []float32) {
	for i := range dst {
		dst[i] = float32(math.Abs(float64(a[i])))
	}
}

func negRef(dst, a []float32) {
	for i := range dst {
		dst[i] = -a[i]
	}
}

func fmaRef(dst, a, b, c []float32) {
	for i := range dst {
		dst[i] = a[i]*b[i] + c[i]
	}
}

func clampRef(dst, a []float32, minVal, maxVal float32) {
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

// =============================================================================
// Test helpers
// =============================================================================

func refAlmostEqual32(a, b float32) bool {
	if math.IsNaN(float64(a)) && math.IsNaN(float64(b)) {
		return true
	}
	if math.IsInf(float64(a), 0) && math.IsInf(float64(b), 0) {
		return math.Signbit(float64(a)) == math.Signbit(float64(b))
	}
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	// Use relative tolerance for larger values (float32 has ~7 digits precision)
	absA := a
	if absA < 0 {
		absA = -absA
	}
	absB := b
	if absB < 0 {
		absB = -absB
	}
	maxAbs := absA
	if absB > maxAbs {
		maxAbs = absB
	}
	if maxAbs > 1 {
		// Relative tolerance: allow 1e-5 relative error
		return diff <= maxAbs*1e-5
	}
	// Absolute tolerance for small values
	return diff <= 1e-5
}

func refSlicesEqual32(got, want []float32) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if !refAlmostEqual32(got[i], want[i]) {
			return false
		}
	}
	return true
}

// makeTestData creates test vectors with predictable values
func makeTestData32(n int) (a, b, c []float32) {
	a = make([]float32, n)
	b = make([]float32, n)
	c = make([]float32, n)
	for i := range n {
		a[i] = float32(i + 1)
		b[i] = float32(n - i)
		c[i] = 0.5
	}
	return
}

// makeMixedSigns creates vectors with negative and positive values
func makeMixedSigns32(n int) (a, b []float32) {
	a = make([]float32, n)
	b = make([]float32, n)
	half := n / 2
	for i := range n {
		if i < half {
			a[i] = -float32(half - i)
			b[i] = float32(half - i)
		} else {
			a[i] = float32(i - half + 1)
			b[i] = -float32(i - half + 1)
		}
	}
	return
}

// Tolerance for float32 comparisons
const refTolerance32 = 1e-5
const refEpsilon32 = 1e-6 // relative tolerance for accumulated values

// =============================================================================
// Reference validation tests
// =============================================================================

// TestDotProduct_Ref validates SIMD DotProduct against pure Go reference
func TestDotProduct_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 256, 1000}

	for _, n := range sizes {
		a, b, _ := makeTestData32(n)

		got := DotProduct(a, b)
		want := dotProductRef(a, b)

		// Use relative tolerance for accumulated values (float32 has ~7 decimal digits precision)
		assert.InEpsilon(t, float64(want), float64(got), refEpsilon32, "DotProduct n=%d", n)
	}

	t.Run("mixed_signs", func(t *testing.T) {
		a, b := makeMixedSigns32(10)
		got := DotProduct(a, b)
		want := dotProductRef(a, b)
		assert.InEpsilon(t, float64(want), float64(got), refEpsilon32, "DotProduct mixed signs")
	})
}

// TestSum_Ref validates SIMD Sum against pure Go reference
func TestSum_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 256, 1000}

	for _, n := range sizes {
		a, _, _ := makeTestData32(n)

		got := Sum(a)
		want := sumRef(a)

		// Use relative tolerance for accumulated values
		assert.InEpsilon(t, float64(want), float64(got), refEpsilon32, "Sum n=%d", n)
	}
}

// TestMinMax_Ref validates SIMD Min/Max against pure Go reference
func TestMinMax_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 256}

	for _, n := range sizes {
		a, _, _ := makeTestData32(n)

		gotMin := Min(a)
		wantMin := minRef(a)
		assert.InDelta(t, float64(wantMin), float64(gotMin), refTolerance32, "Min n=%d", n)

		gotMax := Max(a)
		wantMax := maxRef(a)
		assert.InDelta(t, float64(wantMax), float64(gotMax), refTolerance32, "Max n=%d", n)
	}
}

// TestAdd_Ref validates SIMD Add against pure Go reference
func TestAdd_Ref(t *testing.T) {
	sizes := []int{1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33}

	for _, n := range sizes {
		a, b, _ := makeTestData32(n)
		got := make([]float32, n)
		want := make([]float32, n)

		Add(got, a, b)
		addRef(want, a, b)

		require.Len(t, got, len(want), "Add n=%d length mismatch", n)
		assertFloat32SlicesEqual(t, want, got, "Add n=%d", n)
	}
}

// TestSub_Ref validates SIMD Sub against pure Go reference
func TestSub_Ref(t *testing.T) {
	sizes := []int{1, 3, 4, 5, 7, 8, 9, 15, 16, 17}

	for _, n := range sizes {
		a, b, _ := makeTestData32(n)
		got := make([]float32, n)
		want := make([]float32, n)

		Sub(got, a, b)
		subRef(want, a, b)

		assertFloat32SlicesEqual(t, want, got, "Sub n=%d", n)
	}
}

// TestMul_Ref validates SIMD Mul against pure Go reference
func TestMul_Ref(t *testing.T) {
	sizes := []int{1, 3, 4, 5, 7, 8, 9, 15, 16, 17}

	for _, n := range sizes {
		a, b, _ := makeTestData32(n)
		got := make([]float32, n)
		want := make([]float32, n)

		Mul(got, a, b)
		mulRef(want, a, b)

		assertFloat32SlicesEqual(t, want, got, "Mul n=%d", n)
	}
}

// TestDiv_Ref validates SIMD Div against pure Go reference
func TestDiv_Ref(t *testing.T) {
	sizes := []int{1, 3, 4, 5, 7, 8, 9, 15, 16, 17}

	for _, n := range sizes {
		a, b, _ := makeTestData32(n)
		got := make([]float32, n)
		want := make([]float32, n)

		Div(got, a, b)
		divRef(want, a, b)

		assertFloat32SlicesEqual(t, want, got, "Div n=%d", n)
	}
}

// TestScale_Ref validates SIMD Scale against pure Go reference
func TestScale_Ref(t *testing.T) {
	sizes := []int{1, 3, 4, 5, 7, 8, 9, 15, 16, 17}
	scalar := float32(2.5)

	for _, n := range sizes {
		a, _, _ := makeTestData32(n)
		got := make([]float32, n)
		want := make([]float32, n)

		Scale(got, a, scalar)
		scaleRef(want, a, scalar)

		assertFloat32SlicesEqual(t, want, got, "Scale n=%d", n)
	}
}

// TestAddScalar_Ref validates SIMD AddScalar against pure Go reference
func TestAddScalar_Ref(t *testing.T) {
	sizes := []int{1, 3, 4, 5, 7, 8, 9, 15, 16, 17}
	scalar := float32(10.5)

	for _, n := range sizes {
		a, _, _ := makeTestData32(n)
		got := make([]float32, n)
		want := make([]float32, n)

		AddScalar(got, a, scalar)
		addScalarRef(want, a, scalar)

		assertFloat32SlicesEqual(t, want, got, "AddScalar n=%d", n)
	}
}

// TestAbs_Ref validates SIMD Abs against pure Go reference
func TestAbs_Ref(t *testing.T) {
	sizes := []int{1, 3, 4, 5, 7, 8, 9, 15, 16, 17}

	for _, n := range sizes {
		a, _ := makeMixedSigns32(n)
		got := make([]float32, n)
		want := make([]float32, n)

		Abs(got, a)
		absRef(want, a)

		assertFloat32SlicesEqual(t, want, got, "Abs n=%d", n)
	}
}

// TestNeg_Ref validates SIMD Neg against pure Go reference
func TestNeg_Ref(t *testing.T) {
	sizes := []int{1, 3, 4, 5, 7, 8, 9, 15, 16, 17}

	for _, n := range sizes {
		a, _, _ := makeTestData32(n)
		got := make([]float32, n)
		want := make([]float32, n)

		Neg(got, a)
		negRef(want, a)

		assertFloat32SlicesEqual(t, want, got, "Neg n=%d", n)
	}
}

// TestFMA_Ref validates SIMD FMA against pure Go reference
func TestFMA_Ref(t *testing.T) {
	sizes := []int{1, 3, 4, 5, 7, 8, 9, 15, 16, 17}

	for _, n := range sizes {
		a, b, c := makeTestData32(n)
		got := make([]float32, n)
		want := make([]float32, n)

		FMA(got, a, b, c)
		fmaRef(want, a, b, c)

		assertFloat32SlicesEqual(t, want, got, "FMA n=%d", n)
	}
}

// TestClamp_Ref validates SIMD Clamp against pure Go reference
func TestClamp_Ref(t *testing.T) {
	sizes := []int{1, 3, 4, 5, 7, 8, 9, 15, 16, 17}
	minVal := float32(3.0)
	maxVal := float32(10.0)

	for _, n := range sizes {
		a, _, _ := makeTestData32(n)
		got := make([]float32, n)
		want := make([]float32, n)

		Clamp(got, a, minVal, maxVal)
		clampRef(want, a, minVal, maxVal)

		assertFloat32SlicesEqual(t, want, got, "Clamp n=%d", n)
	}
}

// assertFloat32SlicesEqual is a helper for comparing float32 slices with tolerance
func assertFloat32SlicesEqual(t *testing.T, expected, actual []float32, msgAndArgs ...any) {
	t.Helper()
	require.Len(t, actual, len(expected), msgAndArgs...)
	for i := range expected {
		assert.InDelta(t, float64(expected[i]), float64(actual[i]), refTolerance32, "index %d: %v", i, msgAndArgs)
	}
}

// =============================================================================
// Edge case tests for 100% coverage
// =============================================================================

// TestEdgeCases_Empty tests all functions with empty slices
func TestEdgeCases_Empty(t *testing.T) {
	empty := []float32{}
	dst := make([]float32, 10)

	t.Run("DotProduct_empty", func(t *testing.T) {
		got := DotProduct(empty, empty)
		assert.InDelta(t, float32(0), got, 1e-10, "DotProduct empty should return 0")
	})

	t.Run("Sum_empty", func(t *testing.T) {
		got := Sum(empty)
		assert.InDelta(t, float32(0), got, 1e-10, "Sum empty should return 0")
	})

	t.Run("Min_empty", func(t *testing.T) {
		got := Min(empty)
		assert.True(t, math.IsInf(float64(got), 1), "Min empty should return +Inf")
	})

	t.Run("Max_empty", func(t *testing.T) {
		got := Max(empty)
		assert.True(t, math.IsInf(float64(got), -1), "Max empty should return -Inf")
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

	t.Run("DotProductBatch_empty_results", func(t *testing.T) {
		assert.NotPanics(t, func() {
			DotProductBatch([]float32{}, [][]float32{{1, 2}}, []float32{1, 2})
		})
	})

	t.Run("DotProductBatch_empty_rows", func(t *testing.T) {
		assert.NotPanics(t, func() {
			DotProductBatch([]float32{1}, [][]float32{}, []float32{1, 2})
		})
	})

	t.Run("DotProductBatch_empty_vec", func(t *testing.T) {
		assert.NotPanics(t, func() {
			DotProductBatch([]float32{1}, [][]float32{{1, 2}}, []float32{})
		})
	})

	t.Run("ConvolveValid_empty_kernel", func(t *testing.T) {
		assert.NotPanics(t, func() {
			ConvolveValid(dst, []float32{1, 2, 3}, []float32{})
		})
	})

	t.Run("ConvolveValid_signal_smaller_than_kernel", func(t *testing.T) {
		assert.NotPanics(t, func() {
			ConvolveValid(dst, []float32{1}, []float32{1, 2, 3})
		})
	})

	t.Run("ConvolveValid_empty_dst", func(t *testing.T) {
		assert.NotPanics(t, func() {
			ConvolveValid([]float32{}, []float32{1, 2, 3, 4, 5}, []float32{1, 2})
		})
	})
}

// TestEdgeCases_MismatchedLengths tests functions with different length inputs
func TestEdgeCases_MismatchedLengths(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5}
	b := []float32{1, 2, 3}
	c := []float32{1, 2}

	t.Run("DotProduct_mismatched", func(t *testing.T) {
		got := DotProduct(a, b)
		want := dotProductRef(a[:3], b)
		assert.InDelta(t, float64(want), float64(got), refTolerance32, "DotProduct mismatched lengths")
	})

	t.Run("Add_dst_smaller", func(t *testing.T) {
		dst := make([]float32, 2)
		Add(dst, a, b)
		want := []float32{a[0] + b[0], a[1] + b[1]}
		assertFloat32SlicesEqual(t, want, dst, "Add with smaller dst")
	})

	t.Run("minLen_coverage", func(t *testing.T) {
		// Test all branches of minLen
		assert.Equal(t, 3, minLen(5, 3, 10), "minLen(5,3,10) b < a")
		assert.Equal(t, 2, minLen(5, 6, 2), "minLen(5,6,2) c < a")
		assert.Equal(t, 1, minLen(1, 5, 10), "minLen(1,5,10) a is smallest")
	})

	t.Run("FMA_mismatched", func(t *testing.T) {
		dst := make([]float32, 10)
		FMA(dst, a, b, c)
		// Should use min length of all inputs
		want := make([]float32, 2)
		for i := range 2 {
			want[i] = a[i]*b[i] + c[i]
		}
		assertFloat32SlicesEqual(t, want, dst[:2], refTolerance32, "FMA mismatched lengths")
	})
}

// TestDotProductBatch_Ref validates DotProductBatch against reference
func TestDotProductBatch_Ref(t *testing.T) {
	vec := []float32{1, 2, 3, 4}
	rows := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{1, 1, 1, 1},
		{2, 2, 2, 2},
	}
	results := make([]float32, len(rows))

	DotProductBatch(results, rows, vec)

	expected := []float32{1, 2, 10, 20}
	assertFloat32SlicesEqual(t, expected, results, "DotProductBatch basic")

	t.Run("with_empty_row", func(t *testing.T) {
		rowsWithEmpty := [][]float32{
			{1, 2, 3, 4},
			{}, // empty row - should return 0
			{1, 1, 1, 1},
		}
		res := make([]float32, 3)
		DotProductBatch(res, rowsWithEmpty, vec)
		want := []float32{30, 0, 10}
		assertFloat32SlicesEqual(t, want, res, "DotProductBatch with empty row")
	})

	t.Run("results_smaller", func(t *testing.T) {
		smallResults := make([]float32, 2)
		DotProductBatch(smallResults, rows, vec)
		assertFloat32SlicesEqual(t, expected[:2], smallResults, "DotProductBatch results smaller than rows")
	})

	t.Run("varying_row_lengths", func(t *testing.T) {
		mixedRows := [][]float32{
			{1, 2, 3, 4, 5}, // longer than vec
			{1, 2},          // shorter than vec
			{1, 2, 3, 4},    // same as vec
		}
		res := make([]float32, 3)
		DotProductBatch(res, mixedRows, vec)
		want := []float32{30, 5, 30} // 1*1+2*2+3*3+4*4=30, 1*1+2*2=5, 1*1+2*2+3*3+4*4=30
		assertFloat32SlicesEqual(t, want, res, "DotProductBatch varying row lengths")
	})
}

// TestConvolveValid_Ref validates ConvolveValid against reference
func TestConvolveValid_Ref(t *testing.T) {
	signal := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	kernel := []float32{1, 2, 1}

	// Expected output length: len(signal) - len(kernel) + 1 = 8 - 3 + 1 = 6
	dst := make([]float32, 6)
	ConvolveValid(dst, signal, kernel)

	// Compute expected using pure Go
	expected := make([]float32, 6)
	for i := range 6 {
		for j := range kernel {
			expected[i] += signal[i+j] * kernel[j]
		}
	}

	assertFloat32SlicesEqual(t, expected, dst, "ConvolveValid basic")

	t.Run("dst_smaller", func(t *testing.T) {
		smallDst := make([]float32, 3)
		ConvolveValid(smallDst, signal, kernel)
		assertFloat32SlicesEqual(t, expected[:3], smallDst, "ConvolveValid with smaller dst")
	})
}
