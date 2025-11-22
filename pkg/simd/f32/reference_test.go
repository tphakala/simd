package f32

// Pure Go reference implementations for validating SIMD operations.
// These are intentionally simple and obviously correct.

import (
	"math"
	"testing"
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
	min := a[0]
	for i := 1; i < len(a); i++ {
		if a[i] < min {
			min = a[i]
		}
	}
	return min
}

func maxRef(a []float32) float32 {
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
	for i := 0; i < n; i++ {
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
	for i := 0; i < n; i++ {
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

// =============================================================================
// Reference validation tests
// =============================================================================

// TestDotProduct_Ref validates SIMD DotProduct against pure Go reference
func TestDotProduct_Ref(t *testing.T) {
	// Test various sizes including SIMD boundaries
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 256, 1000}

	for _, n := range sizes {
		a, b, _ := makeTestData32(n)

		got := DotProduct(a, b)
		want := dotProductRef(a, b)

		if !refAlmostEqual32(got, want) {
			t.Errorf("DotProduct n=%d: got %v, want %v", n, got, want)
		}
	}

	// Test mixed signs
	t.Run("mixed_signs", func(t *testing.T) {
		a, b := makeMixedSigns32(10)
		got := DotProduct(a, b)
		want := dotProductRef(a, b)
		if !refAlmostEqual32(got, want) {
			t.Errorf("DotProduct mixed: got %v, want %v", got, want)
		}
	})
}

// TestSum_Ref validates SIMD Sum against pure Go reference
func TestSum_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 256, 1000}

	for _, n := range sizes {
		a, _, _ := makeTestData32(n)

		got := Sum(a)
		want := sumRef(a)

		if !refAlmostEqual32(got, want) {
			t.Errorf("Sum n=%d: got %v, want %v", n, got, want)
		}
	}
}

// TestMinMax_Ref validates SIMD Min/Max against pure Go reference
func TestMinMax_Ref(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 256}

	for _, n := range sizes {
		a, _, _ := makeTestData32(n)

		gotMin := Min(a)
		wantMin := minRef(a)
		if !refAlmostEqual32(gotMin, wantMin) {
			t.Errorf("Min n=%d: got %v, want %v", n, gotMin, wantMin)
		}

		gotMax := Max(a)
		wantMax := maxRef(a)
		if !refAlmostEqual32(gotMax, wantMax) {
			t.Errorf("Max n=%d: got %v, want %v", n, gotMax, wantMax)
		}
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

		if !refSlicesEqual32(got, want) {
			t.Errorf("Add n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual32(got, want) {
			t.Errorf("Sub n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual32(got, want) {
			t.Errorf("Mul n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual32(got, want) {
			t.Errorf("Div n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual32(got, want) {
			t.Errorf("Scale n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual32(got, want) {
			t.Errorf("AddScalar n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual32(got, want) {
			t.Errorf("Abs n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual32(got, want) {
			t.Errorf("Neg n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual32(got, want) {
			t.Errorf("FMA n=%d: got %v, want %v", n, got, want)
		}
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

		if !refSlicesEqual32(got, want) {
			t.Errorf("Clamp n=%d: got %v, want %v", n, got, want)
		}
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

	t.Run("DotProductBatch_empty_results", func(t *testing.T) {
		DotProductBatch([]float32{}, [][]float32{{1, 2}}, []float32{1, 2})
	})

	t.Run("DotProductBatch_empty_rows", func(t *testing.T) {
		DotProductBatch([]float32{1}, [][]float32{}, []float32{1, 2})
	})

	t.Run("DotProductBatch_empty_vec", func(t *testing.T) {
		DotProductBatch([]float32{1}, [][]float32{{1, 2}}, []float32{})
	})

	t.Run("ConvolveValid_empty_kernel", func(t *testing.T) {
		ConvolveValid(dst, []float32{1, 2, 3}, []float32{})
	})

	t.Run("ConvolveValid_signal_smaller_than_kernel", func(t *testing.T) {
		ConvolveValid(dst, []float32{1}, []float32{1, 2, 3})
	})

	t.Run("ConvolveValid_empty_dst", func(t *testing.T) {
		ConvolveValid([]float32{}, []float32{1, 2, 3, 4, 5}, []float32{1, 2})
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
		if !refAlmostEqual32(got, want) {
			t.Errorf("DotProduct mismatched: got %v, want %v", got, want)
		}
	})

	t.Run("Add_dst_smaller", func(t *testing.T) {
		dst := make([]float32, 2)
		Add(dst, a, b)
		want := []float32{a[0] + b[0], a[1] + b[1]}
		if !refSlicesEqual32(dst, want) {
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
		dst := make([]float32, 10)
		FMA(dst, a, b, c)
		// Should use min length of all inputs
		want := make([]float32, 10)
		for i := 0; i < 2; i++ {
			want[i] = a[i]*b[i] + c[i]
		}
		if !refSlicesEqual32(dst[:2], want[:2]) {
			t.Errorf("FMA mismatched: got %v, want %v", dst[:2], want[:2])
		}
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
	if !refSlicesEqual32(results, expected) {
		t.Errorf("DotProductBatch: got %v, want %v", results, expected)
	}

	// Test with empty row (covers ARM64 branch)
	t.Run("with_empty_row", func(t *testing.T) {
		rowsWithEmpty := [][]float32{
			{1, 2, 3, 4},
			{},  // empty row - should return 0
			{1, 1, 1, 1},
		}
		res := make([]float32, 3)
		DotProductBatch(res, rowsWithEmpty, vec)
		want := []float32{30, 0, 10}
		if !refSlicesEqual32(res, want) {
			t.Errorf("DotProductBatch with empty: got %v, want %v", res, want)
		}
	})

	// Test with results smaller than rows
	t.Run("results_smaller", func(t *testing.T) {
		smallResults := make([]float32, 2)
		DotProductBatch(smallResults, rows, vec)
		if !refSlicesEqual32(smallResults, expected[:2]) {
			t.Errorf("DotProductBatch results_smaller: got %v, want %v", smallResults, expected[:2])
		}
	})

	// Test with varying row lengths
	t.Run("varying_row_lengths", func(t *testing.T) {
		mixedRows := [][]float32{
			{1, 2, 3, 4, 5}, // longer than vec
			{1, 2},         // shorter than vec
			{1, 2, 3, 4},   // same as vec
		}
		res := make([]float32, 3)
		DotProductBatch(res, mixedRows, vec)
		// Each row uses min(len(row), len(vec))
		want0 := float32(1*1 + 2*2 + 3*3 + 4*4)
		want1 := float32(1*1 + 2*2)
		want2 := float32(1*1 + 2*2 + 3*3 + 4*4)
		if !refAlmostEqual32(res[0], want0) || !refAlmostEqual32(res[1], want1) || !refAlmostEqual32(res[2], want2) {
			t.Errorf("DotProductBatch varying: got %v, want [%v, %v, %v]", res, want0, want1, want2)
		}
	})
}

// TestConvolveValid_Ref validates ConvolveValid against reference
func TestConvolveValid_Ref(t *testing.T) {
	signal := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	kernel := []float32{1, 2, 1}

	// Expected output length: len(signal) - len(kernel) + 1 = 8 - 3 + 1 = 6
	dst := make([]float32, 6)
	ConvolveValid(dst, signal, kernel)

	// Compute expected
	expected := make([]float32, 6)
	for i := 0; i < 6; i++ {
		for j := 0; j < len(kernel); j++ {
			expected[i] += signal[i+j] * kernel[j]
		}
	}

	if !refSlicesEqual32(dst, expected) {
		t.Errorf("ConvolveValid: got %v, want %v", dst, expected)
	}

	// Test with dst smaller than valid output length
	t.Run("dst_smaller", func(t *testing.T) {
		smallDst := make([]float32, 3)
		ConvolveValid(smallDst, signal, kernel)
		if !refSlicesEqual32(smallDst, expected[:3]) {
			t.Errorf("ConvolveValid dst_smaller: got %v, want %v", smallDst, expected[:3])
		}
	})
}
