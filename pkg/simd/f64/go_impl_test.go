package f64

import (
	"math"
	"testing"
)

// Tests for Go fallback implementations to ensure they work correctly
// even on platforms where assembly is used.

func TestDotProductGo(t *testing.T) {
	tests := []struct {
		name string
		a, b []float64
		want float64
	}{
		{"size_1", []float64{2}, []float64{3}, 6},
		{"size_3", []float64{1, 2, 3}, []float64{4, 5, 6}, 32},
		{"size_4_exact", []float64{1, 2, 3, 4}, []float64{1, 1, 1, 1}, 10},
		{"size_5_remainder", []float64{1, 2, 3, 4, 5}, []float64{1, 1, 1, 1, 1}, 15},
		{"size_8_exact", make8Seq(), make8Ones(), 36},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := dotProductGo(tt.a, tt.b)
			if !refAlmostEqual64(got, tt.want) {
				t.Errorf("dotProductGo() = %v, want %v", got, tt.want)
			}
		})
	}
}

func make8Seq() []float64 {
	a := make([]float64, 8)
	for i := range a {
		a[i] = float64(i + 1)
	}
	return a
}

func make8Ones() []float64 {
	a := make([]float64, 8)
	for i := range a {
		a[i] = 1
	}
	return a
}

func TestMinGo(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want float64
	}{
		{"single", []float64{5}, 5},
		{"min_first", []float64{1, 2, 3}, 1},
		{"min_middle", []float64{3, 1, 2}, 1},
		{"min_last", []float64{3, 2, 1}, 1},
		{"all_same", []float64{5, 5, 5}, 5},
		{"negative", []float64{-1, -5, -2}, -5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := minGo(tt.a)
			if got != tt.want {
				t.Errorf("minGo() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMaxGo(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want float64
	}{
		{"single", []float64{5}, 5},
		{"max_first", []float64{3, 2, 1}, 3},
		{"max_middle", []float64{1, 3, 2}, 3},
		{"max_last", []float64{1, 2, 3}, 3},
		{"all_same", []float64{5, 5, 5}, 5},
		{"negative", []float64{-1, -5, -2}, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := maxGo(tt.a)
			if got != tt.want {
				t.Errorf("maxGo() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestClampGo(t *testing.T) {
	tests := []struct {
		name       string
		a          []float64
		minV, maxV float64
		want       []float64
	}{
		{"below_min", []float64{1, 2, 3}, 5, 10, []float64{5, 5, 5}},
		{"above_max", []float64{11, 12, 13}, 5, 10, []float64{10, 10, 10}},
		{"in_range", []float64{6, 7, 8}, 5, 10, []float64{6, 7, 8}},
		{"mixed", []float64{1, 7, 15}, 5, 10, []float64{5, 7, 10}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			clampGo(dst, tt.a, tt.minV, tt.maxV)
			if !refSlicesEqual64(dst, tt.want) {
				t.Errorf("clampGo() = %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestDotProductBatch64Go(t *testing.T) {
	vec := []float64{1, 2, 3, 4}
	rows := [][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{1, 1, 1, 1},
		{},           // empty row
		{1, 2},       // shorter than vec
	}
	results := make([]float64, len(rows))

	dotProductBatch64Go(results, rows, vec)

	expected := []float64{1, 2, 10, 0, 5}
	if !refSlicesEqual64(results, expected) {
		t.Errorf("dotProductBatch64Go() = %v, want %v", results, expected)
	}
}

func TestConvolveValid64Go(t *testing.T) {
	signal := []float64{1, 2, 3, 4, 5}
	kernel := []float64{1, 2}

	// Valid output length: 5 - 2 + 1 = 4
	dst := make([]float64, 4)
	convolveValid64Go(dst, signal, kernel)

	// dst[i] = signal[i]*kernel[0] + signal[i+1]*kernel[1]
	expected := []float64{
		1*1 + 2*2, // 5
		2*1 + 3*2, // 8
		3*1 + 4*2, // 11
		4*1 + 5*2, // 14
	}

	if !refSlicesEqual64(dst, expected) {
		t.Errorf("convolveValid64Go() = %v, want %v", dst, expected)
	}
}

func TestCumulativeSum64Go(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want []float64
	}{
		{"basic", []float64{1, 2, 3, 4}, []float64{1, 3, 6, 10}},
		{"single", []float64{5}, []float64{5}},
		{"empty", []float64{}, []float64{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			cumulativeSum64Go(dst, tt.a)
			if !refSlicesEqual64(dst, tt.want) {
				t.Errorf("cumulativeSum64Go() = %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestAddGo(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	dst := make([]float64, 3)
	addGo(dst, a, b)
	want := []float64{5, 7, 9}
	if !refSlicesEqual64(dst, want) {
		t.Errorf("addGo() = %v, want %v", dst, want)
	}
}

func TestSubGo(t *testing.T) {
	a := []float64{4, 5, 6}
	b := []float64{1, 2, 3}
	dst := make([]float64, 3)
	subGo(dst, a, b)
	want := []float64{3, 3, 3}
	if !refSlicesEqual64(dst, want) {
		t.Errorf("subGo() = %v, want %v", dst, want)
	}
}

func TestMulGo(t *testing.T) {
	a := []float64{2, 3, 4}
	b := []float64{5, 6, 7}
	dst := make([]float64, 3)
	mulGo(dst, a, b)
	want := []float64{10, 18, 28}
	if !refSlicesEqual64(dst, want) {
		t.Errorf("mulGo() = %v, want %v", dst, want)
	}
}

func TestDivGo(t *testing.T) {
	a := []float64{10, 18, 28}
	b := []float64{5, 6, 7}
	dst := make([]float64, 3)
	divGo(dst, a, b)
	want := []float64{2, 3, 4}
	if !refSlicesEqual64(dst, want) {
		t.Errorf("divGo() = %v, want %v", dst, want)
	}
}

func TestScaleGo(t *testing.T) {
	a := []float64{1, 2, 3}
	dst := make([]float64, 3)
	scaleGo(dst, a, 2)
	want := []float64{2, 4, 6}
	if !refSlicesEqual64(dst, want) {
		t.Errorf("scaleGo() = %v, want %v", dst, want)
	}
}

func TestAddScalarGo(t *testing.T) {
	a := []float64{1, 2, 3}
	dst := make([]float64, 3)
	addScalarGo(dst, a, 10)
	want := []float64{11, 12, 13}
	if !refSlicesEqual64(dst, want) {
		t.Errorf("addScalarGo() = %v, want %v", dst, want)
	}
}

func TestSumGo(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	got := sumGo(a)
	want := float64(15)
	if got != want {
		t.Errorf("sumGo() = %v, want %v", got, want)
	}
}

func TestAbsGo(t *testing.T) {
	a := []float64{-1, 2, -3}
	dst := make([]float64, 3)
	absGo(dst, a)
	want := []float64{1, 2, 3}
	if !refSlicesEqual64(dst, want) {
		t.Errorf("absGo() = %v, want %v", dst, want)
	}
}

func TestNegGo(t *testing.T) {
	a := []float64{1, -2, 3}
	dst := make([]float64, 3)
	negGo(dst, a)
	want := []float64{-1, 2, -3}
	if !refSlicesEqual64(dst, want) {
		t.Errorf("negGo() = %v, want %v", dst, want)
	}
}

func TestFmaGo(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	c := []float64{7, 8, 9}
	dst := make([]float64, 3)
	fmaGo(dst, a, b, c)
	// dst[i] = a[i]*b[i] + c[i]
	want := []float64{1*4 + 7, 2*5 + 8, 3*6 + 9}
	if !refSlicesEqual64(dst, want) {
		t.Errorf("fmaGo() = %v, want %v", dst, want)
	}
}

func TestSqrt64Go(t *testing.T) {
	a := []float64{1, 4, 9, 16}
	dst := make([]float64, 4)
	sqrt64Go(dst, a)
	want := []float64{1, 2, 3, 4}
	if !refSlicesEqual64(dst, want) {
		t.Errorf("sqrt64Go() = %v, want %v", dst, want)
	}
}

func TestReciprocal64Go(t *testing.T) {
	a := []float64{1, 2, 4, 5}
	dst := make([]float64, 4)
	reciprocal64Go(dst, a)
	want := []float64{1, 0.5, 0.25, 0.2}
	if !refSlicesEqual64(dst, want) {
		t.Errorf("reciprocal64Go() = %v, want %v", dst, want)
	}
}

func TestVariance64Go(t *testing.T) {
	// Variance of [1, 2, 3, 4, 5] with mean 3:
	// ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5
	// = (4 + 1 + 0 + 1 + 4) / 5 = 2
	a := []float64{1, 2, 3, 4, 5}
	got := variance64Go(a, 3)
	want := 2.0
	if !refAlmostEqual64(got, want) {
		t.Errorf("variance64Go() = %v, want %v", got, want)
	}
}

func TestEuclideanDistance64Go(t *testing.T) {
	tests := []struct {
		name string
		a, b []float64
		want float64
	}{
		{"same", []float64{1, 2, 3}, []float64{1, 2, 3}, 0},
		{"3d", []float64{0, 0, 0}, []float64{1, 0, 0}, 1},
		{"3d_diagonal", []float64{0, 0, 0}, []float64{3, 4, 0}, 5},
		// Test remainder (not multiple of 4)
		{"size_5", []float64{0, 0, 0, 0, 0}, []float64{3, 4, 0, 0, 0}, 5},
		// Test exact multiple of 4
		{"size_4", []float64{0, 0, 0, 0}, []float64{3, 4, 0, 0}, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := euclideanDistance64Go(tt.a, tt.b)
			if !refAlmostEqual64(got, tt.want) {
				t.Errorf("euclideanDistance64Go() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Test that posInf and negInf are correctly defined
func TestInfConstants(t *testing.T) {
	if !math.IsInf(posInf, 1) {
		t.Errorf("posInf should be +Inf")
	}
	if !math.IsInf(negInf, -1) {
		t.Errorf("negInf should be -Inf")
	}
}
