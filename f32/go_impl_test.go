package f32

import (
	"math"
	"testing"
)

// Tests for Go fallback implementations to ensure they work correctly
// even on platforms where assembly is used.

func TestDotProductGo(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"size_1", []float32{2}, []float32{3}, 6},
		{"size_3", []float32{1, 2, 3}, []float32{4, 5, 6}, 32},
		{"size_8_exact", []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 1, 1, 1, 1, 1, 1, 1}, 36},
		{"size_9_remainder", []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, []float32{1, 1, 1, 1, 1, 1, 1, 1, 1}, 45},
		{"size_16_exact", make16Seq(), make16Ones(), 136},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := dotProductGo(tt.a, tt.b)
			if !refAlmostEqual32(got, tt.want) {
				t.Errorf("dotProductGo() = %v, want %v", got, tt.want)
			}
		})
	}
}

func make16Seq() []float32 {
	a := make([]float32, 16)
	for i := range a {
		a[i] = float32(i + 1)
	}
	return a
}

func make16Ones() []float32 {
	a := make([]float32, 16)
	for i := range a {
		a[i] = 1
	}
	return a
}

func TestMinGo(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		want float32
	}{
		{"single", []float32{5}, 5},
		{"min_first", []float32{1, 2, 3}, 1},
		{"min_middle", []float32{3, 1, 2}, 1},
		{"min_last", []float32{3, 2, 1}, 1},
		{"all_same", []float32{5, 5, 5}, 5},
		{"negative", []float32{-1, -5, -2}, -5},
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
		a    []float32
		want float32
	}{
		{"single", []float32{5}, 5},
		{"max_first", []float32{3, 2, 1}, 3},
		{"max_middle", []float32{1, 3, 2}, 3},
		{"max_last", []float32{1, 2, 3}, 3},
		{"all_same", []float32{5, 5, 5}, 5},
		{"negative", []float32{-1, -5, -2}, -1},
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
		a          []float32
		minV, maxV float32
		want       []float32
	}{
		{"below_min", []float32{1, 2, 3}, 5, 10, []float32{5, 5, 5}},
		{"above_max", []float32{11, 12, 13}, 5, 10, []float32{10, 10, 10}},
		{"in_range", []float32{6, 7, 8}, 5, 10, []float32{6, 7, 8}},
		{"mixed", []float32{1, 7, 15}, 5, 10, []float32{5, 7, 10}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			clampGo(dst, tt.a, tt.minV, tt.maxV)
			if !refSlicesEqual32(dst, tt.want) {
				t.Errorf("clampGo() = %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestDotProductBatch32Go(t *testing.T) {
	vec := []float32{1, 2, 3, 4}
	rows := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{1, 1, 1, 1},
		{},     // empty row
		{1, 2}, // shorter than vec
	}
	results := make([]float32, len(rows))

	dotProductBatch32Go(results, rows, vec)

	expected := []float32{1, 2, 10, 0, 5}
	if !refSlicesEqual32(results, expected) {
		t.Errorf("dotProductBatch32Go() = %v, want %v", results, expected)
	}
}

func TestConvolveValid32Go(t *testing.T) {
	signal := []float32{1, 2, 3, 4, 5}
	kernel := []float32{1, 2}

	// Valid output length: 5 - 2 + 1 = 4
	dst := make([]float32, 4)
	convolveValid32Go(dst, signal, kernel)

	// dst[i] = signal[i]*kernel[0] + signal[i+1]*kernel[1]
	expected := []float32{
		1*1 + 2*2, // 5
		2*1 + 3*2, // 8
		3*1 + 4*2, // 11
		4*1 + 5*2, // 14
	}

	if !refSlicesEqual32(dst, expected) {
		t.Errorf("convolveValid32Go() = %v, want %v", dst, expected)
	}
}

func TestAddGo(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	dst := make([]float32, 3)
	addGo(dst, a, b)
	want := []float32{5, 7, 9}
	if !refSlicesEqual32(dst, want) {
		t.Errorf("addGo() = %v, want %v", dst, want)
	}
}

func TestSubGo(t *testing.T) {
	a := []float32{4, 5, 6}
	b := []float32{1, 2, 3}
	dst := make([]float32, 3)
	subGo(dst, a, b)
	want := []float32{3, 3, 3}
	if !refSlicesEqual32(dst, want) {
		t.Errorf("subGo() = %v, want %v", dst, want)
	}
}

func TestMulGo(t *testing.T) {
	a := []float32{2, 3, 4}
	b := []float32{5, 6, 7}
	dst := make([]float32, 3)
	mulGo(dst, a, b)
	want := []float32{10, 18, 28}
	if !refSlicesEqual32(dst, want) {
		t.Errorf("mulGo() = %v, want %v", dst, want)
	}
}

func TestDivGo(t *testing.T) {
	a := []float32{10, 18, 28}
	b := []float32{5, 6, 7}
	dst := make([]float32, 3)
	divGo(dst, a, b)
	want := []float32{2, 3, 4}
	if !refSlicesEqual32(dst, want) {
		t.Errorf("divGo() = %v, want %v", dst, want)
	}
}

func TestScaleGo(t *testing.T) {
	a := []float32{1, 2, 3}
	dst := make([]float32, 3)
	scaleGo(dst, a, 2)
	want := []float32{2, 4, 6}
	if !refSlicesEqual32(dst, want) {
		t.Errorf("scaleGo() = %v, want %v", dst, want)
	}
}

func TestAddScalarGo(t *testing.T) {
	a := []float32{1, 2, 3}
	dst := make([]float32, 3)
	addScalarGo(dst, a, 10)
	want := []float32{11, 12, 13}
	if !refSlicesEqual32(dst, want) {
		t.Errorf("addScalarGo() = %v, want %v", dst, want)
	}
}

func TestSumGo(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5}
	got := sumGo(a)
	want := float32(15)
	if got != want {
		t.Errorf("sumGo() = %v, want %v", got, want)
	}
}

func TestAbsGo(t *testing.T) {
	a := []float32{-1, 2, -3}
	dst := make([]float32, 3)
	absGo(dst, a)
	want := []float32{1, 2, 3}
	if !refSlicesEqual32(dst, want) {
		t.Errorf("absGo() = %v, want %v", dst, want)
	}
}

func TestNegGo(t *testing.T) {
	a := []float32{1, -2, 3}
	dst := make([]float32, 3)
	negGo(dst, a)
	want := []float32{-1, 2, -3}
	if !refSlicesEqual32(dst, want) {
		t.Errorf("negGo() = %v, want %v", dst, want)
	}
}

func TestFmaGo(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	c := []float32{7, 8, 9}
	dst := make([]float32, 3)
	fmaGo(dst, a, b, c)
	// dst[i] = a[i]*b[i] + c[i]
	want := []float32{1*4 + 7, 2*5 + 8, 3*6 + 9}
	if !refSlicesEqual32(dst, want) {
		t.Errorf("fmaGo() = %v, want %v", dst, want)
	}
}

// Test that posInf and negInf are correctly defined
func TestInfConstants(t *testing.T) {
	if !math.IsInf(float64(posInf), 1) {
		t.Errorf("posInf should be +Inf")
	}
	if !math.IsInf(float64(negInf), -1) {
		t.Errorf("negInf should be -Inf")
	}
}

func TestAccumulateAdd32Go(t *testing.T) {
	dst := []float32{1, 2, 3, 4, 5}
	src := []float32{10, 20, 30, 40, 50}
	accumulateAdd32Go(dst, src)
	want := []float32{11, 22, 33, 44, 55}
	if !refSlicesEqual32(dst, want) {
		t.Errorf("accumulateAdd32Go() = %v, want %v", dst, want)
	}
}

func TestConvolveValidMultiGo(t *testing.T) {
	signal := []float32{1, 2, 3, 4, 5}
	kernels := [][]float32{{1, 1}, {1, -1}}
	kLen := 2
	n := len(signal) - kLen + 1

	dsts := make([][]float32, 2)
	dsts[0] = make([]float32, n)
	dsts[1] = make([]float32, n)

	convolveValidMultiGo(dsts, signal, kernels, n, kLen)

	// kernel[0] = {1, 1}: sum of pairs
	want0 := []float32{3, 5, 7, 9}
	// kernel[1] = {1, -1}: difference of pairs
	want1 := []float32{-1, -1, -1, -1}

	if !refSlicesEqual32(dsts[0], want0) {
		t.Errorf("convolveValidMultiGo() dsts[0] = %v, want %v", dsts[0], want0)
	}
	if !refSlicesEqual32(dsts[1], want1) {
		t.Errorf("convolveValidMultiGo() dsts[1] = %v, want %v", dsts[1], want1)
	}
}

func TestSqrt32Go(t *testing.T) {
	a := []float32{1, 4, 9, 16}
	dst := make([]float32, 4)
	sqrt32Go(dst, a)
	want := []float32{1, 2, 3, 4}
	if !refSlicesEqual32(dst, want) {
		t.Errorf("sqrt32Go() = %v, want %v", dst, want)
	}
}

func TestReciprocal32Go(t *testing.T) {
	a := []float32{1, 2, 4, 5}
	dst := make([]float32, 4)
	reciprocal32Go(dst, a)
	want := []float32{1, 0.5, 0.25, 0.2}
	if !refSlicesEqual32(dst, want) {
		t.Errorf("reciprocal32Go() = %v, want %v", dst, want)
	}
}

func TestAddScaledGo(t *testing.T) {
	dst := []float32{1, 2, 3, 4}
	s := []float32{10, 20, 30, 40}
	addScaledGo(dst, 0.5, s)
	want := []float32{6, 12, 18, 24}
	if !refSlicesEqual32(dst, want) {
		t.Errorf("addScaledGo() = %v, want %v", dst, want)
	}
}

func TestCumulativeSum32Go(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		want []float32
	}{
		{"empty", []float32{}, []float32{}},
		{"single", []float32{5}, []float32{5}},
		{"basic", []float32{1, 2, 3, 4}, []float32{1, 3, 6, 10}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			cumulativeSum32Go(dst, tt.a)
			if !refSlicesEqual32(dst, tt.want) {
				t.Errorf("cumulativeSum32Go() = %v, want %v", dst, tt.want)
			}
		})
	}
}

func TestVariance32Go(t *testing.T) {
	// Variance of [1, 2, 3, 4, 5] with mean 3:
	// ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5 = 2
	a := []float32{1, 2, 3, 4, 5}
	got := variance32Go(a, 3)
	want := float32(2.0)
	if !refAlmostEqual32(got, want) {
		t.Errorf("variance32Go() = %v, want %v", got, want)
	}
}

func TestEuclideanDistance32Go(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"same", []float32{1, 2, 3}, []float32{1, 2, 3}, 0},
		{"3d", []float32{0, 0, 0}, []float32{3, 4, 0}, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := euclideanDistance32Go(tt.a, tt.b)
			if !refAlmostEqual32(got, tt.want) {
				t.Errorf("euclideanDistance32Go() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestInterleave2Go(t *testing.T) {
	a := []float32{1, 3, 5}
	b := []float32{2, 4, 6}
	dst := make([]float32, 6)
	interleave2Go(dst, a, b)
	want := []float32{1, 2, 3, 4, 5, 6}
	if !refSlicesEqual32(dst, want) {
		t.Errorf("interleave2Go() = %v, want %v", dst, want)
	}
}

func TestDeinterleave2Go(t *testing.T) {
	src := []float32{1, 2, 3, 4, 5, 6}
	a := make([]float32, 3)
	b := make([]float32, 3)
	deinterleave2Go(a, b, src)
	wantA := []float32{1, 3, 5}
	wantB := []float32{2, 4, 6}
	if !refSlicesEqual32(a, wantA) {
		t.Errorf("deinterleave2Go() a = %v, want %v", a, wantA)
	}
	if !refSlicesEqual32(b, wantB) {
		t.Errorf("deinterleave2Go() b = %v, want %v", b, wantB)
	}
}

func TestMinIdxGo(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		want int
	}{
		{"empty", []float32{}, -1},
		{"single", []float32{5}, 0},
		{"min_at_start", []float32{1, 2, 3}, 0},
		{"min_at_end", []float32{3, 2, 1}, 2},
		{"min_in_middle", []float32{3, 1, 2}, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := minIdxGo(tt.a)
			if got != tt.want {
				t.Errorf("minIdxGo() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMaxIdxGo(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		want int
	}{
		{"empty", []float32{}, -1},
		{"single", []float32{5}, 0},
		{"max_at_start", []float32{3, 2, 1}, 0},
		{"max_at_end", []float32{1, 2, 3}, 2},
		{"max_in_middle", []float32{1, 3, 2}, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := maxIdxGo(tt.a)
			if got != tt.want {
				t.Errorf("maxIdxGo() = %v, want %v", got, tt.want)
			}
		})
	}
}

