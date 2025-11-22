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
		name        string
		a           []float32
		minV, maxV  float32
		want        []float32
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
		{},           // empty row
		{1, 2},       // shorter than vec
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
