package f32

import (
	"fmt"
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

func TestDotProduct(t *testing.T) {
	t.Logf("CPU: %s", cpu.Info())

	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"empty", nil, nil, 0},
		{"single", []float32{2}, []float32{3}, 6},
		{"two", []float32{1, 2}, []float32{3, 4}, 11},
		{"four", []float32{1, 2, 3, 4}, []float32{5, 6, 7, 8}, 70},
		{"eight", []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{8, 7, 6, 5, 4, 3, 2, 1}, 120},
		{"nine", []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, []float32{9, 8, 7, 6, 5, 4, 3, 2, 1}, 165},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DotProduct(tt.a, tt.b)
			if got != tt.want {
				t.Errorf("DotProduct() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAdd(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	b := []float32{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}
	dst := make([]float32, len(a))

	Add(dst, a, b)

	for i := range dst {
		if dst[i] != 11 {
			t.Errorf("Add()[%d] = %v, want 11", i, dst[i])
		}
	}
}

func TestMul(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	b := []float32{2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
	dst := make([]float32, len(a))

	Mul(dst, a, b)

	for i := range dst {
		want := float32(i+1) * 2
		if dst[i] != want {
			t.Errorf("Mul()[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

func TestSum(t *testing.T) {
	a := make([]float32, 100)
	for i := range a {
		a[i] = float32(i + 1)
	}

	got := Sum(a)
	want := float32(5050)
	if got != want {
		t.Errorf("Sum() = %v, want %v", got, want)
	}
}

func TestMinMax(t *testing.T) {
	a := []float32{5, 2, 8, 1, 9, 3, 7, 4, 6, 10}

	if got := Min(a); got != 1 {
		t.Errorf("Min() = %v, want 1", got)
	}
	if got := Max(a); got != 10 {
		t.Errorf("Max() = %v, want 10", got)
	}
}

func TestAbs(t *testing.T) {
	a := []float32{-1, 2, -3, 4, -5, 6, -7, 8, -9, 10}
	dst := make([]float32, len(a))

	Abs(dst, a)

	for i := range dst {
		want := float32(math.Abs(float64(a[i])))
		if dst[i] != want {
			t.Errorf("Abs()[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

func TestFMA(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	b := []float32{2, 2, 2, 2, 2, 2, 2, 2}
	c := []float32{1, 1, 1, 1, 1, 1, 1, 1}
	dst := make([]float32, len(a))

	FMA(dst, a, b, c)

	for i := range dst {
		want := a[i]*b[i] + c[i]
		if dst[i] != want {
			t.Errorf("FMA()[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

func TestClamp(t *testing.T) {
	a := []float32{-5, -2, 0, 5, 10, 15, 20}
	dst := make([]float32, len(a))

	Clamp(dst, a, 0, 10)

	want := []float32{0, 0, 0, 5, 10, 10, 10}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("Clamp()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestSigmoid(t *testing.T) {
	testCases := []struct {
		name string
		src  []float32
	}{
		{"zeros", []float32{0, 0, 0, 0, 0, 0, 0, 0}},
		{"ones", []float32{1, 1, 1, 1, 1, 1, 1, 1}},
		{"negative", []float32{-1, -2, -3, -4, -5, -6, -7, -8}},
		{"positive", []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		{"mixed", []float32{-3, -1, 0, 1, 3, -5, 5, -8}},
		{"large_positive", []float32{10, 20, 30}},
		{"large_negative", []float32{-10, -20, -30}},
		{"nine_elements", []float32{-2, -1, 0, 1, 2, 3, 4, 5, 6}},
		{"seventeen", make([]float32, 17)},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			dst := make([]float32, len(tc.src))

			// Fill seventeen test case with sequential values
			if tc.name == "seventeen" {
				for i := range tc.src {
					tc.src[i] = float32(i - 8)
				}
			}

			// Test Sigmoid
			Sigmoid(dst, tc.src)

			// Verify results are in valid range (0, 1)
			for i, v := range dst {
				if v < 0 || v > 1 {
					t.Errorf("Sigmoid()[%d] = %v, expected value in range (0, 1)", i, v)
				}

				// Test specific values
				if tc.src[i] == 0 && math.Abs(float64(v-0.5)) > 0.01 {
					t.Errorf("Sigmoid(0)[%d] = %v, expected ~0.5", i, v)
				}
				if tc.src[i] > 10 && v < 0.99 {
					t.Errorf("Sigmoid(%v)[%d] = %v, expected > 0.99", tc.src[i], i, v)
				}
				if tc.src[i] < -10 && v > 0.01 {
					t.Errorf("Sigmoid(%v)[%d] = %v, expected < 0.01", tc.src[i], i, v)
				}
			}

			// Test SigmoidInPlace
			src2 := make([]float32, len(tc.src))
			copy(src2, tc.src)
			SigmoidInPlace(src2)

			// Verify in-place results match
			for i, v := range src2 {
				if math.Abs(float64(v-dst[i])) > 1e-6 {
					t.Errorf("SigmoidInPlace()[%d] = %v, Sigmoid() = %v, expected same", i, v, dst[i])
				}
			}
		})
	}
}

func TestReLU(t *testing.T) {
	src := []float32{-5, -2, -0.5, 0, 0.5, 2, 5}
	dst := make([]float32, len(src))
	want := []float32{0, 0, 0, 0, 0.5, 2, 5}

	ReLU(dst, src)

	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("ReLU()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}

	// Test in-place
	src2 := append([]float32(nil), src...)
	ReLUInPlace(src2)
	for i := range src2 {
		if src2[i] != want[i] {
			t.Errorf("ReLUInPlace()[%d] = %v, want %v", i, src2[i], want[i])
		}
	}
}

func TestClampScale(t *testing.T) {
	src := []float32{-5, 0, 5, 10, 15, 20}
	dst := make([]float32, len(src))

	// Clamp to [0, 10] then scale by 0.1
	ClampScale(dst, src, 0, 10, 0.1)

	// Expected: (clamp(x, 0, 10) - 0) * 0.1
	want := []float32{0, 0, 0.5, 1.0, 1.0, 1.0}
	for i := range dst {
		if math.Abs(float64(dst[i]-want[i])) > 1e-6 {
			t.Errorf("ClampScale()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestTanh(t *testing.T) {
	// Test cases with various input ranges including extreme values that trigger clamping
	testCases := []struct {
		name string
		src  []float32
	}{
		{"zeros", []float32{0, 0, 0, 0, 0, 0, 0, 0}},
		{"ones", []float32{1, 1, 1, 1, 1, 1, 1, 1}},
		{"negative", []float32{-1, -2, -3, -4, -5, -6, -7, -8}},
		{"positive", []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		{"mixed", []float32{-3, -1, 0, 1, 3, -5, 5, -8}},
		// Extreme values that MUST trigger clamping (clamp range is ±20 for -2x)
		// These values test the FMIN/FMAX instructions
		{"extreme_positive", []float32{15, 20, 50, 100}},
		{"extreme_negative", []float32{-15, -20, -50, -100}},
		{"extreme_mixed", []float32{-100, -50, -20, -10, 0, 10, 20, 50, 100}},
		// Various sizes to test NEON vectorized path and scalar remainder
		{"size_1", []float32{2.5}},
		{"size_3", []float32{-1, 0, 1}},
		{"size_4", []float32{-2, -1, 1, 2}},    // exact NEON width for f32
		{"size_5", []float32{-2, -1, 0, 1, 2}}, // NEON + 1 scalar
		{"size_7", []float32{-3, -2, -1, 0, 1, 2, 3}},
		{"size_9", []float32{-4, -3, -2, -1, 0, 1, 2, 3, 4}},
		{"size_17", make([]float32, 17)},
	}

	// Fill size_17 with values including extremes
	for i := range 17 {
		testCases[len(testCases)-1].src[i] = float32(i-8) * 10 // -80 to +80
	}

	const epsilon = 1e-5 // Acceptable relative error for f32

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			dst := make([]float32, len(tc.src))
			Tanh(dst, tc.src)

			for i, v := range dst {
				// Verify results are in valid range [-1, 1]
				if v < -1 || v > 1 {
					t.Errorf("Tanh(%v)[%d] = %v, expected value in range [-1, 1]", tc.src[i], i, v)
				}

				// Verify accuracy against math.Tanh
				expected := float32(math.Tanh(float64(tc.src[i])))
				diff := math.Abs(float64(v - expected))

				// Use relative error for values away from zero, absolute for near-zero
				var relErr float64
				if math.Abs(float64(expected)) > 1e-6 {
					relErr = diff / math.Abs(float64(expected))
				} else {
					relErr = diff
				}

				if relErr > epsilon {
					t.Errorf("Tanh(%v)[%d] = %v, want %v (error: %v)", tc.src[i], i, v, expected, relErr)
				}
			}
		})
	}

	// Test in-place operation
	t.Run("in_place", func(t *testing.T) {
		src := []float32{-100, -50, -10, -1, 0, 1, 10, 50, 100}
		dst := make([]float32, len(src))
		Tanh(dst, src)

		inPlace := make([]float32, len(src))
		copy(inPlace, src)
		TanhInPlace(inPlace)

		for i := range dst {
			if math.Abs(float64(dst[i]-inPlace[i])) > 1e-6 {
				t.Errorf("TanhInPlace()[%d] = %v, Tanh() = %v, expected same", i, inPlace[i], dst[i])
			}
		}
	})
}

func TestExp(t *testing.T) {
	src := []float32{-2, -1, 0, 1, 2}
	dst := make([]float32, len(src))

	Exp(dst, src)

	// Verify exp(0) = 1
	if math.Abs(float64(dst[2]-1.0)) > 0.01 {
		t.Errorf("Exp(0) = %v, want ~1.0", dst[2])
	}

	// Verify exp is positive
	for i, v := range dst {
		if v <= 0 {
			t.Errorf("Exp()[%d] = %v, expected positive value", i, v)
		}
	}

	// Test in-place
	src2 := append([]float32(nil), src...)
	ExpInPlace(src2)
	for i := range src2 {
		if math.Abs(float64(src2[i]-dst[i])) > 0.01 {
			t.Errorf("ExpInPlace()[%d] = %v, Exp() = %v, expected similar", i, src2[i], dst[i])
		}
	}
}

// Benchmarks

func BenchmarkDotProduct_100(b *testing.B) {
	benchmarkDotProduct32(b, 100)
}

func BenchmarkDotProduct_1000(b *testing.B) {
	benchmarkDotProduct32(b, 1000)
}

func BenchmarkDotProduct_10000(b *testing.B) {
	benchmarkDotProduct32(b, 10000)
}

func benchmarkDotProduct32(b *testing.B, n int) {
	b.Helper()
	a := make([]float32, n)
	x := make([]float32, n)
	for i := range a {
		a[i] = float32(i)
		x[i] = float32(n - i)
	}

	b.ResetTimer()
	b.SetBytes(int64(n * 4 * 2))

	var result float32
	for i := 0; i < b.N; i++ {
		result = DotProduct(a, x)
	}
	_ = result
}

func BenchmarkAdd_1000(b *testing.B) {
	a := make([]float32, 1000)
	c := make([]float32, 1000)
	dst := make([]float32, 1000)

	b.SetBytes(1000 * 4 * 3)

	for b.Loop() {
		Add(dst, a, c)
	}
}

func BenchmarkMul_1000(b *testing.B) {
	a := make([]float32, 1000)
	c := make([]float32, 1000)
	dst := make([]float32, 1000)

	b.SetBytes(1000 * 4 * 3)

	for b.Loop() {
		Mul(dst, a, c)
	}
}

func BenchmarkFMA_1000(b *testing.B) {
	a := make([]float32, 1000)
	c := make([]float32, 1000)
	d := make([]float32, 1000)
	dst := make([]float32, 1000)

	b.SetBytes(1000 * 4 * 4)

	for b.Loop() {
		FMA(dst, a, c, d)
	}
}

// Tests for DotProductBatch

func TestDotProductBatch(t *testing.T) {
	tests := []struct {
		name string
		rows [][]float32
		vec  []float32
		want []float32
	}{
		{"empty", nil, nil, nil},
		{"single row", [][]float32{{1, 2, 3}}, []float32{1, 1, 1}, []float32{6}},
		{"two rows", [][]float32{{1, 2}, {3, 4}}, []float32{1, 2}, []float32{5, 11}},
		{"polyphase", [][]float32{{1, 0, 1, 0}, {0, 1, 0, 1}}, []float32{1, 2, 3, 4}, []float32{4, 6}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.rows) == 0 {
				return
			}
			results := make([]float32, len(tt.rows))
			DotProductBatch(results, tt.rows, tt.vec)
			for i, want := range tt.want {
				if results[i] != want {
					t.Errorf("DotProductBatch()[%d] = %v, want %v", i, results[i], want)
				}
			}
		})
	}
}

// Tests for ConvolveValid

func TestConvolveValid(t *testing.T) {
	tests := []struct {
		name   string
		signal []float32
		kernel []float32
		want   []float32
	}{
		{"empty kernel", []float32{1, 2, 3}, nil, nil},
		{"kernel longer", []float32{1, 2}, []float32{1, 2, 3}, nil},
		{"single output", []float32{1, 2, 3}, []float32{1, 1, 1}, []float32{6}},
		{"two outputs", []float32{1, 2, 3, 4}, []float32{1, 1, 1}, []float32{6, 9}},
		{"identity", []float32{1, 2, 3, 4, 5}, []float32{1}, []float32{1, 2, 3, 4, 5}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.kernel) == 0 || len(tt.signal) < len(tt.kernel) {
				return
			}
			validLen := len(tt.signal) - len(tt.kernel) + 1
			dst := make([]float32, validLen)
			ConvolveValid(dst, tt.signal, tt.kernel)
			for i, want := range tt.want {
				if dst[i] != want {
					t.Errorf("ConvolveValid()[%d] = %v, want %v", i, dst[i], want)
				}
			}
		})
	}
}

// Tests for AccumulateAdd

func TestAccumulateAdd(t *testing.T) {
	tests := []struct {
		name   string
		dst    []float32
		src    []float32
		offset int
		want   []float32
	}{
		{
			"basic",
			[]float32{1, 2, 3, 4, 5},
			[]float32{10, 20},
			1,
			[]float32{1, 12, 23, 4, 5},
		},
		{
			"at start",
			[]float32{1, 2, 3, 4},
			[]float32{10, 20, 30},
			0,
			[]float32{11, 22, 33, 4},
		},
		{
			"at end",
			[]float32{1, 2, 3, 4},
			[]float32{10, 20},
			2,
			[]float32{1, 2, 13, 24},
		},
		{
			"empty src",
			[]float32{1, 2, 3},
			[]float32{},
			0,
			[]float32{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.dst))
			copy(dst, tt.dst)
			AccumulateAdd(dst, tt.src, tt.offset)
			for i := range dst {
				if dst[i] != tt.want[i] {
					t.Errorf("AccumulateAdd()[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestAccumulateAdd_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for negative offset")
		}
	}()
	dst := make([]float32, 5)
	src := []float32{1, 2}
	AccumulateAdd(dst, src, -1)
}

func TestAccumulateAdd_PanicsOverflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for overflow")
		}
	}()
	dst := make([]float32, 5)
	src := []float32{1, 2, 3}
	AccumulateAdd(dst, src, 4) // 4+3 > 5
}

// Benchmarks for new functions

func BenchmarkDotProductBatch_2x241(b *testing.B) {
	rows := make([][]float32, 2)
	for i := range rows {
		rows[i] = make([]float32, 241)
		for j := range rows[i] {
			rows[i][j] = float32(j + 1)
		}
	}
	vec := make([]float32, 241)
	for i := range vec {
		vec[i] = float32(i + 1)
	}
	results := make([]float32, 2)

	b.SetBytes(2 * 241 * 4 * 2)

	for b.Loop() {
		DotProductBatch(results, rows, vec)
	}
}

func BenchmarkConvolveValid_1000x64(b *testing.B) {
	signal := make([]float32, 1000)
	kernel := make([]float32, 64)
	for i := range signal {
		signal[i] = float32(i)
	}
	for i := range kernel {
		kernel[i] = 1.0 / 64.0
	}
	validLen := len(signal) - len(kernel) + 1
	dst := make([]float32, validLen)

	b.SetBytes(int64(validLen * 64 * 4 * 2))

	for b.Loop() {
		ConvolveValid(dst, signal, kernel)
	}
}

func BenchmarkAccumulateAdd_1000(b *testing.B) {
	dst := make([]float32, 1000)
	src := make([]float32, 500)
	for i := range src {
		src[i] = float32(i)
	}

	b.SetBytes(500 * 4 * 2) // read src, read+write dst

	for b.Loop() {
		// Reset dst for consistent benchmark
		for i := range dst {
			dst[i] = float32(i)
		}
		AccumulateAdd(dst, src, 250)
	}
}

// Tests for Interleave2 and Deinterleave2

func TestInterleave2(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		b    []float32
		want []float32
	}{
		{"single", []float32{1}, []float32{2}, []float32{1, 2}},
		{"two", []float32{1, 3}, []float32{2, 4}, []float32{1, 2, 3, 4}},
		{"four", []float32{1, 3, 5, 7}, []float32{2, 4, 6, 8}, []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		{"eight", []float32{1, 3, 5, 7, 9, 11, 13, 15}, []float32{2, 4, 6, 8, 10, 12, 14, 16},
			[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a)*2)
			Interleave2(dst, tt.a, tt.b)
			for i, want := range tt.want {
				if dst[i] != want {
					t.Errorf("Interleave2()[%d] = %v, want %v", i, dst[i], want)
				}
			}
		})
	}
}

func TestInterleave2_Large(t *testing.T) {
	sizes := []int{8, 16, 17, 100, 1024, 1025}
	for _, n := range sizes {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := range a {
			a[i] = float32(i * 2)
			b[i] = float32(i*2 + 1)
		}

		dst := make([]float32, n*2)
		Interleave2(dst, a, b)

		for i := range n {
			if dst[i*2] != a[i] {
				t.Errorf("size=%d: Interleave2()[%d] = %v, want %v", n, i*2, dst[i*2], a[i])
			}
			if dst[i*2+1] != b[i] {
				t.Errorf("size=%d: Interleave2()[%d] = %v, want %v", n, i*2+1, dst[i*2+1], b[i])
			}
		}
	}
}

func TestDeinterleave2(t *testing.T) {
	tests := []struct {
		name  string
		src   []float32
		wantA []float32
		wantB []float32
	}{
		{"single", []float32{1, 2}, []float32{1}, []float32{2}},
		{"two", []float32{1, 2, 3, 4}, []float32{1, 3}, []float32{2, 4}},
		{"four", []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{1, 3, 5, 7}, []float32{2, 4, 6, 8}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := len(tt.src) / 2
			a := make([]float32, n)
			b := make([]float32, n)
			Deinterleave2(a, b, tt.src)
			for i, want := range tt.wantA {
				if a[i] != want {
					t.Errorf("Deinterleave2() a[%d] = %v, want %v", i, a[i], want)
				}
			}
			for i, want := range tt.wantB {
				if b[i] != want {
					t.Errorf("Deinterleave2() b[%d] = %v, want %v", i, b[i], want)
				}
			}
		})
	}
}

func TestDeinterleave2_Large(t *testing.T) {
	sizes := []int{8, 16, 17, 100, 1024, 1025}
	for _, n := range sizes {
		src := make([]float32, n*2)
		for i := range src {
			src[i] = float32(i)
		}

		a := make([]float32, n)
		b := make([]float32, n)
		Deinterleave2(a, b, src)

		for i := range n {
			wantA := float32(i * 2)
			wantB := float32(i*2 + 1)
			if a[i] != wantA {
				t.Errorf("size=%d: Deinterleave2() a[%d] = %v, want %v", n, i, a[i], wantA)
			}
			if b[i] != wantB {
				t.Errorf("size=%d: Deinterleave2() b[%d] = %v, want %v", n, i, b[i], wantB)
			}
		}
	}
}

func TestInterleaveDeinterleaveRoundTrip(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 100}
	for _, n := range sizes {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := range a {
			a[i] = float32(i)
			b[i] = float32(i + 1000)
		}

		interleaved := make([]float32, n*2)
		Interleave2(interleaved, a, b)

		aOut := make([]float32, n)
		bOut := make([]float32, n)
		Deinterleave2(aOut, bOut, interleaved)

		for i := range n {
			if aOut[i] != a[i] {
				t.Errorf("size=%d: round-trip a[%d] = %v, want %v", n, i, aOut[i], a[i])
			}
			if bOut[i] != b[i] {
				t.Errorf("size=%d: round-trip b[%d] = %v, want %v", n, i, bOut[i], b[i])
			}
		}
	}
}

func TestConvolveValidMulti(t *testing.T) {
	tests := []struct {
		name    string
		signal  []float32
		kernels [][]float32
		want    [][]float32
	}{
		{
			"two_kernels_simple",
			[]float32{1, 2, 3, 4, 5},
			[][]float32{{1, 1}, {1, -1}},
			[][]float32{{3, 5, 7, 9}, {-1, -1, -1, -1}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kLen := len(tt.kernels[0])
			validLen := len(tt.signal) - kLen + 1
			dsts := make([][]float32, len(tt.kernels))
			for i := range dsts {
				dsts[i] = make([]float32, validLen)
			}

			ConvolveValidMulti(dsts, tt.signal, tt.kernels)

			for k, want := range tt.want {
				for i, v := range want {
					if dsts[k][i] != v {
						t.Errorf("ConvolveValidMulti() dsts[%d][%d] = %v, want %v", k, i, dsts[k][i], v)
					}
				}
			}
		})
	}
}

func BenchmarkInterleave2_1000(b *testing.B) {
	a := make([]float32, 1000)
	c := make([]float32, 1000)
	dst := make([]float32, 2000)
	for i := range a {
		a[i] = float32(i)
		c[i] = float32(i + 1000)
	}

	b.SetBytes(1000 * 4 * 3)

	for b.Loop() {
		Interleave2(dst, a, c)
	}
}

func BenchmarkDeinterleave2_1000(b *testing.B) {
	src := make([]float32, 2000)
	a := make([]float32, 1000)
	c := make([]float32, 1000)
	for i := range src {
		src[i] = float32(i)
	}

	b.SetBytes(1000 * 4 * 3)

	for b.Loop() {
		Deinterleave2(a, c, src)
	}
}

// Tests for new functions

func TestSqrt(t *testing.T) {
	a := []float32{0, 1, 4, 9, 16, 25, 36, 49, 64, 81}
	dst := make([]float32, len(a))
	want := []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

	Sqrt(dst, a)

	for i := range dst {
		if math.Abs(float64(dst[i]-want[i])) > 1e-6 {
			t.Errorf("Sqrt()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestSqrt_Large(t *testing.T) {
	sizes := []int{7, 8, 9, 15, 16, 17, 100}
	for _, n := range sizes {
		a := make([]float32, n)
		dst := make([]float32, n)
		for i := range a {
			a[i] = float32(i * i)
		}

		Sqrt(dst, a)

		for i := range dst {
			want := float32(i)
			if math.Abs(float64(dst[i]-want)) > 1e-5 {
				t.Errorf("size=%d: Sqrt()[%d] = %v, want %v", n, i, dst[i], want)
			}
		}
	}
}

func TestReciprocal(t *testing.T) {
	a := []float32{1, 2, 4, 5, 8, 10, 20, 25}
	dst := make([]float32, len(a))
	want := []float32{1, 0.5, 0.25, 0.2, 0.125, 0.1, 0.05, 0.04}

	Reciprocal(dst, a)

	for i := range dst {
		if math.Abs(float64(dst[i]-want[i])) > 1e-6 {
			t.Errorf("Reciprocal()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestReciprocal_Large(t *testing.T) {
	sizes := []int{7, 8, 9, 15, 16, 17, 100}
	for _, n := range sizes {
		a := make([]float32, n)
		dst := make([]float32, n)
		for i := range a {
			a[i] = float32(i + 1)
		}

		Reciprocal(dst, a)

		for i := range dst {
			want := 1.0 / float32(i+1)
			if math.Abs(float64(dst[i]-want)) > 1e-5 {
				t.Errorf("size=%d: Reciprocal()[%d] = %v, want %v", n, i, dst[i], want)
			}
		}
	}
}

func TestMinIdx(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		want int
	}{
		{"empty", nil, -1},
		{"single", []float32{5}, 0},
		{"first", []float32{1, 2, 3, 4, 5}, 0},
		{"last", []float32{5, 4, 3, 2, 1}, 4},
		{"middle", []float32{5, 2, 1, 3, 4}, 2},
		{"large", []float32{5, 2, 8, 1, 9, 3, 7, 4, 6, 10}, 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MinIdx(tt.a)
			if got != tt.want {
				t.Errorf("MinIdx() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMaxIdx(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		want int
	}{
		{"empty", nil, -1},
		{"single", []float32{5}, 0},
		{"first", []float32{5, 4, 3, 2, 1}, 0},
		{"last", []float32{1, 2, 3, 4, 5}, 4},
		{"middle", []float32{1, 2, 5, 3, 4}, 2},
		{"large", []float32{5, 2, 8, 1, 9, 3, 7, 4, 6, 10}, 9},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MaxIdx(tt.a)
			if got != tt.want {
				t.Errorf("MaxIdx() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAddScaled(t *testing.T) {
	dst := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	s := []float32{1, 1, 1, 1, 1, 1, 1, 1}
	alpha := float32(2.0)
	want := []float32{3, 4, 5, 6, 7, 8, 9, 10}

	AddScaled(dst, alpha, s)

	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("AddScaled()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestAddScaled_Large(t *testing.T) {
	sizes := []int{7, 8, 9, 15, 16, 17, 100}
	for _, n := range sizes {
		dst := make([]float32, n)
		s := make([]float32, n)
		for i := range dst {
			dst[i] = float32(i)
			s[i] = 1.0
		}
		alpha := float32(3.0)

		AddScaled(dst, alpha, s)

		for i := range dst {
			want := float32(i) + 3.0
			if dst[i] != want {
				t.Errorf("size=%d: AddScaled()[%d] = %v, want %v", n, i, dst[i], want)
			}
		}
	}
}

func TestCumulativeSum(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5}
	dst := make([]float32, len(a))
	want := []float32{1, 3, 6, 10, 15}

	CumulativeSum(dst, a)

	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("CumulativeSum()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestNormalize(t *testing.T) {
	a := []float32{3, 4}
	dst := make([]float32, len(a))

	Normalize(dst, a)

	// Magnitude of (3,4) is 5, so normalized is (0.6, 0.8)
	want := []float32{0.6, 0.8}
	for i := range dst {
		if math.Abs(float64(dst[i]-want[i])) > 1e-6 {
			t.Errorf("Normalize()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestNormalize_ZeroVector(t *testing.T) {
	a := []float32{0, 0, 0}
	dst := make([]float32, len(a))

	Normalize(dst, a)

	// Zero vector should be unchanged
	for i := range dst {
		if dst[i] != 0 {
			t.Errorf("Normalize() of zero vector [%d] = %v, want 0", i, dst[i])
		}
	}
}

func TestMean(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5}
	want := float32(3.0)

	got := Mean(a)
	if got != want {
		t.Errorf("Mean() = %v, want %v", got, want)
	}
}

func TestVariance(t *testing.T) {
	a := []float32{2, 4, 4, 4, 5, 5, 7, 9}
	// Mean = 5, Variance = ((2-5)^2 + (4-5)^2*3 + (5-5)^2*2 + (7-5)^2 + (9-5)^2) / 8 = 4
	want := float32(4.0)

	got := Variance(a)
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Errorf("Variance() = %v, want %v", got, want)
	}
}

func TestStdDev(t *testing.T) {
	a := []float32{2, 4, 4, 4, 5, 5, 7, 9}
	want := float32(2.0)

	got := StdDev(a)
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Errorf("StdDev() = %v, want %v", got, want)
	}
}

// Benchmarks for new functions

func BenchmarkSqrt_1000(b *testing.B) {
	a := make([]float32, 1000)
	dst := make([]float32, 1000)
	for i := range a {
		a[i] = float32(i + 1)
	}

	b.SetBytes(1000 * 4 * 2)

	for b.Loop() {
		Sqrt(dst, a)
	}
}

func BenchmarkReciprocal_1000(b *testing.B) {
	a := make([]float32, 1000)
	dst := make([]float32, 1000)
	for i := range a {
		a[i] = float32(i + 1)
	}

	b.SetBytes(1000 * 4 * 2)

	for b.Loop() {
		Reciprocal(dst, a)
	}
}

func BenchmarkMinIdx_1000(b *testing.B) {
	a := make([]float32, 1000)
	for i := range a {
		a[i] = float32(i)
	}
	a[500] = -1 // minimum in middle

	b.SetBytes(1000 * 4)

	var idx int
	for b.Loop() {
		idx = MinIdx(a)
	}
	_ = idx
}

func BenchmarkAddScaled_1000(b *testing.B) {
	dst := make([]float32, 1000)
	s := make([]float32, 1000)
	for i := range dst {
		dst[i] = float32(i)
		s[i] = 1.0
	}

	b.SetBytes(1000 * 4 * 3) // read dst, read s, write dst

	for b.Loop() {
		AddScaled(dst, 2.0, s)
	}
}

func BenchmarkNormalize_1000(b *testing.B) {
	a := make([]float32, 1000)
	dst := make([]float32, 1000)
	for i := range a {
		a[i] = float32(i + 1)
	}

	b.SetBytes(1000 * 4 * 2)

	for b.Loop() {
		Normalize(dst, a)
	}
}

func BenchmarkCumulativeSum_1000(b *testing.B) {
	a := make([]float32, 1000)
	dst := make([]float32, 1000)
	for i := range a {
		a[i] = float32(i + 1)
	}

	b.SetBytes(1000 * 4 * 2)

	for b.Loop() {
		CumulativeSum(dst, a)
	}
}

func BenchmarkMean_1000(b *testing.B) {
	a := make([]float32, 1000)
	for i := range a {
		a[i] = float32(i + 1)
	}

	b.SetBytes(1000 * 4)

	var result float32
	for b.Loop() {
		result = Mean(a)
	}
	_ = result
}

func BenchmarkVariance_1000(b *testing.B) {
	a := make([]float32, 1000)
	for i := range a {
		a[i] = float32(i + 1)
	}

	b.SetBytes(1000 * 4)

	var result float32
	for b.Loop() {
		result = Variance(a)
	}
	_ = result
}

func BenchmarkStdDev_1000(b *testing.B) {
	a := make([]float32, 1000)
	for i := range a {
		a[i] = float32(i + 1)
	}

	b.SetBytes(1000 * 4)

	var result float32
	for b.Loop() {
		result = StdDev(a)
	}
	_ = result
}

func TestEuclideanDistance(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"empty", nil, nil, 0},
		{"same", []float32{1, 2, 3}, []float32{1, 2, 3}, 0},
		{"2d", []float32{0, 0}, []float32{3, 4}, 5},                              // 3-4-5 triangle
		{"3d", []float32{1, 2, 3}, []float32{4, 6, 8}, float32(math.Sqrt(50))},   // sqrt(9+16+25)
		{"negative", []float32{-1, -2}, []float32{1, 2}, float32(math.Sqrt(20))}, // sqrt(4+16)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EuclideanDistance(tt.a, tt.b)
			if math.Abs(float64(got-tt.want)) > 1e-5 {
				t.Errorf("EuclideanDistance() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEuclideanDistance_Large(t *testing.T) {
	n := 1000
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32(i)
		b[i] = float32(i + 1) // diff = 1 for each element
	}

	got := EuclideanDistance(a, b)
	// Each element differs by 1, so sum of squares = n, distance = sqrt(n)
	want := float32(math.Sqrt(float64(n)))
	if math.Abs(float64(got-want)) > 1e-3 {
		t.Errorf("EuclideanDistance large = %v, want %v", got, want)
	}
}

func BenchmarkEuclideanDistance_1000(b *testing.B) {
	a := make([]float32, 1000)
	v := make([]float32, 1000)
	for i := range a {
		a[i] = float32(i)
		v[i] = float32(i + 1)
	}

	b.SetBytes(1000 * 4 * 2)

	var result float32
	for b.Loop() {
		result = EuclideanDistance(a, v)
	}
	_ = result
}

// Edge case tests for empty slices and early returns

func TestInterleave2_Empty(_ *testing.T) {
	var a, b, dst []float32
	Interleave2(dst, a, b)
}

func TestDeinterleave2_Empty(_ *testing.T) {
	var a, b, src []float32
	Deinterleave2(a, b, src)
}

func TestConvolveValidMulti_Empty(_ *testing.T) {
	signal := []float32{1, 2, 3, 4, 5}
	var kernels [][]float32
	var dsts [][]float32
	ConvolveValidMulti(dsts, signal, kernels)
}

func TestConvolveValidMulti_KernelLongerThanSignal(_ *testing.T) {
	signal := []float32{1, 2}
	kernels := [][]float32{{1, 2, 3, 4}}
	dsts := [][]float32{make([]float32, 0)}
	ConvolveValidMulti(dsts, signal, kernels)
}

func TestConvolveValidMulti_MismatchedKernelLengths(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for mismatched kernel lengths")
		}
	}()
	signal := []float32{1, 2, 3, 4, 5}
	kernels := [][]float32{{1, 1}, {1, 1, 1}}
	dsts := [][]float32{make([]float32, 4), make([]float32, 3)}
	ConvolveValidMulti(dsts, signal, kernels)
}

func TestSqrt_Empty(_ *testing.T) {
	var a, dst []float32
	Sqrt(dst, a)
}

func TestReciprocal_Empty(_ *testing.T) {
	var a, dst []float32
	Reciprocal(dst, a)
}

func TestAddScaled_Empty(_ *testing.T) {
	var dst, s []float32
	AddScaled(dst, 2.0, s)
}

func TestCumulativeSum_Empty(_ *testing.T) {
	var a, dst []float32
	CumulativeSum(dst, a)
}

func TestNormalize_Empty(_ *testing.T) {
	var a, dst []float32
	Normalize(dst, a)
}

func TestMean_Empty(t *testing.T) {
	var a []float32
	got := Mean(a)
	if got != 0 {
		t.Errorf("Mean(empty) = %v, want 0", got)
	}
}

func TestVariance_Empty(t *testing.T) {
	var a []float32
	got := Variance(a)
	if got != 0 {
		t.Errorf("Variance(empty) = %v, want 0", got)
	}
}

func TestVariance_Single(t *testing.T) {
	a := []float32{5}
	got := Variance(a)
	if got != 0 {
		t.Errorf("Variance(single) = %v, want 0", got)
	}
}

func TestConvolveValidMulti_SmallDst(t *testing.T) {
	// Test when dst is smaller than validLen
	signal := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	kernels := [][]float32{{1, 1}, {1, -1}}
	// validLen would be 9, but we provide smaller dst
	dsts := [][]float32{make([]float32, 3), make([]float32, 3)}

	ConvolveValidMulti(dsts, signal, kernels)

	// Should only fill first 3 elements
	if len(dsts[0]) != 3 {
		t.Errorf("dst[0] length = %d, want 3", len(dsts[0]))
	}
	// Verify values: {1,1} kernel gives sums
	want0 := []float32{3, 5, 7}
	for i, v := range want0 {
		if dsts[0][i] != v {
			t.Errorf("dsts[0][%d] = %v, want %v", i, dsts[0][i], v)
		}
	}
}

func TestConvolveValidMulti_ZeroLengthDst(_ *testing.T) {
	signal := []float32{1, 2, 3, 4, 5}
	kernels := [][]float32{{1, 1}, {1, -1}}
	dsts := [][]float32{make([]float32, 0), make([]float32, 0)}
	ConvolveValidMulti(dsts, signal, kernels)
}

func TestConvolveValidMulti_FewerDstsThanKernels(_ *testing.T) {
	signal := []float32{1, 2, 3, 4, 5}
	kernels := [][]float32{{1, 1}, {1, -1}}
	dsts := [][]float32{make([]float32, 4)}
	ConvolveValidMulti(dsts, signal, kernels)
}

// Tests for CubicInterpDot

// cubicInterpDotRef32 is a reference implementation for testing
func cubicInterpDotRef32(hist, a, b, c, d []float32, x float32) float32 {
	var sum float32
	for i := range hist {
		coef := a[i] + x*(b[i]+x*(c[i]+x*d[i]))
		sum += hist[i] * coef
	}
	return sum
}

func TestCubicInterpDot(t *testing.T) {
	t.Logf("CPU: %s", cpu.Info())

	tests := []struct {
		name string
		n    int
		x    float32
	}{
		{"empty", 0, 0.5},
		{"single", 1, 0.5},
		{"two", 2, 0.5},
		{"three", 3, 0.5},
		{"four", 4, 0.5},          // NEON vector width
		{"five", 5, 0.5},          // NEON + scalar remainder
		{"seven", 7, 0.5},         // Below AVX threshold
		{"eight", 8, 0.5},         // AVX vector width
		{"nine", 9, 0.5},          // AVX + scalar remainder
		{"fifteen", 15, 0.5},      // Below 2x AVX
		{"sixteen", 16, 0.5},      // 2x AVX / AVX-512 vector width
		{"seventeen", 17, 0.5},    // 2x AVX + scalar remainder
		{"thirty_two", 32, 0.5},   // 2x AVX-512
		{"thirty_three", 33, 0.5}, // 2x AVX-512 + remainder
		{"x=0", 16, 0.0},          // Edge case: x=0 means only a[] matters
		{"x=0.999", 16, 0.999},    // Near boundary
		{"x=1", 16, 1.0},          // Edge case
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.n == 0 {
				got := CubicInterpDot(nil, nil, nil, nil, nil, tt.x)
				if got != 0 {
					t.Errorf("CubicInterpDot(empty) = %v, want 0", got)
				}
				return
			}

			hist := make([]float32, tt.n)
			a := make([]float32, tt.n)
			b := make([]float32, tt.n)
			c := make([]float32, tt.n)
			d := make([]float32, tt.n)

			for i := range tt.n {
				hist[i] = float32(i+1) * 0.1
				a[i] = float32(i) * 0.5
				b[i] = float32(i) * 0.3
				c[i] = float32(i) * 0.2
				d[i] = float32(i) * 0.1
			}

			got := CubicInterpDot(hist, a, b, c, d, tt.x)
			want := cubicInterpDotRef32(hist, a, b, c, d, tt.x)

			// Use relative tolerance for floating point comparison
			tol := float32(1e-5)
			if want != 0 {
				tol = float32(math.Abs(float64(want)) * 1e-5)
			}
			if math.Abs(float64(got-want)) > float64(tol) {
				t.Errorf("CubicInterpDot() = %v, want %v, diff = %v", got, want, got-want)
			}
		})
	}
}

func TestCubicInterpDot_Large(t *testing.T) {
	// Test with various sizes typical for audio resampling
	sizes := []int{32, 64, 100, 241, 1000}

	for _, n := range sizes {
		hist := make([]float32, n)
		a := make([]float32, n)
		b := make([]float32, n)
		c := make([]float32, n)
		d := make([]float32, n)

		for i := range n {
			hist[i] = float32(i+1) * 0.01
			a[i] = float32(i) * 0.1
			b[i] = float32(i) * 0.05
			c[i] = float32(i) * 0.02
			d[i] = float32(i) * 0.01
		}

		x := float32(0.75)
		got := CubicInterpDot(hist, a, b, c, d, x)
		want := cubicInterpDotRef32(hist, a, b, c, d, x)

		// Use relative tolerance (float32 has less precision)
		tol := float32(math.Abs(float64(want)) * 1e-4)
		if math.Abs(float64(got-want)) > float64(tol) {
			t.Errorf("CubicInterpDot(n=%d) = %v, want %v, diff = %v", n, got, want, got-want)
		}
	}
}

func TestCubicInterpDot_DifferentLengths(t *testing.T) {
	// Test with slices of different lengths - should use minimum
	hist := []float32{1, 2, 3, 4, 5}
	a := []float32{1, 1, 1, 1, 1, 1, 1}
	b := []float32{0.5, 0.5, 0.5}
	c := []float32{0.1, 0.1, 0.1, 0.1}
	d := []float32{0.01, 0.01, 0.01, 0.01, 0.01, 0.01}

	// Minimum length is 3 (from b)
	n := 3
	x := float32(0.5)
	got := CubicInterpDot(hist, a, b, c, d, x)
	want := cubicInterpDotRef32(hist[:n], a[:n], b[:n], c[:n], d[:n], x)

	if math.Abs(float64(got-want)) > 1e-6 {
		t.Errorf("CubicInterpDot(different lengths) = %v, want %v", got, want)
	}
}

func TestCubicInterpDotUnsafe(t *testing.T) {
	n := 16
	hist := make([]float32, n)
	a := make([]float32, n)
	b := make([]float32, n)
	c := make([]float32, n)
	d := make([]float32, n)

	for i := range n {
		hist[i] = float32(i+1) * 0.1
		a[i] = float32(i) * 0.5
		b[i] = float32(i) * 0.3
		c[i] = float32(i) * 0.2
		d[i] = float32(i) * 0.1
	}

	x := float32(0.5)
	got := CubicInterpDotUnsafe(hist, a, b, c, d, x)
	want := cubicInterpDotRef32(hist, a, b, c, d, x)

	if math.Abs(float64(got-want)) > 1e-5 {
		t.Errorf("CubicInterpDotUnsafe() = %v, want %v", got, want)
	}
}

func BenchmarkCubicInterpDot_64(b *testing.B) {
	benchmarkCubicInterpDot32(b, 64)
}

func BenchmarkCubicInterpDot_241(b *testing.B) {
	benchmarkCubicInterpDot32(b, 241)
}

func BenchmarkCubicInterpDot_1000(b *testing.B) {
	benchmarkCubicInterpDot32(b, 1000)
}

func benchmarkCubicInterpDot32(b *testing.B, n int) {
	b.Helper()
	hist := make([]float32, n)
	a := make([]float32, n)
	coefB := make([]float32, n)
	c := make([]float32, n)
	d := make([]float32, n)

	for i := range n {
		hist[i] = float32(i+1) * 0.01
		a[i] = float32(i) * 0.1
		coefB[i] = float32(i) * 0.05
		c[i] = float32(i) * 0.02
		d[i] = float32(i) * 0.01
	}

	x := float32(0.75)
	b.ResetTimer()
	b.SetBytes(int64(n * 4 * 5)) // 5 slices of float32

	var result float32
	for i := 0; i < b.N; i++ {
		result = CubicInterpDot(hist, a, coefB, c, d, x)
	}
	_ = result
}

// =============================================================================
// Int32ToFloat32Scale Tests
// =============================================================================

func TestInt32ToFloat32Scale(t *testing.T) {
	testCases := []struct {
		name  string
		src   []int32
		scale float32
	}{
		{"empty", nil, 1.0},
		{"single", []int32{32767}, 1.0 / 32768.0},
		{"four", []int32{-32768, -16384, 0, 32767}, 1.0 / 32768.0},
		{"eight", []int32{-32768, -16384, -8192, 0, 8192, 16384, 32767, -1}, 1.0 / 32768.0},
		{"nine", []int32{-32768, -16384, -8192, 0, 8192, 16384, 32767, -1, 100}, 1.0 / 32768.0},
		{"sixteen", make([]int32, 16), 1.0 / 32768.0},
		{"seventeen", make([]int32, 17), 1.0 / 32768.0},
		{"audio_16bit", []int32{-32768, 0, 32767, -16384, 16384, -8192, 8192, 0}, 1.0 / 32768.0},
		{"audio_32bit", []int32{-2147483648, 0, 2147483647, -1073741824, 1073741824}, 1.0 / 2147483648.0},
	}

	// Initialize test data for sized test cases
	for i := range testCases {
		if testCases[i].name == "sixteen" || testCases[i].name == "seventeen" {
			for j := range testCases[i].src {
				testCases[i].src[j] = int32((j - len(testCases[i].src)/2) * 1000)
			}
		}
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if len(tc.src) == 0 {
				dst := make([]float32, 0)
				Int32ToFloat32Scale(dst, tc.src, tc.scale)
				return
			}

			dst := make([]float32, len(tc.src))
			Int32ToFloat32Scale(dst, tc.src, tc.scale)

			// Verify results
			for i, v := range dst {
				want := float32(tc.src[i]) * tc.scale
				if math.Abs(float64(v-want)) > 1e-6 {
					t.Errorf("Int32ToFloat32Scale()[%d] = %v, want %v", i, v, want)
				}
			}
		})
	}
}

func TestInt32ToFloat32Scale_Large(t *testing.T) {
	// Test with large array that exercises SIMD paths
	sizes := []int{100, 1000, 10000}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			src := make([]int32, size)
			dst := make([]float32, size)
			scale := float32(1.0 / 32768.0)

			// Fill with test data
			for i := range src {
				src[i] = int32((i%65536 - 32768) * 2)
			}

			Int32ToFloat32Scale(dst, src, scale)

			// Verify results
			for i := range dst {
				want := float32(src[i]) * scale
				if math.Abs(float64(dst[i]-want)) > 1e-5 {
					t.Errorf("Int32ToFloat32Scale()[%d] = %v, want %v", i, dst[i], want)
				}
			}
		})
	}
}

func TestInt32ToFloat32Scale_AudioConversion(t *testing.T) {
	// Test realistic audio conversion scenarios

	t.Run("16bit_audio", func(t *testing.T) {
		// 16-bit signed audio: range [-32768, 32767]
		src := []int32{-32768, -16384, -8192, 0, 8192, 16384, 32767}
		dst := make([]float32, len(src))
		scale := float32(1.0 / 32768.0)

		Int32ToFloat32Scale(dst, src, scale)

		// Verify conversion
		expected := []float32{-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.999969482} // 32767/32768
		for i := range dst {
			if math.Abs(float64(dst[i]-expected[i])) > 1e-5 {
				t.Errorf("16bit[%d]: got %v, want %v", i, dst[i], expected[i])
			}
		}
	})

	t.Run("32bit_audio", func(t *testing.T) {
		// 32-bit signed audio: range [-2147483648, 2147483647]
		src := []int32{-2147483648, 0, 2147483647}
		dst := make([]float32, len(src))
		scale := float32(1.0 / 2147483648.0)

		Int32ToFloat32Scale(dst, src, scale)

		// Verify conversion
		if dst[0] != -1.0 {
			t.Errorf("32bit min: got %v, want -1.0", dst[0])
		}
		if dst[1] != 0.0 {
			t.Errorf("32bit zero: got %v, want 0.0", dst[1])
		}
		// 2147483647/2147483648 ≈ 0.999999999534
		if math.Abs(float64(dst[2]-0.9999999995)) > 1e-6 {
			t.Errorf("32bit max: got %v, want ~0.9999999995", dst[2])
		}
	})
}

func TestInt32ToFloat32ScaleUnsafe(t *testing.T) {
	src := []int32{-32768, 0, 32767, -16384, 16384, -8192, 8192, 0}
	dst := make([]float32, len(src))
	scale := float32(1.0 / 32768.0)

	Int32ToFloat32ScaleUnsafe(dst, src, scale)

	for i := range dst {
		want := float32(src[i]) * scale
		if math.Abs(float64(dst[i]-want)) > 1e-6 {
			t.Errorf("Int32ToFloat32ScaleUnsafe()[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

// =============================================================================
// Split-Format Complex Operations Tests
// =============================================================================

// mulComplexRef computes complex multiplication using reference implementation
func mulComplexRef(dstRe, dstIm, aRe, aIm, bRe, bIm []float32) {
	for i := range dstRe {
		dstRe[i] = aRe[i]*bRe[i] - aIm[i]*bIm[i]
		dstIm[i] = aRe[i]*bIm[i] + aIm[i]*bRe[i]
	}
}

// mulConjComplexRef computes multiply-by-conjugate using reference implementation
func mulConjComplexRef(dstRe, dstIm, aRe, aIm, bRe, bIm []float32) {
	for i := range dstRe {
		dstRe[i] = aRe[i]*bRe[i] + aIm[i]*bIm[i]
		dstIm[i] = aIm[i]*bRe[i] - aRe[i]*bIm[i]
	}
}

// absSqComplexRef computes magnitude squared using reference implementation
func absSqComplexRef(dst, aRe, aIm []float32) {
	for i := range dst {
		dst[i] = aRe[i]*aRe[i] + aIm[i]*aIm[i]
	}
}

// complexMulTestCase defines a test case for complex multiplication operations
type complexMulTestCase struct {
	name    string
	fn      func(dstRe, dstIm, aRe, aIm, bRe, bIm []float32)
	emptyFn func()
	ref     func(dstRe, dstIm, aRe, aIm, bRe, bIm []float32)
}

func TestMulComplex(t *testing.T) {
	testComplexMul(t, complexMulTestCase{
		name:    "MulComplex",
		fn:      MulComplex,
		emptyFn: func() { MulComplex(nil, nil, nil, nil, nil, nil) },
		ref:     mulComplexRef,
	})
}

func TestMulConjComplex(t *testing.T) {
	testComplexMul(t, complexMulTestCase{
		name:    "MulConjComplex",
		fn:      MulConjComplex,
		emptyFn: func() { MulConjComplex(nil, nil, nil, nil, nil, nil) },
		ref:     mulConjComplexRef,
	})
}

func testComplexMul(t *testing.T, tc complexMulTestCase) {
	t.Helper()
	sizes := []int{0, 1, 4, 8, 9, 16, 1000}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			if n == 0 {
				tc.emptyFn()
				return
			}

			aRe, aIm := make([]float32, n), make([]float32, n)
			bRe, bIm := make([]float32, n), make([]float32, n)
			dstRe, dstIm := make([]float32, n), make([]float32, n)
			wantRe, wantIm := make([]float32, n), make([]float32, n)

			for i := range n {
				aRe[i], aIm[i] = float32(i+1)*0.1, float32(i+2)*0.2
				bRe[i], bIm[i] = float32(i+3)*0.3, float32(i+4)*0.4
			}

			tc.fn(dstRe, dstIm, aRe, aIm, bRe, bIm)
			tc.ref(wantRe, wantIm, aRe, aIm, bRe, bIm)

			for i := range n {
				reErr := math.Abs(float64(dstRe[i]-wantRe[i])) / (math.Abs(float64(wantRe[i])) + 1e-10)
				imErr := math.Abs(float64(dstIm[i]-wantIm[i])) / (math.Abs(float64(wantIm[i])) + 1e-10)
				if reErr > 1e-6 {
					t.Errorf("Re[%d] = %v, want %v (relErr=%v)", i, dstRe[i], wantRe[i], reErr)
				}
				if imErr > 1e-6 {
					t.Errorf("Im[%d] = %v, want %v (relErr=%v)", i, dstIm[i], wantIm[i], imErr)
				}
			}
		})
	}
}

func TestAbsSqComplex(t *testing.T) {
	testCases := []struct {
		name string
		n    int
	}{
		{"empty", 0},
		{"single", 1},
		{"four", 4},
		{"eight", 8},
		{"nine", 9},
		{"sixteen", 16},
		{"large", 1000},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.n == 0 {
				AbsSqComplex(nil, nil, nil)
				return
			}

			aRe := make([]float32, tc.n)
			aIm := make([]float32, tc.n)
			dst := make([]float32, tc.n)
			want := make([]float32, tc.n)

			for i := range tc.n {
				aRe[i] = float32(i+1) * 0.1
				aIm[i] = float32(i+2) * 0.2
			}

			AbsSqComplex(dst, aRe, aIm)
			absSqComplexRef(want, aRe, aIm)

			for i := range tc.n {
				// Use relative tolerance (1e-6) for FMA vs separate mul+add precision differences
				relErr := math.Abs(float64(dst[i]-want[i])) / (math.Abs(float64(want[i])) + 1e-10)
				if relErr > 1e-6 {
					t.Errorf("AbsSqComplex()[%d] = %v, want %v (relErr=%v)", i, dst[i], want[i], relErr)
				}
			}
		})
	}
}

func TestMulComplex_KnownValues(t *testing.T) {
	// Test with known complex multiplication results
	// (1 + 2i) * (3 + 4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
	aRe := []float32{1}
	aIm := []float32{2}
	bRe := []float32{3}
	bIm := []float32{4}
	dstRe := make([]float32, 1)
	dstIm := make([]float32, 1)

	MulComplex(dstRe, dstIm, aRe, aIm, bRe, bIm)

	if dstRe[0] != -5 {
		t.Errorf("MulComplex() Re = %v, want -5", dstRe[0])
	}
	if dstIm[0] != 10 {
		t.Errorf("MulComplex() Im = %v, want 10", dstIm[0])
	}
}

func TestMulConjComplex_KnownValues(t *testing.T) {
	// Test with known complex conjugate multiplication results
	// (1 + 2i) * conj(3 + 4i) = (1 + 2i) * (3 - 4i)
	// = (1*3 + 2*4) + (2*3 - 1*4)i = 11 + 2i
	aRe := []float32{1}
	aIm := []float32{2}
	bRe := []float32{3}
	bIm := []float32{4}
	dstRe := make([]float32, 1)
	dstIm := make([]float32, 1)

	MulConjComplex(dstRe, dstIm, aRe, aIm, bRe, bIm)

	if dstRe[0] != 11 {
		t.Errorf("MulConjComplex() Re = %v, want 11", dstRe[0])
	}
	if dstIm[0] != 2 {
		t.Errorf("MulConjComplex() Im = %v, want 2", dstIm[0])
	}
}

func TestAbsSqComplex_KnownValues(t *testing.T) {
	// |3 + 4i|^2 = 3^2 + 4^2 = 25
	aRe := []float32{3}
	aIm := []float32{4}
	dst := make([]float32, 1)

	AbsSqComplex(dst, aRe, aIm)

	if dst[0] != 25 {
		t.Errorf("AbsSqComplex() = %v, want 25", dst[0])
	}
}

// =============================================================================
// Split-Format Complex Benchmarks
// =============================================================================

func BenchmarkMulComplex(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}

	for _, n := range sizes {
		b.Run(fmt.Sprintf("SIMD_%d", n), func(b *testing.B) {
			aRe := make([]float32, n)
			aIm := make([]float32, n)
			bRe := make([]float32, n)
			bIm := make([]float32, n)
			dstRe := make([]float32, n)
			dstIm := make([]float32, n)

			for i := range n {
				aRe[i] = float32(i) * 0.1
				aIm[i] = float32(i) * 0.2
				bRe[i] = float32(i) * 0.3
				bIm[i] = float32(i) * 0.4
			}

			b.ResetTimer()
			b.SetBytes(int64(n * 4 * 6)) // 6 slices of float32

			for i := 0; i < b.N; i++ {
				MulComplex(dstRe, dstIm, aRe, aIm, bRe, bIm)
			}
		})

		b.Run(fmt.Sprintf("Go_%d", n), func(b *testing.B) {
			aRe := make([]float32, n)
			aIm := make([]float32, n)
			bRe := make([]float32, n)
			bIm := make([]float32, n)
			dstRe := make([]float32, n)
			dstIm := make([]float32, n)

			for i := range n {
				aRe[i] = float32(i) * 0.1
				aIm[i] = float32(i) * 0.2
				bRe[i] = float32(i) * 0.3
				bIm[i] = float32(i) * 0.4
			}

			b.ResetTimer()
			b.SetBytes(int64(n * 4 * 6))

			for i := 0; i < b.N; i++ {
				mulComplex32Go(dstRe, dstIm, aRe, aIm, bRe, bIm)
			}
		})
	}
}

func BenchmarkAbsSqComplex(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}

	for _, n := range sizes {
		b.Run(fmt.Sprintf("SIMD_%d", n), func(b *testing.B) {
			aRe := make([]float32, n)
			aIm := make([]float32, n)
			dst := make([]float32, n)

			for i := range n {
				aRe[i] = float32(i) * 0.1
				aIm[i] = float32(i) * 0.2
			}

			b.ResetTimer()
			b.SetBytes(int64(n * 4 * 3)) // 3 slices of float32

			for i := 0; i < b.N; i++ {
				AbsSqComplex(dst, aRe, aIm)
			}
		})

		b.Run(fmt.Sprintf("Go_%d", n), func(b *testing.B) {
			aRe := make([]float32, n)
			aIm := make([]float32, n)
			dst := make([]float32, n)

			for i := range n {
				aRe[i] = float32(i) * 0.1
				aIm[i] = float32(i) * 0.2
			}

			b.ResetTimer()
			b.SetBytes(int64(n * 4 * 3))

			for i := 0; i < b.N; i++ {
				absSqComplex32Go(dst, aRe, aIm)
			}
		})
	}
}

// =============================================================================
// FFT BUTTERFLY SIMULATION TESTS
// These tests specifically target bugs that manifest in FFT butterfly operations
// =============================================================================

// complexMulSIMDvsGoHelper tests SIMD vs Go implementations for complex multiplication.
func complexMulSIMDvsGoHelper(t *testing.T, sizes []int,
	simdFn func(dstRe, dstIm, aRe, aIm, bRe, bIm []float32),
	goFn func(dstRe, dstIm, aRe, aIm, bRe, bIm []float32),
) {
	t.Helper()
	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			aRe, aIm := make([]float32, n), make([]float32, n)
			bRe, bIm := make([]float32, n), make([]float32, n)

			for i := range n {
				angle := 2 * math.Pi * float64(i) / float64(n)
				aRe[i] = float32(math.Cos(angle))
				aIm[i] = float32(math.Sin(angle))
				bRe[i] = float32(math.Cos(2 * angle))
				bIm[i] = float32(math.Sin(2 * angle))
			}

			simdRe, simdIm := make([]float32, n), make([]float32, n)
			simdFn(simdRe, simdIm, aRe, aIm, bRe, bIm)

			goRe, goIm := make([]float32, n), make([]float32, n)
			goFn(goRe, goIm, aRe, aIm, bRe, bIm)

			for i := range n {
				reErr := math.Abs(float64(simdRe[i] - goRe[i]))
				imErr := math.Abs(float64(simdIm[i] - goIm[i]))
				if reErr > 1e-6 {
					t.Errorf("Re[%d]: SIMD=%v, Go=%v, diff=%v", i, simdRe[i], goRe[i], reErr)
				}
				if imErr > 1e-6 {
					t.Errorf("Im[%d]: SIMD=%v, Go=%v, diff=%v", i, simdIm[i], goIm[i], imErr)
				}
			}
		})
	}
}

// TestMulComplex_SIMDvsGo explicitly compares SIMD and pure Go implementations
// to identify bugs that only appear in the SIMD path.
func TestMulComplex_SIMDvsGo(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 63, 64, 65, 128, 256, 512, 1024}
	complexMulSIMDvsGoHelper(t, sizes, MulComplex, mulComplex32Go)
}

// TestMulComplex_FFTButterflyPattern simulates the exact access pattern used in
// FFT butterfly operations to identify bugs in strided twiddle factor application.
func TestMulComplex_FFTButterflyPattern(t *testing.T) {
	// Simulate FFT sizes
	fftSizes := []int{8, 16, 32, 64, 128, 256, 512, 1024}

	for _, fftSize := range fftSizes {
		t.Run(fmt.Sprintf("fft=%d", fftSize), func(t *testing.T) {
			// Precompute twiddle factors like an FFT would
			twRe := make([]float32, fftSize/2)
			twIm := make([]float32, fftSize/2)
			for k := range fftSize / 2 {
				angle := -2 * math.Pi * float64(k) / float64(fftSize)
				twRe[k] = float32(math.Cos(angle))
				twIm[k] = float32(math.Sin(angle))
			}

			// Simulate butterfly at different stages
			for stage := 0; (1 << stage) < fftSize; stage++ {
				halfSize := 1 << stage
				fullSize := halfSize * 2
				twStep := fftSize / fullSize

				t.Run(fmt.Sprintf("stage=%d", stage), func(t *testing.T) {
					// Create sample data and butterfly data
					dataRe := make([]float32, halfSize)
					dataIm := make([]float32, halfSize)

					// Gather twiddle factors with stride (this is where bugs can occur)
					gatheredTwRe := make([]float32, halfSize)
					gatheredTwIm := make([]float32, halfSize)

					for k := range halfSize {
						twIdx := k * twStep
						if twIdx >= len(twRe) {
							t.Fatalf("twiddle index %d out of bounds (len=%d) at k=%d, twStep=%d",
								twIdx, len(twRe), k, twStep)
						}
						gatheredTwRe[k] = twRe[twIdx]
						gatheredTwIm[k] = twIm[twIdx]

						// Sample data (use twiddle-like values)
						angle := 2 * math.Pi * float64(k) / float64(halfSize)
						dataRe[k] = float32(math.Cos(angle))
						dataIm[k] = float32(math.Sin(angle))
					}

					// Perform complex multiplication (butterfly core operation)
					simdRe := make([]float32, halfSize)
					simdIm := make([]float32, halfSize)
					MulComplex(simdRe, simdIm, dataRe, dataIm, gatheredTwRe, gatheredTwIm)

					// Compute reference
					refRe := make([]float32, halfSize)
					refIm := make([]float32, halfSize)
					mulComplex32Go(refRe, refIm, dataRe, dataIm, gatheredTwRe, gatheredTwIm)

					// Compare
					for k := range halfSize {
						reErr := math.Abs(float64(simdRe[k] - refRe[k]))
						imErr := math.Abs(float64(simdIm[k] - refIm[k]))
						if reErr > 1e-5 {
							t.Errorf("stage=%d k=%d Re: SIMD=%v, ref=%v, diff=%v",
								stage, k, simdRe[k], refRe[k], reErr)
						}
						if imErr > 1e-5 {
							t.Errorf("stage=%d k=%d Im: SIMD=%v, ref=%v, diff=%v",
								stage, k, simdIm[k], refIm[k], imErr)
						}
					}
				})
			}
		})
	}
}

// TestMulComplex_InPlaceLike tests patterns where inputs and outputs may overlap
// or share underlying memory, which can reveal aliasing bugs.
func TestMulComplex_InPlaceLike(t *testing.T) {
	sizes := []int{8, 16, 32, 64}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			// Create a larger buffer and use overlapping slices
			bufRe := make([]float32, n*2)
			bufIm := make([]float32, n*2)

			// Initialize with test data
			for i := range len(bufRe) {
				bufRe[i] = float32(i + 1)
				bufIm[i] = float32(i + 2)
			}

			// Use first half as input A, second half as input B, first half as output
			aRe := bufRe[:n]
			aIm := bufIm[:n]
			bRe := bufRe[n:]
			bIm := bufIm[n:]

			// Save copies for reference computation
			aReCopy := make([]float32, n)
			aImCopy := make([]float32, n)
			copy(aReCopy, aRe)
			copy(aImCopy, aIm)

			// Compute reference first (before any overwrites)
			refRe := make([]float32, n)
			refIm := make([]float32, n)
			mulComplex32Go(refRe, refIm, aReCopy, aImCopy, bRe, bIm)

			// Compute with SIMD (output overwrites input A)
			MulComplex(aRe, aIm, aReCopy, aImCopy, bRe, bIm)

			// Compare
			for i := range n {
				if math.Abs(float64(aRe[i]-refRe[i])) > 1e-5 {
					t.Errorf("Re[%d]: got=%v, want=%v", i, aRe[i], refRe[i])
				}
				if math.Abs(float64(aIm[i]-refIm[i])) > 1e-5 {
					t.Errorf("Im[%d]: got=%v, want=%v", i, aIm[i], refIm[i])
				}
			}
		})
	}
}

// TestMulComplex_ZeroAndExtreme tests edge cases with zeros and extreme values
// that might reveal numerical issues in SIMD implementations.
func TestMulComplex_ZeroAndExtreme(t *testing.T) {
	testCases := []struct {
		name string
		aRe  []float32
		aIm  []float32
		bRe  []float32
		bIm  []float32
	}{
		{
			"zeros_times_nonzero",
			[]float32{0, 0, 0, 0, 0, 0, 0, 0},
			[]float32{0, 0, 0, 0, 0, 0, 0, 0},
			[]float32{1, 2, 3, 4, 5, 6, 7, 8},
			[]float32{8, 7, 6, 5, 4, 3, 2, 1},
		},
		{
			"nonzero_times_zeros",
			[]float32{1, 2, 3, 4, 5, 6, 7, 8},
			[]float32{8, 7, 6, 5, 4, 3, 2, 1},
			[]float32{0, 0, 0, 0, 0, 0, 0, 0},
			[]float32{0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			"identity_real",
			[]float32{1, 2, 3, 4, 5, 6, 7, 8},
			[]float32{0, 0, 0, 0, 0, 0, 0, 0},
			[]float32{1, 1, 1, 1, 1, 1, 1, 1},
			[]float32{0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			"identity_imag",
			[]float32{0, 0, 0, 0, 0, 0, 0, 0},
			[]float32{1, 2, 3, 4, 5, 6, 7, 8},
			[]float32{0, 0, 0, 0, 0, 0, 0, 0},
			[]float32{1, 1, 1, 1, 1, 1, 1, 1},
		},
		{
			"alternating_signs",
			[]float32{1, -1, 1, -1, 1, -1, 1, -1},
			[]float32{-1, 1, -1, 1, -1, 1, -1, 1},
			[]float32{1, 1, 1, 1, 1, 1, 1, 1},
			[]float32{1, 1, 1, 1, 1, 1, 1, 1},
		},
		{
			"unit_circle_quadrants",
			[]float32{1, 0, -1, 0, 1, 0, -1, 0},
			[]float32{0, 1, 0, -1, 0, 1, 0, -1},
			[]float32{0, 1, 0, -1, 0, 1, 0, -1},
			[]float32{1, 0, -1, 0, 1, 0, -1, 0},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			n := len(tc.aRe)

			simdRe := make([]float32, n)
			simdIm := make([]float32, n)
			MulComplex(simdRe, simdIm, tc.aRe, tc.aIm, tc.bRe, tc.bIm)

			refRe := make([]float32, n)
			refIm := make([]float32, n)
			mulComplex32Go(refRe, refIm, tc.aRe, tc.aIm, tc.bRe, tc.bIm)

			for i := range n {
				if math.Abs(float64(simdRe[i]-refRe[i])) > 1e-6 {
					t.Errorf("Re[%d]: SIMD=%v, Go=%v", i, simdRe[i], refRe[i])
				}
				if math.Abs(float64(simdIm[i]-refIm[i])) > 1e-6 {
					t.Errorf("Im[%d]: SIMD=%v, Go=%v", i, simdIm[i], refIm[i])
				}
			}
		})
	}
}

// TestMulComplex_TwiddleGathering tests the exact pattern used when gathering
// twiddle factors with a stride, which is where the FFT butterfly bug occurred.
func TestMulComplex_TwiddleGathering(t *testing.T) {
	// This test simulates what happens in butterflyStageVectorized:
	// for k := 0; k < halfSize; k++ {
	//     twBufRe[k] = twRe[k*twStep]
	//     twBufIm[k] = twIm[k*twStep]
	// }

	n := 256 // Full FFT size
	twRe := make([]float32, n/2)
	twIm := make([]float32, n/2)

	// Initialize twiddle factors
	for k := range n / 2 {
		angle := -2 * math.Pi * float64(k) / float64(n)
		twRe[k] = float32(math.Cos(angle))
		twIm[k] = float32(math.Sin(angle))
	}

	// Test at different stages with different twStep values
	stages := []struct {
		halfSize int
		twStep   int
	}{
		{1, 128}, // stage 0: twStep = 256/2 = 128
		{2, 64},  // stage 1: twStep = 256/4 = 64
		{4, 32},  // stage 2: twStep = 256/8 = 32
		{8, 16},  // stage 3: twStep = 256/16 = 16
		{16, 8},  // stage 4: twStep = 256/32 = 8
		{32, 4},  // stage 5: twStep = 256/64 = 4
		{64, 2},  // stage 6: twStep = 256/128 = 2
		{128, 1}, // stage 7: twStep = 256/256 = 1
	}

	for _, s := range stages {
		t.Run(fmt.Sprintf("halfSize=%d_twStep=%d", s.halfSize, s.twStep), func(t *testing.T) {
			// Gather twiddle factors
			gatheredRe := make([]float32, s.halfSize)
			gatheredIm := make([]float32, s.halfSize)
			for k := range s.halfSize {
				gatheredRe[k] = twRe[k*s.twStep]
				gatheredIm[k] = twIm[k*s.twStep]
			}

			// Create data to multiply
			dataRe := make([]float32, s.halfSize)
			dataIm := make([]float32, s.halfSize)
			for k := range s.halfSize {
				dataRe[k] = float32(k + 1)
				dataIm[k] = float32(k + 2)
			}

			// SIMD computation
			simdRe := make([]float32, s.halfSize)
			simdIm := make([]float32, s.halfSize)
			MulComplex(simdRe, simdIm, dataRe, dataIm, gatheredRe, gatheredIm)

			// Go reference computation
			refRe := make([]float32, s.halfSize)
			refIm := make([]float32, s.halfSize)
			mulComplex32Go(refRe, refIm, dataRe, dataIm, gatheredRe, gatheredIm)

			// Compare - use relative tolerance for larger values
			for k := range s.halfSize {
				reErr := math.Abs(float64(simdRe[k] - refRe[k]))
				imErr := math.Abs(float64(simdIm[k] - refIm[k]))
				// Use relative tolerance for larger values, absolute for small values
				reTol := max(1e-5, math.Abs(float64(refRe[k]))*1e-5)
				imTol := max(1e-5, math.Abs(float64(refIm[k]))*1e-5)
				if reErr > reTol {
					t.Errorf("k=%d Re: SIMD=%v, ref=%v, diff=%v", k, simdRe[k], refRe[k], reErr)
				}
				if imErr > imTol {
					t.Errorf("k=%d Im: SIMD=%v, ref=%v, diff=%v", k, simdIm[k], refIm[k], imErr)
				}
			}
		})
	}
}

// TestAbsSqComplex_SIMDvsGo explicitly compares SIMD and pure Go implementations.
func TestAbsSqComplex_SIMDvsGo(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 256}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			aRe := make([]float32, n)
			aIm := make([]float32, n)

			for i := range n {
				angle := 2 * math.Pi * float64(i) / float64(n)
				aRe[i] = float32(math.Cos(angle))
				aIm[i] = float32(math.Sin(angle))
			}

			simdDst := make([]float32, n)
			AbsSqComplex(simdDst, aRe, aIm)

			goDst := make([]float32, n)
			absSqComplex32Go(goDst, aRe, aIm)

			for i := range n {
				err := math.Abs(float64(simdDst[i] - goDst[i]))
				if err > 1e-6 {
					t.Errorf("[%d]: SIMD=%v, Go=%v, diff=%v", i, simdDst[i], goDst[i], err)
				}
			}
		})
	}
}

// TestMulConjComplex_SIMDvsGo explicitly compares SIMD and pure Go implementations.
func TestMulConjComplex_SIMDvsGo(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 256}
	complexMulSIMDvsGoHelper(t, sizes, MulConjComplex, mulConjComplex32Go)
}

// fftButterflyStages performs FFT butterfly stages using the provided complex multiplication function.
func fftButterflyStages(dataRe, dataIm, twRe, twIm []float32, n int,
	mulFn func(dstRe, dstIm, aRe, aIm, bRe, bIm []float32),
) {
	for stage := 0; (1 << stage) < n; stage++ {
		halfSize := 1 << stage
		fullSize := halfSize * 2
		twStep := n / fullSize

		for group := 0; group < n; group += fullSize {
			gatheredRe := make([]float32, halfSize)
			gatheredIm := make([]float32, halfSize)
			for k := range halfSize {
				gatheredRe[k] = twRe[k*twStep]
				gatheredIm[k] = twIm[k*twStep]
			}

			topRe := dataRe[group : group+halfSize]
			topIm := dataIm[group : group+halfSize]
			botRe := dataRe[group+halfSize : group+fullSize]
			botIm := dataIm[group+halfSize : group+fullSize]

			tempRe := make([]float32, halfSize)
			tempIm := make([]float32, halfSize)
			mulFn(tempRe, tempIm, botRe, botIm, gatheredRe, gatheredIm)

			for k := range halfSize {
				tr, ti := topRe[k], topIm[k]
				topRe[k] = tr + tempRe[k]
				topIm[k] = ti + tempIm[k]
				botRe[k] = tr - tempRe[k]
				botIm[k] = ti - tempIm[k]
			}
		}
	}
}

// TestMulComplex_FullFFTSimulation performs a complete FFT-like computation
// to ensure SIMD operations work correctly in the full butterfly context.
func TestMulComplex_FullFFTSimulation(t *testing.T) {
	for _, n := range []int{8, 16, 32, 64, 128, 256} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			dataRe, dataIm := make([]float32, n), make([]float32, n)
			dataRe[0] = 1.0 // Impulse - FFT should give all 1s

			twRe, twIm := make([]float32, n/2), make([]float32, n/2)
			for k := range n / 2 {
				angle := -2 * math.Pi * float64(k) / float64(n)
				twRe[k] = float32(math.Cos(angle))
				twIm[k] = float32(math.Sin(angle))
			}

			dataReGo, dataImGo := make([]float32, n), make([]float32, n)
			copy(dataReGo, dataRe)
			copy(dataImGo, dataIm)

			fftButterflyStages(dataRe, dataIm, twRe, twIm, n, MulComplex)
			fftButterflyStages(dataReGo, dataImGo, twRe, twIm, n, mulComplex32Go)

			for i := range n {
				reErr := math.Abs(float64(dataRe[i] - dataReGo[i]))
				imErr := math.Abs(float64(dataIm[i] - dataImGo[i]))
				if reErr > 1e-4 {
					t.Errorf("bin %d Re: SIMD=%v, Go=%v, diff=%v", i, dataRe[i], dataReGo[i], reErr)
				}
				if imErr > 1e-4 {
					t.Errorf("bin %d Im: SIMD=%v, Go=%v, diff=%v", i, dataIm[i], dataImGo[i], imErr)
				}
			}

			for i := range n {
				if math.Abs(float64(dataRe[i]-1.0)) > 1e-4 {
					t.Errorf("FFT[%d] Re = %v, want 1.0 (impulse response)", i, dataRe[i])
				}
				if math.Abs(float64(dataIm[i])) > 1e-4 {
					t.Errorf("FFT[%d] Im = %v, want 0.0 (impulse response)", i, dataIm[i])
				}
			}
		})
	}
}

// =============================================================================
// BUTTERFLY COMPLEX TESTS
// =============================================================================

// butterflyComplexRef is a reference implementation of the fused butterfly operation.
func butterflyComplexRef(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm []float32) {
	for i := range upperRe {
		// Complex multiply: temp = lower * twiddle
		lr, li := lowerRe[i], lowerIm[i]
		tr, ti := twRe[i], twIm[i]
		tempRe := lr*tr - li*ti
		tempIm := lr*ti + li*tr

		// Butterfly: upper' = upper + temp, lower' = upper - temp
		ur, ui := upperRe[i], upperIm[i]
		upperRe[i] = ur + tempRe
		upperIm[i] = ui + tempIm
		lowerRe[i] = ur - tempRe
		lowerIm[i] = ui - tempIm
	}
}

func TestButterflyComplex(t *testing.T) {
	sizes := []int{0, 1, 4, 8, 9, 16, 17, 31, 32, 33, 63, 64, 65, 128, 256, 512, 1000}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			if n == 0 {
				ButterflyComplex(nil, nil, nil, nil, nil, nil)
				return
			}

			upperRe := make([]float32, n)
			upperIm := make([]float32, n)
			lowerRe := make([]float32, n)
			lowerIm := make([]float32, n)
			twRe := make([]float32, n)
			twIm := make([]float32, n)

			// Reference copies
			upperReRef := make([]float32, n)
			upperImRef := make([]float32, n)
			lowerReRef := make([]float32, n)
			lowerImRef := make([]float32, n)

			// Initialize with test data
			for i := range n {
				angle := 2 * math.Pi * float64(i) / float64(n)
				upperRe[i] = float32(i+1) * 0.1
				upperIm[i] = float32(i+2) * 0.2
				lowerRe[i] = float32(i+3) * 0.3
				lowerIm[i] = float32(i+4) * 0.4
				twRe[i] = float32(math.Cos(angle))
				twIm[i] = float32(math.Sin(angle))
			}

			// Copy for reference
			copy(upperReRef, upperRe)
			copy(upperImRef, upperIm)
			copy(lowerReRef, lowerRe)
			copy(lowerImRef, lowerIm)

			// Apply SIMD version
			ButterflyComplex(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm)

			// Apply reference
			butterflyComplexRef(upperReRef, upperImRef, lowerReRef, lowerImRef, twRe, twIm)

			// Compare results - use relative tolerance for FMA vs separate mul+add precision differences
			for i := range n {
				relTol := func(got, want float32) bool {
					diff := math.Abs(float64(got - want))
					return diff <= 2e-4 || diff <= math.Abs(float64(want))*1e-5
				}
				if !relTol(upperRe[i], upperReRef[i]) {
					t.Errorf("upperRe[%d] = %v, want %v", i, upperRe[i], upperReRef[i])
				}
				if !relTol(upperIm[i], upperImRef[i]) {
					t.Errorf("upperIm[%d] = %v, want %v", i, upperIm[i], upperImRef[i])
				}
				if !relTol(lowerRe[i], lowerReRef[i]) {
					t.Errorf("lowerRe[%d] = %v, want %v", i, lowerRe[i], lowerReRef[i])
				}
				if !relTol(lowerIm[i], lowerImRef[i]) {
					t.Errorf("lowerIm[%d] = %v, want %v", i, lowerIm[i], lowerImRef[i])
				}
			}
		})
	}
}

func TestButterflyComplex_KnownValues(t *testing.T) {
	// Test with known values:
	// upper = 1+2i, lower = 3+4i, twiddle = cos(π/4)+i*sin(π/4) ≈ 0.707+0.707i
	// temp = lower * twiddle = (3+4i) * (0.707+0.707i)
	//      = (3*0.707 - 4*0.707) + (3*0.707 + 4*0.707)i
	//      ≈ -0.707 + 4.95i
	// upper' = upper + temp = (1-0.707) + (2+4.95)i ≈ 0.293 + 6.95i
	// lower' = upper - temp = (1+0.707) + (2-4.95)i ≈ 1.707 - 2.95i
	cos45 := float32(math.Cos(math.Pi / 4))
	sin45 := float32(math.Sin(math.Pi / 4))

	upperRe := []float32{1}
	upperIm := []float32{2}
	lowerRe := []float32{3}
	lowerIm := []float32{4}
	twRe := []float32{cos45}
	twIm := []float32{sin45}

	ButterflyComplex(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm)

	// Expected values computed manually
	tempRe := 3*cos45 - 4*sin45 // ≈ -0.707
	tempIm := 3*sin45 + 4*cos45 // ≈ 4.95
	wantUpperRe := 1 + tempRe   // ≈ 0.293
	wantUpperIm := 2 + tempIm   // ≈ 6.95
	wantLowerRe := 1 - tempRe   // ≈ 1.707
	wantLowerIm := 2 - tempIm   // ≈ -2.95

	if math.Abs(float64(upperRe[0]-wantUpperRe)) > 1e-5 {
		t.Errorf("upperRe = %v, want %v", upperRe[0], wantUpperRe)
	}
	if math.Abs(float64(upperIm[0]-wantUpperIm)) > 1e-5 {
		t.Errorf("upperIm = %v, want %v", upperIm[0], wantUpperIm)
	}
	if math.Abs(float64(lowerRe[0]-wantLowerRe)) > 1e-5 {
		t.Errorf("lowerRe = %v, want %v", lowerRe[0], wantLowerRe)
	}
	if math.Abs(float64(lowerIm[0]-wantLowerIm)) > 1e-5 {
		t.Errorf("lowerIm = %v, want %v", lowerIm[0], wantLowerIm)
	}
}

func TestButterflyComplex_SIMDvsGo(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 256, 512, 1024}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			upperRe := make([]float32, n)
			upperIm := make([]float32, n)
			lowerRe := make([]float32, n)
			lowerIm := make([]float32, n)
			twRe := make([]float32, n)
			twIm := make([]float32, n)

			// Go copies
			upperReGo := make([]float32, n)
			upperImGo := make([]float32, n)
			lowerReGo := make([]float32, n)
			lowerImGo := make([]float32, n)

			// Initialize with varied data to catch bugs
			for i := range n {
				angle := 2 * math.Pi * float64(i) / float64(n)
				upperRe[i] = float32(math.Sin(float64(i) * 0.123))
				upperIm[i] = float32(math.Cos(float64(i) * 0.456))
				lowerRe[i] = float32(math.Sin(float64(i) * 0.789))
				lowerIm[i] = float32(math.Cos(float64(i) * 0.012))
				twRe[i] = float32(math.Cos(angle))
				twIm[i] = float32(math.Sin(angle))
			}

			copy(upperReGo, upperRe)
			copy(upperImGo, upperIm)
			copy(lowerReGo, lowerRe)
			copy(lowerImGo, lowerIm)

			// SIMD version
			ButterflyComplex(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm)

			// Pure Go version
			butterflyComplex32Go(upperReGo, upperImGo, lowerReGo, lowerImGo, twRe, twIm)

			// Compare
			for i := range n {
				if math.Abs(float64(upperRe[i]-upperReGo[i])) > 1e-5 {
					t.Errorf("upperRe[%d]: SIMD=%v, Go=%v", i, upperRe[i], upperReGo[i])
				}
				if math.Abs(float64(upperIm[i]-upperImGo[i])) > 1e-5 {
					t.Errorf("upperIm[%d]: SIMD=%v, Go=%v", i, upperIm[i], upperImGo[i])
				}
				if math.Abs(float64(lowerRe[i]-lowerReGo[i])) > 1e-5 {
					t.Errorf("lowerRe[%d]: SIMD=%v, Go=%v", i, lowerRe[i], lowerReGo[i])
				}
				if math.Abs(float64(lowerIm[i]-lowerImGo[i])) > 1e-5 {
					t.Errorf("lowerIm[%d]: SIMD=%v, Go=%v", i, lowerIm[i], lowerImGo[i])
				}
			}
		})
	}
}

func TestButterflyComplex_FFTPattern(t *testing.T) {
	// Test exact FFT butterfly pattern at different stages
	fftSizes := []int{8, 16, 32, 64, 128, 256, 512, 1024}

	for _, fftSize := range fftSizes {
		t.Run(fmt.Sprintf("fft=%d", fftSize), func(t *testing.T) {
			// Precompute twiddle factors
			twRe := make([]float32, fftSize/2)
			twIm := make([]float32, fftSize/2)
			for k := range fftSize / 2 {
				angle := -2 * math.Pi * float64(k) / float64(fftSize)
				twRe[k] = float32(math.Cos(angle))
				twIm[k] = float32(math.Sin(angle))
			}

			// Test at different stages
			for stage := 0; (1 << stage) < fftSize; stage++ {
				halfSize := 1 << stage
				fullSize := halfSize * 2
				twStep := fftSize / fullSize

				t.Run(fmt.Sprintf("stage=%d", stage), func(t *testing.T) {
					// Create test data
					upperRe := make([]float32, halfSize)
					upperIm := make([]float32, halfSize)
					lowerRe := make([]float32, halfSize)
					lowerIm := make([]float32, halfSize)
					gatheredTwRe := make([]float32, halfSize)
					gatheredTwIm := make([]float32, halfSize)

					// Copy data for reference
					upperReRef := make([]float32, halfSize)
					upperImRef := make([]float32, halfSize)
					lowerReRef := make([]float32, halfSize)
					lowerImRef := make([]float32, halfSize)

					for k := range halfSize {
						angle := 2 * math.Pi * float64(k) / float64(halfSize)
						upperRe[k] = float32(math.Cos(angle))
						upperIm[k] = float32(math.Sin(angle))
						lowerRe[k] = float32(math.Sin(angle * 2))
						lowerIm[k] = float32(math.Cos(angle * 2))
						gatheredTwRe[k] = twRe[k*twStep]
						gatheredTwIm[k] = twIm[k*twStep]
					}

					copy(upperReRef, upperRe)
					copy(upperImRef, upperIm)
					copy(lowerReRef, lowerRe)
					copy(lowerImRef, lowerIm)

					// Apply SIMD butterfly
					ButterflyComplex(upperRe, upperIm, lowerRe, lowerIm, gatheredTwRe, gatheredTwIm)

					// Apply reference butterfly
					butterflyComplex32Go(upperReRef, upperImRef, lowerReRef, lowerImRef, gatheredTwRe, gatheredTwIm)

					// Compare
					for k := range halfSize {
						if math.Abs(float64(upperRe[k]-upperReRef[k])) > 1e-5 {
							t.Errorf("stage=%d k=%d upperRe: SIMD=%v, ref=%v", stage, k, upperRe[k], upperReRef[k])
						}
						if math.Abs(float64(upperIm[k]-upperImRef[k])) > 1e-5 {
							t.Errorf("stage=%d k=%d upperIm: SIMD=%v, ref=%v", stage, k, upperIm[k], upperImRef[k])
						}
						if math.Abs(float64(lowerRe[k]-lowerReRef[k])) > 1e-5 {
							t.Errorf("stage=%d k=%d lowerRe: SIMD=%v, ref=%v", stage, k, lowerRe[k], lowerReRef[k])
						}
						if math.Abs(float64(lowerIm[k]-lowerImRef[k])) > 1e-5 {
							t.Errorf("stage=%d k=%d lowerIm: SIMD=%v, ref=%v", stage, k, lowerIm[k], lowerImRef[k])
						}
					}
				})
			}
		})
	}
}

func BenchmarkButterflyComplex(b *testing.B) {
	sizes := []int{64, 128, 256, 512, 1024, 4096}

	benchFn := func(b *testing.B, n int, fn func(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm []float32)) {
		b.Helper()
		upperRe := make([]float32, n)
		upperIm := make([]float32, n)
		lowerRe := make([]float32, n)
		lowerIm := make([]float32, n)
		twRe := make([]float32, n)
		twIm := make([]float32, n)

		for i := range n {
			angle := 2 * math.Pi * float64(i) / float64(n)
			upperRe[i] = float32(i) * 0.1
			upperIm[i] = float32(i) * 0.2
			lowerRe[i] = float32(i) * 0.3
			lowerIm[i] = float32(i) * 0.4
			twRe[i] = float32(math.Cos(angle))
			twIm[i] = float32(math.Sin(angle))
		}

		b.ResetTimer()
		b.SetBytes(int64(n * 4 * 6)) // 6 slices of float32

		for i := 0; i < b.N; i++ {
			fn(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm)
		}
	}

	for _, n := range sizes {
		b.Run(fmt.Sprintf("SIMD_%d", n), func(b *testing.B) {
			benchFn(b, n, ButterflyComplex)
		})
		b.Run(fmt.Sprintf("Go_%d", n), func(b *testing.B) {
			benchFn(b, n, butterflyComplex32Go)
		})
	}
}
