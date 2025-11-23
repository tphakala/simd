package f32

import (
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
		{"2d", []float32{0, 0}, []float32{3, 4}, 5},                       // 3-4-5 triangle
		{"3d", []float32{1, 2, 3}, []float32{4, 6, 8}, float32(math.Sqrt(50))}, // sqrt(9+16+25)
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
