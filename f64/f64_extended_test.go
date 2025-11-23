package f64

import (
	"math"
	"testing"
)

// TestSqrt validates Sqrt operation
func TestSqrt(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want []float64
	}{
		{"empty", nil, nil},
		{"single", []float64{4}, []float64{2}},
		{"perfect_squares", []float64{0, 1, 4, 9, 16, 25}, []float64{0, 1, 2, 3, 4, 5}},
		{"decimals", []float64{0.25, 0.5, 2, 8}, []float64{0.5, 0.7071067811865476, 1.4142135623730951, 2.8284271247461903}},
		{"large", []float64{1e6, 1e8, 1e10}, []float64{1000, 10000, 100000}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			Sqrt(dst, tt.a)
			for i := range dst {
				if i < len(tt.want) && !almostEqual(dst[i], tt.want[i], 1e-14) {
					t.Errorf("Sqrt()[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

// TestReciprocal validates Reciprocal operation
func TestReciprocal(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want []float64
	}{
		{"simple", []float64{1, 2, 4, 5, 10}, []float64{1, 0.5, 0.25, 0.2, 0.1}},
		{"negative", []float64{-1, -2, -4}, []float64{-1, -0.5, -0.25}},
		{"decimals", []float64{0.5, 0.25, 0.1}, []float64{2, 4, 10}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			Reciprocal(dst, tt.a)
			for i := range dst {
				if i < len(tt.want) && !almostEqual(dst[i], tt.want[i], 1e-14) {
					t.Errorf("Reciprocal()[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

// TestMean validates Mean operation
func TestMean(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want float64
	}{
		{"empty", nil, 0},
		{"single", []float64{5}, 5},
		{"simple", []float64{1, 2, 3, 4, 5}, 3},
		{"negative", []float64{-5, -3, -1, 1, 3, 5}, 0},
		{"decimals", []float64{1.5, 2.5, 3.5}, 2.5},
		{"large", make100(), 50.5}, // 1 to 100, mean = 50.5
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Mean(tt.a)
			if !almostEqual(got, tt.want, 1e-14) {
				t.Errorf("Mean() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestVariance validates Variance operation
func TestVariance(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want float64
	}{
		{"empty", nil, 0},
		{"single", []float64{5}, 0},
		{"constant", []float64{3, 3, 3, 3}, 0},
		{"simple", []float64{1, 2, 3, 4, 5}, 2}, // Population variance
		{"binary", []float64{0, 0, 1, 1}, 0.25},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Variance(tt.a)
			if !almostEqual(got, tt.want, 1e-14) {
				t.Errorf("Variance() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestStdDev validates StdDev operation
func TestStdDev(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want float64
	}{
		{"empty", nil, 0},
		{"single", []float64{5}, 0},
		{"constant", []float64{3, 3, 3, 3}, 0},
		{"simple", []float64{1, 2, 3, 4, 5}, math.Sqrt(2)}, // sqrt(variance)
		{"binary", []float64{0, 0, 1, 1}, 0.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := StdDev(tt.a)
			if !almostEqual(got, tt.want, 1e-14) {
				t.Errorf("StdDev() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestEuclideanDistance validates EuclideanDistance operation
func TestEuclideanDistance(t *testing.T) {
	tests := []struct {
		name string
		a, b []float64
		want float64
	}{
		{"empty", nil, nil, 0},
		{"same", []float64{1, 2, 3}, []float64{1, 2, 3}, 0},
		{"2d", []float64{0, 0}, []float64{3, 4}, 5},                     // 3-4-5 triangle
		{"3d", []float64{1, 2, 3}, []float64{4, 6, 8}, math.Sqrt(50)},   // sqrt(9+16+25)
		{"negative", []float64{-1, -2}, []float64{1, 2}, math.Sqrt(20)}, // sqrt(4+16)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EuclideanDistance(tt.a, tt.b)
			if !almostEqual(got, tt.want, 1e-14) {
				t.Errorf("EuclideanDistance() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestNormalize validates Normalize operation
func TestNormalize(t *testing.T) {
	tests := []struct {
		name      string
		a         []float64
		wantNorm  float64 // Expected norm of result (should be 1 for non-zero vectors)
		checkZero bool    // If true, check that output equals input (for zero vectors)
	}{
		{"empty", nil, 0, false},
		{"unit", []float64{1, 0, 0}, 1, false},
		{"2d", []float64{3, 4}, 1, false},    // Will become {0.6, 0.8}
		{"3d", []float64{2, 2, 1}, 1, false}, // Norm = 3, normalized = {2/3, 2/3, 1/3}
		{"negative", []float64{-1, 0, 1}, 1, false},
		{"zero", []float64{0, 0, 0}, 0, true},      // Should remain zero
		{"tiny", []float64{1e-15, 1e-15}, 0, true}, // Below threshold, should remain unchanged
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			Normalize(dst, tt.a)

			if tt.checkZero {
				// For zero/tiny vectors, output should equal input
				for i := range dst {
					if i < len(tt.a) && dst[i] != tt.a[i] {
						t.Errorf("Normalize()[%d] = %v, want %v (zero vector)", i, dst[i], tt.a[i])
					}
				}
			} else if len(dst) > 0 {
				// Calculate norm of result
				norm := 0.0
				for _, v := range dst {
					norm += v * v
				}
				norm = math.Sqrt(norm)

				if !almostEqual(norm, tt.wantNorm, 1e-10) {
					t.Errorf("Normalize() norm = %v, want %v", norm, tt.wantNorm)
				}
			}
		})
	}
}

// TestCumulativeSum validates CumulativeSum operation
func TestCumulativeSum(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want []float64
	}{
		{"empty", nil, nil},
		{"single", []float64{5}, []float64{5}},
		{"simple", []float64{1, 2, 3, 4, 5}, []float64{1, 3, 6, 10, 15}},
		{"negative", []float64{5, -3, 2, -1}, []float64{5, 2, 4, 3}},
		{"zeros", []float64{0, 1, 0, 2, 0}, []float64{0, 1, 1, 3, 3}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			CumulativeSum(dst, tt.a)
			for i := range dst {
				if i < len(tt.want) && !almostEqual(dst[i], tt.want[i], 1e-14) {
					t.Errorf("CumulativeSum()[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

// TestSqrt_Large tests Sqrt with large arrays
func TestSqrt_Large(t *testing.T) {
	// Test different sizes to exercise SIMD paths
	sizes := []int{7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 256, 277, 1000}

	for _, n := range sizes {
		t.Run(string(rune('0'+n%10)), func(t *testing.T) {
			a := make([]float64, n)
			dst := make([]float64, n)

			// Fill with perfect squares
			for i := range a {
				a[i] = float64((i + 1) * (i + 1))
			}

			Sqrt(dst, a)

			// Verify
			for i := range dst {
				expected := float64(i + 1)
				if !almostEqual(dst[i], expected, 1e-10) {
					t.Errorf("Sqrt[%d] = %v, want %v", i, dst[i], expected)
				}
			}
		})
	}
}

// TestEuclideanDistance_HighDim tests with high-dimensional vectors
func TestEuclideanDistance_HighDim(t *testing.T) {
	// Create two high-dimensional vectors
	n := 1000
	a := make([]float64, n)
	b := make([]float64, n)

	// Unit vectors in different directions
	a[0] = 1.0 // [1, 0, 0, ...]
	b[1] = 1.0 // [0, 1, 0, ...]

	got := EuclideanDistance(a, b)
	want := math.Sqrt(2) // sqrt(1^2 + 1^2)

	if !almostEqual(got, want, 1e-14) {
		t.Errorf("EuclideanDistance(high-dim) = %v, want %v", got, want)
	}

	// Parallel vectors
	for i := range a {
		a[i] = float64(i)
		b[i] = float64(i)
	}

	got = EuclideanDistance(a, b)
	if got != 0 {
		t.Errorf("EuclideanDistance(same vectors) = %v, want 0", got)
	}
}

// Benchmarks for new operations

func BenchmarkSqrt_100(b *testing.B) {
	a := make([]float64, 100)
	dst := make([]float64, 100)
	for i := range a {
		a[i] = float64(i + 1)
	}

	b.SetBytes(100 * 8 * 2)

	for b.Loop() {
		Sqrt(dst, a)
	}
}

func BenchmarkReciprocal_100(b *testing.B) {
	a := make([]float64, 100)
	dst := make([]float64, 100)
	for i := range a {
		a[i] = float64(i + 1)
	}

	b.SetBytes(100 * 8 * 2)

	for b.Loop() {
		Reciprocal(dst, a)
	}
}

func BenchmarkMean_1000(b *testing.B) {
	a := make([]float64, 1000)
	for i := range a {
		a[i] = float64(i)
	}

	b.SetBytes(1000 * 8)

	var result float64
	for b.Loop() {
		result = Mean(a)
	}
	_ = result
}

func BenchmarkVariance_1000(b *testing.B) {
	a := make([]float64, 1000)
	for i := range a {
		a[i] = float64(i)
	}

	b.SetBytes(1000 * 8)

	var result float64
	for b.Loop() {
		result = Variance(a)
	}
	_ = result
}

func BenchmarkEuclideanDistance_100(b *testing.B) {
	a := make([]float64, 100)
	c := make([]float64, 100)
	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i + 1)
	}

	b.SetBytes(100 * 8 * 2)

	var result float64
	for b.Loop() {
		result = EuclideanDistance(a, c)
	}
	_ = result
}

func BenchmarkNormalize_100(b *testing.B) {
	a := make([]float64, 100)
	dst := make([]float64, 100)
	for i := range a {
		a[i] = float64(i + 1)
	}

	b.SetBytes(100 * 8 * 2)

	for b.Loop() {
		Normalize(dst, a)
	}
}

func BenchmarkCumulativeSum_1000(b *testing.B) {
	a := make([]float64, 1000)
	dst := make([]float64, 1000)
	for i := range a {
		a[i] = float64(i)
	}

	b.SetBytes(1000 * 8 * 2)

	for b.Loop() {
		CumulativeSum(dst, a)
	}
}

// Tests for DotProductBatch

func TestDotProductBatch(t *testing.T) {
	tests := []struct {
		name string
		rows [][]float64
		vec  []float64
		want []float64
	}{
		{"empty", nil, nil, nil},
		{"single row", [][]float64{{1, 2, 3}}, []float64{1, 1, 1}, []float64{6}},
		{"two rows", [][]float64{{1, 2}, {3, 4}}, []float64{1, 2}, []float64{5, 11}},
		{"polyphase", [][]float64{{1, 0, 1, 0}, {0, 1, 0, 1}}, []float64{1, 2, 3, 4}, []float64{4, 6}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.rows) == 0 {
				return
			}
			results := make([]float64, len(tt.rows))
			DotProductBatch(results, tt.rows, tt.vec)
			for i, want := range tt.want {
				if results[i] != want {
					t.Errorf("DotProductBatch()[%d] = %v, want %v", i, results[i], want)
				}
			}
		})
	}
}

func TestDotProductBatch_Large(t *testing.T) {
	// Simulate polyphase filter with 2 phases and 241-tap kernel
	numPhases := 2
	taps := 241
	rows := make([][]float64, numPhases)
	for i := range rows {
		rows[i] = make([]float64, taps)
		for j := range rows[i] {
			rows[i][j] = float64(i*taps + j + 1)
		}
	}
	vec := make([]float64, taps)
	for i := range vec {
		vec[i] = 1.0
	}

	results := make([]float64, numPhases)
	DotProductBatch(results, rows, vec)

	// First phase: sum of 1..241
	want0 := float64(taps * (taps + 1) / 2)
	// Second phase: sum of 242..482
	want1 := float64(taps*(taps+1)/2 + taps*taps)

	if !almostEqual(results[0], want0, 1e-10) {
		t.Errorf("DotProductBatch()[0] = %v, want %v", results[0], want0)
	}
	if !almostEqual(results[1], want1, 1e-10) {
		t.Errorf("DotProductBatch()[1] = %v, want %v", results[1], want1)
	}
}

// Tests for ConvolveValid

func TestConvolveValid(t *testing.T) {
	tests := []struct {
		name   string
		signal []float64
		kernel []float64
		want   []float64
	}{
		{"empty kernel", []float64{1, 2, 3}, nil, nil},
		{"kernel longer", []float64{1, 2}, []float64{1, 2, 3}, nil},
		{"single output", []float64{1, 2, 3}, []float64{1, 1, 1}, []float64{6}},
		{"two outputs", []float64{1, 2, 3, 4}, []float64{1, 1, 1}, []float64{6, 9}},
		{"identity", []float64{1, 2, 3, 4, 5}, []float64{1}, []float64{1, 2, 3, 4, 5}},
		{"sum2", []float64{1, 2, 3, 4, 5}, []float64{1, 1}, []float64{3, 5, 7, 9}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.kernel) == 0 || len(tt.signal) < len(tt.kernel) {
				return
			}
			validLen := len(tt.signal) - len(tt.kernel) + 1
			dst := make([]float64, validLen)
			ConvolveValid(dst, tt.signal, tt.kernel)
			for i, want := range tt.want {
				if dst[i] != want {
					t.Errorf("ConvolveValid()[%d] = %v, want %v", i, dst[i], want)
				}
			}
		})
	}
}

func TestConvolveValid_FIR(t *testing.T) {
	// Test with a simple 5-tap FIR filter (moving average)
	signal := make([]float64, 100)
	for i := range signal {
		signal[i] = float64(i + 1)
	}
	kernel := []float64{0.2, 0.2, 0.2, 0.2, 0.2} // 5-tap moving average

	validLen := len(signal) - len(kernel) + 1
	dst := make([]float64, validLen)
	ConvolveValid(dst, signal, kernel)

	// Check first few outputs
	// dst[0] = (1+2+3+4+5) * 0.2 = 3
	// dst[1] = (2+3+4+5+6) * 0.2 = 4
	if !almostEqual(dst[0], 3.0, 1e-10) {
		t.Errorf("ConvolveValid()[0] = %v, want 3.0", dst[0])
	}
	if !almostEqual(dst[1], 4.0, 1e-10) {
		t.Errorf("ConvolveValid()[1] = %v, want 4.0", dst[1])
	}
}

// Benchmarks for new functions

func BenchmarkDotProductBatch_2x241(b *testing.B) {
	// Simulate polyphase filter: 2 phases, 241 taps each
	rows := make([][]float64, 2)
	for i := range rows {
		rows[i] = make([]float64, 241)
		for j := range rows[i] {
			rows[i][j] = float64(j + 1)
		}
	}
	vec := make([]float64, 241)
	for i := range vec {
		vec[i] = float64(i + 1)
	}
	results := make([]float64, 2)

	b.SetBytes(2 * 241 * 8 * 2) // 2 rows * 241 elements * 8 bytes * 2 (rows + vec)

	for b.Loop() {
		DotProductBatch(results, rows, vec)
	}
}

func BenchmarkConvolveValid_1000x64(b *testing.B) {
	// 1000-sample signal, 64-tap kernel
	signal := make([]float64, 1000)
	kernel := make([]float64, 64)
	for i := range signal {
		signal[i] = float64(i)
	}
	for i := range kernel {
		kernel[i] = 1.0 / 64.0
	}
	validLen := len(signal) - len(kernel) + 1
	dst := make([]float64, validLen)

	b.SetBytes(int64(validLen * 64 * 8 * 2)) // Each output reads 64 signal + 64 kernel

	for b.Loop() {
		ConvolveValid(dst, signal, kernel)
	}
}

func BenchmarkConvolveValid_1000x241(b *testing.B) {
	// 1000-sample signal, 241-tap kernel (typical DFT size)
	signal := make([]float64, 1000)
	kernel := make([]float64, 241)
	for i := range signal {
		signal[i] = float64(i)
	}
	for i := range kernel {
		kernel[i] = 1.0 / 241.0
	}
	validLen := len(signal) - len(kernel) + 1
	dst := make([]float64, validLen)

	b.SetBytes(int64(validLen * 241 * 8 * 2))

	for b.Loop() {
		ConvolveValid(dst, signal, kernel)
	}
}

// Tests for Interleave2 and Deinterleave2

func TestInterleave2(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		b    []float64
		want []float64
	}{
		{"empty", nil, nil, nil},
		{"single", []float64{1}, []float64{2}, []float64{1, 2}},
		{"two", []float64{1, 3}, []float64{2, 4}, []float64{1, 2, 3, 4}},
		{"four", []float64{1, 3, 5, 7}, []float64{2, 4, 6, 8}, []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		{"five", []float64{1, 3, 5, 7, 9}, []float64{2, 4, 6, 8, 10}, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
		{"stereo", []float64{-1, 0, 1, 0, -1}, []float64{1, 0, -1, 0, 1}, []float64{-1, 1, 0, 0, 1, -1, 0, 0, -1, 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.a) == 0 {
				return
			}
			dst := make([]float64, len(tt.a)*2)
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
	// Test with larger arrays to exercise SIMD paths
	sizes := []int{8, 16, 17, 100, 1024, 1025}
	for _, n := range sizes {
		a := make([]float64, n)
		b := make([]float64, n)
		for i := range a {
			a[i] = float64(i * 2)
			b[i] = float64(i*2 + 1)
		}

		dst := make([]float64, n*2)
		Interleave2(dst, a, b)

		// Verify interleaving
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
		src   []float64
		wantA []float64
		wantB []float64
	}{
		{"empty", nil, nil, nil},
		{"single", []float64{1, 2}, []float64{1}, []float64{2}},
		{"two", []float64{1, 2, 3, 4}, []float64{1, 3}, []float64{2, 4}},
		{"four", []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1, 3, 5, 7}, []float64{2, 4, 6, 8}},
		{"five", []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, []float64{1, 3, 5, 7, 9}, []float64{2, 4, 6, 8, 10}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.src) < 2 {
				return
			}
			n := len(tt.src) / 2
			a := make([]float64, n)
			b := make([]float64, n)
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
		src := make([]float64, n*2)
		for i := range src {
			src[i] = float64(i)
		}

		a := make([]float64, n)
		b := make([]float64, n)
		Deinterleave2(a, b, src)

		for i := range n {
			wantA := float64(i * 2)
			wantB := float64(i*2 + 1)
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
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 100, 1000}
	for _, n := range sizes {
		// Original data
		a := make([]float64, n)
		b := make([]float64, n)
		for i := range a {
			a[i] = float64(i)
			b[i] = float64(i + 1000)
		}

		// Interleave then deinterleave
		interleaved := make([]float64, n*2)
		Interleave2(interleaved, a, b)

		aOut := make([]float64, n)
		bOut := make([]float64, n)
		Deinterleave2(aOut, bOut, interleaved)

		// Should match original
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

// Tests for ConvolveValidMulti

func TestConvolveValidMulti(t *testing.T) {
	tests := []struct {
		name    string
		signal  []float64
		kernels [][]float64
		want    [][]float64
	}{
		{
			"two_kernels_simple",
			[]float64{1, 2, 3, 4, 5},
			[][]float64{{1, 1}, {1, -1}},
			[][]float64{{3, 5, 7, 9}, {-1, -1, -1, -1}},
		},
		{
			"polyphase_like",
			[]float64{1, 2, 3, 4, 5, 6},
			[][]float64{{1, 0, 1}, {0, 1, 0}},
			[][]float64{{4, 6, 8, 10}, {2, 3, 4, 5}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kLen := len(tt.kernels[0])
			validLen := len(tt.signal) - kLen + 1
			dsts := make([][]float64, len(tt.kernels))
			for i := range dsts {
				dsts[i] = make([]float64, validLen)
			}

			ConvolveValidMulti(dsts, tt.signal, tt.kernels)

			for k, want := range tt.want {
				for i, v := range want {
					if !almostEqual(dsts[k][i], v, 1e-10) {
						t.Errorf("ConvolveValidMulti() dsts[%d][%d] = %v, want %v", k, i, dsts[k][i], v)
					}
				}
			}
		})
	}
}

func TestConvolveValidMulti_MatchesSingle(t *testing.T) {
	// Verify ConvolveValidMulti gives same results as multiple ConvolveValid calls
	signal := make([]float64, 1000)
	for i := range signal {
		signal[i] = float64(i) * 0.1
	}

	kernels := make([][]float64, 3)
	for k := range kernels {
		kernels[k] = make([]float64, 64)
		for i := range kernels[k] {
			kernels[k][i] = float64(k*64+i) * 0.01
		}
	}

	kLen := len(kernels[0])
	validLen := len(signal) - kLen + 1

	// Using ConvolveValidMulti
	dstsMulti := make([][]float64, len(kernels))
	for i := range dstsMulti {
		dstsMulti[i] = make([]float64, validLen)
	}
	ConvolveValidMulti(dstsMulti, signal, kernels)

	// Using individual ConvolveValid calls
	for k, kernel := range kernels {
		dstSingle := make([]float64, validLen)
		ConvolveValid(dstSingle, signal, kernel)

		for i := range dstSingle {
			if !almostEqual(dstsMulti[k][i], dstSingle[i], 1e-10) {
				t.Errorf("kernel %d, pos %d: multi=%v, single=%v", k, i, dstsMulti[k][i], dstSingle[i])
			}
		}
	}
}

// Benchmarks for Interleave2 and Deinterleave2

func BenchmarkInterleave2_1000(b *testing.B) {
	a := make([]float64, 1000)
	c := make([]float64, 1000)
	dst := make([]float64, 2000)
	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i + 1000)
	}

	b.SetBytes(1000 * 8 * 3) // Read 2 arrays, write 1

	for b.Loop() {
		Interleave2(dst, a, c)
	}
}

func BenchmarkDeinterleave2_1000(b *testing.B) {
	src := make([]float64, 2000)
	a := make([]float64, 1000)
	c := make([]float64, 1000)
	for i := range src {
		src[i] = float64(i)
	}

	b.SetBytes(1000 * 8 * 3)

	for b.Loop() {
		Deinterleave2(a, c, src)
	}
}

func BenchmarkConvolveValidMulti_2x1000x64(b *testing.B) {
	// 2 kernels, 1000-sample signal, 64-tap kernels (polyphase resampler scenario)
	signal := make([]float64, 1000)
	kernels := make([][]float64, 2)
	for i := range signal {
		signal[i] = float64(i)
	}
	for k := range kernels {
		kernels[k] = make([]float64, 64)
		for i := range kernels[k] {
			kernels[k][i] = 1.0 / 64.0
		}
	}
	validLen := len(signal) - len(kernels[0]) + 1
	dsts := make([][]float64, 2)
	for i := range dsts {
		dsts[i] = make([]float64, validLen)
	}

	b.SetBytes(int64(validLen * 64 * 8 * 2 * 2)) // 2 kernels

	for b.Loop() {
		ConvolveValidMulti(dsts, signal, kernels)
	}
}
