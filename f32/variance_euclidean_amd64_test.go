//go:build amd64

package f32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// varEuclidSeq builds a deterministic float32 slice that mixes signs and
// magnitudes so the kernel's lane reduction is exercised across every unroll
// remainder.
func varEuclidSeq(n, seed int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = (float32((i*7+seed)%17) - 8) * 0.5 // range [-4, 4]
	}
	return s
}

// reduceMatch32 reports whether got and want agree within the tolerance used for
// f32 reductions, comparing NaN and Inf by class rather than by value because the
// SIMD path accumulates in a different order than the scalar reference.
func reduceMatch32(got, want float32) bool {
	g, w := float64(got), float64(want)
	if math.IsNaN(w) || math.IsNaN(g) {
		return math.IsNaN(w) == math.IsNaN(g)
	}
	if math.IsInf(w, 0) || math.IsInf(g, 0) {
		return math.IsInf(w, 1) == math.IsInf(g, 1) && math.IsInf(w, -1) == math.IsInf(g, -1)
	}
	return math.Abs(g-w) <= 1e-4*math.Abs(w)+1e-5
}

func TestVarianceKernelsAMD64(t *testing.T) {
	for n := 0; n <= 40; n++ {
		a := varEuclidSeq(n, 1)
		var mean float32
		if n > 0 {
			mean = Mean(a)
		}
		want := variance32Go(a, mean)

		if cpu.X86.SSE2 {
			if got := varianceSSE(a, mean); !reduceMatch32(got, want) {
				t.Errorf("varianceSSE(n=%d) = %v, want %v", n, got, want)
			}
		}
		if cpu.X86.AVX && cpu.X86.FMA {
			if got := varianceAVX(a, mean); !reduceMatch32(got, want) {
				t.Errorf("varianceAVX(n=%d) = %v, want %v", n, got, want)
			}
		}
	}
}

func TestEuclideanDistanceKernelsAMD64(t *testing.T) {
	for n := 0; n <= 40; n++ {
		a := varEuclidSeq(n, 1)
		b := varEuclidSeq(n, 9)
		want := euclideanDistance32Go(a, b)

		if cpu.X86.SSE2 {
			if got := euclideanDistanceSSE(a, b); !reduceMatch32(got, want) {
				t.Errorf("euclideanDistanceSSE(n=%d) = %v, want %v", n, got, want)
			}
		}
		if cpu.X86.AVX && cpu.X86.FMA {
			if got := euclideanDistanceAVX(a, b); !reduceMatch32(got, want) {
				t.Errorf("euclideanDistanceAVX(n=%d) = %v, want %v", n, got, want)
			}
		}
	}
}

// TestVarianceEuclideanSpecialValuesAMD64 feeds NaN, +/-Inf, and denormal inputs
// through the kernels and asserts they propagate the same special class as the
// scalar reference.
func TestVarianceEuclideanSpecialValuesAMD64(t *testing.T) {
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	denorm := math.Float32frombits(1) // smallest positive subnormal

	cases := [][]float32{
		{nan, 1, 2, 3, 4, 5, 6, 7, 8},
		{1, 2, 3, inf, 5, 6, 7, 8, 9},
		{denorm, denorm, denorm, denorm, denorm},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, nan},
		{-inf, 1, 2, 3},
	}
	for ci, a := range cases {
		mean := Mean(a)
		vwant := variance32Go(a, mean)

		b := make([]float32, len(a))
		for i := range a {
			b[i] = a[i] * 0.5
		}
		ewant := euclideanDistance32Go(a, b)

		if cpu.X86.SSE2 {
			if got := varianceSSE(a, mean); !reduceMatch32(got, vwant) {
				t.Errorf("case %d varianceSSE = %v, want %v", ci, got, vwant)
			}
			if got := euclideanDistanceSSE(a, b); !reduceMatch32(got, ewant) {
				t.Errorf("case %d euclideanDistanceSSE = %v, want %v", ci, got, ewant)
			}
		}
		if cpu.X86.AVX && cpu.X86.FMA {
			if got := varianceAVX(a, mean); !reduceMatch32(got, vwant) {
				t.Errorf("case %d varianceAVX = %v, want %v", ci, got, vwant)
			}
			if got := euclideanDistanceAVX(a, b); !reduceMatch32(got, ewant) {
				t.Errorf("case %d euclideanDistanceAVX = %v, want %v", ci, got, ewant)
			}
		}
	}
}

// TestVarianceEuclideanAllocsAMD64 confirms the public APIs stay allocation-free
// through the SIMD dispatch path.
func TestVarianceEuclideanAllocsAMD64(t *testing.T) {
	a := varEuclidSeq(100, 1)
	b := varEuclidSeq(100, 2)
	if allocs := testing.AllocsPerRun(100, func() { _ = Variance(a) }); allocs != 0 {
		t.Errorf("Variance allocs = %v, want 0", allocs)
	}
	if allocs := testing.AllocsPerRun(100, func() { _ = StdDev(a) }); allocs != 0 {
		t.Errorf("StdDev allocs = %v, want 0", allocs)
	}
	if allocs := testing.AllocsPerRun(100, func() { _ = EuclideanDistance(a, b) }); allocs != 0 {
		t.Errorf("EuclideanDistance allocs = %v, want 0", allocs)
	}
}
