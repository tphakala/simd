package i32

import (
	"math"
	"math/rand"
	"testing"
)

// Tests for MaxAbs, the peak-magnitude reduction defined as libopus celtMaxabs32:
// max(maxVal, -minVal) over a single signed min/max scan, with -minVal a WRAPPING
// int32 negate. The load-bearing case is a MinInt32 element beside a negative one,
// where celtMaxabs32 (max of maxVal and the wrapped -minVal) diverges from the
// intuitive per-lane abs-then-max: for [MinInt32, -3] the answer is -3, not 3.

// maxAbsOracle computes celtMaxabs32 independently of maxAbsGo: it scans for the
// signed min and max, then wraps -min through int64 -> int32 (a different code
// path than the int32 unary minus in the reference) and returns the larger. It
// pins the reference rather than trusting a formula shared with it.
func maxAbsOracle(a []int32) int32 {
	if len(a) == 0 {
		return 0
	}
	lo, hi := a[0], a[0]
	for _, v := range a {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	neg := int32(-int64(lo)) // wrap -lo into int32: -MinInt32 -> MinInt32
	if hi > neg {
		return hi
	}
	return neg
}

// absThenMax is the WRONG definition MaxAbs must NOT implement: per-lane wrapping
// abs, then the maximum. It differs from celtMaxabs32 exactly when a MinInt32
// element rides beside a negative one; TestMaxAbs_Divergence pins that gap so a
// VPABSD-based rewrite could never pass silently.
func absThenMax(a []int32) int32 {
	var m int32
	for i, v := range a {
		av := v
		if av < 0 {
			av = -av // wrapping abs: -MinInt32 == MinInt32
		}
		if i == 0 || av > m {
			m = av
		}
	}
	return m
}

// TestMaxAbsOracle confirms the oracle itself encodes the wrap and the divergence,
// so the parity tests below rest on a checked foundation.
func TestMaxAbsOracle(t *testing.T) {
	if got := maxAbsOracle([]int32{math.MinInt32, -3}); got != -3 {
		t.Fatalf("oracle([MinInt32,-3]) = %d, want -3", got)
	}
	if got := maxAbsOracle([]int32{math.MinInt32}); got != math.MinInt32 {
		t.Fatalf("oracle([MinInt32]) = %d, want %d", got, int32(math.MinInt32))
	}
	if got := absThenMax([]int32{math.MinInt32, -3}); got != 3 {
		t.Fatalf("absThenMax([MinInt32,-3]) = %d, want 3 (the wrong answer MaxAbs must avoid)", got)
	}
}

// TestMaxAbs sweeps every tier-3 length against both the pure-Go reference and the
// independent oracle, so a fault cannot hide by agreeing with the reference alone.
// MinInt32 rides index 0 so the wrapping -minVal combine is exercised at every
// length, and MaxInt32 rides the last index so the scalar tail must be folded in.
func TestMaxAbs(t *testing.T) {
	for _, n := range tier3Lengths {
		a := genI32(n, 71)
		if n > 0 {
			a[0] = math.MinInt32
			a[n-1] = math.MaxInt32
		}
		got := MaxAbs(a)
		// maxAbsGo requires a non-empty slice (only the public MaxAbs guards
		// empty), so the reference parity is checked for n > 0 only; the oracle
		// covers the empty case.
		if n > 0 {
			if want := maxAbsGo(a); got != want {
				t.Fatalf("MaxAbs n=%d = %d, want %d (reference)", n, got, want)
			}
		}
		if want := maxAbsOracle(a); got != want {
			t.Fatalf("MaxAbs n=%d = %d, want %d (oracle)", n, got, want)
		}
	}
}

// TestMaxAbs_Random crosses random int32 (organically hitting the full range) with
// the hand-picked specials, at every length that straddles a vector block and its
// scalar tail on both arches, checking against the reference and the oracle.
func TestMaxAbs_Random(t *testing.T) {
	rng := rand.New(rand.NewSource(19))
	specials := []int32{math.MinInt32, math.MaxInt32, 0, -1, 1}
	for _, n := range []int{1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 33, 100, 1000} {
		for trial := range 20 {
			a := make([]int32, n)
			for i := range a {
				if rng.Intn(3) == 0 {
					a[i] = specials[rng.Intn(len(specials))]
				} else {
					a[i] = int32(rng.Uint32())
				}
			}
			got := MaxAbs(a)
			if want := maxAbsGo(a); got != want {
				t.Fatalf("MaxAbs n=%d trial=%d = %d, want %d (reference)", n, trial, got, want)
			}
			if want := maxAbsOracle(a); got != want {
				t.Fatalf("MaxAbs n=%d trial=%d = %d, want %d (oracle)", n, trial, got, want)
			}
		}
	}
}

// TestMaxAbs_Cases pins the hand-crafted contracts in isolation, spanning both the
// Go path (n below the SIMD thresholds) and the vector path.
func TestMaxAbs_Cases(t *testing.T) {
	cases := []struct {
		name string
		a    []int32
		want int32
	}{
		{"empty", []int32{}, 0},
		{"nil", nil, 0},
		{"single-min", []int32{math.MinInt32}, math.MinInt32},
		{"single-max", []int32{math.MaxInt32}, math.MaxInt32},
		{"single-neg", []int32{-42}, 42},
		{"min-beside-neg", []int32{math.MinInt32, -3}, -3},
		{"all-positive", []int32{1, 2, 3, 4, 5}, 5},
		{"all-negative", []int32{-1, -2, -3, -4, -5}, 5},
		{"mixed", []int32{-10, 3, -7, 2}, 10},
		{"all-minus-one", []int32{-1, -1, -1, -1}, 1},
		// The divergence at a full AVX2 block (8) and a NEON block (4): a MinInt32
		// beside negatives keeps the answer at -3 (celtMaxabs32), not 3 (abs-max).
		{"min-beside-neg-neon", []int32{math.MinInt32, -3, -3, -3}, -3},
		{"min-beside-neg-avx", []int32{math.MinInt32, -3, -3, -3, -3, -3, -3, -3}, -3},
	}
	for _, c := range cases {
		if got := MaxAbs(c.a); got != c.want {
			t.Errorf("MaxAbs(%s) = %d, want %d", c.name, got, c.want)
		}
		if got := maxAbsOracle(c.a); got != c.want {
			t.Errorf("oracle(%s) = %d, want %d", c.name, got, c.want)
		}
	}
}

// TestMaxAbs_Divergence pins the whole point of the primitive: the celtMaxabs32
// definition diverges from per-lane abs-then-max whenever a MinInt32 element rides
// beside a negative one. It checks lengths on the Go path and both SIMD paths so a
// VPABSD/ABS-based kernel would be caught on every backend.
func TestMaxAbs_Divergence(t *testing.T) {
	for _, n := range []int{2, 3, 4, 5, 8, 9, 16, 17} {
		a := make([]int32, n)
		for i := range a {
			a[i] = -3
		}
		a[0] = math.MinInt32
		got := MaxAbs(a)
		if got != -3 {
			t.Fatalf("MaxAbs divergence n=%d = %d, want -3 (celtMaxabs32)", n, got)
		}
		if wrong := absThenMax(a); got == wrong {
			t.Fatalf("MaxAbs divergence n=%d matched abs-then-max (%d): it must NOT be built on VPABSD", n, wrong)
		}
	}
}

// TestMaxAbs_Unaligned sweeps every element offset so neither an aligned-load
// substitution nor an off-by-one block boundary can survive. The driving extremes
// ride the ends so the head block and the scalar tail both stay load-bearing.
func TestMaxAbs_Unaligned(t *testing.T) {
	const span = 300
	backing := genI32(span, 33)
	for _, n := range []int{4, 5, 7, 8, 9, 11, 17, 25, 33, 64, 240} {
		for off := range 8 {
			a := backing[off+1 : off+1+n]
			// Save and plant extremes at both ends, then restore afterwards.
			first, last := a[0], a[n-1]
			a[0] = math.MinInt32
			a[n-1] = math.MaxInt32
			got := MaxAbs(a)
			if want := maxAbsGo(a); got != want {
				t.Fatalf("MaxAbs unaligned n=%d off=%d = %d, want %d (reference)", n, off, got, want)
			}
			if want := maxAbsOracle(a); got != want {
				t.Fatalf("MaxAbs unaligned n=%d off=%d = %d, want %d (oracle)", n, off, got, want)
			}
			a[0], a[n-1] = first, last
		}
	}
}

// TestMaxAbs_AllocFree confirms the scalar return forces no caller allocation. The
// buffer is declared INSIDE the measured closure so only allocations forced by
// MaxAbs itself are counted.
func TestMaxAbs_AllocFree(t *testing.T) {
	if got := testing.AllocsPerRun(50, func() {
		var a [1000]int32
		_ = MaxAbs(a[:])
	}); got != 0 {
		t.Errorf("MaxAbs forces %v caller allocations per run, want 0", got)
	}
}

// FuzzMaxAbs differentially fuzzes the dispatched MaxAbs against the pure-Go
// reference and the independent oracle over arbitrary int32 samples, so tail
// handling and the wrap are explored past the hand-picked seeds. It reuses
// addLenSeeds, whose element counts bracket the 4/8-lane block boundaries.
func FuzzMaxAbs(f *testing.F) {
	addLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		a := i32sFromBits(raw)
		got := MaxAbs(a)
		// maxAbsGo requires a non-empty slice; the oracle covers the empty case.
		if len(a) > 0 {
			if want := maxAbsGo(a); got != want {
				t.Fatalf("MaxAbs = %d, want %d (reference, len=%d)", got, want, len(a))
			}
		}
		if want := maxAbsOracle(a); got != want {
			t.Fatalf("MaxAbs = %d, want %d (oracle, len=%d)", got, want, len(a))
		}
	})
}
