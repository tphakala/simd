package f32

import "math"

// MinIdxOfSum returns the index and value of the minimum of a[i]+b[i] over
// i in [0, min(len(a), len(b))). Each candidate is computed with a single
// float32 addition (exactly one rounding; implementations never use FMA).
// Ties resolve to the lowest index. NaN candidates never win a comparison;
// if the first candidate is NaN it is never displaced, so an input whose
// candidates are all NaN returns index 0. Returns (-1, 0) for empty input.
//
// MinIdxOfSum is scalar on every path by design: at the motivating sizes
// (n around 11 to 17) dispatch overhead eats a pairwise kernel. Use
// MinIdxOfSumRows to batch many argmin rows into one call.
// a and b are read-only; the call allocates nothing.
func MinIdxOfSum(a, b []float32) (int, float32) {
	return minIdxOfSumGo(a, b)
}

// MinIdxOfSumRows computes, for each row r in [0, m) where
// m = min(len(vals), len(idxs)):
//
//	c(r, i) = a[i] + k[base+r*slide+i]  for i in [0, len(a))
//	vals[r] = the minimum c(r, i)
//	idxs[r] = the smallest i attaining it
//
// following MinIdxOfSum's contract per row: strict less-than, so ties
// resolve to the lowest i, each candidate is exactly one float32 addition
// (never an FMA), and NaN candidates never displace the incumbent. +Inf
// entries in k act as padding that never beats any finite candidate; a row
// whose candidates are all +Inf yields (0, +Inf).
//
// Every k index reached by a processed row must be in range:
// MinIdxOfSumRows panics before writing any output if any processed
// row's window falls outside k, or if len(a) does not fit in int32.
// If len(a) == 0 no k index is reached; every processed row gets
// vals[r] = 0, idxs[r] = -1. vals[m:] and idxs[m:] are left untouched.
//
// Uses NEON on ARM64 and AVX2 on AMD64 for slide values +1 and -1 (the
// sliding-window shapes), with a pure Go fallback elsewhere. All paths
// produce bit-identical results. Inputs are read-only; the call allocates
// nothing.
func MinIdxOfSumRows(vals []float32, idxs []int32, a, k []float32, base, slide int) {
	m := min(len(vals), len(idxs))
	if m == 0 {
		return
	}
	if len(a) == 0 {
		for r := range m {
			vals[r] = 0
			idxs[r] = -1
		}
		return
	}
	if len(a) > math.MaxInt32 {
		panic("f32.MinIdxOfSumRows: len(a) exceeds int32 index range")
	}
	// Walk the row offsets incrementally instead of computing the extreme
	// row as base+(m-1)*slide: the multiplication can wrap for adversarial
	// slide values and bypass the bound. Incremental addition cannot: off
	// is confirmed within [0, lim] before each step, so a single wrapping
	// add always lands negative and the next check catches it. O(m) is
	// noise next to the O(m*n) the operation itself performs.
	lim := len(k) - len(a)
	off := base
	for r := 0; r < m; r++ {
		if off < 0 || off > lim {
			panic("f32.MinIdxOfSumRows: k window out of range")
		}
		off += slide
	}
	minIdxOfSumRows32(vals[:m], idxs[:m], a, k, base, slide)
}
