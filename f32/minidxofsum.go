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
// (n around 11 to 17) a pairwise kernel projects to cap near 1.5x, not
// enough to justify a separate assembly path. Use MinIdxOfSumRows to
// batch many argmin rows into one call.
// a and b are read-only; the call allocates nothing.
func MinIdxOfSum(a, b []float32) (idx int, val float32) {
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
// nothing. MinIdxOfSumRowsAdd is the fused form that also adds a per-row node
// value behind a limit gate.
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
	// Confirm every processed row's window lands in range before writing any
	// output; panic before any write if not.
	if !minIdxOfSumRowsWindowsOK(m, len(a), len(k), base, slide) {
		panic("f32.MinIdxOfSumRows: k window out of range")
	}
	minIdxOfSumRows32(vals[:m], idxs[:m], nil, a, k, base, slide, 0, 0)
}

// minIdxOfSumRowsWindowsOK reports whether every one of the m >= 1 rows' windows
// of width lenA lands inside a k of length lenK, without forming any product
// that can wrap.
func minIdxOfSumRowsWindowsOK(m, lenA, lenK, base, slide int) bool {
	// Confirm every processed row's window lands in [0, lim] before writing any
	// output; panic before any write if not. lim < 0 means lenA > lenK, so
	// no window of width lenA fits and every row (m >= 1) is out of range.
	// Rejecting that up front also keeps lim >= 0 inside the switch, so the
	// lim-(m-1) term below cannot underflow for any m. The row offsets form the
	// arithmetic sequence base + r*slide for r in [0, m). For the two slides
	// that reach a kernel (+1 and -1) the sequence is monotonic, so its two
	// extremes decide the whole range in O(1); any other slide keeps the
	// overflow-safe incremental walk (base+(m-1)*slide can wrap for adversarial
	// slide values, incremental addition cannot).
	lim := lenK - lenA
	ok := lim >= 0
	if ok {
		switch slide {
		case 1:
			// Ascending offsets base .. base+(m-1). The high bound is written
			// as lim-(m-1) (safe: lim >= 0 and m >= 1) rather than base+(m-1),
			// so an adversarial base never enters the arithmetic.
			ok = base >= 0 && base <= lim-(m-1)
		case -1:
			// Descending offsets base .. base-(m-1). base >= m-1 keeps the low
			// extreme non-negative; base <= lim keeps the high extreme in range.
			ok = base >= m-1 && base <= lim
		default:
			// General slide (only +1 and -1 reach a kernel; everything else
			// falls to the Go path). Walk the offsets incrementally rather than
			// forming base+(m-1)*slide, whose multiplication can wrap for
			// adversarial slide values and bypass the bound; incremental
			// addition cannot, since off is confirmed within [0, lim] before
			// each step. O(m) is noise next to the O(m*n) the operation does.
			off := base
			for range m {
				if off < 0 || off > lim {
					ok = false
					break
				}
				off += slide
			}
		}
	}
	return ok
}

// MinIdxOfSumRowsAdd is MinIdxOfSumRows fused with a gated add of a per-row node
// value: one Viterbi step. For each row r in [0, m), where
// m = min(len(dst), len(idxs), len(node)), it takes the row's minimum v and its
// smallest attaining index i exactly as MinIdxOfSumRows does (same window,
// same strict less-than, same lowest-index tie-break, same NaN rules), then
//
//	if v < limit { dst[r] = node[r] + v; idxs[r] = i } else { dst[r] = sentinel; idxs[r] = -1 }
//
// so idxs[r] == -1 exactly on rows that took the sentinel. The gate tests the
// minimum alone, never the sum: a NaN v, a NaN limit, v == limit and a row whose
// candidates are all +Inf are dead. node[r]+v is one float32 addition (one
// rounding, never fused); a live row whose sum overflows writes +Inf.
//
// If len(a) == 0 no k index is reached or validated and every processed row is
// dead (sentinel, -1) for every limit: the minimum over the empty set is +Inf,
// which is never below limit.
//
// MinIdxOfSumRowsAdd panics before writing any output if any processed row's
// window falls outside k, or if len(a) does not fit in int32. dst[m:] and
// idxs[m:] are left untouched.
//
// # Aliasing
//
// dst may overlay node exactly (dst[r] is written after node[r] is read). dst
// must not overlap a or k, nor node at a shifted offset. a, k and node are
// read-only and may overlap one another.
//
// Uses NEON on ARM64 and AVX2 on AMD64 for slide values +1 and -1, with a pure Go
// fallback elsewhere, sharing the MinIdxOfSumRows search kernels. All paths
// produce bit-identical results. The call allocates nothing.
func MinIdxOfSumRowsAdd(dst []float32, idxs []int32, node, a, k []float32, base, slide int, limit, sentinel float32) {
	m := min(len(dst), len(idxs), len(node))
	if m == 0 {
		return
	}
	if len(a) == 0 {
		for r := range m {
			dst[r] = sentinel
			idxs[r] = -1
		}
		return
	}
	if len(a) > math.MaxInt32 {
		panic("f32.MinIdxOfSumRowsAdd: len(a) exceeds int32 index range")
	}
	if !minIdxOfSumRowsWindowsOK(m, len(a), len(k), base, slide) {
		panic("f32.MinIdxOfSumRowsAdd: k window out of range")
	}
	minIdxOfSumRows32(dst[:m], idxs[:m], node[:m], a, k, base, slide, limit, sentinel)
}
