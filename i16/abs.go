package i16

// Wrapping absolute value at 16-bit width, element-wise and as a reduction.
// Both serve the fixed-point envelope and peak scans that precede gain
// decisions: Abs feeds an envelope follower, MaxAbs is the headroom probe.
// Neither saturates, so the SIMD lanes and the Go reference are bit-identical
// for every input including the type minimum.

// Abs writes dst[i] = |a[i]| for i in [0, n), n = min(len(dst), len(a)). The
// negation wraps in int16 rather than saturating: |MinInt16| does not fit
// int16, so Abs maps -32768 to -32768 (compare [github.com/tphakala/simd/i8.Abs],
// which saturates by design). Every other input yields the true magnitude.
// Any trailing capacity in dst is left untouched.
//
// dst and a may overlap only if they start at the same address.
func Abs(dst, a []int16) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	absI16(dst[:n], a[:n])
}

// MaxAbs returns max_i |a[i]| as int, in [0, 32768]: |-32768| = 32768 does not
// fit int16, which is why the result widens (mirroring
// [github.com/tphakala/simd/i8.MaxAbs] and libopus celt_maxabs16) and why this
// is not equivalent to Abs followed by a signed max reduction. An empty a
// returns 0. a is read-only; the call allocates nothing.
func MaxAbs(a []int16) int {
	if len(a) == 0 {
		return 0
	}
	return maxAbsI16(a)
}
