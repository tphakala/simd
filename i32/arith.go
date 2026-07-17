package i32

// Element-wise integer arithmetic on int32 slices.
//
// Add and Sub are the generic building blocks for combining two int32 streams
// element by element (for example summing or differencing two channels); Abs
// is the single-operand magnitude. All clamp to the shortest operand, write
// into the caller-provided dst, and use int32 wraparound (two's complement) so
// the SIMD and pure-Go paths are bit-identical across the full int32 range.

// Add writes dst[i] = a[i] + b[i] for i in [0, n), n = min(len(dst), len(a),
// len(b)). Any trailing capacity in dst is left untouched.
func Add(dst, a, b []int32) {
	n := min(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	addI32(dst[:n], a[:n], b[:n])
}

// Sub writes dst[i] = a[i] - b[i] for i in [0, n), n = min(len(dst), len(a),
// len(b)). Any trailing capacity in dst is left untouched.
func Sub(dst, a, b []int32) {
	n := min(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	subI32(dst[:n], a[:n], b[:n])
}

// Abs writes dst[i] = |a[i]| for i in [0, n), n = min(len(dst), len(a)). The
// negation wraps in int32 rather than saturating: |MinInt32| does not fit
// int32, so Abs maps MinInt32 to MinInt32 (as does
// [github.com/tphakala/simd/i16.Abs] at its own minimum; compare
// [github.com/tphakala/simd/i8.Abs], which saturates by design). Every other
// input yields the true magnitude. Any trailing capacity in dst is left
// untouched.
//
// dst and a may overlap only if they start at the same address.
func Abs(dst, a []int32) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	absI32(dst[:n], a[:n])
}
