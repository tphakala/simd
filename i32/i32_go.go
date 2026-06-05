package i32

// Pure-Go reference implementations.
//
// These are the source of truth for behavior: every SIMD kernel is validated
// for bit-exact parity against the functions here. They are compiled on every
// architecture and used directly as the fallback when no SIMD path applies.

func interleave2Go(dst, a, b []int32) {
	for i := range a {
		dst[i*2] = a[i]
		dst[i*2+1] = b[i]
	}
}

func deinterleave2Go(a, b, src []int32) {
	for i := range a {
		a[i] = src[i*2]
		b[i] = src[i*2+1]
	}
}

func addGo(dst, a, b []int32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func subGo(dst, a, b []int32) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func midSideEncodeGo(mid, side, left, right []int32) {
	for i := range mid {
		mid[i] = (left[i] + right[i]) >> 1
		side[i] = left[i] - right[i]
	}
}

func midSideDecodeGo(left, right, mid, side []int32) {
	for i := range left {
		// The encoder's mid = (l+r)>>1 dropped the low bit of l+r. Because l+r
		// and l-r have the same parity, side&1 restores it.
		sum := (mid[i] << 1) | (side[i] & 1)
		left[i] = (sum + side[i]) >> 1
		right[i] = (sum - side[i]) >> 1
	}
}

// The order-K fixed-predictor residual coefficients: the signed binomials
// (-1)^j * C(K,j) applied to src[n], src[n-1], ..., src[n-K]. The order is the
// slice length minus one, so diffGo needs no separate order argument.
var (
	fixedCoeffs1 = []int32{1, -1}
	fixedCoeffs2 = []int32{1, -2, 1}
	fixedCoeffs3 = []int32{1, -3, 3, -1}
	fixedCoeffs4 = []int32{1, -4, 6, -4, 1}
)

// diffGo writes the fixed-predictor residual for coefficients c into dst: the
// first order=len(c)-1 entries are the verbatim warm-up (dst[i]=src[i]) and
// dst[n] for n>=order is the binomial combination. int32 arithmetic wraps,
// matching the SIMD kernels.
func diffGo(dst, src, c []int32) {
	order := len(c) - 1
	w := min(order, len(src))
	copy(dst[:w], src[:w])
	for n := order; n < len(src); n++ {
		var acc int32
		for j, cj := range c {
			acc += cj * src[n-j]
		}
		dst[n] = acc
	}
}

func diff1Go(dst, src []int32) { diffGo(dst, src, fixedCoeffs1) }
func diff2Go(dst, src []int32) { diffGo(dst, src, fixedCoeffs2) }
func diff3Go(dst, src []int32) { diffGo(dst, src, fixedCoeffs3) }
func diff4Go(dst, src []int32) { diffGo(dst, src, fixedCoeffs4) }

// cumsumGo is the in-place inclusive prefix sum (cumulative sum): a[i] becomes
// a[0]+...+a[i], with a[0] left unchanged. It is the building block the
// SIMD-accelerated Restore kernels compose: the order-K fixed predictor is the
// K-th forward difference, so its inverse is K cumulative-sum passes. int32
// arithmetic wraps, matching the SIMD kernels.
func cumsumGo(a []int32) {
	for i := 1; i < len(a); i++ {
		a[i] += a[i-1]
	}
}

// lpcResidualEncodeGo writes the quantized-LPC residual for an order-len(coeffs)
// predictor. The first order entries are the verbatim warm-up; for i >= order,
//
//	res[i] = samples[i] - int32((Σ_j coeffs[j]*samples[i-1-j]) >> shift)
//
// The prediction sum is accumulated in int64 (matching libFLAC) so it does not
// overflow for FLAC's coefficient precision and order; only the shifted result
// is truncated to int32. res and samples are the caller-clamped equal-length
// slices; res must not alias samples. int32 truncation wraps, matching the SIMD
// kernels.
func lpcResidualEncodeGo(res, samples, coeffs []int32, shift uint) {
	order := len(coeffs)
	w := min(order, len(res))
	copy(res[:w], samples[:w])
	for i := order; i < len(res); i++ {
		var acc int64
		for j, c := range coeffs {
			acc += int64(c) * int64(samples[i-1-j])
		}
		res[i] = samples[i] - int32(acc>>shift)
	}
}

// int32SignShift sign-extends an int32 to its all-zero/all-one mask: r>>31
// (arithmetic) is 0 for non-negative r and -1 (0xFFFFFFFF) for negative r, the
// sign-mask half of the zigzag fold.
const int32SignShift = 31

// riceSumsGo writes the per-parameter Rice unary-bit sums into sums:
//
//	sums[k] = Σ_i (zigzag(res[i]) >> k)   for k in [0, len(sums))
//
// where zigzag(r) = (r<<1) ^ (r>>31) folds a signed residual to its unsigned
// Rice symbol. It fully overwrites sums and is the bit-exact source of truth the
// SIMD kernels are validated against. The fold uses int32 arithmetic (matching
// the kernels): r<<1 wraps and r>>31 is the arithmetic sign-extension, so a
// math.MinInt32 residual folds to 2^32-1 before the unsigned widen.
func riceSumsGo(sums []uint64, res []int32) {
	for k := range sums {
		sums[k] = 0
	}
	for _, r := range res {
		u := uint64(uint32((r << 1) ^ (r >> int32SignShift)))
		for k := range sums {
			sums[k] += u >> uint(k)
		}
	}
}

// zigzagSumGo returns Σ_i zigzag(res[i]), the FLAC residual zigzag fold summed
// in uint64. It is exactly the k=0 column of riceSumsGo and the bit-exact source
// of truth the SIMD kernels are validated against. The fold uses int32
// arithmetic (matching the kernels): r<<1 wraps and r>>31 is the arithmetic
// sign-extension, so a math.MinInt32 residual folds to 2^32-1 before the
// unsigned widen.
func zigzagSumGo(res []int32) uint64 {
	var s uint64
	for _, r := range res {
		s += uint64(uint32((r << 1) ^ (r >> int32SignShift)))
	}
	return s
}

// minMaxGo returns the smallest and largest int32 in res via a single signed
// scan. res must be non-empty (the public MinMax guards the empty case); it is
// the bit-exact source of truth the SIMD MinMax kernels are validated against.
func minMaxGo(res []int32) (minVal, maxVal int32) {
	lo, hi := res[0], res[0]
	for _, r := range res[1:] {
		if r < lo {
			lo = r
		}
		if r > hi {
			hi = r
		}
	}
	return lo, hi
}

// absU64 returns the absolute value of v as a uint64. v is bounded well inside
// the int64 range here (at most a 4th finite difference of int32 samples, ~2^35),
// so the math.MinInt64 edge case (where -v overflows) cannot arise.
func absU64(v int64) uint64 {
	if v < 0 {
		return uint64(-v)
	}
	return uint64(v)
}

// fixedAbsSumsGo writes the five fixed-predictor residual abs-sums into sums:
//
//	sums[order] = Σ_{i>=order} |e_order[i]|   for order in 0..4
//
// where e_order is the order-th forward finite difference of src (order 0 is src
// itself, order 1 the first difference, and so on). The differences are computed
// in int64 (a 4th difference of int32 samples can reach ~2^35, beyond int32), and
// sums[order] excludes the first order warm-up samples, matching the cost FLAC
// uses to choose a fixed predictor order. It fully overwrites sums and is the
// bit-exact source of truth the SIMD kernels are validated against.
//
// The accumulators are kept in a local array and stored once at the end: writing
// through *sums each iteration would force them to memory (the compiler cannot
// prove sums does not alias src). This mirrors go-flac's FixedAbsSums exactly so
// swapping it for the SIMD path leaves the encoded stream byte-identical.
func fixedAbsSumsGo(src []int32, sums *[5]uint64) {
	var s [5]uint64
	var p1, p2, p3, p4 int64 // p1=prev e0, p2=prev e1, p3=prev e2, p4=prev e3
	n := len(src)
	// Warm-up: order o's difference e_o is first defined at sample index o, so at
	// position i only orders 0..i are active. Handled in this short bounded loop
	// (len(s)-1 = 4 samples at most) so the main loop below can stay branchless.
	w := min(n, len(s)-1)
	for i := range w {
		e0 := int64(src[i])
		e1 := e0 - p1
		e2 := e1 - p2
		e3 := e2 - p3
		e := [4]int64{e0, e1, e2, e3}
		for o := 0; o <= i; o++ {
			s[o] += absU64(e[o])
		}
		p1, p2, p3, p4 = e0, e1, e2, e3
	}
	// Main region i >= len(s)-1: every order is active, so this is branchless.
	for i := w; i < n; i++ {
		e0 := int64(src[i])
		e1 := e0 - p1
		e2 := e1 - p2
		e3 := e2 - p3
		e4 := e3 - p4
		s[0] += absU64(e0)
		s[1] += absU64(e1)
		s[2] += absU64(e2)
		s[3] += absU64(e3)
		s[4] += absU64(e4)
		p1, p2, p3, p4 = e0, e1, e2, e3
	}
	*sums = s
}

// lpcRestoreGo is the exact inverse of lpcResidualEncodeGo: it reconstructs the
// samples from the [order warm-up | residual] layout via the serial recurrence
//
//	out[i] = residual[i] + int32((Σ_j coeffs[j]*out[i-1-j]) >> shift)
//
// Each out[i] depends on the order previously reconstructed outputs, so unlike
// the encode FIR this cannot be vectorized across i. out and residual are the
// caller-clamped equal-length slices; out must not alias residual.
func lpcRestoreGo(out, residual, coeffs []int32, shift uint) {
	order := len(coeffs)
	w := min(order, len(out))
	copy(out[:w], residual[:w])
	for i := order; i < len(out); i++ {
		var acc int64
		for j, c := range coeffs {
			acc += int64(c) * int64(out[i-1-j])
		}
		out[i] = residual[i] + int32(acc>>shift)
	}
}
