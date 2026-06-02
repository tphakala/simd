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
