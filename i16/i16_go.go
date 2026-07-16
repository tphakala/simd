package i16

// Pure-Go reference implementations.
//
// These are the source of truth for behavior: every SIMD kernel is validated
// for bit-exact parity against the functions here. They are compiled on every
// architecture and used directly as the fallback when no SIMD path applies.

func interleave2Go(dst, a, b []int16) {
	for i := range a {
		dst[i*2] = a[i]
		dst[i*2+1] = b[i]
	}
}

func deinterleave2Go(a, b, src []int16) {
	for i := range a {
		a[i] = src[i*2]
		b[i] = src[i*2+1]
	}
}

// dotGo computes the widening int16 dot product, accumulating each int32
// product into an int32 running sum with two's-complement wraparound. It
// clamps to min(len(a), len(b)) itself so the unsafe entry point can hand it
// mismatched slices, and it is the bit-exact source of truth the SIMD Dot
// kernels are validated against.
//
// Wrapping (rather than saturating) accumulation is the contract: wrapping
// addition is associative and commutative modulo 2^32, so the SIMD lane
// grouping and horizontal reduction produce bit-identical results to this
// sequential loop for every input, including forced overflow.
func dotGo(a, b []int16) int32 {
	var s int32
	for i := range min(len(a), len(b)) {
		s += int32(a[i]) * int32(b[i])
	}
	return s
}

// xcorrLagBlock is the number of lags the SIMD kernels evaluate per call. Four
// is the libopus xcorr_kernel shape: it is the point where each x load is
// amortised across enough accumulators to hide the multiply-accumulate latency,
// while still fitting the lag accumulators in registers.
const xcorrLagBlock = 4

// xcorrLags returns the number of lags whose full window fits in y, which is
// what XCorr computes. Lags past this point would read off the end of y, so
// they are not computed and dst is left untouched there.
func xcorrLags(dst []int32, x, y []int16) int {
	if len(x) == 0 || len(y) < len(x) {
		return 0
	}
	return min(len(dst), len(y)-len(x)+1)
}

// xcorrGo is the pure-Go reference: one wrapping int32 dot product per lag. It
// is the bit-exact source of truth the SIMD XCorr kernels are validated
// against, and it is defined in terms of dotGo so the two primitives cannot
// drift apart.
func xcorrGo(dst []int32, x, y []int16) {
	for k := range xcorrLags(dst, x, y) {
		dst[k] = dotGo(x, y[k:])
	}
}

// xcorrBlocked drives a 4-lag SIMD kernel over as many full blocks of lags as
// fit, then finishes the remaining one to three lags with the dot kernel. It is
// shared by the amd64 and arm64 dispatchers, which differ only in the kernels
// they pass.
//
// The y slice handed to kernel4 is exactly the window that block may read:
// lag k+3 reaches y[k+3+len(x)-1], so the block needs len(x)+3 elements from
// y[k]. That is in bounds precisely when k+xcorrLagBlock <= m, which is the
// loop condition, so the slice expression cannot panic.
func xcorrBlocked(dst []int32, x, y []int16,
	kernel4 func(dst []int32, x, y []int16),
	dot func(a, b []int16) int32,
) {
	m := xcorrLags(dst, x, y)
	k := 0
	for ; k+xcorrLagBlock <= m; k += xcorrLagBlock {
		kernel4(dst[k:k+xcorrLagBlock], x, y[k:k+len(x)+xcorrLagBlock-1])
	}
	for ; k < m; k++ {
		dst[k] = dot(x, y[k:k+len(x)])
	}
}
