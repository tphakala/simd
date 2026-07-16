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
//
// This is NOT a tunable. All three kernels hardcode four lags: their
// accumulator count, their y load offsets (+0/+2/+4/+6 bytes), their
// len(y)-3 clamp, and their four dst stores. Changing this constant compiles
// cleanly and then breaks at runtime, in both directions: raising it leaves
// the extra dst words unwritten, and lowering it makes the assembly store past
// the end of the dst slice the dispatcher hands it, which is an out-of-bounds
// write. Moving it means rewriting xcorr4NEON, xcorr4SSE2 and xcorr4AVX2.
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

// xcorrWindow returns the y window the 4-lag kernel may read for the block at
// lag k: lag k+3 reaches y[k+3+len(x)-1], so the block needs len(x)+3 elements
// from y[k]. That is in bounds precisely when k+xcorrLagBlock <= m, which is
// the loop condition every dispatcher uses, so the slice cannot panic.
//
// The per-arch dispatchers each inline their own block loop rather than sharing
// one driver parameterised by the kernel. That looks like needless duplication
// and is not: passing the kernel as a func value makes the call indirect, and
// escape analysis cannot see through an indirect call, so it must assume the
// slices escape. That defeats the //go:noescape on the kernels and propagates
// out to XCorr itself, forcing every CALLER to heap-allocate its dst, x and y.
// Direct calls keep the whole path allocation-free. Do not "simplify" the
// dispatchers back into a shared higher-order driver.
func xcorrWindow(x, y []int16, k int) []int16 {
	return y[k : k+len(x)+xcorrLagBlock-1]
}
