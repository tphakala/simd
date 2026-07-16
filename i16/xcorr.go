package i16

// XCorr fills dst[k] with the wrapping int32 dot product of x against y[k:]:
//
//	dst[k] = sum_j int32(x[j]) * int32(y[k+j])   for j in [0, len(x))
//
// for k in [0, m), where m = min(len(dst), len(y)-len(x)+1). This is multi-lag
// cross-correlation, the shape pitch analysis and filtering call at: it reuses
// each x load across several lag accumulators instead of re-reading x per lag.
//
// Only lags whose full window fits in y are computed. dst[m:] is left untouched
// rather than zeroed, matching the library's clamp-and-leave-the-tail
// convention, so a caller can size dst generously without losing prior contents.
// A call with len(x) == 0, len(y) < len(x), or empty dst computes nothing.
//
// Each lag accumulates in int32 with two's-complement wraparound, exactly as
// [DotProduct] does and for the same reason: wrapping addition is associative,
// so the lane grouping and horizontal reduction are bit-identical to the scalar
// loop for every input, including operands engineered to overflow. dst[k] is
// therefore always identical to DotProduct(x, y[k:k+len(x)]), which is the
// property the tests pin.
//
// x and y are read-only; the call allocates nothing, including for the caller
// (the parameters do not escape).
//
// Note the operand order relative to [github.com/tphakala/simd/f32.ConvolveValid],
// which computes the same thing for float32: that one takes the long slid
// operand second and the short stationary operand third, where XCorr takes the
// short one (x) second. Passing them in f32's order here binds x to the long
// operand, trips the len(y) < len(x) guard, and returns having written nothing,
// with no panic and no diagnostic.
func XCorr(dst []int32, x, y []int16) {
	if len(dst) == 0 || len(x) == 0 || len(y) < len(x) {
		return
	}
	xcorrI16(dst, x, y)
}
