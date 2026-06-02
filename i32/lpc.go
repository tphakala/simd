package i32

// Quantized-LPC residual encode and restore for the FLAC codec.
//
// FLAC's LPC subframes predict each sample from a fixed-point linear
// combination of the preceding `order` samples, using integer coefficients
// quantized to a shared right-shift. The encoder Rice-codes the residual
// LPCResidualEncode produces; the decoder reconstructs the samples with
// LPCRestore. The predictor order is taken from len(coeffs) (FLAC allows 1..32),
// and shift is the quantization right-shift applied to the prediction sum.
//
// Both functions lay the data out as a FLAC subframe stores it: the first
// `order` entries are the verbatim warm-up samples and the rest are the residual
// (encode) or the reconstructed samples (decode). The prediction sum is
// accumulated in int64 so it does not overflow for FLAC's coefficient precision
// and order; only the final shifted prediction is truncated to int32. Each
// function clamps to n = min of its two slice lengths, writes only into the
// caller's destination, and uses int32 wraparound. The destination must not
// alias the source.
//
// LPCResidualEncode is a FIR that vectorizes across output samples; LPCRestore
// is a serial recurrence (each output feeds the next prediction) and so is far
// less parallel. Both have SIMD kernels where they pay off, with a pure-Go
// fallback that is the bit-exact source of truth.

// maxLPCShift is the largest shift the kernels handle. A 64-bit accumulator
// shifted by 63 already saturates to 0 or -1, and the SIMD kernels' emulated
// arithmetic shift (and x86's shift-count masking) are only correct for counts
// in [0, 63], so the public functions clamp shift to this value.
const maxLPCShift = 63

// maxLPCRestoreOrder caps the predictor order the SIMD decode kernels accept. It
// matches FLAC's maximum LPC order (32) and sizes the stack array the dispatch
// reverses the coefficients into, so the reversal stays allocation-free. Orders
// above it fall back to the pure-Go recurrence.
const maxLPCRestoreOrder = 32

// LPCResidualEncode writes the quantized-LPC residual into res:
//
//	res[i] = samples[i]                                              for i < order
//	res[i] = samples[i] - int32((Σ_j coeffs[j]*samples[i-1-j]) >> shift) for i >= order
//
// where order = len(coeffs). It processes n = min(len(res), len(samples)) and
// leaves any trailing capacity in res untouched. An empty coeffs (order 0) or
// fewer than order+1 samples yields all warm-up (res is samples verbatim).
//
// shift is the quantization right-shift (FLAC uses 0..31). It is clamped to 63
// so the result stays consistent across the Go and SIMD paths: a shift of 63 or
// more of a 64-bit accumulator already yields 0 or -1, and the SIMD kernels'
// emulated shift is only valid for counts in [0, 63].
func LPCResidualEncode(res, samples, coeffs []int32, shift uint) {
	n := min(len(res), len(samples))
	if n == 0 {
		return
	}
	if order := len(coeffs); order == 0 || order >= n {
		copy(res[:n], samples[:n])
		return
	}
	if shift > maxLPCShift {
		shift = maxLPCShift
	}
	lpcResidualEncodeI32(res[:n], samples[:n], coeffs, shift)
}

// LPCRestore reconstructs the samples LPCResidualEncode encoded, inverting it
// with the serial recurrence:
//
//	out[i] = residual[i]                                            for i < order
//	out[i] = residual[i] + int32((Σ_j coeffs[j]*out[i-1-j]) >> shift)   for i >= order
//
// where order = len(coeffs). It processes n = min(len(out), len(residual)) and
// leaves any trailing capacity in out untouched. Given the same coeffs and shift,
// LPCRestore(LPCResidualEncode(samples)) == samples. shift is clamped to 63, as
// in LPCResidualEncode, so encode and decode stay matched across all paths.
func LPCRestore(out, residual, coeffs []int32, shift uint) {
	n := min(len(out), len(residual))
	if n == 0 {
		return
	}
	if order := len(coeffs); order == 0 || order >= n {
		copy(out[:n], residual[:n])
		return
	}
	if shift > maxLPCShift {
		shift = maxLPCShift
	}
	lpcRestoreI32(out[:n], residual[:n], coeffs, shift)
}
