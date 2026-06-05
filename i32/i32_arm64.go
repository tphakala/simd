//go:build arm64

package i32

import "github.com/tphakala/simd/cpu"

// NEON processes 4 int32 pairs (one .4S register) per iteration.
const minNEONElements = 4

var hasNEON = cpu.ARM64.NEON

func interleave2I32(dst, a, b []int32) {
	if hasNEON && len(a) >= minNEONElements {
		interleave2NEON(dst, a, b)
		return
	}
	interleave2Go(dst, a, b)
}

func deinterleave2I32(a, b, src []int32) {
	if hasNEON && len(a) >= minNEONElements {
		deinterleave2NEON(a, b, src)
		return
	}
	deinterleave2Go(a, b, src)
}

//go:noescape
func interleave2NEON(dst, a, b []int32)

//go:noescape
func deinterleave2NEON(a, b, src []int32)

func addI32(dst, a, b []int32) {
	if hasNEON && len(dst) >= minNEONElements {
		addNEON(dst, a, b)
		return
	}
	addGo(dst, a, b)
}

func subI32(dst, a, b []int32) {
	if hasNEON && len(dst) >= minNEONElements {
		subNEON(dst, a, b)
		return
	}
	subGo(dst, a, b)
}

func midSideEncodeI32(mid, side, left, right []int32) {
	if hasNEON && len(mid) >= minNEONElements {
		midSideEncodeNEON(mid, side, left, right)
		return
	}
	midSideEncodeGo(mid, side, left, right)
}

func midSideDecodeI32(left, right, mid, side []int32) {
	if hasNEON && len(left) >= minNEONElements {
		midSideDecodeNEON(left, right, mid, side)
		return
	}
	midSideDecodeGo(left, right, mid, side)
}

func diff1I32(dst, src []int32) {
	if hasNEON && len(dst) >= minNEONElements {
		diff1NEON(dst, src)
		return
	}
	diff1Go(dst, src)
}

func diff2I32(dst, src []int32) {
	if hasNEON && len(dst) >= minNEONElements {
		diff2NEON(dst, src)
		return
	}
	diff2Go(dst, src)
}

func diff3I32(dst, src []int32) {
	if hasNEON && len(dst) >= minNEONElements {
		diff3NEON(dst, src)
		return
	}
	diff3Go(dst, src)
}

func diff4I32(dst, src []int32) {
	if hasNEON && len(dst) >= minNEONElements {
		diff4NEON(dst, src)
		return
	}
	diff4Go(dst, src)
}

//go:noescape
func addNEON(dst, a, b []int32)

//go:noescape
func subNEON(dst, a, b []int32)

//go:noescape
func midSideEncodeNEON(mid, side, left, right []int32)

//go:noescape
func midSideDecodeNEON(left, right, mid, side []int32)

func cumsumI32(a []int32) {
	if hasNEON && len(a) >= minNEONElements {
		cumsumNEON(a)
		return
	}
	cumsumGo(a)
}

//go:noescape
func cumsumNEON(a []int32)

//go:noescape
func diff1NEON(dst, src []int32)

//go:noescape
func diff2NEON(dst, src []int32)

//go:noescape
func diff3NEON(dst, src []int32)

//go:noescape
func diff4NEON(dst, src []int32)

// minNEONRestoreOrder is the smallest predictor order at which the SIMD decode
// recurrence kernel beats the scalar Go recurrence on NEON (tuned from the
// Raspberry Pi 5 benchmarks). Below it the serial dependency leaves too little
// per-output tap work to amortize the horizontal reduction.
const minNEONRestoreOrder = 8

func lpcResidualEncodeI32(res, samples, coeffs []int32, shift uint) {
	if hasNEON && len(res)-len(coeffs) >= minNEONElements {
		lpcResidualEncodeNEON(res, samples, coeffs, shift)
		return
	}
	lpcResidualEncodeGo(res, samples, coeffs, shift)
}

func lpcRestoreI32(out, residual, coeffs []int32, shift uint) {
	order := len(coeffs)
	if hasNEON && order >= minNEONRestoreOrder && order <= maxLPCRestoreOrder && len(out)-order >= 1 {
		// The kernel dots the ascending window with the coefficients reversed,
		// so reverse once into a stack array (no heap alloc; the array does not
		// escape through the //go:noescape kernel).
		var rc [maxLPCRestoreOrder]int32
		for k := range order {
			rc[k] = coeffs[order-1-k]
		}
		lpcRestoreNEON(out, residual, rc[:order], shift)
		return
	}
	lpcRestoreGo(out, residual, coeffs, shift)
}

//go:noescape
func lpcResidualEncodeNEON(res, samples, coeffs []int32, shift uint)

//go:noescape
func lpcRestoreNEON(out, residual, rcoeffs []int32, shift uint)

// riceSumsI32 dispatches the Rice per-parameter unary-bit sums. The NEON kernel
// always writes the full riceParamCount (15) FLAC sums, so it is used only for
// that width; other widths and short inputs use the pure-Go reference.
func riceSumsI32(sums []uint64, res []int32) {
	if hasNEON && len(sums) == riceParamCount && len(res) >= minNEONElements {
		riceSumsNEON(sums, res)
		return
	}
	riceSumsGo(sums, res)
}

// zigzagSumI32 dispatches the residual zigzag-fold sum (RiceSums' k=0 column).
// The NEON kernel widens to int64 lanes; it gates on NEON and at least one full
// 4-element block, falling back to the pure-Go reference otherwise.
func zigzagSumI32(res []int32) uint64 {
	if hasNEON && len(res) >= minNEONElements {
		return zigzagSumNEON(res)
	}
	return zigzagSumGo(res)
}

//go:noescape
func zigzagSumNEON(res []int32) uint64

// fixedAbsSumsI32 dispatches the five fixed-predictor residual abs-sums. The
// NEON kernel computes the order-0..4 finite differences in int64 lanes via a
// windowed sign-extending cascade; it handles the first 4 warm-up samples and
// the tail scalar, so it only needs the warm-up samples to exist. Shorter inputs
// use the pure-Go reference.
func fixedAbsSumsI32(src []int32, sums *[5]uint64) {
	if hasNEON && len(src) >= minNEONElements {
		fixedAbsSumsNEON(src, sums)
		return
	}
	fixedAbsSumsGo(src, sums)
}

//go:noescape
func fixedAbsSumsNEON(src []int32, sums *[5]uint64)

//go:noescape
func riceSumsNEON(sums []uint64, res []int32)

// riceSumsWideI32 dispatches the FLAC 5-bit Rice sums (len(sums) ==
// riceMaxParam5+1 = 31). The NEON path reuses the 15-wide kernel for columns
// 0..14 and a high kernel for columns 15..30, so the whole range is vectorized
// instead of falling to the scalar tail; both gate on NEON and one full block.
func riceSumsWideI32(sums []uint64, res []int32) {
	// The exact-width gate mirrors riceSumsI32: riceSumsHighNEON is a fixed
	// 16-column writer, so only dispatch it when sums is the full 31-wide slice;
	// any other length goes to the pure-Go reference (which handles all widths).
	if hasNEON && len(sums) == riceMaxParam5+1 && len(res) >= minNEONElements {
		riceSumsNEON(sums[:riceParamCount], res)     // columns 0..14
		riceSumsHighNEON(sums[riceParamCount:], res) // columns 15..30
		return
	}
	riceSumsGo(sums, res)
}

//go:noescape
func riceSumsHighNEON(sums []uint64, res []int32)

// minMaxI32 dispatches the signed int32 min/max reduction. The NEON kernel does
// the reduction in 4-wide SMIN/SMAX lanes with a single-instruction SMINV/SMAXV
// across-vector fold and a scalar tail, so it gates on NEON and at least one full
// 4-element block; shorter slices use the pure-Go reference. res is non-empty
// (the public MinMax guards the empty case).
func minMaxI32(res []int32) (minVal, maxVal int32) {
	if hasNEON && len(res) >= minNEONElements {
		return minMaxNEON(res)
	}
	return minMaxGo(res)
}

//go:noescape
func minMaxNEON(res []int32) (minVal, maxVal int32)
