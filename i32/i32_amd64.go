//go:build amd64

package i32

import "github.com/tphakala/simd/cpu"

// Minimum number of int32 pairs before the AVX kernel beats the scalar loop.
// AVX processes 8 pairs (8 int32 per 256-bit register) per iteration.
const minAVXElements = 8

// hasAVX gates the SIMD kernels. The interleave kernels use only AVX1
// instructions (VUNPCKLPS / VPERM2F128 / VSHUFPS), so AVX without AVX2 is
// sufficient. This checks the CPU feature explicitly rather than relying on
// length alone, so the package is safe on the (now rare) AVX-less amd64
// baseline.
var hasAVX = cpu.X86.AVX

func interleave2I32(dst, a, b []int32) {
	if hasAVX && len(a) >= minAVXElements {
		interleave2AVX(dst, a, b)
		return
	}
	interleave2Go(dst, a, b)
}

func deinterleave2I32(a, b, src []int32) {
	if hasAVX && len(a) >= minAVXElements {
		deinterleave2AVX(a, b, src)
		return
	}
	deinterleave2Go(a, b, src)
}

//go:noescape
func interleave2AVX(dst, a, b []int32)

//go:noescape
func deinterleave2AVX(a, b, src []int32)

// The arithmetic / decorrelation / fixed-predictor kernels operate on 256-bit
// integer lanes (VPADDD/VPSUBD/VPSRAD/...), which require AVX2 rather than the
// AVX1 that suffices for the float-shuffle interleave kernels above. They gate
// on AVX2 explicitly and fall back to the pure-Go reference otherwise.
var hasAVX2 = cpu.X86.AVX2

// minLPCRestoreOrder is the smallest predictor order at which the SIMD decode
// recurrence kernel beats the scalar Go recurrence. The recurrence is serial
// (each output feeds the next), so SIMD only helps once the per-output tap dot
// product has enough work to amortize its horizontal reduction; below this the
// scalar path wins. Tuned from the benchmarks.
const minLPCRestoreOrder = 8

func addI32(dst, a, b []int32) {
	if hasAVX2 && len(dst) >= minAVXElements {
		addAVX2(dst, a, b)
		return
	}
	addGo(dst, a, b)
}

func subI32(dst, a, b []int32) {
	if hasAVX2 && len(dst) >= minAVXElements {
		subAVX2(dst, a, b)
		return
	}
	subGo(dst, a, b)
}

func midSideEncodeI32(mid, side, left, right []int32) {
	if hasAVX2 && len(mid) >= minAVXElements {
		midSideEncodeAVX2(mid, side, left, right)
		return
	}
	midSideEncodeGo(mid, side, left, right)
}

func midSideDecodeI32(left, right, mid, side []int32) {
	if hasAVX2 && len(left) >= minAVXElements {
		midSideDecodeAVX2(left, right, mid, side)
		return
	}
	midSideDecodeGo(left, right, mid, side)
}

func diff1I32(dst, src []int32) {
	if hasAVX2 && len(dst) >= minAVXElements {
		diff1AVX2(dst, src)
		return
	}
	diff1Go(dst, src)
}

func diff2I32(dst, src []int32) {
	if hasAVX2 && len(dst) >= minAVXElements {
		diff2AVX2(dst, src)
		return
	}
	diff2Go(dst, src)
}

func diff3I32(dst, src []int32) {
	if hasAVX2 && len(dst) >= minAVXElements {
		diff3AVX2(dst, src)
		return
	}
	diff3Go(dst, src)
}

func diff4I32(dst, src []int32) {
	if hasAVX2 && len(dst) >= minAVXElements {
		diff4AVX2(dst, src)
		return
	}
	diff4Go(dst, src)
}

//go:noescape
func addAVX2(dst, a, b []int32)

//go:noescape
func subAVX2(dst, a, b []int32)

//go:noescape
func midSideEncodeAVX2(mid, side, left, right []int32)

//go:noescape
func midSideDecodeAVX2(left, right, mid, side []int32)

func cumsumI32(a []int32) {
	if hasAVX2 && len(a) >= minAVXElements {
		cumsumAVX2(a)
		return
	}
	cumsumGo(a)
}

//go:noescape
func cumsumAVX2(a []int32)

//go:noescape
func diff1AVX2(dst, src []int32)

//go:noescape
func diff2AVX2(dst, src []int32)

//go:noescape
func diff3AVX2(dst, src []int32)

//go:noescape
func diff4AVX2(dst, src []int32)

// lpcResidualEncodeI32 dispatches the quantized-LPC encode FIR. The kernel
// vectorizes across output samples and accumulates the prediction in int64, so
// it gates on AVX2 (VPMULDQ widening multiply) and needs at least one full
// 8-output block past the warm-up.
func lpcResidualEncodeI32(res, samples, coeffs []int32, shift uint) {
	if hasAVX2 && len(res)-len(coeffs) >= minAVXElements {
		lpcResidualEncodeAVX2(res, samples, coeffs, shift)
		return
	}
	lpcResidualEncodeGo(res, samples, coeffs, shift)
}

// lpcRestoreI32 dispatches the quantized-LPC decode recurrence. Each output
// feeds the next prediction, so the kernel vectorizes only the per-output tap
// dot product and pays off only once the order is large enough to amortize the
// horizontal reduction; below that it stays on the scalar Go recurrence.
func lpcRestoreI32(out, residual, coeffs []int32, shift uint) {
	order := len(coeffs)
	if hasAVX2 && order >= minLPCRestoreOrder && order <= maxLPCRestoreOrder && len(out)-order >= 1 {
		// The kernel dots the ascending window out[i-order..i-1] with the
		// coefficients reversed, so it lines up two contiguous loads. Reverse
		// once into a stack array (no heap alloc; the array does not escape
		// through the //go:noescape kernel).
		var rc [maxLPCRestoreOrder]int32
		for k := range order {
			rc[k] = coeffs[order-1-k]
		}
		lpcRestoreAVX2(out, residual, rc[:order], shift)
		return
	}
	lpcRestoreGo(out, residual, coeffs, shift)
}

//go:noescape
func lpcResidualEncodeAVX2(res, samples, coeffs []int32, shift uint)

//go:noescape
func lpcRestoreAVX2(out, residual, rcoeffs []int32, shift uint)

// riceSumsI32 dispatches the Rice per-parameter unary-bit sums. The AVX2 kernel
// always writes the full riceParamCount (15) FLAC sums, so it is used only for
// that width; other widths and short inputs use the pure-Go reference.
func riceSumsI32(sums []uint64, res []int32) {
	if hasAVX2 && len(sums) == riceParamCount && len(res) >= minAVXElements {
		riceSumsAVX2(sums, res)
		return
	}
	riceSumsGo(sums, res)
}

// zigzagSumI32 dispatches the residual zigzag-fold sum (RiceSums' k=0 column).
// The AVX2 kernel widens to int64 lanes; it gates on AVX2 and at least one full
// 8-element block, falling back to the pure-Go reference otherwise.
func zigzagSumI32(res []int32) uint64 {
	if hasAVX2 && len(res) >= minAVXElements {
		return zigzagSumAVX2(res)
	}
	return zigzagSumGo(res)
}

//go:noescape
func zigzagSumAVX2(res []int32) uint64

// fixedAbsSumsI32 dispatches the five fixed-predictor residual abs-sums. The
// AVX2 kernel computes the order-0..4 finite differences in int64 lanes via a
// windowed sign-extending cascade, so it gates on AVX2 and at least one full
// 8-element block (the kernel handles the first 4 warm-up samples and the tail
// scalar). Shorter inputs use the pure-Go reference.
func fixedAbsSumsI32(src []int32, sums *[5]uint64) {
	if hasAVX2 && len(src) >= minAVXElements {
		fixedAbsSumsAVX2(src, sums)
		return
	}
	fixedAbsSumsGo(src, sums)
}

//go:noescape
func fixedAbsSumsAVX2(src []int32, sums *[5]uint64)

//go:noescape
func riceSumsAVX2(sums []uint64, res []int32)

// riceSumsWideI32 dispatches the FLAC 5-bit Rice sums (len(sums) ==
// riceMaxParam5+1 = 31). The AVX2 path reuses the 15-wide kernel for columns
// 0..14 and a high kernel for columns 15..30, so the whole range is vectorized
// instead of falling to the scalar tail; both gate on AVX2 and one full block.
func riceSumsWideI32(sums []uint64, res []int32) {
	// The exact-width gate mirrors riceSumsI32: riceSumsHighAVX2 is a fixed
	// 16-column writer, so only dispatch it when sums is the full 31-wide slice;
	// any other length goes to the pure-Go reference (which handles all widths).
	if hasAVX2 && len(sums) == riceMaxParam5+1 && len(res) >= minAVXElements {
		riceSumsAVX2(sums[:riceParamCount], res)     // columns 0..14
		riceSumsHighAVX2(sums[riceParamCount:], res) // columns 15..30
		return
	}
	riceSumsGo(sums, res)
}

//go:noescape
func riceSumsHighAVX2(sums []uint64, res []int32)

// minMaxI32 dispatches the signed int32 min/max reduction. The AVX2 kernel does
// the reduction in 8-wide VPMINSD/VPMAXSD lanes with a scalar tail, so it gates
// on AVX2 and at least one full 8-element block; shorter slices use the pure-Go
// reference. res is non-empty (the public MinMax guards the empty case).
func minMaxI32(res []int32) (minVal, maxVal int32) {
	if hasAVX2 && len(res) >= minAVXElements {
		return minMaxAVX2(res)
	}
	return minMaxGo(res)
}

//go:noescape
func minMaxAVX2(res []int32) (minVal, maxVal int32)
