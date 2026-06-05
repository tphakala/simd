//go:build !amd64 && !arm64

package i32

func interleave2I32(dst, a, b []int32)   { interleave2Go(dst, a, b) }
func deinterleave2I32(a, b, src []int32) { deinterleave2Go(a, b, src) }

func addI32(dst, a, b []int32) { addGo(dst, a, b) }
func subI32(dst, a, b []int32) { subGo(dst, a, b) }

func midSideEncodeI32(mid, side, left, right []int32) { midSideEncodeGo(mid, side, left, right) }
func midSideDecodeI32(left, right, mid, side []int32) { midSideDecodeGo(left, right, mid, side) }

func cumsumI32(a []int32) { cumsumGo(a) }

func diff1I32(dst, src []int32) { diff1Go(dst, src) }
func diff2I32(dst, src []int32) { diff2Go(dst, src) }
func diff3I32(dst, src []int32) { diff3Go(dst, src) }
func diff4I32(dst, src []int32) { diff4Go(dst, src) }

func lpcResidualEncodeI32(res, samples, coeffs []int32, shift uint) {
	lpcResidualEncodeGo(res, samples, coeffs, shift)
}

func lpcRestoreI32(out, residual, coeffs []int32, shift uint) {
	lpcRestoreGo(out, residual, coeffs, shift)
}

func riceSumsI32(sums []uint64, res []int32) { riceSumsGo(sums, res) }

func zigzagSumI32(res []int32) uint64 { return zigzagSumGo(res) }

func fixedAbsSumsI32(src []int32, sums *[5]uint64) { fixedAbsSumsGo(src, sums) }

func riceSumsWideI32(sums []uint64, res []int32) { riceSumsGo(sums, res) }

func minMaxI32(res []int32) (minVal, maxVal int32) { return minMaxGo(res) }
