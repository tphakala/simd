//go:build amd64

package i8

import "github.com/tphakala/simd/cpu"

// The int8 kernels operate on 256-bit integer lanes (VPADDSB/VPSUBSB/VPMINSB/
// VPMAXSB/VPMOVSXB*/VPMADDWD), which require AVX2. They gate on AVX2 explicitly
// and fall back to the pure-Go reference on the (now rare) AVX-less baseline and
// for slices shorter than one vector block.
var (
	hasAVX2 = cpu.X86.AVX2
	// hasAVXVNNI gates the VPDPBUSD tier above AVX2 for DotProductBatch. The VEX
	// form of AVX-VNNI runs on YMM state and its dispatch sits above AVX2, so
	// cpu.clearAVX2 clears it too; AVXVNNI therefore implies AVX2 in this repo.
	hasAVXVNNI = cpu.X86.AVXVNNI
)

// Per-kernel minimum element counts: one full vector iteration's worth of int8
// inputs. Shorter slices use the pure-Go reference.
const (
	blockSat32   = 32 // VPADDSB/VPSUBSB process 32 bytes per iteration
	blockMinMax  = 32 // VPMINSB/VPMAXSB process 32 bytes per iteration
	blockReduce  = 16 // Sum/DotProduct widen 16 bytes per iteration (VPMOVSXBW)
	blockWiden16 = 16 // ToInt16 widens 16 bytes per iteration (VPMOVSXBW)
	blockWiden32 = 8  // ToInt32 widens 8 bytes per iteration (VPMOVSXBD)
)

// minAVXVNNIBatch is the vec-length cut for the AVX-VNNI DotProductBatch kernel.
// It is an independent literal (not an alias of blockReduce) so the AVX2 batch
// threshold and the VNNI threshold can be retuned separately. The VNNI kernel is
// correct at any vec length (it falls through to a scalar tail); this is a
// performance cut only.
//
// It sits well above blockReduce because VPDPBUSD's edge is per-block, while the
// bias+correction adds a fixed per-call cost the AVX2 kernel does not pay: a
// fifth (vec-sum) accumulator to reduce and the 128*sum(vec) subtraction. On an
// i7-1260P (Alder Lake) that fixed cost is not amortized on a single 4-row group
// below ~64: measured VNNI/AVX2 was ~1.03 at dims 32, a noisy ~0.95-1.02 across
// 40/48/56, and a decisive 0.87 at dims 64 (then 0.73 at 128 and 0.58 at 256).
// 64 is the first length with a stable, large-margin win, and it is where the
// quantized-matmul callers that motivate this kernel operate, so vec shorter than
// 64 stays on the AVX2 kernel with no regression.
const minAVXVNNIBatch = 64

func addSatI8(dst, a, b []int8) {
	if hasAVX2 && len(dst) >= blockSat32 {
		addSatAVX2(dst, a, b)
		return
	}
	addSatGo(dst, a, b)
}

func subSatI8(dst, a, b []int8) {
	if hasAVX2 && len(dst) >= blockSat32 {
		subSatAVX2(dst, a, b)
		return
	}
	subSatGo(dst, a, b)
}

func toI16(dst []int16, src []int8) {
	if hasAVX2 && len(src) >= blockWiden16 {
		toI16AVX2(dst, src)
		return
	}
	toI16Go(dst, src)
}

func toI32(dst []int32, src []int8) {
	if hasAVX2 && len(src) >= blockWiden32 {
		toI32AVX2(dst, src)
		return
	}
	toI32Go(dst, src)
}

func sumI8(a []int8) int32 {
	if hasAVX2 && len(a) >= blockReduce {
		return sumAVX2(a)
	}
	return sumGo(a)
}

func dotI8(a, b []int8) int32 {
	if hasAVX2 && len(a) >= blockReduce {
		return dotAVX2(a, b)
	}
	return dotGo(a, b)
}

func dotProductBatchI8(results []int32, rows [][]int8, vec []int8) {
	vecLen := len(vec)
	switch {
	case hasAVXVNNI && len(rows) >= 4 && vecLen >= minAVXVNNIBatch:
		dotProductBatch4AVXVNNI(results, rows, vec, vecLen)
	case hasAVX2 && len(rows) >= 4 && vecLen >= blockReduce:
		dotProductBatch4AVX2(results, rows, vec, vecLen)
	default:
		dotProductBatchRows(results, rows, vec)
	}
}

// dotProductBatch4AVX2 scores rows against vec in groups of four so vec stays in
// registers across the group instead of being re-loaded per row. A group whose
// four rows are each at least vecLen long takes the fused 4-row AVX2 kernel; a
// ragged group (any row shorter than vec) and the trailing rows past the last
// full group both go through the shared dotProductBatchRows fallback (per-row
// dotI8, scoring 0 for an empty clamped row). The caller guarantees AVX2,
// len(rows) >= 4, vecLen >= blockReduce, and len(results) == len(rows).
func dotProductBatch4AVX2(results []int32, rows [][]int8, vec []int8, vecLen int) {
	i := 0
	for i+3 < len(rows) {
		r0, r1, r2, r3 := rows[i], rows[i+1], rows[i+2], rows[i+3]
		if len(r0) >= vecLen && len(r1) >= vecLen && len(r2) >= vecLen && len(r3) >= vecLen {
			dotProduct4AVX2(results[i:i+4], r0, r1, r2, r3, vec)
		} else {
			dotProductBatchRows(results[i:i+4], rows[i:i+4], vec)
		}
		i += 4
	}
	dotProductBatchRows(results[i:], rows[i:], vec)
}

// dotProductBatch4AVXVNNI mirrors dotProductBatch4AVX2 but scores full 4-row
// groups with the AVX-VNNI kernel; ragged groups (any row shorter than vec) and
// the trailing rows past the last full group share the same per-row
// dotProductBatchRows fallback (the per-row dotI8 stays on the AVX2 tier). The
// group loop is duplicated rather than shared with the AVX2 driver behind a
// kernel func value on purpose: an indirect call defeats escape analysis and the
// kernel's //go:noescape, forcing every caller to heap-allocate. The caller
// guarantees AVX-VNNI, len(rows) >= 4, vecLen >= minAVXVNNIBatch, and
// len(results) == len(rows).
func dotProductBatch4AVXVNNI(results []int32, rows [][]int8, vec []int8, vecLen int) {
	i := 0
	for i+3 < len(rows) {
		r0, r1, r2, r3 := rows[i], rows[i+1], rows[i+2], rows[i+3]
		if len(r0) >= vecLen && len(r1) >= vecLen && len(r2) >= vecLen && len(r3) >= vecLen {
			dotProduct4AVXVNNI(results[i:i+4], r0, r1, r2, r3, vec)
		} else {
			dotProductBatchRows(results[i:i+4], rows[i:i+4], vec)
		}
		i += 4
	}
	dotProductBatchRows(results[i:], rows[i:], vec)
}

func minMaxI8(a []int8) (minVal, maxVal int8) {
	if hasAVX2 && len(a) >= blockMinMax {
		return minMaxAVX2(a)
	}
	return minMaxGo(a)
}

func minI8(dst, a, b []int8) {
	if hasAVX2 && len(dst) >= blockSat32 {
		minAVX2(dst, a, b)
		return
	}
	minGo(dst, a, b)
}

func maxI8(dst, a, b []int8) {
	if hasAVX2 && len(dst) >= blockSat32 {
		maxAVX2(dst, a, b)
		return
	}
	maxGo(dst, a, b)
}

func clampElemI8(dst, src []int8, lo, hi int8) {
	if hasAVX2 && len(dst) >= blockSat32 {
		clampAVX2(dst, src, lo, hi)
		return
	}
	clampGo(dst, src, lo, hi)
}

func absI8(dst, a []int8) {
	if hasAVX2 && len(dst) >= blockSat32 {
		absAVX2(dst, a)
		return
	}
	absGo(dst, a)
}

func negI8(dst, a []int8) {
	if hasAVX2 && len(dst) >= blockSat32 {
		negAVX2(dst, a)
		return
	}
	negGo(dst, a)
}

func maxAbsI8(a []int8) int {
	if hasAVX2 && len(a) >= blockMinMax {
		return maxAbsAVX2(a)
	}
	return maxAbsGo(a)
}

func absDiffI8(dst, a, b []int8) {
	if hasAVX2 && len(dst) >= blockSat32 {
		absDiffAVX2(dst, a, b)
		return
	}
	absDiffGo(dst, a, b)
}

func addScalarSatI8(dst, a []int8, s int8) {
	if hasAVX2 && len(dst) >= blockSat32 {
		addScalarSatAVX2(dst, a, s)
		return
	}
	addScalarSatGo(dst, a, s)
}

func sumAbsI8(a []int8) int32 {
	if hasAVX2 && len(a) >= blockSat32 {
		return sumAbsAVX2(a)
	}
	return sumAbsGo(a)
}

func sadI8(a, b []int8) int32 {
	if hasAVX2 && len(a) >= blockSat32 {
		return sadAVX2(a, b)
	}
	return sadGo(a, b)
}

func subScalarSatI8(dst, a []int8, s int8) {
	if hasAVX2 && len(dst) >= blockSat32 {
		subScalarSatAVX2(dst, a, s)
		return
	}
	subScalarSatGo(dst, a, s)
}

//go:noescape
func addSatAVX2(dst, a, b []int8)

//go:noescape
func subSatAVX2(dst, a, b []int8)

//go:noescape
func toI16AVX2(dst []int16, src []int8)

//go:noescape
func toI32AVX2(dst []int32, src []int8)

//go:noescape
func sumAVX2(a []int8) int32

//go:noescape
func dotAVX2(a, b []int8) int32

// dotProduct4AVX2 scores four rows (each at least len(vec) long) against vec,
// writing the four int32 dot products to res[0:4]. vec is expanded to int16 once
// per 16-byte block and reused across the four rows. res must have len >= 4.
//
//go:noescape
func dotProduct4AVX2(res []int32, r0, r1, r2, r3, vec []int8)

// dotProduct4AVXVNNI scores four rows (each at least len(vec) long) against vec
// like dotProduct4AVX2, but fuses the widen-multiply-accumulate with VPDPBUSD
// (AVX-VNNI). VPDPBUSD is unsigned x signed, so each row byte is biased to
// unsigned (row XOR 0x80 = row + 128) and the shared 128*sum(vec) is subtracted
// back off per row; sum(vec) is accumulated in-kernel over the same VNNI prefix.
// res must have len >= 4.
//
//go:noescape
func dotProduct4AVXVNNI(res []int32, r0, r1, r2, r3, vec []int8)

//go:noescape
func minMaxAVX2(a []int8) (minVal, maxVal int8)

//go:noescape
func minAVX2(dst, a, b []int8)

//go:noescape
func maxAVX2(dst, a, b []int8)

//go:noescape
func clampAVX2(dst, src []int8, lo, hi int8)

//go:noescape
func absAVX2(dst, a []int8)

//go:noescape
func negAVX2(dst, a []int8)

//go:noescape
func maxAbsAVX2(a []int8) int

//go:noescape
func absDiffAVX2(dst, a, b []int8)

//go:noescape
func addScalarSatAVX2(dst, a []int8, s int8)

//go:noescape
func subScalarSatAVX2(dst, a []int8, s int8)

//go:noescape
func sumAbsAVX2(a []int8) int32

//go:noescape
func sadAVX2(a, b []int8) int32

// Quantization dispatch (Part of #132). The AVX2 kernels process 16/8/8 lanes
// per iteration; shorter slices use the pure-Go reference. Requantize also
// routes out-of-contract inputs (multiplier == math.MinInt32, or shift outside
// [-31, 30]) to Go, since the kernel assumes the sane domain.
const (
	blockQuant   = 16 // quantizeAVX2 packs 16 int8 per iteration
	blockDequant = 8  // dequantizeAVX2 widens 8 int8 per iteration
	blockRequant = 8  // requantizeAVX2 processes 8 int32 per iteration
)

func quantizeI8(dst []int8, src []float32, scale float32, zeroPoint int8) {
	if hasAVX2 && len(dst) >= blockQuant {
		quantizeAVX2(dst, src, scale, zeroPoint)
		return
	}
	quantizeGo(dst, src, scale, zeroPoint)
}

func dequantizeI8(dst []float32, src []int8, scale float32, zeroPoint int8) {
	if hasAVX2 && len(dst) >= blockDequant {
		dequantizeAVX2(dst, src, scale, zeroPoint)
		return
	}
	dequantizeGo(dst, src, scale, zeroPoint)
}

func requantizeI8(dst []int8, acc []int32, multiplier int32, shift int, zeroPoint int8) {
	if hasAVX2 && !requantizeOutOfContract(multiplier, shift) && len(dst) >= blockRequant {
		requantizeAVX2(dst, acc, multiplier, shift, zeroPoint)
		return
	}
	requantizeGo(dst, acc, multiplier, shift, zeroPoint)
}

//go:noescape
func quantizeAVX2(dst []int8, src []float32, scale float32, zeroPoint int8)

//go:noescape
func dequantizeAVX2(dst []float32, src []int8, scale float32, zeroPoint int8)

//go:noescape
func requantizeAVX2(dst []int8, acc []int32, multiplier int32, shift int, zeroPoint int8)
