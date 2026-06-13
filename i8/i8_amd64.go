//go:build amd64

package i8

import "github.com/tphakala/simd/cpu"

// The int8 kernels operate on 256-bit integer lanes (VPADDSB/VPSUBSB/VPMINSB/
// VPMAXSB/VPMOVSXB*/VPMADDWD), which require AVX2. They gate on AVX2 explicitly
// and fall back to the pure-Go reference on the (now rare) AVX-less baseline and
// for slices shorter than one vector block.
var hasAVX2 = cpu.X86.AVX2

// Per-kernel minimum element counts: one full vector iteration's worth of int8
// inputs. Shorter slices use the pure-Go reference.
const (
	blockSat32   = 32 // VPADDSB/VPSUBSB process 32 bytes per iteration
	blockMinMax  = 32 // VPMINSB/VPMAXSB process 32 bytes per iteration
	blockReduce  = 16 // Sum/DotProduct widen 16 bytes per iteration (VPMOVSXBW)
	blockWiden16 = 16 // ToInt16 widens 16 bytes per iteration (VPMOVSXBW)
	blockWiden32 = 8  // ToInt32 widens 8 bytes per iteration (VPMOVSXBD)
)

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

func minMaxI8(a []int8) (minVal, maxVal int8) {
	if hasAVX2 && len(a) >= blockMinMax {
		return minMaxAVX2(a)
	}
	return minMaxGo(a)
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

//go:noescape
func minMaxAVX2(a []int8) (minVal, maxVal int8)
