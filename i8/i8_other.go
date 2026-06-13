//go:build !amd64 && !arm64

package i8

// Pure-Go dispatch for architectures without a SIMD backend.

func addSatI8(dst, a, b []int8)   { addSatGo(dst, a, b) }
func subSatI8(dst, a, b []int8)   { subSatGo(dst, a, b) }
func toI16(dst []int16, s []int8) { toI16Go(dst, s) }
func toI32(dst []int32, s []int8) { toI32Go(dst, s) }
func sumI8(a []int8) int32        { return sumGo(a) }
func dotI8(a, b []int8) int32     { return dotGo(a, b) }

func minMaxI8(a []int8) (minVal, maxVal int8) { return minMaxGo(a) }
