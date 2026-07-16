//go:build !amd64 && !arm64

package i16

func interleave2I16(dst, a, b []int16)   { interleave2Go(dst, a, b) }
func deinterleave2I16(a, b, src []int16) { deinterleave2Go(a, b, src) }
func dotI16(a, b []int16) int32          { return dotGo(a, b) }
func xcorrI16(dst []int32, x, y []int16) { xcorrGo(dst, x, y) }
