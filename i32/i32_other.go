//go:build !amd64 && !arm64

package i32

func interleave2I32(dst, a, b []int32)   { interleave2Go(dst, a, b) }
func deinterleave2I32(a, b, src []int32) { deinterleave2Go(a, b, src) }
