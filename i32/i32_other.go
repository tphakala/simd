//go:build !amd64 && !arm64

package i32

func interleave2I32(dst, a, b []int32)   { interleave2Go(dst, a, b) }
func deinterleave2I32(a, b, src []int32) { deinterleave2Go(a, b, src) }

func addI32(dst, a, b []int32) { addGo(dst, a, b) }
func subI32(dst, a, b []int32) { subGo(dst, a, b) }
func absI32(dst, a []int32)    { absGo(dst, a) }
func sumI32(a []int32) int32   { return sumGo(a) }

func minMaxI32(res []int32) (minVal, maxVal int32) { return minMaxGo(res) }

func negWhereNegI32(dst, mag []int32, sign []float32) { negWhereNegGo(dst, mag, sign) }

func scaleQ31I32(dst, a []int32, k int32) { scaleQ31Go(dst, a, k) }
func scaleQ15I32(dst, a []int32, k int16) { scaleQ15Go(dst, a, k) }
