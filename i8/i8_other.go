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

func minI8(dst, a, b []int8)                 { minGo(dst, a, b) }
func maxI8(dst, a, b []int8)                 { maxGo(dst, a, b) }
func clampElemI8(dst, s []int8, lo, hi int8) { clampGo(dst, s, lo, hi) }
func absI8(dst, a []int8)                    { absGo(dst, a) }
func negI8(dst, a []int8)                    { negGo(dst, a) }
func maxAbsI8(a []int8) int                  { return maxAbsGo(a) }
func absDiffI8(dst, a, b []int8)             { absDiffGo(dst, a, b) }

func addScalarSatI8(dst, a []int8, s int8) { addScalarSatGo(dst, a, s) }
func subScalarSatI8(dst, a []int8, s int8) { subScalarSatGo(dst, a, s) }
func sumAbsI8(a []int8) int32              { return sumAbsGo(a) }
func sadI8(a, b []int8) int32              { return sadGo(a, b) }
