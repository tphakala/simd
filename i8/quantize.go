package i8

// Per-tensor affine int8 quantization (Part of #132).
//
// These three functions cover the fixed-point boundary of a quantized pipeline:
// Quantize maps float32 activations to int8, Dequantize maps them back, and
// Requantize rescales an int32 accumulator (the output of a quantized matmul or
// convolution) to int8 with a Q31 multiplier and a shift. The signed,
// per-tensor affine convention matches ONNX / PyTorch / TFLite: a real value r
// and its quantized value q relate by q = round(r/scale) + zeroPoint and
// r = (q - zeroPoint) * scale.
//
// dst and src always have distinct element types, so in safe Go they cannot
// alias; the kernels rely on that to reprocess an overlapping final block
// instead of a scalar tail. Callers must not construct aliasing views through
// unsafe.

// Quantize writes dst[i] = clamp(rne(src[i]/scale) + zeroPoint, -128, 127) for i
// in [0, n), n = min(len(dst), len(src)). The divide is a true IEEE-754 float32
// division (not a reciprocal multiply) and rne is round-half-to-even, so the
// result is bit-identical across the Go, AVX2 and NEON paths and matches the
// documented formula literally.
//
// Saturation follows from the clamp: +Inf maps to 127, -Inf to -128. NaN maps to
// zeroPoint. scale is expected to be finite and positive; that is the caller's
// contract, not enforced here. Any trailing capacity in dst is left untouched.
func Quantize(dst []int8, src []float32, scale float32, zeroPoint int8) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	quantizeI8(dst[:n], src[:n], scale, zeroPoint)
}

// Dequantize writes dst[i] = float32(int32(src[i]) - int32(zeroPoint)) * scale
// for i in [0, n), n = min(len(dst), len(src)). The subtraction is exact and the
// int-to-float conversion is exact (the difference is at most 255 in magnitude),
// so the single multiply is the only rounding and the result is bit-identical
// across the Go, AVX2 and NEON paths. Any trailing capacity in dst is left
// untouched.
func Dequantize(dst []float32, src []int8, scale float32, zeroPoint int8) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	dequantizeI8(dst[:n], src[:n], scale, zeroPoint)
}

// Requantize rescales an int32 accumulator to int8 using the gemmlowp / TFLite
// double-rounding epilogue, writing dst[i] for i in [0, n),
// n = min(len(dst), len(acc)). multiplier is a Q31 fixed-point value (normally
// in [2^30, 2^31)) and shift is a power-of-two exponent. Per element:
//
//	x = acc[i] << max(shift, 0)                       (wrapping int32)
//	y = SaturatingRoundingDoublingHighMul(x, multiplier)
//	z = RoundingDivideByPOT(y, max(-shift, 0))        (ties away from zero)
//	dst[i] = clamp(z + zeroPoint, -128, 127)
//
// Inputs outside the SIMD kernels' domain (multiplier == math.MinInt32, or shift
// outside [-31, 30]) are handled by the full-width Go reference. Any trailing
// capacity in dst is left untouched.
func Requantize(dst []int8, acc []int32, multiplier int32, shift int, zeroPoint int8) {
	n := min(len(dst), len(acc))
	if n == 0 {
		return
	}
	requantizeI8(dst[:n], acc[:n], multiplier, shift, zeroPoint)
}
