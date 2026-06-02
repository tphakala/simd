package i32

// Fixed-predictor encode differences for the FLAC codec.
//
// FLAC's fixed predictors of order 1..4 predict each sample from the previous
// ones with fixed integer coefficients; the residual the encoder Rice-codes is
// the order-K forward finite difference of the samples. DiffK writes that
// residual into dst:
//
//	dst[i] = src[i]                            for i < K  (verbatim warm-up)
//	dst[n] = sum_j (-1)^j * C(K,j) * src[n-j]  for n >= K (the residual)
//
// so dst is laid out as [K warm-up samples | residuals], exactly the order a
// FLAC subframe stores. Each function clamps to n = min(len(dst), len(src)),
// uses int32 wraparound (the residual can exceed the source range), and writes
// only into the caller's dst. Inputs shorter than K hold warm-up only.

// Diff1 writes the order-1 residual: dst[0]=src[0], dst[n]=src[n]-src[n-1].
func Diff1(dst, src []int32) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	diff1I32(dst[:n], src[:n])
}

// Diff2 writes the order-2 residual: dst[n]=src[n]-2src[n-1]+src[n-2].
func Diff2(dst, src []int32) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	diff2I32(dst[:n], src[:n])
}

// Diff3 writes the order-3 residual: dst[n]=src[n]-3src[n-1]+3src[n-2]-src[n-3].
func Diff3(dst, src []int32) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	diff3I32(dst[:n], src[:n])
}

// Diff4 writes the order-4 residual:
// dst[n]=src[n]-4src[n-1]+6src[n-2]-4src[n-3]+src[n-4].
func Diff4(dst, src []int32) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	diff4I32(dst[:n], src[:n])
}
