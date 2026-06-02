package i32

// Fixed-predictor decode restoration for the FLAC codec.
//
// RestoreK is the exact inverse of DiffK (see fixed.go): given a subframe laid
// out as [K warm-up samples | residuals], it reconstructs the original samples:
//
//	dst[i] = src[i]                              for i < K  (verbatim warm-up)
//	dst[n] = src[n] - sum_{j>=1} c[j]*dst[n-j]   for n >= K (the recurrence)
//
// where c are the signed binomials (-1)^j*C(K,j). Each function clamps to
// n = min(len(dst), len(src)), uses int32 wraparound, and writes only into the
// caller's dst. Inputs of K or fewer samples are all warm-up (copied verbatim).
//
// Implementation: the order-K fixed predictor is the K-th forward difference, so
// its inverse is K cumulative-sum passes. Over int32 (the ring Z/2^32) the
// forward-difference operator and the cumulative-sum operator are exact
// inverses, so K passes of the SIMD cumsumI32 kernel reproduce the serial
// recurrence bit-for-bit (cross-checked against restoreGo in the tests). The
// only wrinkle is the warm-up: DiffK stores the first K samples verbatim rather
// than differenced, so before integrating we rewrite that prefix into the K-th
// difference form. That fix-up is causal (it reads only the prefix), O(K^2) with
// K <= 4, and never touches the residual region.

// Restore1 inverts Diff1: dst[0]=src[0], dst[n]=src[n]+dst[n-1].
func Restore1(dst, src []int32) { restoreI32(dst, src, fixedCoeffs1) }

// Restore2 inverts Diff2: dst[n]=src[n]+2*dst[n-1]-dst[n-2].
func Restore2(dst, src []int32) { restoreI32(dst, src, fixedCoeffs2) }

// Restore3 inverts Diff3: dst[n]=src[n]+3*dst[n-1]-3*dst[n-2]+dst[n-3].
func Restore3(dst, src []int32) { restoreI32(dst, src, fixedCoeffs3) }

// Restore4 inverts Diff4: dst[n]=src[n]+4*dst[n-1]-6*dst[n-2]+4*dst[n-3]-dst[n-4].
func Restore4(dst, src []int32) { restoreI32(dst, src, fixedCoeffs4) }

// restoreI32 reconstructs samples using the cumulative-sum decomposition. It
// takes the same fixedCoeffsK table diffGo uses and derives the predictor order
// from it (order = len(c)-1), so Restore and Diff stay locked to matching orders
// and are provably inverses; the cumulative-sum path needs only that order, not
// the coefficient values. cumsumI32 dispatches to the SIMD kernel (with a
// pure-Go fallback), so this one orchestration covers every architecture.
func restoreI32(dst, src, c []int32) {
	order := len(c) - 1
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	copy(dst[:n], src[:n])
	if n <= order {
		// Fewer samples than the predictor order: all warm-up, no residual to
		// integrate. The verbatim copy above is already the answer.
		return
	}
	// Rewrite the verbatim warm-up dst[0:order] into its K-th forward-difference
	// form by applying the first difference 'order' times to the prefix. Each
	// pass holds index 0 and differences 1..order-1 in place (backward so a
	// value is read before it is overwritten), mirroring diffGo's boundary.
	for range order {
		for i := order - 1; i >= 1; i-- {
			dst[i] -= dst[i-1]
		}
	}
	// 'order' cumulative-sum passes invert the order-K forward difference.
	for range order {
		cumsumI32(dst[:n])
	}
}
