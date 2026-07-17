package i16

// Q15 fixed-point multiply.
//
// Kernel selection constraint: NEON's single-instruction SQRDMULH computes the
// same rounding product but SATURATES (-32768 * -32768) to 32767, so it can
// never implement this op. The NEON kernel widens with SMULL, rounds with
// SRSHR and narrows with XTN, which wraps; amd64's VPMULHRSW computes the
// identical ((a*b >> 14) + 1) >> 1 and wraps that pair too, so it is safe.
// Do not "optimize" either kernel toward the saturating instruction.

// MulQ15 writes the rounding Q15 fixed-point product
//
//	dst[i] = int16((int32(a[i])*int32(b[i]) + 1<<14) >> 15)
//
// for i in [0, n), n = min(len(dst), len(a), len(b)). This is the rounding
// form (libopus MULT16_16_P15), not the truncating MULT16_16_Q15: the 1<<14
// term rounds to nearest, so MulQ15(1, 16384) = 1 where truncation would
// give 0.
//
// The int16 conversion wraps rather than saturates. Exactly one input pair
// produces a value outside int16 range: MulQ15(-32768, -32768) is +32768 and
// wraps to -32768. That wrap is a guarantee the SIMD paths reproduce
// bit-exactly. Any trailing capacity in dst is left untouched.
//
// dst may overlap a or b only if it starts at the same address.
func MulQ15(dst, a, b []int16) {
	n := min(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	mulQ15I16(dst[:n], a[:n], b[:n])
}
