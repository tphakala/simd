package i32

// In-place radix-2 butterfly on two int32 slices.
//
// Butterfly is the Hadamard/Haar radix-2 step of a fast Walsh-Hadamard
// transform (FWHT): for each lane it replaces the pair (lo, hi) with their sum
// and difference, (lo+hi, lo-hi). Both the add and the subtract wrap in int32
// rather than saturating, so the SIMD and pure-Go paths are bit-identical across
// the full int32 range and there is no relaxed tier. It is the haar1 combine
// applied after a Q31 scale of both lanes: lo, hi = lo+hi, lo-hi.

// Butterfly writes lo[i], hi[i] = lo[i]+hi[i], lo[i]-hi[i] for i in [0, n),
// n = min(len(lo), len(hi)). The sum and difference are two's-complement
// wrapping int32 operations (a+b and a-b modulo 2^32), matching a VPADDD/VPSUBD
// or ADD/SUB .4S lane exactly, so a sum that overflows or a difference that
// underflows wraps rather than saturating. Any trailing capacity past n in
// either slice is left untouched.
//
// Each lane reads both lo[i] and hi[i] before writing either, and iteration is
// forward, so the operation is well defined in place lane by lane. lo and hi
// must NOT overlap each other, though: the SIMD kernels load a whole block of
// both lo and hi before storing either block, so an overlapping region would
// clobber input lanes a later iteration has not yet read. Pass two distinct slices.
func Butterfly(lo, hi []int32) {
	n := min(len(lo), len(hi))
	if n == 0 {
		return
	}
	butterflyI32(lo[:n], hi[:n])
}
