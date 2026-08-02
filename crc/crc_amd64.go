//go:build amd64

package crc

import "github.com/tphakala/simd/cpu"

// minFoldBytes is the smallest input routed through the PCLMULQDQ fold. Below it
// the scalar slice-by-16 loop wins, because the fold pays a fixed scalar
// reduction over the 16 accumulator bytes regardless of input length. Tuned from
// the benchmarks; FLAC frames are kilobytes, so the hot path always folds.
const minFoldBytes = 64

// hasFoldISA reports whether every instruction crc16FoldBlocks uses is present,
// which is more than its name suggests. The kernel folds with PCLMULQDQ, but it
// also byte-swaps with PSHUFB (SSSE3) and moves the accumulator halves with
// PINSRQ/PEXTRQ (SSE4.1), and PCLMULQDQ implies neither: they are separate CPUID
// bits, exactly as FMA3 is separate from AVX (#201).
//
// No shipping part has PCLMULQDQ without SSE4.1 (Westmere onward, Bulldozer
// onward and Jaguar onward all carry both), so this is a latent mismatch rather
// than an observed fault. Stating the real requirement costs one CPUID read at
// init and removes the need to know that history to audit the guard. SSE4.1
// implies SSSE3, so testing it covers PSHUFB too.
var hasFoldISA = cpu.X86.PCLMULQDQ && cpu.X86.SSE41

// foldSupported reports whether the carry-less-multiply kernel is active.
func foldSupported() bool { return hasFoldISA }

// checksum16 folds the bulk of p with PCLMULQDQ when available, then reduces the
// folded accumulator plus the trailing tail with the scalar path.
func checksum16(p []byte) uint16 {
	if hasFoldISA && len(p) >= minFoldBytes {
		full := len(p) &^ (blockBytes - 1)
		var acc [2]uint64
		crc16FoldBlocks(&acc, p[:full])
		return checksum16FromAcc(acc[0], acc[1], p[full:])
	}
	return checksum16Go(p)
}

//go:noescape
func crc16FoldBlocks(acc *[2]uint64, p []byte)
