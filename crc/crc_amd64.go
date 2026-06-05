//go:build amd64

package crc

import "github.com/tphakala/simd/cpu"

// minFoldBytes is the smallest input routed through the PCLMULQDQ fold. Below it
// the scalar slice-by-16 loop wins, because the fold pays a fixed scalar
// reduction over the 16 accumulator bytes regardless of input length. Tuned from
// the benchmarks; FLAC frames are kilobytes, so the hot path always folds.
const minFoldBytes = 64

var hasPCLMULQDQ = cpu.X86.PCLMULQDQ

// foldSupported reports whether the carry-less-multiply kernel is active.
func foldSupported() bool { return hasPCLMULQDQ }

// checksum16 folds the bulk of p with PCLMULQDQ when available, then reduces the
// folded accumulator plus the trailing tail with the scalar path.
func checksum16(p []byte) uint16 {
	if hasPCLMULQDQ && len(p) >= minFoldBytes {
		full := len(p) &^ (blockBytes - 1)
		var acc [2]uint64
		crc16FoldBlocks(&acc, p[:full])
		return checksum16FromAcc(acc[0], acc[1], p[full:])
	}
	return checksum16Go(p)
}

//go:noescape
func crc16FoldBlocks(acc *[2]uint64, p []byte)
