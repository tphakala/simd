//go:build amd64 || arm64

package crc

import (
	"math/rand"
	"testing"
)

// TestCRC16FoldBlocksMatchesModel pins the carry-less-multiply kernel (PCLMULQDQ
// on amd64, PMULL on arm64) against the pure-Go fold model. A random nonzero
// starting accumulator exercises the loop-carried fold dependency, not just the
// acc==0 first block. It runs only when the kernel is actually wired in, i.e.
// when the CPU has the polynomial-multiply instruction.
func TestCRC16FoldBlocksMatchesModel(t *testing.T) {
	if !foldSupported() {
		t.Skip("CPU has no carry-less-multiply instruction")
	}
	r := rand.New(rand.NewSource(99))
	for _, blocks := range []int{0, 1, 2, 3, 4, 5, 8, 16, 64, 256, 1024} {
		buf := make([]byte, blocks*16)
		for i := range buf {
			buf[i] = byte(r.Intn(256))
		}
		var model, asm [2]uint64
		model[0], model[1] = r.Uint64(), r.Uint64()
		asm = model // identical nonzero start
		crc16FoldGo(&model, buf)
		crc16FoldBlocks(&asm, buf)
		if asm != model {
			t.Fatalf("blocks=%d: asm={%#016x,%#016x} model={%#016x,%#016x}",
				blocks, asm[0], asm[1], model[0], model[1])
		}
	}
}
