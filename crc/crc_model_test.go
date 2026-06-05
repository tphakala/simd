package crc

import (
	"encoding/binary"
	"math/rand"
	"testing"
)

// This file holds the pure-Go reference model for the carry-less-multiply fold.
// It is the executable specification the PCLMULQDQ and PMULL kernels must match:
// the assembly is a 1:1 translation of crc16FoldGo, and the per-arch tests pin
// the assembly output against this model. Keeping the model in test code means
// production builds never carry the slow bit-serial clmul emulation.

// xnModPRef computes x^n mod P for the FLAC CRC-16 polynomial (P = 0x18005,
// degree 16); the result has degree < 16.
func xnModPRef(n int) uint16 {
	r := uint32(1)
	for range n {
		r <<= 1
		if r&0x10000 != 0 {
			r ^= 0x18005
		}
	}
	return uint16(r)
}

// clmul64 is a carry-less (GF(2)) multiply of two 64-bit polynomials, returning
// the 128-bit product as (hi, lo). It mirrors one PCLMULQDQ / PMULL lane.
func clmul64(a, b uint64) (hi, lo uint64) {
	for i := range 64 {
		if (b>>uint(i))&1 != 0 {
			lo ^= a << uint(i)
			hi ^= a >> uint(64-i) // i==0 -> a>>64 == 0 per Go shift semantics
		}
	}
	return hi, lo
}

// crc16FoldGo folds every full 16-byte block of p into the 128-bit accumulator
// acc (acc[0]=high bits 127..64, acc[1]=low bits 63..0), updating it in place.
// Each block computes acc = (acc*x^128 mod P) ^ block, which keeps acc congruent
// to the processed prefix modulo P. This is the reference for crc16FoldBlocks.
func crc16FoldGo(acc *[2]uint64, p []byte) {
	accHi, accLo := acc[0], acc[1]
	for len(p) >= 16 {
		dHi := binary.BigEndian.Uint64(p[0:8])
		dLo := binary.BigEndian.Uint64(p[8:16])
		f1h, f1l := clmul64(accHi, crc16K1) // high half * x^192 mod P
		f2h, f2l := clmul64(accLo, crc16K2) // low half  * x^128 mod P
		accHi = f1h ^ f2h ^ dHi
		accLo = f1l ^ f2l ^ dLo
		p = p[16:]
	}
	acc[0], acc[1] = accHi, accLo
}

// foldModelChecksum is the full CRC-16 via the fold model plus scalar reduction,
// mirroring exactly what the SIMD dispatch does with the assembly kernel.
func foldModelChecksum(p []byte) uint16 {
	full := len(p) &^ 15
	var acc [2]uint64
	crc16FoldGo(&acc, p[:full])
	return checksum16FromAcc(acc[0], acc[1], p[full:])
}

// TestFoldConstantsAreXnModP pins the embedded fold constants to their algebraic
// definitions so the Go model and the assembly literals cannot drift.
func TestFoldConstantsAreXnModP(t *testing.T) {
	if got := xnModPRef(192); got != crc16K1 {
		t.Errorf("crc16K1 = %#04x, want x^192 mod P = %#04x", uint16(crc16K1), got)
	}
	if got := xnModPRef(128); got != crc16K2 {
		t.Errorf("crc16K2 = %#04x, want x^128 mod P = %#04x", uint16(crc16K2), got)
	}
}

// TestFoldModelMatchesScalar validates the fold algorithm end-to-end against the
// independent scalar reference, independent of any assembly.
func TestFoldModelMatchesScalar(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	for n := 0; n <= 600; n++ {
		buf := make([]byte, n)
		for i := range buf {
			buf[i] = byte(r.Intn(256))
		}
		if got, want := foldModelChecksum(buf), refChecksum16(buf); got != want {
			t.Fatalf("n=%d: foldModel=%#04x want %#04x", n, got, want)
		}
	}
	for _, n := range []int{1024, 4096, 4097, 8192, 16384, 16385} {
		buf := make([]byte, n)
		for i := range buf {
			buf[i] = byte(r.Intn(256))
		}
		if got, want := foldModelChecksum(buf), refChecksum16(buf); got != want {
			t.Fatalf("n=%d: foldModel=%#04x want %#04x", n, got, want)
		}
	}
}
