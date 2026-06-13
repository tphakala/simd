// Package crc provides SIMD-accelerated cyclic redundancy checks.
//
// Checksum16 computes a CRC-16 over the generator polynomial x^16+x^15+x^2+1 =
// 0x8005 with init 0, MSB-first and no input/output reflection. The 0x8005
// polynomial is shared by many CRC-16 variants (ARC, Modbus, USB, and others,
// which differ in reflection, init, and xorout); this MSB-first, unreflected
// parameterization is the one a FLAC encoder computes over every coded frame,
// but it is not FLAC-specific. Checksum16 folds the bulk of the buffer 16 bytes
// at a time using a carry-less-multiply kernel (PCLMULQDQ on amd64, PMULL on
// arm64) and falls back to a slice-by-16 table loop on architectures without a
// polynomial-multiply instruction.
//
// All functions are bit-identical to the scalar reference and allocation-free.
//
// Thread Safety: All functions are safe for concurrent use.
package crc

import "encoding/binary"

// poly16 is the CRC-16 generator polynomial x^16+x^15+x^2+1 (0x8005) without its
// implicit x^16 term (the full 17-bit form is 0x18005).
const poly16 = 0x8005

const (
	// bitsPerByte is the MSB-first byte stride: folding one byte into the CRC
	// shifts the 16-bit register left by one byte and indexes on the byte that
	// falls out of the top.
	bitsPerByte = 8
	// blockBytes is the carry-less-multiply fold stride and the width of the
	// 128-bit accumulator in bytes.
	blockBytes = 16
)

// Fold constants for the carry-less-multiply kernels. Folding a 128-bit
// accumulator forward by one 16-byte block computes acc*x^128 mod P, which
// splits into the two 64-bit halves of acc multiplied (carry-less) by:
//
//	crc16K1 = x^192 mod P  (applied to the high half, bits 127..64)
//	crc16K2 = x^128 mod P  (applied to the low half,  bits  63..0)
//
// Both reduce to degree < 16, so each carry-less product stays below 80 bits.
// The assembly kernels embed these same literals; crc_model_test.go asserts
// they equal x^192 mod P and x^128 mod P so the two never drift apart.
const (
	crc16K1 = 0x1666 // x^192 mod 0x18005
	crc16K2 = 0x0106 // x^128 mod 0x18005
)

var (
	table16  [256]uint16
	table16x [16][256]uint16
)

func init() {
	for i := range 256 {
		c := uint16(i) << bitsPerByte
		for range bitsPerByte {
			if c&0x8000 != 0 {
				c = (c << 1) ^ poly16
			} else {
				c <<= 1
			}
		}
		table16[i] = c
	}

	// Slice-by-16 derived tables: table16x[0] == table16; table16x[n][b] is the
	// CRC-16 of byte b followed by n zero bytes, so sixteen input bytes fold in
	// one step. MSB-first, no reflection, matching update16.
	for b := range 256 {
		table16x[0][b] = table16[b]
	}
	for n := 1; n < 16; n++ {
		for b := range 256 {
			prev := table16x[n-1][b]
			table16x[n][b] = (prev << bitsPerByte) ^ table16[byte(prev>>bitsPerByte)]
		}
	}
}

// update16 folds one byte into a running CRC-16.
func update16(c uint16, b byte) uint16 {
	return (c << bitsPerByte) ^ table16[byte(c>>bitsPerByte)^b]
}

// Checksum16 returns the CRC-16 of p (polynomial 0x8005, init 0, MSB-first,
// no reflection). It is bit-identical to a byte-at-a-time reference loop and
// allocation-free.
func Checksum16(p []byte) uint16 {
	return checksum16(p)
}

// checksum16Go is the pure-Go slice-by-16 reference. It is the fallback on
// architectures without a carry-less-multiply instruction and the final
// reduction step shared by the SIMD kernels.
func checksum16Go(p []byte) uint16 {
	var c uint16
	for len(p) >= blockBytes {
		c = table16x[15][byte(c>>bitsPerByte)^p[0]] ^
			table16x[14][byte(c)^p[1]] ^
			table16x[13][p[2]] ^
			table16x[12][p[3]] ^
			table16x[11][p[4]] ^
			table16x[10][p[5]] ^
			table16x[9][p[6]] ^
			table16x[8][p[7]] ^
			table16x[7][p[8]] ^
			table16x[6][p[9]] ^
			table16x[5][p[10]] ^
			table16x[4][p[11]] ^
			table16x[3][p[12]] ^
			table16x[2][p[13]] ^
			table16x[1][p[14]] ^
			table16x[0][p[15]]
		p = p[16:]
	}
	for _, b := range p {
		c = update16(c, b)
	}
	return c
}

// checksum16FromAcc finishes a folded accumulator. After the fold loop, the
// 128-bit accumulator (accHi:accLo, most-significant byte first) is congruent
// to the already-processed prefix modulo the CRC polynomial, so the final CRC
// is the scalar CRC-16 of the accumulator's 16 bytes followed by the unfolded
// tail (fewer than 16 bytes). The fixed 31-byte stack buffer keeps it
// allocation-free.
func checksum16FromAcc(accHi, accLo uint64, tail []byte) uint16 {
	var b [2*blockBytes - 1]byte // 16 accumulator bytes + at most 15 tail bytes
	binary.BigEndian.PutUint64(b[0:8], accHi)
	binary.BigEndian.PutUint64(b[8:blockBytes], accLo)
	n := blockBytes + copy(b[blockBytes:], tail)
	return checksum16Go(b[:n])
}
