//go:build arm64

#include "textflag.h"

// FLAC CRC-16 (polynomial 0x8005, MSB-first, no reflection) carry-less-multiply
// fold for ARM64 (PMULL / FEAT_PMULL).
//
// crc16FoldBlocks folds every full 16-byte block of p into the 128-bit
// accumulator *acc (acc[0] = high bits 127..64, acc[1] = low bits 63..0). Each
// block computes acc = (acc * x^128 mod P) ^ block via two carry-less products:
//
//	fold = clmul(acc_lo, x^128 mod P) ^ clmul(acc_hi, x^192 mod P)
//	acc  = fold ^ block
//
// which keeps acc congruent to the processed prefix modulo P; the Go caller
// finishes with the scalar reduction. This is a 1:1 translation of crc16FoldGo
// in crc_model_test.go and is pinned against it by the parity tests.
//
// The internal vector layout keeps logical bit i in vector bit i, so the high 64
// bits live in lane d[1] (where PMULL2 reads) and the low 64 bits in lane d[0]
// (where PMULL reads). Input blocks are byte-reversed (VREV64 + VEXT) so the
// first, most significant byte lands in the high lane, matching the MSB-first
// CRC. Carry-less multiply and EOR are commutative, so operand order is free.
//
// Every instruction here is a native Go arm64 mnemonic, so there are no
// hand-encoded WORD directives for asmcheck to validate.

// foldConst holds the fold constants: lane d[0] = K2 (x^128 mod P) for the low
// half, lane d[1] = K1 (x^192 mod P) for the high half.
DATA foldConst<>+0(SB)/8, $0x0106
DATA foldConst<>+8(SB)/8, $0x1666
GLOBL foldConst<>(SB), RODATA|NOPTR, $16

// func crc16FoldBlocks(acc *[2]uint64, p []byte)
TEXT ·crc16FoldBlocks(SB), NOSPLIT, $0-32
	MOVD acc+0(FP), R0     // R0 = &acc
	MOVD p_base+8(FP), R1  // R1 = &p[0]
	MOVD p_len+16(FP), R2  // R2 = len(p)
	LSR  $4, R2, R2        // R2 = number of full 16-byte blocks

	// Load accumulator. VLD1 fills d[0] from acc[0] (=accHi) and d[1] from
	// acc[1] (=accLo); swap the lanes so the high bits sit in d[1].
	VLD1 (R0), [V0.D2]
	VEXT $8, V0.B16, V0.B16, V0.B16 // d[0]=accLo, d[1]=accHi

	// Load fold constants: V1.d[0]=K2, V1.d[1]=K1.
	MOVD $foldConst<>(SB), R3
	VLD1 (R3), [V1.D2]

	CBZ R2, done

loop:
	VLD1.P 16(R1), [V2.B16]         // V2 = next 16 bytes (little-endian)
	VREV64 V2.B16, V2.B16           // reverse bytes within each doubleword
	VEXT   $8, V2.B16, V2.B16, V2.B16 // swap doublewords -> full 16-byte reverse

	VPMULL  V1.D1, V0.D1, V3.Q1     // V3 = clmul(acc_lo, K2)
	VPMULL2 V1.D2, V0.D2, V4.Q1     // V4 = clmul(acc_hi, K1)
	VEOR    V4.B16, V3.B16, V0.B16  // V0 = clmul(acc_lo,K2) ^ clmul(acc_hi,K1)
	VEOR    V2.B16, V0.B16, V0.B16  // V0 = fold ^ block  ->  new accumulator

	SUB  $1, R2
	CBNZ R2, loop

done:
	// Swap lanes back to memory order (d[0]=accHi, d[1]=accLo) and store.
	VEXT $8, V0.B16, V0.B16, V0.B16
	VST1 [V0.D2], (R0)
	RET
