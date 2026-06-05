//go:build amd64

#include "textflag.h"

// FLAC CRC-16 (polynomial 0x8005, MSB-first, no reflection) carry-less-multiply
// fold for AMD64 (PCLMULQDQ).
//
// crc16FoldBlocks folds every full 16-byte block of p into the 128-bit
// accumulator *acc (acc[0] = high bits 127..64, acc[1] = low bits 63..0). Each
// block computes acc = (acc * x^128 mod P) ^ block via two carry-less products:
//
//	fold = clmul(acc_hi, x^192 mod P) ^ clmul(acc_lo, x^128 mod P)
//	acc  = fold ^ block
//
// which keeps acc congruent to the processed prefix modulo P; the Go caller
// finishes with the scalar reduction. This is a 1:1 translation of crc16FoldGo
// in crc_model_test.go and is pinned against it by the parity tests.
//
// A single accumulator means the loop is bound by PCLMULQDQ latency rather than
// throughput, but that still clears the slice-by-16 table loop several times
// over. The internal xmm layout keeps logical bit i in xmm bit i: lane0 holds
// the low 64 bits, lane1 the high 64 bits. Input blocks are byte-reversed with
// PSHUFB so the first (most significant) byte lands in the high lane, matching
// the MSB-first CRC.

// bswapMask reverses all 16 bytes of a block: out[i] = in[15-i].
DATA bswapMask<>+0(SB)/8, $0x08090a0b0c0d0e0f
DATA bswapMask<>+8(SB)/8, $0x0001020304050607
GLOBL bswapMask<>(SB), RODATA|NOPTR, $16

// func crc16FoldBlocks(acc *[2]uint64, p []byte)
TEXT ·crc16FoldBlocks(SB), NOSPLIT, $0-32
	MOVQ acc+0(FP), AX     // AX = &acc
	MOVQ p_base+8(FP), SI  // SI = &p[0]
	MOVQ p_len+16(FP), CX  // CX = len(p)
	SHRQ $4, CX            // CX = number of full 16-byte blocks

	// Load accumulator into X0: lane0 = accLo = acc[1], lane1 = accHi = acc[0].
	MOVQ   8(AX), X0
	MOVQ   0(AX), BX
	PINSRQ $1, BX, X0

	// Fold constants in low lanes: X1 = K1 (x^192 mod P), X2 = K2 (x^128 mod P).
	MOVQ $0x1666, BX
	MOVQ BX, X1
	MOVQ $0x0106, BX
	MOVQ BX, X2

	MOVOU bswapMask<>(SB), X5 // byte-reverse shuffle control

	TESTQ CX, CX
	JZ    done

loop:
	MOVOU  (SI), X3          // X3 = next 16 bytes (little-endian)
	PSHUFB X5, X3            // byte-reverse: first byte -> high lane (MSB-first)

	MOVOU     X0, X4         // X4 = acc copy for the high-half product
	PCLMULQDQ $0x01, X1, X4  // X4 = clmul(acc_hi, K1)  [dst hi qword, src lo qword]
	PCLMULQDQ $0x00, X2, X0  // X0 = clmul(acc_lo, K2)  [dst lo qword, src lo qword]
	PXOR      X4, X0         // X0 = clmul(acc_hi,K1) ^ clmul(acc_lo,K2)
	PXOR      X3, X0         // X0 = fold ^ block  ->  new accumulator

	ADDQ $16, SI
	DECQ CX
	JNZ  loop

done:
	// Store accumulator back: acc[0] = accHi = lane1, acc[1] = accLo = lane0.
	MOVQ   X0, 8(AX)
	PEXTRQ $1, X0, BX
	MOVQ   BX, 0(AX)
	RET
