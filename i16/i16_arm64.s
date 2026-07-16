//go:build arm64

#include "textflag.h"

// int16 (de)interleave on ARM64 (NEON / ASIMD).
//
// ZIP1/ZIP2/UZP1/UZP2 on .8H operands and the VLD1/VST1 .H8 loads/stores all
// move 16-bit lanes by bit pattern, so these kernels mirror the int32 .4S
// kernels in ../i32/i32_arm64.s with the lane width halved (8 lanes per 128-bit
// register instead of 4); only the scalar tails differ (16-bit MOVH). The
// ZIP/UZP instructions are hand-encoded as WORD because the Go assembler lacks
// mnemonics for them; the trailing comment is the decoded form and is
// cross-checked by asmcheck_test.go.

// func interleave2NEON(dst, a, b []int16)
// Interleaves: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], dst[3]=b[1], ...
TEXT ·interleave2NEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0    // dst pointer
    MOVD a_base+24(FP), R1     // a pointer
    MOVD a_len+32(FP), R3      // n = len(a)
    MOVD b_base+48(FP), R2     // b pointer

    // Process 8 pairs at a time
    LSR $3, R3, R4             // R4 = n / 8
    CBZ R4, interleave2_neon_remainder

interleave2_neon_loop8:
    VLD1.P 16(R1), [V0.H8]     // V0 = [a0, a1, a2, a3, a4, a5, a6, a7]
    VLD1.P 16(R2), [V1.H8]     // V1 = [b0, b1, b2, b3, b4, b5, b6, b7]
    WORD $0x4E413802           // ZIP1 V2.8H, V0.8H, V1.8H -> [a0,b0,a1,b1,a2,b2,a3,b3]
    WORD $0x4E417803           // ZIP2 V3.8H, V0.8H, V1.8H -> [a4,b4,a5,b5,a6,b6,a7,b7]
    VST1.P [V2.H8], 16(R0)     // Store [a0,b0,a1,b1,a2,b2,a3,b3]
    VST1.P [V3.H8], 16(R0)     // Store [a4,b4,a5,b5,a6,b6,a7,b7]
    SUB $1, R4
    CBNZ R4, interleave2_neon_loop8

interleave2_neon_remainder:
    AND $7, R3
    CBZ R3, interleave2_neon_done

interleave2_neon_loop1:
    MOVH (R1), R5
    MOVH (R2), R6
    MOVH R5, (R0)
    MOVH R6, 2(R0)
    ADD $2, R1
    ADD $2, R2
    ADD $4, R0
    SUB $1, R3
    CBNZ R3, interleave2_neon_loop1

interleave2_neon_done:
    RET

// func deinterleave2NEON(a, b, src []int16)
// Deinterleaves: a[0]=src[0], b[0]=src[1], a[1]=src[2], b[1]=src[3], ...
TEXT ·deinterleave2NEON(SB), NOSPLIT, $0-72
    MOVD a_base+0(FP), R0      // a pointer
    MOVD a_len+8(FP), R3       // n = len(a)
    MOVD b_base+24(FP), R1     // b pointer
    MOVD src_base+48(FP), R2   // src pointer

    // Process 8 pairs at a time
    LSR $3, R3, R4             // R4 = n / 8
    CBZ R4, deinterleave2_neon_remainder

deinterleave2_neon_loop8:
    VLD1.P 16(R2), [V0.H8]     // V0 = [a0,b0,a1,b1,a2,b2,a3,b3]
    VLD1.P 16(R2), [V1.H8]     // V1 = [a4,b4,a5,b5,a6,b6,a7,b7]
    WORD $0x4E411802           // UZP1 V2.8H, V0.8H, V1.8H -> [a0,a1,a2,a3,a4,a5,a6,a7]
    WORD $0x4E415803           // UZP2 V3.8H, V0.8H, V1.8H -> [b0,b1,b2,b3,b4,b5,b6,b7]
    VST1.P [V2.H8], 16(R0)     // Store a
    VST1.P [V3.H8], 16(R1)     // Store b
    SUB $1, R4
    CBNZ R4, deinterleave2_neon_loop8

deinterleave2_neon_remainder:
    AND $7, R3
    CBZ R3, deinterleave2_neon_done

deinterleave2_neon_loop1:
    MOVH (R2), R5
    MOVH 2(R2), R6
    MOVH R5, (R0)
    MOVH R6, (R1)
    ADD $4, R2
    ADD $2, R0
    ADD $2, R1
    SUB $1, R3
    CBNZ R3, deinterleave2_neon_loop1

deinterleave2_neon_done:
    RET

// func dotNEON(a, b []int16) int32
// Handles mismatched slice lengths: uses min(len(a), len(b)).
//
// Widening int16 dot product. SMLAL/SMLAL2 each multiply four int16 pairs into
// int32 and accumulate, so one iteration retires 16 products into four
// independent accumulators; the four chains keep the multiply-accumulate
// latency off the critical path. VADD folds them, ADDV reduces, and an 8-wide
// block plus a scalar tail finish n mod 16 (short CELT bands are common, so the
// 8-wide block earns its keep).
//
// Accumulation wraps in int32, matching dotGo bit-for-bit: SMLAL wraps per lane
// and wrapping addition is associative, so the lane split and the ADDV
// reduction cannot change the result.
//
// The Go assembler has no mnemonic for any integer vector multiply (SMLAL and
// friends are all "unrecognized instruction"), so these are hand-encoded as
// WORD; the trailing comment is the decoded form and asmcheck_test.go
// cross-checks it against arm64asm.
TEXT ·dotNEON(SB), NOSPLIT, $0-52
    MOVD a_base+0(FP), R0      // a pointer
    MOVD a_len+8(FP), R2
    MOVD b_len+32(FP), R3
    CMP  R3, R2
    CSEL LT, R2, R3, R2        // R2 = n = min(len(a), len(b))
    MOVD b_base+24(FP), R1     // b pointer

    VEOR V16.B16, V16.B16, V16.B16
    VEOR V17.B16, V17.B16, V17.B16
    VEOR V18.B16, V18.B16, V18.B16
    VEOR V19.B16, V19.B16, V19.B16

    LSR  $4, R2, R4            // R4 = n / 16
    CBZ  R4, dot_neon_block8

dot_neon_loop16:
    VLD1.P 32(R0), [V0.H8, V1.H8]
    VLD1.P 32(R1), [V2.H8, V3.H8]
    WORD $0x0E628010           // SMLAL V16.4S, V0.4H, V2.4H
    WORD $0x4E628011           // SMLAL2 V17.4S, V0.8H, V2.8H
    WORD $0x0E638032           // SMLAL V18.4S, V1.4H, V3.4H
    WORD $0x4E638033           // SMLAL2 V19.4S, V1.8H, V3.8H
    SUB  $1, R4
    CBNZ R4, dot_neon_loop16

dot_neon_block8:
    AND  $15, R2, R3           // R3 = n mod 16
    TBZ  $3, R3, dot_neon_fold // bit 3 clear => fewer than 8 left
    VLD1.P 16(R0), [V0.H8]
    VLD1.P 16(R1), [V2.H8]
    WORD $0x0E628010           // SMLAL V16.4S, V0.4H, V2.4H
    WORD $0x4E628011           // SMLAL2 V17.4S, V0.8H, V2.8H

dot_neon_fold:
    VADD V17.S4, V16.S4, V16.S4
    VADD V19.S4, V18.S4, V18.S4
    VADD V18.S4, V16.S4, V16.S4
    VADDV V16.S4, V16          // ADDV S16, V16.4S
    FMOVS F16, R5              // R5 = vector partial sum

    AND  $7, R2, R3            // R3 = n mod 8
    CBZ  R3, dot_neon_done

dot_neon_scalar:
    MOVH.P 2(R0), R6           // sign-extending 16-bit load
    MOVH.P 2(R1), R7
    MUL  R7, R6, R6
    ADDW R6, R5, R5            // 32-bit add: wraps like dotGo
    SUB  $1, R3
    CBNZ R3, dot_neon_scalar

dot_neon_done:
    MOVW R5, ret+48(FP)
    RET
