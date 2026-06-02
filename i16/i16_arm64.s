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
