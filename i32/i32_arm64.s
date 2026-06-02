//go:build arm64

#include "textflag.h"

// int32 (de)interleave on ARM64 (NEON / ASIMD).
//
// ZIP1/ZIP2/UZP1/UZP2 on .4S operands and the VLD1/VST1 .S4 loads/stores all
// move 32-bit lanes by bit pattern, so these kernels are the int32 counterparts
// of interleave2NEON / deinterleave2NEON in ../f32/f32_arm64.s with identical
// vector bodies; only the scalar tails differ (integer MOVW instead of FMOVS).
// The ZIP/UZP instructions are hand-encoded as WORD because the Go assembler
// lacks mnemonics for them; the trailing comment is the decoded form and is
// cross-checked by asmcheck_test.go.

// func interleave2NEON(dst, a, b []int32)
// Interleaves: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], dst[3]=b[1], ...
TEXT ·interleave2NEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0    // dst pointer
    MOVD a_base+24(FP), R1     // a pointer
    MOVD a_len+32(FP), R3      // n = len(a)
    MOVD b_base+48(FP), R2     // b pointer

    // Process 4 pairs at a time
    LSR $2, R3, R4             // R4 = n / 4
    CBZ R4, interleave2_neon_remainder

interleave2_neon_loop4:
    VLD1.P 16(R1), [V0.S4]     // V0 = [a0, a1, a2, a3]
    VLD1.P 16(R2), [V1.S4]     // V1 = [b0, b1, b2, b3]
    WORD $0x4E813802           // ZIP1 V2.4S, V0.4S, V1.4S -> [a0, b0, a1, b1]
    WORD $0x4E817803           // ZIP2 V3.4S, V0.4S, V1.4S -> [a2, b2, a3, b3]
    VST1.P [V2.S4], 16(R0)     // Store [a0, b0, a1, b1]
    VST1.P [V3.S4], 16(R0)     // Store [a2, b2, a3, b3]
    SUB $1, R4
    CBNZ R4, interleave2_neon_loop4

interleave2_neon_remainder:
    AND $3, R3
    CBZ R3, interleave2_neon_done

interleave2_neon_loop1:
    MOVW (R1), R5
    MOVW (R2), R6
    MOVW R5, (R0)
    MOVW R6, 4(R0)
    ADD $4, R1
    ADD $4, R2
    ADD $8, R0
    SUB $1, R3
    CBNZ R3, interleave2_neon_loop1

interleave2_neon_done:
    RET

// func deinterleave2NEON(a, b, src []int32)
// Deinterleaves: a[0]=src[0], b[0]=src[1], a[1]=src[2], b[1]=src[3], ...
TEXT ·deinterleave2NEON(SB), NOSPLIT, $0-72
    MOVD a_base+0(FP), R0      // a pointer
    MOVD a_len+8(FP), R3       // n = len(a)
    MOVD b_base+24(FP), R1     // b pointer
    MOVD src_base+48(FP), R2   // src pointer

    // Process 4 pairs at a time
    LSR $2, R3, R4             // R4 = n / 4
    CBZ R4, deinterleave2_neon_remainder

deinterleave2_neon_loop4:
    VLD1.P 16(R2), [V0.S4]     // V0 = [a0, b0, a1, b1]
    VLD1.P 16(R2), [V1.S4]     // V1 = [a2, b2, a3, b3]
    WORD $0x4E811802           // UZP1 V2.4S, V0.4S, V1.4S -> [a0, a1, a2, a3]
    WORD $0x4E815803           // UZP2 V3.4S, V0.4S, V1.4S -> [b0, b1, b2, b3]
    VST1.P [V2.S4], 16(R0)     // Store a
    VST1.P [V3.S4], 16(R1)     // Store b
    SUB $1, R4
    CBNZ R4, deinterleave2_neon_loop4

deinterleave2_neon_remainder:
    AND $3, R3
    CBZ R3, deinterleave2_neon_done

deinterleave2_neon_loop1:
    MOVW (R2), R5
    MOVW 4(R2), R6
    MOVW R5, (R0)
    MOVW R6, (R1)
    ADD $8, R2
    ADD $4, R0
    ADD $4, R1
    SUB $1, R3
    CBNZ R3, deinterleave2_neon_loop1

deinterleave2_neon_done:
    RET

// Arithmetic, decorrelation and fixed-predictor kernels (NEON / ASIMD).
//
// These do integer ALU work on .4S vectors (4 int32/iter): ADD/SUB/SSHR/SHL and
// the AND/ORR/MOVI used for the mid/side parity bit. The Go assembler has no
// mnemonics for these vector ops, so they are hand-encoded as WORD with the
// decoded GNU form in the trailing comment (cross-checked by asmcheck_test.go).
// Each vector lane is 32 bits, so the SIMD path wraps exactly like the int32 Go
// reference; the scalar tails use the W-register (32-bit) ALU forms (ADDW/SUBW/
// ASRW/...) so they wrap identically. Dispatched from i32_arm64.go gated on the
// NEON CPU feature, with the pure-Go reference as the fallback.

// func addNEON(dst, a, b []int32)
TEXT ·addNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $2, R3, R4
    CBZ R4, add_neon_remainder

add_neon_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    WORD $0x4EA18402           // ADD V2.4S, V0.4S, V1.4S
    VST1.P [V2.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, add_neon_loop4

add_neon_remainder:
    AND $3, R3
    CBZ R3, add_neon_done

add_neon_loop1:
    MOVW (R1), R5
    MOVW (R2), R6
    ADDW R6, R5, R5
    MOVW R5, (R0)
    ADD $4, R1
    ADD $4, R2
    ADD $4, R0
    SUB $1, R3
    CBNZ R3, add_neon_loop1

add_neon_done:
    RET

// func subNEON(dst, a, b []int32)
TEXT ·subNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $2, R3, R4
    CBZ R4, sub_neon_remainder

sub_neon_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    WORD $0x6EA18402           // SUB V2.4S, V0.4S, V1.4S
    VST1.P [V2.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, sub_neon_loop4

sub_neon_remainder:
    AND $3, R3
    CBZ R3, sub_neon_done

sub_neon_loop1:
    MOVW (R1), R5
    MOVW (R2), R6
    SUBW R6, R5, R5
    MOVW R5, (R0)
    ADD $4, R1
    ADD $4, R2
    ADD $4, R0
    SUB $1, R3
    CBNZ R3, sub_neon_loop1

sub_neon_done:
    RET

// func midSideEncodeNEON(mid, side, left, right []int32)
// mid = (left + right) >> 1 (arithmetic), side = left - right
TEXT ·midSideEncodeNEON(SB), NOSPLIT, $0-96
    MOVD mid_base+0(FP), R0
    MOVD mid_len+8(FP), R3
    MOVD side_base+24(FP), R1
    MOVD left_base+48(FP), R2
    MOVD right_base+72(FP), R5

    LSR $2, R3, R4
    CBZ R4, mse_neon_remainder

mse_neon_loop4:
    VLD1.P 16(R2), [V0.S4]     // left
    VLD1.P 16(R5), [V1.S4]     // right
    WORD $0x4EA18402           // ADD V2.4S, V0.4S, V1.4S   (left + right)
    WORD $0x4F3F0442           // SSHR V2.4S, V2.4S, #1      ((left+right)>>1)
    VST1.P [V2.S4], 16(R0)     // mid
    WORD $0x6EA18403           // SUB V3.4S, V0.4S, V1.4S    (left - right)
    VST1.P [V3.S4], 16(R1)     // side
    SUB $1, R4
    CBNZ R4, mse_neon_loop4

mse_neon_remainder:
    AND $3, R3
    CBZ R3, mse_neon_done

mse_neon_loop1:
    MOVW (R2), R6              // left
    MOVW (R5), R7             // right
    ADDW R7, R6, R8           // left + right
    ASRW $1, R8, R8           // >> 1 arithmetic
    MOVW R8, (R0)             // mid
    SUBW R7, R6, R9           // left - right
    MOVW R9, (R1)             // side
    ADD $4, R2
    ADD $4, R5
    ADD $4, R0
    ADD $4, R1
    SUB $1, R3
    CBNZ R3, mse_neon_loop1

mse_neon_done:
    RET

// func midSideDecodeNEON(left, right, mid, side []int32)
// sum = (mid<<1)|(side&1); left = (sum+side)>>1; right = (sum-side)>>1
TEXT ·midSideDecodeNEON(SB), NOSPLIT, $0-96
    MOVD left_base+0(FP), R0
    MOVD left_len+8(FP), R3
    MOVD right_base+24(FP), R1
    MOVD mid_base+48(FP), R2
    MOVD side_base+72(FP), R5

    LSR $2, R3, R4
    CBZ R4, msd_neon_remainder
    WORD $0x4F000424           // MOVI V4.4S, #0x1   (per-lane parity mask)

msd_neon_loop4:
    VLD1.P 16(R2), [V0.S4]     // mid
    VLD1.P 16(R5), [V1.S4]     // side
    WORD $0x4F215402           // SHL V2.4S, V0.4S, #1        (mid << 1)
    WORD $0x4E241C23           // AND V3.16B, V1.16B, V4.16B  (side & 1)
    WORD $0x4EA31C42           // ORR V2.16B, V2.16B, V3.16B  (sum)
    WORD $0x4EA18445           // ADD V5.4S, V2.4S, V1.4S     (sum + side)
    WORD $0x4F3F04A5           // SSHR V5.4S, V5.4S, #1       (left)
    VST1.P [V5.S4], 16(R0)
    WORD $0x6EA18446           // SUB V6.4S, V2.4S, V1.4S     (sum - side)
    WORD $0x4F3F04C6           // SSHR V6.4S, V6.4S, #1       (right)
    VST1.P [V6.S4], 16(R1)
    SUB $1, R4
    CBNZ R4, msd_neon_loop4

msd_neon_remainder:
    AND $3, R3
    CBZ R3, msd_neon_done

msd_neon_loop1:
    MOVW (R2), R6              // mid
    MOVW (R5), R7             // side
    LSLW $1, R6, R8           // mid << 1
    ANDW $1, R7, R9           // side & 1
    ORRW R9, R8, R8           // sum
    ADDW R7, R8, R10
    ASRW $1, R10, R10         // (sum+side)>>1
    MOVW R10, (R0)            // left
    SUBW R7, R8, R11
    ASRW $1, R11, R11         // (sum-side)>>1
    MOVW R11, (R1)            // right
    ADD $4, R2
    ADD $4, R5
    ADD $4, R0
    ADD $4, R1
    SUB $1, R3
    CBNZ R3, msd_neon_loop1

msd_neon_done:
    RET

// func diff1NEON(dst, src []int32)
// dst[0]=src[0]; dst[n]=src[n]-src[n-1]
TEXT ·diff1NEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD src_base+24(FP), R1
    MOVD src_len+32(FP), R3

    MOVW (R1), R5
    MOVW R5, (R0)              // warm-up dst[0]=src[0]
    SUB $1, R3, R3             // residual count = len-1
    CBZ R3, diff1_neon_done
    ADD $4, R0                 // &dst[1]
    ADD $4, R1, R6             // R6 = &src[1] (src[n]); R1 = &src[0] (src[n-1])

    LSR $2, R3, R4
    CBZ R4, diff1_neon_remainder

diff1_neon_loop4:
    VLD1.P 16(R6), [V0.S4]     // src[n]
    VLD1.P 16(R1), [V1.S4]     // src[n-1]
    WORD $0x6EA18402           // SUB V2.4S, V0.4S, V1.4S
    VST1.P [V2.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, diff1_neon_loop4

diff1_neon_remainder:
    AND $3, R3
    CBZ R3, diff1_neon_done

diff1_neon_loop1:
    MOVW (R6), R5
    MOVW (R1), R7
    SUBW R7, R5, R5
    MOVW R5, (R0)
    ADD $4, R6
    ADD $4, R1
    ADD $4, R0
    SUB $1, R3
    CBNZ R3, diff1_neon_loop1

diff1_neon_done:
    RET

// func diff2NEON(dst, src []int32)
// dst[0:2]=src[0:2]; dst[n]=(v0+v2)-2*v1 where v0=src[n],v1=src[n-1],v2=src[n-2]
TEXT ·diff2NEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD src_base+24(FP), R1
    MOVD src_len+32(FP), R3

    MOVW (R1), R5
    MOVW R5, (R0)
    MOVW 4(R1), R5
    MOVW R5, 4(R0)             // warm-up dst[0:2]=src[0:2]
    SUB $2, R3, R3             // residual count = len-2
    ADD $8, R0                 // &dst[2]
    ADD $8, R1, R5             // pv0 = &src[2]
    ADD $4, R1, R6             // pv1 = &src[1]; R1 = pv2 = &src[0]

    LSR $2, R3, R4
    CBZ R4, diff2_neon_remainder

diff2_neon_loop4:
    VLD1.P 16(R5), [V0.S4]     // v0
    VLD1.P 16(R1), [V2.S4]     // v2
    WORD $0x4EA28403           // ADD V3.4S, V0.4S, V2.4S   (v0 + v2)
    VLD1.P 16(R6), [V1.S4]     // v1
    WORD $0x4F215424           // SHL V4.4S, V1.4S, #1      (2*v1)
    WORD $0x6EA48463           // SUB V3.4S, V3.4S, V4.4S   ((v0+v2) - 2*v1)
    VST1.P [V3.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, diff2_neon_loop4

diff2_neon_remainder:
    AND $3, R3
    CBZ R3, diff2_neon_done

diff2_neon_loop1:
    MOVW (R5), R7              // v0
    MOVW (R6), R8             // v1
    LSLW $1, R8, R8           // 2*v1
    SUBW R8, R7, R7           // v0 - 2*v1
    MOVW (R1), R9             // v2
    ADDW R9, R7, R7           // + v2
    MOVW R7, (R0)
    ADD $4, R5
    ADD $4, R6
    ADD $4, R1
    ADD $4, R0
    SUB $1, R3
    CBNZ R3, diff2_neon_loop1

diff2_neon_done:
    RET

// func diff3NEON(dst, src []int32)
// dst[0:3]=src[0:3]; dst[n]=(v0-v3)-3*(v1-v2)
TEXT ·diff3NEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD src_base+24(FP), R1
    MOVD src_len+32(FP), R3

    MOVW (R1), R5
    MOVW R5, (R0)
    MOVW 4(R1), R5
    MOVW R5, 4(R0)
    MOVW 8(R1), R5
    MOVW R5, 8(R0)             // warm-up dst[0:3]=src[0:3]
    SUB $3, R3, R3             // residual count = len-3
    ADD $12, R0                // &dst[3]
    ADD $12, R1, R5            // pv0 = &src[3]
    ADD $8, R1, R6             // pv1 = &src[2]
    ADD $4, R1, R7             // pv2 = &src[1]; R1 = pv3 = &src[0]

    LSR $2, R3, R4
    CBZ R4, diff3_neon_remainder

diff3_neon_loop4:
    VLD1.P 16(R5), [V0.S4]     // v0
    VLD1.P 16(R1), [V3.S4]     // v3
    WORD $0x6EA38400           // SUB V0.4S, V0.4S, V3.4S   (a = v0 - v3)
    VLD1.P 16(R6), [V1.S4]     // v1
    VLD1.P 16(R7), [V2.S4]     // v2
    WORD $0x6EA28421           // SUB V1.4S, V1.4S, V2.4S   (b = v1 - v2)
    WORD $0x4F215424           // SHL V4.4S, V1.4S, #1      (2*b)
    WORD $0x4EA18484           // ADD V4.4S, V4.4S, V1.4S   (3*b)
    WORD $0x6EA48400           // SUB V0.4S, V0.4S, V4.4S   (a - 3*b)
    VST1.P [V0.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, diff3_neon_loop4

diff3_neon_remainder:
    AND $3, R3
    CBZ R3, diff3_neon_done

diff3_neon_loop1:
    MOVW (R5), R8             // v0
    MOVW (R1), R9             // v3
    SUBW R9, R8, R8           // v0 - v3
    MOVW (R6), R9             // v1
    MOVW (R7), R10            // v2
    SUBW R10, R9, R9          // b = v1 - v2
    LSLW $1, R9, R11
    ADDW R9, R11, R11         // 3*b
    SUBW R11, R8, R8          // (v0-v3) - 3*b
    MOVW R8, (R0)
    ADD $4, R5
    ADD $4, R6
    ADD $4, R7
    ADD $4, R1
    ADD $4, R0
    SUB $1, R3
    CBNZ R3, diff3_neon_loop1

diff3_neon_done:
    RET

// func diff4NEON(dst, src []int32)
// dst[0:4]=src[0:4]; dst[n]=(v0+v4)-4*(v1+v3)+6*v2
TEXT ·diff4NEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD src_base+24(FP), R1
    MOVD src_len+32(FP), R3

    MOVW (R1), R5
    MOVW R5, (R0)
    MOVW 4(R1), R5
    MOVW R5, 4(R0)
    MOVW 8(R1), R5
    MOVW R5, 8(R0)
    MOVW 12(R1), R5
    MOVW R5, 12(R0)            // warm-up dst[0:4]=src[0:4]
    SUB $4, R3, R3             // residual count = len-4
    ADD $16, R0                // &dst[4]
    ADD $16, R1, R5            // pv0 = &src[4]
    ADD $12, R1, R6            // pv1 = &src[3]
    ADD $8, R1, R7             // pv2 = &src[2]
    ADD $4, R1, R8             // pv3 = &src[1]; R1 = pv4 = &src[0]

    LSR $2, R3, R4
    CBZ R4, diff4_neon_remainder

diff4_neon_loop4:
    VLD1.P 16(R5), [V0.S4]     // v0
    VLD1.P 16(R1), [V4.S4]     // v4
    WORD $0x4EA48400           // ADD V0.4S, V0.4S, V4.4S   (s04 = v0 + v4)
    VLD1.P 16(R6), [V1.S4]     // v1
    VLD1.P 16(R8), [V3.S4]     // v3
    WORD $0x4EA38421           // ADD V1.4S, V1.4S, V3.4S   (s13 = v1 + v3)
    WORD $0x4F225421           // SHL V1.4S, V1.4S, #2      (4*s13)
    WORD $0x6EA18400           // SUB V0.4S, V0.4S, V1.4S   (s04 - 4*s13)
    VLD1.P 16(R7), [V2.S4]     // v2
    WORD $0x4F225445           // SHL V5.4S, V2.4S, #2      (4*v2)
    WORD $0x4F215446           // SHL V6.4S, V2.4S, #1      (2*v2)
    WORD $0x4EA684A5           // ADD V5.4S, V5.4S, V6.4S   (6*v2)
    WORD $0x4EA58400           // ADD V0.4S, V0.4S, V5.4S   (+ 6*v2)
    VST1.P [V0.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, diff4_neon_loop4

diff4_neon_remainder:
    AND $3, R3
    CBZ R3, diff4_neon_done

diff4_neon_loop1:
    MOVW (R5), R9             // v0
    MOVW (R1), R10            // v4
    ADDW R10, R9, R9          // s04
    MOVW (R6), R10            // v1
    MOVW (R8), R11            // v3
    ADDW R11, R10, R10        // s13
    LSLW $2, R10, R10         // 4*s13
    SUBW R10, R9, R9          // s04 - 4*s13
    MOVW (R7), R10            // v2
    LSLW $2, R10, R11         // 4*v2
    LSLW $1, R10, R10         // 2*v2
    ADDW R10, R11, R11        // 6*v2
    ADDW R11, R9, R9          // + 6*v2
    MOVW R9, (R0)
    ADD $4, R5
    ADD $4, R6
    ADD $4, R7
    ADD $4, R8
    ADD $4, R1
    ADD $4, R0
    SUB $1, R3
    CBNZ R3, diff4_neon_loop1

diff4_neon_done:
    RET
