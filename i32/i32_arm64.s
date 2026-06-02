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

// func cumsumNEON(a []int32)
// In-place inclusive prefix sum: a[i] += a[i-1] for i in 1..n-1 (a[0] kept).
// This is the order-1 fixed-predictor restore and the building block the
// Restore1..Restore4 wrappers compose. Each .4S block computes a standalone
// 4-element inclusive prefix sum (two zero-filled EXT shift-adds), adds the
// running total of earlier blocks, then broadcasts its last lane as the next
// block's running total. The scalar tail reads the previous cumulative value
// from memory, so it needs no vector carry. EXT/DUP/MOVI and ADD .4S are
// hand-encoded as WORD (the Go assembler lacks these mnemonics); the trailing
// comment is the decoded form, cross-checked by asmcheck_test.go.
TEXT ·cumsumNEON(SB), NOSPLIT, $0-24
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R3

    LSR $2, R3, R4             // R4 = n / 4 (block count; >=1, dispatch gates n>=4)
    CBZ R4, cumsum_neon_remainder

    WORD $0x4F000407           // MOVI V7.4S, #0x0   (zero source for the EXT shifts)
    WORD $0x4F000408           // MOVI V8.4S, #0x0   (running carry = 0)

cumsum_neon_loop4:
    VLD1 (R0), [V0.S4]         // block [x0, x1, x2, x3]
    WORD $0x6E0060E1           // EXT V1.16B, V7.16B, V0.16B, #12   ([0, x0, x1, x2])
    WORD $0x4EA18400           // ADD V0.4S, V0.4S, V1.4S
    WORD $0x6E0040E1           // EXT V1.16B, V7.16B, V0.16B, #8    ([0, 0, y0, y1])
    WORD $0x4EA18400           // ADD V0.4S, V0.4S, V1.4S   (4-elem inclusive prefix)
    WORD $0x4EA88400           // ADD V0.4S, V0.4S, V8.4S   (+ carry from earlier blocks)
    VST1 [V0.S4], (R0)
    WORD $0x4E1C0408           // DUP V8.4S, V0.S[3]        (carry = last lane)
    ADD $16, R0
    SUB $1, R4
    CBNZ R4, cumsum_neon_loop4

cumsum_neon_remainder:
    AND $3, R3
    CBZ R3, cumsum_neon_done

    MOVW -4(R0), R5            // carry = previous cumulative value from memory

cumsum_neon_loop1:
    MOVW (R0), R6
    ADDW R5, R6, R6
    MOVW R6, (R0)
    MOVW R6, R5
    ADD $4, R0
    SUB $1, R3
    CBNZ R3, cumsum_neon_loop1

cumsum_neon_done:
    RET

// func lpcResidualEncodeNEON(res, samples, coeffs []int32, shift uint)
// Quantized-LPC encode FIR, vectorized across 4 output samples per iteration:
//
//	res[i] = samples[i] - int32((Σ_j coeffs[j]*samples[i-1-j]) >> shift)
//
// For each tap j the 4-sample window samples[i-1-j..i+2-j] is widened to int64
// (SMLAL on the low two lanes, SMLAL2 on the high two) and multiply-accumulated
// into two int64x2 accumulators. After the tap loop each accumulator is
// arithmetic-shifted right by shift (SSHL by the broadcast -shift, NEON's native
// signed 64-bit shift), the int64 predictions are narrowed back to int32 (XTN /
// XTN2) and subtracted from samples[i..i+3]. The first 'order' samples are the
// verbatim warm-up; the (n-order) mod 4 trailing outputs use a scalar int64
// recurrence. The dispatch guarantees order >= 1 and n-order >= 4.
TEXT ·lpcResidualEncodeNEON(SB), NOSPLIT, $0-80
    MOVD res_base+0(FP), R0
    MOVD res_len+8(FP), R4           // n
    MOVD samples_base+24(FP), R1
    MOVD coeffs_base+48(FP), R2
    MOVD coeffs_len+56(FP), R3       // order
    MOVD shift+72(FP), R6

    // warm-up: res[0:order] = samples[0:order]
    MOVD R1, R10
    MOVD R0, R11
    MOVD R3, R9
lpcenc_neon_warmup:
    MOVW.P 4(R10), R12
    MOVW.P R12, 4(R11)
    SUB $1, R9
    CBNZ R9, lpcenc_neon_warmup

    SUB $1, R3, R10
    LSL $2, R10, R10
    ADD R1, R10, R5                  // winBase = &samples[order-1]
    LSL $2, R3, R10
    ADD R0, R10, R0                  // R0 = &res[order]
    SUB R3, R4, R4                   // residual count = n - order
    LSR $2, R4, R7                   // R7 = full 4-blocks
    AND $3, R4, R8                   // R8 = remaining tail outputs
    NEG R6, R12
    WORD $0x4e080d84                 // DUP V4.2D, X12   (-shift, for the SSHL)

    CBZ R7, lpcenc_neon_tail
lpcenc_neon_block:
    WORD $0x6f00e400                 // MOVI V0.2D, #0x0   (acc_lo = 0)
    WORD $0x6f00e401                 // MOVI V1.2D, #0x0   (acc_hi = 0)
    MOVD R5, R10                     // window ptr = winBase - j*4
    MOVD R2, R11                     // coeff ptr
    MOVD R3, R9                      // j = order
lpcenc_neon_tap:
    VLD1 (R10), [V2.S4]              // S = samples[i-1-j..i+2-j]
    MOVW (R11), R12                  // coeff[j]
    WORD $0x4e040d83                 // DUP V3.4S, W12
    WORD $0x0ea38040                 // SMLAL V0.2D, V2.2S, V3.2S
    WORD $0x4ea38041                 // SMLAL2 V1.2D, V2.4S, V3.4S
    ADD $4, R11
    SUB $4, R10
    SUB $1, R9
    CBNZ R9, lpcenc_neon_tap

    WORD $0x4ee44400                 // SSHL V0.2D, V0.2D, V4.2D   (>>shift arithmetic)
    WORD $0x4ee44421                 // SSHL V1.2D, V1.2D, V4.2D
    WORD $0x0ea12805                 // XTN V5.2S, V0.2D
    WORD $0x4ea12825                 // XTN2 V5.4S, V1.2D
    ADD $4, R5, R10                  // &samples[i]
    VLD1 (R10), [V6.S4]              // samples[i..i+3]
    WORD $0x6ea584c6                 // SUB V6.4S, V6.4S, V5.4S    (res = samples - pred)
    VST1.P [V6.S4], 16(R0)
    ADD $16, R5                      // winBase += 4 samples
    SUB $1, R7
    CBNZ R7, lpcenc_neon_block

lpcenc_neon_tail:
    CBZ R8, lpcenc_neon_done
lpcenc_neon_tail_out:
    MOVD ZR, R4                      // acc = 0 (int64)
    MOVD R5, R10                     // window ptr
    MOVD R2, R11                     // coeff ptr
    MOVD R3, R9                      // j = order
lpcenc_neon_tail_tap:
    MOVW (R10), R12                  // sign-extend samples[i-1-j]
    MOVW (R11), R13                  // sign-extend coeff[j]
    MUL R12, R13, R13
    ADD R13, R4, R4                  // acc += coeff[j]*samples[i-1-j]
    ADD $4, R11
    SUB $4, R10
    SUB $1, R9
    CBNZ R9, lpcenc_neon_tail_tap
    ASR R6, R4, R4                   // pred = acc >> shift (arithmetic)
    ADD $4, R5, R10
    MOVW (R10), R12                  // samples[i]
    SUBW R4, R12, R12                // - pred
    MOVW.P R12, 4(R0)
    ADD $4, R5                       // winBase += 1 sample
    SUB $1, R8
    CBNZ R8, lpcenc_neon_tail_out

lpcenc_neon_done:
    RET

// func lpcRestoreNEON(out, residual, rcoeffs []int32, shift uint)
// Quantized-LPC decode recurrence:
//
//	out[i] = residual[i] + int32((Σ_j coeffs[j]*out[i-1-j]) >> shift)
//
// Serial across i, so only the per-output tap dot product is vectorized. The
// caller passes rcoeffs (coeffs reversed), so the ascending window out[i-order..]
// and rcoeffs line up. The newest scalarTaps taps (which include the just-stored
// out[i-1]) are summed scalar to keep them on forwardable narrow loads; the
// oldest vecTaps (a multiple of 4) are widened (SMLAL/SMLAL2) into int64
// accumulators, summed (ADD.2D), and folded to a scalar (ADDP). scalarTaps =
// ((order+2) & 3) + 2 keeps the newest 2..5 taps scalar. The dispatch gates order
// in [minNEONRestoreOrder, 32] and n-order >= 1.
TEXT ·lpcRestoreNEON(SB), NOSPLIT, $0-80
    MOVD out_base+0(FP), R5
    MOVD out_len+8(FP), R4           // n
    MOVD residual_base+24(FP), R1
    MOVD rcoeffs_base+48(FP), R2
    MOVD rcoeffs_len+56(FP), R3      // order
    MOVD shift+72(FP), R6

    // warm-up: out[0:order] = residual[0:order]
    MOVD R5, R10
    MOVD R1, R11
    MOVD R3, R9
lpcdec_neon_warmup:
    MOVW.P 4(R11), R12
    MOVW.P R12, 4(R10)
    SUB $1, R9
    CBNZ R9, lpcdec_neon_warmup

    LSL $2, R4, R10
    ADD R5, R10, R0                  // R0 = &out[n] (end sentinel)
    MOVD R5, R7                      // R7 = &out[i-order] (window base, i=order -> &out[0])
    LSL $2, R3, R10
    ADD R5, R10, R5                  // R5 = &out[order]
    ADD R1, R10, R1                  // R1 = &residual[order]
    // scalarTaps = ((order+2) & 3) + 2 ; vecTaps = order - scalarTaps ; numVec = vecTaps/4
    ADD $2, R3, R10
    AND $3, R10, R10
    ADD $2, R10, R10                 // scalarTaps (2..5)
    SUB R10, R3, R8                  // vecTaps (multiple of 4)
    LSR $2, R8, R8                   // R8 = numVec

lpcdec_neon_out:
    CMP R0, R5
    BHS lpcdec_neon_done
    WORD $0x6f00e400                 // MOVI V0.2D, #0x0   (acc_lo)
    WORD $0x6f00e401                 // MOVI V1.2D, #0x0   (acc_hi)
    MOVD R7, R10                     // window ptr = &out[i-order]
    MOVD R2, R11                     // rcoeffs ptr
    MOVD R8, R13                     // groups remaining = numVec
    CBZ R13, lpcdec_neon_reduce
lpcdec_neon_vec:
    VLD1.P 16(R10), [V2.S4]          // W = out[i-order + g*4 ..]
    VLD1.P 16(R11), [V3.S4]          // rcoeffs[g*4 ..]
    WORD $0x0ea38040                 // SMLAL V0.2D, V2.2S, V3.2S
    WORD $0x4ea38041                 // SMLAL2 V1.2D, V2.4S, V3.4S
    SUB $1, R13
    CBNZ R13, lpcdec_neon_vec
lpcdec_neon_reduce:
    WORD $0x4ee18400                 // ADD V0.2D, V0.2D, V1.2D
    WORD $0x5ef1b800                 // ADDP D0, V0.2D
    FMOVD F0, R12                    // R12 = vector partial sum (int64)
    // scalar leftover taps: R10/R11 already advanced to tap = vecTaps
lpcdec_neon_rem:
    CMP R5, R10
    BHS lpcdec_neon_pred
    MOVW.P 4(R10), R13               // out[i-order+k]
    MOVW.P 4(R11), R9                // rcoeffs[k]
    MUL R13, R9, R13
    ADD R13, R12, R12                // acc += rcoeffs[k]*out[i-order+k]
    B   lpcdec_neon_rem
lpcdec_neon_pred:
    ASR R6, R12, R12                 // pred = acc >> shift (arithmetic)
    MOVW (R1), R13                   // residual[i]
    ADDW R12, R13, R13               // + pred
    MOVW.P R13, 4(R5)                // out[i]
    ADD $4, R1                       // &residual[i]++
    ADD $4, R7                       // window base++
    B   lpcdec_neon_out

lpcdec_neon_done:
    RET

// func riceSumsNEON(sums []uint64, res []int32)
// Rice per-parameter unary-bit sums, sums[k] = Σ_i (zigzag(res[i]) >> k) for
// k = 0..14 (the fixed FLAC 4-bit range; the dispatch guarantees len(sums)==15).
//
// Each residual is folded to its unsigned Rice symbol with
// zigzag(r) = (r<<1) ^ (r>>31): VSHL gives r<<1 and the arithmetic r>>31 is built
// without a signed-shift mnemonic as -(r>>>31), the logical sign bit negated
// (VUSHR then VSUB from zero). The fold is zero-extended to int64 (VUSHLL /
// VUSHLL2 on the low and high S-lane pairs) so the per-k shifts and running
// totals never overflow. ARM64's 32 vector registers hold all 15 int64x2
// accumulators in a single sweep: each 4-lane block contributes its low pair
// (V2) and high pair (V3), which are halved progressively (u>>k is u>>(k-1)
// shifted once more, exact for the logical VUSHR) and added to one accumulator
// per k. After the loop each accumulator's two lanes are folded (VADDP) and
// stored. The (n mod 4) trailing residuals are then added to all 15 sums with a
// scalar progressive-halving tail. The dispatch gates len(res) >= 4 (>=1 block).
TEXT ·riceSumsNEON(SB), NOSPLIT, $0-48
    MOVD sums_base+0(FP), R0
    MOVD res_base+24(FP), R2
    MOVD res_len+32(FP), R3
    LSR  $2, R3, R4                  // R4 = full 4-element blocks (>=1)
    VEOR V7.B16, V7.B16, V7.B16      // V7 = 0 (zero source for the sign negate)
    VEOR V8.B16, V8.B16, V8.B16
    VEOR V9.B16, V9.B16, V9.B16
    VEOR V10.B16, V10.B16, V10.B16
    VEOR V11.B16, V11.B16, V11.B16
    VEOR V12.B16, V12.B16, V12.B16
    VEOR V13.B16, V13.B16, V13.B16
    VEOR V14.B16, V14.B16, V14.B16
    VEOR V15.B16, V15.B16, V15.B16
    VEOR V16.B16, V16.B16, V16.B16
    VEOR V17.B16, V17.B16, V17.B16
    VEOR V18.B16, V18.B16, V18.B16
    VEOR V19.B16, V19.B16, V19.B16
    VEOR V20.B16, V20.B16, V20.B16
    VEOR V21.B16, V21.B16, V21.B16
    VEOR V22.B16, V22.B16, V22.B16

rice_neon_loop:
    VLD1.P 16(R2), [V0.S4]           // v = res[i..i+3]
    VSHL   $1, V0.S4, V1.S4          // r<<1
    VUSHR  $31, V0.S4, V4.S4         // (uint32)r >> 31  (sign bit, 0 or 1)
    VSUB   V4.S4, V7.S4, V4.S4       // mask = 0 - signbit = (r>>31 arithmetic)
    VEOR   V4.B16, V1.B16, V1.B16    // u = zigzag(r)
    VUSHLL  $0, V1.S2, V2.D2         // cur_lo = zero-extend lanes 0,1 -> int64x2
    VUSHLL2 $0, V1.S4, V3.D2         // cur_hi = zero-extend lanes 2,3 -> int64x2
    VADD V2.D2, V8.D2, V8.D2          // k=0
    VADD V3.D2, V8.D2, V8.D2
    VUSHR $1, V2.D2, V2.D2
    VUSHR $1, V3.D2, V3.D2
    VADD V2.D2, V9.D2, V9.D2          // k=1
    VADD V3.D2, V9.D2, V9.D2
    VUSHR $1, V2.D2, V2.D2
    VUSHR $1, V3.D2, V3.D2
    VADD V2.D2, V10.D2, V10.D2          // k=2
    VADD V3.D2, V10.D2, V10.D2
    VUSHR $1, V2.D2, V2.D2
    VUSHR $1, V3.D2, V3.D2
    VADD V2.D2, V11.D2, V11.D2          // k=3
    VADD V3.D2, V11.D2, V11.D2
    VUSHR $1, V2.D2, V2.D2
    VUSHR $1, V3.D2, V3.D2
    VADD V2.D2, V12.D2, V12.D2          // k=4
    VADD V3.D2, V12.D2, V12.D2
    VUSHR $1, V2.D2, V2.D2
    VUSHR $1, V3.D2, V3.D2
    VADD V2.D2, V13.D2, V13.D2          // k=5
    VADD V3.D2, V13.D2, V13.D2
    VUSHR $1, V2.D2, V2.D2
    VUSHR $1, V3.D2, V3.D2
    VADD V2.D2, V14.D2, V14.D2          // k=6
    VADD V3.D2, V14.D2, V14.D2
    VUSHR $1, V2.D2, V2.D2
    VUSHR $1, V3.D2, V3.D2
    VADD V2.D2, V15.D2, V15.D2          // k=7
    VADD V3.D2, V15.D2, V15.D2
    VUSHR $1, V2.D2, V2.D2
    VUSHR $1, V3.D2, V3.D2
    VADD V2.D2, V16.D2, V16.D2          // k=8
    VADD V3.D2, V16.D2, V16.D2
    VUSHR $1, V2.D2, V2.D2
    VUSHR $1, V3.D2, V3.D2
    VADD V2.D2, V17.D2, V17.D2          // k=9
    VADD V3.D2, V17.D2, V17.D2
    VUSHR $1, V2.D2, V2.D2
    VUSHR $1, V3.D2, V3.D2
    VADD V2.D2, V18.D2, V18.D2          // k=10
    VADD V3.D2, V18.D2, V18.D2
    VUSHR $1, V2.D2, V2.D2
    VUSHR $1, V3.D2, V3.D2
    VADD V2.D2, V19.D2, V19.D2          // k=11
    VADD V3.D2, V19.D2, V19.D2
    VUSHR $1, V2.D2, V2.D2
    VUSHR $1, V3.D2, V3.D2
    VADD V2.D2, V20.D2, V20.D2          // k=12
    VADD V3.D2, V20.D2, V20.D2
    VUSHR $1, V2.D2, V2.D2
    VUSHR $1, V3.D2, V3.D2
    VADD V2.D2, V21.D2, V21.D2          // k=13
    VADD V3.D2, V21.D2, V21.D2
    VUSHR $1, V2.D2, V2.D2
    VUSHR $1, V3.D2, V3.D2
    VADD V2.D2, V22.D2, V22.D2          // k=14
    VADD V3.D2, V22.D2, V22.D2
    SUB  $1, R4
    CBNZ R4, rice_neon_loop

    // fold each int64x2 accumulator to a scalar and store sums[0..14]
    VADDP V8.D2, V8.D2, V1.D2
    FMOVD F1, R5
    MOVD  R5, 0(R0)
    VADDP V9.D2, V9.D2, V1.D2
    FMOVD F1, R5
    MOVD  R5, 8(R0)
    VADDP V10.D2, V10.D2, V1.D2
    FMOVD F1, R5
    MOVD  R5, 16(R0)
    VADDP V11.D2, V11.D2, V1.D2
    FMOVD F1, R5
    MOVD  R5, 24(R0)
    VADDP V12.D2, V12.D2, V1.D2
    FMOVD F1, R5
    MOVD  R5, 32(R0)
    VADDP V13.D2, V13.D2, V1.D2
    FMOVD F1, R5
    MOVD  R5, 40(R0)
    VADDP V14.D2, V14.D2, V1.D2
    FMOVD F1, R5
    MOVD  R5, 48(R0)
    VADDP V15.D2, V15.D2, V1.D2
    FMOVD F1, R5
    MOVD  R5, 56(R0)
    VADDP V16.D2, V16.D2, V1.D2
    FMOVD F1, R5
    MOVD  R5, 64(R0)
    VADDP V17.D2, V17.D2, V1.D2
    FMOVD F1, R5
    MOVD  R5, 72(R0)
    VADDP V18.D2, V18.D2, V1.D2
    FMOVD F1, R5
    MOVD  R5, 80(R0)
    VADDP V19.D2, V19.D2, V1.D2
    FMOVD F1, R5
    MOVD  R5, 88(R0)
    VADDP V20.D2, V20.D2, V1.D2
    FMOVD F1, R5
    MOVD  R5, 96(R0)
    VADDP V21.D2, V21.D2, V1.D2
    FMOVD F1, R5
    MOVD  R5, 104(R0)
    VADDP V22.D2, V22.D2, V1.D2
    FMOVD F1, R5
    MOVD  R5, 112(R0)

    // scalar tail: (n mod 4) residuals, add to all 15 sums
    AND  $3, R3, R4
    CBZ  R4, rice_neon_done
rice_neon_tail_out:
    MOVW (R2), R5                    // sign-extend r
    ADDW R5, R5, R6                  // r<<1
    ASRW $31, R5, R7                 // r>>31 (arithmetic)
    EORW R7, R6, R6                  // u = zigzag(r) (zero-extended)
    MOVD R0, R8                      // sums ptr
    MOVD $15, R9                     // k counter
rice_neon_tail_k:
    MOVD (R8), R7
    ADD  R6, R7, R7
    MOVD R7, (R8)
    LSR  $1, R6, R6
    ADD  $8, R8
    SUB  $1, R9
    CBNZ R9, rice_neon_tail_k
    ADD  $4, R2
    SUB  $1, R4
    CBNZ R4, rice_neon_tail_out

rice_neon_done:
    RET
