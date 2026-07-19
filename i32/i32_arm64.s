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

// Arithmetic and reduction kernels (NEON / ASIMD).
//
// These do integer ALU work on .4S vectors (4 int32/iter). Each vector lane is
// 32 bits, so the SIMD path wraps exactly like the int32 Go reference; the
// scalar tails use the W-register (32-bit) ALU forms (ADDW/SUBW/...) so they
// wrap identically. SMIN/SMAX/SMINV/SMAXV and vector ABS have no Go assembler
// mnemonics and are hand-encoded as WORD with the decoded GNU form in the
// trailing comment (cross-checked by asmcheck_test.go). The ADD/SUB vector
// WORDs below are hand-encoded too even though the assembler does accept the
// native VADD/VSUB spellings on .S4 operands (sumNEON uses one); they stay as
// verified WORD encodings rather than churn. Dispatched from i32_arm64.go
// gated on the NEON CPU feature, with the pure-Go reference as the fallback.

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

// func minMaxNEON(res []int32) (minVal, maxVal int32)
// Signed int32 min and max over res in one pass. The dispatch gates len(res) >=
// 4, so at least one full 4-element (.4S) block exists: the min and max
// accumulators start from block 0 and fold the remaining full blocks with
// SMIN/SMAX, then SMINV/SMAXV reduce each accumulator across its 4 lanes to a
// scalar and a scalar tail folds the (n mod 4) remainder. SMIN/SMAX/SMINV/SMAXV
// have no Go assembler mnemonic, so they are hand-encoded WORD directives (the
// trailing comment is the decoded form, cross-checked by asmcheck_test.go).
// Every compare is signed, matching minMaxGo exactly.
TEXT ·minMaxNEON(SB), NOSPLIT, $0-32
    MOVD res_base+0(FP), R2
    MOVD res_len+8(FP), R3
    LSR  $2, R3, R4                  // R4 = full 4-element blocks (>=1)
    VLD1 (R2), [V0.S4]               // V0 = block 0 (min acc), no advance
    VLD1.P 16(R2), [V1.S4]           // V1 = block 0 (max acc), advance to block 1
    SUB  $1, R4                      // blocks remaining after block 0
    CBZ  R4, mm_neon_reduce          // single block: accumulators hold it; R2 at tail
mm_neon_loop:
    VLD1.P 16(R2), [V2.S4]           // load block + advance
    WORD $0x4EA26C00                 // SMIN V0.4S, V0.4S, V2.4S
    WORD $0x4EA26421                 // SMAX V1.4S, V1.4S, V2.4S
    SUB  $1, R4
    CBNZ R4, mm_neon_loop

mm_neon_reduce:
    WORD $0x4EB1A803                 // SMINV S3, V0.4S
    WORD $0x4EB0A824                 // SMAXV S4, V1.4S
    FMOVS F3, R5                     // R5 = running min (low 32 = int32)
    FMOVS F4, R6                     // R6 = running max (low 32 = int32)

    // scalar tail: (n mod 4) residuals (R2 already at &res[fullBlocks*4])
    AND  $3, R3, R4
    CBZ  R4, mm_neon_done
mm_neon_tail:
    MOVW.P 4(R2), R7                 // r (sign-extended; low 32 = int32)
    CMPW R5, R7                      // (R7 - R5), signed 32-bit
    CSEL LT, R7, R5, R5             // R5 = min(r, R5)
    CMPW R6, R7
    CSEL GT, R7, R6, R6             // R6 = max(r, R6)
    SUB  $1, R4
    CBNZ R4, mm_neon_tail

mm_neon_done:
    MOVW R5, minVal+24(FP)
    MOVW R6, maxVal+28(FP)
    RET

// func sumNEON(a []int32) int32
// Wrapping int32 sum: VADD folds 4-lane blocks into a vector accumulator,
// ADDV reduces it to a scalar, and a 32-bit scalar tail adds the (n mod 4)
// remainder. Every add wraps in a 32-bit lane, and wrapping addition is
// associative, so the lane split and reduction order are bit-identical to
// sumGo for every input, including forced overflow. The slice arrives
// pre-clamped from the public Sum, so a_len is the trusted element count.
TEXT ·sumNEON(SB), NOSPLIT, $0-28
    MOVD a_base+0(FP), R1
    MOVD a_len+8(FP), R3

    VEOR V0.B16, V0.B16, V0.B16
    LSR  $2, R3, R4            // R4 = n / 4
    CBZ  R4, sum_neon_reduce

sum_neon_loop4:
    VLD1.P 16(R1), [V1.S4]
    VADD V1.S4, V0.S4, V0.S4   // accumulate (wrapping)
    SUB  $1, R4
    CBNZ R4, sum_neon_loop4

sum_neon_reduce:
    VADDV V0.S4, V0            // ADDV S0, V0.4S
    FMOVS F0, R5               // vector partial sum (low 32 = int32)

    AND  $3, R3
    CBZ  R3, sum_neon_done

sum_neon_scalar:
    MOVW.P 4(R1), R6
    ADDW R6, R5, R5            // 32-bit add: wraps like sumGo
    SUB  $1, R3
    CBNZ R3, sum_neon_scalar

sum_neon_done:
    MOVW R5, ret+24(FP)
    RET

// func absNEON(dst, a []int32)
// Wrapping absolute value, 4 lanes per iteration: vector ABS wraps at the
// type minimum (abs(MinInt32) = MinInt32 in a 32-bit lane), which is absGo's
// contract. The scalar tail computes |v| in a 64-bit GPR, where -(MinInt32)
// is +2^31, and the MOVW store keeps the low 32 bits, wrapping it back to
// MinInt32 identically.
TEXT ·absNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1

    LSR  $2, R3, R4            // R4 = n / 4
    CBZ  R4, abs_neon_remainder

abs_neon_loop4:
    VLD1.P 16(R1), [V0.S4]
    WORD $0x4EA0B801           // ABS V1.4S, V0.4S
    VST1.P [V1.S4], 16(R0)
    SUB  $1, R4
    CBNZ R4, abs_neon_loop4

abs_neon_remainder:
    AND  $3, R3
    CBZ  R3, abs_neon_done

abs_neon_scalar:
    MOVW.P 4(R1), R5           // v, sign-extended
    NEG  R5, R6                // -v
    CMP  $0, R5
    CSEL LT, R6, R5, R5        // |v| = v < 0 ? -v : v   (can be 2^31)
    MOVW.P R5, 4(R0)           // low 32 bits: 2^31 wraps to MinInt32
    SUB  $1, R3
    CBNZ R3, abs_neon_scalar

abs_neon_done:
    RET

// func negWhereNegNEON(dst, mag []int32, sign []float32)
// Branchless conditional negate, 4 lanes per iteration. SSHR V1.4S,#31 broadcasts
// each sign lane's IEEE-754 sign bit to a full-width int32 mask m (all-ones iff
// the sign bit is set, so -0.0/-Inf/-NaN negate); EOR then SUB apply (mag ^ m) -
// m, which is -mag when m = -1 (MinInt32 wraps to itself) and mag when m = 0.
// Bit-identical to negWhereNegGo. The scalar tail does the same in 64-bit GPRs:
// ASR $31 of the sign-extended sign word yields m = 0 or -1, and the MOVW store
// keeps the low 32 bits so -MinInt32 wraps back to MinInt32. The SSHR/EOR/SUB
// WORDs are cross-checked against arm64asm by TestArm64WordEncodings. Frame is 3
// slice headers: dst+0, mag+24, sign+48.
TEXT ·negWhereNegNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD mag_base+24(FP), R1
    MOVD sign_base+48(FP), R2

    LSR  $2, R3, R4            // R4 = n / 4
    CBZ  R4, negwhereneg_neon_remainder

negwhereneg_neon_loop4:
    VLD1.P 16(R2), [V1.S4]     // sign
    WORD $0x4F210421           // SSHR V1.4S, V1.4S, #31   -> mask m
    VLD1.P 16(R1), [V0.S4]     // mag
    WORD $0x6E211C00           // EOR V0.16B, V0.16B, V1.16B  -> mag ^ m
    WORD $0x6EA18400           // SUB V0.4S, V0.4S, V1.4S     -> (mag ^ m) - m
    VST1.P [V0.S4], 16(R0)
    SUB  $1, R4
    CBNZ R4, negwhereneg_neon_loop4

negwhereneg_neon_remainder:
    AND  $3, R3
    CBZ  R3, negwhereneg_neon_done

negwhereneg_neon_scalar:
    MOVW.P 4(R2), R6           // sign bits, sign-extended
    ASR  $31, R6, R6           // m = sign >> 31 = 0 or -1
    MOVW.P 4(R1), R5           // mag, sign-extended
    EOR  R6, R5, R5            // mag ^ m
    SUB  R6, R5, R5            // (mag ^ m) - m
    MOVW.P R5, 4(R0)           // low 32 bits: -MinInt32 wraps to MinInt32
    SUB  $1, R3
    CBNZ R3, negwhereneg_neon_scalar

negwhereneg_neon_done:
    RET

// Fixed-point scale-by-scalar kernels (NEON / ASIMD).
//
// Unlike AVX2, NEON has a native 64-bit ARITHMETIC shift (SSHR .2D), so the Q31/
// Q15 scale is a clean widen-shift-narrow: SMULL/SMULL2 multiply the int32 lanes
// by the broadcast coefficient into int64 products (Q=0 takes lanes 0,1; Q=1 the
// SMULL2 form takes lanes 2,3), SSHR .2D arithmetically shifts each 64-bit product
// right by the fixed-point position, and XTN/XTN2 narrow the low 32 bits of each
// int64 back to int32 (truncation = the int32() wrap, so a=k=MinInt32's 2^62 >> 31
// = 2^31 lands as MinInt32). k is broadcast with DUP from a W register: MOVW
// sign-extends the int32 for Q31, MOVH the int16 for Q15, and both leave int64(k)
// in the register for the scalar tail (MUL then ASR then a MOVW store that keeps
// the low 32 bits). dst may alias a exactly: each block/lane reads a before it
// stores dst. SMULL/SMULL2/SSHR/XTN/XTN2/DUP have no Go assembler mnemonic and are
// hand-encoded WORD directives with the decoded form in the trailing comment
// (cross-checked against arm64asm by TestArm64WordEncodings). Frame is two slice
// headers plus the scalar k: dst+0, a+24, k+48.

// func scaleQ31NEON(dst, a []int32, k int32)
TEXT ·scaleQ31NEON(SB), NOSPLIT, $0-52
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVW k+48(FP), R2          // R2 = int64(k), sign-extended int32 (also tail k)
    WORD $0x4E040C41           // DUP V1.4S, W2   (k in all 4 int32 lanes)

    LSR  $2, R3, R4            // R4 = n / 4
    CBZ  R4, scaleq31_neon_remainder

scaleq31_neon_loop4:
    VLD1.P 16(R1), [V0.S4]     // a[i..i+3]
    WORD $0x0EA1C002           // SMULL V2.2D, V0.2S, V1.2S   (lanes 0,1 -> 2 int64)
    WORD $0x4EA1C003           // SMULL2 V3.2D, V0.4S, V1.4S  (lanes 2,3 -> 2 int64)
    WORD $0x4F610442           // SSHR V2.2D, V2.2D, #31
    WORD $0x4F610463           // SSHR V3.2D, V3.2D, #31
    WORD $0x0EA12844           // XTN V4.2S, V2.2D   (low 32 of results 0,1)
    WORD $0x4EA12864           // XTN2 V4.4S, V3.2D  (low 32 of results 2,3)
    VST1.P [V4.S4], 16(R0)
    SUB  $1, R4
    CBNZ R4, scaleq31_neon_loop4

scaleq31_neon_remainder:
    AND  $3, R3
    CBZ  R3, scaleq31_neon_done

scaleq31_neon_scalar:
    MOVW.P 4(R1), R5           // a[i], sign-extended to 64-bit
    MUL  R2, R5, R5            // a[i] * k (64-bit, |p| <= 2^62)
    ASR  $31, R5, R5           // arithmetic shift right 31
    MOVW.P R5, 4(R0)           // low 32 bits: wraps like int32()
    SUB  $1, R3
    CBNZ R3, scaleq31_neon_scalar

scaleq31_neon_done:
    RET

// func scaleQ15NEON(dst, a []int32, k int16)
// Identical widen-shift-narrow to scaleQ31NEON with a shift of 15. k is a signed
// int16, sign-extended by MOVH to int64(k); |k * a[i]| <= 2^46, well inside the
// int64 product.
TEXT ·scaleQ15NEON(SB), NOSPLIT, $0-50
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVH k+48(FP), R2          // R2 = int64(k), sign-extended int16 (also tail k)
    WORD $0x4E040C41           // DUP V1.4S, W2   (k in all 4 int32 lanes)

    LSR  $2, R3, R4            // R4 = n / 4
    CBZ  R4, scaleq15_neon_remainder

scaleq15_neon_loop4:
    VLD1.P 16(R1), [V0.S4]     // a[i..i+3]
    WORD $0x0EA1C002           // SMULL V2.2D, V0.2S, V1.2S   (lanes 0,1 -> 2 int64)
    WORD $0x4EA1C003           // SMULL2 V3.2D, V0.4S, V1.4S  (lanes 2,3 -> 2 int64)
    WORD $0x4F710442           // SSHR V2.2D, V2.2D, #15
    WORD $0x4F710463           // SSHR V3.2D, V3.2D, #15
    WORD $0x0EA12844           // XTN V4.2S, V2.2D   (low 32 of results 0,1)
    WORD $0x4EA12864           // XTN2 V4.4S, V3.2D  (low 32 of results 2,3)
    VST1.P [V4.S4], 16(R0)
    SUB  $1, R4
    CBNZ R4, scaleq15_neon_loop4

scaleq15_neon_remainder:
    AND  $3, R3
    CBZ  R3, scaleq15_neon_done

scaleq15_neon_scalar:
    MOVW.P 4(R1), R5           // a[i], sign-extended to 64-bit
    MUL  R2, R5, R5            // k * a[i] (64-bit, |p| <= 2^46)
    ASR  $15, R5, R5           // arithmetic shift right 15
    MOVW.P R5, 4(R0)           // low 32 bits: wraps like int32()
    SUB  $1, R3
    CBNZ R3, scaleq15_neon_scalar

scaleq15_neon_done:
    RET
