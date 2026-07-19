//go:build arm64

#include "textflag.h"

// Fixed-point complex arithmetic kernels (NEON / ASIMD).
//
// Add and Sub are flat wrapping int32 lane ops (ADD / SUB .4S), 4 int32 per
// iteration with a scalar tail. MulByScalar is the truncating Q15 scale-by-scalar
// (MULT16_32_Q15) applied in place over the flat lane view, reusing the i32
// ScaleQ15 widen-shift-narrow. Mul and MulConj are the C_MUL / conjugated complex
// multiply, deinterleaved with LD2 and re-interleaved with ST2.
//
// The truncating Q15 product is a clean widen-shift-narrow: SMULL/SMULL2 multiply
// the int32 lanes into int64 products, SSHR .2D #15 arithmetically shifts each
// product right by 15 (truncating toward -inf, no rounding), and XTN/XTN2 narrow
// the low 32 bits back to int32 (the narrowing is the int32 wrap, so
// MinInt32 * MinInt16 = 2^46 >> 15 = 2^31 lands as MinInt32). The complex kernels
// deinterleave a into V_ar/V_ai with LD2 .4S and the int16 twiddle into V_br/V_bi
// with LD2 .4H, sign-extend the twiddle to int32 with SXTL (the #0 shift alias of
// SSHLL), form the four half-products, combine with ADD/SUB .4S, and ST2 the
// [re, im] pair back to interleaved [r0,i0,r1,i1,...]. dst may alias a exactly:
// each block deinterleaves its whole a and tw block before it stores dst.
//
// SXTL/SMULL/SMULL2/SSHR/XTN/XTN2/DUP have no Go assembler mnemonic, and the
// standalone Add/Sub kernels pin ADD/SUB .4S as verified WORD encodings (matching
// i32); all are hand-encoded WORD directives with the decoded GNU form in the
// trailing comment, cross-checked against golang.org/x/arch/arm64asm by
// TestArm64WordEncodings. The Mul/MulConj re/im combines use the Go VADD/VSUB
// mnemonics, and LD2/ST2 use VLD2/VST2. Reserved registers (g in R28, R27/R18/R16/R17) are
// untouched. The scalar tails use the W-register (32-bit) ALU forms so they wrap
// identically to the vector lanes.

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

// func mulByScalarNEON(a []int32, s int16)
// In-place truncating Q15 scale, 4 int32 per iteration: the i32 scaleQ15NEON
// widen-shift-narrow with the load and store sharing R0 (dst aliases a). s is a
// signed int16 broadcast with DUP from W2 (MOVH sign-extends it), and int64(s) in
// R2 also serves the scalar tail. Frame is one slice header plus the int16 scalar:
// a+0, s+24.
TEXT ·mulByScalarNEON(SB), NOSPLIT, $0-26
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R3
    MOVH s+24(FP), R2          // R2 = int64(s), sign-extended int16 (also tail s)
    WORD $0x4E040C41           // DUP V1.4S, W2   (s in all 4 int32 lanes)

    LSR  $2, R3, R4            // R4 = n / 4
    CBZ  R4, mulbyscalar_neon_remainder

mulbyscalar_neon_loop4:
    VLD1 (R0), [V0.S4]         // a[i..i+3] (no advance; in-place store reuses R0)
    WORD $0x0EA1C002           // SMULL V2.2D, V0.2S, V1.2S
    WORD $0x4EA1C003           // SMULL2 V3.2D, V0.4S, V1.4S
    WORD $0x4F710442           // SSHR V2.2D, V2.2D, #15
    WORD $0x4F710463           // SSHR V3.2D, V3.2D, #15
    WORD $0x0EA12844           // XTN V4.2S, V2.2D
    WORD $0x4EA12864           // XTN2 V4.4S, V3.2D
    VST1 [V4.S4], (R0)         // in place
    ADD  $16, R0
    SUB  $1, R4
    CBNZ R4, mulbyscalar_neon_loop4

mulbyscalar_neon_remainder:
    AND  $3, R3
    CBZ  R3, mulbyscalar_neon_done

mulbyscalar_neon_scalar:
    MOVW (R0), R5              // a[i], sign-extended
    MUL  R2, R5, R5            // s * a[i] (64-bit, |p| <= 2^46)
    ASR  $15, R5, R5           // arithmetic shift right 15
    MOVW.P R5, 4(R0)           // low 32 bits: wraps like int32(); store + advance
    SUB  $1, R3
    CBNZ R3, mulbyscalar_neon_scalar

mulbyscalar_neon_done:
    RET

// func mulNEON(dst, a []int32, tw []int16)
// C_MUL complex multiply, 4 complex (8 int32) per iteration. dst+0, a+24, tw+48.
TEXT ·mulNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3     // n (int32 count, even)
    MOVD a_base+24(FP), R1
    MOVD tw_base+48(FP), R2

    LSR  $3, R3, R4            // R4 = n / 8 = 4-complex blocks
    CBZ  R4, mul_neon_tail

mul_neon_loop:
    VLD2.P 32(R1), [V0.S4, V1.S4]    // V0=ar[0..3], V1=ai[0..3]
    VLD2.P 16(R2), [V2.H4, V3.H4]    // V2=br[0..3] int16, V3=bi[0..3] int16
    WORD $0x0F10A444           // SXTL V4.4S, V2.4H   (br int16 -> int32)
    WORD $0x0F10A465           // SXTL V5.4S, V3.4H   (bi int16 -> int32)
    WORD $0x0EA4C010           // SMULL V16.2D, V0.2S, V4.2S    (prr lanes 0,1)
    WORD $0x4EA4C011           // SMULL2 V17.2D, V0.4S, V4.4S   (prr lanes 2,3)
    WORD $0x4F710610           // SSHR V16.2D, V16.2D, #15
    WORD $0x4F710631           // SSHR V17.2D, V17.2D, #15
    WORD $0x0EA12A06           // XTN V6.2S, V16.2D    (prr low 32 of lanes 0,1)
    WORD $0x4EA12A26           // XTN2 V6.4S, V17.2D   (prr low 32 of lanes 2,3)
    WORD $0x0EA5C030           // SMULL V16.2D, V1.2S, V5.2S    (pii)
    WORD $0x4EA5C031           // SMULL2 V17.2D, V1.4S, V5.4S
    WORD $0x4F710610           // SSHR V16.2D, V16.2D, #15
    WORD $0x4F710631           // SSHR V17.2D, V17.2D, #15
    WORD $0x0EA12A07           // XTN V7.2S, V16.2D    (pii)
    WORD $0x4EA12A27           // XTN2 V7.4S, V17.2D
    WORD $0x0EA5C010           // SMULL V16.2D, V0.2S, V5.2S    (pri)
    WORD $0x4EA5C011           // SMULL2 V17.2D, V0.4S, V5.4S
    WORD $0x4F710610           // SSHR V16.2D, V16.2D, #15
    WORD $0x4F710631           // SSHR V17.2D, V17.2D, #15
    WORD $0x0EA12A08           // XTN V8.2S, V16.2D    (pri)
    WORD $0x4EA12A28           // XTN2 V8.4S, V17.2D
    WORD $0x0EA4C030           // SMULL V16.2D, V1.2S, V4.2S    (pir)
    WORD $0x4EA4C031           // SMULL2 V17.2D, V1.4S, V4.4S
    WORD $0x4F710610           // SSHR V16.2D, V16.2D, #15
    WORD $0x4F710631           // SSHR V17.2D, V17.2D, #15
    WORD $0x0EA12A09           // XTN V9.2S, V16.2D    (pir)
    WORD $0x4EA12A29           // XTN2 V9.4S, V17.2D
    VSUB V7.S4, V6.S4, V10.S4        // re = prr - pii
    VADD V9.S4, V8.S4, V11.S4        // im = pri + pir
    VST2.P [V10.S4, V11.S4], 32(R0)  // re-interleave -> dst
    SUB  $1, R4
    CBNZ R4, mul_neon_loop

mul_neon_tail:
    AND  $7, R3, R4            // leftover int32 (even)
    LSR  $1, R4, R4            // leftover complex count
    CBZ  R4, mul_neon_done

mul_neon_scalar:
    MOVW.P 4(R1), R5           // ar (sign-extended)
    MOVW.P 4(R1), R6           // ai
    MOVH.P 2(R2), R7           // br (sign-extended int16)
    MOVH.P 2(R2), R8           // bi
    MUL  R7, R5, R9            // ar*br
    ASR  $15, R9, R9           // prr
    MUL  R8, R6, R10           // ai*bi
    ASR  $15, R10, R10         // pii
    SUBW R10, R9, R11          // re = prr - pii (32-bit)
    MOVW.P R11, 4(R0)          // dst[2k] = re
    MUL  R8, R5, R12           // ar*bi
    ASR  $15, R12, R12         // pri
    MUL  R7, R6, R13           // ai*br
    ASR  $15, R13, R13         // pir
    ADDW R13, R12, R11         // im = pri + pir (32-bit)
    MOVW.P R11, 4(R0)          // dst[2k+1] = im
    SUB  $1, R4
    CBNZ R4, mul_neon_scalar

mul_neon_done:
    RET

// func mulConjNEON(dst, a []int32, tw []int16)
// Conjugated complex multiply: same four half-products as mulNEON, but the real
// combine adds (prr + pii) and the imaginary combine subtracts (pir - pri).
TEXT ·mulConjNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3     // n (int32 count, even)
    MOVD a_base+24(FP), R1
    MOVD tw_base+48(FP), R2

    LSR  $3, R3, R4            // R4 = n / 8 = 4-complex blocks
    CBZ  R4, mulconj_neon_tail

mulconj_neon_loop:
    VLD2.P 32(R1), [V0.S4, V1.S4]    // V0=ar, V1=ai
    VLD2.P 16(R2), [V2.H4, V3.H4]    // V2=br int16, V3=bi int16
    WORD $0x0F10A444           // SXTL V4.4S, V2.4H
    WORD $0x0F10A465           // SXTL V5.4S, V3.4H
    WORD $0x0EA4C010           // SMULL V16.2D, V0.2S, V4.2S    (prr)
    WORD $0x4EA4C011           // SMULL2 V17.2D, V0.4S, V4.4S
    WORD $0x4F710610           // SSHR V16.2D, V16.2D, #15
    WORD $0x4F710631           // SSHR V17.2D, V17.2D, #15
    WORD $0x0EA12A06           // XTN V6.2S, V16.2D    (prr)
    WORD $0x4EA12A26           // XTN2 V6.4S, V17.2D
    WORD $0x0EA5C030           // SMULL V16.2D, V1.2S, V5.2S    (pii)
    WORD $0x4EA5C031           // SMULL2 V17.2D, V1.4S, V5.4S
    WORD $0x4F710610           // SSHR V16.2D, V16.2D, #15
    WORD $0x4F710631           // SSHR V17.2D, V17.2D, #15
    WORD $0x0EA12A07           // XTN V7.2S, V16.2D    (pii)
    WORD $0x4EA12A27           // XTN2 V7.4S, V17.2D
    WORD $0x0EA5C010           // SMULL V16.2D, V0.2S, V5.2S    (pri)
    WORD $0x4EA5C011           // SMULL2 V17.2D, V0.4S, V5.4S
    WORD $0x4F710610           // SSHR V16.2D, V16.2D, #15
    WORD $0x4F710631           // SSHR V17.2D, V17.2D, #15
    WORD $0x0EA12A08           // XTN V8.2S, V16.2D    (pri)
    WORD $0x4EA12A28           // XTN2 V8.4S, V17.2D
    WORD $0x0EA4C030           // SMULL V16.2D, V1.2S, V4.2S    (pir)
    WORD $0x4EA4C031           // SMULL2 V17.2D, V1.4S, V4.4S
    WORD $0x4F710610           // SSHR V16.2D, V16.2D, #15
    WORD $0x4F710631           // SSHR V17.2D, V17.2D, #15
    WORD $0x0EA12A09           // XTN V9.2S, V16.2D    (pir)
    WORD $0x4EA12A29           // XTN2 V9.4S, V17.2D
    VADD V7.S4, V6.S4, V10.S4        // re = prr + pii
    VSUB V8.S4, V9.S4, V11.S4        // im = pir - pri
    VST2.P [V10.S4, V11.S4], 32(R0)  // re-interleave -> dst
    SUB  $1, R4
    CBNZ R4, mulconj_neon_loop

mulconj_neon_tail:
    AND  $7, R3, R4            // leftover int32 (even)
    LSR  $1, R4, R4            // leftover complex count
    CBZ  R4, mulconj_neon_done

mulconj_neon_scalar:
    MOVW.P 4(R1), R5           // ar
    MOVW.P 4(R1), R6           // ai
    MOVH.P 2(R2), R7           // br
    MOVH.P 2(R2), R8           // bi
    MUL  R7, R5, R9            // ar*br
    ASR  $15, R9, R9           // prr
    MUL  R8, R6, R10           // ai*bi
    ASR  $15, R10, R10         // pii
    ADDW R10, R9, R11          // re = prr + pii (32-bit)
    MOVW.P R11, 4(R0)          // dst[2k] = re
    MUL  R8, R5, R12           // ar*bi
    ASR  $15, R12, R12         // pri
    MUL  R7, R6, R13           // ai*br
    ASR  $15, R13, R13         // pir
    SUBW R12, R13, R11         // im = pir - pri (32-bit)
    MOVW.P R11, 4(R0)          // dst[2k+1] = im
    SUB  $1, R4
    CBNZ R4, mulconj_neon_scalar

mulconj_neon_done:
    RET
