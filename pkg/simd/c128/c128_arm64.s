//go:build arm64

#include "textflag.h"

// ============================================================================
// ARM64 NEON IMPLEMENTATIONS (128-bit, 1x complex128 per iteration)
// ============================================================================
//
// complex128 layout: [real, imag] pairs
// NEON processes 128 bits = 2 float64 = 1 complex128 per iteration
//
// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i

// func mulNEON(dst, a, b []complex128)
TEXT ·mulNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0      // dst pointer
    MOVD dst_len+8(FP), R1       // length
    MOVD a_base+24(FP), R2       // a pointer
    MOVD b_base+48(FP), R3       // b pointer

    CBZ  R1, mul_neon_done

mul_neon_loop:
    // Load one complex128 from a and b
    VLD1 (R2), [V0.D2]           // V0 = [ar, ai]
    VLD1 (R3), [V1.D2]           // V1 = [br, bi]

    // Duplicate real part of a: V2 = [ar, ar]
    VDUP V0.D[0], V2.D2

    // Duplicate imag part of a: V3 = [ai, ai]
    VDUP V0.D[1], V3.D2

    // Swap b: V4 = [bi, br]
    VEXT $8, V1.B16, V1.B16, V4.B16

    // V2 = ar * b = [ar*br, ar*bi]
    FMULD V1.D2, V2.D2, V2.D2

    // V5 = ai * swapped_b = [ai*bi, ai*br]
    FMULD V4.D2, V3.D2, V5.D2

    // result_real = ar*br - ai*bi
    // result_imag = ar*bi + ai*br
    // V2[0] = V2[0] - V5[0], V2[1] = V2[1] + V5[1]
    // Use FSUB for real, FADD for imag
    FSUBD V5.D[0], V2.D[0], V6.D[0]
    FADDD V5.D[1], V2.D[1], V6.D[1]

    VST1 [V6.D2], (R0)

    ADD  $16, R2
    ADD  $16, R3
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, mul_neon_loop

mul_neon_done:
    RET

// func mulConjNEON(dst, a, b []complex128)
TEXT ·mulConjNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2
    MOVD b_base+48(FP), R3

    CBZ  R1, mulconj_neon_done

mulconj_neon_loop:
    VLD1 (R2), [V0.D2]           // [ar, ai]
    VLD1 (R3), [V1.D2]           // [br, bi]

    VDUP V0.D[0], V2.D2          // [ar, ar]
    VDUP V0.D[1], V3.D2          // [ai, ai]

    // For a * conj(b) = [ar*br + ai*bi, ai*br - ar*bi]
    // V4 = ar * b = [ar*br, ar*bi]
    FMULD V1.D2, V2.D2, V4.D2

    // V5 = ai * b = [ai*br, ai*bi]
    FMULD V1.D2, V3.D2, V5.D2

    // result_real = ar*br + ai*bi
    // result_imag = ai*br - ar*bi
    FADDD V5.D[1], V4.D[0], V6.D[0]    // ar*br + ai*bi
    FSUBD V4.D[1], V5.D[0], V6.D[1]    // ai*br - ar*bi

    VST1 [V6.D2], (R0)

    ADD  $16, R2
    ADD  $16, R3
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, mulconj_neon_loop

mulconj_neon_done:
    RET

// func scaleNEON(dst, a []complex128, s complex128)
TEXT ·scaleNEON(SB), NOSPLIT, $0-64
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2
    // s is at s+48(FP) (real) and s+56(FP) (imag)

    FMOVD s+48(FP), F16          // sr
    FMOVD s+56(FP), F17          // si

    VDUP F16, V6.D2              // [sr, sr]
    VDUP F17, V7.D2              // [si, si]

    CBZ  R1, scale_neon_done

scale_neon_loop:
    VLD1 (R2), [V0.D2]           // [ar, ai]

    VDUP V0.D[0], V2.D2          // [ar, ar]
    VDUP V0.D[1], V3.D2          // [ai, ai]

    // V4 = ar * [sr, si] = [ar*sr, ar*si]
    // V5 = ai * [si, sr] = [ai*si, ai*sr]
    // We need: [ar*sr - ai*si, ar*si + ai*sr]

    // Create [sr, si] and [si, sr]
    VMOV V6.D[0], V4.D[0]
    VMOV V7.D[0], V4.D[1]        // V4 = [sr, si]

    VMOV V7.D[0], V5.D[0]
    VMOV V6.D[0], V5.D[1]        // V5 = [si, sr]

    FMULD V4.D2, V2.D2, V2.D2    // [ar*sr, ar*si]
    FMULD V5.D2, V3.D2, V3.D2    // [ai*si, ai*sr]

    FSUBD V3.D[0], V2.D[0], V8.D[0]    // ar*sr - ai*si
    FADDD V3.D[1], V2.D[1], V8.D[1]    // ar*si + ai*sr

    VST1 [V8.D2], (R0)

    ADD  $16, R2
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, scale_neon_loop

scale_neon_done:
    RET

// func addNEON(dst, a, b []complex128)
TEXT ·addNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2
    MOVD b_base+48(FP), R3

    CBZ  R1, add_neon_done

add_neon_loop:
    VLD1 (R2), [V0.D2]
    VLD1 (R3), [V1.D2]
    FADDD V0.D2, V1.D2, V2.D2
    VST1 [V2.D2], (R0)
    ADD  $16, R2
    ADD  $16, R3
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, add_neon_loop

add_neon_done:
    RET

// func subNEON(dst, a, b []complex128)
TEXT ·subNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2
    MOVD b_base+48(FP), R3

    CBZ  R1, sub_neon_done

sub_neon_loop:
    VLD1 (R2), [V0.D2]
    VLD1 (R3), [V1.D2]
    FSUBD V1.D2, V0.D2, V2.D2
    VST1 [V2.D2], (R0)
    ADD  $16, R2
    ADD  $16, R3
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, sub_neon_loop

sub_neon_done:
    RET
