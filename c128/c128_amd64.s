//go:build amd64

#include "textflag.h"

// ============================================================================
// AVX+FMA IMPLEMENTATIONS (256-bit, 2x complex128 per iteration)
// ============================================================================
//
// complex128 layout: [real, imag] pairs in memory
// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i

// func mulAVX(dst, a, b []complex128)
TEXT ·mulAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   mul_avx_remainder

mul_avx_loop2:
    VMOVUPD (SI), Y0             // [ar0, ai0, ar1, ai1]
    VMOVUPD (DI), Y1             // [br0, bi0, br1, bi1]

    // Compute real parts: ar*br - ai*bi
    // Compute imag parts: ar*bi + ai*br
    VMOVDDUP Y0, Y2              // [ar, ar, ar, ar]
    VSHUFPD $0x0F, Y0, Y0, Y3    // [ai, ai, ai, ai]
    VSHUFPD $0x05, Y1, Y1, Y4    // [bi, br, bi, br]

    VMULPD Y1, Y2, Y2            // [ar*br, ar*bi, ar*br, ar*bi]
    VMULPD Y4, Y3, Y5            // [ai*bi, ai*br, ai*bi, ai*br]

    // Use separate add/sub then interleave for correct result
    // real = ar*br - ai*bi (need Y2[even] - Y5[even])
    // imag = ar*bi + ai*br (need Y2[odd] + Y5[odd])
    VSUBPD Y5, Y2, Y6            // [ar*br-ai*bi, ar*bi-ai*br, ...]
    VADDPD Y5, Y2, Y7            // [ar*br+ai*bi, ar*bi+ai*br, ...]

    // Blend: take even from Y6, odd from Y7
    VBLENDPD $0x0A, Y7, Y6, Y2   // 0x0A = 0b1010, take lanes 1,3 from Y7

    VMOVUPD Y2, (DX)

    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  mul_avx_loop2

mul_avx_remainder:
    ANDQ $1, CX
    JZ   mul_avx_done

    // Handle single complex128 using XMM registers
    VMOVUPD (SI), X0
    VMOVUPD (DI), X1

    VMOVDDUP X0, X2              // [ar, ar]
    VSHUFPD $0x03, X0, X0, X3    // [ai, ai]
    VSHUFPD $0x01, X1, X1, X4    // [bi, br]

    VMULPD X1, X2, X2            // [ar*br, ar*bi]
    VMULPD X4, X3, X5            // [ai*bi, ai*br]

    VSUBPD X5, X2, X6            // [ar*br-ai*bi, ar*bi-ai*br]
    VADDPD X5, X2, X7            // [ar*br+ai*bi, ar*bi+ai*br]

    VBLENDPD $0x02, X7, X6, X2   // Take lane 1 from X7

    VMOVUPD X2, (DX)

mul_avx_done:
    VZEROUPPER
    RET

// func mulConjAVX(dst, a, b []complex128)
// a * conj(b) = (ar + ai*i)(br - bi*i) = (ar*br + ai*bi) + (ai*br - ar*bi)*i
TEXT ·mulConjAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   mulconj_avx_remainder

mulconj_avx_loop2:
    VMOVUPD (SI), Y0
    VMOVUPD (DI), Y1

    VMOVDDUP Y0, Y2              // [ar, ar, ...]
    VSHUFPD $0x0F, Y0, Y0, Y3    // [ai, ai, ...]
    VSHUFPD $0x05, Y1, Y1, Y4    // [bi, br, ...]

    VMULPD Y1, Y2, Y2            // [ar*br, ar*bi, ...]
    VMULPD Y4, Y3, Y5            // [ai*bi, ai*br, ...]

    // For conj: real = ar*br + ai*bi, imag = ai*br - ar*bi
    VADDPD Y5, Y2, Y6            // [ar*br+ai*bi, ar*bi+ai*br, ...]
    VSUBPD Y2, Y5, Y7            // [ai*bi-ar*br, ai*br-ar*bi, ...]

    // Blend: even from Y6 (add result), odd from Y7 (sub result)
    VBLENDPD $0x0A, Y7, Y6, Y2

    VMOVUPD Y2, (DX)

    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  mulconj_avx_loop2

mulconj_avx_remainder:
    ANDQ $1, CX
    JZ   mulconj_avx_done

    VMOVUPD (SI), X0
    VMOVUPD (DI), X1

    VMOVDDUP X0, X2
    VSHUFPD $0x03, X0, X0, X3
    VSHUFPD $0x01, X1, X1, X4

    VMULPD X1, X2, X2
    VMULPD X4, X3, X5

    VADDPD X5, X2, X6
    VSUBPD X2, X5, X7

    VBLENDPD $0x02, X7, X6, X2

    VMOVUPD X2, (DX)

mulconj_avx_done:
    VZEROUPPER
    RET

// func scaleAVX(dst, a []complex128, s complex128)
TEXT ·scaleAVX(SB), NOSPLIT, $0-64
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Load scalar and create [sr, si, sr, si] pattern
    VMOVSD s+48(FP), X8
    VMOVSD s+56(FP), X9
    VUNPCKLPD X9, X8, X1         // X1 = [sr, si]
    VINSERTF128 $1, X1, Y1, Y1   // Y1 = [sr, si, sr, si]

    // Create swapped [si, sr, si, sr]
    VSHUFPD $0x05, Y1, Y1, Y4

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   scale_avx_remainder

scale_avx_loop2:
    VMOVUPD (SI), Y0

    VMOVDDUP Y0, Y2              // [ar, ar, ...]
    VSHUFPD $0x0F, Y0, Y0, Y3    // [ai, ai, ...]

    VMULPD Y1, Y2, Y2            // [ar*sr, ar*si, ...]
    VMULPD Y4, Y3, Y5            // [ai*si, ai*sr, ...]

    VSUBPD Y5, Y2, Y6            // [ar*sr-ai*si, ar*si-ai*sr]
    VADDPD Y5, Y2, Y7            // [ar*sr+ai*si, ar*si+ai*sr]

    VBLENDPD $0x0A, Y7, Y6, Y2

    VMOVUPD Y2, (DX)

    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  scale_avx_loop2

scale_avx_remainder:
    ANDQ $1, CX
    JZ   scale_avx_done

    VMOVUPD (SI), X0
    VMOVSD s+48(FP), X8
    VMOVSD s+56(FP), X9
    VUNPCKLPD X9, X8, X1
    VSHUFPD $0x01, X1, X1, X4

    VMOVDDUP X0, X2
    VSHUFPD $0x03, X0, X0, X3

    VMULPD X1, X2, X2
    VMULPD X4, X3, X5

    VSUBPD X5, X2, X6
    VADDPD X5, X2, X7

    VBLENDPD $0x02, X7, X6, X2

    VMOVUPD X2, (DX)

scale_avx_done:
    VZEROUPPER
    RET

// func addAVX(dst, a, b []complex128)
TEXT ·addAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   add_avx_remainder

add_avx_loop2:
    VMOVUPD (SI), Y0
    VMOVUPD (DI), Y1
    VADDPD Y0, Y1, Y2
    VMOVUPD Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  add_avx_loop2

add_avx_remainder:
    ANDQ $1, CX
    JZ   add_avx_done

    VMOVUPD (SI), X0
    VMOVUPD (DI), X1
    VADDPD X0, X1, X2
    VMOVUPD X2, (DX)

add_avx_done:
    VZEROUPPER
    RET

// func subAVX(dst, a, b []complex128)
TEXT ·subAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   sub_avx_remainder

sub_avx_loop2:
    VMOVUPD (SI), Y0
    VMOVUPD (DI), Y1
    VSUBPD Y1, Y0, Y2
    VMOVUPD Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  sub_avx_loop2

sub_avx_remainder:
    ANDQ $1, CX
    JZ   sub_avx_done

    VMOVUPD (SI), X0
    VMOVUPD (DI), X1
    VSUBPD X1, X0, X2
    VMOVUPD X2, (DX)

sub_avx_done:
    VZEROUPPER
    RET

// ============================================================================
// AVX-512 IMPLEMENTATIONS (512-bit, 4x complex128 per iteration)
// ============================================================================

// func mulAVX512(dst, a, b []complex128)
TEXT ·mulAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   mul_avx512_remainder

mul_avx512_loop4:
    VMOVUPD (SI), Z0
    VMOVUPD (DI), Z1

    VMOVDDUP Z0, Z2
    VSHUFPD $0xFF, Z0, Z0, Z3
    VSHUFPD $0x55, Z1, Z1, Z4

    VMULPD Z1, Z2, Z2
    VMULPD Z4, Z3, Z5

    VSUBPD Z5, Z2, Z6
    VADDPD Z5, Z2, Z7

    // Blend with mask 0xAA (10101010) for odd lanes
    MOVB $0xAA, R8
    KMOVB R8, K1
    VBLENDMPD Z7, Z6, K1, Z2

    VMOVUPD Z2, (DX)

    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, DX
    DECQ AX
    JNZ  mul_avx512_loop4

mul_avx512_remainder:
    ANDQ $3, CX
    JZ   mul_avx512_done

mul_avx512_tail:
    VMOVUPD (SI), X0
    VMOVUPD (DI), X1

    VMOVDDUP X0, X2
    VSHUFPD $0x03, X0, X0, X3
    VSHUFPD $0x01, X1, X1, X4

    VMULPD X1, X2, X2
    VMULPD X4, X3, X5

    VSUBPD X5, X2, X6
    VADDPD X5, X2, X7

    VBLENDPD $0x02, X7, X6, X2

    VMOVUPD X2, (DX)

    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ CX
    JNZ  mul_avx512_tail

mul_avx512_done:
    VZEROUPPER
    RET

// func mulConjAVX512(dst, a, b []complex128)
TEXT ·mulConjAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   mulconj_avx512_remainder

mulconj_avx512_loop4:
    VMOVUPD (SI), Z0
    VMOVUPD (DI), Z1

    VMOVDDUP Z0, Z2
    VSHUFPD $0xFF, Z0, Z0, Z3
    VSHUFPD $0x55, Z1, Z1, Z4

    VMULPD Z1, Z2, Z2
    VMULPD Z4, Z3, Z5

    VADDPD Z5, Z2, Z6
    VSUBPD Z2, Z5, Z7

    MOVB $0xAA, R8
    KMOVB R8, K1
    VBLENDMPD Z7, Z6, K1, Z2

    VMOVUPD Z2, (DX)

    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, DX
    DECQ AX
    JNZ  mulconj_avx512_loop4

mulconj_avx512_remainder:
    ANDQ $3, CX
    JZ   mulconj_avx512_done

mulconj_avx512_tail:
    VMOVUPD (SI), X0
    VMOVUPD (DI), X1

    VMOVDDUP X0, X2
    VSHUFPD $0x03, X0, X0, X3
    VSHUFPD $0x01, X1, X1, X4

    VMULPD X1, X2, X2
    VMULPD X4, X3, X5

    VADDPD X5, X2, X6
    VSUBPD X2, X5, X7

    VBLENDPD $0x02, X7, X6, X2

    VMOVUPD X2, (DX)

    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ CX
    JNZ  mulconj_avx512_tail

mulconj_avx512_done:
    VZEROUPPER
    RET

// func scaleAVX512(dst, a []complex128, s complex128)
TEXT ·scaleAVX512(SB), NOSPLIT, $0-64
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VBROADCASTSD s+48(FP), Z6
    VBROADCASTSD s+56(FP), Z7
    VSHUFPD $0x00, Z7, Z6, Z1    // [sr, si, sr, si, ...]
    VSHUFPD $0xFF, Z6, Z7, Z4    // [si, sr, si, sr, ...]

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   scale_avx512_remainder

scale_avx512_loop4:
    VMOVUPD (SI), Z0

    VMOVDDUP Z0, Z2
    VSHUFPD $0xFF, Z0, Z0, Z3

    VMULPD Z1, Z2, Z2
    VMULPD Z4, Z3, Z5

    VSUBPD Z5, Z2, Z6
    VADDPD Z5, Z2, Z7

    MOVB $0xAA, R8
    KMOVB R8, K1
    VBLENDMPD Z7, Z6, K1, Z2

    VMOVUPD Z2, (DX)

    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  scale_avx512_loop4

scale_avx512_remainder:
    ANDQ $3, CX
    JZ   scale_avx512_done

scale_avx512_tail:
    VMOVUPD (SI), X0
    VMOVSD s+48(FP), X8
    VMOVSD s+56(FP), X9
    VUNPCKLPD X9, X8, X1
    VSHUFPD $0x01, X1, X1, X4

    VMOVDDUP X0, X2
    VSHUFPD $0x03, X0, X0, X3

    VMULPD X1, X2, X2
    VMULPD X4, X3, X5

    VSUBPD X5, X2, X6
    VADDPD X5, X2, X7

    VBLENDPD $0x02, X7, X6, X2

    VMOVUPD X2, (DX)

    ADDQ $16, SI
    ADDQ $16, DX
    DECQ CX
    JNZ  scale_avx512_tail

scale_avx512_done:
    VZEROUPPER
    RET

// func addAVX512(dst, a, b []complex128)
TEXT ·addAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   add_avx512_remainder

add_avx512_loop4:
    VMOVUPD (SI), Z0
    VMOVUPD (DI), Z1
    VADDPD Z0, Z1, Z2
    VMOVUPD Z2, (DX)
    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, DX
    DECQ AX
    JNZ  add_avx512_loop4

add_avx512_remainder:
    ANDQ $3, CX
    JZ   add_avx512_done

add_avx512_tail:
    VMOVUPD (SI), X0
    VMOVUPD (DI), X1
    VADDPD X0, X1, X2
    VMOVUPD X2, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ CX
    JNZ  add_avx512_tail

add_avx512_done:
    VZEROUPPER
    RET

// func subAVX512(dst, a, b []complex128)
TEXT ·subAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   sub_avx512_remainder

sub_avx512_loop4:
    VMOVUPD (SI), Z0
    VMOVUPD (DI), Z1
    VSUBPD Z1, Z0, Z2
    VMOVUPD Z2, (DX)
    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, DX
    DECQ AX
    JNZ  sub_avx512_loop4

sub_avx512_remainder:
    ANDQ $3, CX
    JZ   sub_avx512_done

sub_avx512_tail:
    VMOVUPD (SI), X0
    VMOVUPD (DI), X1
    VSUBPD X1, X0, X2
    VMOVUPD X2, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ CX
    JNZ  sub_avx512_tail

sub_avx512_done:
    VZEROUPPER
    RET

// ============================================================================
// SSE2/SSE3 IMPLEMENTATIONS (128-bit, 1x complex128 per iteration)
// ============================================================================

// func mulSSE2(dst, a, b []complex128)
TEXT ·mulSSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    TESTQ CX, CX
    JZ   mul_sse2_done

mul_sse2_loop:
    MOVUPD (SI), X0
    MOVUPD (DI), X1

    MOVAPD X0, X2
    SHUFPD $0x00, X2, X2         // [ar, ar]

    MOVAPD X0, X3
    SHUFPD $0x03, X3, X3         // [ai, ai]

    MOVAPD X1, X4
    SHUFPD $0x01, X4, X4         // [bi, br]

    MULPD X1, X2                 // [ar*br, ar*bi]
    MULPD X4, X3                 // [ai*bi, ai*br]

    // Separate add/sub then blend
    MOVAPD X2, X5
    SUBPD X3, X5                 // [ar*br-ai*bi, ar*bi-ai*br]
    ADDPD X3, X2                 // [ar*br+ai*bi, ar*bi+ai*br]

    BLENDPD $0x02, X2, X5        // Take lane 1 from X2 (add result)

    MOVUPD X5, (DX)

    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ CX
    JNZ  mul_sse2_loop

mul_sse2_done:
    RET

// func mulConjSSE2(dst, a, b []complex128)
TEXT ·mulConjSSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    TESTQ CX, CX
    JZ   mulconj_sse2_done

mulconj_sse2_loop:
    MOVUPD (SI), X0
    MOVUPD (DI), X1

    MOVAPD X0, X2
    SHUFPD $0x00, X2, X2

    MOVAPD X0, X3
    SHUFPD $0x03, X3, X3

    MOVAPD X1, X4
    SHUFPD $0x01, X4, X4

    MULPD X1, X2                 // [ar*br, ar*bi]
    MULPD X4, X3                 // [ai*bi, ai*br]

    MOVAPD X2, X5
    ADDPD X3, X5                 // [ar*br+ai*bi, ar*bi+ai*br]
    MOVAPD X3, X6
    SUBPD X2, X6                 // [ai*bi-ar*br, ai*br-ar*bi]

    BLENDPD $0x02, X6, X5        // Take lane 1 from X6

    MOVUPD X5, (DX)

    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ CX
    JNZ  mulconj_sse2_loop

mulconj_sse2_done:
    RET

// func scaleSSE2(dst, a []complex128, s complex128)
TEXT ·scaleSSE2(SB), NOSPLIT, $0-64
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVSD s+48(FP), X6
    MOVSD s+56(FP), X7
    UNPCKLPD X7, X6              // X6 = [sr, si]
    MOVAPD X6, X7
    SHUFPD $0x01, X7, X7         // X7 = [si, sr]

    TESTQ CX, CX
    JZ   scale_sse2_done

scale_sse2_loop:
    MOVUPD (SI), X0

    MOVAPD X0, X2
    SHUFPD $0x00, X2, X2

    MOVAPD X0, X3
    SHUFPD $0x03, X3, X3

    MULPD X6, X2                 // [ar*sr, ar*si]
    MULPD X7, X3                 // [ai*si, ai*sr]

    MOVAPD X2, X4
    SUBPD X3, X4                 // [ar*sr-ai*si, ar*si-ai*sr]
    ADDPD X3, X2                 // [ar*sr+ai*si, ar*si+ai*sr]

    BLENDPD $0x02, X2, X4

    MOVUPD X4, (DX)

    ADDQ $16, SI
    ADDQ $16, DX
    DECQ CX
    JNZ  scale_sse2_loop

scale_sse2_done:
    RET

// func addSSE2(dst, a, b []complex128)
TEXT ·addSSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    TESTQ CX, CX
    JZ   add_sse2_done

add_sse2_loop:
    MOVUPD (SI), X0
    MOVUPD (DI), X1
    ADDPD X1, X0
    MOVUPD X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ CX
    JNZ  add_sse2_loop

add_sse2_done:
    RET

// func subSSE2(dst, a, b []complex128)
TEXT ·subSSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    TESTQ CX, CX
    JZ   sub_sse2_done

sub_sse2_loop:
    MOVUPD (SI), X0
    MOVUPD (DI), X1
    SUBPD X1, X0
    MOVUPD X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ CX
    JNZ  sub_sse2_loop

sub_sse2_done:
    RET

// ============================================================================
// ABS - COMPLEX MAGNITUDE: |a + bi| = sqrt(a² + b²)
// ============================================================================

// func absAVX512(dst []float64, a []complex128)
// Computes |z| = sqrt(real² + imag²) for each complex number
TEXT ·absAVX512(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $1, AX            // Process 2 elements per iteration
    JZ   abs_avx512_remainder

abs_avx512_loop2:
    // Load 2 complex128 = 4 float64: [r0, i0, r1, i1]
    VMOVUPD (SI), Y0

    // Square all elements: [r0², i0², r1², i1²]
    VMULPD Y0, Y0, Y0

    // Horizontal add pairs within each 128-bit lane
    // VHADDPD Y0, Y0, Y0 produces:
    //   Lane 0: [r0²+i0², r0²+i0²]
    //   Lane 1: [r1²+i1², r1²+i1²]
    // Result: [r0²+i0², r0²+i0², r1²+i1², r1²+i1²]
    VHADDPD Y0, Y0, Y0

    // Pack results: elements 0 and 2 contain the sums we need
    // Extract upper 128 bits: X1 = [r1²+i1², r1²+i1²]
    VEXTRACTF128 $1, Y0, X1
    // Unpack low elements: X0 = [r0²+i0², r1²+i1²]
    VUNPCKLPD X1, X0, X0

    // Take sqrt: X0 = [|z0|, |z1|]
    VSQRTPD X0, X0

    // Store 2 results
    VMOVUPD X0, (DX)

    ADDQ $32, SI           // 2 complex128 = 32 bytes
    ADDQ $16, DX           // 2 float64 = 16 bytes
    DECQ AX
    JNZ  abs_avx512_loop2

abs_avx512_remainder:
    ANDQ $1, CX
    JZ   abs_avx512_done

    // Handle 1 remaining element using XMM
abs_avx512_remainder_loop:
    MOVUPD (SI), X0        // Load one complex128

    // X0 = [real, imag]
    VMOVDDUP X0, X1        // X1 = [real, real]
    VSHUFPD $1, X0, X0, X2 // X2 = [imag, imag]

    VMULPD X1, X1, X1      // real²
    VMULPD X2, X2, X2      // imag²
    VADDPD X2, X1, X1      // r² + i²
    VSQRTPD X1, X1

    VMOVSD X1, (DX)

    ADDQ $16, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  abs_avx512_remainder_loop

abs_avx512_done:
    VZEROUPPER
    RET

// func absAVX(dst []float64, a []complex128)
TEXT ·absAVX(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    TESTQ CX, CX
    JZ   abs_avx_done

abs_avx_loop:
    VMOVUPD (SI), X0       // Load one complex128

    // X0 = [real, imag]
    VMOVDDUP X0, X1        // X1 = [real, real]
    VSHUFPD $1, X0, X0, X2 // X2 = [imag, imag]

    VMULPD X1, X1, X1      // real²
    VMULPD X2, X2, X2      // imag²
    VADDPD X2, X1, X1      // r² + i²
    VSQRTPD X1, X1

    VMOVSD X1, (DX)

    ADDQ $16, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  abs_avx_loop

abs_avx_done:
    VZEROUPPER
    RET

// func absSSE2(dst []float64, a []complex128)
TEXT ·absSSE2(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    TESTQ CX, CX
    JZ   abs_sse2_done

abs_sse2_loop:
    MOVUPD (SI), X0        // Load one complex128 [real, imag]

    // X0 = [real, imag]
    MOVDDUP X0, X1         // X1 = [real, real]
    SHUFPD $1, X0, X0      // X0 = [imag, imag]

    MULPD X1, X1           // real²
    MULPD X0, X0           // imag²
    ADDPD X0, X1           // r² + i²
    SQRTPD X1, X1

    MOVSD X1, (DX)

    ADDQ $16, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  abs_sse2_loop

abs_sse2_done:
    RET

// ============================================================================
// ABSSQ - MAGNITUDE SQUARED: |a + bi|² = a² + b²
// ============================================================================

// func absSqAVX512(dst []float64, a []complex128)
// Fixed: Use VHADDPD + proper packing (same approach as absAVX512)
TEXT ·absSqAVX512(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $1, AX            // Process 2 complex128 per iteration
    JZ   abssq_avx512_remainder

abssq_avx512_loop2:
    // Load 2 complex128 = 4 float64: [r0, i0, r1, i1]
    VMOVUPD (SI), Y0

    // Square all elements: [r0², i0², r1², i1²]
    VMULPD Y0, Y0, Y0

    // Horizontal add pairs within each 128-bit lane
    // VHADDPD Y0, Y0, Y0 produces:
    //   Lane 0: [r0²+i0², r0²+i0²]
    //   Lane 1: [r1²+i1², r1²+i1²]
    // Result: [r0²+i0², r0²+i0², r1²+i1², r1²+i1²]
    VHADDPD Y0, Y0, Y0

    // Pack results: elements 0 and 2 contain the sums we need
    // Extract upper 128 bits: X1 = [r1²+i1², r1²+i1²]
    VEXTRACTF128 $1, Y0, X1
    // Unpack low elements: X0 = [r0²+i0², r1²+i1²]
    VUNPCKLPD X1, X0, X0

    // Store 2 results (no sqrt for AbsSq)
    VMOVUPD X0, (DX)

    ADDQ $32, SI           // 2 complex128 = 32 bytes
    ADDQ $16, DX           // 2 float64 = 16 bytes
    DECQ AX
    JNZ  abssq_avx512_loop2

abssq_avx512_remainder:
    ANDQ $1, CX
    JZ   abssq_avx512_done

    // Handle 1 remaining element using XMM
abssq_avx512_remainder_loop:
    MOVUPD (SI), X0        // Load one complex128

    // X0 = [real, imag]
    VMOVDDUP X0, X1        // X1 = [real, real]
    VSHUFPD $1, X0, X0, X2 // X2 = [imag, imag]

    VMULPD X1, X1, X1      // real²
    VMULPD X2, X2, X2      // imag²
    VADDPD X2, X1, X1      // r² + i²

    VMOVSD X1, (DX)

    ADDQ $16, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  abssq_avx512_remainder_loop

abssq_avx512_done:
    VZEROUPPER
    RET

// func absSqAVX(dst []float64, a []complex128)
TEXT ·absSqAVX(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    TESTQ CX, CX
    JZ   abssq_avx_done

abssq_avx_loop:
    VMOVUPD (SI), X0

    VMULPD X0, X0, X1      // [r², i²]

    VMOVDDUP X1, X2        // X2 = [r², r²]
    SHUFPD $1, X1, X1      // X1 = [i², i²]
    VADDPD X1, X2, X2      // r² + i²

    VMOVSD X2, (DX)

    ADDQ $16, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  abssq_avx_loop

abssq_avx_done:
    VZEROUPPER
    RET

// func absSqSSE2(dst []float64, a []complex128)
TEXT ·absSqSSE2(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    TESTQ CX, CX
    JZ   abssq_sse2_done

abssq_sse2_loop:
    MOVUPD (SI), X0

    MULPD X0, X0           // [r², i²]

    MOVDDUP X0, X1         // [r², r²]
    SHUFPD $1, X0, X0      // [i², i²]
    ADDPD X0, X1           // r² + i²

    MOVSD X1, (DX)

    ADDQ $16, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  abssq_sse2_loop

abssq_sse2_done:
    RET

// ============================================================================
// CONJ - COMPLEX CONJUGATE: conj(a + bi) = a - bi
// ============================================================================

// func conjAVX512(dst, a []complex128)
TEXT ·conjAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    TESTQ CX, CX
    JZ   conj_avx512_done

    // Use scalar loop (same as SSE2 but through dispatch)
    // AVX512 doesn't provide a simpler conjugate operation than mixing scalars
conj_avx512_loop:
    MOVUPD (SI), X0        // [real, imag]

    // Create negated copy: [-real, -imag]
    XORPD X1, X1           // Clear X1 = [0, 0]
    SUBPD X0, X1           // X1 = -X0 = [-real, -imag]

    // Use SHUFPD to blend/reconstruct: [real, -imag]
    SHUFPD $2, X1, X0      // X0 = [low(X0)=real, high(X1)=-imag]

    MOVUPD X0, (DX)

    ADDQ $16, SI
    ADDQ $16, DX
    DECQ CX
    JNZ  conj_avx512_loop

conj_avx512_done:
    VZEROUPPER
    RET

// func conjAVX(dst, a []complex128)
TEXT ·conjAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    TESTQ CX, CX
    JZ   conj_avx_done

conj_avx_loop:
    VMOVUPD (SI), X0       // [real, imag]

    // Create negated copy: [-real, -imag]
    VXORPD X1, X1, X1      // Clear X1 = [0, 0]
    VSUBPD X0, X1, X1      // X1 = -X0 = [-real, -imag]

    // Use SHUFPD to blend/reconstruct: [real, -imag]
    SHUFPD $2, X1, X0      // X0 = [low(X0)=real, high(X1)=-imag]

    VMOVUPD X0, (DX)

    ADDQ $16, SI
    ADDQ $16, DX
    DECQ CX
    JNZ  conj_avx_loop

conj_avx_done:
    VZEROUPPER
    RET

// func conjSSE2(dst, a []complex128)
TEXT ·conjSSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    TESTQ CX, CX
    JZ   conj_sse2_done

conj_sse2_loop:
    MOVUPD (SI), X0        // [real, imag]

    // Create negated copy: [-real, -imag]
    XORPD X1, X1           // Clear X1 = [0, 0]
    SUBPD X0, X1           // X1 = -X0 = [-real, -imag]

    // Blend: keep real from X0, imaginary from X1 (negated)
    // For SSE2, we need to manually blend (use SHUFPD for blending)
    SHUFPD $2, X1, X0      // Take low from X0, high from X1: [real, -imag]

    MOVUPD X0, (DX)

    ADDQ $16, SI
    ADDQ $16, DX
    DECQ CX
    JNZ  conj_sse2_loop

conj_sse2_done:
    RET
