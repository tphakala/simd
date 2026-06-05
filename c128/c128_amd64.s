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

    // Compute cross products, then fused multiply-subtract/add
    VMULPD Y4, Y3, Y5              // Y5 = [ai*bi, ai*br, ...]
    VFMADDSUB213PD Y5, Y1, Y2      // Y2 = Y1*Y2 ± Y5
                                    // even: ar*br - ai*bi (real)
                                    // odd:  ar*bi + ai*br (imag)

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

    VMULPD X4, X3, X5              // X5 = [ai*bi, ai*br]
    VFMADDSUB213PD X5, X1, X2      // X2 = X1*X2 ± X5

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

    // For conj: real = ar*br + ai*bi, imag = ai*br - ar*bi
    // Compute ar×b_swapped, then FMA with ai×b, producing result in swapped order
    VMULPD Y4, Y2, Y5              // Y5 = [ar*bi, ar*br, ...]
    VFMADDSUB213PD Y5, Y1, Y3      // Y3 = Y1*Y3 ± Y5
                                    // even: ai*br - ar*bi, odd: ai*bi + ar*br
    VSHUFPD $0x05, Y3, Y3, Y2      // Swap pairs → [ar*br+ai*bi, ai*br-ar*bi, ...]

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

    VMULPD X4, X2, X5
    VFMADDSUB213PD X5, X1, X3
    VSHUFPD $0x01, X3, X3, X2

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
    VMOVSD s_real+48(FP), X8
    VMOVSD s_imag+56(FP), X9
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

    VMULPD Y4, Y3, Y5              // Y5 = [ai*si, ai*sr, ...]
    VFMADDSUB213PD Y5, Y1, Y2      // Y2 = Y1*Y2 ± Y5
                                    // even: ar*sr - ai*si, odd: ar*si + ai*sr

    VMOVUPD Y2, (DX)

    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  scale_avx_loop2

scale_avx_remainder:
    ANDQ $1, CX
    JZ   scale_avx_done

    VMOVUPD (SI), X0
    VMOVSD s_real+48(FP), X8
    VMOVSD s_imag+56(FP), X9
    VUNPCKLPD X9, X8, X1
    VSHUFPD $0x01, X1, X1, X4

    VMOVDDUP X0, X2
    VSHUFPD $0x03, X0, X0, X3

    VMULPD X4, X3, X5
    VFMADDSUB213PD X5, X1, X2

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

    VMULPD Z4, Z3, Z5              // Z5 = [ai*bi, ai*br, ...]
    VFMADDSUB213PD Z5, Z1, Z2      // Z2 = Z1*Z2 ± Z5

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

    VMULPD X4, X3, X5
    VFMADDSUB213PD X5, X1, X2

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

    VMULPD Z4, Z2, Z5              // Z5 = [ar*bi, ar*br, ...]
    VFMADDSUB213PD Z5, Z1, Z3      // Z3 = Z1*Z3 ± Z5
    VSHUFPD $0x55, Z3, Z3, Z2      // Swap pairs

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

    VMULPD X4, X2, X5
    VFMADDSUB213PD X5, X1, X3
    VSHUFPD $0x01, X3, X3, X2

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

    VBROADCASTSD s_real+48(FP), Z6
    VBROADCASTSD s_imag+56(FP), Z7
    VSHUFPD $0x00, Z7, Z6, Z1    // [sr, si, sr, si, ...]
    VSHUFPD $0xFF, Z6, Z7, Z4    // [si, sr, si, sr, ...]

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   scale_avx512_remainder

scale_avx512_loop4:
    VMOVUPD (SI), Z0

    VMOVDDUP Z0, Z2
    VSHUFPD $0xFF, Z0, Z0, Z3

    VMULPD Z4, Z3, Z5              // Z5 = [ai*si, ai*sr, ...]
    VFMADDSUB213PD Z5, Z1, Z2      // Z2 = Z1*Z2 ± Z5

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
    VMOVSD s_real+48(FP), X8
    VMOVSD s_imag+56(FP), X9
    VUNPCKLPD X9, X8, X1
    VSHUFPD $0x01, X1, X1, X4

    VMOVDDUP X0, X2
    VSHUFPD $0x03, X0, X0, X3

    VMULPD X4, X3, X5
    VFMADDSUB213PD X5, X1, X2

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

    SHUFPD $0x2, X2, X5          // re<-X5[0] (sub), im<-X2[1] (add); SSE2 form of BLENDPD $0x02

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

    SHUFPD $0x2, X6, X5          // re<-X5[0] (add), im<-X6[1] (sub); SSE2 form of BLENDPD $0x02

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

    MOVSD s_real+48(FP), X6
    MOVSD s_imag+56(FP), X7
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

    SHUFPD $0x2, X2, X4          // re<-X4[0] (sub), im<-X2[1] (add); SSE2 form of BLENDPD $0x02

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
TEXT ·absAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Mask for VCOMPRESSPD: pick even lanes (0,2,4,6) from duplicated sums.
    // Use KMOVW (AVX512F) instead of KMOVB (AVX512DQ) for baseline compatibility.
    MOVL $0x55, AX
    KMOVW AX, K1

    MOVQ CX, AX
    SHRQ $2, AX            // Process 4 elements per iteration
    JZ   abs_avx512_loop2_check

abs_avx512_loop4:
    // Load 4 complex128 = 8 float64: [r0, i0, r1, i1, r2, i2, r3, i3]
    VMOVUPD (SI), Z0

    // Compute [r²+i²] with shuffle+add (no horizontal add dependency).
    VMULPD Z0, Z0, Z0
    VSHUFPD $0x55, Z0, Z0, Z1
    VADDPD Z0, Z1, Z2      // [s0, s0, s1, s1, s2, s2, s3, s3]

    // Pack duplicated sums into contiguous lanes: [s0, s1, s2, s3].
    VCOMPRESSPD Z2, K1, Z2

    VSQRTPD Y2, Y2
    VMOVUPD Y2, (DX)

    ADDQ $64, SI           // 4 complex128 = 64 bytes
    ADDQ $32, DX           // 4 float64 = 32 bytes
    DECQ AX
    JNZ  abs_avx512_loop4

abs_avx512_loop2_check:
    ANDQ $3, CX
    MOVQ CX, AX
    SHRQ $1, AX
    JZ   abs_avx512_remainder1

abs_avx512_loop2:
    // Load 2 complex128 = 4 float64: [r0, i0, r1, i1]
    VMOVUPD (SI), Y0
    VMULPD Y0, Y0, Y0
    VSHUFPD $0x55, Y0, Y0, Y1
    VADDPD Y0, Y1, Y0      // [s0, s0, s1, s1]

    VEXTRACTF128 $1, Y0, X1
    VUNPCKLPD X1, X0, X0   // [s0, s1]
    VSQRTPD X0, X0
    VMOVUPD X0, (DX)

    ADDQ $32, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  abs_avx512_loop2

abs_avx512_remainder1:
    ANDQ $1, CX
    JZ   abs_avx512_done

    // Handle 1 remaining element using XMM
    VMOVUPD (SI), X0       // Load one complex128 (VEX-encoded: avoid AVX-SSE transition)
    VMOVDDUP X0, X1        // X1 = [real, real]
    VSHUFPD $1, X0, X0, X2 // X2 = [imag, imag]

    VMULPD X1, X1, X1      // real²
    VMULPD X2, X2, X2      // imag²
    VADDPD X2, X1, X1      // r² + i²
    VSQRTPD X1, X1
    VMOVSD X1, (DX)

abs_avx512_done:
    VZEROUPPER
    RET

// func absAVX(dst []float64, a []complex128)
TEXT ·absAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   abs_avx_remainder

abs_avx_loop2:
    // Load 2 complex128 = 4 float64: [r0, i0, r1, i1]
    VMOVUPD (SI), Y0
    VMULPD Y0, Y0, Y0
    VSHUFPD $0x55, Y0, Y0, Y1
    VADDPD Y0, Y1, Y0      // [s0, s0, s1, s1]

    VEXTRACTF128 $1, Y0, X1
    VUNPCKLPD X1, X0, X0   // [s0, s1]
    VSQRTPD X0, X0
    VMOVUPD X0, (DX)

    ADDQ $32, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  abs_avx_loop2

abs_avx_remainder:
    ANDQ $1, CX
    JZ   abs_avx_done

    // Handle 1 remaining element
    VMOVUPD (SI), X0       // VEX-encoded: avoid AVX-SSE transition penalty
    VMOVDDUP X0, X1
    VSHUFPD $1, X0, X0, X2
    VMULPD X1, X1, X1
    VMULPD X2, X2, X2
    VADDPD X2, X1, X1
    VSQRTPD X1, X1
    VMOVSD X1, (DX)

abs_avx_done:
    VZEROUPPER
    RET

// func absSSE2(dst []float64, a []complex128)
TEXT ·absSSE2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    TESTQ CX, CX
    JZ   abs_sse2_done

abs_sse2_loop:
    MOVUPD (SI), X0        // Load one complex128 [real, imag]

    // X0 = [real, imag]
    MOVAPD X0, X1          // X1[0]=real; high lane discarded by MOVSD store (SSE2 form of MOVDDUP)
    SHUFPD $1, X0, X0      // X0[0]=imag

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
// Mirrors absAVX512 (4 complex128 per iteration) without the final VSQRTPD.
TEXT ·absSqAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Mask for VCOMPRESSPD: pick even lanes (0,2,4,6) from duplicated sums.
    // Use KMOVW (AVX512F) instead of KMOVB (AVX512DQ) for baseline compatibility.
    MOVL $0x55, AX
    KMOVW AX, K1

    MOVQ CX, AX
    SHRQ $2, AX            // Process 4 elements per iteration
    JZ   abssq_avx512_loop2_check

abssq_avx512_loop4:
    // Load 4 complex128 = 8 float64: [r0, i0, r1, i1, r2, i2, r3, i3]
    VMOVUPD (SI), Z0

    // Compute [r²+i²] with shuffle+add (no horizontal add dependency).
    VMULPD Z0, Z0, Z0
    VSHUFPD $0x55, Z0, Z0, Z1
    VADDPD Z0, Z1, Z2      // [s0, s0, s1, s1, s2, s2, s3, s3]

    // Pack duplicated sums into contiguous lanes: [s0, s1, s2, s3].
    VCOMPRESSPD Z2, K1, Z2

    VMOVUPD Y2, (DX)       // no sqrt for AbsSq

    ADDQ $64, SI           // 4 complex128 = 64 bytes
    ADDQ $32, DX           // 4 float64 = 32 bytes
    DECQ AX
    JNZ  abssq_avx512_loop4

abssq_avx512_loop2_check:
    ANDQ $3, CX
    MOVQ CX, AX
    SHRQ $1, AX
    JZ   abssq_avx512_remainder1

abssq_avx512_loop2:
    // Load 2 complex128 = 4 float64: [r0, i0, r1, i1]
    VMOVUPD (SI), Y0
    VMULPD Y0, Y0, Y0
    VSHUFPD $0x55, Y0, Y0, Y1
    VADDPD Y0, Y1, Y0      // [s0, s0, s1, s1]

    VEXTRACTF128 $1, Y0, X1
    VUNPCKLPD X1, X0, X0   // [s0, s1]
    VMOVUPD X0, (DX)       // no sqrt for AbsSq

    ADDQ $32, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  abssq_avx512_loop2

abssq_avx512_remainder1:
    ANDQ $1, CX
    JZ   abssq_avx512_done

    // Handle 1 remaining element using XMM
    VMOVUPD (SI), X0       // Load one complex128 (VEX-encoded: avoid AVX-SSE transition)
    VMOVDDUP X0, X1        // X1 = [real, real]
    VSHUFPD $1, X0, X0, X2 // X2 = [imag, imag]

    VMULPD X1, X1, X1      // real²
    VMULPD X2, X2, X2      // imag²
    VADDPD X2, X1, X1      // r² + i²
    VMOVSD X1, (DX)        // no sqrt for AbsSq

abssq_avx512_done:
    VZEROUPPER
    RET

// func absSqAVX(dst []float64, a []complex128)
// Mirrors absAVX (2 complex128 per iteration) without the final VSQRTPD.
TEXT ·absSqAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $1, AX            // Process 2 complex128 per iteration
    JZ   abssq_avx_remainder

abssq_avx_loop2:
    // Load 2 complex128 = 4 float64: [r0, i0, r1, i1]
    VMOVUPD (SI), Y0
    VMULPD Y0, Y0, Y0
    VSHUFPD $0x55, Y0, Y0, Y1
    VADDPD Y0, Y1, Y0      // [s0, s0, s1, s1]

    VEXTRACTF128 $1, Y0, X1
    VUNPCKLPD X1, X0, X0   // [s0, s1]
    VMOVUPD X0, (DX)       // no sqrt for AbsSq

    ADDQ $32, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  abssq_avx_loop2

abssq_avx_remainder:
    ANDQ $1, CX
    JZ   abssq_avx_done

    // Handle 1 remaining element
    VMOVUPD (SI), X0       // VEX-encoded: avoid AVX-SSE transition penalty
    VMOVDDUP X0, X1
    VSHUFPD $1, X0, X0, X2
    VMULPD X1, X1, X1
    VMULPD X2, X2, X2
    VADDPD X2, X1, X1
    VMOVSD X1, (DX)        // no sqrt for AbsSq

abssq_avx_done:
    VZEROUPPER
    RET

// func absSqSSE2(dst []float64, a []complex128)
TEXT ·absSqSSE2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    TESTQ CX, CX
    JZ   abssq_sse2_done

abssq_sse2_loop:
    MOVUPD (SI), X0

    MULPD X0, X0           // [r², i²]

    MOVAPD X0, X1          // X1[0]=r²; high lane discarded by MOVSD store (SSE2 form of MOVDDUP)
    SHUFPD $1, X0, X0      // X0[0]=i²
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
TEXT ·conjAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VMOVUPD conjSignMask<>(SB), Z2  // Z2 = [0,-0,0,-0,0,-0,0,-0]

    MOVQ CX, AX
    SHRQ $2, AX            // AX = groups of 4 complex128
    JZ   conj_avx512_loop2_check

conj_avx512_loop4:
    VMOVUPD (SI), Z0
    VPXORQ  Z2, Z0, Z0     // flip imaginary sign bits (VPXORQ is AVX512F; VXORPD needs AVX512DQ)
    VMOVUPD Z0, (DX)
    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  conj_avx512_loop4

conj_avx512_loop2_check:
    ANDQ $3, CX
    TESTQ $2, CX
    JZ   conj_avx512_remainder1

    VMOVUPD (SI), Y0
    VXORPD  Y2, Y0, Y0     // Y2 = low half of Z2
    VMOVUPD Y0, (DX)
    ADDQ $32, SI
    ADDQ $32, DX

conj_avx512_remainder1:
    ANDQ $1, CX
    JZ   conj_avx512_done

    VMOVUPD (SI), X0
    VXORPD  X2, X0, X0     // X2 = low quarter of Z2
    VMOVUPD X0, (DX)

conj_avx512_done:
    VZEROUPPER
    RET

// conjSignMask flips the sign bit of the imaginary lane of each packed
// complex128: XOR [re0, im0, re1, im1] -> [re0, -im0, re1, -im1].
// 64 bytes: low 32 bytes serve AVX (YMM), full 64 bytes serve AVX-512 (ZMM).
DATA conjSignMask<>+0(SB)/8, $0x0000000000000000
DATA conjSignMask<>+8(SB)/8, $0x8000000000000000
DATA conjSignMask<>+16(SB)/8, $0x0000000000000000
DATA conjSignMask<>+24(SB)/8, $0x8000000000000000
DATA conjSignMask<>+32(SB)/8, $0x0000000000000000
DATA conjSignMask<>+40(SB)/8, $0x8000000000000000
DATA conjSignMask<>+48(SB)/8, $0x0000000000000000
DATA conjSignMask<>+56(SB)/8, $0x8000000000000000
GLOBL conjSignMask<>(SB), RODATA|NOPTR, $64

// func conjAVX(dst, a []complex128)
TEXT ·conjAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VMOVUPD conjSignMask<>(SB), Y2 // [0.0, -0.0, 0.0, -0.0]

    MOVQ CX, AX
    SHRQ $1, AX            // AX = pairs (2 complex128 per iter, YMM)
    JZ   conj_avx_remainder

conj_avx_loop2:
    VMOVUPD (SI), Y0       // [re0, im0, re1, im1]
    VXORPD  Y2, Y0, Y0     // [re0, -im0, re1, -im1]
    VMOVUPD Y0, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  conj_avx_loop2

conj_avx_remainder:
    ANDQ $1, CX
    JZ   conj_avx_done

    VMOVUPD (SI), X0       // [re, im]
    VXORPD  X2, X0, X0     // [re, -im] (X2 = low lane of Y2)
    VMOVUPD X0, (DX)

conj_avx_done:
    VZEROUPPER
    RET

// func conjSSE2(dst, a []complex128)
TEXT ·conjSSE2(SB), NOSPLIT, $0-48
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

// ============================================================================
// FROMREAL - CONVERT REAL TO COMPLEX: complex(x, 0)
// ============================================================================

// func fromRealAVX512(dst []complex128, src []float64)
// Processes 2 complex128 per iteration using YMM. Mirrors fromRealAVX: a
// dedicated ZMM kernel buys nothing for this store-bandwidth-bound interleave,
// exactly as c64's fromRealAVX512 reuses the YMM approach.
TEXT ·fromRealAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   fromreal_avx512_remainder

    VXORPD Y15, Y15, Y15          // zero for interleaving

fromreal_avx512_loop2:
    VMOVUPD (SI), X0              // X0 = [r0, r1]
    VUNPCKLPD X15, X0, X1         // X1 = [r0, 0]
    VUNPCKHPD X15, X0, X2         // X2 = [r1, 0]
    VINSERTF128 $1, X2, Y1, Y0    // Y0 = [r0, 0, r1, 0]
    VMOVUPD Y0, (DX)             // store 2 complex128 = 32 bytes
    ADDQ $16, SI                 // 2 float64 = 16 bytes
    ADDQ $32, DX                 // 2 complex128 = 32 bytes
    DECQ AX
    JNZ  fromreal_avx512_loop2

fromreal_avx512_remainder:
    ANDQ $1, CX
    JZ   fromreal_avx512_done

    VMOVSD (SI), X0              // X0 = [r, 0] (load form zeros the high 64)
    VMOVUPD X0, (DX)            // store one complex128

fromreal_avx512_done:
    VZEROUPPER
    RET

// func fromRealAVX(dst []complex128, src []float64)
TEXT ·fromRealAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   fromreal_avx_remainder

    VXORPD Y15, Y15, Y15

fromreal_avx_loop2:
    VMOVUPD (SI), X0              // X0 = [r0, r1]
    VUNPCKLPD X15, X0, X1         // X1 = [r0, 0]
    VUNPCKHPD X15, X0, X2         // X2 = [r1, 0]
    VINSERTF128 $1, X2, Y1, Y0    // Y0 = [r0, 0, r1, 0]
    VMOVUPD Y0, (DX)
    ADDQ $16, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  fromreal_avx_loop2

fromreal_avx_remainder:
    ANDQ $1, CX
    JZ   fromreal_avx_done

    VMOVSD (SI), X0
    VMOVUPD X0, (DX)

fromreal_avx_done:
    VZEROUPPER
    RET

// func fromRealSSE2(dst []complex128, src []float64)
// Each complex128 fills a full XMM, so MOVSD's load-form zeroing of the upper
// 64 bits yields [r, 0.0] directly; one element per iteration.
TEXT ·fromRealSSE2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    TESTQ CX, CX
    JZ   fromreal_sse2_done

fromreal_sse2_loop:
    MOVSD (SI), X0               // X0 = [r, 0.0] (load form zeros the high 64)
    MOVUPS X0, (DX)             // store one complex128 = 16 bytes
    ADDQ $8, SI
    ADDQ $16, DX
    DECQ CX
    JNZ  fromreal_sse2_loop

fromreal_sse2_done:
    RET
