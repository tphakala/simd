//go:build amd64

#include "textflag.h"

// ============================================================================
// AVX+FMA IMPLEMENTATIONS (256-bit, 4x complex64 per iteration)
// ============================================================================
//
// complex64 layout: [real, imag] pairs in memory (8 bytes per complex64)
// YMM register holds 8 float32 = 4 complex64: [r0,i0,r1,i1,r2,i2,r3,i3]
// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i

// func mulAVX(dst, a, b []complex64)
TEXT ·mulAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX              // Process 4 complex64 per iteration
    JZ   mul_avx_remainder

mul_avx_loop4:
    // Load 4 complex64: Y0 = [ar0,ai0,ar1,ai1,ar2,ai2,ar3,ai3]
    VMOVUPS (SI), Y0
    VMOVUPS (DI), Y1         // Y1 = [br0,bi0,br1,bi1,br2,bi2,br3,bi3]

    // Deinterleave a: get real parts duplicated
    VMOVSLDUP Y0, Y2         // Y2 = [ar0,ar0,ar1,ar1,ar2,ar2,ar3,ar3]
    VMOVSHDUP Y0, Y3         // Y3 = [ai0,ai0,ai1,ai1,ai2,ai2,ai3,ai3]

    // Swap b pairs for cross products: [bi,br,bi,br,...]
    VSHUFPS $0xB1, Y1, Y1, Y4  // Y4 = [bi0,br0,bi1,br1,bi2,br2,bi3,br3]

    // Compute products
    VMULPS Y1, Y2, Y2        // Y2 = [ar*br, ar*bi, ar*br, ar*bi, ...]
    VMULPS Y4, Y3, Y5        // Y5 = [ai*bi, ai*br, ai*bi, ai*br, ...]

    // Combine: real = ar*br - ai*bi, imag = ar*bi + ai*br
    // Use VADDSUBPS: subtracts odd elements, adds even elements
    // But we need: sub even (real), add odd (imag) - opposite of ADDSUBPS
    // So use VFMSUBADD231PS or separate add/sub with blend
    VSUBPS Y5, Y2, Y6        // Y6 = [ar*br-ai*bi, ar*bi-ai*br, ...]
    VADDPS Y5, Y2, Y7        // Y7 = [ar*br+ai*bi, ar*bi+ai*br, ...]

    // Blend: take even from Y6 (sub result), odd from Y7 (add result)
    VBLENDPS $0xAA, Y7, Y6, Y2  // 0xAA = 10101010, take lanes 1,3,5,7 from Y7

    VMOVUPS Y2, (DX)

    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  mul_avx_loop4

mul_avx_remainder:
    ANDQ $3, CX
    JZ   mul_avx_done

mul_avx_tail:
    // Handle single complex64 using XMM registers
    VMOVSD (SI), X0          // Load one complex64 (8 bytes)
    VMOVSD (DI), X1

    // X0 = [ar, ai, ?, ?], X1 = [br, bi, ?, ?]
    VMOVSLDUP X0, X2         // X2 = [ar, ar, ?, ?]
    VMOVSHDUP X0, X3         // X3 = [ai, ai, ?, ?]
    VSHUFPS $0xB1, X1, X1, X4  // X4 = [bi, br, ?, ?]

    VMULPS X1, X2, X2        // X2 = [ar*br, ar*bi, ?, ?]
    VMULPS X4, X3, X5        // X5 = [ai*bi, ai*br, ?, ?]

    VSUBPS X5, X2, X6        // X6 = [ar*br-ai*bi, ar*bi-ai*br, ?, ?]
    VADDPS X5, X2, X7        // X7 = [ar*br+ai*bi, ar*bi+ai*br, ?, ?]

    VBLENDPS $0x02, X7, X6, X2  // Take lane 1 from X7

    VMOVSD X2, (DX)

    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  mul_avx_tail

mul_avx_done:
    VZEROUPPER
    RET

// func mulConjAVX(dst, a, b []complex64)
// a * conj(b) = (ar + ai*i)(br - bi*i) = (ar*br + ai*bi) + (ai*br - ar*bi)*i
TEXT ·mulConjAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   mulconj_avx_remainder

mulconj_avx_loop4:
    VMOVUPS (SI), Y0
    VMOVUPS (DI), Y1

    VMOVSLDUP Y0, Y2         // [ar, ar, ...]
    VMOVSHDUP Y0, Y3         // [ai, ai, ...]
    VSHUFPS $0xB1, Y1, Y1, Y4  // [bi, br, ...]

    VMULPS Y1, Y2, Y2        // [ar*br, ar*bi, ...]
    VMULPS Y4, Y3, Y5        // [ai*bi, ai*br, ...]

    // For conj: real = ar*br + ai*bi, imag = ai*br - ar*bi
    VADDPS Y5, Y2, Y6        // [ar*br+ai*bi, ar*bi+ai*br, ...]
    VSUBPS Y2, Y5, Y7        // [ai*bi-ar*br, ai*br-ar*bi, ...]

    // Blend: even from Y6 (add result), odd from Y7 (sub result)
    VBLENDPS $0xAA, Y7, Y6, Y2

    VMOVUPS Y2, (DX)

    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  mulconj_avx_loop4

mulconj_avx_remainder:
    ANDQ $3, CX
    JZ   mulconj_avx_done

mulconj_avx_tail:
    VMOVSD (SI), X0
    VMOVSD (DI), X1

    VMOVSLDUP X0, X2
    VMOVSHDUP X0, X3
    VSHUFPS $0xB1, X1, X1, X4

    VMULPS X1, X2, X2
    VMULPS X4, X3, X5

    VADDPS X5, X2, X6
    VSUBPS X2, X5, X7

    VBLENDPS $0x02, X7, X6, X2

    VMOVSD X2, (DX)

    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  mulconj_avx_tail

mulconj_avx_done:
    VZEROUPPER
    RET

// func scaleAVX(dst, a []complex64, s complex64)
TEXT ·scaleAVX(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Load scalar s and broadcast
    // s is at offset 48 (complex64 = 8 bytes)
    VMOVSD s+48(FP), X8      // X8 = [sr, si, ?, ?]
    // Broadcast to YMM: [sr,si,sr,si,sr,si,sr,si]
    VBROADCASTSD X8, Y1

    // Create swapped [si,sr,si,sr,...] for cross products
    VSHUFPS $0xB1, Y1, Y1, Y4

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   scale_avx_remainder

scale_avx_loop4:
    VMOVUPS (SI), Y0

    VMOVSLDUP Y0, Y2         // [ar, ar, ...]
    VMOVSHDUP Y0, Y3         // [ai, ai, ...]

    VMULPS Y1, Y2, Y2        // [ar*sr, ar*si, ...]
    VMULPS Y4, Y3, Y5        // [ai*si, ai*sr, ...]

    VSUBPS Y5, Y2, Y6        // [ar*sr-ai*si, ar*si-ai*sr]
    VADDPS Y5, Y2, Y7        // [ar*sr+ai*si, ar*si+ai*sr]

    VBLENDPS $0xAA, Y7, Y6, Y2

    VMOVUPS Y2, (DX)

    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  scale_avx_loop4

scale_avx_remainder:
    ANDQ $3, CX
    JZ   scale_avx_done

scale_avx_tail:
    VMOVSD (SI), X0
    VMOVSD s+48(FP), X1
    VSHUFPS $0xB1, X1, X1, X4

    VMOVSLDUP X0, X2
    VMOVSHDUP X0, X3

    VMULPS X1, X2, X2
    VMULPS X4, X3, X5

    VSUBPS X5, X2, X6
    VADDPS X5, X2, X7

    VBLENDPS $0x02, X7, X6, X2

    VMOVSD X2, (DX)

    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  scale_avx_tail

scale_avx_done:
    VZEROUPPER
    RET

// func addAVX(dst, a, b []complex64)
TEXT ·addAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   add_avx_remainder

add_avx_loop4:
    VMOVUPS (SI), Y0
    VMOVUPS (DI), Y1
    VADDPS Y0, Y1, Y2
    VMOVUPS Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  add_avx_loop4

add_avx_remainder:
    ANDQ $3, CX
    JZ   add_avx_done

add_avx_tail:
    VMOVSD (SI), X0
    VMOVSD (DI), X1
    VADDPS X0, X1, X2
    VMOVSD X2, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  add_avx_tail

add_avx_done:
    VZEROUPPER
    RET

// func subAVX(dst, a, b []complex64)
TEXT ·subAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   sub_avx_remainder

sub_avx_loop4:
    VMOVUPS (SI), Y0
    VMOVUPS (DI), Y1
    VSUBPS Y1, Y0, Y2
    VMOVUPS Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  sub_avx_loop4

sub_avx_remainder:
    ANDQ $3, CX
    JZ   sub_avx_done

sub_avx_tail:
    VMOVSD (SI), X0
    VMOVSD (DI), X1
    VSUBPS X1, X0, X2
    VMOVSD X2, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  sub_avx_tail

sub_avx_done:
    VZEROUPPER
    RET

// ============================================================================
// AVX-512 IMPLEMENTATIONS (512-bit, 8x complex64 per iteration)
// ============================================================================

// func mulAVX512(dst, a, b []complex64)
TEXT ·mulAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX              // Process 8 complex64 per iteration
    JZ   mul_avx512_remainder

mul_avx512_loop8:
    VMOVUPS (SI), Z0         // 8 complex64
    VMOVUPS (DI), Z1

    // Deinterleave
    VMOVSLDUP Z0, Z2         // Broadcast real parts
    VMOVSHDUP Z0, Z3         // Broadcast imag parts
    VSHUFPS $0xB1, Z1, Z1, Z4  // Swap pairs

    VMULPS Z1, Z2, Z2        // [ar*br, ar*bi, ...]
    VMULPS Z4, Z3, Z5        // [ai*bi, ai*br, ...]

    VSUBPS Z5, Z2, Z6
    VADDPS Z5, Z2, Z7

    // Blend with mask 0xAAAA (alternating)
    MOVW $0xAAAA, R8
    KMOVW R8, K1
    VBLENDMPS Z7, Z6, K1, Z2

    VMOVUPS Z2, (DX)

    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, DX
    DECQ AX
    JNZ  mul_avx512_loop8

mul_avx512_remainder:
    ANDQ $7, CX
    JZ   mul_avx512_done

mul_avx512_tail:
    VMOVSD (SI), X0
    VMOVSD (DI), X1

    VMOVSLDUP X0, X2
    VMOVSHDUP X0, X3
    VSHUFPS $0xB1, X1, X1, X4

    VMULPS X1, X2, X2
    VMULPS X4, X3, X5

    VSUBPS X5, X2, X6
    VADDPS X5, X2, X7

    VBLENDPS $0x02, X7, X6, X2

    VMOVSD X2, (DX)

    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  mul_avx512_tail

mul_avx512_done:
    VZEROUPPER
    RET

// func mulConjAVX512(dst, a, b []complex64)
TEXT ·mulConjAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   mulconj_avx512_remainder

mulconj_avx512_loop8:
    VMOVUPS (SI), Z0
    VMOVUPS (DI), Z1

    VMOVSLDUP Z0, Z2
    VMOVSHDUP Z0, Z3
    VSHUFPS $0xB1, Z1, Z1, Z4

    VMULPS Z1, Z2, Z2
    VMULPS Z4, Z3, Z5

    VADDPS Z5, Z2, Z6
    VSUBPS Z2, Z5, Z7

    MOVW $0xAAAA, R8
    KMOVW R8, K1
    VBLENDMPS Z7, Z6, K1, Z2

    VMOVUPS Z2, (DX)

    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, DX
    DECQ AX
    JNZ  mulconj_avx512_loop8

mulconj_avx512_remainder:
    ANDQ $7, CX
    JZ   mulconj_avx512_done

mulconj_avx512_tail:
    VMOVSD (SI), X0
    VMOVSD (DI), X1

    VMOVSLDUP X0, X2
    VMOVSHDUP X0, X3
    VSHUFPS $0xB1, X1, X1, X4

    VMULPS X1, X2, X2
    VMULPS X4, X3, X5

    VADDPS X5, X2, X6
    VSUBPS X2, X5, X7

    VBLENDPS $0x02, X7, X6, X2

    VMOVSD X2, (DX)

    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  mulconj_avx512_tail

mulconj_avx512_done:
    VZEROUPPER
    RET

// func scaleAVX512(dst, a []complex64, s complex64)
TEXT ·scaleAVX512(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Broadcast scalar to ZMM
    VBROADCASTSD s+48(FP), Z1
    VSHUFPS $0xB1, Z1, Z1, Z4

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   scale_avx512_remainder

scale_avx512_loop8:
    VMOVUPS (SI), Z0

    VMOVSLDUP Z0, Z2
    VMOVSHDUP Z0, Z3

    VMULPS Z1, Z2, Z2
    VMULPS Z4, Z3, Z5

    VSUBPS Z5, Z2, Z6
    VADDPS Z5, Z2, Z7

    MOVW $0xAAAA, R8
    KMOVW R8, K1
    VBLENDMPS Z7, Z6, K1, Z2

    VMOVUPS Z2, (DX)

    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  scale_avx512_loop8

scale_avx512_remainder:
    ANDQ $7, CX
    JZ   scale_avx512_done

scale_avx512_tail:
    VMOVSD (SI), X0
    VMOVSD s+48(FP), X1
    VSHUFPS $0xB1, X1, X1, X4

    VMOVSLDUP X0, X2
    VMOVSHDUP X0, X3

    VMULPS X1, X2, X2
    VMULPS X4, X3, X5

    VSUBPS X5, X2, X6
    VADDPS X5, X2, X7

    VBLENDPS $0x02, X7, X6, X2

    VMOVSD X2, (DX)

    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  scale_avx512_tail

scale_avx512_done:
    VZEROUPPER
    RET

// func addAVX512(dst, a, b []complex64)
TEXT ·addAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   add_avx512_remainder

add_avx512_loop8:
    VMOVUPS (SI), Z0
    VMOVUPS (DI), Z1
    VADDPS Z0, Z1, Z2
    VMOVUPS Z2, (DX)
    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, DX
    DECQ AX
    JNZ  add_avx512_loop8

add_avx512_remainder:
    ANDQ $7, CX
    JZ   add_avx512_done

add_avx512_tail:
    VMOVSD (SI), X0
    VMOVSD (DI), X1
    VADDPS X0, X1, X2
    VMOVSD X2, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  add_avx512_tail

add_avx512_done:
    VZEROUPPER
    RET

// func subAVX512(dst, a, b []complex64)
TEXT ·subAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   sub_avx512_remainder

sub_avx512_loop8:
    VMOVUPS (SI), Z0
    VMOVUPS (DI), Z1
    VSUBPS Z1, Z0, Z2
    VMOVUPS Z2, (DX)
    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, DX
    DECQ AX
    JNZ  sub_avx512_loop8

sub_avx512_remainder:
    ANDQ $7, CX
    JZ   sub_avx512_done

sub_avx512_tail:
    VMOVSD (SI), X0
    VMOVSD (DI), X1
    VSUBPS X1, X0, X2
    VMOVSD X2, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  sub_avx512_tail

sub_avx512_done:
    VZEROUPPER
    RET

// ============================================================================
// SSE2 IMPLEMENTATIONS (128-bit, 2x complex64 per iteration)
// ============================================================================

// func mulSSE2(dst, a, b []complex64)
TEXT ·mulSSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $1, AX              // Process 2 complex64 per iteration
    JZ   mul_sse2_remainder

mul_sse2_loop2:
    MOVUPS (SI), X0          // 2 complex64: [ar0,ai0,ar1,ai1]
    MOVUPS (DI), X1          // [br0,bi0,br1,bi1]

    MOVSLDUP X0, X2          // [ar0,ar0,ar1,ar1]
    MOVSHDUP X0, X3          // [ai0,ai0,ai1,ai1]
    MOVAPS X1, X4
    SHUFPS $0xB1, X4, X4     // [bi0,br0,bi1,br1]

    MULPS X1, X2             // [ar*br, ar*bi, ...]
    MULPS X4, X3             // [ai*bi, ai*br, ...]

    MOVAPS X2, X5
    SUBPS X3, X5             // [ar*br-ai*bi, ar*bi-ai*br, ...]
    ADDPS X3, X2             // [ar*br+ai*bi, ar*bi+ai*br, ...]

    BLENDPS $0x0A, X2, X5    // Take lanes 1,3 from X2

    MOVUPS X5, (DX)

    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ AX
    JNZ  mul_sse2_loop2

mul_sse2_remainder:
    ANDQ $1, CX
    JZ   mul_sse2_done

    // Handle single complex64
    MOVSD (SI), X0           // [ar, ai, ?, ?]
    MOVSD (DI), X1           // [br, bi, ?, ?]

    MOVSLDUP X0, X2
    MOVSHDUP X0, X3
    MOVAPS X1, X4
    SHUFPS $0xB1, X4, X4

    MULPS X1, X2
    MULPS X4, X3

    MOVAPS X2, X5
    SUBPS X3, X5
    ADDPS X3, X2

    BLENDPS $0x02, X2, X5

    MOVSD X5, (DX)

mul_sse2_done:
    RET

// func mulConjSSE2(dst, a, b []complex64)
TEXT ·mulConjSSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   mulconj_sse2_remainder

mulconj_sse2_loop2:
    MOVUPS (SI), X0
    MOVUPS (DI), X1

    MOVSLDUP X0, X2
    MOVSHDUP X0, X3
    MOVAPS X1, X4
    SHUFPS $0xB1, X4, X4

    MULPS X1, X2
    MULPS X4, X3

    MOVAPS X2, X5
    ADDPS X3, X5             // [ar*br+ai*bi, ar*bi+ai*br, ...]
    MOVAPS X3, X6
    SUBPS X2, X6             // [ai*bi-ar*br, ai*br-ar*bi, ...]

    BLENDPS $0x0A, X6, X5    // Take lanes 1,3 from X6

    MOVUPS X5, (DX)

    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ AX
    JNZ  mulconj_sse2_loop2

mulconj_sse2_remainder:
    ANDQ $1, CX
    JZ   mulconj_sse2_done

    MOVSD (SI), X0
    MOVSD (DI), X1

    MOVSLDUP X0, X2
    MOVSHDUP X0, X3
    MOVAPS X1, X4
    SHUFPS $0xB1, X4, X4

    MULPS X1, X2
    MULPS X4, X3

    MOVAPS X2, X5
    ADDPS X3, X5
    MOVAPS X3, X6
    SUBPS X2, X6

    BLENDPS $0x02, X6, X5

    MOVSD X5, (DX)

mulconj_sse2_done:
    RET

// func scaleSSE2(dst, a []complex64, s complex64)
TEXT ·scaleSSE2(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Load scalar and broadcast to [sr,si,sr,si]
    MOVSD s+48(FP), X6
    MOVLHPS X6, X6           // X6 = [sr,si,sr,si]
    MOVAPS X6, X7
    SHUFPS $0xB1, X7, X7     // X7 = [si,sr,si,sr]

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   scale_sse2_remainder

scale_sse2_loop2:
    MOVUPS (SI), X0

    MOVSLDUP X0, X2
    MOVSHDUP X0, X3

    MULPS X6, X2
    MULPS X7, X3

    MOVAPS X2, X4
    SUBPS X3, X4
    ADDPS X3, X2

    BLENDPS $0x0A, X2, X4

    MOVUPS X4, (DX)

    ADDQ $16, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  scale_sse2_loop2

scale_sse2_remainder:
    ANDQ $1, CX
    JZ   scale_sse2_done

    MOVSD (SI), X0
    MOVSD s+48(FP), X1
    SHUFPS $0xB1, X1, X1

    MOVSLDUP X0, X2
    MOVSHDUP X0, X3

    MULPS X6, X2
    MULPS X1, X3

    MOVAPS X2, X4
    SUBPS X3, X4
    ADDPS X3, X2

    BLENDPS $0x02, X2, X4

    MOVSD X4, (DX)

scale_sse2_done:
    RET

// func addSSE2(dst, a, b []complex64)
TEXT ·addSSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   add_sse2_remainder

add_sse2_loop2:
    MOVUPS (SI), X0
    MOVUPS (DI), X1
    ADDPS X1, X0
    MOVUPS X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ AX
    JNZ  add_sse2_loop2

add_sse2_remainder:
    ANDQ $1, CX
    JZ   add_sse2_done

    MOVSD (SI), X0
    MOVSD (DI), X1
    ADDPS X1, X0
    MOVSD X0, (DX)

add_sse2_done:
    RET

// func subSSE2(dst, a, b []complex64)
TEXT ·subSSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   sub_sse2_remainder

sub_sse2_loop2:
    MOVUPS (SI), X0
    MOVUPS (DI), X1
    SUBPS X1, X0
    MOVUPS X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ AX
    JNZ  sub_sse2_loop2

sub_sse2_remainder:
    ANDQ $1, CX
    JZ   sub_sse2_done

    MOVSD (SI), X0
    MOVSD (DI), X1
    SUBPS X1, X0
    MOVSD X0, (DX)

sub_sse2_done:
    RET

// ============================================================================
// ABS - COMPLEX MAGNITUDE: |a + bi| = sqrt(a^2 + b^2)
// ============================================================================

// func absAVX512(dst []float32, a []complex64)
// Process using YMM registers since VHADDPS doesn't exist for ZMM
TEXT ·absAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $2, AX              // Process 4 complex64 per iteration
    JZ   abs_avx512_remainder

abs_avx512_loop4:
    // Load 4 complex64 = 8 float32
    VMOVUPS (SI), Y0         // [r0,i0,r1,i1,r2,i2,r3,i3]

    // Square all elements
    VMULPS Y0, Y0, Y0        // [r0^2,i0^2,r1^2,i1^2,r2^2,i2^2,r3^2,i3^2]

    // Horizontal add pairs
    // After VHADDPS Y0,Y0,Y0:
    //   Lane 0: [mag0, mag1, mag0, mag1]
    //   Lane 1: [mag2, mag3, mag2, mag3]
    VHADDPS Y0, Y0, Y0

    // Extract high 128 bits and pack results
    // X0 (low 128 bits) = [mag0, mag1, mag0, mag1]
    // X1 (high 128 bits) = [mag2, mag3, mag2, mag3]
    VEXTRACTF128 $1, Y0, X1
    // VSHUFPS to get [mag0, mag1, mag2, mag3]
    VSHUFPS $0x44, X1, X0, X0

    // Take sqrt
    VSQRTPS X0, X0

    // Store 4 float32 results
    VMOVUPS X0, (DX)

    ADDQ $32, SI             // 4 complex64 = 32 bytes
    ADDQ $16, DX             // 4 float32 = 16 bytes
    DECQ AX
    JNZ  abs_avx512_loop4

abs_avx512_remainder:
    ANDQ $3, CX
    JZ   abs_avx512_done

abs_avx512_tail:
    // Load one complex64
    VMOVSD (SI), X0          // [r, i, ?, ?]
    VMULPS X0, X0, X0        // [r^2, i^2, ?, ?]
    VHADDPS X0, X0, X0       // [r^2+i^2, r^2+i^2, ...]
    VSQRTSS X0, X0, X0
    VMOVSS X0, (DX)

    ADDQ $8, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  abs_avx512_tail

abs_avx512_done:
    VZEROUPPER
    RET

// func absAVX(dst []float32, a []complex64)
TEXT ·absAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $2, AX              // Process 4 complex64 per iteration
    JZ   abs_avx_remainder

abs_avx_loop4:
    // Load 4 complex64 = 8 float32
    VMOVUPS (SI), Y0         // [r0,i0,r1,i1,r2,i2,r3,i3]

    // Square all
    VMULPS Y0, Y0, Y0        // [r0^2,i0^2,r1^2,i1^2,r2^2,i2^2,r3^2,i3^2]

    // Horizontal add pairs
    // After VHADDPS: Lane 0: [mag0, mag1, mag0, mag1], Lane 1: [mag2, mag3, mag2, mag3]
    VHADDPS Y0, Y0, Y0

    // Extract and pack results
    VEXTRACTF128 $1, Y0, X1
    VSHUFPS $0x44, X1, X0, X0  // [mag0, mag1, mag2, mag3]

    // Take sqrt
    VSQRTPS X0, X0

    // Store 4 float32 results
    VMOVUPS X0, (DX)

    ADDQ $32, SI             // 4 complex64 = 32 bytes
    ADDQ $16, DX             // 4 float32 = 16 bytes
    DECQ AX
    JNZ  abs_avx_loop4

abs_avx_remainder:
    ANDQ $3, CX
    JZ   abs_avx_done

abs_avx_tail:
    VMOVSD (SI), X0
    VMULPS X0, X0, X0
    VHADDPS X0, X0, X0
    VSQRTSS X0, X0, X0
    VMOVSS X0, (DX)

    ADDQ $8, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  abs_avx_tail

abs_avx_done:
    VZEROUPPER
    RET

// func absSSE2(dst []float32, a []complex64)
// Uses SSE2-only instructions (no HADDPS which is SSE3)
TEXT ·absSSE2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    TESTQ CX, CX
    JZ   abs_sse2_done

abs_sse2_loop:
    MOVSD (SI), X0           // X0 = [r, i, 0, 0]
    MULPS X0, X0             // X0 = [r^2, i^2, 0, 0]
    // SSE2-compatible horizontal add (no HADDPS)
    MOVAPS X0, X1            // X1 = [r^2, i^2, 0, 0]
    SHUFPS $0x01, X1, X1     // X1 = [i^2, r^2, r^2, r^2]
    ADDSS X1, X0             // X0[0] = r^2 + i^2
    SQRTSS X0, X0
    MOVSS X0, (DX)

    ADDQ $8, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  abs_sse2_loop

abs_sse2_done:
    RET

// ============================================================================
// ABSSQ - MAGNITUDE SQUARED: |a + bi|^2 = a^2 + b^2
// ============================================================================

// func absSqAVX512(dst []float32, a []complex64)
// Process using YMM registers since VHADDPS doesn't exist for ZMM
TEXT ·absSqAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   abssq_avx512_remainder

abssq_avx512_loop4:
    VMOVUPS (SI), Y0
    VMULPS Y0, Y0, Y0
    // After VHADDPS: Lane 0: [mag0, mag1, mag0, mag1], Lane 1: [mag2, mag3, mag2, mag3]
    VHADDPS Y0, Y0, Y0

    // Extract and pack results
    VEXTRACTF128 $1, Y0, X1
    VSHUFPS $0x44, X1, X0, X0  // [mag0, mag1, mag2, mag3]

    // Store 4 float32 results (no sqrt needed)
    VMOVUPS X0, (DX)

    ADDQ $32, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  abssq_avx512_loop4

abssq_avx512_remainder:
    ANDQ $3, CX
    JZ   abssq_avx512_done

abssq_avx512_tail:
    VMOVSD (SI), X0
    VMULPS X0, X0, X0
    VHADDPS X0, X0, X0
    VMOVSS X0, (DX)

    ADDQ $8, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  abssq_avx512_tail

abssq_avx512_done:
    VZEROUPPER
    RET

// func absSqAVX(dst []float32, a []complex64)
TEXT ·absSqAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   abssq_avx_remainder

abssq_avx_loop4:
    VMOVUPS (SI), Y0
    VMULPS Y0, Y0, Y0
    // After VHADDPS: Lane 0: [mag0, mag1, mag0, mag1], Lane 1: [mag2, mag3, mag2, mag3]
    VHADDPS Y0, Y0, Y0

    // Extract and pack results
    VEXTRACTF128 $1, Y0, X1
    VSHUFPS $0x44, X1, X0, X0  // [mag0, mag1, mag2, mag3]

    // Store 4 float32 (no sqrt)
    VMOVUPS X0, (DX)

    ADDQ $32, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  abssq_avx_loop4

abssq_avx_remainder:
    ANDQ $3, CX
    JZ   abssq_avx_done

abssq_avx_tail:
    VMOVSD (SI), X0
    VMULPS X0, X0, X0
    VHADDPS X0, X0, X0
    VMOVSS X0, (DX)

    ADDQ $8, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  abssq_avx_tail

abssq_avx_done:
    VZEROUPPER
    RET

// func absSqSSE2(dst []float32, a []complex64)
// Uses SSE2-only instructions (no HADDPS which is SSE3)
TEXT ·absSqSSE2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    TESTQ CX, CX
    JZ   abssq_sse2_done

abssq_sse2_loop:
    MOVSD (SI), X0           // X0 = [r, i, 0, 0]
    MULPS X0, X0             // X0 = [r^2, i^2, 0, 0]
    // SSE2-compatible horizontal add (no HADDPS)
    MOVAPS X0, X1            // X1 = [r^2, i^2, 0, 0]
    SHUFPS $0x01, X1, X1     // X1 = [i^2, r^2, r^2, r^2]
    ADDSS X1, X0             // X0[0] = r^2 + i^2
    MOVSS X0, (DX)

    ADDQ $8, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  abssq_sse2_loop

abssq_sse2_done:
    RET

// ============================================================================
// CONJ - COMPLEX CONJUGATE: conj(a + bi) = a - bi
// ============================================================================

// Sign mask for negating imaginary parts: negate odd float32 elements
DATA signmask64<>+0x00(SB)/4, $0x00000000  // +0 for real
DATA signmask64<>+0x04(SB)/4, $0x80000000  // -0 for imag (flip sign)
DATA signmask64<>+0x08(SB)/4, $0x00000000
DATA signmask64<>+0x0c(SB)/4, $0x80000000
DATA signmask64<>+0x10(SB)/4, $0x00000000
DATA signmask64<>+0x14(SB)/4, $0x80000000
DATA signmask64<>+0x18(SB)/4, $0x00000000
DATA signmask64<>+0x1c(SB)/4, $0x80000000
GLOBL signmask64<>(SB), RODATA|NOPTR, $32

// func conjAVX512(dst, a []complex64)
TEXT ·conjAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Load sign mask for negating imag parts
    VBROADCASTSD signmask64<>+0(SB), Z15  // [+0,-0,+0,-0,...]

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   conj_avx512_remainder

conj_avx512_loop8:
    VMOVUPS (SI), Z0
    VXORPS Z15, Z0, Z0       // Negate imag parts via sign bit XOR
    VMOVUPS Z0, (DX)

    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  conj_avx512_loop8

conj_avx512_remainder:
    ANDQ $7, CX
    JZ   conj_avx512_done

conj_avx512_tail:
    VMOVSD (SI), X0
    VXORPS X15, X0, X0
    VMOVSD X0, (DX)

    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  conj_avx512_tail

conj_avx512_done:
    VZEROUPPER
    RET

// func conjAVX(dst, a []complex64)
TEXT ·conjAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Load sign mask
    VBROADCASTSD signmask64<>+0(SB), Y15

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   conj_avx_remainder

conj_avx_loop4:
    VMOVUPS (SI), Y0
    VXORPS Y15, Y0, Y0
    VMOVUPS Y0, (DX)

    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  conj_avx_loop4

conj_avx_remainder:
    ANDQ $3, CX
    JZ   conj_avx_done

conj_avx_tail:
    VMOVSD (SI), X0
    VXORPS X15, X0, X0
    VMOVSD X0, (DX)

    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  conj_avx_tail

conj_avx_done:
    VZEROUPPER
    RET

// func conjSSE2(dst, a []complex64)
TEXT ·conjSSE2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Load sign mask
    MOVUPS signmask64<>+0(SB), X15

    TESTQ CX, CX
    JZ   conj_sse2_done

conj_sse2_loop:
    MOVSD (SI), X0
    XORPS X15, X0
    MOVSD X0, (DX)

    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  conj_sse2_loop

conj_sse2_done:
    RET

// ============================================================================
// FROMREAL - CONVERT REAL TO COMPLEX: complex(x, 0)
// ============================================================================

// func fromRealAVX512(dst []complex64, src []float32)
// Process using YMM registers since VPERM2F128 doesn't support ZMM
TEXT ·fromRealAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    // Process 4 elements per iteration using YMM
    MOVQ CX, AX
    SHRQ $2, AX
    JZ   fromreal_avx512_remainder

    VXORPS Y15, Y15, Y15      // Zero for interleaving

fromreal_avx512_loop4:
    // Load 4 float32 from src
    VMOVUPS (SI), X0          // X0 = [r0,r1,r2,r3]

    // Interleave with zeros
    VUNPCKLPS X15, X0, X1     // X1 = [r0,0,r1,0]
    VUNPCKHPS X15, X0, X2     // X2 = [r2,0,r3,0]

    // Combine into YMM
    VINSERTF128 $1, X2, Y1, Y0

    // Store 4 complex64 = 32 bytes
    VMOVUPS Y0, (DX)

    ADDQ $16, SI              // 4 float32 = 16 bytes
    ADDQ $32, DX              // 4 complex64 = 32 bytes
    DECQ AX
    JNZ  fromreal_avx512_loop4

fromreal_avx512_remainder:
    ANDQ $3, CX
    JZ   fromreal_avx512_done

fromreal_avx512_tail:
    VMOVSS (SI), X0           // Load one float32
    VXORPS X1, X1, X1         // Zero
    VUNPCKLPS X1, X0, X0      // Interleave: [r, 0, ?, ?]
    VMOVSD X0, (DX)           // Store one complex64

    ADDQ $4, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  fromreal_avx512_tail

fromreal_avx512_done:
    VZEROUPPER
    RET

// func fromRealAVX(dst []complex64, src []float32)
TEXT ·fromRealAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   fromreal_avx_remainder

    VXORPS Y15, Y15, Y15

fromreal_avx_loop4:
    // Load 4 float32
    VMOVUPS (SI), X0          // X0 = [r0,r1,r2,r3]

    // Interleave with zeros
    VUNPCKLPS X15, X0, X1     // X1 = [r0,0,r1,0]
    VUNPCKHPS X15, X0, X2     // X2 = [r2,0,r3,0]

    // Combine into YMM
    VINSERTF128 $1, X2, Y1, Y0

    // Store 4 complex64 = 32 bytes
    VMOVUPS Y0, (DX)

    ADDQ $16, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  fromreal_avx_loop4

fromreal_avx_remainder:
    ANDQ $3, CX
    JZ   fromreal_avx_done

fromreal_avx_tail:
    VMOVSS (SI), X0
    VXORPS X1, X1, X1
    VUNPCKLPS X1, X0, X0
    VMOVSD X0, (DX)

    ADDQ $4, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  fromreal_avx_tail

fromreal_avx_done:
    VZEROUPPER
    RET

// func fromRealSSE2(dst []complex64, src []float32)
TEXT ·fromRealSSE2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    XORPS X15, X15            // Zero register

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   fromreal_sse2_remainder

fromreal_sse2_loop2:
    // Load 2 float32
    MOVSD (SI), X0            // X0 = [r0, r1, ?, ?]

    // Interleave with zeros
    UNPCKLPS X15, X0          // X0 = [r0, 0, r1, 0]

    // Store 2 complex64 = 16 bytes
    MOVUPS X0, (DX)

    ADDQ $8, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  fromreal_sse2_loop2

fromreal_sse2_remainder:
    ANDQ $1, CX
    JZ   fromreal_sse2_done

    MOVSS (SI), X0
    UNPCKLPS X15, X0
    MOVSD X0, (DX)

fromreal_sse2_done:
    RET
