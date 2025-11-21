//go:build amd64

#include "textflag.h"

// Constants
DATA absf32mask<>+0x00(SB)/4, $0x7fffffff
DATA absf32mask<>+0x04(SB)/4, $0x7fffffff
DATA absf32mask<>+0x08(SB)/4, $0x7fffffff
DATA absf32mask<>+0x0c(SB)/4, $0x7fffffff
DATA absf32mask<>+0x10(SB)/4, $0x7fffffff
DATA absf32mask<>+0x14(SB)/4, $0x7fffffff
DATA absf32mask<>+0x18(SB)/4, $0x7fffffff
DATA absf32mask<>+0x1c(SB)/4, $0x7fffffff
GLOBL absf32mask<>(SB), RODATA|NOPTR, $32

// func dotProductAVX(a, b []float32) float32
// Processes 8 float32s per iteration (256-bit YMM)
TEXT ·dotProductAVX(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    VXORPS Y0, Y0, Y0          // Accumulator

    MOVQ CX, AX
    SHRQ $3, AX                // len / 8
    JZ   dot32_remainder

dot32_loop8:
    VMOVUPS (SI), Y1
    VMOVUPS (DI), Y2
    VFMADD231PS Y1, Y2, Y0
    ADDQ $32, SI
    ADDQ $32, DI
    DECQ AX
    JNZ  dot32_loop8

dot32_remainder:
    // Reduce Y0 to X0 BEFORE scalar ops (VEX scalar ops zero upper YMM)
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    ANDQ $7, CX
    JZ   dot32_done

dot32_scalar:
    VMOVSS (SI), X1
    VMOVSS (DI), X2
    VFMADD231SS X1, X2, X0
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ CX
    JNZ  dot32_scalar

dot32_done:
    VMOVSS X0, ret+48(FP)
    VZEROUPPER
    RET

// func addAVX(dst, a, b []float32)
TEXT ·addAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   add32_remainder

add32_loop8:
    VMOVUPS (SI), Y0
    VMOVUPS (DI), Y1
    VADDPS Y0, Y1, Y2
    VMOVUPS Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  add32_loop8

add32_remainder:
    ANDQ $7, CX
    JZ   add32_done

add32_scalar:
    VMOVSS (SI), X0
    VADDSS (DI), X0, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  add32_scalar

add32_done:
    VZEROUPPER
    RET

// func subAVX(dst, a, b []float32)
TEXT ·subAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   sub32_remainder

sub32_loop8:
    VMOVUPS (SI), Y0
    VMOVUPS (DI), Y1
    VSUBPS Y1, Y0, Y2
    VMOVUPS Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  sub32_loop8

sub32_remainder:
    ANDQ $7, CX
    JZ   sub32_done

sub32_scalar:
    VMOVSS (SI), X0
    VSUBSS (DI), X0, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  sub32_scalar

sub32_done:
    VZEROUPPER
    RET

// func mulAVX(dst, a, b []float32)
TEXT ·mulAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   mul32_remainder

mul32_loop8:
    VMOVUPS (SI), Y0
    VMOVUPS (DI), Y1
    VMULPS Y0, Y1, Y2
    VMOVUPS Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  mul32_loop8

mul32_remainder:
    ANDQ $7, CX
    JZ   mul32_done

mul32_scalar:
    VMOVSS (SI), X0
    VMULSS (DI), X0, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  mul32_scalar

mul32_done:
    VZEROUPPER
    RET

// func divAVX(dst, a, b []float32)
TEXT ·divAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   div32_remainder

div32_loop8:
    VMOVUPS (SI), Y0
    VMOVUPS (DI), Y1
    VDIVPS Y1, Y0, Y2
    VMOVUPS Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  div32_loop8

div32_remainder:
    ANDQ $7, CX
    JZ   div32_done

div32_scalar:
    VMOVSS (SI), X0
    VDIVSS (DI), X0, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  div32_scalar

div32_done:
    VZEROUPPER
    RET

// func scaleAVX(dst, a []float32, s float32)
TEXT ·scaleAVX(SB), NOSPLIT, $0-52
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VBROADCASTSS s+48(FP), Y1

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   scale32_remainder

scale32_loop8:
    VMOVUPS (SI), Y0
    VMULPS Y0, Y1, Y2
    VMOVUPS Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  scale32_loop8

scale32_remainder:
    ANDQ $7, CX
    JZ   scale32_done
    VMOVSS s+48(FP), X1

scale32_scalar:
    VMOVSS (SI), X0
    VMULSS X0, X1, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  scale32_scalar

scale32_done:
    VZEROUPPER
    RET

// func addScalarAVX(dst, a []float32, s float32)
TEXT ·addScalarAVX(SB), NOSPLIT, $0-52
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VBROADCASTSS s+48(FP), Y1

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   addsc32_remainder

addsc32_loop8:
    VMOVUPS (SI), Y0
    VADDPS Y0, Y1, Y2
    VMOVUPS Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  addsc32_loop8

addsc32_remainder:
    ANDQ $7, CX
    JZ   addsc32_done
    VMOVSS s+48(FP), X1

addsc32_scalar:
    VMOVSS (SI), X0
    VADDSS X0, X1, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  addsc32_scalar

addsc32_done:
    VZEROUPPER
    RET

// func sumAVX(a []float32) float32
TEXT ·sumAVX(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VXORPS Y0, Y0, Y0

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   sum32_remainder

sum32_loop8:
    VMOVUPS (SI), Y1
    VADDPS Y0, Y1, Y0
    ADDQ $32, SI
    DECQ AX
    JNZ  sum32_loop8

sum32_remainder:
    // Reduce Y0 to X0 BEFORE scalar ops
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    ANDQ $7, CX
    JZ   sum32_done

sum32_scalar:
    VMOVSS (SI), X1
    VADDSS X0, X1, X0
    ADDQ $4, SI
    DECQ CX
    JNZ  sum32_scalar

sum32_done:
    VMOVSS X0, ret+24(FP)
    VZEROUPPER
    RET

// func minAVX(a []float32) float32
TEXT ·minAVX(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VMOVUPS (SI), Y0
    ADDQ $32, SI
    SUBQ $8, CX

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   min32_reduce_first

min32_loop8:
    VMOVUPS (SI), Y1
    VMINPS Y0, Y1, Y0
    ADDQ $32, SI
    DECQ AX
    JNZ  min32_loop8

min32_reduce_first:
    // Reduce Y0 to scalar BEFORE scalar ops
    VEXTRACTF128 $1, Y0, X1
    VMINPS X0, X1, X0
    VPERMILPS $0x0E, X0, X1
    VMINPS X0, X1, X0
    VPERMILPS $0x01, X0, X1
    VMINSS X0, X1, X0

    ANDQ $7, CX
    JZ   min32_done

min32_scalar:
    VMOVSS (SI), X1
    VMINSS X0, X1, X0
    ADDQ $4, SI
    DECQ CX
    JNZ  min32_scalar

min32_done:
    VMOVSS X0, ret+24(FP)
    VZEROUPPER
    RET

// func maxAVX(a []float32) float32
TEXT ·maxAVX(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VMOVUPS (SI), Y0
    ADDQ $32, SI
    SUBQ $8, CX

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   max32_reduce_first

max32_loop8:
    VMOVUPS (SI), Y1
    VMAXPS Y0, Y1, Y0
    ADDQ $32, SI
    DECQ AX
    JNZ  max32_loop8

max32_reduce_first:
    // Reduce Y0 to scalar BEFORE scalar ops
    VEXTRACTF128 $1, Y0, X1
    VMAXPS X0, X1, X0
    VPERMILPS $0x0E, X0, X1
    VMAXPS X0, X1, X0
    VPERMILPS $0x01, X0, X1
    VMAXSS X0, X1, X0

    ANDQ $7, CX
    JZ   max32_done

max32_scalar:
    VMOVSS (SI), X1
    VMAXSS X0, X1, X0
    ADDQ $4, SI
    DECQ CX
    JNZ  max32_scalar

max32_done:
    VMOVSS X0, ret+24(FP)
    VZEROUPPER
    RET

// func absAVX(dst, a []float32)
TEXT ·absAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VMOVUPS absf32mask<>(SB), Y2

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   abs32_remainder

abs32_loop8:
    VMOVUPS (SI), Y0
    VANDPS Y0, Y2, Y1
    VMOVUPS Y1, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  abs32_loop8

abs32_remainder:
    ANDQ $7, CX
    JZ   abs32_done

abs32_scalar:
    VMOVSS (SI), X0
    VANDPS X0, X2, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  abs32_scalar

abs32_done:
    VZEROUPPER
    RET

// func negAVX(dst, a []float32)
TEXT ·negAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VXORPS Y2, Y2, Y2

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   neg32_remainder

neg32_loop8:
    VMOVUPS (SI), Y0
    VSUBPS Y0, Y2, Y1
    VMOVUPS Y1, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  neg32_loop8

neg32_remainder:
    ANDQ $7, CX
    JZ   neg32_done

neg32_scalar:
    VMOVSS (SI), X0
    VSUBSS X0, X2, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  neg32_scalar

neg32_done:
    VZEROUPPER
    RET

// func fmaAVX(dst, a, b, c []float32)
TEXT ·fmaAVX(SB), NOSPLIT, $0-96
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI
    MOVQ c_base+72(FP), R8

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   fma32_remainder

fma32_loop8:
    VMOVUPS (SI), Y0
    VMOVUPS (DI), Y1
    VMOVUPS (R8), Y2
    VFMADD213PS Y2, Y1, Y0
    VMOVUPS Y0, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, R8
    ADDQ $32, DX
    DECQ AX
    JNZ  fma32_loop8

fma32_remainder:
    ANDQ $7, CX
    JZ   fma32_done

fma32_scalar:
    VMOVSS (SI), X0
    VMOVSS (DI), X1
    VMOVSS (R8), X2
    VFMADD213SS X2, X1, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, R8
    ADDQ $4, DX
    DECQ CX
    JNZ  fma32_scalar

fma32_done:
    VZEROUPPER
    RET

// func clampAVX(dst, a []float32, minVal, maxVal float32)
TEXT ·clampAVX(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VBROADCASTSS minVal+48(FP), Y1
    VBROADCASTSS maxVal+52(FP), Y2

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   clamp32_remainder

clamp32_loop8:
    VMOVUPS (SI), Y0
    VMAXPS Y0, Y1, Y0
    VMINPS Y0, Y2, Y0
    VMOVUPS Y0, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  clamp32_loop8

clamp32_remainder:
    ANDQ $7, CX
    JZ   clamp32_done
    VMOVSS minVal+48(FP), X1
    VMOVSS maxVal+52(FP), X2

clamp32_scalar:
    VMOVSS (SI), X0
    VMAXSS X0, X1, X0
    VMINSS X0, X2, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  clamp32_scalar

clamp32_done:
    VZEROUPPER
    RET

// ============================================================================
// AVX-512 implementations (16x float32 per iteration)
// ============================================================================

// func dotProductAVX512(a, b []float32) float32
TEXT ·dotProductAVX512(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    VXORPS Z0, Z0, Z0

    MOVQ CX, AX
    SHRQ $4, AX                // len / 16
    JZ   dot32_512_remainder

dot32_512_loop16:
    VMOVUPS (SI), Z1
    VMOVUPS (DI), Z2
    VFMADD231PS Z1, Z2, Z0
    ADDQ $64, SI
    ADDQ $64, DI
    DECQ AX
    JNZ  dot32_512_loop16

dot32_512_remainder:
    VEXTRACTF32X8 $1, Z0, Y1
    VADDPS Y1, Y0, Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    ANDQ $15, CX
    JZ   dot32_512_done

dot32_512_scalar:
    VMOVSS (SI), X1
    VMOVSS (DI), X2
    VFMADD231SS X1, X2, X0
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ CX
    JNZ  dot32_512_scalar

dot32_512_done:
    VMOVSS X0, ret+48(FP)
    VZEROUPPER
    RET

// func addAVX512(dst, a, b []float32)
TEXT ·addAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $4, AX
    JZ   add32_512_remainder

add32_512_loop16:
    VMOVUPS (SI), Z0
    VMOVUPS (DI), Z1
    VADDPS Z0, Z1, Z2
    VMOVUPS Z2, (DX)
    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, DX
    DECQ AX
    JNZ  add32_512_loop16

add32_512_remainder:
    ANDQ $15, CX
    JZ   add32_512_done

add32_512_scalar:
    VMOVSS (SI), X0
    VADDSS (DI), X0, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  add32_512_scalar

add32_512_done:
    VZEROUPPER
    RET

// func subAVX512(dst, a, b []float32)
TEXT ·subAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $4, AX
    JZ   sub32_512_remainder

sub32_512_loop16:
    VMOVUPS (SI), Z0
    VMOVUPS (DI), Z1
    VSUBPS Z1, Z0, Z2
    VMOVUPS Z2, (DX)
    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, DX
    DECQ AX
    JNZ  sub32_512_loop16

sub32_512_remainder:
    ANDQ $15, CX
    JZ   sub32_512_done

sub32_512_scalar:
    VMOVSS (SI), X0
    VSUBSS (DI), X0, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  sub32_512_scalar

sub32_512_done:
    VZEROUPPER
    RET

// func mulAVX512(dst, a, b []float32)
TEXT ·mulAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $4, AX
    JZ   mul32_512_remainder

mul32_512_loop16:
    VMOVUPS (SI), Z0
    VMOVUPS (DI), Z1
    VMULPS Z0, Z1, Z2
    VMOVUPS Z2, (DX)
    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, DX
    DECQ AX
    JNZ  mul32_512_loop16

mul32_512_remainder:
    ANDQ $15, CX
    JZ   mul32_512_done

mul32_512_scalar:
    VMOVSS (SI), X0
    VMULSS (DI), X0, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  mul32_512_scalar

mul32_512_done:
    VZEROUPPER
    RET

// func divAVX512(dst, a, b []float32)
TEXT ·divAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $4, AX
    JZ   div32_512_remainder

div32_512_loop16:
    VMOVUPS (SI), Z0
    VMOVUPS (DI), Z1
    VDIVPS Z1, Z0, Z2
    VMOVUPS Z2, (DX)
    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, DX
    DECQ AX
    JNZ  div32_512_loop16

div32_512_remainder:
    ANDQ $15, CX
    JZ   div32_512_done

div32_512_scalar:
    VMOVSS (SI), X0
    VDIVSS (DI), X0, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  div32_512_scalar

div32_512_done:
    VZEROUPPER
    RET

// func scaleAVX512(dst, a []float32, s float32)
TEXT ·scaleAVX512(SB), NOSPLIT, $0-52
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VBROADCASTSS s+48(FP), Z1

    MOVQ CX, AX
    SHRQ $4, AX
    JZ   scale32_512_remainder

scale32_512_loop16:
    VMOVUPS (SI), Z0
    VMULPS Z0, Z1, Z2
    VMOVUPS Z2, (DX)
    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  scale32_512_loop16

scale32_512_remainder:
    ANDQ $15, CX
    JZ   scale32_512_done
    VMOVSS s+48(FP), X1

scale32_512_scalar:
    VMOVSS (SI), X0
    VMULSS X0, X1, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  scale32_512_scalar

scale32_512_done:
    VZEROUPPER
    RET

// func addScalarAVX512(dst, a []float32, s float32)
TEXT ·addScalarAVX512(SB), NOSPLIT, $0-52
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VBROADCASTSS s+48(FP), Z1

    MOVQ CX, AX
    SHRQ $4, AX
    JZ   addsc32_512_remainder

addsc32_512_loop16:
    VMOVUPS (SI), Z0
    VADDPS Z0, Z1, Z2
    VMOVUPS Z2, (DX)
    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  addsc32_512_loop16

addsc32_512_remainder:
    ANDQ $15, CX
    JZ   addsc32_512_done
    VMOVSS s+48(FP), X1

addsc32_512_scalar:
    VMOVSS (SI), X0
    VADDSS X0, X1, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  addsc32_512_scalar

addsc32_512_done:
    VZEROUPPER
    RET

// func sumAVX512(a []float32) float32
TEXT ·sumAVX512(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VXORPS Z0, Z0, Z0

    MOVQ CX, AX
    SHRQ $4, AX
    JZ   sum32_512_remainder

sum32_512_loop16:
    VMOVUPS (SI), Z1
    VADDPS Z0, Z1, Z0
    ADDQ $64, SI
    DECQ AX
    JNZ  sum32_512_loop16

sum32_512_remainder:
    VEXTRACTF32X8 $1, Z0, Y1
    VADDPS Y1, Y0, Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    ANDQ $15, CX
    JZ   sum32_512_done

sum32_512_scalar:
    VMOVSS (SI), X1
    VADDSS X0, X1, X0
    ADDQ $4, SI
    DECQ CX
    JNZ  sum32_512_scalar

sum32_512_done:
    VMOVSS X0, ret+24(FP)
    VZEROUPPER
    RET

// func minAVX512(a []float32) float32
TEXT ·minAVX512(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VMOVUPS (SI), Z0
    ADDQ $64, SI
    SUBQ $16, CX

    MOVQ CX, AX
    SHRQ $4, AX
    JZ   min32_512_reduce

min32_512_loop16:
    VMOVUPS (SI), Z1
    VMINPS Z0, Z1, Z0
    ADDQ $64, SI
    DECQ AX
    JNZ  min32_512_loop16

min32_512_reduce:
    VEXTRACTF32X8 $1, Z0, Y1
    VMINPS Y0, Y1, Y0
    VEXTRACTF128 $1, Y0, X1
    VMINPS X0, X1, X0
    VPERMILPS $0x0E, X0, X1
    VMINPS X0, X1, X0
    VPERMILPS $0x01, X0, X1
    VMINSS X0, X1, X0

    ANDQ $15, CX
    JZ   min32_512_done

min32_512_scalar:
    VMOVSS (SI), X1
    VMINSS X0, X1, X0
    ADDQ $4, SI
    DECQ CX
    JNZ  min32_512_scalar

min32_512_done:
    VMOVSS X0, ret+24(FP)
    VZEROUPPER
    RET

// func maxAVX512(a []float32) float32
TEXT ·maxAVX512(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VMOVUPS (SI), Z0
    ADDQ $64, SI
    SUBQ $16, CX

    MOVQ CX, AX
    SHRQ $4, AX
    JZ   max32_512_reduce

max32_512_loop16:
    VMOVUPS (SI), Z1
    VMAXPS Z0, Z1, Z0
    ADDQ $64, SI
    DECQ AX
    JNZ  max32_512_loop16

max32_512_reduce:
    VEXTRACTF32X8 $1, Z0, Y1
    VMAXPS Y0, Y1, Y0
    VEXTRACTF128 $1, Y0, X1
    VMAXPS X0, X1, X0
    VPERMILPS $0x0E, X0, X1
    VMAXPS X0, X1, X0
    VPERMILPS $0x01, X0, X1
    VMAXSS X0, X1, X0

    ANDQ $15, CX
    JZ   max32_512_done

max32_512_scalar:
    VMOVSS (SI), X1
    VMAXSS X0, X1, X0
    ADDQ $4, SI
    DECQ CX
    JNZ  max32_512_scalar

max32_512_done:
    VMOVSS X0, ret+24(FP)
    VZEROUPPER
    RET

// func absAVX512(dst, a []float32)
TEXT ·absAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VPBROADCASTD absf32mask<>(SB), Z2

    MOVQ CX, AX
    SHRQ $4, AX
    JZ   abs32_512_remainder

abs32_512_loop16:
    VMOVUPS (SI), Z0
    VPANDD Z0, Z2, Z1
    VMOVUPS Z1, (DX)
    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  abs32_512_loop16

abs32_512_remainder:
    ANDQ $15, CX
    JZ   abs32_512_done

abs32_512_scalar:
    VMOVSS (SI), X0
    VANDPS X0, X2, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  abs32_512_scalar

abs32_512_done:
    VZEROUPPER
    RET

// func negAVX512(dst, a []float32)
TEXT ·negAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VXORPS Z2, Z2, Z2

    MOVQ CX, AX
    SHRQ $4, AX
    JZ   neg32_512_remainder

neg32_512_loop16:
    VMOVUPS (SI), Z0
    VSUBPS Z0, Z2, Z1
    VMOVUPS Z1, (DX)
    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  neg32_512_loop16

neg32_512_remainder:
    ANDQ $15, CX
    JZ   neg32_512_done

neg32_512_scalar:
    VMOVSS (SI), X0
    VSUBSS X0, X2, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  neg32_512_scalar

neg32_512_done:
    VZEROUPPER
    RET

// func fmaAVX512(dst, a, b, c []float32)
TEXT ·fmaAVX512(SB), NOSPLIT, $0-96
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI
    MOVQ c_base+72(FP), R8

    MOVQ CX, AX
    SHRQ $4, AX
    JZ   fma32_512_remainder

fma32_512_loop16:
    VMOVUPS (SI), Z0
    VMOVUPS (DI), Z1
    VMOVUPS (R8), Z2
    VFMADD213PS Z2, Z1, Z0
    VMOVUPS Z0, (DX)
    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, R8
    ADDQ $64, DX
    DECQ AX
    JNZ  fma32_512_loop16

fma32_512_remainder:
    ANDQ $15, CX
    JZ   fma32_512_done

fma32_512_scalar:
    VMOVSS (SI), X0
    VMOVSS (DI), X1
    VMOVSS (R8), X2
    VFMADD213SS X2, X1, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, R8
    ADDQ $4, DX
    DECQ CX
    JNZ  fma32_512_scalar

fma32_512_done:
    VZEROUPPER
    RET

// func clampAVX512(dst, a []float32, minVal, maxVal float32)
TEXT ·clampAVX512(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VBROADCASTSS minVal+48(FP), Z1
    VBROADCASTSS maxVal+52(FP), Z2

    MOVQ CX, AX
    SHRQ $4, AX
    JZ   clamp32_512_remainder

clamp32_512_loop16:
    VMOVUPS (SI), Z0
    VMAXPS Z0, Z1, Z0
    VMINPS Z0, Z2, Z0
    VMOVUPS Z0, (DX)
    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  clamp32_512_loop16

clamp32_512_remainder:
    ANDQ $15, CX
    JZ   clamp32_512_done
    VMOVSS minVal+48(FP), X1
    VMOVSS maxVal+52(FP), X2

clamp32_512_scalar:
    VMOVSS (SI), X0
    VMAXSS X0, X1, X0
    VMINSS X0, X2, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  clamp32_512_scalar

clamp32_512_done:
    VZEROUPPER
    RET

// ============================================================================
// SSE implementations (4x float32 per iteration)
// ============================================================================

// func dotProductSSE(a, b []float32) float32
TEXT ·dotProductSSE(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    XORPS X0, X0

    MOVQ CX, AX
    SHRQ $2, AX                // len / 4
    JZ   dot32_sse_remainder

dot32_sse_loop4:
    MOVUPS (SI), X1
    MOVUPS (DI), X2
    MULPS X2, X1
    ADDPS X1, X0
    ADDQ $16, SI
    ADDQ $16, DI
    DECQ AX
    JNZ  dot32_sse_loop4

dot32_sse_remainder:
    // Horizontal sum
    MOVAPS X0, X1
    SHUFPS $0x0E, X1, X1
    ADDPS X1, X0
    MOVAPS X0, X1
    SHUFPS $0x01, X1, X1
    ADDSS X1, X0

    ANDQ $3, CX
    JZ   dot32_sse_done

dot32_sse_scalar:
    MOVSS (SI), X1
    MOVSS (DI), X2
    MULSS X2, X1
    ADDSS X1, X0
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ CX
    JNZ  dot32_sse_scalar

dot32_sse_done:
    MOVSS X0, ret+48(FP)
    RET

// func addSSE(dst, a, b []float32)
TEXT ·addSSE(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   add32_sse_remainder

add32_sse_loop4:
    MOVUPS (SI), X0
    MOVUPS (DI), X1
    ADDPS X1, X0
    MOVUPS X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ AX
    JNZ  add32_sse_loop4

add32_sse_remainder:
    ANDQ $3, CX
    JZ   add32_sse_done

add32_sse_scalar:
    MOVSS (SI), X0
    ADDSS (DI), X0
    MOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  add32_sse_scalar

add32_sse_done:
    RET

// func subSSE(dst, a, b []float32)
TEXT ·subSSE(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   sub32_sse_remainder

sub32_sse_loop4:
    MOVUPS (SI), X0
    MOVUPS (DI), X1
    SUBPS X1, X0
    MOVUPS X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ AX
    JNZ  sub32_sse_loop4

sub32_sse_remainder:
    ANDQ $3, CX
    JZ   sub32_sse_done

sub32_sse_scalar:
    MOVSS (SI), X0
    SUBSS (DI), X0
    MOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  sub32_sse_scalar

sub32_sse_done:
    RET

// func mulSSE(dst, a, b []float32)
TEXT ·mulSSE(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   mul32_sse_remainder

mul32_sse_loop4:
    MOVUPS (SI), X0
    MOVUPS (DI), X1
    MULPS X1, X0
    MOVUPS X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ AX
    JNZ  mul32_sse_loop4

mul32_sse_remainder:
    ANDQ $3, CX
    JZ   mul32_sse_done

mul32_sse_scalar:
    MOVSS (SI), X0
    MULSS (DI), X0
    MOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  mul32_sse_scalar

mul32_sse_done:
    RET

// func divSSE(dst, a, b []float32)
TEXT ·divSSE(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   div32_sse_remainder

div32_sse_loop4:
    MOVUPS (SI), X0
    MOVUPS (DI), X1
    DIVPS X1, X0
    MOVUPS X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ AX
    JNZ  div32_sse_loop4

div32_sse_remainder:
    ANDQ $3, CX
    JZ   div32_sse_done

div32_sse_scalar:
    MOVSS (SI), X0
    DIVSS (DI), X0
    MOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  div32_sse_scalar

div32_sse_done:
    RET

// func scaleSSE(dst, a []float32, s float32)
TEXT ·scaleSSE(SB), NOSPLIT, $0-52
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVSS s+48(FP), X1
    SHUFPS $0, X1, X1

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   scale32_sse_remainder

scale32_sse_loop4:
    MOVUPS (SI), X0
    MULPS X1, X0
    MOVUPS X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  scale32_sse_loop4

scale32_sse_remainder:
    ANDQ $3, CX
    JZ   scale32_sse_done

scale32_sse_scalar:
    MOVSS (SI), X0
    MULSS X1, X0
    MOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  scale32_sse_scalar

scale32_sse_done:
    RET

// func addScalarSSE(dst, a []float32, s float32)
TEXT ·addScalarSSE(SB), NOSPLIT, $0-52
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVSS s+48(FP), X1
    SHUFPS $0, X1, X1

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   addsc32_sse_remainder

addsc32_sse_loop4:
    MOVUPS (SI), X0
    ADDPS X1, X0
    MOVUPS X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  addsc32_sse_loop4

addsc32_sse_remainder:
    ANDQ $3, CX
    JZ   addsc32_sse_done

addsc32_sse_scalar:
    MOVSS (SI), X0
    ADDSS X1, X0
    MOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  addsc32_sse_scalar

addsc32_sse_done:
    RET

// func sumSSE(a []float32) float32
TEXT ·sumSSE(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    XORPS X0, X0

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   sum32_sse_remainder

sum32_sse_loop4:
    MOVUPS (SI), X1
    ADDPS X1, X0
    ADDQ $16, SI
    DECQ AX
    JNZ  sum32_sse_loop4

sum32_sse_remainder:
    MOVAPS X0, X1
    SHUFPS $0x0E, X1, X1
    ADDPS X1, X0
    MOVAPS X0, X1
    SHUFPS $0x01, X1, X1
    ADDSS X1, X0

    ANDQ $3, CX
    JZ   sum32_sse_done

sum32_sse_scalar:
    MOVSS (SI), X1
    ADDSS X1, X0
    ADDQ $4, SI
    DECQ CX
    JNZ  sum32_sse_scalar

sum32_sse_done:
    MOVSS X0, ret+24(FP)
    RET

// func minSSE(a []float32) float32
TEXT ·minSSE(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    MOVUPS (SI), X0
    ADDQ $16, SI
    SUBQ $4, CX

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   min32_sse_reduce

min32_sse_loop4:
    MOVUPS (SI), X1
    MINPS X1, X0
    ADDQ $16, SI
    DECQ AX
    JNZ  min32_sse_loop4

min32_sse_reduce:
    MOVAPS X0, X1
    SHUFPS $0x0E, X1, X1
    MINPS X1, X0
    MOVAPS X0, X1
    SHUFPS $0x01, X1, X1
    MINSS X1, X0

    ANDQ $3, CX
    JZ   min32_sse_done

min32_sse_scalar:
    MOVSS (SI), X1
    MINSS X1, X0
    ADDQ $4, SI
    DECQ CX
    JNZ  min32_sse_scalar

min32_sse_done:
    MOVSS X0, ret+24(FP)
    RET

// func maxSSE(a []float32) float32
TEXT ·maxSSE(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    MOVUPS (SI), X0
    ADDQ $16, SI
    SUBQ $4, CX

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   max32_sse_reduce

max32_sse_loop4:
    MOVUPS (SI), X1
    MAXPS X1, X0
    ADDQ $16, SI
    DECQ AX
    JNZ  max32_sse_loop4

max32_sse_reduce:
    MOVAPS X0, X1
    SHUFPS $0x0E, X1, X1
    MAXPS X1, X0
    MOVAPS X0, X1
    SHUFPS $0x01, X1, X1
    MAXSS X1, X0

    ANDQ $3, CX
    JZ   max32_sse_done

max32_sse_scalar:
    MOVSS (SI), X1
    MAXSS X1, X0
    ADDQ $4, SI
    DECQ CX
    JNZ  max32_sse_scalar

max32_sse_done:
    MOVSS X0, ret+24(FP)
    RET

// func absSSE(dst, a []float32)
TEXT ·absSSE(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVUPS absf32mask<>(SB), X2

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   abs32_sse_remainder

abs32_sse_loop4:
    MOVUPS (SI), X0
    ANDPS X2, X0
    MOVUPS X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  abs32_sse_loop4

abs32_sse_remainder:
    ANDQ $3, CX
    JZ   abs32_sse_done

abs32_sse_scalar:
    MOVSS (SI), X0
    ANDPS X2, X0
    MOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  abs32_sse_scalar

abs32_sse_done:
    RET

// func negSSE(dst, a []float32)
TEXT ·negSSE(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    XORPS X2, X2

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   neg32_sse_remainder

neg32_sse_loop4:
    MOVUPS (SI), X0
    MOVAPS X2, X1
    SUBPS X0, X1
    MOVUPS X1, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  neg32_sse_loop4

neg32_sse_remainder:
    ANDQ $3, CX
    JZ   neg32_sse_done

neg32_sse_scalar:
    MOVSS (SI), X0
    MOVAPS X2, X1
    SUBSS X0, X1
    MOVSS X1, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  neg32_sse_scalar

neg32_sse_done:
    RET

// func fmaSSE(dst, a, b, c []float32)
// Note: SSE doesn't have native FMA, emulate with mul+add
TEXT ·fmaSSE(SB), NOSPLIT, $0-96
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI
    MOVQ c_base+72(FP), R8

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   fma32_sse_remainder

fma32_sse_loop4:
    MOVUPS (SI), X0
    MOVUPS (DI), X1
    MOVUPS (R8), X2
    MULPS X1, X0
    ADDPS X2, X0
    MOVUPS X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, R8
    ADDQ $16, DX
    DECQ AX
    JNZ  fma32_sse_loop4

fma32_sse_remainder:
    ANDQ $3, CX
    JZ   fma32_sse_done

fma32_sse_scalar:
    MOVSS (SI), X0
    MOVSS (DI), X1
    MOVSS (R8), X2
    MULSS X1, X0
    ADDSS X2, X0
    MOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, R8
    ADDQ $4, DX
    DECQ CX
    JNZ  fma32_sse_scalar

fma32_sse_done:
    RET

// func clampSSE(dst, a []float32, minVal, maxVal float32)
TEXT ·clampSSE(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVSS minVal+48(FP), X1
    SHUFPS $0, X1, X1
    MOVSS maxVal+52(FP), X2
    SHUFPS $0, X2, X2

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   clamp32_sse_remainder

clamp32_sse_loop4:
    MOVUPS (SI), X0
    MAXPS X1, X0
    MINPS X2, X0
    MOVUPS X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  clamp32_sse_loop4

clamp32_sse_remainder:
    ANDQ $3, CX
    JZ   clamp32_sse_done

clamp32_sse_scalar:
    MOVSS (SI), X0
    MAXSS X1, X0
    MINSS X2, X0
    MOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  clamp32_sse_scalar

clamp32_sse_done:
    RET
