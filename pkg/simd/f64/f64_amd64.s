//go:build amd64

#include "textflag.h"

// Constants for abs (sign bit mask)
DATA absf64mask<>+0x00(SB)/8, $0x7fffffffffffffff
DATA absf64mask<>+0x08(SB)/8, $0x7fffffffffffffff
DATA absf64mask<>+0x10(SB)/8, $0x7fffffffffffffff
DATA absf64mask<>+0x18(SB)/8, $0x7fffffffffffffff
DATA absf64mask<>+0x20(SB)/8, $0x7fffffffffffffff
DATA absf64mask<>+0x28(SB)/8, $0x7fffffffffffffff
DATA absf64mask<>+0x30(SB)/8, $0x7fffffffffffffff
DATA absf64mask<>+0x38(SB)/8, $0x7fffffffffffffff
GLOBL absf64mask<>(SB), RODATA|NOPTR, $64

// ============================================================================
// AVX+FMA IMPLEMENTATIONS (256-bit, 4x float64 per iteration)
// ============================================================================

// func dotProductAVX(a, b []float64) float64
TEXT ·dotProductAVX(SB), NOSPLIT, $0-56
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    VXORPD Y0, Y0, Y0

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   dot_avx_remainder

dot_avx_loop4:
    VMOVUPD (SI), Y1
    VMOVUPD (DI), Y2
    VFMADD231PD Y1, Y2, Y0
    ADDQ $32, SI
    ADDQ $32, DI
    DECQ AX
    JNZ  dot_avx_loop4

dot_avx_remainder:
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0

    ANDQ $3, CX
    JZ   dot_avx_done

dot_avx_scalar:
    VMOVSD (SI), X1
    VMOVSD (DI), X2
    VFMADD231SD X1, X2, X0
    ADDQ $8, SI
    ADDQ $8, DI
    DECQ CX
    JNZ  dot_avx_scalar

dot_avx_done:
    VMOVSD X0, ret+48(FP)
    VZEROUPPER
    RET

// func addAVX(dst, a, b []float64)
TEXT ·addAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   add_avx_remainder

add_avx_loop4:
    VMOVUPD (SI), Y0
    VMOVUPD (DI), Y1
    VADDPD Y0, Y1, Y2
    VMOVUPD Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  add_avx_loop4

add_avx_remainder:
    ANDQ $3, CX
    JZ   add_avx_done

add_avx_scalar:
    VMOVSD (SI), X0
    VADDSD (DI), X0, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  add_avx_scalar

add_avx_done:
    VZEROUPPER
    RET

// func subAVX(dst, a, b []float64)
TEXT ·subAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   sub_avx_remainder

sub_avx_loop4:
    VMOVUPD (SI), Y0
    VMOVUPD (DI), Y1
    VSUBPD Y1, Y0, Y2
    VMOVUPD Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  sub_avx_loop4

sub_avx_remainder:
    ANDQ $3, CX
    JZ   sub_avx_done

sub_avx_scalar:
    VMOVSD (SI), X0
    VSUBSD (DI), X0, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  sub_avx_scalar

sub_avx_done:
    VZEROUPPER
    RET

// func mulAVX(dst, a, b []float64)
TEXT ·mulAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   mul_avx_remainder

mul_avx_loop4:
    VMOVUPD (SI), Y0
    VMOVUPD (DI), Y1
    VMULPD Y0, Y1, Y2
    VMOVUPD Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  mul_avx_loop4

mul_avx_remainder:
    ANDQ $3, CX
    JZ   mul_avx_done

mul_avx_scalar:
    VMOVSD (SI), X0
    VMULSD (DI), X0, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  mul_avx_scalar

mul_avx_done:
    VZEROUPPER
    RET

// func divAVX(dst, a, b []float64)
TEXT ·divAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   div_avx_remainder

div_avx_loop4:
    VMOVUPD (SI), Y0
    VMOVUPD (DI), Y1
    VDIVPD Y1, Y0, Y2
    VMOVUPD Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  div_avx_loop4

div_avx_remainder:
    ANDQ $3, CX
    JZ   div_avx_done

div_avx_scalar:
    VMOVSD (SI), X0
    VDIVSD (DI), X0, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  div_avx_scalar

div_avx_done:
    VZEROUPPER
    RET

// func scaleAVX(dst, a []float64, s float64)
TEXT ·scaleAVX(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VBROADCASTSD s+48(FP), Y1

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   scale_avx_remainder

scale_avx_loop4:
    VMOVUPD (SI), Y0
    VMULPD Y0, Y1, Y2
    VMOVUPD Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  scale_avx_loop4

scale_avx_remainder:
    ANDQ $3, CX
    JZ   scale_avx_done
    VMOVSD s+48(FP), X1

scale_avx_scalar:
    VMOVSD (SI), X0
    VMULSD X0, X1, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  scale_avx_scalar

scale_avx_done:
    VZEROUPPER
    RET

// func addScalarAVX(dst, a []float64, s float64)
TEXT ·addScalarAVX(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VBROADCASTSD s+48(FP), Y1

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   addsc_avx_remainder

addsc_avx_loop4:
    VMOVUPD (SI), Y0
    VADDPD Y0, Y1, Y2
    VMOVUPD Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  addsc_avx_loop4

addsc_avx_remainder:
    ANDQ $3, CX
    JZ   addsc_avx_done
    VMOVSD s+48(FP), X1

addsc_avx_scalar:
    VMOVSD (SI), X0
    VADDSD X0, X1, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  addsc_avx_scalar

addsc_avx_done:
    VZEROUPPER
    RET

// func sumAVX(a []float64) float64
TEXT ·sumAVX(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VXORPD Y0, Y0, Y0

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   sum_avx_remainder

sum_avx_loop4:
    VMOVUPD (SI), Y1
    VADDPD Y0, Y1, Y0
    ADDQ $32, SI
    DECQ AX
    JNZ  sum_avx_loop4

sum_avx_remainder:
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0

    ANDQ $3, CX
    JZ   sum_avx_done

sum_avx_scalar:
    VMOVSD (SI), X1
    VADDSD X0, X1, X0
    ADDQ $8, SI
    DECQ CX
    JNZ  sum_avx_scalar

sum_avx_done:
    VMOVSD X0, ret+24(FP)
    VZEROUPPER
    RET

// func minAVX(a []float64) float64
TEXT ·minAVX(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VMOVUPD (SI), Y0
    ADDQ $32, SI
    SUBQ $4, CX

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   min_avx_reduce

min_avx_loop4:
    VMOVUPD (SI), Y1
    VMINPD Y0, Y1, Y0
    ADDQ $32, SI
    DECQ AX
    JNZ  min_avx_loop4

min_avx_reduce:
    VEXTRACTF128 $1, Y0, X1
    VMINPD X0, X1, X0
    VPERMILPD $1, X0, X1
    VMINSD X0, X1, X0

    ANDQ $3, CX
    JZ   min_avx_done

min_avx_scalar:
    VMOVSD (SI), X1
    VMINSD X0, X1, X0
    ADDQ $8, SI
    DECQ CX
    JNZ  min_avx_scalar

min_avx_done:
    VMOVSD X0, ret+24(FP)
    VZEROUPPER
    RET

// func maxAVX(a []float64) float64
TEXT ·maxAVX(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VMOVUPD (SI), Y0
    ADDQ $32, SI
    SUBQ $4, CX

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   max_avx_reduce

max_avx_loop4:
    VMOVUPD (SI), Y1
    VMAXPD Y0, Y1, Y0
    ADDQ $32, SI
    DECQ AX
    JNZ  max_avx_loop4

max_avx_reduce:
    VEXTRACTF128 $1, Y0, X1
    VMAXPD X0, X1, X0
    VPERMILPD $1, X0, X1
    VMAXSD X0, X1, X0

    ANDQ $3, CX
    JZ   max_avx_done

max_avx_scalar:
    VMOVSD (SI), X1
    VMAXSD X0, X1, X0
    ADDQ $8, SI
    DECQ CX
    JNZ  max_avx_scalar

max_avx_done:
    VMOVSD X0, ret+24(FP)
    VZEROUPPER
    RET

// func absAVX(dst, a []float64)
TEXT ·absAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VMOVUPD absf64mask<>(SB), Y2

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   abs_avx_remainder

abs_avx_loop4:
    VMOVUPD (SI), Y0
    VANDPD Y0, Y2, Y1
    VMOVUPD Y1, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  abs_avx_loop4

abs_avx_remainder:
    ANDQ $3, CX
    JZ   abs_avx_done

abs_avx_scalar:
    VMOVSD (SI), X0
    VANDPD X0, X2, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  abs_avx_scalar

abs_avx_done:
    VZEROUPPER
    RET

// func negAVX(dst, a []float64)
TEXT ·negAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VXORPD Y2, Y2, Y2

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   neg_avx_remainder

neg_avx_loop4:
    VMOVUPD (SI), Y0
    VSUBPD Y0, Y2, Y1
    VMOVUPD Y1, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  neg_avx_loop4

neg_avx_remainder:
    ANDQ $3, CX
    JZ   neg_avx_done

neg_avx_scalar:
    VMOVSD (SI), X0
    VSUBSD X0, X2, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  neg_avx_scalar

neg_avx_done:
    VZEROUPPER
    RET

// func fmaAVX(dst, a, b, c []float64)
TEXT ·fmaAVX(SB), NOSPLIT, $0-96
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI
    MOVQ c_base+72(FP), R8

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   fma_avx_remainder

fma_avx_loop4:
    VMOVUPD (SI), Y0
    VMOVUPD (DI), Y1
    VMOVUPD (R8), Y2
    VFMADD213PD Y2, Y1, Y0
    VMOVUPD Y0, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, R8
    ADDQ $32, DX
    DECQ AX
    JNZ  fma_avx_loop4

fma_avx_remainder:
    ANDQ $3, CX
    JZ   fma_avx_done

fma_avx_scalar:
    VMOVSD (SI), X0
    VMOVSD (DI), X1
    VMOVSD (R8), X2
    VFMADD213SD X2, X1, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, R8
    ADDQ $8, DX
    DECQ CX
    JNZ  fma_avx_scalar

fma_avx_done:
    VZEROUPPER
    RET

// func clampAVX(dst, a []float64, minVal, maxVal float64)
TEXT ·clampAVX(SB), NOSPLIT, $0-64
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VBROADCASTSD minVal+48(FP), Y1
    VBROADCASTSD maxVal+56(FP), Y2

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   clamp_avx_remainder

clamp_avx_loop4:
    VMOVUPD (SI), Y0
    VMAXPD Y0, Y1, Y0
    VMINPD Y0, Y2, Y0
    VMOVUPD Y0, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  clamp_avx_loop4

clamp_avx_remainder:
    ANDQ $3, CX
    JZ   clamp_avx_done
    VMOVSD minVal+48(FP), X1
    VMOVSD maxVal+56(FP), X2

clamp_avx_scalar:
    VMOVSD (SI), X0
    VMAXSD X0, X1, X0
    VMINSD X0, X2, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  clamp_avx_scalar

clamp_avx_done:
    VZEROUPPER
    RET

// func sqrtAVX(dst, a []float64)
TEXT ·sqrtAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   sqrt_avx_remainder

sqrt_avx_loop4:
    VMOVUPD (SI), Y0
    VSQRTPD Y0, Y1
    VMOVUPD Y1, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  sqrt_avx_loop4

sqrt_avx_remainder:
    ANDQ $3, CX
    JZ   sqrt_avx_done

sqrt_avx_scalar:
    VMOVSD (SI), X0
    VSQRTSD X0, X0, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  sqrt_avx_scalar

sqrt_avx_done:
    VZEROUPPER
    RET

// func reciprocalAVX(dst, a []float64)
TEXT ·reciprocalAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Load 1.0 for division
    MOVQ $0x3FF0000000000000, AX  // 1.0 in IEEE 754
    MOVQ AX, X3
    VBROADCASTSD X3, Y3

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   recip_avx_remainder

recip_avx_loop4:
    VMOVUPD (SI), Y0
    VDIVPD Y0, Y3, Y1
    VMOVUPD Y1, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  recip_avx_loop4

recip_avx_remainder:
    ANDQ $3, CX
    JZ   recip_avx_done

recip_avx_scalar:
    VMOVSD (SI), X0
    VDIVSD X0, X3, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  recip_avx_scalar

recip_avx_done:
    VZEROUPPER
    RET

// func varianceAVX(a []float64, mean float64) float64
TEXT ·varianceAVX(SB), NOSPLIT, $0-40
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    VBROADCASTSD mean+24(FP), Y2

    VXORPD Y0, Y0, Y0  // accumulator

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   var_avx_remainder

var_avx_loop4:
    VMOVUPD (SI), Y1
    VSUBPD Y2, Y1, Y1        // diff = val - mean
    VFMADD231PD Y1, Y1, Y0   // sum += diff * diff
    ADDQ $32, SI
    DECQ AX
    JNZ  var_avx_loop4

var_avx_remainder:
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0

    ANDQ $3, CX
    JZ   var_avx_divide
    VMOVSD mean+24(FP), X2

var_avx_scalar:
    VMOVSD (SI), X1
    VSUBSD X2, X1, X1
    VFMADD231SD X1, X1, X0
    ADDQ $8, SI
    DECQ CX
    JNZ  var_avx_scalar

var_avx_divide:
    // Divide by n
    MOVQ a_len+8(FP), CX
    CVTSQ2SD CX, X1
    VDIVSD X1, X0, X0
    VMOVSD X0, ret+32(FP)
    VZEROUPPER
    RET

// func euclideanDistanceAVX(a, b []float64) float64
TEXT ·euclideanDistanceAVX(SB), NOSPLIT, $0-56
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    VXORPD Y0, Y0, Y0

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   euclid_avx_remainder

euclid_avx_loop4:
    VMOVUPD (SI), Y1
    VMOVUPD (DI), Y2
    VSUBPD Y2, Y1, Y1        // diff
    VFMADD231PD Y1, Y1, Y0   // sum += diff * diff
    ADDQ $32, SI
    ADDQ $32, DI
    DECQ AX
    JNZ  euclid_avx_loop4

euclid_avx_remainder:
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0

    ANDQ $3, CX
    JZ   euclid_avx_sqrt

euclid_avx_scalar:
    VMOVSD (SI), X1
    VMOVSD (DI), X2
    VSUBSD X2, X1, X1
    VFMADD231SD X1, X1, X0
    ADDQ $8, SI
    ADDQ $8, DI
    DECQ CX
    JNZ  euclid_avx_scalar

euclid_avx_sqrt:
    VSQRTSD X0, X0, X0
    VMOVSD X0, ret+48(FP)
    VZEROUPPER
    RET

// ============================================================================
// AVX-512 IMPLEMENTATIONS (512-bit, 8x float64 per iteration)
// ============================================================================

// func dotProductAVX512(a, b []float64) float64
TEXT ·dotProductAVX512(SB), NOSPLIT, $0-56
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    VXORPD Z0, Z0, Z0

    MOVQ CX, AX
    SHRQ $3, AX                // len / 8
    JZ   dot_512_remainder

dot_512_loop8:
    VMOVUPD (SI), Z1
    VMOVUPD (DI), Z2
    VFMADD231PD Z1, Z2, Z0
    ADDQ $64, SI
    ADDQ $64, DI
    DECQ AX
    JNZ  dot_512_loop8

dot_512_remainder:
    // Reduce Z0 to scalar
    VEXTRACTF64X4 $1, Z0, Y1
    VADDPD Y1, Y0, Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0

    ANDQ $7, CX
    JZ   dot_512_done

dot_512_scalar:
    VMOVSD (SI), X1
    VMOVSD (DI), X2
    VFMADD231SD X1, X2, X0
    ADDQ $8, SI
    ADDQ $8, DI
    DECQ CX
    JNZ  dot_512_scalar

dot_512_done:
    VMOVSD X0, ret+48(FP)
    VZEROUPPER
    RET

// func addAVX512(dst, a, b []float64)
TEXT ·addAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   add_512_remainder

add_512_loop8:
    VMOVUPD (SI), Z0
    VMOVUPD (DI), Z1
    VADDPD Z0, Z1, Z2
    VMOVUPD Z2, (DX)
    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, DX
    DECQ AX
    JNZ  add_512_loop8

add_512_remainder:
    ANDQ $7, CX
    JZ   add_512_done

add_512_scalar:
    VMOVSD (SI), X0
    VADDSD (DI), X0, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  add_512_scalar

add_512_done:
    VZEROUPPER
    RET

// func subAVX512(dst, a, b []float64)
TEXT ·subAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   sub_512_remainder

sub_512_loop8:
    VMOVUPD (SI), Z0
    VMOVUPD (DI), Z1
    VSUBPD Z1, Z0, Z2
    VMOVUPD Z2, (DX)
    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, DX
    DECQ AX
    JNZ  sub_512_loop8

sub_512_remainder:
    ANDQ $7, CX
    JZ   sub_512_done

sub_512_scalar:
    VMOVSD (SI), X0
    VSUBSD (DI), X0, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  sub_512_scalar

sub_512_done:
    VZEROUPPER
    RET

// func mulAVX512(dst, a, b []float64)
TEXT ·mulAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   mul_512_remainder

mul_512_loop8:
    VMOVUPD (SI), Z0
    VMOVUPD (DI), Z1
    VMULPD Z0, Z1, Z2
    VMOVUPD Z2, (DX)
    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, DX
    DECQ AX
    JNZ  mul_512_loop8

mul_512_remainder:
    ANDQ $7, CX
    JZ   mul_512_done

mul_512_scalar:
    VMOVSD (SI), X0
    VMULSD (DI), X0, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  mul_512_scalar

mul_512_done:
    VZEROUPPER
    RET

// func divAVX512(dst, a, b []float64)
TEXT ·divAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   div_512_remainder

div_512_loop8:
    VMOVUPD (SI), Z0
    VMOVUPD (DI), Z1
    VDIVPD Z1, Z0, Z2
    VMOVUPD Z2, (DX)
    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, DX
    DECQ AX
    JNZ  div_512_loop8

div_512_remainder:
    ANDQ $7, CX
    JZ   div_512_done

div_512_scalar:
    VMOVSD (SI), X0
    VDIVSD (DI), X0, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  div_512_scalar

div_512_done:
    VZEROUPPER
    RET

// func scaleAVX512(dst, a []float64, s float64)
TEXT ·scaleAVX512(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VBROADCASTSD s+48(FP), Z1

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   scale_512_remainder

scale_512_loop8:
    VMOVUPD (SI), Z0
    VMULPD Z0, Z1, Z2
    VMOVUPD Z2, (DX)
    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  scale_512_loop8

scale_512_remainder:
    ANDQ $7, CX
    JZ   scale_512_done
    VMOVSD s+48(FP), X1

scale_512_scalar:
    VMOVSD (SI), X0
    VMULSD X0, X1, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  scale_512_scalar

scale_512_done:
    VZEROUPPER
    RET

// func addScalarAVX512(dst, a []float64, s float64)
TEXT ·addScalarAVX512(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VBROADCASTSD s+48(FP), Z1

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   addsc_512_remainder

addsc_512_loop8:
    VMOVUPD (SI), Z0
    VADDPD Z0, Z1, Z2
    VMOVUPD Z2, (DX)
    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  addsc_512_loop8

addsc_512_remainder:
    ANDQ $7, CX
    JZ   addsc_512_done
    VMOVSD s+48(FP), X1

addsc_512_scalar:
    VMOVSD (SI), X0
    VADDSD X0, X1, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  addsc_512_scalar

addsc_512_done:
    VZEROUPPER
    RET

// func sumAVX512(a []float64) float64
TEXT ·sumAVX512(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VXORPD Z0, Z0, Z0

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   sum_512_remainder

sum_512_loop8:
    VMOVUPD (SI), Z1
    VADDPD Z0, Z1, Z0
    ADDQ $64, SI
    DECQ AX
    JNZ  sum_512_loop8

sum_512_remainder:
    VEXTRACTF64X4 $1, Z0, Y1
    VADDPD Y1, Y0, Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0

    ANDQ $7, CX
    JZ   sum_512_done

sum_512_scalar:
    VMOVSD (SI), X1
    VADDSD X0, X1, X0
    ADDQ $8, SI
    DECQ CX
    JNZ  sum_512_scalar

sum_512_done:
    VMOVSD X0, ret+24(FP)
    VZEROUPPER
    RET

// func minAVX512(a []float64) float64
TEXT ·minAVX512(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VMOVUPD (SI), Z0
    ADDQ $64, SI
    SUBQ $8, CX

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   min_512_reduce

min_512_loop8:
    VMOVUPD (SI), Z1
    VMINPD Z0, Z1, Z0
    ADDQ $64, SI
    DECQ AX
    JNZ  min_512_loop8

min_512_reduce:
    VEXTRACTF64X4 $1, Z0, Y1
    VMINPD Y0, Y1, Y0
    VEXTRACTF128 $1, Y0, X1
    VMINPD X0, X1, X0
    VPERMILPD $1, X0, X1
    VMINSD X0, X1, X0

    ANDQ $7, CX
    JZ   min_512_done

min_512_scalar:
    VMOVSD (SI), X1
    VMINSD X0, X1, X0
    ADDQ $8, SI
    DECQ CX
    JNZ  min_512_scalar

min_512_done:
    VMOVSD X0, ret+24(FP)
    VZEROUPPER
    RET

// func maxAVX512(a []float64) float64
TEXT ·maxAVX512(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VMOVUPD (SI), Z0
    ADDQ $64, SI
    SUBQ $8, CX

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   max_512_reduce

max_512_loop8:
    VMOVUPD (SI), Z1
    VMAXPD Z0, Z1, Z0
    ADDQ $64, SI
    DECQ AX
    JNZ  max_512_loop8

max_512_reduce:
    VEXTRACTF64X4 $1, Z0, Y1
    VMAXPD Y0, Y1, Y0
    VEXTRACTF128 $1, Y0, X1
    VMAXPD X0, X1, X0
    VPERMILPD $1, X0, X1
    VMAXSD X0, X1, X0

    ANDQ $7, CX
    JZ   max_512_done

max_512_scalar:
    VMOVSD (SI), X1
    VMAXSD X0, X1, X0
    ADDQ $8, SI
    DECQ CX
    JNZ  max_512_scalar

max_512_done:
    VMOVSD X0, ret+24(FP)
    VZEROUPPER
    RET

// func absAVX512(dst, a []float64)
TEXT ·absAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VMOVUPD absf64mask<>(SB), Z2

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   abs_512_remainder

abs_512_loop8:
    VMOVUPD (SI), Z0
    VANDPD Z0, Z2, Z1
    VMOVUPD Z1, (DX)
    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  abs_512_loop8

abs_512_remainder:
    ANDQ $7, CX
    JZ   abs_512_done

abs_512_scalar:
    VMOVSD (SI), X0
    VANDPD X0, X2, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  abs_512_scalar

abs_512_done:
    VZEROUPPER
    RET

// func negAVX512(dst, a []float64)
TEXT ·negAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VXORPD Z2, Z2, Z2

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   neg_512_remainder

neg_512_loop8:
    VMOVUPD (SI), Z0
    VSUBPD Z0, Z2, Z1
    VMOVUPD Z1, (DX)
    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  neg_512_loop8

neg_512_remainder:
    ANDQ $7, CX
    JZ   neg_512_done

neg_512_scalar:
    VMOVSD (SI), X0
    VSUBSD X0, X2, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  neg_512_scalar

neg_512_done:
    VZEROUPPER
    RET

// func fmaAVX512(dst, a, b, c []float64)
TEXT ·fmaAVX512(SB), NOSPLIT, $0-96
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI
    MOVQ c_base+72(FP), R8

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   fma_512_remainder

fma_512_loop8:
    VMOVUPD (SI), Z0
    VMOVUPD (DI), Z1
    VMOVUPD (R8), Z2
    VFMADD213PD Z2, Z1, Z0
    VMOVUPD Z0, (DX)
    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, R8
    ADDQ $64, DX
    DECQ AX
    JNZ  fma_512_loop8

fma_512_remainder:
    ANDQ $7, CX
    JZ   fma_512_done

fma_512_scalar:
    VMOVSD (SI), X0
    VMOVSD (DI), X1
    VMOVSD (R8), X2
    VFMADD213SD X2, X1, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, R8
    ADDQ $8, DX
    DECQ CX
    JNZ  fma_512_scalar

fma_512_done:
    VZEROUPPER
    RET

// func clampAVX512(dst, a []float64, minVal, maxVal float64)
TEXT ·clampAVX512(SB), NOSPLIT, $0-64
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VBROADCASTSD minVal+48(FP), Z1
    VBROADCASTSD maxVal+56(FP), Z2

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   clamp_512_remainder

clamp_512_loop8:
    VMOVUPD (SI), Z0
    VMAXPD Z0, Z1, Z0
    VMINPD Z0, Z2, Z0
    VMOVUPD Z0, (DX)
    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  clamp_512_loop8

clamp_512_remainder:
    ANDQ $7, CX
    JZ   clamp_512_done
    VMOVSD minVal+48(FP), X1
    VMOVSD maxVal+56(FP), X2

clamp_512_scalar:
    VMOVSD (SI), X0
    VMAXSD X0, X1, X0
    VMINSD X0, X2, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  clamp_512_scalar

clamp_512_done:
    VZEROUPPER
    RET

// func sqrtAVX512(dst, a []float64)
TEXT ·sqrtAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   sqrt_512_remainder

sqrt_512_loop8:
    VMOVUPD (SI), Z0
    VSQRTPD Z0, Z1
    VMOVUPD Z1, (DX)
    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  sqrt_512_loop8

sqrt_512_remainder:
    ANDQ $7, CX
    JZ   sqrt_512_done

sqrt_512_scalar:
    VMOVSD (SI), X0
    VSQRTSD X0, X0, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  sqrt_512_scalar

sqrt_512_done:
    VZEROUPPER
    RET

// func reciprocalAVX512(dst, a []float64)
TEXT ·reciprocalAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVQ $0x3FF0000000000000, AX
    MOVQ AX, X3
    VBROADCASTSD X3, Z3

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   recip_512_remainder

recip_512_loop8:
    VMOVUPD (SI), Z0
    VDIVPD Z0, Z3, Z1
    VMOVUPD Z1, (DX)
    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  recip_512_loop8

recip_512_remainder:
    ANDQ $7, CX
    JZ   recip_512_done

recip_512_scalar:
    VMOVSD (SI), X0
    VDIVSD X0, X3, X0
    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  recip_512_scalar

recip_512_done:
    VZEROUPPER
    RET

// func varianceAVX512(a []float64, mean float64) float64
TEXT ·varianceAVX512(SB), NOSPLIT, $0-40
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    VBROADCASTSD mean+24(FP), Z2

    VXORPD Z0, Z0, Z0

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   var_512_remainder

var_512_loop8:
    VMOVUPD (SI), Z1
    VSUBPD Z2, Z1, Z1
    VFMADD231PD Z1, Z1, Z0
    ADDQ $64, SI
    DECQ AX
    JNZ  var_512_loop8

var_512_remainder:
    VEXTRACTF64X4 $1, Z0, Y1
    VADDPD Y1, Y0, Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0

    ANDQ $7, CX
    JZ   var_512_divide
    VMOVSD mean+24(FP), X2

var_512_scalar:
    VMOVSD (SI), X1
    VSUBSD X2, X1, X1
    VFMADD231SD X1, X1, X0
    ADDQ $8, SI
    DECQ CX
    JNZ  var_512_scalar

var_512_divide:
    MOVQ a_len+8(FP), CX
    CVTSQ2SD CX, X1
    VDIVSD X1, X0, X0
    VMOVSD X0, ret+32(FP)
    VZEROUPPER
    RET

// func euclideanDistanceAVX512(a, b []float64) float64
TEXT ·euclideanDistanceAVX512(SB), NOSPLIT, $0-56
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    VXORPD Z0, Z0, Z0

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   euclid_512_remainder

euclid_512_loop8:
    VMOVUPD (SI), Z1
    VMOVUPD (DI), Z2
    VSUBPD Z2, Z1, Z1
    VFMADD231PD Z1, Z1, Z0
    ADDQ $64, SI
    ADDQ $64, DI
    DECQ AX
    JNZ  euclid_512_loop8

euclid_512_remainder:
    VEXTRACTF64X4 $1, Z0, Y1
    VADDPD Y1, Y0, Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0

    ANDQ $7, CX
    JZ   euclid_512_sqrt

euclid_512_scalar:
    VMOVSD (SI), X1
    VMOVSD (DI), X2
    VSUBSD X2, X1, X1
    VFMADD231SD X1, X1, X0
    ADDQ $8, SI
    ADDQ $8, DI
    DECQ CX
    JNZ  euclid_512_scalar

euclid_512_sqrt:
    VSQRTSD X0, X0, X0
    VMOVSD X0, ret+48(FP)
    VZEROUPPER
    RET

// ============================================================================
// SSE2 IMPLEMENTATIONS (128-bit, 2x float64 per iteration)
// ============================================================================

// func dotProductSSE2(a, b []float64) float64
TEXT ·dotProductSSE2(SB), NOSPLIT, $0-56
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    XORPD X0, X0

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   dot_sse2_remainder

dot_sse2_loop2:
    MOVUPD (SI), X1
    MOVUPD (DI), X2
    MULPD X2, X1
    ADDPD X1, X0
    ADDQ $16, SI
    ADDQ $16, DI
    DECQ AX
    JNZ  dot_sse2_loop2

dot_sse2_remainder:
    // Reduce X0
    MOVAPD X0, X1
    SHUFPD $1, X1, X1
    ADDSD X1, X0

    ANDQ $1, CX
    JZ   dot_sse2_done

    MOVSD (SI), X1
    MOVSD (DI), X2
    MULSD X2, X1
    ADDSD X1, X0

dot_sse2_done:
    MOVSD X0, ret+48(FP)
    RET

// func addSSE2(dst, a, b []float64)
TEXT ·addSSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   add_sse2_remainder

add_sse2_loop2:
    MOVUPD (SI), X0
    MOVUPD (DI), X1
    ADDPD X1, X0
    MOVUPD X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ AX
    JNZ  add_sse2_loop2

add_sse2_remainder:
    ANDQ $1, CX
    JZ   add_sse2_done

    MOVSD (SI), X0
    ADDSD (DI), X0
    MOVSD X0, (DX)

add_sse2_done:
    RET

// func subSSE2(dst, a, b []float64)
TEXT ·subSSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   sub_sse2_remainder

sub_sse2_loop2:
    MOVUPD (SI), X0
    MOVUPD (DI), X1
    SUBPD X1, X0
    MOVUPD X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ AX
    JNZ  sub_sse2_loop2

sub_sse2_remainder:
    ANDQ $1, CX
    JZ   sub_sse2_done

    MOVSD (SI), X0
    SUBSD (DI), X0
    MOVSD X0, (DX)

sub_sse2_done:
    RET

// func mulSSE2(dst, a, b []float64)
TEXT ·mulSSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   mul_sse2_remainder

mul_sse2_loop2:
    MOVUPD (SI), X0
    MOVUPD (DI), X1
    MULPD X1, X0
    MOVUPD X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ AX
    JNZ  mul_sse2_loop2

mul_sse2_remainder:
    ANDQ $1, CX
    JZ   mul_sse2_done

    MOVSD (SI), X0
    MULSD (DI), X0
    MOVSD X0, (DX)

mul_sse2_done:
    RET

// func divSSE2(dst, a, b []float64)
TEXT ·divSSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   div_sse2_remainder

div_sse2_loop2:
    MOVUPD (SI), X0
    MOVUPD (DI), X1
    DIVPD X1, X0
    MOVUPD X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ AX
    JNZ  div_sse2_loop2

div_sse2_remainder:
    ANDQ $1, CX
    JZ   div_sse2_done

    MOVSD (SI), X0
    DIVSD (DI), X0
    MOVSD X0, (DX)

div_sse2_done:
    RET

// func scaleSSE2(dst, a []float64, s float64)
TEXT ·scaleSSE2(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVSD s+48(FP), X1
    SHUFPD $0, X1, X1  // Broadcast to both lanes

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   scale_sse2_remainder

scale_sse2_loop2:
    MOVUPD (SI), X0
    MULPD X1, X0
    MOVUPD X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  scale_sse2_loop2

scale_sse2_remainder:
    ANDQ $1, CX
    JZ   scale_sse2_done

    MOVSD (SI), X0
    MULSD X1, X0
    MOVSD X0, (DX)

scale_sse2_done:
    RET

// func addScalarSSE2(dst, a []float64, s float64)
TEXT ·addScalarSSE2(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVSD s+48(FP), X1
    SHUFPD $0, X1, X1

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   addsc_sse2_remainder

addsc_sse2_loop2:
    MOVUPD (SI), X0
    ADDPD X1, X0
    MOVUPD X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  addsc_sse2_loop2

addsc_sse2_remainder:
    ANDQ $1, CX
    JZ   addsc_sse2_done

    MOVSD (SI), X0
    ADDSD X1, X0
    MOVSD X0, (DX)

addsc_sse2_done:
    RET

// func sumSSE2(a []float64) float64
TEXT ·sumSSE2(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    XORPD X0, X0

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   sum_sse2_remainder

sum_sse2_loop2:
    MOVUPD (SI), X1
    ADDPD X1, X0
    ADDQ $16, SI
    DECQ AX
    JNZ  sum_sse2_loop2

sum_sse2_remainder:
    MOVAPD X0, X1
    SHUFPD $1, X1, X1
    ADDSD X1, X0

    ANDQ $1, CX
    JZ   sum_sse2_done

    MOVSD (SI), X1
    ADDSD X1, X0

sum_sse2_done:
    MOVSD X0, ret+24(FP)
    RET

// func minSSE2(a []float64) float64
TEXT ·minSSE2(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    MOVUPD (SI), X0
    ADDQ $16, SI
    SUBQ $2, CX

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   min_sse2_reduce

min_sse2_loop2:
    MOVUPD (SI), X1
    MINPD X1, X0
    ADDQ $16, SI
    DECQ AX
    JNZ  min_sse2_loop2

min_sse2_reduce:
    MOVAPD X0, X1
    SHUFPD $1, X1, X1
    MINSD X1, X0

    ANDQ $1, CX
    JZ   min_sse2_done

    MOVSD (SI), X1
    MINSD X1, X0

min_sse2_done:
    MOVSD X0, ret+24(FP)
    RET

// func maxSSE2(a []float64) float64
TEXT ·maxSSE2(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    MOVUPD (SI), X0
    ADDQ $16, SI
    SUBQ $2, CX

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   max_sse2_reduce

max_sse2_loop2:
    MOVUPD (SI), X1
    MAXPD X1, X0
    ADDQ $16, SI
    DECQ AX
    JNZ  max_sse2_loop2

max_sse2_reduce:
    MOVAPD X0, X1
    SHUFPD $1, X1, X1
    MAXSD X1, X0

    ANDQ $1, CX
    JZ   max_sse2_done

    MOVSD (SI), X1
    MAXSD X1, X0

max_sse2_done:
    MOVSD X0, ret+24(FP)
    RET

// func absSSE2(dst, a []float64)
TEXT ·absSSE2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVUPD absf64mask<>(SB), X2

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   abs_sse2_remainder

abs_sse2_loop2:
    MOVUPD (SI), X0
    ANDPD X2, X0
    MOVUPD X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  abs_sse2_loop2

abs_sse2_remainder:
    ANDQ $1, CX
    JZ   abs_sse2_done

    MOVSD (SI), X0
    ANDPD X2, X0
    MOVSD X0, (DX)

abs_sse2_done:
    RET

// func negSSE2(dst, a []float64)
TEXT ·negSSE2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    XORPD X2, X2

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   neg_sse2_remainder

neg_sse2_loop2:
    MOVUPD (SI), X0
    MOVAPD X2, X1
    SUBPD X0, X1
    MOVUPD X1, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  neg_sse2_loop2

neg_sse2_remainder:
    ANDQ $1, CX
    JZ   neg_sse2_done

    MOVSD (SI), X0
    MOVAPD X2, X1
    SUBSD X0, X1
    MOVSD X1, (DX)

neg_sse2_done:
    RET

// func fmaSSE2(dst, a, b, c []float64) - emulated, no hardware FMA
TEXT ·fmaSSE2(SB), NOSPLIT, $0-96
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI
    MOVQ c_base+72(FP), R8

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   fma_sse2_remainder

fma_sse2_loop2:
    MOVUPD (SI), X0
    MOVUPD (DI), X1
    MOVUPD (R8), X2
    MULPD X1, X0
    ADDPD X2, X0
    MOVUPD X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, R8
    ADDQ $16, DX
    DECQ AX
    JNZ  fma_sse2_loop2

fma_sse2_remainder:
    ANDQ $1, CX
    JZ   fma_sse2_done

    MOVSD (SI), X0
    MOVSD (DI), X1
    MOVSD (R8), X2
    MULSD X1, X0
    ADDSD X2, X0
    MOVSD X0, (DX)

fma_sse2_done:
    RET

// func clampSSE2(dst, a []float64, minVal, maxVal float64)
TEXT ·clampSSE2(SB), NOSPLIT, $0-64
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVSD minVal+48(FP), X1
    MOVSD maxVal+56(FP), X2
    SHUFPD $0, X1, X1
    SHUFPD $0, X2, X2

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   clamp_sse2_remainder

clamp_sse2_loop2:
    MOVUPD (SI), X0
    MAXPD X1, X0
    MINPD X2, X0
    MOVUPD X0, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  clamp_sse2_loop2

clamp_sse2_remainder:
    ANDQ $1, CX
    JZ   clamp_sse2_done

    MOVSD (SI), X0
    MAXSD X1, X0
    MINSD X2, X0
    MOVSD X0, (DX)

clamp_sse2_done:
    RET

// func sqrtSSE2(dst, a []float64)
TEXT ·sqrtSSE2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   sqrt_sse2_remainder

sqrt_sse2_loop2:
    MOVUPD (SI), X0
    SQRTPD X0, X1
    MOVUPD X1, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  sqrt_sse2_loop2

sqrt_sse2_remainder:
    ANDQ $1, CX
    JZ   sqrt_sse2_done

    MOVSD (SI), X0
    SQRTSD X0, X0
    MOVSD X0, (DX)

sqrt_sse2_done:
    RET

// func reciprocalSSE2(dst, a []float64)
TEXT ·reciprocalSSE2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVQ $0x3FF0000000000000, AX
    MOVQ AX, X3
    SHUFPD $0, X3, X3

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   recip_sse2_remainder

recip_sse2_loop2:
    MOVUPD (SI), X0
    MOVAPD X3, X1
    DIVPD X0, X1
    MOVUPD X1, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  recip_sse2_loop2

recip_sse2_remainder:
    ANDQ $1, CX
    JZ   recip_sse2_done

    MOVSD (SI), X0
    MOVAPD X3, X1
    DIVSD X0, X1
    MOVSD X1, (DX)

recip_sse2_done:
    RET

// func varianceSSE2(a []float64, mean float64) float64
TEXT ·varianceSSE2(SB), NOSPLIT, $0-40
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVSD mean+24(FP), X2
    SHUFPD $0, X2, X2

    XORPD X0, X0

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   var_sse2_remainder

var_sse2_loop2:
    MOVUPD (SI), X1
    SUBPD X2, X1
    MULPD X1, X1
    ADDPD X1, X0
    ADDQ $16, SI
    DECQ AX
    JNZ  var_sse2_loop2

var_sse2_remainder:
    MOVAPD X0, X1
    SHUFPD $1, X1, X1
    ADDSD X1, X0

    ANDQ $1, CX
    JZ   var_sse2_divide

    MOVSD (SI), X1
    MOVSD mean+24(FP), X2
    SUBSD X2, X1
    MULSD X1, X1
    ADDSD X1, X0

var_sse2_divide:
    MOVQ a_len+8(FP), CX
    CVTSQ2SD CX, X1
    DIVSD X1, X0
    MOVSD X0, ret+32(FP)
    RET

// func euclideanDistanceSSE2(a, b []float64) float64
TEXT ·euclideanDistanceSSE2(SB), NOSPLIT, $0-56
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    XORPD X0, X0

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   euclid_sse2_remainder

euclid_sse2_loop2:
    MOVUPD (SI), X1
    MOVUPD (DI), X2
    SUBPD X2, X1
    MULPD X1, X1
    ADDPD X1, X0
    ADDQ $16, SI
    ADDQ $16, DI
    DECQ AX
    JNZ  euclid_sse2_loop2

euclid_sse2_remainder:
    MOVAPD X0, X1
    SHUFPD $1, X1, X1
    ADDSD X1, X0

    ANDQ $1, CX
    JZ   euclid_sse2_sqrt

    MOVSD (SI), X1
    MOVSD (DI), X2
    SUBSD X2, X1
    MULSD X1, X1
    ADDSD X1, X0

euclid_sse2_sqrt:
    SQRTSD X0, X0
    MOVSD X0, ret+48(FP)
    RET
