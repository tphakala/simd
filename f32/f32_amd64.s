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

// Constants for roundAVX (round-half-away-from-zero). absf32mask above doubles
// as the |frac| mask; these supply the sign bit, the 0.5 threshold and 1.0.
DATA roundf32_signmask<>+0x00(SB)/4, $0x80000000
DATA roundf32_signmask<>+0x04(SB)/4, $0x80000000
DATA roundf32_signmask<>+0x08(SB)/4, $0x80000000
DATA roundf32_signmask<>+0x0c(SB)/4, $0x80000000
DATA roundf32_signmask<>+0x10(SB)/4, $0x80000000
DATA roundf32_signmask<>+0x14(SB)/4, $0x80000000
DATA roundf32_signmask<>+0x18(SB)/4, $0x80000000
DATA roundf32_signmask<>+0x1c(SB)/4, $0x80000000
GLOBL roundf32_signmask<>(SB), RODATA|NOPTR, $32

DATA roundf32_half<>+0x00(SB)/4, $0x3f000000
DATA roundf32_half<>+0x04(SB)/4, $0x3f000000
DATA roundf32_half<>+0x08(SB)/4, $0x3f000000
DATA roundf32_half<>+0x0c(SB)/4, $0x3f000000
DATA roundf32_half<>+0x10(SB)/4, $0x3f000000
DATA roundf32_half<>+0x14(SB)/4, $0x3f000000
DATA roundf32_half<>+0x18(SB)/4, $0x3f000000
DATA roundf32_half<>+0x1c(SB)/4, $0x3f000000
GLOBL roundf32_half<>(SB), RODATA|NOPTR, $32

DATA roundf32_one<>+0x00(SB)/4, $0x3f800000
DATA roundf32_one<>+0x04(SB)/4, $0x3f800000
DATA roundf32_one<>+0x08(SB)/4, $0x3f800000
DATA roundf32_one<>+0x0c(SB)/4, $0x3f800000
DATA roundf32_one<>+0x10(SB)/4, $0x3f800000
DATA roundf32_one<>+0x14(SB)/4, $0x3f800000
DATA roundf32_one<>+0x18(SB)/4, $0x3f800000
DATA roundf32_one<>+0x1c(SB)/4, $0x3f800000
GLOBL roundf32_one<>(SB), RODATA|NOPTR, $32

// func dotProductAVX(a, b []float32) float32
// Optimized with 4 independent accumulators to hide FMA latency.
// Processes 32 float32s per iteration (4 vectors × 8 floats).
// Handles mismatched slice lengths: uses min(len(a), len(b)).
TEXT ·dotProductAVX(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_len+32(FP), AX
    CMPQ AX, CX
    CMOVQLT AX, CX             // CX = min(len(a), len(b))
    MOVQ b_base+24(FP), DI

    // Initialize 4 independent accumulators
    VXORPS Y0, Y0, Y0          // acc0
    VXORPS Y3, Y3, Y3          // acc1
    VXORPS Y4, Y4, Y4          // acc2
    VXORPS Y5, Y5, Y5          // acc3

    // Process 32 elements per iteration (4 vectors × 8 floats)
    MOVQ CX, AX
    SHRQ $5, AX                // len / 32
    JZ   dot32_loop8_check

dot32_loop32:
    // Load and FMA for acc0
    VMOVUPS (SI), Y1
    VMOVUPS (DI), Y2
    VFMADD231PS Y1, Y2, Y0

    // Load and FMA for acc1
    VMOVUPS 32(SI), Y1
    VMOVUPS 32(DI), Y2
    VFMADD231PS Y1, Y2, Y3

    // Load and FMA for acc2
    VMOVUPS 64(SI), Y1
    VMOVUPS 64(DI), Y2
    VFMADD231PS Y1, Y2, Y4

    // Load and FMA for acc3
    VMOVUPS 96(SI), Y1
    VMOVUPS 96(DI), Y2
    VFMADD231PS Y1, Y2, Y5

    ADDQ $128, SI
    ADDQ $128, DI
    DECQ AX
    JNZ  dot32_loop32

    // Combine accumulators: Y0 = Y0 + Y3 + Y4 + Y5
    VADDPS Y3, Y0, Y0
    VADDPS Y4, Y0, Y0
    VADDPS Y5, Y0, Y0

dot32_loop8_check:
    // Handle remaining 8-element chunks
    ANDQ $31, CX
    MOVQ CX, AX
    SHRQ $3, AX
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
//
// Division Latency Hiding via 4x Loop Unrolling (float32)
// =======================================================
// VDIVPS has high latency (~11 cycles) but good throughput (~5 cycles).
// By issuing 4 independent VDIVPS instructions per iteration, we allow
// the CPU to overlap their execution, achieving closer to throughput limit.
//
// float32 advantage: YMM registers hold 8 floats vs 4 doubles, so we process
// 32 elements per 4x-unrolled iteration (vs 16 for float64).
//
// Timing (Alder Lake P-core, from uops.info):
//   - VDIVPS YMM latency: ~11 cycles
//   - VDIVPS YMM throughput: ~5 cycles
//
TEXT ·divAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    // Process 32 elements per iteration (4 vectors × 8 floats)
    // This allows 4 independent VDIVPS operations to be in-flight
    MOVQ CX, AX
    SHRQ $5, AX                // len / 32
    JZ   div32_loop8_check

div32_loop32:
    // Load 4 vectors from a (128 bytes = 32 floats)
    VMOVUPS 0(SI), Y0
    VMOVUPS 32(SI), Y2
    VMOVUPS 64(SI), Y4
    VMOVUPS 96(SI), Y6
    // Load 4 vectors from b (128 bytes = 32 floats)
    VMOVUPS 0(DI), Y1
    VMOVUPS 32(DI), Y3
    VMOVUPS 64(DI), Y5
    VMOVUPS 96(DI), Y7
    // Issue 4 independent divisions - no data dependencies between them
    // CPU can execute these in parallel using pipelined divider unit
    VDIVPS Y1, Y0, Y0
    VDIVPS Y3, Y2, Y2
    VDIVPS Y5, Y4, Y4
    VDIVPS Y7, Y6, Y6
    // Store results (128 bytes = 32 floats)
    VMOVUPS Y0, 0(DX)
    VMOVUPS Y2, 32(DX)
    VMOVUPS Y4, 64(DX)
    VMOVUPS Y6, 96(DX)
    ADDQ $128, SI
    ADDQ $128, DI
    ADDQ $128, DX
    DECQ AX
    JNZ  div32_loop32

div32_loop8_check:
    // Handle remaining 8-element chunks
    ANDQ $31, CX
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
// Optimized with 4 independent accumulators to hide ADD latency (4 cycles).
TEXT ·sumAVX(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    // Initialize 4 independent accumulators
    VXORPS Y0, Y0, Y0          // acc0
    VXORPS Y3, Y3, Y3          // acc1
    VXORPS Y4, Y4, Y4          // acc2
    VXORPS Y5, Y5, Y5          // acc3

    // Process 32 elements per iteration (4 vectors × 8 floats)
    MOVQ CX, AX
    SHRQ $5, AX                // len / 32
    JZ   sum32_loop8_check

sum32_loop32:
    VADDPS (SI), Y0, Y0
    VADDPS 32(SI), Y3, Y3
    VADDPS 64(SI), Y4, Y4
    VADDPS 96(SI), Y5, Y5
    ADDQ $128, SI
    DECQ AX
    JNZ  sum32_loop32

    // Combine accumulators
    VADDPS Y3, Y0, Y0
    VADDPS Y4, Y0, Y0
    VADDPS Y5, Y0, Y0

sum32_loop8_check:
    ANDQ $31, CX
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   sum32_remainder

sum32_loop8:
    VADDPS (SI), Y0, Y0
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

// func maxAbsAVX(a []float32) float32
// max_i |a[i]|. Mirrors maxAVX with the sign bit cleared (VANDPS absf32mask)
// on each loaded vector and the scalar tail before the running max.
TEXT ·maxAbsAVX(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VMOVUPS absf32mask<>(SB), Y2   // Y2 = abs mask (X2 = low 128 for the tail)
    VMOVUPS (SI), Y0
    VANDPS Y0, Y2, Y0             // Y0 = |a[0:8]|
    ADDQ $32, SI
    SUBQ $8, CX

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   maxabs32_avx_reduce

maxabs32_avx_loop8:
    VMOVUPS (SI), Y1
    VANDPS Y1, Y2, Y1
    VMAXPS Y0, Y1, Y0
    ADDQ $32, SI
    DECQ AX
    JNZ  maxabs32_avx_loop8

maxabs32_avx_reduce:
    VEXTRACTF128 $1, Y0, X1
    VMAXPS X0, X1, X0
    VPERMILPS $0x0E, X0, X1
    VMAXPS X0, X1, X0
    VPERMILPS $0x01, X0, X1
    VMAXSS X0, X1, X0

    ANDQ $7, CX
    JZ   maxabs32_avx_done

maxabs32_avx_scalar:
    VMOVSS (SI), X1
    VANDPS X1, X2, X1
    VMAXSS X0, X1, X0
    ADDQ $4, SI
    DECQ CX
    JNZ  maxabs32_avx_scalar

maxabs32_avx_done:
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
// Optimized with 4 independent accumulators to hide FMA latency.
// Processes 64 float32s per iteration (4 vectors × 16 floats).
// Handles mismatched slice lengths: uses min(len(a), len(b)).
TEXT ·dotProductAVX512(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_len+32(FP), AX
    CMPQ AX, CX
    CMOVQLT AX, CX             // CX = min(len(a), len(b))
    MOVQ b_base+24(FP), DI

    // Initialize 4 independent accumulators
    VPXORD Z0, Z0, Z0          // acc0
    VPXORD Z3, Z3, Z3          // acc1
    VPXORD Z4, Z4, Z4          // acc2
    VPXORD Z5, Z5, Z5          // acc3

    // Process 64 elements per iteration (4 vectors × 16 floats)
    MOVQ CX, AX
    SHRQ $6, AX                // len / 64
    JZ   dot32_512_loop16_check

PCALIGN $64
dot32_512_loop64:
    // Load and FMA for acc0
    VMOVUPS (SI), Z1
    VMOVUPS (DI), Z2
    VFMADD231PS Z1, Z2, Z0

    // Load and FMA for acc1
    VMOVUPS 64(SI), Z1
    VMOVUPS 64(DI), Z2
    VFMADD231PS Z1, Z2, Z3

    // Load and FMA for acc2
    VMOVUPS 128(SI), Z1
    VMOVUPS 128(DI), Z2
    VFMADD231PS Z1, Z2, Z4

    // Load and FMA for acc3
    VMOVUPS 192(SI), Z1
    VMOVUPS 192(DI), Z2
    VFMADD231PS Z1, Z2, Z5

    ADDQ $256, SI
    ADDQ $256, DI
    DECQ AX
    JNZ  dot32_512_loop64

    // Combine accumulators: Z0 = Z0 + Z3 + Z4 + Z5
    VADDPS Z3, Z0, Z0
    VADDPS Z4, Z0, Z0
    VADDPS Z5, Z0, Z0

dot32_512_loop16_check:
    // Handle remaining 16-element chunks
    ANDQ $63, CX
    MOVQ CX, AX
    SHRQ $4, AX
    JZ   dot32_512_remainder

PCALIGN $64
dot32_512_loop16:
    VMOVUPS (SI), Z1
    VMOVUPS (DI), Z2
    VFMADD231PS Z1, Z2, Z0
    ADDQ $64, SI
    ADDQ $64, DI
    DECQ AX
    JNZ  dot32_512_loop16

dot32_512_remainder:
    // VEXTRACTF64X4 (AVX512F): same upper-256 extract as VEXTRACTF32X8, no AVX512DQ dep.
    VEXTRACTF64X4 $1, Z0, Y1
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
// Optimized with 4x unrolling to hide VDIVPS latency.
TEXT ·divAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    // Process 64 elements per iteration (4 vectors × 16 floats)
    MOVQ CX, AX
    SHRQ $6, AX                // len / 64
    JZ   div32_512_loop16_check

div32_512_loop64:
    // Load 4 vectors from a
    VMOVUPS 0(SI), Z0
    VMOVUPS 64(SI), Z2
    VMOVUPS 128(SI), Z4
    VMOVUPS 192(SI), Z6
    // Load 4 vectors from b
    VMOVUPS 0(DI), Z1
    VMOVUPS 64(DI), Z3
    VMOVUPS 128(DI), Z5
    VMOVUPS 192(DI), Z7
    // Issue 4 independent divisions
    VDIVPS Z1, Z0, Z0
    VDIVPS Z3, Z2, Z2
    VDIVPS Z5, Z4, Z4
    VDIVPS Z7, Z6, Z6
    // Store results
    VMOVUPS Z0, 0(DX)
    VMOVUPS Z2, 64(DX)
    VMOVUPS Z4, 128(DX)
    VMOVUPS Z6, 192(DX)
    ADDQ $256, SI
    ADDQ $256, DI
    ADDQ $256, DX
    DECQ AX
    JNZ  div32_512_loop64

div32_512_loop16_check:
    // Handle remaining 16-element chunks
    ANDQ $63, CX
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
// Optimized with 4 independent accumulators to hide ADD latency (4 cycles).
TEXT ·sumAVX512(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    // Initialize 4 independent accumulators
    VPXORD Z0, Z0, Z0          // acc0
    VPXORD Z3, Z3, Z3          // acc1
    VPXORD Z4, Z4, Z4          // acc2
    VPXORD Z5, Z5, Z5          // acc3

    // Process 64 elements per iteration (4 vectors × 16 floats)
    MOVQ CX, AX
    SHRQ $6, AX                // len / 64
    JZ   sum32_512_loop16_check

sum32_512_loop64:
    VADDPS (SI), Z0, Z0
    VADDPS 64(SI), Z3, Z3
    VADDPS 128(SI), Z4, Z4
    VADDPS 192(SI), Z5, Z5
    ADDQ $256, SI
    DECQ AX
    JNZ  sum32_512_loop64

    // Combine accumulators
    VADDPS Z3, Z0, Z0
    VADDPS Z4, Z0, Z0
    VADDPS Z5, Z0, Z0

sum32_512_loop16_check:
    ANDQ $63, CX
    MOVQ CX, AX
    SHRQ $4, AX
    JZ   sum32_512_remainder

sum32_512_loop16:
    VADDPS (SI), Z0, Z0
    ADDQ $64, SI
    DECQ AX
    JNZ  sum32_512_loop16

sum32_512_remainder:
    // VEXTRACTF64X4 (AVX512F): same upper-256 extract as VEXTRACTF32X8, no AVX512DQ dep.
    VEXTRACTF64X4 $1, Z0, Y1
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
    // VEXTRACTF64X4 (AVX512F): same upper-256 extract as VEXTRACTF32X8, no AVX512DQ dep.
    VEXTRACTF64X4 $1, Z0, Y1
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
    // VEXTRACTF64X4 (AVX512F): same upper-256 extract as VEXTRACTF32X8, no AVX512DQ dep.
    VEXTRACTF64X4 $1, Z0, Y1
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

    VPXORD Z2, Z2, Z2

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
// Handles mismatched slice lengths: uses min(len(a), len(b)).
TEXT ·dotProductSSE(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_len+32(FP), AX
    CMPQ AX, CX
    CMOVQLT AX, CX             // CX = min(len(a), len(b))
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

// func maxAbsSSE(a []float32) float32
// max_i |a[i]|. Mirrors maxSSE with ANDPS absf32mask folded into each load.
TEXT ·maxAbsSSE(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    MOVUPS absf32mask<>(SB), X2
    MOVUPS (SI), X0
    ANDPS X2, X0                 // X0 = |a[0:4]|
    ADDQ $16, SI
    SUBQ $4, CX

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   maxabs32_sse_reduce

maxabs32_sse_loop4:
    MOVUPS (SI), X1
    ANDPS X2, X1
    MAXPS X1, X0
    ADDQ $16, SI
    DECQ AX
    JNZ  maxabs32_sse_loop4

maxabs32_sse_reduce:
    MOVAPS X0, X1
    SHUFPS $0x0E, X1, X1
    MAXPS X1, X0
    MOVAPS X0, X1
    SHUFPS $0x01, X1, X1
    MAXSS X1, X0

    ANDQ $3, CX
    JZ   maxabs32_sse_done

maxabs32_sse_scalar:
    MOVSS (SI), X1
    ANDPS X2, X1
    MAXSS X1, X0
    ADDQ $4, SI
    DECQ CX
    JNZ  maxabs32_sse_scalar

maxabs32_sse_done:
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

// ============================================================================
// INTERLEAVE/DEINTERLEAVE IMPLEMENTATIONS
// ============================================================================

// func interleave2AVX(dst, a, b []float32)
// Interleaves two slices: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], dst[3]=b[1], ...
TEXT ·interleave2AVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX    // dst pointer
    MOVQ a_base+24(FP), SI     // a pointer
    MOVQ a_len+32(FP), CX      // n = len(a)
    MOVQ b_base+48(FP), DI     // b pointer

    // Process 8 pairs at a time (16 output elements)
    MOVQ CX, AX
    SHRQ $3, AX                // AX = n / 8
    JZ   interleave2_avx32_remainder

interleave2_avx32_loop8:
    // Load 8 from a: Y0 = [a0..a7]
    VMOVUPS (SI), Y0
    // Load 8 from b: Y1 = [b0..b7]
    VMOVUPS (DI), Y1

    // Unpack within 128-bit lanes
    VUNPCKLPS Y1, Y0, Y2       // [a0,b0,a1,b1 | a4,b4,a5,b5]
    VUNPCKHPS Y1, Y0, Y3       // [a2,b2,a3,b3 | a6,b6,a7,b7]

    // Permute to get final order
    VPERM2F128 $0x20, Y3, Y2, Y4  // [a0,b0,a1,b1,a2,b2,a3,b3]
    VPERM2F128 $0x31, Y3, Y2, Y5  // [a4,b4,a5,b5,a6,b6,a7,b7]

    // Store 16 elements to dst
    VMOVUPS Y4, (DX)
    VMOVUPS Y5, 32(DX)

    ADDQ $32, SI               // a += 8 * 4
    ADDQ $32, DI               // b += 8 * 4
    ADDQ $64, DX               // dst += 16 * 4
    DECQ AX
    JNZ  interleave2_avx32_loop8

interleave2_avx32_remainder:
    ANDQ $7, CX
    JZ   interleave2_avx32_done

interleave2_avx32_scalar:
    VMOVSS (SI), X0
    VMOVSS (DI), X1
    VMOVSS X0, (DX)
    VMOVSS X1, 4(DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  interleave2_avx32_scalar

interleave2_avx32_done:
    VZEROUPPER
    RET

// func deinterleave2AVX(a, b, src []float32)
// Deinterleaves: a[0]=src[0], b[0]=src[1], a[1]=src[2], b[1]=src[3], ...
// Algorithm based on Intel AVX intrinsics guide:
// 1. VSHUFPS $0x88 extracts evens (a's), $0xDD extracts odds (b's)
// 2. VPERM2F128 + VUNPCKLPD/VUNPCKHPD reorders 64-bit chunks across lanes
TEXT ·deinterleave2AVX(SB), NOSPLIT, $0-72
    MOVQ a_base+0(FP), DX      // a pointer
    MOVQ a_len+8(FP), CX       // n = len(a)
    MOVQ b_base+24(FP), R8     // b pointer
    MOVQ src_base+48(FP), SI   // src pointer

    // Process 8 pairs at a time
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   deinterleave2_avx32_remainder

deinterleave2_avx32_loop8:
    // Load 16 interleaved elements
    // Y0 = [a0,b0,a1,b1 | a2,b2,a3,b3]
    // Y1 = [a4,b4,a5,b5 | a6,b6,a7,b7]
    VMOVUPS (SI), Y0
    VMOVUPS 32(SI), Y1

    // Step 1: Extract evens and odds using VSHUFPS
    // VSHUFPS $0x88: select indices 0,2 from each source per lane
    // Y2 = [a0,a1,a4,a5 | a2,a3,a6,a7]
    VSHUFPS $0x88, Y1, Y0, Y2
    // VSHUFPS $0xDD: select indices 1,3 from each source per lane
    // Y3 = [b0,b1,b4,b5 | b2,b3,b6,b7]
    VSHUFPS $0xDD, Y1, Y0, Y3

    // Step 2: Reorder Y2 to get [a0,a1,a2,a3 | a4,a5,a6,a7]
    // Swap lanes: Y4 = [a2,a3,a6,a7 | a0,a1,a4,a5]
    VPERM2F128 $0x01, Y2, Y2, Y4
    // Unpack low 64-bit pairs: Y5 = [a0,a1,a2,a3 | ...]
    VUNPCKLPD Y4, Y2, Y5
    // Unpack high 64-bit pairs: Y6 = [a4,a5,a6,a7 | ...]
    VUNPCKHPD Y4, Y2, Y6
    // Combine low lanes: Y7 = [a0,a1,a2,a3 | a4,a5,a6,a7]
    VPERM2F128 $0x20, Y6, Y5, Y7

    // Step 3: Reorder Y3 to get [b0,b1,b2,b3 | b4,b5,b6,b7]
    // Swap lanes: Y4 = [b2,b3,b6,b7 | b0,b1,b4,b5]
    VPERM2F128 $0x01, Y3, Y3, Y4
    // Unpack low 64-bit pairs: Y5 = [b0,b1,b2,b3 | ...]
    VUNPCKLPD Y4, Y3, Y5
    // Unpack high 64-bit pairs: Y6 = [b4,b5,b6,b7 | ...]
    VUNPCKHPD Y4, Y3, Y6
    // Combine low lanes: Y4 = [b0,b1,b2,b3 | b4,b5,b6,b7]
    VPERM2F128 $0x20, Y6, Y5, Y4

    // Store results
    VMOVUPS Y7, (DX)           // store a
    VMOVUPS Y4, (R8)           // store b

    ADDQ $64, SI               // src += 16 * 4
    ADDQ $32, DX               // a += 8 * 4
    ADDQ $32, R8               // b += 8 * 4
    DECQ AX
    JNZ  deinterleave2_avx32_loop8

deinterleave2_avx32_remainder:
    ANDQ $7, CX
    JZ   deinterleave2_avx32_done

deinterleave2_avx32_scalar:
    VMOVSS (SI), X0
    VMOVSS 4(SI), X1
    VMOVSS X0, (DX)
    VMOVSS X1, (R8)
    ADDQ $8, SI
    ADDQ $4, DX
    ADDQ $4, R8
    DECQ CX
    JNZ  deinterleave2_avx32_scalar

deinterleave2_avx32_done:
    VZEROUPPER
    RET

// 3-stream interleave/deinterleave gather indices (AVX2 VPERMPS).
//
// N=3 does not map onto a clean register transpose the way N=4 (4x4) and N=8
// (8x8) do, because 8 frames span 24 interleaved slots that straddle the
// 128-bit lane boundaries. Instead each 256-bit output is assembled from three
// VPERMPS gathers (one per source stream), merged with two VPBLENDD masks. The
// index vectors are loop-invariant, so the kernels load all nine into YMM
// registers once before the loop. "don't care" lanes are 0; the blend discards
// them. See interleave3AVX / deinterleave3AVX below for the slot maps.

// interleave: per 8 frames, dst = [a0 b0 c0 a1 b1 c1 ... a7 b7 c7].
// O0 = dst[0:8]   A@{0,3,6} B@{1,4,7} C@{2,5}
DATA il3idxA0<>+0(SB)/4, $0
DATA il3idxA0<>+4(SB)/4, $0
DATA il3idxA0<>+8(SB)/4, $0
DATA il3idxA0<>+12(SB)/4, $1
DATA il3idxA0<>+16(SB)/4, $0
DATA il3idxA0<>+20(SB)/4, $0
DATA il3idxA0<>+24(SB)/4, $2
DATA il3idxA0<>+28(SB)/4, $0
GLOBL il3idxA0<>(SB), RODATA|NOPTR, $32
DATA il3idxB0<>+0(SB)/4, $0
DATA il3idxB0<>+4(SB)/4, $0
DATA il3idxB0<>+8(SB)/4, $0
DATA il3idxB0<>+12(SB)/4, $0
DATA il3idxB0<>+16(SB)/4, $1
DATA il3idxB0<>+20(SB)/4, $0
DATA il3idxB0<>+24(SB)/4, $0
DATA il3idxB0<>+28(SB)/4, $2
GLOBL il3idxB0<>(SB), RODATA|NOPTR, $32
DATA il3idxC0<>+0(SB)/4, $0
DATA il3idxC0<>+4(SB)/4, $0
DATA il3idxC0<>+8(SB)/4, $0
DATA il3idxC0<>+12(SB)/4, $0
DATA il3idxC0<>+16(SB)/4, $0
DATA il3idxC0<>+20(SB)/4, $1
DATA il3idxC0<>+24(SB)/4, $0
DATA il3idxC0<>+28(SB)/4, $0
GLOBL il3idxC0<>(SB), RODATA|NOPTR, $32
// O1 = dst[8:16]  C@{0,3,6} A@{1,4,7} B@{2,5}
DATA il3idxA1<>+0(SB)/4, $0
DATA il3idxA1<>+4(SB)/4, $3
DATA il3idxA1<>+8(SB)/4, $0
DATA il3idxA1<>+12(SB)/4, $0
DATA il3idxA1<>+16(SB)/4, $4
DATA il3idxA1<>+20(SB)/4, $0
DATA il3idxA1<>+24(SB)/4, $0
DATA il3idxA1<>+28(SB)/4, $5
GLOBL il3idxA1<>(SB), RODATA|NOPTR, $32
DATA il3idxB1<>+0(SB)/4, $0
DATA il3idxB1<>+4(SB)/4, $0
DATA il3idxB1<>+8(SB)/4, $3
DATA il3idxB1<>+12(SB)/4, $0
DATA il3idxB1<>+16(SB)/4, $0
DATA il3idxB1<>+20(SB)/4, $4
DATA il3idxB1<>+24(SB)/4, $0
DATA il3idxB1<>+28(SB)/4, $0
GLOBL il3idxB1<>(SB), RODATA|NOPTR, $32
DATA il3idxC1<>+0(SB)/4, $2
DATA il3idxC1<>+4(SB)/4, $0
DATA il3idxC1<>+8(SB)/4, $0
DATA il3idxC1<>+12(SB)/4, $3
DATA il3idxC1<>+16(SB)/4, $0
DATA il3idxC1<>+20(SB)/4, $0
DATA il3idxC1<>+24(SB)/4, $4
DATA il3idxC1<>+28(SB)/4, $0
GLOBL il3idxC1<>(SB), RODATA|NOPTR, $32
// O2 = dst[16:24] B@{0,3,6} C@{1,4,7} A@{2,5}
DATA il3idxA2<>+0(SB)/4, $0
DATA il3idxA2<>+4(SB)/4, $0
DATA il3idxA2<>+8(SB)/4, $6
DATA il3idxA2<>+12(SB)/4, $0
DATA il3idxA2<>+16(SB)/4, $0
DATA il3idxA2<>+20(SB)/4, $7
DATA il3idxA2<>+24(SB)/4, $0
DATA il3idxA2<>+28(SB)/4, $0
GLOBL il3idxA2<>(SB), RODATA|NOPTR, $32
DATA il3idxB2<>+0(SB)/4, $5
DATA il3idxB2<>+4(SB)/4, $0
DATA il3idxB2<>+8(SB)/4, $0
DATA il3idxB2<>+12(SB)/4, $6
DATA il3idxB2<>+16(SB)/4, $0
DATA il3idxB2<>+20(SB)/4, $0
DATA il3idxB2<>+24(SB)/4, $7
DATA il3idxB2<>+28(SB)/4, $0
GLOBL il3idxB2<>(SB), RODATA|NOPTR, $32
DATA il3idxC2<>+0(SB)/4, $0
DATA il3idxC2<>+4(SB)/4, $5
DATA il3idxC2<>+8(SB)/4, $0
DATA il3idxC2<>+12(SB)/4, $0
DATA il3idxC2<>+16(SB)/4, $6
DATA il3idxC2<>+20(SB)/4, $0
DATA il3idxC2<>+24(SB)/4, $0
DATA il3idxC2<>+28(SB)/4, $7
GLOBL il3idxC2<>(SB), RODATA|NOPTR, $32

// deinterleave: src = [a0 b0 c0 a1 ...]; S0=src[0:8] S1=src[8:16] S2=src[16:24].
// A=d0[0:8]  A@out{0,1,2}<-S0{0,3,6}  {3,4,5}<-S1{1,4,7}  {6,7}<-S2{2,5}
DATA dl3idxS0A<>+0(SB)/4, $0
DATA dl3idxS0A<>+4(SB)/4, $3
DATA dl3idxS0A<>+8(SB)/4, $6
DATA dl3idxS0A<>+12(SB)/4, $0
DATA dl3idxS0A<>+16(SB)/4, $0
DATA dl3idxS0A<>+20(SB)/4, $0
DATA dl3idxS0A<>+24(SB)/4, $0
DATA dl3idxS0A<>+28(SB)/4, $0
GLOBL dl3idxS0A<>(SB), RODATA|NOPTR, $32
DATA dl3idxS1A<>+0(SB)/4, $0
DATA dl3idxS1A<>+4(SB)/4, $0
DATA dl3idxS1A<>+8(SB)/4, $0
DATA dl3idxS1A<>+12(SB)/4, $1
DATA dl3idxS1A<>+16(SB)/4, $4
DATA dl3idxS1A<>+20(SB)/4, $7
DATA dl3idxS1A<>+24(SB)/4, $0
DATA dl3idxS1A<>+28(SB)/4, $0
GLOBL dl3idxS1A<>(SB), RODATA|NOPTR, $32
DATA dl3idxS2A<>+0(SB)/4, $0
DATA dl3idxS2A<>+4(SB)/4, $0
DATA dl3idxS2A<>+8(SB)/4, $0
DATA dl3idxS2A<>+12(SB)/4, $0
DATA dl3idxS2A<>+16(SB)/4, $0
DATA dl3idxS2A<>+20(SB)/4, $0
DATA dl3idxS2A<>+24(SB)/4, $2
DATA dl3idxS2A<>+28(SB)/4, $5
GLOBL dl3idxS2A<>(SB), RODATA|NOPTR, $32
// B=d1  {0,1,2}<-S0{1,4,7}  {3,4}<-S1{2,5}  {5,6,7}<-S2{0,3,6}
DATA dl3idxS0B<>+0(SB)/4, $1
DATA dl3idxS0B<>+4(SB)/4, $4
DATA dl3idxS0B<>+8(SB)/4, $7
DATA dl3idxS0B<>+12(SB)/4, $0
DATA dl3idxS0B<>+16(SB)/4, $0
DATA dl3idxS0B<>+20(SB)/4, $0
DATA dl3idxS0B<>+24(SB)/4, $0
DATA dl3idxS0B<>+28(SB)/4, $0
GLOBL dl3idxS0B<>(SB), RODATA|NOPTR, $32
DATA dl3idxS1B<>+0(SB)/4, $0
DATA dl3idxS1B<>+4(SB)/4, $0
DATA dl3idxS1B<>+8(SB)/4, $0
DATA dl3idxS1B<>+12(SB)/4, $2
DATA dl3idxS1B<>+16(SB)/4, $5
DATA dl3idxS1B<>+20(SB)/4, $0
DATA dl3idxS1B<>+24(SB)/4, $0
DATA dl3idxS1B<>+28(SB)/4, $0
GLOBL dl3idxS1B<>(SB), RODATA|NOPTR, $32
DATA dl3idxS2B<>+0(SB)/4, $0
DATA dl3idxS2B<>+4(SB)/4, $0
DATA dl3idxS2B<>+8(SB)/4, $0
DATA dl3idxS2B<>+12(SB)/4, $0
DATA dl3idxS2B<>+16(SB)/4, $0
DATA dl3idxS2B<>+20(SB)/4, $0
DATA dl3idxS2B<>+24(SB)/4, $3
DATA dl3idxS2B<>+28(SB)/4, $6
GLOBL dl3idxS2B<>(SB), RODATA|NOPTR, $32
// C=d2  {0,1}<-S0{2,5}  {2,3,4}<-S1{0,3,6}  {5,6,7}<-S2{1,4,7}
DATA dl3idxS0C<>+0(SB)/4, $2
DATA dl3idxS0C<>+4(SB)/4, $5
DATA dl3idxS0C<>+8(SB)/4, $0
DATA dl3idxS0C<>+12(SB)/4, $0
DATA dl3idxS0C<>+16(SB)/4, $0
DATA dl3idxS0C<>+20(SB)/4, $0
DATA dl3idxS0C<>+24(SB)/4, $0
DATA dl3idxS0C<>+28(SB)/4, $0
GLOBL dl3idxS0C<>(SB), RODATA|NOPTR, $32
DATA dl3idxS1C<>+0(SB)/4, $0
DATA dl3idxS1C<>+4(SB)/4, $0
DATA dl3idxS1C<>+8(SB)/4, $0
DATA dl3idxS1C<>+12(SB)/4, $3
DATA dl3idxS1C<>+16(SB)/4, $6
DATA dl3idxS1C<>+20(SB)/4, $0
DATA dl3idxS1C<>+24(SB)/4, $0
DATA dl3idxS1C<>+28(SB)/4, $0
GLOBL dl3idxS1C<>(SB), RODATA|NOPTR, $32
DATA dl3idxS2C<>+0(SB)/4, $0
DATA dl3idxS2C<>+4(SB)/4, $0
DATA dl3idxS2C<>+8(SB)/4, $0
DATA dl3idxS2C<>+12(SB)/4, $0
DATA dl3idxS2C<>+16(SB)/4, $0
DATA dl3idxS2C<>+20(SB)/4, $1
DATA dl3idxS2C<>+24(SB)/4, $4
DATA dl3idxS2C<>+28(SB)/4, $7
GLOBL dl3idxS2C<>(SB), RODATA|NOPTR, $32

// func interleave3AVX(dst, s0, s1, s2 []float32, n int)
// Interleaves 3 planar streams (dst[i*3+c] = s_c[i]) into 24 contiguous floats
// per 8 frames via per-stream VPERMPS gathers + VPBLENDD merges. n is a
// multiple of 8 (the caller handles the tail). Requires AVX2.
TEXT ·interleave3AVX(SB), NOSPLIT, $0-104
    MOVQ dst_base+0(FP), DI
    MOVQ s0_base+24(FP), AX
    MOVQ s1_base+48(FP), BX
    MOVQ s2_base+72(FP), CX
    MOVQ n+96(FP), SI
    SHRQ $3, SI                // SI = n/8 blocks
    TESTQ SI, SI
    JZ interleave3_avx_done
    VMOVUPS il3idxA0<>(SB), Y3
    VMOVUPS il3idxB0<>(SB), Y4
    VMOVUPS il3idxC0<>(SB), Y5
    VMOVUPS il3idxA1<>(SB), Y6
    VMOVUPS il3idxB1<>(SB), Y7
    VMOVUPS il3idxC1<>(SB), Y8
    VMOVUPS il3idxA2<>(SB), Y9
    VMOVUPS il3idxB2<>(SB), Y10
    VMOVUPS il3idxC2<>(SB), Y11

interleave3_avx_loop:
    VMOVUPS (AX), Y0           // A = s0[i:i+8]
    VMOVUPS (BX), Y1           // B = s1[i:i+8]
    VMOVUPS (CX), Y2           // C = s2[i:i+8]
    VPERMPS Y0, Y3, Y12        // O0: gather A
    VPERMPS Y1, Y4, Y15        // gather B
    VPBLENDD $0x92, Y15, Y12, Y12
    VPERMPS Y2, Y5, Y15        // gather C
    VPBLENDD $0x24, Y15, Y12, Y12
    VMOVUPS Y12, (DI)
    VPERMPS Y0, Y6, Y12        // O1
    VPERMPS Y1, Y7, Y15
    VPBLENDD $0x24, Y15, Y12, Y12
    VPERMPS Y2, Y8, Y15
    VPBLENDD $0x49, Y15, Y12, Y12
    VMOVUPS Y12, 32(DI)
    VPERMPS Y0, Y9, Y12        // O2
    VPERMPS Y1, Y10, Y15
    VPBLENDD $0x49, Y15, Y12, Y12
    VPERMPS Y2, Y11, Y15
    VPBLENDD $0x92, Y15, Y12, Y12
    VMOVUPS Y12, 64(DI)
    ADDQ $32, AX
    ADDQ $32, BX
    ADDQ $32, CX
    ADDQ $96, DI
    DECQ SI
    JNZ interleave3_avx_loop

interleave3_avx_done:
    VZEROUPPER
    RET

// func deinterleave3AVX(d0, d1, d2, src []float32, n int)
// Splits a 3-stream interleaved buffer (d_c[i] = src[i*3+c]) into planar
// streams, 8 frames per iteration, via per-stream VPERMPS gathers + VPBLENDD
// merges. n is a multiple of 8 (the caller handles the tail). Requires AVX2.
TEXT ·deinterleave3AVX(SB), NOSPLIT, $0-104
    MOVQ d0_base+0(FP), AX
    MOVQ d1_base+24(FP), BX
    MOVQ d2_base+48(FP), CX
    MOVQ src_base+72(FP), SI
    MOVQ n+96(FP), DI
    SHRQ $3, DI                // DI = n/8 blocks
    TESTQ DI, DI
    JZ deinterleave3_avx_done
    VMOVUPS dl3idxS0A<>(SB), Y3
    VMOVUPS dl3idxS1A<>(SB), Y4
    VMOVUPS dl3idxS2A<>(SB), Y5
    VMOVUPS dl3idxS0B<>(SB), Y6
    VMOVUPS dl3idxS1B<>(SB), Y7
    VMOVUPS dl3idxS2B<>(SB), Y8
    VMOVUPS dl3idxS0C<>(SB), Y9
    VMOVUPS dl3idxS1C<>(SB), Y10
    VMOVUPS dl3idxS2C<>(SB), Y11

deinterleave3_avx_loop:
    VMOVUPS (SI), Y0           // S0 = src[0:8]
    VMOVUPS 32(SI), Y1         // S1 = src[8:16]
    VMOVUPS 64(SI), Y2         // S2 = src[16:24]
    VPERMPS Y0, Y3, Y12        // d0 = A
    VPERMPS Y1, Y4, Y15
    VPBLENDD $0x38, Y15, Y12, Y12
    VPERMPS Y2, Y5, Y15
    VPBLENDD $0xC0, Y15, Y12, Y12
    VMOVUPS Y12, (AX)
    VPERMPS Y0, Y6, Y12        // d1 = B
    VPERMPS Y1, Y7, Y15
    VPBLENDD $0x18, Y15, Y12, Y12
    VPERMPS Y2, Y8, Y15
    VPBLENDD $0xE0, Y15, Y12, Y12
    VMOVUPS Y12, (BX)
    VPERMPS Y0, Y9, Y12        // d2 = C
    VPERMPS Y1, Y10, Y15
    VPBLENDD $0x1C, Y15, Y12, Y12
    VPERMPS Y2, Y11, Y15
    VPBLENDD $0xE0, Y15, Y12, Y12
    VMOVUPS Y12, (CX)
    ADDQ $32, AX
    ADDQ $32, BX
    ADDQ $32, CX
    ADDQ $96, SI
    DECQ DI
    JNZ deinterleave3_avx_loop

deinterleave3_avx_done:
    VZEROUPPER
    RET

// func interleave4AVX(dst, s0, s1, s2, s3 []float32, n int)
// Interleaves 4 planar streams (dst[i*4+c] = s_c[i]) via a 4x4 transpose of
// XMM registers, 4 frames per iteration. n is a multiple of 4 (the caller
// handles the tail).
TEXT ·interleave4AVX(SB), NOSPLIT, $0-128
    MOVQ dst_base+0(FP), DI
    MOVQ s0_base+24(FP), AX
    MOVQ s1_base+48(FP), BX
    MOVQ s2_base+72(FP), CX
    MOVQ s3_base+96(FP), DX
    MOVQ n+120(FP), SI
    SHRQ $2, SI                // SI = n/4 blocks
    TESTQ SI, SI
    JZ interleave4_avx_done

interleave4_avx_loop:
    VMOVUPS (AX), X0           // r0 = s0[i:i+4]
    VMOVUPS (BX), X1           // r1 = s1[i:i+4]
    VMOVUPS (CX), X2           // r2 = s2[i:i+4]
    VMOVUPS (DX), X3           // r3 = s3[i:i+4]
    VUNPCKLPS X1, X0, X4       // [r0.0,r1.0,r0.1,r1.1]
    VUNPCKHPS X1, X0, X5       // [r0.2,r1.2,r0.3,r1.3]
    VUNPCKLPS X3, X2, X6       // [r2.0,r3.0,r2.1,r3.1]
    VUNPCKHPS X3, X2, X7       // [r2.2,r3.2,r2.3,r3.3]
    VSHUFPS $0x44, X6, X4, X8  // frame0 = [r0.0,r1.0,r2.0,r3.0]
    VSHUFPS $0xEE, X6, X4, X9  // frame1 = [r0.1,r1.1,r2.1,r3.1]
    VSHUFPS $0x44, X7, X5, X10 // frame2 = [r0.2,r1.2,r2.2,r3.2]
    VSHUFPS $0xEE, X7, X5, X11 // frame3 = [r0.3,r1.3,r2.3,r3.3]
    VMOVUPS X8, (DI)
    VMOVUPS X9, 16(DI)
    VMOVUPS X10, 32(DI)
    VMOVUPS X11, 48(DI)
    ADDQ $16, AX
    ADDQ $16, BX
    ADDQ $16, CX
    ADDQ $16, DX
    ADDQ $64, DI
    DECQ SI
    JNZ interleave4_avx_loop

interleave4_avx_done:
    VZEROUPPER
    RET

// func deinterleave4AVX(d0, d1, d2, d3, src []float32, n int)
// Splits an interleaved 4-stream buffer (d_c[i] = src[i*4+c]) via a 4x4
// transpose, 4 frames per iteration. n is a multiple of 4.
TEXT ·deinterleave4AVX(SB), NOSPLIT, $0-128
    MOVQ d0_base+0(FP), AX
    MOVQ d1_base+24(FP), BX
    MOVQ d2_base+48(FP), CX
    MOVQ d3_base+72(FP), DX
    MOVQ src_base+96(FP), SI
    MOVQ n+120(FP), DI
    SHRQ $2, DI                // DI = n/4 blocks
    TESTQ DI, DI
    JZ deinterleave4_avx_done

deinterleave4_avx_loop:
    VMOVUPS (SI), X0           // frame0 = src[0:4]
    VMOVUPS 16(SI), X1         // frame1
    VMOVUPS 32(SI), X2         // frame2
    VMOVUPS 48(SI), X3         // frame3
    VUNPCKLPS X1, X0, X4       // [f0c0,f1c0,f0c1,f1c1]
    VUNPCKHPS X1, X0, X5       // [f0c2,f1c2,f0c3,f1c3]
    VUNPCKLPS X3, X2, X6       // [f2c0,f3c0,f2c1,f3c1]
    VUNPCKHPS X3, X2, X7       // [f2c2,f3c2,f2c3,f3c3]
    VSHUFPS $0x44, X6, X4, X8  // chan0 = [f0c0,f1c0,f2c0,f3c0]
    VSHUFPS $0xEE, X6, X4, X9  // chan1
    VSHUFPS $0x44, X7, X5, X10 // chan2
    VSHUFPS $0xEE, X7, X5, X11 // chan3
    VMOVUPS X8, (AX)
    VMOVUPS X9, (BX)
    VMOVUPS X10, (CX)
    VMOVUPS X11, (DX)
    ADDQ $16, AX
    ADDQ $16, BX
    ADDQ $16, CX
    ADDQ $16, DX
    ADDQ $64, SI
    DECQ DI
    JNZ deinterleave4_avx_loop

deinterleave4_avx_done:
    VZEROUPPER
    RET

// func interleave8AVX(dst []float32, srcs [][]float32, n int)
// Interleaves 8 planar streams (dst[i*8+c] = srcs[c][i]) via an 8x8 transpose
// of YMM registers, 8 frames per iteration. n is a multiple of 8 (the caller
// handles the tail). srcs must have exactly 8 elements.
TEXT ·interleave8AVX(SB), NOSPLIT, $0-56
    MOVQ srcs_base+24(FP), R12   // R12 = &srcs[0] (array of slice headers)
    MOVQ 0(R12), AX              // srcs[0].ptr
    MOVQ 24(R12), BX             // srcs[1].ptr
    MOVQ 48(R12), CX             // srcs[2].ptr
    MOVQ 72(R12), DX             // srcs[3].ptr
    MOVQ 96(R12), R8             // srcs[4].ptr
    MOVQ 120(R12), R9            // srcs[5].ptr
    MOVQ 144(R12), R10           // srcs[6].ptr
    MOVQ 168(R12), R11           // srcs[7].ptr
    MOVQ dst_base+0(FP), DI
    MOVQ n+48(FP), SI
    SHRQ $3, SI                  // SI = n/8 blocks
    TESTQ SI, SI
    JZ interleave8_avx_done

interleave8_avx_loop:
    VMOVUPS (AX), Y0             // row k = srcs[k][i:i+8]
    VMOVUPS (BX), Y1
    VMOVUPS (CX), Y2
    VMOVUPS (DX), Y3
    VMOVUPS (R8), Y4
    VMOVUPS (R9), Y5
    VMOVUPS (R10), Y6
    VMOVUPS (R11), Y7
    VUNPCKLPS Y1, Y0, Y8
    VUNPCKHPS Y1, Y0, Y9
    VUNPCKLPS Y3, Y2, Y10
    VUNPCKHPS Y3, Y2, Y11
    VUNPCKLPS Y5, Y4, Y12
    VUNPCKHPS Y5, Y4, Y13
    VUNPCKLPS Y7, Y6, Y14
    VUNPCKHPS Y7, Y6, Y15
    VSHUFPS $0x44, Y10, Y8, Y0
    VSHUFPS $0xEE, Y10, Y8, Y1
    VSHUFPS $0x44, Y11, Y9, Y2
    VSHUFPS $0xEE, Y11, Y9, Y3
    VSHUFPS $0x44, Y14, Y12, Y4
    VSHUFPS $0xEE, Y14, Y12, Y5
    VSHUFPS $0x44, Y15, Y13, Y6
    VSHUFPS $0xEE, Y15, Y13, Y7
    VPERM2F128 $0x20, Y4, Y0, Y8   // frame0
    VPERM2F128 $0x20, Y5, Y1, Y9   // frame1
    VPERM2F128 $0x20, Y6, Y2, Y10  // frame2
    VPERM2F128 $0x20, Y7, Y3, Y11  // frame3
    VPERM2F128 $0x31, Y4, Y0, Y12  // frame4
    VPERM2F128 $0x31, Y5, Y1, Y13  // frame5
    VPERM2F128 $0x31, Y6, Y2, Y14  // frame6
    VPERM2F128 $0x31, Y7, Y3, Y15  // frame7
    VMOVUPS Y8, (DI)
    VMOVUPS Y9, 32(DI)
    VMOVUPS Y10, 64(DI)
    VMOVUPS Y11, 96(DI)
    VMOVUPS Y12, 128(DI)
    VMOVUPS Y13, 160(DI)
    VMOVUPS Y14, 192(DI)
    VMOVUPS Y15, 224(DI)
    ADDQ $32, AX
    ADDQ $32, BX
    ADDQ $32, CX
    ADDQ $32, DX
    ADDQ $32, R8
    ADDQ $32, R9
    ADDQ $32, R10
    ADDQ $32, R11
    ADDQ $256, DI
    DECQ SI
    JNZ interleave8_avx_loop

interleave8_avx_done:
    VZEROUPPER
    RET

// func deinterleave8AVX(dsts [][]float32, src []float32, n int)
// Splits an interleaved 8-stream buffer (dsts[c][i] = src[i*8+c]) via an 8x8
// transpose, 8 frames per iteration. n is a multiple of 8. dsts must have
// exactly 8 elements.
TEXT ·deinterleave8AVX(SB), NOSPLIT, $0-56
    MOVQ dsts_base+0(FP), R12    // R12 = &dsts[0] (array of slice headers)
    MOVQ 0(R12), AX             // dsts[0].ptr
    MOVQ 24(R12), BX            // dsts[1].ptr
    MOVQ 48(R12), CX            // dsts[2].ptr
    MOVQ 72(R12), DX            // dsts[3].ptr
    MOVQ 96(R12), R8            // dsts[4].ptr
    MOVQ 120(R12), R9           // dsts[5].ptr
    MOVQ 144(R12), R10          // dsts[6].ptr
    MOVQ 168(R12), R11          // dsts[7].ptr
    MOVQ src_base+24(FP), SI
    MOVQ n+48(FP), DI
    SHRQ $3, DI                 // DI = n/8 blocks
    TESTQ DI, DI
    JZ deinterleave8_avx_done

deinterleave8_avx_loop:
    VMOVUPS (SI), Y0            // frame k = src[k*8:k*8+8]
    VMOVUPS 32(SI), Y1
    VMOVUPS 64(SI), Y2
    VMOVUPS 96(SI), Y3
    VMOVUPS 128(SI), Y4
    VMOVUPS 160(SI), Y5
    VMOVUPS 192(SI), Y6
    VMOVUPS 224(SI), Y7
    VUNPCKLPS Y1, Y0, Y8
    VUNPCKHPS Y1, Y0, Y9
    VUNPCKLPS Y3, Y2, Y10
    VUNPCKHPS Y3, Y2, Y11
    VUNPCKLPS Y5, Y4, Y12
    VUNPCKHPS Y5, Y4, Y13
    VUNPCKLPS Y7, Y6, Y14
    VUNPCKHPS Y7, Y6, Y15
    VSHUFPS $0x44, Y10, Y8, Y0
    VSHUFPS $0xEE, Y10, Y8, Y1
    VSHUFPS $0x44, Y11, Y9, Y2
    VSHUFPS $0xEE, Y11, Y9, Y3
    VSHUFPS $0x44, Y14, Y12, Y4
    VSHUFPS $0xEE, Y14, Y12, Y5
    VSHUFPS $0x44, Y15, Y13, Y6
    VSHUFPS $0xEE, Y15, Y13, Y7
    VPERM2F128 $0x20, Y4, Y0, Y8   // chan0
    VPERM2F128 $0x20, Y5, Y1, Y9   // chan1
    VPERM2F128 $0x20, Y6, Y2, Y10  // chan2
    VPERM2F128 $0x20, Y7, Y3, Y11  // chan3
    VPERM2F128 $0x31, Y4, Y0, Y12  // chan4
    VPERM2F128 $0x31, Y5, Y1, Y13  // chan5
    VPERM2F128 $0x31, Y6, Y2, Y14  // chan6
    VPERM2F128 $0x31, Y7, Y3, Y15  // chan7
    VMOVUPS Y8, (AX)
    VMOVUPS Y9, (BX)
    VMOVUPS Y10, (CX)
    VMOVUPS Y11, (DX)
    VMOVUPS Y12, (R8)
    VMOVUPS Y13, (R9)
    VMOVUPS Y14, (R10)
    VMOVUPS Y15, (R11)
    ADDQ $32, AX
    ADDQ $32, BX
    ADDQ $32, CX
    ADDQ $32, DX
    ADDQ $32, R8
    ADDQ $32, R9
    ADDQ $32, R10
    ADDQ $32, R11
    ADDQ $256, SI
    DECQ DI
    JNZ deinterleave8_avx_loop

deinterleave8_avx_done:
    VZEROUPPER
    RET

// func interleave6AVX(dst, s0, s1, s2, s3, s4, s5 []float32, n int)
// Interleaves 6 planar streams (dst[i*6+c] = s_c[i]) into 48 contiguous floats
// per 8 frames. N=6 has no clean register transpose, so the kernel zips each
// stream pair (s0,s1)(s2,s3)(s4,s5) into a float64-pair stream via VUNPCKLPS/
// VUNPCKHPS + VPERM2F128, then runs the f64 N=3 interleave (immediate VPERMPD/
// VBLENDPD on those pairs) twice: once for frames 0-3, once for frames 4-7. n is
// a multiple of 8 (the caller handles the tail). Requires AVX2.
//
// After the zip, PA=[a0,b0,a1,b1,...] holds pair k = (s0[k],s1[k]) as one f64
// lane; PB/PC hold (s2,s3)/(s4,s5). Interleaving the three pair-streams with the
// f64 N=3 rows O0=[A0,B0,C0,A1] O1=[B1,C1,A2,B2] O2=[C2,A3,B3,C3] lays the six
// streams down frame-by-frame.
TEXT ·interleave6AVX(SB), NOSPLIT, $0-176
    MOVQ dst_base+0(FP), DI
    MOVQ s0_base+24(FP), AX
    MOVQ s1_base+48(FP), BX
    MOVQ s2_base+72(FP), CX
    MOVQ s3_base+96(FP), DX
    MOVQ s4_base+120(FP), R8
    MOVQ s5_base+144(FP), R9
    MOVQ n+168(FP), SI
    SHRQ $3, SI                  // SI = n/8 blocks
    TESTQ SI, SI
    JZ interleave6_avx_done

interleave6_avx_loop:
    VMOVUPS (AX), Y0             // s0[i:i+8]
    VMOVUPS (BX), Y1             // s1
    VMOVUPS (CX), Y2             // s2
    VMOVUPS (DX), Y3             // s3
    VMOVUPS (R8), Y4             // s4
    VMOVUPS (R9), Y5             // s5
    VUNPCKLPS Y1, Y0, Y6         // zip (s0,s1): [a0,b0,a1,b1|a4,b4,a5,b5]
    VUNPCKHPS Y1, Y0, Y7         // [a2,b2,a3,b3|a6,b6,a7,b7]
    VPERM2F128 $0x20, Y7, Y6, Y8 // PA_lo=[a0,b0,a1,b1,a2,b2,a3,b3]
    VPERM2F128 $0x31, Y7, Y6, Y9 // PA_hi=[a4,b4,a5,b5,a6,b6,a7,b7]
    VUNPCKLPS Y3, Y2, Y6         // zip (s2,s3)
    VUNPCKHPS Y3, Y2, Y7
    VPERM2F128 $0x20, Y7, Y6, Y10 // PB_lo
    VPERM2F128 $0x31, Y7, Y6, Y11 // PB_hi
    VUNPCKLPS Y5, Y4, Y6         // zip (s4,s5)
    VUNPCKHPS Y5, Y4, Y7
    VPERM2F128 $0x20, Y7, Y6, Y12 // PC_lo
    VPERM2F128 $0x31, Y7, Y6, Y13 // PC_hi
    // interleave3 of pairs (frames 0-3): a=PA_lo b=PB_lo c=PC_lo
    VPERMPD $0x40, Y8, Y0        // [a0,a0,a0,a1]
    VPERMPD $0x00, Y10, Y1       // [b0,b0,b0,b0]
    VPERMPD $0x00, Y12, Y2       // [c0,c0,c0,c0]
    VBLENDPD $0x2, Y1, Y0, Y0    // lane1 <- b0
    VBLENDPD $0x4, Y2, Y0, Y0    // lane2 <- c0
    VMOVUPD Y0, (DI)             // O0=[a0,b0,c0,a1]
    VPERMPD $0x81, Y10, Y1       // [b1,b0,b0,b2]
    VPERMPD $0x04, Y12, Y2       // [c0,c1,c0,c0]
    VPERMPD $0x20, Y8, Y0        // [a0,a0,a2,a0]
    VBLENDPD $0x2, Y2, Y1, Y1    // lane1 <- c1
    VBLENDPD $0x4, Y0, Y1, Y1    // lane2 <- a2
    VMOVUPD Y1, 32(DI)           // O1=[b1,c1,a2,b2]
    VPERMPD $0xC2, Y12, Y2       // [c2,c0,c0,c3]
    VPERMPD $0x0C, Y8, Y0        // [a0,a3,a0,a0]
    VPERMPD $0x30, Y10, Y1       // [b0,b0,b3,b0]
    VBLENDPD $0x2, Y0, Y2, Y2    // lane1 <- a3
    VBLENDPD $0x4, Y1, Y2, Y2    // lane2 <- b3
    VMOVUPD Y2, 64(DI)           // O2=[c2,a3,b3,c3]
    // interleave3 of pairs (frames 4-7): a=PA_hi b=PB_hi c=PC_hi
    VPERMPD $0x40, Y9, Y0
    VPERMPD $0x00, Y11, Y1
    VPERMPD $0x00, Y13, Y2
    VBLENDPD $0x2, Y1, Y0, Y0
    VBLENDPD $0x4, Y2, Y0, Y0
    VMOVUPD Y0, 96(DI)
    VPERMPD $0x81, Y11, Y1
    VPERMPD $0x04, Y13, Y2
    VPERMPD $0x20, Y9, Y0
    VBLENDPD $0x2, Y2, Y1, Y1
    VBLENDPD $0x4, Y0, Y1, Y1
    VMOVUPD Y1, 128(DI)
    VPERMPD $0xC2, Y13, Y2
    VPERMPD $0x0C, Y9, Y0
    VPERMPD $0x30, Y11, Y1
    VBLENDPD $0x2, Y0, Y2, Y2
    VBLENDPD $0x4, Y1, Y2, Y2
    VMOVUPD Y2, 160(DI)
    ADDQ $32, AX
    ADDQ $32, BX
    ADDQ $32, CX
    ADDQ $32, DX
    ADDQ $32, R8
    ADDQ $32, R9
    ADDQ $192, DI
    DECQ SI
    JNZ interleave6_avx_loop

interleave6_avx_done:
    VZEROUPPER
    RET

// func deinterleave6AVX(d0, d1, d2, d3, d4, d5, src []float32, n int)
// Splits a 6-stream interleaved buffer (d_c[i] = src[i*6+c]) into planar streams,
// 8 frames per iteration. The inverse of interleave6AVX: reinterpreting src as a
// 3-stream interleaved float64-pair buffer, the f64 N=3 deinterleave recovers the
// three pair-streams PA/PB/PC, then VSHUFPS/VPERMPD unzip each pair back into its
// two float32 planar streams. n is a multiple of 8 (the caller handles the tail).
// Requires AVX2.
TEXT ·deinterleave6AVX(SB), NOSPLIT, $0-176
    MOVQ d0_base+0(FP), AX
    MOVQ d1_base+24(FP), BX
    MOVQ d2_base+48(FP), CX
    MOVQ d3_base+72(FP), DX
    MOVQ d4_base+96(FP), R8
    MOVQ d5_base+120(FP), R9
    MOVQ src_base+144(FP), SI
    MOVQ n+168(FP), DI
    SHRQ $3, DI                  // DI = n/8 blocks
    TESTQ DI, DI
    JZ deinterleave6_avx_done

deinterleave6_avx_loop:
    VMOVUPD (SI), Y0             // S0=[A0,B0,C0,A1] (frames 0-3, pair-interleaved)
    VMOVUPD 32(SI), Y1           // S1=[B1,C1,A2,B2]
    VMOVUPD 64(SI), Y2           // S2=[C2,A3,B3,C3]
    VMOVUPD 96(SI), Y3           // S3..S5 = frames 4-7
    VMOVUPD 128(SI), Y4
    VMOVUPD 160(SI), Y5
    // f64 N=3 deinterleave (frames 0-3) -> PA_lo,PB_lo,PC_lo
    VPERMPD $0x0C, Y0, Y8        // A base [a0,a1,a0,a0]
    VPERMPD $0x20, Y1, Y6        // [.,.,a2,.]
    VPERMPD $0x40, Y2, Y7        // [.,.,.,a3]
    VBLENDPD $0x4, Y6, Y8, Y8    // lane2 <- a2
    VBLENDPD $0x8, Y7, Y8, Y8    // lane3 <- a3; PA_lo=[a0,a1,a2,a3]
    VPERMPD $0x01, Y0, Y10       // B base [b0,a0,a0,a0]
    VPERMPD $0x30, Y1, Y6        // [b1,b1,b2,.]
    VPERMPD $0x80, Y2, Y7        // [.,.,.,b3]
    VBLENDPD $0x6, Y6, Y10, Y10  // lanes1,2 <- b1,b2
    VBLENDPD $0x8, Y7, Y10, Y10  // lane3 <- b3; PB_lo=[b0,b1,b2,b3]
    VPERMPD $0x02, Y0, Y12       // C base [c0,.,.,.]
    VPERMPD $0x04, Y1, Y6        // [.,c1,.,.]
    VPERMPD $0xC0, Y2, Y7        // [c2,.,c2,c3]
    VBLENDPD $0x2, Y6, Y12, Y12  // lane1 <- c1
    VBLENDPD $0xC, Y7, Y12, Y12  // lanes2,3 <- c2,c3; PC_lo=[c0,c1,c2,c3]
    // f64 N=3 deinterleave (frames 4-7) -> PA_hi,PB_hi,PC_hi
    VPERMPD $0x0C, Y3, Y9
    VPERMPD $0x20, Y4, Y6
    VPERMPD $0x40, Y5, Y7
    VBLENDPD $0x4, Y6, Y9, Y9
    VBLENDPD $0x8, Y7, Y9, Y9
    VPERMPD $0x01, Y3, Y11
    VPERMPD $0x30, Y4, Y6
    VPERMPD $0x80, Y5, Y7
    VBLENDPD $0x6, Y6, Y11, Y11
    VBLENDPD $0x8, Y7, Y11, Y11
    VPERMPD $0x02, Y3, Y13
    VPERMPD $0x04, Y4, Y6
    VPERMPD $0xC0, Y5, Y7
    VBLENDPD $0x2, Y6, Y13, Y13
    VBLENDPD $0xC, Y7, Y13, Y13
    // unzip pair-streams back to planar float32 (a even lanes, b odd lanes)
    VSHUFPS $0x88, Y9, Y8, Y0    // [a0,a1,a4,a5,a2,a3,a6,a7]
    VSHUFPS $0xDD, Y9, Y8, Y1    // [b0,b1,b4,b5,b2,b3,b6,b7]
    VPERMPD $0xD8, Y0, Y0        // s0=[a0..a7]
    VPERMPD $0xD8, Y1, Y1        // s1=[b0..b7]
    VMOVUPS Y0, (AX)
    VMOVUPS Y1, (BX)
    VSHUFPS $0x88, Y11, Y10, Y2
    VSHUFPS $0xDD, Y11, Y10, Y6
    VPERMPD $0xD8, Y2, Y2        // s2
    VPERMPD $0xD8, Y6, Y6        // s3
    VMOVUPS Y2, (CX)
    VMOVUPS Y6, (DX)
    VSHUFPS $0x88, Y13, Y12, Y3
    VSHUFPS $0xDD, Y13, Y12, Y7
    VPERMPD $0xD8, Y3, Y3        // s4
    VPERMPD $0xD8, Y7, Y7        // s5
    VMOVUPS Y3, (R8)
    VMOVUPS Y7, (R9)
    ADDQ $192, SI
    ADDQ $32, AX
    ADDQ $32, BX
    ADDQ $32, CX
    ADDQ $32, DX
    ADDQ $32, R8
    ADDQ $32, R9
    DECQ DI
    JNZ deinterleave6_avx_loop

deinterleave6_avx_done:
    VZEROUPPER
    RET

// ============================================================================
// SQRT, RECIPROCAL, ADDSCALED IMPLEMENTATIONS
// ============================================================================

// func sqrtAVX(dst, a []float32)
//
// Square Root Latency Hiding via 4x Loop Unrolling (float32)
// ==========================================================
// VSQRTPS has high latency (~12 cycles) but good throughput (~6 cycles).
// By issuing 4 independent VSQRTPS instructions, the CPU can overlap
// their execution using its pipelined sqrt unit.
//
// float32 advantage: YMM registers hold 8 floats vs 4 doubles, so we process
// 32 elements per 4x-unrolled iteration (vs 16 for float64).
//
// Timing (Alder Lake P-core, from uops.info):
//   - VSQRTPS YMM latency: ~12 cycles
//   - VSQRTPS YMM throughput: ~6 cycles
//
TEXT ·sqrtAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Process 32 elements per iteration (4 vectors × 8 floats)
    // This allows 4 independent VSQRTPS operations to overlap
    MOVQ CX, AX
    SHRQ $5, AX                // len / 32
    JZ   sqrt32_avx_loop8_check

sqrt32_avx_loop32:
    // Load 4 vectors from source (128 bytes = 32 floats)
    VMOVUPS 0(SI), Y0
    VMOVUPS 32(SI), Y2
    VMOVUPS 64(SI), Y4
    VMOVUPS 96(SI), Y6
    // Issue 4 independent sqrts - no data dependencies between them
    // CPU can execute these in parallel using pipelined sqrt unit
    VSQRTPS Y0, Y0
    VSQRTPS Y2, Y2
    VSQRTPS Y4, Y4
    VSQRTPS Y6, Y6
    // Store results (128 bytes = 32 floats)
    VMOVUPS Y0, 0(DX)
    VMOVUPS Y2, 32(DX)
    VMOVUPS Y4, 64(DX)
    VMOVUPS Y6, 96(DX)
    ADDQ $128, SI
    ADDQ $128, DX
    DECQ AX
    JNZ  sqrt32_avx_loop32

sqrt32_avx_loop8_check:
    // Handle remaining 8-element chunks
    ANDQ $31, CX
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   sqrt32_avx_remainder

sqrt32_avx_loop8:
    VMOVUPS (SI), Y0
    VSQRTPS Y0, Y1
    VMOVUPS Y1, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  sqrt32_avx_loop8

sqrt32_avx_remainder:
    ANDQ $7, CX
    JZ   sqrt32_avx_done

sqrt32_avx_scalar:
    VMOVSS (SI), X0
    VSQRTSS X0, X0, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  sqrt32_avx_scalar

sqrt32_avx_done:
    VZEROUPPER
    RET

// roundAVX: round-half-away-from-zero, the float32 analogue of f64's roundAVX.
// There is no single AVX instruction for round-half-away (VROUNDPS $0 rounds
// half-to-even), so it is emulated:
//   t   = trunc(x)
//   f   = x - t                     // signed fractional part in (-1, 1)
//   inc = (|f| >= 0.5) ? sign(x)*1.0 : 0
//   result = t + inc
// This avoids the trunc(|x|+0.5) overcounting at |x|=nextafter(0.5,0), where the
// FP add rounds up to exactly 1.0.
//
// func roundAVX(dst, src []float32)
TEXT ·roundAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    VMOVUPS absf32mask<>(SB), Y3      // |frac| mask (0x7fffffff per lane)
    VMOVUPS roundf32_signmask<>(SB), Y4
    VMOVUPS roundf32_half<>(SB), Y5
    VMOVUPS roundf32_one<>(SB), Y11

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   round32_avx_remainder

round32_avx_loop8:
    VMOVUPS (SI), Y0                  // Y0 = x
    VROUNDPS $3, Y0, Y1               // Y1 = trunc(x)
    VSUBPS Y1, Y0, Y2                 // Y2 = x - trunc(x) = signed frac
    VANDPS Y3, Y2, Y6                 // Y6 = |frac|
    VCMPPS $5, Y5, Y6, Y7             // Y7 = mask: |frac| NLT 0.5  (|frac| >= 0.5)
    VANDPS Y11, Y7, Y8                // Y8 = 1.0 where mask, else 0
    VANDPS Y4, Y0, Y9                 // Y9 = signbit(x)
    VORPS Y9, Y8, Y8                  // Y8 = ±1.0 (or ±0.0 when no increment)
    VADDPS Y8, Y1, Y1                 // Y1 = trunc(x) + ±1 or +0
    VMOVUPS Y1, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  round32_avx_loop8

round32_avx_remainder:
    ANDQ $7, CX
    JZ   round32_avx_done

    VMOVSS absf32mask<>(SB), X3
    VMOVSS roundf32_signmask<>(SB), X4
    VMOVSS roundf32_half<>(SB), X5
    VMOVSS roundf32_one<>(SB), X11

round32_avx_scalar:
    VMOVSS (SI), X0                   // X0 = x
    VROUNDSS $3, X0, X0, X1           // X1 = trunc(x)
    VSUBSS X1, X0, X2                 // X2 = signed frac
    VANDPS X3, X2, X6                 // X6 = |frac|
    VCMPSS $5, X5, X6, X7             // X7 = mask: |frac| NLT 0.5
    VANDPS X11, X7, X8                // X8 = 1.0 where mask else 0
    VANDPS X4, X0, X9                 // X9 = signbit(x)
    VORPS X9, X8, X8                  // X8 = ±1 or ±0
    VADDSS X8, X1, X1                 // X1 = trunc(x) + inc
    VMOVSS X1, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  round32_avx_scalar

round32_avx_done:
    VZEROUPPER
    RET

// func sqrtAVX512(dst, a []float32)
// Optimized with 4x unrolling to hide VSQRTPS latency.
TEXT ·sqrtAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Process 64 elements per iteration (4 vectors × 16 floats)
    MOVQ CX, AX
    SHRQ $6, AX                // len / 64
    JZ   sqrt32_avx512_loop16_check

sqrt32_avx512_loop64:
    // Load 4 vectors
    VMOVUPS 0(SI), Z0
    VMOVUPS 64(SI), Z2
    VMOVUPS 128(SI), Z4
    VMOVUPS 192(SI), Z6
    // Issue 4 independent sqrts
    VSQRTPS Z0, Z0
    VSQRTPS Z2, Z2
    VSQRTPS Z4, Z4
    VSQRTPS Z6, Z6
    // Store results
    VMOVUPS Z0, 0(DX)
    VMOVUPS Z2, 64(DX)
    VMOVUPS Z4, 128(DX)
    VMOVUPS Z6, 192(DX)
    ADDQ $256, SI
    ADDQ $256, DX
    DECQ AX
    JNZ  sqrt32_avx512_loop64

sqrt32_avx512_loop16_check:
    // Handle remaining 16-element chunks
    ANDQ $63, CX
    MOVQ CX, AX
    SHRQ $4, AX
    JZ   sqrt32_avx512_remainder

sqrt32_avx512_loop16:
    VMOVUPS (SI), Z0
    VSQRTPS Z0, Z1
    VMOVUPS Z1, (DX)
    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  sqrt32_avx512_loop16

sqrt32_avx512_remainder:
    ANDQ $15, CX
    JZ   sqrt32_avx512_done

sqrt32_avx512_scalar:
    VMOVSS (SI), X0
    VSQRTSS X0, X0, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  sqrt32_avx512_scalar

sqrt32_avx512_done:
    VZEROUPPER
    RET

// func sqrtSSE(dst, a []float32)
TEXT ·sqrtSSE(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   sqrt32_sse_remainder

sqrt32_sse_loop4:
    MOVUPS (SI), X0
    SQRTPS X0, X1
    MOVUPS X1, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  sqrt32_sse_loop4

sqrt32_sse_remainder:
    ANDQ $3, CX
    JZ   sqrt32_sse_done

sqrt32_sse_scalar:
    MOVSS (SI), X0
    SQRTSS X0, X0
    MOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  sqrt32_sse_scalar

sqrt32_sse_done:
    RET

// func reciprocalAVX(dst, a []float32)
//
// Reciprocal via Division with Latency Hiding (float32)
// =====================================================
// Computes 1.0/a[i] using VDIVPS for full IEEE 754 precision.
//
// Why not VRCPPS? While VRCPPS provides fast approximate reciprocal,
// it only delivers ~12-bit precision (~0.024% max relative error).
// For applications requiring full float32 precision (23-bit mantissa),
// we use actual division with 1.0 as the dividend.
//
// Same latency hiding strategy: 4x unrolling allows 4 independent
// VDIVPS operations to overlap, hiding the ~11 cycle latency.
//
// Timing (Alder Lake P-core, from uops.info):
//   - VDIVPS YMM latency: ~11 cycles
//   - VDIVPS YMM throughput: ~5 cycles
//
TEXT ·reciprocalAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Broadcast 1.0 to Y7 (preserved across all iterations)
    // 0x3f800000 is 1.0 in IEEE 754 single precision
    MOVL $0x3f800000, AX
    MOVD AX, X7
    VBROADCASTSS X7, Y7

    // Process 32 elements per iteration (4 vectors × 8 floats)
    // This allows 4 independent VDIVPS operations to be in-flight
    MOVQ CX, AX
    SHRQ $5, AX                // len / 32
    JZ   recip32_avx_loop8_check

recip32_avx_loop32:
    // Load 4 vectors from source (128 bytes = 32 floats)
    VMOVUPS 0(SI), Y0
    VMOVUPS 32(SI), Y1
    VMOVUPS 64(SI), Y2
    VMOVUPS 96(SI), Y3
    // Issue 4 independent divisions: dst[i] = 1.0 / a[i]
    // No data dependencies - CPU can execute in parallel
    VDIVPS Y0, Y7, Y0
    VDIVPS Y1, Y7, Y1
    VDIVPS Y2, Y7, Y2
    VDIVPS Y3, Y7, Y3
    // Store results (128 bytes = 32 floats)
    VMOVUPS Y0, 0(DX)
    VMOVUPS Y1, 32(DX)
    VMOVUPS Y2, 64(DX)
    VMOVUPS Y3, 96(DX)
    ADDQ $128, SI
    ADDQ $128, DX
    DECQ AX
    JNZ  recip32_avx_loop32

recip32_avx_loop8_check:
    // Handle remaining 8-element chunks
    ANDQ $31, CX
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   recip32_avx_remainder

recip32_avx_loop8:
    VMOVUPS (SI), Y0
    VDIVPS Y0, Y7, Y1
    VMOVUPS Y1, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  recip32_avx_loop8

recip32_avx_remainder:
    ANDQ $7, CX
    JZ   recip32_avx_done

recip32_avx_scalar:
    VMOVSS (SI), X0
    VDIVSS X0, X7, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  recip32_avx_scalar

recip32_avx_done:
    VZEROUPPER
    RET

// func reciprocalAVX512(dst, a []float32)
// Optimized with 4x unrolling to hide VDIVPS latency.
TEXT ·reciprocalAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Broadcast 1.0 to Z7 (preserved across iterations)
    MOVL $0x3f800000, AX      // 1.0 in float32
    VPBROADCASTD AX, Z7

    // Process 64 elements per iteration (4 vectors × 16 floats)
    MOVQ CX, AX
    SHRQ $6, AX                // len / 64
    JZ   recip32_avx512_loop16_check

recip32_avx512_loop64:
    // Load 4 vectors
    VMOVUPS 0(SI), Z0
    VMOVUPS 64(SI), Z1
    VMOVUPS 128(SI), Z2
    VMOVUPS 192(SI), Z3
    // Issue 4 independent divisions
    VDIVPS Z0, Z7, Z0
    VDIVPS Z1, Z7, Z1
    VDIVPS Z2, Z7, Z2
    VDIVPS Z3, Z7, Z3
    // Store results
    VMOVUPS Z0, 0(DX)
    VMOVUPS Z1, 64(DX)
    VMOVUPS Z2, 128(DX)
    VMOVUPS Z3, 192(DX)
    ADDQ $256, SI
    ADDQ $256, DX
    DECQ AX
    JNZ  recip32_avx512_loop64

recip32_avx512_loop16_check:
    // Handle remaining 16-element chunks
    ANDQ $63, CX
    MOVQ CX, AX
    SHRQ $4, AX
    JZ   recip32_avx512_remainder

recip32_avx512_loop16:
    VMOVUPS (SI), Z0
    VDIVPS Z0, Z7, Z1
    VMOVUPS Z1, (DX)
    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  recip32_avx512_loop16

recip32_avx512_remainder:
    ANDQ $15, CX
    JZ   recip32_avx512_done

recip32_avx512_scalar:
    VMOVSS (SI), X0
    VDIVSS X0, X7, X0
    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  recip32_avx512_scalar

recip32_avx512_done:
    VZEROUPPER
    RET

// func reciprocalSSE(dst, a []float32)
TEXT ·reciprocalSSE(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Load 1.0 to X2
    MOVL $0x3f800000, AX
    MOVD AX, X2
    SHUFPS $0, X2, X2         // Broadcast to all lanes

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   recip32_sse_remainder

recip32_sse_loop4:
    MOVUPS (SI), X0
    MOVAPS X2, X1
    DIVPS X0, X1
    MOVUPS X1, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  recip32_sse_loop4

recip32_sse_remainder:
    ANDQ $3, CX
    JZ   recip32_sse_done

recip32_sse_scalar:
    MOVSS (SI), X0
    MOVSS X2, X1
    DIVSS X0, X1
    MOVSS X1, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  recip32_sse_scalar

recip32_sse_done:
    RET

// func addScaledAVX(dst []float32, alpha float32, s []float32)
// dst[i] += alpha * s[i]
// Frame: dst(24) + alpha(4 padded to 8) + s(24) = 56 bytes
TEXT ·addScaledAVX(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVSS alpha+24(FP), X3
    MOVQ s_base+32(FP), SI

    // Broadcast alpha to Y3
    VBROADCASTSS X3, Y3

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   addscaled32_avx_remainder

addscaled32_avx_loop8:
    VMOVUPS (SI), Y0          // s[i:i+8]
    VMOVUPS (DX), Y1          // dst[i:i+8]
    VFMADD231PS Y0, Y3, Y1    // dst += alpha * s
    VMOVUPS Y1, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  addscaled32_avx_loop8

addscaled32_avx_remainder:
    ANDQ $7, CX
    JZ   addscaled32_avx_done

addscaled32_avx_scalar:
    VMOVSS (SI), X0
    VMOVSS (DX), X1
    VFMADD231SS X0, X3, X1
    VMOVSS X1, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  addscaled32_avx_scalar

addscaled32_avx_done:
    VZEROUPPER
    RET

// func addScaledAVX512(dst []float32, alpha float32, s []float32)
TEXT ·addScaledAVX512(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVSS alpha+24(FP), X3
    MOVQ s_base+32(FP), SI

    // Broadcast alpha to Z3
    VBROADCASTSS X3, Z3

    MOVQ CX, AX
    SHRQ $4, AX
    JZ   addscaled32_avx512_remainder

addscaled32_avx512_loop16:
    VMOVUPS (SI), Z0          // s[i:i+16]
    VMOVUPS (DX), Z1          // dst[i:i+16]
    VFMADD231PS Z0, Z3, Z1    // dst += alpha * s
    VMOVUPS Z1, (DX)
    ADDQ $64, SI
    ADDQ $64, DX
    DECQ AX
    JNZ  addscaled32_avx512_loop16

addscaled32_avx512_remainder:
    ANDQ $15, CX
    JZ   addscaled32_avx512_done

addscaled32_avx512_scalar:
    VMOVSS (SI), X0
    VMOVSS (DX), X1
    VFMADD231SS X0, X3, X1
    VMOVSS X1, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  addscaled32_avx512_scalar

addscaled32_avx512_done:
    VZEROUPPER
    RET

// func addScaledSSE(dst []float32, alpha float32, s []float32)
// dst[i] += alpha * s[i]
TEXT ·addScaledSSE(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVSS alpha+24(FP), X3
    MOVQ s_base+32(FP), SI

    // Broadcast alpha to X3
    SHUFPS $0, X3, X3

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   addscaled32_sse_remainder

addscaled32_sse_loop4:
    MOVUPS (SI), X0           // s[i:i+4]
    MOVUPS (DX), X1           // dst[i:i+4]
    MULPS X3, X0              // alpha * s
    ADDPS X0, X1              // dst + alpha * s
    MOVUPS X1, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  addscaled32_sse_loop4

addscaled32_sse_remainder:
    ANDQ $3, CX
    JZ   addscaled32_sse_done

addscaled32_sse_scalar:
    MOVSS (SI), X0
    MOVSS (DX), X1
    MULSS X3, X0
    ADDSS X0, X1
    MOVSS X1, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  addscaled32_sse_scalar

addscaled32_sse_done:
    RET

// ============================================================================
// CUBIC INTERPOLATION DOT PRODUCT
// ============================================================================

// func cubicInterpDotAVX(hist, a, b, c, d []float32, x float32) float32
// Computes: Σ hist[i] * (a[i] + x*(b[i] + x*(c[i] + x*d[i])))
// Uses Horner's method for polynomial evaluation with FMA.
// Processes 8 float32 per iteration with dual accumulators.
//
// Frame layout (5 slices + 1 float32 + padding + 1 return):
//   hist: base+0, len+8, cap+16
//   a:    base+24, len+32, cap+40
//   b:    base+48, len+56, cap+64
//   c:    base+72, len+80, cap+88
//   d:    base+96, len+104, cap+112
//   x:    +120 (float32)
//   pad:  +124 (4 bytes padding for alignment)
//   ret:  +128 (float32)
TEXT ·cubicInterpDotAVX(SB), NOSPLIT, $0-132
    MOVQ hist_base+0(FP), SI   // SI = hist pointer
    MOVQ hist_len+8(FP), CX    // CX = length
    MOVQ a_base+24(FP), DI     // DI = a pointer
    MOVQ b_base+48(FP), R8     // R8 = b pointer
    MOVQ c_base+72(FP), R9     // R9 = c pointer
    MOVQ d_base+96(FP), R10    // R10 = d pointer

    // Broadcast x to all 8 lanes of Y7
    VBROADCASTSS x+120(FP), Y7

    // Initialize dual accumulators to zero for ILP
    VXORPS Y0, Y0, Y0          // acc0
    VXORPS Y6, Y6, Y6          // acc1

    // Process 16 elements per iteration (2 vectors × 8 floats)
    MOVQ CX, AX
    SHRQ $4, AX                // AX = len / 16
    JZ   cubic32_avx_loop8_check

cubic32_avx_loop16:
    // Multi-stage Horner with two independent chains for ILP.
    // Each VFMADD213PS dst, mul, add computes dst = dst*mul + add.

    // Stage 1: p = c + x*d (two parallel chains)
    VMOVUPS 0(R9), Y1                       // Y1 = c[0:8]
    VMOVUPS 32(R9), Y2                      // Y2 = c[8:16]
    VMOVUPS 0(R10), Y5
    VFMADD231PS Y5, Y7, Y1                  // Y1 = d*x + c
    VMOVUPS 32(R10), Y5
    VFMADD231PS Y5, Y7, Y2                  // Y2 = d*x + c

    // Stage 2: p = b + x*p
    VMOVUPS 0(R8), Y5
    VFMADD213PS Y5, Y7, Y1                  // Y1 = Y1*x + b
    VMOVUPS 32(R8), Y5
    VFMADD213PS Y5, Y7, Y2                  // Y2 = Y2*x + b

    // Stage 3: coef = a + x*p
    VMOVUPS 0(DI), Y3
    VFMADD213PS Y3, Y7, Y1                  // Y1 = Y1*x + a
    VMOVUPS 32(DI), Y4
    VFMADD213PS Y4, Y7, Y2                  // Y2 = Y2*x + a

    // Accumulate with independent accumulators.
    VMOVUPS 0(SI), Y5
    VFMADD231PS Y5, Y1, Y0                  // acc0 += hist * coef
    VMOVUPS 32(SI), Y5
    VFMADD231PS Y5, Y2, Y6                  // acc1 += hist * coef

    // Advance pointers
    ADDQ $64, SI
    ADDQ $64, DI
    ADDQ $64, R8
    ADDQ $64, R9
    ADDQ $64, R10
    DECQ AX
    JNZ  cubic32_avx_loop16

    // Combine accumulators: Y0 = Y0 + Y6
    VADDPS Y6, Y0, Y0

cubic32_avx_loop8_check:
    // Handle remaining 8-element chunks
    ANDQ $15, CX
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   cubic32_avx_remainder

cubic32_avx_loop8:
    VMOVUPS (R10), Y1          // Y1 = d
    VMOVUPS (R9), Y2           // Y2 = c
    VMOVUPS (R8), Y3           // Y3 = b
    VMOVUPS (DI), Y4           // Y4 = a
    VMOVUPS (SI), Y5           // Y5 = hist

    VFMADD231PS Y1, Y7, Y2     // Y2 = d*x + c
    VFMADD231PS Y2, Y7, Y3     // Y3 = (d*x+c)*x + b
    VFMADD231PS Y3, Y7, Y4     // Y4 = coef
    VFMADD231PS Y5, Y4, Y0     // acc += hist * coef

    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, R8
    ADDQ $32, R9
    ADDQ $32, R10
    DECQ AX
    JNZ  cubic32_avx_loop8

cubic32_avx_remainder:
    // Reduce Y0 to scalar first (before scalar ops that zero upper bits)
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0         // X0[0] = sum of all 8 elements

    // Handle remaining 1-7 elements
    ANDQ $7, CX
    JZ   cubic32_avx_done

cubic32_avx_scalar:
    // Load single elements
    VMOVSS (R10), X1           // X1 = d[i]
    VMOVSS (R9), X2            // X2 = c[i]
    VMOVSS (R8), X3            // X3 = b[i]
    VMOVSS (DI), X4            // X4 = a[i]
    VMOVSS (SI), X5            // X5 = hist[i]
    VMOVSS x+120(FP), X6       // X6 = x

    // Horner's method for scalar
    VFMADD231SS X1, X6, X2     // X2 = d*x + c
    VFMADD231SS X2, X6, X3     // X3 = (d*x+c)*x + b
    VFMADD231SS X3, X6, X4     // X4 = coef

    // Accumulate
    VFMADD231SS X5, X4, X0     // X0 = hist * coef + X0

    // Advance pointers
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, R8
    ADDQ $4, R9
    ADDQ $4, R10
    DECQ CX
    JNZ  cubic32_avx_scalar

cubic32_avx_done:
    VMOVSS X0, ret+128(FP)
    VZEROUPPER
    RET

// Constants for exp/sigmoid computation using range reduction + polynomial
// exp(x) = 2^k * exp(r) where k = round(x * log2e), r = x - k * ln2
// Then polynomial approximation for exp(r) on [-ln2/2, ln2/2]

// log2(e) = 1/ln(2) ≈ 1.442695041
DATA exp_log2e<>+0x00(SB)/4, $0x3fb8aa3b
DATA exp_log2e<>+0x04(SB)/4, $0x3fb8aa3b
DATA exp_log2e<>+0x08(SB)/4, $0x3fb8aa3b
DATA exp_log2e<>+0x0c(SB)/4, $0x3fb8aa3b
DATA exp_log2e<>+0x10(SB)/4, $0x3fb8aa3b
DATA exp_log2e<>+0x14(SB)/4, $0x3fb8aa3b
DATA exp_log2e<>+0x18(SB)/4, $0x3fb8aa3b
DATA exp_log2e<>+0x1c(SB)/4, $0x3fb8aa3b
GLOBL exp_log2e<>(SB), RODATA|NOPTR, $32

// ln(2) ≈ 0.693147181
DATA exp_ln2<>+0x00(SB)/4, $0x3f317218
DATA exp_ln2<>+0x04(SB)/4, $0x3f317218
DATA exp_ln2<>+0x08(SB)/4, $0x3f317218
DATA exp_ln2<>+0x0c(SB)/4, $0x3f317218
DATA exp_ln2<>+0x10(SB)/4, $0x3f317218
DATA exp_ln2<>+0x14(SB)/4, $0x3f317218
DATA exp_ln2<>+0x18(SB)/4, $0x3f317218
DATA exp_ln2<>+0x1c(SB)/4, $0x3f317218
GLOBL exp_ln2<>(SB), RODATA|NOPTR, $32

// 0.5 for rounding
DATA exp_half<>+0x00(SB)/4, $0x3f000000
DATA exp_half<>+0x04(SB)/4, $0x3f000000
DATA exp_half<>+0x08(SB)/4, $0x3f000000
DATA exp_half<>+0x0c(SB)/4, $0x3f000000
DATA exp_half<>+0x10(SB)/4, $0x3f000000
DATA exp_half<>+0x14(SB)/4, $0x3f000000
DATA exp_half<>+0x18(SB)/4, $0x3f000000
DATA exp_half<>+0x1c(SB)/4, $0x3f000000
GLOBL exp_half<>(SB), RODATA|NOPTR, $32

// 1.0
DATA exp_one<>+0x00(SB)/4, $0x3f800000
DATA exp_one<>+0x04(SB)/4, $0x3f800000
DATA exp_one<>+0x08(SB)/4, $0x3f800000
DATA exp_one<>+0x0c(SB)/4, $0x3f800000
DATA exp_one<>+0x10(SB)/4, $0x3f800000
DATA exp_one<>+0x14(SB)/4, $0x3f800000
DATA exp_one<>+0x18(SB)/4, $0x3f800000
DATA exp_one<>+0x1c(SB)/4, $0x3f800000
GLOBL exp_one<>(SB), RODATA|NOPTR, $32

// Polynomial coefficients for exp(r) ≈ 1 + r + c2*r^2 + c3*r^3 + c4*r^4 + c5*r^5
// c2 = 0.5
DATA exp_c2<>+0x00(SB)/4, $0x3f000000
DATA exp_c2<>+0x04(SB)/4, $0x3f000000
DATA exp_c2<>+0x08(SB)/4, $0x3f000000
DATA exp_c2<>+0x0c(SB)/4, $0x3f000000
DATA exp_c2<>+0x10(SB)/4, $0x3f000000
DATA exp_c2<>+0x14(SB)/4, $0x3f000000
DATA exp_c2<>+0x18(SB)/4, $0x3f000000
DATA exp_c2<>+0x1c(SB)/4, $0x3f000000
GLOBL exp_c2<>(SB), RODATA|NOPTR, $32

// c3 = 1/6 ≈ 0.16666667
DATA exp_c3<>+0x00(SB)/4, $0x3e2aaaab
DATA exp_c3<>+0x04(SB)/4, $0x3e2aaaab
DATA exp_c3<>+0x08(SB)/4, $0x3e2aaaab
DATA exp_c3<>+0x0c(SB)/4, $0x3e2aaaab
DATA exp_c3<>+0x10(SB)/4, $0x3e2aaaab
DATA exp_c3<>+0x14(SB)/4, $0x3e2aaaab
DATA exp_c3<>+0x18(SB)/4, $0x3e2aaaab
DATA exp_c3<>+0x1c(SB)/4, $0x3e2aaaab
GLOBL exp_c3<>(SB), RODATA|NOPTR, $32

// c4 = 1/24 ≈ 0.041666668
DATA exp_c4<>+0x00(SB)/4, $0x3d2aaaab
DATA exp_c4<>+0x04(SB)/4, $0x3d2aaaab
DATA exp_c4<>+0x08(SB)/4, $0x3d2aaaab
DATA exp_c4<>+0x0c(SB)/4, $0x3d2aaaab
DATA exp_c4<>+0x10(SB)/4, $0x3d2aaaab
DATA exp_c4<>+0x14(SB)/4, $0x3d2aaaab
DATA exp_c4<>+0x18(SB)/4, $0x3d2aaaab
DATA exp_c4<>+0x1c(SB)/4, $0x3d2aaaab
GLOBL exp_c4<>(SB), RODATA|NOPTR, $32

// c5 = 1/120 ≈ 0.008333334
DATA exp_c5<>+0x00(SB)/4, $0x3c088889
DATA exp_c5<>+0x04(SB)/4, $0x3c088889
DATA exp_c5<>+0x08(SB)/4, $0x3c088889
DATA exp_c5<>+0x0c(SB)/4, $0x3c088889
DATA exp_c5<>+0x10(SB)/4, $0x3c088889
DATA exp_c5<>+0x14(SB)/4, $0x3c088889
DATA exp_c5<>+0x18(SB)/4, $0x3c088889
DATA exp_c5<>+0x1c(SB)/4, $0x3c088889
GLOBL exp_c5<>(SB), RODATA|NOPTR, $32

// Clamp threshold for sigmoid: beyond ±20, sigmoid saturates to 0 or 1
DATA sigmoid_clamp_hi<>+0x00(SB)/4, $0x41a00000  // 20.0
DATA sigmoid_clamp_hi<>+0x04(SB)/4, $0x41a00000
DATA sigmoid_clamp_hi<>+0x08(SB)/4, $0x41a00000
DATA sigmoid_clamp_hi<>+0x0c(SB)/4, $0x41a00000
DATA sigmoid_clamp_hi<>+0x10(SB)/4, $0x41a00000
DATA sigmoid_clamp_hi<>+0x14(SB)/4, $0x41a00000
DATA sigmoid_clamp_hi<>+0x18(SB)/4, $0x41a00000
DATA sigmoid_clamp_hi<>+0x1c(SB)/4, $0x41a00000
GLOBL sigmoid_clamp_hi<>(SB), RODATA|NOPTR, $32

DATA sigmoid_clamp_lo<>+0x00(SB)/4, $0xc1a00000  // -20.0
DATA sigmoid_clamp_lo<>+0x04(SB)/4, $0xc1a00000
DATA sigmoid_clamp_lo<>+0x08(SB)/4, $0xc1a00000
DATA sigmoid_clamp_lo<>+0x0c(SB)/4, $0xc1a00000
DATA sigmoid_clamp_lo<>+0x10(SB)/4, $0xc1a00000
DATA sigmoid_clamp_lo<>+0x14(SB)/4, $0xc1a00000
DATA sigmoid_clamp_lo<>+0x18(SB)/4, $0xc1a00000
DATA sigmoid_clamp_lo<>+0x1c(SB)/4, $0xc1a00000
GLOBL sigmoid_clamp_lo<>(SB), RODATA|NOPTR, $32

// Magic number for float->int conversion: 2^23 + 2^22 = 12582912.0
DATA exp_magic<>+0x00(SB)/4, $0x4b400000
DATA exp_magic<>+0x04(SB)/4, $0x4b400000
DATA exp_magic<>+0x08(SB)/4, $0x4b400000
DATA exp_magic<>+0x0c(SB)/4, $0x4b400000
DATA exp_magic<>+0x10(SB)/4, $0x4b400000
DATA exp_magic<>+0x14(SB)/4, $0x4b400000
DATA exp_magic<>+0x18(SB)/4, $0x4b400000
DATA exp_magic<>+0x1c(SB)/4, $0x4b400000
GLOBL exp_magic<>(SB), RODATA|NOPTR, $32

// 127 << 23 = 1065353216 (exponent bias)
DATA exp_bias<>+0x00(SB)/4, $0x3f800000  // This is 1.0 as integer = 127<<23
DATA exp_bias<>+0x04(SB)/4, $0x3f800000
DATA exp_bias<>+0x08(SB)/4, $0x3f800000
DATA exp_bias<>+0x0c(SB)/4, $0x3f800000
DATA exp_bias<>+0x10(SB)/4, $0x3f800000
DATA exp_bias<>+0x14(SB)/4, $0x3f800000
DATA exp_bias<>+0x18(SB)/4, $0x3f800000
DATA exp_bias<>+0x1c(SB)/4, $0x3f800000
GLOBL exp_bias<>(SB), RODATA|NOPTR, $32

// Exp clamp thresholds: ±88.0 keeps the 2^k reconstruction inside the
// representable float32 range (exp(88) ~= 1.65e38 < MaxFloat32). Matches the
// pure-Go fallback's overflow/underflow clamp.
DATA exp_clamp_hi<>+0x00(SB)/4, $0x42b00000  // 88.0
DATA exp_clamp_hi<>+0x04(SB)/4, $0x42b00000
DATA exp_clamp_hi<>+0x08(SB)/4, $0x42b00000
DATA exp_clamp_hi<>+0x0c(SB)/4, $0x42b00000
DATA exp_clamp_hi<>+0x10(SB)/4, $0x42b00000
DATA exp_clamp_hi<>+0x14(SB)/4, $0x42b00000
DATA exp_clamp_hi<>+0x18(SB)/4, $0x42b00000
DATA exp_clamp_hi<>+0x1c(SB)/4, $0x42b00000
GLOBL exp_clamp_hi<>(SB), RODATA|NOPTR, $32

DATA exp_clamp_lo<>+0x00(SB)/4, $0xc2b00000  // -88.0
DATA exp_clamp_lo<>+0x04(SB)/4, $0xc2b00000
DATA exp_clamp_lo<>+0x08(SB)/4, $0xc2b00000
DATA exp_clamp_lo<>+0x0c(SB)/4, $0xc2b00000
DATA exp_clamp_lo<>+0x10(SB)/4, $0xc2b00000
DATA exp_clamp_lo<>+0x14(SB)/4, $0xc2b00000
DATA exp_clamp_lo<>+0x18(SB)/4, $0xc2b00000
DATA exp_clamp_lo<>+0x1c(SB)/4, $0xc2b00000
GLOBL exp_clamp_lo<>(SB), RODATA|NOPTR, $32

// Constants for exp-based tanh (float32)
DATA tanh32_two<>+0x00(SB)/4, $0x40000000  // 2.0
DATA tanh32_two<>+0x04(SB)/4, $0x40000000
DATA tanh32_two<>+0x08(SB)/4, $0x40000000
DATA tanh32_two<>+0x0c(SB)/4, $0x40000000
DATA tanh32_two<>+0x10(SB)/4, $0x40000000
DATA tanh32_two<>+0x14(SB)/4, $0x40000000
DATA tanh32_two<>+0x18(SB)/4, $0x40000000
DATA tanh32_two<>+0x1c(SB)/4, $0x40000000
GLOBL tanh32_two<>(SB), RODATA|NOPTR, $32

DATA tanh32_clamp_hi<>+0x00(SB)/4, $0x41a00000  // 20.0
DATA tanh32_clamp_hi<>+0x04(SB)/4, $0x41a00000
DATA tanh32_clamp_hi<>+0x08(SB)/4, $0x41a00000
DATA tanh32_clamp_hi<>+0x0c(SB)/4, $0x41a00000
DATA tanh32_clamp_hi<>+0x10(SB)/4, $0x41a00000
DATA tanh32_clamp_hi<>+0x14(SB)/4, $0x41a00000
DATA tanh32_clamp_hi<>+0x18(SB)/4, $0x41a00000
DATA tanh32_clamp_hi<>+0x1c(SB)/4, $0x41a00000
GLOBL tanh32_clamp_hi<>(SB), RODATA|NOPTR, $32

DATA tanh32_clamp_lo<>+0x00(SB)/4, $0xc1a00000  // -20.0
DATA tanh32_clamp_lo<>+0x04(SB)/4, $0xc1a00000
DATA tanh32_clamp_lo<>+0x08(SB)/4, $0xc1a00000
DATA tanh32_clamp_lo<>+0x0c(SB)/4, $0xc1a00000
DATA tanh32_clamp_lo<>+0x10(SB)/4, $0xc1a00000
DATA tanh32_clamp_lo<>+0x14(SB)/4, $0xc1a00000
DATA tanh32_clamp_lo<>+0x18(SB)/4, $0xc1a00000
DATA tanh32_clamp_lo<>+0x1c(SB)/4, $0xc1a00000
GLOBL tanh32_clamp_lo<>(SB), RODATA|NOPTR, $32

// func sigmoidAVX(dst, src []float32)
// Computes accurate sigmoid: σ(x) = 1 / (1 + exp(-x))
// Uses range reduction and polynomial approximation for exp.
TEXT ·sigmoidAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DI
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    // Load all constants into registers
    VMOVUPS exp_log2e<>(SB), Y8         // Y8 = log2(e)
    VMOVUPS exp_ln2<>(SB), Y9           // Y9 = ln(2)
    VMOVUPS exp_one<>(SB), Y10          // Y10 = 1.0
    VMOVUPS exp_c2<>(SB), Y11           // Y11 = c2 = 0.5
    VMOVUPS exp_c3<>(SB), Y12           // Y12 = c3 = 1/6
    VMOVUPS exp_c4<>(SB), Y13           // Y13 = c4 = 1/24
    VMOVUPS exp_c5<>(SB), Y14           // Y14 = c5 = 1/120
    VMOVUPS exp_magic<>(SB), Y15        // Y15 = magic for rounding

    // Process 8 elements per iteration
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   sigmoid32_remainder

sigmoid32_loop8:
    // Load x and compute -x (we need exp(-x))
    VMOVUPS (SI), Y0                    // Y0 = x
    VXORPS Y1, Y1, Y1                   // Y1 = 0
    VSUBPS Y0, Y1, Y0                   // Y0 = -x

    // Clamp -x to [-20, 20] to prevent overflow
    VMOVUPS sigmoid_clamp_hi<>(SB), Y1
    VMOVUPS sigmoid_clamp_lo<>(SB), Y2
    VMINPS Y1, Y0, Y0                   // clamp upper
    VMAXPS Y2, Y0, Y0                   // clamp lower

    // Range reduction: k = round(-x * log2e), r = -x - k * ln2
    // Using magic number rounding: floor(x + magic) - magic
    VMULPS Y8, Y0, Y1                   // Y1 = -x * log2e
    VADDPS Y15, Y1, Y2                  // Y2 = -x * log2e + magic
    VSUBPS Y15, Y2, Y3                  // Y3 = k = round(-x * log2e) as float

    // r = -x - k * ln2
    VMULPS Y9, Y3, Y4                   // Y4 = k * ln2
    VSUBPS Y4, Y0, Y0                   // Y0 = r = -x - k * ln2

    // Polynomial: exp(r) ≈ 1 + r*(1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
    // Using Horner's method
    VMULPS Y0, Y14, Y1                  // Y1 = r * c5
    VADDPS Y13, Y1, Y1                  // Y1 = c4 + r*c5
    VMULPS Y0, Y1, Y1                   // Y1 = r*(c4 + r*c5)
    VADDPS Y12, Y1, Y1                  // Y1 = c3 + r*(c4 + r*c5)
    VMULPS Y0, Y1, Y1                   // Y1 = r*(c3 + ...)
    VADDPS Y11, Y1, Y1                  // Y1 = c2 + r*(c3 + ...)
    VMULPS Y0, Y1, Y1                   // Y1 = r*(c2 + ...)
    VADDPS Y10, Y1, Y1                  // Y1 = 1 + r*(c2 + ...)
    VMULPS Y0, Y1, Y1                   // Y1 = r*(1 + r*(c2 + ...))
    VADDPS Y10, Y1, Y1                  // Y1 = exp(r) ≈ 1 + r*(1 + ...)

    // Reconstruct: exp(-x) = exp(r) * 2^k
    // 2^k is computed by adding k*2^23 to the exponent bits
    // k is already a float, convert to int by reinterpret after adding magic
    VADDPS Y15, Y3, Y2                  // Y2 = k + magic (k is integer in float form)
    VPSLLD $23, Y2, Y2                  // Y2 = k << 23 (shift to exponent position)
    // Note: the magic already includes the bias adjustment via subtraction
    // We need: 2^k where k in range [-29, 29] approximately
    // Actually, we need to convert k to integer properly

    // Simpler approach: use VCVTPS2DQ and then shift
    VCVTPS2DQ Y3, Y4                    // Y4 = int(k)
    VPSLLD $23, Y4, Y4                  // Y4 = k << 23
    VPADDD Y10, Y4, Y4                  // Y4 = 2^k as float (add to 1.0's bits)

    VMULPS Y4, Y1, Y1                   // Y1 = exp(-x) = exp(r) * 2^k

    // Sigmoid: 1 / (1 + exp(-x))
    VADDPS Y10, Y1, Y1                  // Y1 = 1 + exp(-x)
    VDIVPS Y1, Y10, Y0                  // Y0 = 1 / (1 + exp(-x))

    VMOVUPS Y0, (DI)                    // store result

    ADDQ $32, SI
    ADDQ $32, DI
    DECQ AX
    JNZ  sigmoid32_loop8

sigmoid32_remainder:
    ANDQ $7, CX
    JZ   sigmoid32_done

sigmoid32_scalar:
    // Scalar path: load single float
    VMOVSS (SI), X0                     // X0 = x
    VXORPS X1, X1, X1
    VSUBSS X0, X1, X0                   // X0 = -x

    // Clamp
    VMOVSS sigmoid_clamp_hi<>(SB), X1
    VMOVSS sigmoid_clamp_lo<>(SB), X2
    VMINSS X1, X0, X0
    VMAXSS X2, X0, X0

    // Range reduction
    VMOVSS exp_log2e<>(SB), X8
    VMOVSS exp_ln2<>(SB), X9
    VMOVSS exp_magic<>(SB), X15
    VMULSS X8, X0, X1                   // X1 = -x * log2e
    VADDSS X15, X1, X2
    VSUBSS X15, X2, X3                  // X3 = k
    VMULSS X9, X3, X4
    VSUBSS X4, X0, X0                   // X0 = r

    // Polynomial
    VMOVSS exp_one<>(SB), X10
    VMOVSS exp_c2<>(SB), X11
    VMOVSS exp_c3<>(SB), X12
    VMOVSS exp_c4<>(SB), X13
    VMOVSS exp_c5<>(SB), X14

    VMULSS X0, X14, X1
    VADDSS X13, X1, X1
    VMULSS X0, X1, X1
    VADDSS X12, X1, X1
    VMULSS X0, X1, X1
    VADDSS X11, X1, X1
    VMULSS X0, X1, X1
    VADDSS X10, X1, X1
    VMULSS X0, X1, X1
    VADDSS X10, X1, X1                  // X1 = exp(r)

    // Reconstruct 2^k
    VCVTSS2SI X3, AX                    // AX = int(k)
    SHLL $23, AX                        // AX = k << 23
    ADDL $0x3f800000, AX                // AX = 2^k as float bits (add 127<<23 bias)
    VMOVD AX, X4
    VMULSS X4, X1, X1                   // X1 = exp(-x)

    // Sigmoid
    VADDSS X10, X1, X1
    VDIVSS X1, X10, X0
    VMOVSS X0, (DI)

    ADDQ $4, SI
    ADDQ $4, DI
    DECQ CX
    JNZ  sigmoid32_scalar

sigmoid32_done:
    VZEROUPPER
    RET

// func expAVX(dst, src []float32)
// Computes e^x using range reduction and a degree-5 polynomial, the same exp
// core as sigmoidAVX but without the negation and the final 1/(1+exp) wrap.
// Inputs are clamped to [-88, 88] to match the pure-Go fallback: results stay
// finite and large-negative inputs underflow to 0.
TEXT ·expAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DI
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    // Load constants into registers
    VMOVUPS exp_log2e<>(SB), Y8         // Y8 = log2(e)
    VMOVUPS exp_ln2<>(SB), Y9           // Y9 = ln(2)
    VMOVUPS exp_one<>(SB), Y10          // Y10 = 1.0
    VMOVUPS exp_c2<>(SB), Y11           // Y11 = c2 = 0.5
    VMOVUPS exp_c3<>(SB), Y12           // Y12 = c3 = 1/6
    VMOVUPS exp_c4<>(SB), Y13           // Y13 = c4 = 1/24
    VMOVUPS exp_c5<>(SB), Y14           // Y14 = c5 = 1/120
    VMOVUPS exp_magic<>(SB), Y15        // Y15 = magic for rounding

    // Process 8 elements per iteration
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   exp32_remainder

exp32_loop8:
    VMOVUPS (SI), Y0                    // Y0 = x

    // Clamp x to [-88, 88]
    VMOVUPS exp_clamp_hi<>(SB), Y1
    VMOVUPS exp_clamp_lo<>(SB), Y2
    VMINPS Y1, Y0, Y0                   // clamp upper
    VMAXPS Y2, Y0, Y0                   // clamp lower

    // Range reduction: k = round(x * log2e), r = x - k * ln2
    VMULPS Y8, Y0, Y1                   // Y1 = x * log2e
    VADDPS Y15, Y1, Y2                  // Y2 = x*log2e + magic
    VSUBPS Y15, Y2, Y3                  // Y3 = k = round(x * log2e) as float
    VMULPS Y9, Y3, Y4                   // Y4 = k * ln2
    VSUBPS Y4, Y0, Y0                   // Y0 = r = x - k * ln2

    // Polynomial: exp(r) ~= 1 + r*(1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
    VMULPS Y0, Y14, Y1                  // Y1 = r * c5
    VADDPS Y13, Y1, Y1                  // Y1 = c4 + r*c5
    VMULPS Y0, Y1, Y1                   // Y1 = r*(c4 + r*c5)
    VADDPS Y12, Y1, Y1                  // Y1 = c3 + r*(...)
    VMULPS Y0, Y1, Y1                   // Y1 = r*(c3 + ...)
    VADDPS Y11, Y1, Y1                  // Y1 = c2 + r*(...)
    VMULPS Y0, Y1, Y1                   // Y1 = r*(c2 + ...)
    VADDPS Y10, Y1, Y1                  // Y1 = 1 + r*(...)
    VMULPS Y0, Y1, Y1                   // Y1 = r*(1 + ...)
    VADDPS Y10, Y1, Y1                  // Y1 = exp(r)

    // Reconstruct: exp(x) = exp(r) * 2^k
    VCVTPS2DQ Y3, Y4                    // Y4 = int(k)
    VPSLLD $23, Y4, Y4                  // Y4 = k << 23
    VPADDD Y10, Y4, Y4                  // Y4 = 2^k (add to 1.0's bits)
    VMULPS Y4, Y1, Y1                   // Y1 = exp(x)

    VMOVUPS Y1, (DI)                    // store result

    ADDQ $32, SI
    ADDQ $32, DI
    DECQ AX
    JNZ  exp32_loop8

exp32_remainder:
    ANDQ $7, CX
    JZ   exp32_done

    VMOVSS exp_log2e<>(SB), X8
    VMOVSS exp_ln2<>(SB), X9
    VMOVSS exp_magic<>(SB), X15
    VMOVSS exp_one<>(SB), X10
    VMOVSS exp_c2<>(SB), X11
    VMOVSS exp_c3<>(SB), X12
    VMOVSS exp_c4<>(SB), X13
    VMOVSS exp_c5<>(SB), X14
    VMOVSS exp_clamp_hi<>(SB), X6        // clamp bounds in X6/X7: X1/X2 are
    VMOVSS exp_clamp_lo<>(SB), X7        // reused as temporaries in the loop

exp32_scalar:
    VMOVSS (SI), X0                     // X0 = x

    // Clamp
    VMINSS X6, X0, X0
    VMAXSS X7, X0, X0

    // Range reduction
    VMULSS X8, X0, X1                   // X1 = x * log2e
    VADDSS X15, X1, X2
    VSUBSS X15, X2, X3                  // X3 = k
    VMULSS X9, X3, X4
    VSUBSS X4, X0, X0                   // X0 = r

    // Polynomial
    VMULSS X0, X14, X1
    VADDSS X13, X1, X1
    VMULSS X0, X1, X1
    VADDSS X12, X1, X1
    VMULSS X0, X1, X1
    VADDSS X11, X1, X1
    VMULSS X0, X1, X1
    VADDSS X10, X1, X1
    VMULSS X0, X1, X1
    VADDSS X10, X1, X1                  // X1 = exp(r)

    // Reconstruct 2^k
    VCVTSS2SI X3, AX                    // AX = int(k)
    SHLL $23, AX                        // AX = k << 23
    ADDL $0x3f800000, AX                // AX = 2^k bits (add 127<<23 bias)
    VMOVD AX, X4
    VMULSS X4, X1, X1                   // X1 = exp(x)
    VMOVSS X1, (DI)

    ADDQ $4, SI
    ADDQ $4, DI
    DECQ CX
    JNZ  exp32_scalar

exp32_done:
    VZEROUPPER
    RET

// func clampScaleAVX(dst, src []float32, minVal, maxVal, scale float32)
// Performs fused clamp and scale: dst[i] = (clamp(src[i], minVal, maxVal) - minVal) * scale
TEXT ·clampScaleAVX(SB), NOSPLIT, $0-60
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI
    VBROADCASTSS minVal+48(FP), Y1     // Y1 = minVal (broadcast to all lanes)
    VBROADCASTSS maxVal+52(FP), Y2     // Y2 = maxVal
    VBROADCASTSS scale+56(FP), Y3      // Y3 = scale

    // Process 8 elements per iteration
    MOVQ CX, AX
    SHRQ $3, AX                        // len / 8
    JZ   clampscale32_remainder

clampscale32_loop8:
    VMOVUPS (SI), Y0                   // Y0 = src[i]
    VMAXPS Y0, Y1, Y0                  // Y0 = max(src[i], minVal)
    VMINPS Y0, Y2, Y0                  // Y0 = min(max(src[i], minVal), maxVal)
    VSUBPS Y1, Y0, Y0                  // Y0 = clamped - minVal
    VMULPS Y3, Y0, Y0                  // Y0 = (clamped - minVal) * scale
    VMOVUPS Y0, (DX)                   // store result
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  clampscale32_loop8

clampscale32_remainder:
    ANDQ $7, CX                        // remainder = len % 8
    JZ   clampscale32_done
    VMOVSS minVal+48(FP), X1           // X1 = minVal (scalar)
    VMOVSS maxVal+52(FP), X2           // X2 = maxVal
    VMOVSS scale+56(FP), X3            // X3 = scale

clampscale32_scalar:
    VMOVSS (SI), X0                    // X0 = src[i]
    VMAXSS X0, X1, X0                  // X0 = max(src[i], minVal)
    VMINSS X0, X2, X0                  // X0 = min(max(src[i], minVal), maxVal)
    VSUBSS X1, X0, X0                  // X0 = clamped - minVal
    VMULSS X3, X0, X0                  // X0 = (clamped - minVal) * scale
    VMOVSS X0, (DX)                    // store result
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  clampscale32_scalar

clampscale32_done:
    VZEROUPPER
    RET

// func reluAVX(dst, src []float32)
// Computes ReLU: dst[i] = max(0, src[i])
TEXT ·reluAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    // Create zero vector
    VXORPS Y1, Y1, Y1                  // Y1 = 0

    // Process 8 elements per iteration
    MOVQ CX, AX
    SHRQ $3, AX                        // len / 8
    JZ   relu32_remainder

relu32_loop8:
    VMOVUPS (SI), Y0                   // Y0 = src[i]
    VMAXPS Y0, Y1, Y0                  // Y0 = max(src[i], 0)
    VMOVUPS Y0, (DX)                   // store result
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  relu32_loop8

relu32_remainder:
    ANDQ $7, CX                        // remainder = len % 8
    JZ   relu32_done
    VXORPS X1, X1, X1                  // X1 = 0 (scalar)

relu32_scalar:
    VMOVSS (SI), X0                    // X0 = src[i]
    VMAXSS X0, X1, X0                  // X0 = max(src[i], 0)
    VMOVSS X0, (DX)                    // store result
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  relu32_scalar

relu32_done:
    VZEROUPPER
    RET

// func tanhAVX(dst, src []float32)
// Computes accurate tanh: tanh(x) = (1 - e^(-2x)) / (1 + e^(-2x))
// Uses range reduction and polynomial approximation for exp.
TEXT ·tanhAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    // Load constants into Y8-Y15
    VMOVUPS tanh32_two<>(SB), Y8        // Y8 = 2.0
    VMOVUPS exp_log2e<>(SB), Y9         // Y9 = log2(e)
    VMOVUPS exp_ln2<>(SB), Y10          // Y10 = ln(2)
    VMOVUPS exp_one<>(SB), Y11          // Y11 = 1.0
    VMOVUPS exp_c2<>(SB), Y12           // Y12 = c2 = 0.5
    VMOVUPS exp_c3<>(SB), Y13           // Y13 = c3 = 1/6
    VMOVUPS exp_c4<>(SB), Y14           // Y14 = c4 = 1/24
    VMOVUPS exp_c5<>(SB), Y15           // Y15 = c5 = 1/120

    // Process 8 elements per iteration
    MOVQ CX, AX
    SHRQ $3, AX                         // len / 8
    JZ   tanh32_remainder

tanh32_loop8:
    VMOVUPS (SI), Y0                    // Y0 = x

    // Compute z = -2x
    VXORPS Y1, Y1, Y1                   // Y1 = 0
    VSUBPS Y0, Y1, Y0                   // Y0 = -x
    VMULPS Y8, Y0, Y0                   // Y0 = -2x = z

    // Clamp z to [-20, 20]
    VMOVUPS tanh32_clamp_hi<>(SB), Y1   // Y1 = 20.0
    VMOVUPS tanh32_clamp_lo<>(SB), Y2   // Y2 = -20.0
    VMINPS Y1, Y0, Y0                   // Y0 = min(z, 20)
    VMAXPS Y2, Y0, Y0                   // Y0 = max(min(z, 20), -20)

    // Range reduction: k = round(z * log2e), r = z - k * ln2
    VMULPS Y9, Y0, Y1                   // Y1 = z * log2e
    VROUNDPS $0, Y1, Y2                 // Y2 = k = round(Y1) (nearest)
    VMULPS Y10, Y2, Y3                  // Y3 = k * ln2
    VSUBPS Y3, Y0, Y3                   // Y3 = r = z - k * ln2

    // Polynomial: exp(r) ≈ 1 + r*(1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
    // Horner's method
    VMULPS Y3, Y15, Y4                  // Y4 = r * c5
    VADDPS Y14, Y4, Y4                  // Y4 = c4 + r*c5
    VMULPS Y3, Y4, Y4                   // Y4 = r*(c4 + r*c5)
    VADDPS Y13, Y4, Y4                  // Y4 = c3 + r*(...)
    VMULPS Y3, Y4, Y4                   // Y4 = r*(c3 + ...)
    VADDPS Y12, Y4, Y4                  // Y4 = c2 + r*(...)
    VMULPS Y3, Y4, Y4                   // Y4 = r*(c2 + ...)
    VADDPS Y11, Y4, Y4                  // Y4 = 1 + r*(...)
    VMULPS Y3, Y4, Y4                   // Y4 = r*(1 + ...)
    VADDPS Y11, Y4, Y4                  // Y4 = exp(r)

    // Reconstruct exp(z) = exp(r) * 2^k
    // For float32: convert k to int, shift left by 23, add to 1.0's bits
    VCVTTPS2DQ Y2, Y5                   // Y5 = int32(k)
    VPSLLD $23, Y5, Y5                  // Y5 = k << 23
    VPADDD Y11, Y5, Y5                  // Y5 = 2^k (add 1.0's bit pattern)
    VMULPS Y5, Y4, Y4                   // Y4 = exp(z) = exp(-2x)

    // tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
    VSUBPS Y4, Y11, Y5                  // Y5 = 1 - exp(-2x)
    VADDPS Y4, Y11, Y6                  // Y6 = 1 + exp(-2x)
    VDIVPS Y6, Y5, Y0                   // Y0 = tanh(x)

    VMOVUPS Y0, (DX)                    // store result
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  tanh32_loop8

tanh32_remainder:
    ANDQ $7, CX                         // remainder = len % 8
    JZ   tanh32_done

tanh32_scalar:
    // Scalar path for remaining elements
    VMOVSS (SI), X0                     // X0 = x

    // z = -2x
    VXORPS X1, X1, X1
    VSUBSS X0, X1, X0                   // X0 = -x
    VMULSS X8, X0, X0                   // X0 = -2x

    // Clamp to [-20, 20]
    MOVL $0x41a00000, AX                // 20.0
    VMOVD AX, X1
    MOVL $0xc1a00000, AX                // -20.0
    VMOVD AX, X2
    VMINSS X1, X0, X0
    VMAXSS X2, X0, X0

    // Range reduction
    VMULSS X9, X0, X1                   // X1 = z * log2e
    VROUNDSS $0, X1, X1, X2             // X2 = k = round(X1)
    VMULSS X10, X2, X3                  // X3 = k * ln2
    VSUBSS X3, X0, X3                   // X3 = r = z - k * ln2

    // Horner's polynomial
    VMULSS X3, X15, X4                  // X4 = r * c5
    VADDSS X14, X4, X4
    VMULSS X3, X4, X4
    VADDSS X13, X4, X4
    VMULSS X3, X4, X4
    VADDSS X12, X4, X4
    VMULSS X3, X4, X4
    VADDSS X11, X4, X4
    VMULSS X3, X4, X4
    VADDSS X11, X4, X4                  // X4 = exp(r)

    // Reconstruct 2^k
    VCVTTSS2SI X2, AX                   // AX = int32(k)
    SHLL $23, AX                        // AX = k << 23
    MOVL $0x3f800000, BX                // 1.0's bits
    ADDL BX, AX                         // AX = 2^k bits
    VMOVD AX, X5
    VMULSS X5, X4, X4                   // X4 = exp(-2x)

    // tanh = (1 - exp) / (1 + exp)
    VSUBSS X4, X11, X5                  // X5 = 1 - exp
    VADDSS X4, X11, X6                  // X6 = 1 + exp
    VDIVSS X6, X5, X0                   // X0 = tanh

    VMOVSS X0, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  tanh32_scalar

tanh32_done:
    VZEROUPPER
    RET

// func int32ToFloat32ScaleAVX(dst []float32, src []int32, scale float32)
// Converts int32 samples to float32 and multiplies by scale in one pass.
// dst[i] = float32(src[i]) * scale
// Optimized for audio PCM conversion (e.g., scale = 1.0/32768 for 16-bit).
// Frame: dst(24) + src(24) + scale(4) = 52 bytes
TEXT ·int32ToFloat32ScaleAVX(SB), NOSPLIT, $0-52
    MOVQ dst_base+0(FP), DX        // DX = dst pointer
    MOVQ dst_len+8(FP), CX         // CX = length
    MOVQ src_base+24(FP), SI       // SI = src pointer

    // Broadcast scale to all 8 lanes
    VBROADCASTSS scale+48(FP), Y2  // Y2 = {scale, scale, ..., scale}

    // Process 8 int32 elements per iteration
    MOVQ CX, AX
    SHRQ $3, AX                    // len / 8
    JZ   i32tof32_remainder

i32tof32_loop8:
    VMOVDQU (SI), Y0               // Y0 = 8 x int32
    VCVTDQ2PS Y0, Y1               // Y1 = convert int32 to float32
    VMULPS Y2, Y1, Y1              // Y1 = Y1 * scale
    VMOVUPS Y1, (DX)               // store 8 x float32
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  i32tof32_loop8

i32tof32_remainder:
    ANDQ $7, CX                    // remainder = len % 8
    JZ   i32tof32_done

i32tof32_scalar:
    VCVTSI2SSL (SI), X0, X0        // convert single int32 to float32
    VMULSS X2, X0, X0              // X0 = X0 * scale
    VMOVSS X0, (DX)                // store float32
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  i32tof32_scalar

i32tof32_done:
    VZEROUPPER
    RET

// func int16ToFloat32ScaleAVX(dst []float32, src []int16, scale float32)
// Converts int16 samples to float32 and multiplies by scale in one pass.
// dst[i] = float32(src[i]) * scale
// Optimized for 16-bit PCM audio (e.g., scale = 1.0/32768 to normalize to [-1, 1)).
// Requires AVX2 (VPMOVSXWD ymm form to widen 8 int16 to 8 int32).
// Frame: dst(24) + src(24) + scale(4) = 52 bytes
TEXT ·int16ToFloat32ScaleAVX(SB), NOSPLIT, $0-52
    MOVQ dst_base+0(FP), DX        // DX = dst pointer (float32 out)
    MOVQ dst_len+8(FP), CX         // CX = length (len(dst) == len(src))
    MOVQ src_base+24(FP), SI       // SI = src pointer (int16 in)

    // Broadcast scale to all 8 lanes
    VBROADCASTSS scale+48(FP), Y2  // Y2 = {scale, scale, ..., scale}

    // Process 8 int16 elements per iteration
    MOVQ CX, AX
    SHRQ $3, AX                    // len / 8
    JZ   i16tof32_remainder

i16tof32_loop8:
    VPMOVSXWD (SI), Y0             // load 8 int16, sign-extend to 8 x int32
    VCVTDQ2PS Y0, Y1              // convert int32 to float32
    VMULPS Y2, Y1, Y1            // multiply by scale
    VMOVUPS Y1, (DX)            // store 8 x float32
    ADDQ $16, SI               // advance src by 8 x int16 = 16 bytes
    ADDQ $32, DX               // advance dst by 8 x float32 = 32 bytes
    DECQ AX
    JNZ  i16tof32_loop8

i16tof32_remainder:
    ANDQ $7, CX                    // remainder = len % 8
    JZ   i16tof32_done

i16tof32_scalar:
    MOVWLSX (SI), AX               // load int16, sign-extend to 32-bit
    VCVTSI2SSL AX, X0, X0         // convert int32 to float32
    VMULSS X2, X0, X0            // multiply by scale (X2 = scale lane 0)
    VMOVSS X0, (DX)            // store float32
    ADDQ $2, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  i16tof32_scalar

i16tof32_done:
    VZEROUPPER
    RET

// Clamp constants for float32 -> int16 saturation (broadcast per lane).
DATA f32toi16max<>+0(SB)/4, $0x46FFFE00  // 32767.0
GLOBL f32toi16max<>(SB), RODATA|NOPTR, $4
DATA f32toi16min<>+0(SB)/4, $0xC7000000  // -32768.0
GLOBL f32toi16min<>(SB), RODATA|NOPTR, $4

// func float32ToInt16ScaleAVX(dst []int16, src []float32, scale float32)
// Scales float32 samples and converts to int16 PCM in one pass.
// dst[i] = clamp(roundTiesToEven(src[i]*scale), -32768, 32767), NaN -> 0.
// Requires AVX2.
//
// Matches ARM64 FCVTNS+SQXTN bit-for-bit. The hardware VCVTPS2DQ+VPACKSSDW path
// alone would map NaN and +Inf to -32768, so NaN and the saturation bounds are
// handled explicitly before the convert:
//   1. multiply by scale
//   2. self-compare (VCMPPS EQ) and AND: NaN lanes -> 0
//   3. VMINPS 32767.0 / VMAXPS -32768.0: clamp, mapping +Inf -> 32767, -Inf -> -32768
//   4. VCVTPS2DQ: round to nearest-even (in range, so exact)
//   5. VPACKSSDW (via VEXTRACTF128): pack 8 int32 to 8 int16 in order
//
// The 1-7 element tail reprocesses the final aligned block of 8 (an overlapping
// store of identical values); the dispatcher guarantees len >= 8.
//
// Frame: dst(24) + src(24) + scale(4) = 52 bytes
TEXT ·float32ToInt16ScaleAVX(SB), NOSPLIT, $0-52
    MOVQ dst_base+0(FP), DX        // DX = dst pointer (int16 out)
    MOVQ dst_len+8(FP), CX         // CX = length (len(dst) == len(src))
    MOVQ src_base+24(FP), SI       // SI = src pointer (float32 in)

    VBROADCASTSS scale+48(FP), Y3      // Y3 = scale x8
    VBROADCASTSS f32toi16max<>(SB), Y4 // Y4 = 32767.0 x8
    VBROADCASTSS f32toi16min<>(SB), Y5 // Y5 = -32768.0 x8

    MOVQ CX, AX
    SHRQ $3, AX                    // len / 8
    JZ   f32toi16_tail

f32toi16_loop8:
    VMOVUPS (SI), Y0               // 8 x float32
    VMULPS Y3, Y0, Y0            // * scale
    VCMPPS $0, Y0, Y0, Y1       // Y1 = (Y0 == Y0) ? ones : 0  (0 for NaN)
    VANDPS Y1, Y0, Y0          // NaN lanes -> 0
    VMINPS Y4, Y0, Y0         // clamp high (+Inf -> 32767)
    VMAXPS Y5, Y0, Y0        // clamp low (-Inf -> -32768)
    VCVTPS2DQ Y0, Y0         // 8 x int32, round nearest-even
    VEXTRACTF128 $1, Y0, X1  // high 4 int32 -> X1
    VPACKSSDW X1, X0, X0     // pack to 8 x int16 (in order)
    VMOVDQU X0, (DX)         // store 8 x int16 (16 bytes)
    ADDQ $32, SI            // advance src by 8 x float32
    ADDQ $16, DX           // advance dst by 8 x int16
    DECQ AX
    JNZ  f32toi16_loop8

f32toi16_tail:
    ANDQ $7, CX                    // remainder = len % 8
    JZ   f32toi16_done

    // Back up to the final aligned block of 8 and reprocess it (overlap).
    MOVQ $8, BX
    SUBQ CX, BX                    // BX = 8 - (len % 8)  (1..7)
    MOVQ BX, AX
    SHLQ $2, AX                    // (8 - rem) * 4 src bytes
    SUBQ AX, SI
    MOVQ BX, AX
    SHLQ $1, AX                    // (8 - rem) * 2 dst bytes
    SUBQ AX, DX

    VMOVUPS (SI), Y0               // final 8 x float32
    VMULPS Y3, Y0, Y0
    VCMPPS $0, Y0, Y0, Y1
    VANDPS Y1, Y0, Y0
    VMINPS Y4, Y0, Y0
    VMAXPS Y5, Y0, Y0
    VCVTPS2DQ Y0, Y0
    VEXTRACTF128 $1, Y0, X1
    VPACKSSDW X1, X0, X0
    VMOVDQU X0, (DX)               // store final 8 x int16

f32toi16_done:
    VZEROUPPER
    RET

// ============================================================================
// SPLIT-FORMAT COMPLEX OPERATIONS
// ============================================================================
//
// These operate on split real/imag arrays - much simpler than interleaved
// because we can load real and imag values directly without shuffling.

// func mulComplexAVX(dstRe, dstIm, aRe, aIm, bRe, bIm []float32)
// Computes element-wise complex multiplication using split arrays:
//   dstRe[i] = aRe[i]*bRe[i] - aIm[i]*bIm[i]
//   dstIm[i] = aRe[i]*bIm[i] + aIm[i]*bRe[i]
// Uses FMA for efficiency.
// Frame: dstRe(24) + dstIm(24) + aRe(24) + aIm(24) + bRe(24) + bIm(24) = 144 bytes
TEXT ·mulComplexAVX(SB), NOSPLIT, $0-144
    MOVQ dstRe_base+0(FP), DX      // DX = dstRe pointer
    MOVQ dstRe_len+8(FP), CX       // CX = length
    MOVQ dstIm_base+24(FP), R8     // R8 = dstIm pointer
    MOVQ aRe_base+48(FP), SI       // SI = aRe pointer
    MOVQ aIm_base+72(FP), DI       // DI = aIm pointer
    MOVQ bRe_base+96(FP), R9       // R9 = bRe pointer
    MOVQ bIm_base+120(FP), R10     // R10 = bIm pointer

    // Process 8 elements per iteration (AVX 256-bit = 8x float32)
    MOVQ CX, AX
    SHRQ $3, AX                    // len / 8
    JZ   mulcplx_remainder

mulcplx_loop8:
    // Load inputs
    VMOVUPS (SI), Y0               // Y0 = aRe[0:8]
    VMOVUPS (DI), Y1               // Y1 = aIm[0:8]
    VMOVUPS (R9), Y2               // Y2 = bRe[0:8]
    VMOVUPS (R10), Y3              // Y3 = bIm[0:8]

    // dstRe = aRe*bRe - aIm*bIm
    VMULPS Y0, Y2, Y4              // Y4 = aRe * bRe
    VFNMADD231PS Y1, Y3, Y4        // Y4 = Y4 - aIm*bIm = aRe*bRe - aIm*bIm

    // dstIm = aRe*bIm + aIm*bRe
    VMULPS Y0, Y3, Y5              // Y5 = aRe * bIm
    VFMADD231PS Y1, Y2, Y5         // Y5 = Y5 + aIm*bRe = aRe*bIm + aIm*bRe

    // Store results
    VMOVUPS Y4, (DX)               // dstRe[0:8]
    VMOVUPS Y5, (R8)               // dstIm[0:8]

    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, R9
    ADDQ $32, R10
    ADDQ $32, DX
    ADDQ $32, R8
    DECQ AX
    JNZ  mulcplx_loop8

mulcplx_remainder:
    ANDQ $7, CX
    JZ   mulcplx_done

mulcplx_scalar:
    // Load single elements
    VMOVSS (SI), X0                // X0 = aRe
    VMOVSS (DI), X1                // X1 = aIm
    VMOVSS (R9), X2                // X2 = bRe
    VMOVSS (R10), X3               // X3 = bIm

    // dstRe = aRe*bRe - aIm*bIm
    VMULSS X0, X2, X4              // X4 = aRe * bRe
    VFNMADD231SS X1, X3, X4        // X4 = X4 - aIm*bIm

    // dstIm = aRe*bIm + aIm*bRe
    VMULSS X0, X3, X5              // X5 = aRe * bIm
    VFMADD231SS X1, X2, X5         // X5 = X5 + aIm*bRe

    // Store results
    VMOVSS X4, (DX)
    VMOVSS X5, (R8)

    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, R9
    ADDQ $4, R10
    ADDQ $4, DX
    ADDQ $4, R8
    DECQ CX
    JNZ  mulcplx_scalar

mulcplx_done:
    VZEROUPPER
    RET

// func mulConjComplexAVX(dstRe, dstIm, aRe, aIm, bRe, bIm []float32)
// Computes element-wise multiplication by conjugate using split arrays:
//   dstRe[i] = aRe[i]*bRe[i] + aIm[i]*bIm[i]
//   dstIm[i] = aIm[i]*bRe[i] - aRe[i]*bIm[i]
// Frame: dstRe(24) + dstIm(24) + aRe(24) + aIm(24) + bRe(24) + bIm(24) = 144 bytes
TEXT ·mulConjComplexAVX(SB), NOSPLIT, $0-144
    MOVQ dstRe_base+0(FP), DX      // DX = dstRe pointer
    MOVQ dstRe_len+8(FP), CX       // CX = length
    MOVQ dstIm_base+24(FP), R8     // R8 = dstIm pointer
    MOVQ aRe_base+48(FP), SI       // SI = aRe pointer
    MOVQ aIm_base+72(FP), DI       // DI = aIm pointer
    MOVQ bRe_base+96(FP), R9       // R9 = bRe pointer
    MOVQ bIm_base+120(FP), R10     // R10 = bIm pointer

    // Process 8 elements per iteration
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   mulconjcplx_remainder

mulconjcplx_loop8:
    // Load inputs
    VMOVUPS (SI), Y0               // Y0 = aRe[0:8]
    VMOVUPS (DI), Y1               // Y1 = aIm[0:8]
    VMOVUPS (R9), Y2               // Y2 = bRe[0:8]
    VMOVUPS (R10), Y3              // Y3 = bIm[0:8]

    // dstRe = aRe*bRe + aIm*bIm
    VMULPS Y0, Y2, Y4              // Y4 = aRe * bRe
    VFMADD231PS Y1, Y3, Y4         // Y4 = Y4 + aIm*bIm = aRe*bRe + aIm*bIm

    // dstIm = aIm*bRe - aRe*bIm
    VMULPS Y1, Y2, Y5              // Y5 = aIm * bRe
    VFNMADD231PS Y0, Y3, Y5        // Y5 = Y5 - aRe*bIm = aIm*bRe - aRe*bIm

    // Store results
    VMOVUPS Y4, (DX)
    VMOVUPS Y5, (R8)

    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, R9
    ADDQ $32, R10
    ADDQ $32, DX
    ADDQ $32, R8
    DECQ AX
    JNZ  mulconjcplx_loop8

mulconjcplx_remainder:
    ANDQ $7, CX
    JZ   mulconjcplx_done

mulconjcplx_scalar:
    VMOVSS (SI), X0
    VMOVSS (DI), X1
    VMOVSS (R9), X2
    VMOVSS (R10), X3

    // dstRe = aRe*bRe + aIm*bIm
    VMULSS X0, X2, X4
    VFMADD231SS X1, X3, X4

    // dstIm = aIm*bRe - aRe*bIm
    VMULSS X1, X2, X5
    VFNMADD231SS X0, X3, X5

    VMOVSS X4, (DX)
    VMOVSS X5, (R8)

    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, R9
    ADDQ $4, R10
    ADDQ $4, DX
    ADDQ $4, R8
    DECQ CX
    JNZ  mulconjcplx_scalar

mulconjcplx_done:
    VZEROUPPER
    RET

// func absSqComplexAVX(dst, aRe, aIm []float32)
// Computes element-wise magnitude squared using split arrays:
//   dst[i] = aRe[i]^2 + aIm[i]^2
// Frame: dst(24) + aRe(24) + aIm(24) = 72 bytes
TEXT ·absSqComplexAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX        // DX = dst pointer
    MOVQ dst_len+8(FP), CX         // CX = length
    MOVQ aRe_base+24(FP), SI       // SI = aRe pointer
    MOVQ aIm_base+48(FP), DI       // DI = aIm pointer

    // Process 8 elements per iteration
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   abssqcplx_remainder

abssqcplx_loop8:
    // Load inputs
    VMOVUPS (SI), Y0               // Y0 = aRe[0:8]
    VMOVUPS (DI), Y1               // Y1 = aIm[0:8]

    // dst = aRe^2 + aIm^2
    VMULPS Y0, Y0, Y2              // Y2 = aRe^2
    VFMADD231PS Y1, Y1, Y2         // Y2 = Y2 + aIm^2 = aRe^2 + aIm^2

    // Store result
    VMOVUPS Y2, (DX)

    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  abssqcplx_loop8

abssqcplx_remainder:
    ANDQ $7, CX
    JZ   abssqcplx_done

abssqcplx_scalar:
    VMOVSS (SI), X0                // X0 = aRe
    VMOVSS (DI), X1                // X1 = aIm

    // dst = aRe^2 + aIm^2
    VMULSS X0, X0, X2              // X2 = aRe^2
    VFMADD231SS X1, X1, X2         // X2 = X2 + aIm^2

    VMOVSS X2, (DX)

    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  abssqcplx_scalar

abssqcplx_done:
    VZEROUPPER
    RET

// ============================================================================
// BUTTERFLY COMPLEX - FUSED FFT BUTTERFLY WITH TWIDDLE MULTIPLY
// ============================================================================

// func butterflyComplexAVX(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm []float32)
// Performs FFT butterfly with twiddle factor multiply:
//   temp_re = lower_re*tw_re - lower_im*tw_im
//   temp_im = lower_re*tw_im + lower_im*tw_re
//   upper_re, lower_re = upper_re+temp_re, upper_re-temp_re
//   upper_im, lower_im = upper_im+temp_im, upper_im-temp_im
// Frame: 6 slices × 24 bytes = 144 bytes
TEXT ·butterflyComplexAVX(SB), NOSPLIT, $0-144
    MOVQ upperRe_base+0(FP), DX      // DX = upperRe pointer
    MOVQ upperRe_len+8(FP), CX       // CX = length
    MOVQ upperIm_base+24(FP), R8     // R8 = upperIm pointer
    MOVQ lowerRe_base+48(FP), SI     // SI = lowerRe pointer
    MOVQ lowerIm_base+72(FP), DI     // DI = lowerIm pointer
    MOVQ twRe_base+96(FP), R9        // R9 = twRe pointer
    MOVQ twIm_base+120(FP), R10      // R10 = twIm pointer

    // Process 8 elements per iteration
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   butterfly_remainder

butterfly_loop8:
    // Load lower and twiddle
    VMOVUPS (SI), Y0                 // Y0 = lower_re[0:8]
    VMOVUPS (DI), Y1                 // Y1 = lower_im[0:8]
    VMOVUPS (R9), Y2                 // Y2 = tw_re[0:8]
    VMOVUPS (R10), Y3                // Y3 = tw_im[0:8]

    // Complex multiply: temp = lower * twiddle
    // temp_re = lower_re*tw_re - lower_im*tw_im
    // temp_im = lower_re*tw_im + lower_im*tw_re
    VMULPS Y0, Y2, Y4                // Y4 = lower_re * tw_re
    VMULPS Y1, Y3, Y5                // Y5 = lower_im * tw_im
    VSUBPS Y5, Y4, Y4                // Y4 = temp_re = lr*tr - li*ti

    VMULPS Y0, Y3, Y5                // Y5 = lower_re * tw_im
    VFMADD231PS Y1, Y2, Y5           // Y5 = temp_im = lr*ti + li*tr

    // Load upper
    VMOVUPS (DX), Y0                 // Y0 = upper_re[0:8]
    VMOVUPS (R8), Y1                 // Y1 = upper_im[0:8]

    // Butterfly: upper' = upper + temp, lower' = upper - temp
    VADDPS Y0, Y4, Y2                // Y2 = upper_re + temp_re
    VSUBPS Y4, Y0, Y3                // Y3 = upper_re - temp_re
    VMOVUPS Y2, (DX)                 // store upper_re'
    VMOVUPS Y3, (SI)                 // store lower_re'

    VADDPS Y1, Y5, Y2                // Y2 = upper_im + temp_im
    VSUBPS Y5, Y1, Y3                // Y3 = upper_im - temp_im
    VMOVUPS Y2, (R8)                 // store upper_im'
    VMOVUPS Y3, (DI)                 // store lower_im'

    // Advance pointers
    ADDQ $32, DX
    ADDQ $32, R8
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, R9
    ADDQ $32, R10
    DECQ AX
    JNZ  butterfly_loop8

butterfly_remainder:
    ANDQ $7, CX
    JZ   butterfly_done

butterfly_scalar:
    // Load lower and twiddle (scalar)
    VMOVSS (SI), X0                  // X0 = lower_re
    VMOVSS (DI), X1                  // X1 = lower_im
    VMOVSS (R9), X2                  // X2 = tw_re
    VMOVSS (R10), X3                 // X3 = tw_im

    // Complex multiply: temp = lower * twiddle
    VMULSS X0, X2, X4                // X4 = lower_re * tw_re
    VMULSS X1, X3, X5                // X5 = lower_im * tw_im
    VSUBSS X5, X4, X4                // X4 = temp_re

    VMULSS X0, X3, X5                // X5 = lower_re * tw_im
    VFMADD231SS X1, X2, X5           // X5 = temp_im

    // Load upper
    VMOVSS (DX), X0                  // X0 = upper_re
    VMOVSS (R8), X1                  // X1 = upper_im

    // Butterfly
    VADDSS X0, X4, X2                // X2 = upper_re + temp_re
    VSUBSS X4, X0, X3                // X3 = upper_re - temp_re
    VMOVSS X2, (DX)
    VMOVSS X3, (SI)

    VADDSS X1, X5, X2                // X2 = upper_im + temp_im
    VSUBSS X5, X1, X3                // X3 = upper_im - temp_im
    VMOVSS X2, (R8)
    VMOVSS X3, (DI)

    // Advance pointers
    ADDQ $4, DX
    ADDQ $4, R8
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, R9
    ADDQ $4, R10
    DECQ CX
    JNZ  butterfly_scalar

butterfly_done:
    VZEROUPPER
    RET

// ============================================================================
// REAL FFT UNPACK - UNPACKING STEP FOR REAL-VALUED FFT
// ============================================================================

// func realFFTUnpackAVX(outRe, outIm, zRe, zIm, twRe, twIm []float32, n int)
// Performs the unpacking step of real FFT:
//   For k in [1, n-1]:
//     conj_z = conj(Z[n-k])
//     even = 0.5 * (Z[k] + conj_z)
//     diff = Z[k] - conj_z
//     odd = W[k] * (-0.5i) * diff
//     X[k] = even + odd
// Frame: 6 slices × 24 bytes + 1 int × 8 bytes = 152 bytes
TEXT ·realFFTUnpackAVX(SB), NOSPLIT, $0-152
    // Load parameters
    MOVQ outRe_base+0(FP), DX        // DX = outRe pointer
    MOVQ outIm_base+24(FP), SI       // SI = outIm pointer
    MOVQ zRe_base+48(FP), DI         // DI = zRe pointer (forward)
    MOVQ zIm_base+72(FP), R8         // R8 = zIm pointer (forward)
    MOVQ twRe_base+96(FP), R11       // R11 = twRe pointer
    MOVQ twIm_base+120(FP), R12      // R12 = twIm pointer
    MOVQ n+144(FP), CX               // CX = n

    // Calculate number of iterations: (n-1) / 8
    MOVQ CX, AX
    DECQ AX                          // AX = n - 1
    MOVQ AX, R13                     // R13 = n - 1 (save for remainder)
    SHRQ $3, AX                      // AX = (n-1) / 8 = number of SIMD iterations
    JZ   realfft_remainder           // Skip SIMD loop if < 8 elements

    // Set up reverse pointers: R9 = &zRe[n-8], R10 = &zIm[n-8].
    // BX (not R14) holds byte offsets: R14 is the goroutine g pointer on Go amd64.
    MOVQ CX, BX                      // BX = n
    SUBQ $8, BX                      // BX = n - 8
    SHLQ $2, BX                      // BX = (n-8) * 4 = byte offset
    MOVQ DI, R9
    ADDQ BX, R9                      // R9 = &zRe[n-8]
    MOVQ R8, R10
    ADDQ BX, R10                     // R10 = &zIm[n-8]

    // Offset forward pointers to start at index 1
    ADDQ $4, DI                      // DI = &zRe[1]
    ADDQ $4, R8                      // R8 = &zIm[1]
    ADDQ $4, DX                      // DX = &outRe[1]
    ADDQ $4, SI                      // SI = &outIm[1]

    // Load constants. The reverse permutation is performed per iteration in the
    // loop with VPERM2F128 + VPERMILPS, so no permutation mask is needed here.

    // Broadcast 0.5 (0x3F000000 = 0.5f)
    MOVL $0x3F000000, BX
    MOVD BX, X13
    VBROADCASTSS X13, Y13            // Y13 = 0.5 broadcast

    // Sign mask for negation (0x80000000)
    VPCMPEQD Y14, Y14, Y14           // Y14 = all 1s
    VPSLLD $31, Y14, Y14             // Y14 = 0x80000000 (sign bit only)

realfft_loop8:
    // Load forward Z[k:k+8]
    VMOVUPS (DI), Y0                 // Y0 = zRe[k:k+8] (forward)
    VMOVUPS (R8), Y1                 // Y1 = zIm[k:k+8] (forward)

    // Load reverse Z[n-k-7:n-k+1] and reverse the order
    // Memory has: z[n-k-7], z[n-k-6], ..., z[n-k]
    // We need:    z[n-k], z[n-k-1], ..., z[n-k-7]
    VMOVUPS (R9), Y2                 // Y2 = zRe[n-k-7:n-k+1] (to be reversed)
    VMOVUPS (R10), Y3                // Y3 = zIm[n-k-7:n-k+1] (to be reversed)

    // Reverse Y2 and Y3 using VPERM2F128 + VPERMILPS
    // Step 1: Swap 128-bit lanes
    VPERM2F128 $0x01, Y2, Y2, Y2     // Swap high/low 128-bit lanes
    VPERM2F128 $0x01, Y3, Y3, Y3
    // Step 2: Reverse within each 128-bit lane using VPERMILPS
    VPERMILPS $0x1B, Y2, Y2          // 0x1B = 0b00011011 = [3,2,1,0] within each lane
    VPERMILPS $0x1B, Y3, Y3

    // Now Y2 = znkRe (reversed), Y3 = zIm[n-k] (reversed, not yet negated)
    // For conjugate: znkIm = -zIm[n-k]
    VXORPS Y14, Y3, Y3               // Y3 = znkIm = -zIm[n-k] (conjugate)

    // Compute even = 0.5 * (Z[k] + conj(Z[n-k]))
    // evenRe = 0.5 * (zkRe + znkRe)
    // evenIm = 0.5 * (zkIm + znkIm) = 0.5 * (zkIm - zIm[n-k])
    VADDPS Y0, Y2, Y4                // Y4 = zkRe + znkRe
    VMULPS Y4, Y13, Y4               // Y4 = evenRe = 0.5 * (zkRe + znkRe)
    VADDPS Y1, Y3, Y5                // Y5 = zkIm + znkIm (znkIm already negated)
    VMULPS Y5, Y13, Y5               // Y5 = evenIm = 0.5 * (zkIm + znkIm)

    // Compute diff = Z[k] - conj(Z[n-k])
    // diffRe = zkRe - znkRe
    // diffIm = zkIm - znkIm = zkIm - (-zIm[n-k]) = zkIm + zIm[n-k]
    VSUBPS Y2, Y0, Y6                // Y6 = diffRe = zkRe - znkRe
    VSUBPS Y3, Y1, Y7                // Y7 = diffIm = zkIm - znkIm

    // Load twiddles W[k]
    VMOVUPS (R11), Y8                // Y8 = twRe (wr)
    VMOVUPS (R12), Y9                // Y9 = twIm (wi)

    // Compute odd = W[k] * (-0.5i) * diff
    // oddRe = 0.5 * (wr*diffIm + wi*diffRe)
    // oddIm = 0.5 * (wi*diffIm - wr*diffRe)
    VMULPS Y8, Y7, Y10               // Y10 = wr * diffIm
    VFMADD231PS Y9, Y6, Y10          // Y10 = wr*diffIm + wi*diffRe
    VMULPS Y10, Y13, Y10             // Y10 = oddRe = 0.5 * (wr*diffIm + wi*diffRe)

    VMULPS Y9, Y7, Y11               // Y11 = wi * diffIm
    VFNMADD231PS Y8, Y6, Y11         // Y11 = wi*diffIm - wr*diffRe
    VMULPS Y11, Y13, Y11             // Y11 = oddIm = 0.5 * (wi*diffIm - wr*diffRe)

    // Compute output X[k] = even + odd
    VADDPS Y4, Y10, Y0               // Y0 = outRe = evenRe + oddRe
    VADDPS Y5, Y11, Y1               // Y1 = outIm = evenIm + oddIm

    // Store results
    VMOVUPS Y0, (DX)                 // store outRe[k:k+8]
    VMOVUPS Y1, (SI)                 // store outIm[k:k+8]

    // Advance pointers
    ADDQ $32, DI                     // forward zRe += 8
    ADDQ $32, R8                     // forward zIm += 8
    SUBQ $32, R9                     // reverse zRe -= 8
    SUBQ $32, R10                    // reverse zIm -= 8
    ADDQ $32, R11                    // twRe += 8
    ADDQ $32, R12                    // twIm += 8
    ADDQ $32, DX                     // outRe += 8
    ADDQ $32, SI                     // outIm += 8

    DECQ AX
    JNZ  realfft_loop8

realfft_remainder:
    // Handle remaining elements (n-1) % 8
    ANDQ $7, R13                     // R13 = remainder count
    JZ   realfft_done

    // Reload base pointers for remainder (need to recalculate positions)
    MOVQ outRe_base+0(FP), DX
    MOVQ outIm_base+24(FP), SI
    MOVQ zRe_base+48(FP), DI
    MOVQ zIm_base+72(FP), R8
    MOVQ twRe_base+96(FP), R11
    MOVQ twIm_base+120(FP), R12
    MOVQ n+144(FP), CX

    // Calculate starting k for remainder: 1 + 8 * num_full_iterations
    MOVQ CX, AX
    DECQ AX                          // AX = n - 1
    SHRQ $3, AX                      // AX = num_full_iterations
    SHLQ $3, AX                      // AX = 8 * num_full_iterations
    INCQ AX                          // AX = 1 + 8 * num_full_iterations = starting k

    // Offset pointers to starting k
    MOVQ AX, BX
    SHLQ $2, BX                      // BX = k * 4 bytes
    ADDQ BX, DX                      // DX = &outRe[k]
    ADDQ BX, SI                      // SI = &outIm[k]
    ADDQ BX, DI                      // DI = &zRe[k]
    ADDQ BX, R8                      // R8 = &zIm[k]

    // Twiddle offset is (k-1)
    DECQ AX
    MOVQ AX, BX
    SHLQ $2, BX
    ADDQ BX, R11                     // R11 = &twRe[k-1]
    ADDQ BX, R12                     // R12 = &twIm[k-1]
    INCQ AX                          // Restore AX = k

realfft_scalar:
    // Calculate mirror index: nk = n - k
    MOVQ CX, BX
    SUBQ AX, BX                      // BX = n - k = nk

    // Load Z[k]
    VMOVSS (DI), X0                  // X0 = zRe[k]
    VMOVSS (R8), X1                  // X1 = zIm[k]

    // Load conj(Z[n-k])
    MOVQ zRe_base+48(FP), R15
    MOVQ BX, R9
    SHLQ $2, R9
    ADDQ R9, R15
    VMOVSS (R15), X2                 // X2 = zRe[nk]

    MOVQ zIm_base+72(FP), R15
    ADDQ R9, R15
    VMOVSS (R15), X3                 // X3 = zIm[nk]

    // Load 0.5 constant (0x3F000000 = 0.5f)
    MOVL $0x3F000000, BX
    MOVD BX, X13

    // Negate X3 for conjugate: znkIm = -zIm[nk]
    VXORPS X14, X14, X14
    VSUBSS X3, X14, X3               // X3 = -zIm[nk] = znkIm

    // evenRe = 0.5 * (zkRe + znkRe)
    VADDSS X0, X2, X4
    VMULSS X4, X13, X4               // X4 = evenRe

    // evenIm = 0.5 * (zkIm + znkIm)
    VADDSS X1, X3, X5
    VMULSS X5, X13, X5               // X5 = evenIm

    // diffRe = zkRe - znkRe
    VSUBSS X2, X0, X6                // X6 = diffRe

    // diffIm = zkIm - znkIm
    VSUBSS X3, X1, X7                // X7 = diffIm

    // Load twiddles
    VMOVSS (R11), X8                 // X8 = wr
    VMOVSS (R12), X9                 // X9 = wi

    // oddRe = 0.5 * (wr*diffIm + wi*diffRe)
    VMULSS X8, X7, X10               // X10 = wr * diffIm
    VFMADD231SS X9, X6, X10          // X10 = wr*diffIm + wi*diffRe
    VMULSS X10, X13, X10             // X10 = oddRe

    // oddIm = 0.5 * (wi*diffIm - wr*diffRe)
    VMULSS X9, X7, X11               // X11 = wi * diffIm
    VFNMADD231SS X8, X6, X11         // X11 = wi*diffIm - wr*diffRe
    VMULSS X11, X13, X11             // X11 = oddIm

    // output = even + odd
    VADDSS X4, X10, X0               // X0 = outRe
    VADDSS X5, X11, X1               // X1 = outIm

    // Store
    VMOVSS X0, (DX)
    VMOVSS X1, (SI)

    // Advance pointers
    ADDQ $4, DI
    ADDQ $4, R8
    ADDQ $4, R11
    ADDQ $4, R12
    ADDQ $4, DX
    ADDQ $4, SI
    INCQ AX                          // k++

    DECQ R13
    JNZ  realfft_scalar

realfft_done:
    VZEROUPPER
    RET

// ============================================================================
// REVERSE - REVERSE SLICE ELEMENTS
// ============================================================================

// func reverseAVX(dst, src []float32)
// Reverses elements: dst[i] = src[len-1-i]
// Frame: dst(24) + src(24) = 48 bytes
TEXT ·reverseAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX        // DX = dst pointer
    MOVQ dst_len+8(FP), CX         // CX = length
    MOVQ src_base+24(FP), SI       // SI = src pointer

    // Check for in-place reversal (not supported in SIMD path)
    CMPQ DX, SI
    JE   reverse_inplace

    // Calculate src end pointer: SI + (n-8)*4 (points to last full block)
    MOVQ CX, AX
    SUBQ $8, AX                    // AX = n - 8
    SHLQ $2, AX                    // AX = (n-8) * 4
    ADDQ SI, AX                    // AX = &src[n-8]
    MOVQ AX, R8                    // R8 = reverse src pointer

    // Process 8 elements per iteration (from end of src to beginning of dst)
    MOVQ CX, AX
    SHRQ $3, AX                    // AX = n / 8
    JZ   reverse_remainder

reverse_loop8:
    // Load 8 elements from reverse position
    VMOVUPS (R8), Y0               // Y0 = src[n-8:n]

    // Reverse order: swap 128-bit lanes then reverse within lanes
    VPERM2F128 $0x01, Y0, Y0, Y0   // Swap high/low 128-bit lanes
    VPERMILPS $0x1B, Y0, Y0        // Reverse within each lane: [3,2,1,0]

    // Store to forward position
    VMOVUPS Y0, (DX)

    ADDQ $32, DX                   // dst += 8
    SUBQ $32, R8                   // src_rev -= 8
    DECQ AX
    JNZ  reverse_loop8

reverse_remainder:
    ANDQ $7, CX
    JZ   reverse_done

    // Handle remaining elements
    // Calculate remaining src position
    MOVQ dst_len+8(FP), AX         // AX = original length
    SHRQ $3, AX
    SHLQ $3, AX                    // AX = processed count
    MOVQ dst_len+8(FP), R9         // R9 = n
    SUBQ AX, R9                    // R9 = remaining count
    DECQ R9                        // R9 = n - processed - 1

reverse_scalar:
    // Get src[n-1-i] where i is current dst index
    MOVQ dst_len+8(FP), R10
    SHRQ $3, R10
    SHLQ $3, R10                   // R10 = processed count
    ADDQ R10, R9                   // Adjust for processed
    MOVQ dst_len+8(FP), R10
    DECQ R10                       // R10 = n - 1
    SUBQ R9, R10                   // R10 = n - 1 - (processed + remaining_idx)

    // Actually, let's just do a simple scalar loop
    MOVQ src_base+24(FP), SI
    MOVQ dst_len+8(FP), AX
    SHRQ $3, AX
    SHLQ $3, AX                    // AX = starting dst index
    MOVQ dst_len+8(FP), R10
    DECQ R10                       // R10 = n - 1

reverse_scalar_loop:
    MOVQ R10, R11
    SUBQ AX, R11                   // R11 = n - 1 - i (src index)
    SHLQ $2, R11
    ADDQ SI, R11                   // R11 = &src[n-1-i]
    VMOVSS (R11), X0
    VMOVSS X0, (DX)

    ADDQ $4, DX
    INCQ AX
    DECQ CX
    JNZ  reverse_scalar_loop
    JMP  reverse_done

reverse_inplace:
    // In-place reversal: swap from both ends toward middle
    // SI = start, calculate end pointer
    MOVQ CX, AX
    DECQ AX
    SHLQ $2, AX
    ADDQ SI, AX                    // AX = &src[n-1]

    SHRQ $1, CX                    // CX = n / 2 swaps needed
    JZ   reverse_done

reverse_inplace_loop:
    VMOVSS (SI), X0                // X0 = front element
    VMOVSS (AX), X1                // X1 = back element
    VMOVSS X1, (SI)                // store back to front
    VMOVSS X0, (AX)                // store front to back
    ADDQ $4, SI
    SUBQ $4, AX
    DECQ CX
    JNZ  reverse_inplace_loop

reverse_done:
    VZEROUPPER
    RET

// ============================================================================
// ADD-SUB - FUSED SUM AND DIFFERENCE
// ============================================================================

// func addSubAVX(sumDst, diffDst, a, b []float32)
// Computes element-wise sum and difference:
//   sumDst[i] = a[i] + b[i]
//   diffDst[i] = a[i] - b[i]
// Frame: sumDst(24) + diffDst(24) + a(24) + b(24) = 96 bytes
TEXT ·addSubAVX(SB), NOSPLIT, $0-96
    MOVQ sumDst_base+0(FP), DX     // DX = sumDst pointer
    MOVQ sumDst_len+8(FP), CX      // CX = length
    MOVQ diffDst_base+24(FP), R8   // R8 = diffDst pointer
    MOVQ a_base+48(FP), SI         // SI = a pointer
    MOVQ b_base+72(FP), DI         // DI = b pointer

    // Process 8 elements per iteration
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   addsub_remainder

addsub_loop8:
    // Load inputs
    VMOVUPS (SI), Y0               // Y0 = a[0:8]
    VMOVUPS (DI), Y1               // Y1 = b[0:8]

    // Compute sum and diff
    VADDPS Y0, Y1, Y2              // Y2 = a + b (sum)
    VSUBPS Y1, Y0, Y3              // Y3 = a - b (diff)

    // Store results
    VMOVUPS Y2, (DX)               // sumDst[0:8]
    VMOVUPS Y3, (R8)               // diffDst[0:8]

    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    ADDQ $32, R8
    DECQ AX
    JNZ  addsub_loop8

addsub_remainder:
    ANDQ $7, CX
    JZ   addsub_done

addsub_scalar:
    VMOVSS (SI), X0                // X0 = a
    VMOVSS (DI), X1                // X1 = b
    VADDSS X0, X1, X2              // X2 = a + b
    VSUBSS X1, X0, X3              // X3 = a - b
    VMOVSS X2, (DX)
    VMOVSS X3, (R8)

    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    ADDQ $4, R8
    DECQ CX
    JNZ  addsub_scalar

addsub_done:
    VZEROUPPER
    RET

// func convolveDecimateAVX(dst, signal, kernel []float32, factor, phase int)
//
// Fused decimating valid convolution: for each output k it computes the dot
// product of signal[pos:pos+kLen] with kernel, then advances pos by factor.
// The inner dot replicates dotProductAVX exactly (4 accumulators, 32/8/scalar
// reduction) so results are bit-identical to a per-window DotProductUnsafe.
// Outer state lives in R8-R13 plus BX (pos); the inner dot uses SI/DI/CX/AX and
// Y0-Y5. BX (not R14) holds pos: R14 is the goroutine g pointer in Go's amd64
// ABI and must not be clobbered.
TEXT ·convolveDecimateAVX(SB), NOSPLIT, $0-88
    MOVQ dst_base+0(FP), R8        // output pointer
    MOVQ dst_len+8(FP), R9         // n outputs
    MOVQ signal_base+24(FP), R10   // signal base
    MOVQ kernel_base+48(FP), R11   // kernel base
    MOVQ kernel_len+56(FP), R12    // kLen
    MOVQ factor+72(FP), R13        // factor (elements)
    MOVQ phase+80(FP), BX         // pos (elements)

    TESTQ R9, R9
    JZ    cd_avx_ret

cd_avx_outer:
    LEAQ (R10)(BX*4), SI          // SI = &signal[pos]
    MOVQ R11, DI                   // DI = &kernel[0]

    VXORPS Y0, Y0, Y0
    VXORPS Y3, Y3, Y3
    VXORPS Y4, Y4, Y4
    VXORPS Y5, Y5, Y5

    MOVQ R12, CX                   // CX = kLen
    MOVQ CX, AX
    SHRQ $5, AX                    // kLen / 32
    JZ   cd_avx_loop8_check

cd_avx_loop32:
    VMOVUPS (SI), Y1
    VMOVUPS (DI), Y2
    VFMADD231PS Y1, Y2, Y0
    VMOVUPS 32(SI), Y1
    VMOVUPS 32(DI), Y2
    VFMADD231PS Y1, Y2, Y3
    VMOVUPS 64(SI), Y1
    VMOVUPS 64(DI), Y2
    VFMADD231PS Y1, Y2, Y4
    VMOVUPS 96(SI), Y1
    VMOVUPS 96(DI), Y2
    VFMADD231PS Y1, Y2, Y5
    ADDQ $128, SI
    ADDQ $128, DI
    DECQ AX
    JNZ  cd_avx_loop32

    VADDPS Y3, Y0, Y0
    VADDPS Y4, Y0, Y0
    VADDPS Y5, Y0, Y0

cd_avx_loop8_check:
    MOVQ R12, CX
    ANDQ $31, CX
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   cd_avx_reduce

cd_avx_loop8:
    VMOVUPS (SI), Y1
    VMOVUPS (DI), Y2
    VFMADD231PS Y1, Y2, Y0
    ADDQ $32, SI
    ADDQ $32, DI
    DECQ AX
    JNZ  cd_avx_loop8

cd_avx_reduce:
    // Reduce Y0 to X0 before scalar ops (VEX scalar ops zero upper YMM).
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    MOVQ R12, CX
    ANDQ $7, CX
    JZ   cd_avx_store

cd_avx_scalar:
    VMOVSS (SI), X1
    VMOVSS (DI), X2
    VFMADD231SS X1, X2, X0
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ CX
    JNZ  cd_avx_scalar

cd_avx_store:
    VMOVSS X0, (R8)
    ADDQ $4, R8
    ADDQ R13, BX                  // pos += factor
    DECQ R9
    JNZ  cd_avx_outer

cd_avx_ret:
    VZEROUPPER
    RET

// func convolveValidMaxAbsAVX(signal, kernel []float32) float32
//
// Fused valid convolution + abs-max: returns max_i |dot(signal[i:i+kLen], kernel)|
// over the n = len(signal)-kLen+1 windows without materializing the output. The
// inner dot replicates convolveDecimateAVX / dotProductAVX (4 Y accumulators,
// 32/8/scalar reduction), so each window is bit-identical to ConvolveValid; the
// per-window store is replaced by an abs (VANDPS) into a running max (X7).
// Caller guarantees kLen >= 1 and len(signal) >= kLen, so n >= 1.
TEXT ·convolveValidMaxAbsAVX(SB), NOSPLIT, $0-52
    MOVQ signal_base+0(FP), R10    // R10 = &signal[pos], advances 4 bytes/output
    MOVQ signal_len+8(FP), R9
    MOVQ kernel_base+24(FP), R11
    MOVQ kernel_len+32(FP), R12    // R12 = kLen

    SUBQ R12, R9
    INCQ R9                        // R9 = n

    VMOVUPS absf32mask<>(SB), X6   // abs mask
    VXORPS X7, X7, X7              // running max = 0

cvma_avx_outer:
    MOVQ R10, SI
    MOVQ R11, DI

    VXORPS Y0, Y0, Y0
    VXORPS Y3, Y3, Y3
    VXORPS Y4, Y4, Y4
    VXORPS Y5, Y5, Y5

    MOVQ R12, CX
    MOVQ CX, AX
    SHRQ $5, AX                    // kLen / 32
    JZ   cvma_avx_loop8_check

cvma_avx_loop32:
    VMOVUPS (SI), Y1
    VMOVUPS (DI), Y2
    VFMADD231PS Y1, Y2, Y0
    VMOVUPS 32(SI), Y1
    VMOVUPS 32(DI), Y2
    VFMADD231PS Y1, Y2, Y3
    VMOVUPS 64(SI), Y1
    VMOVUPS 64(DI), Y2
    VFMADD231PS Y1, Y2, Y4
    VMOVUPS 96(SI), Y1
    VMOVUPS 96(DI), Y2
    VFMADD231PS Y1, Y2, Y5
    ADDQ $128, SI
    ADDQ $128, DI
    DECQ AX
    JNZ  cvma_avx_loop32

    VADDPS Y3, Y0, Y0
    VADDPS Y4, Y0, Y0
    VADDPS Y5, Y0, Y0

cvma_avx_loop8_check:
    MOVQ R12, CX
    ANDQ $31, CX
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   cvma_avx_reduce

cvma_avx_loop8:
    VMOVUPS (SI), Y1
    VMOVUPS (DI), Y2
    VFMADD231PS Y1, Y2, Y0
    ADDQ $32, SI
    ADDQ $32, DI
    DECQ AX
    JNZ  cvma_avx_loop8

cvma_avx_reduce:
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    MOVQ R12, CX
    ANDQ $7, CX
    JZ   cvma_avx_absmax

cvma_avx_scalar:
    VMOVSS (SI), X1
    VMOVSS (DI), X2
    VFMADD231SS X1, X2, X0
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ CX
    JNZ  cvma_avx_scalar

cvma_avx_absmax:
    VANDPS X0, X6, X0             // |dot|
    VMAXSS X0, X7, X7            // running max = max(running, |dot|)
    ADDQ $4, R10                  // pos += 1
    DECQ R9
    JNZ  cvma_avx_outer

    VMOVSS X7, ret+48(FP)
    VZEROUPPER
    RET

// func convolveDecimateAVX512(dst, signal, kernel []float32, factor, phase int)
//
// AVX-512 fused decimating valid convolution. Inner dot replicates
// dotProductAVX512 (4 Z accumulators, 64/16/scalar reduction) for bit-identical
// results vs a per-window DotProductUnsafe on the AVX-512 path.
TEXT ·convolveDecimateAVX512(SB), NOSPLIT, $0-88
    MOVQ dst_base+0(FP), R8
    MOVQ dst_len+8(FP), R9
    MOVQ signal_base+24(FP), R10
    MOVQ kernel_base+48(FP), R11
    MOVQ kernel_len+56(FP), R12
    MOVQ factor+72(FP), R13
    MOVQ phase+80(FP), BX

    TESTQ R9, R9
    JZ    cd_avx512_ret

cd_avx512_outer:
    LEAQ (R10)(BX*4), SI
    MOVQ R11, DI

    VPXORD Z0, Z0, Z0
    VPXORD Z3, Z3, Z3
    VPXORD Z4, Z4, Z4
    VPXORD Z5, Z5, Z5

    MOVQ R12, CX
    MOVQ CX, AX
    SHRQ $6, AX                    // kLen / 64
    JZ   cd_avx512_loop16_check

cd_avx512_loop64:
    VMOVUPS (SI), Z1
    VMOVUPS (DI), Z2
    VFMADD231PS Z1, Z2, Z0
    VMOVUPS 64(SI), Z1
    VMOVUPS 64(DI), Z2
    VFMADD231PS Z1, Z2, Z3
    VMOVUPS 128(SI), Z1
    VMOVUPS 128(DI), Z2
    VFMADD231PS Z1, Z2, Z4
    VMOVUPS 192(SI), Z1
    VMOVUPS 192(DI), Z2
    VFMADD231PS Z1, Z2, Z5
    ADDQ $256, SI
    ADDQ $256, DI
    DECQ AX
    JNZ  cd_avx512_loop64

    VADDPS Z3, Z0, Z0
    VADDPS Z4, Z0, Z0
    VADDPS Z5, Z0, Z0

cd_avx512_loop16_check:
    MOVQ R12, CX
    ANDQ $63, CX
    MOVQ CX, AX
    SHRQ $4, AX
    JZ   cd_avx512_reduce

cd_avx512_loop16:
    VMOVUPS (SI), Z1
    VMOVUPS (DI), Z2
    VFMADD231PS Z1, Z2, Z0
    ADDQ $64, SI
    ADDQ $64, DI
    DECQ AX
    JNZ  cd_avx512_loop16

cd_avx512_reduce:
    // VEXTRACTF64X4 (AVX512F): same upper-256 extract as VEXTRACTF32X8, no AVX512DQ dep.
    VEXTRACTF64X4 $1, Z0, Y1
    VADDPS Y1, Y0, Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    MOVQ R12, CX
    ANDQ $15, CX
    JZ   cd_avx512_store

cd_avx512_scalar:
    VMOVSS (SI), X1
    VMOVSS (DI), X2
    VFMADD231SS X1, X2, X0
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ CX
    JNZ  cd_avx512_scalar

cd_avx512_store:
    VMOVSS X0, (R8)
    ADDQ $4, R8
    ADDQ R13, BX
    DECQ R9
    JNZ  cd_avx512_outer

cd_avx512_ret:
    VZEROUPPER
    RET

// func convolveDecimateSSE(dst, signal, kernel []float32, factor, phase int)
//
// SSE2 fused decimating valid convolution. Inner dot replicates dotProductSSE
// (single accumulator, 4-wide loop, SHUFPS horizontal sum) for bit-identical
// results vs a per-window DotProductUnsafe on the SSE path.
TEXT ·convolveDecimateSSE(SB), NOSPLIT, $0-88
    MOVQ dst_base+0(FP), R8
    MOVQ dst_len+8(FP), R9
    MOVQ signal_base+24(FP), R10
    MOVQ kernel_base+48(FP), R11
    MOVQ kernel_len+56(FP), R12
    MOVQ factor+72(FP), R13
    MOVQ phase+80(FP), BX

    TESTQ R9, R9
    JZ    cd_sse_ret

cd_sse_outer:
    LEAQ (R10)(BX*4), SI
    MOVQ R11, DI
    XORPS X0, X0

    MOVQ R12, CX
    MOVQ CX, AX
    SHRQ $2, AX                    // kLen / 4
    JZ   cd_sse_reduce

cd_sse_loop4:
    MOVUPS (SI), X1
    MOVUPS (DI), X2
    MULPS X2, X1
    ADDPS X1, X0
    ADDQ $16, SI
    ADDQ $16, DI
    DECQ AX
    JNZ  cd_sse_loop4

cd_sse_reduce:
    MOVAPS X0, X1
    SHUFPS $0x0E, X1, X1
    ADDPS X1, X0
    MOVAPS X0, X1
    SHUFPS $0x01, X1, X1
    ADDSS X1, X0

    MOVQ R12, CX
    ANDQ $3, CX
    JZ   cd_sse_store

cd_sse_scalar:
    MOVSS (SI), X1
    MOVSS (DI), X2
    MULSS X2, X1
    ADDSS X1, X0
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ CX
    JNZ  cd_sse_scalar

cd_sse_store:
    MOVSS X0, (R8)
    ADDQ $4, R8
    ADDQ R13, BX
    DECQ R9
    JNZ  cd_sse_outer

cd_sse_ret:
    RET

// func convolveValidMaxAbsSSE(signal, kernel []float32) float32
//
// SSE fused valid convolution + abs-max. Inner dot replicates convolveDecimateSSE
// / dotProductSSE (single accumulator, MULPS+ADDPS, 4/scalar reduction); the
// per-window store is replaced by an abs (ANDPS) into a running max (X7). Caller
// guarantees kLen >= 1 and len(signal) >= kLen, so n >= 1.
TEXT ·convolveValidMaxAbsSSE(SB), NOSPLIT, $0-52
    MOVQ signal_base+0(FP), R10
    MOVQ signal_len+8(FP), R9
    MOVQ kernel_base+24(FP), R11
    MOVQ kernel_len+32(FP), R12

    SUBQ R12, R9
    INCQ R9                        // n

    MOVUPS absf32mask<>(SB), X6
    XORPS X7, X7                   // running max = 0

cvma_sse_outer:
    MOVQ R10, SI
    MOVQ R11, DI
    XORPS X0, X0

    MOVQ R12, CX
    MOVQ CX, AX
    SHRQ $2, AX                    // kLen / 4
    JZ   cvma_sse_reduce

cvma_sse_loop4:
    MOVUPS (SI), X1
    MOVUPS (DI), X2
    MULPS X2, X1
    ADDPS X1, X0
    ADDQ $16, SI
    ADDQ $16, DI
    DECQ AX
    JNZ  cvma_sse_loop4

cvma_sse_reduce:
    MOVAPS X0, X1
    SHUFPS $0x0E, X1, X1
    ADDPS X1, X0
    MOVAPS X0, X1
    SHUFPS $0x01, X1, X1
    ADDSS X1, X0

    MOVQ R12, CX
    ANDQ $3, CX
    JZ   cvma_sse_absmax

cvma_sse_scalar:
    MOVSS (SI), X1
    MOVSS (DI), X2
    MULSS X2, X1
    ADDSS X1, X0
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ CX
    JNZ  cvma_sse_scalar

cvma_sse_absmax:
    ANDPS X6, X0                  // |dot|
    MAXSS X0, X7                 // running max = max(X7, X0)
    ADDQ $4, R10
    DECQ R9
    JNZ  cvma_sse_outer

    MOVSS X7, ret+48(FP)
    RET

// func dotProduct4AVX(results, row0, row1, row2, row3, vec *float32, n int)
// Computes four dot products against the same vec, reusing each vec load.
TEXT ·dotProduct4AVX(SB), NOSPLIT, $0-56
    MOVQ results+0(FP), DX
    MOVQ row0+8(FP), SI
    MOVQ row1+16(FP), R8
    MOVQ row2+24(FP), R9
    MOVQ row3+32(FP), R10
    MOVQ vec+40(FP), DI
    MOVQ n+48(FP), CX

    VXORPS Y0, Y0, Y0          // acc0a
    VXORPS Y3, Y3, Y3          // acc1a
    VXORPS Y4, Y4, Y4          // acc2a
    VXORPS Y5, Y5, Y5          // acc3a
    VXORPS Y6, Y6, Y6          // acc0b
    VXORPS Y7, Y7, Y7          // acc1b
    VXORPS Y8, Y8, Y8          // acc2b
    VXORPS Y9, Y9, Y9          // acc3b

    MOVQ CX, AX
    SHRQ $5, AX                // n / 32
    JZ   dot4_avx_loop8_check

dot4_avx_loop32:
    VMOVUPS (DI), Y1
    VMOVUPS (SI), Y2
    VFMADD231PS Y1, Y2, Y0
    VMOVUPS (R8), Y2
    VFMADD231PS Y1, Y2, Y3
    VMOVUPS (R9), Y2
    VFMADD231PS Y1, Y2, Y4
    VMOVUPS (R10), Y2
    VFMADD231PS Y1, Y2, Y5

    VMOVUPS 32(DI), Y1
    VMOVUPS 32(SI), Y2
    VFMADD231PS Y1, Y2, Y6
    VMOVUPS 32(R8), Y2
    VFMADD231PS Y1, Y2, Y7
    VMOVUPS 32(R9), Y2
    VFMADD231PS Y1, Y2, Y8
    VMOVUPS 32(R10), Y2
    VFMADD231PS Y1, Y2, Y9

    VMOVUPS 64(DI), Y1
    VMOVUPS 64(SI), Y2
    VFMADD231PS Y1, Y2, Y0
    VMOVUPS 64(R8), Y2
    VFMADD231PS Y1, Y2, Y3
    VMOVUPS 64(R9), Y2
    VFMADD231PS Y1, Y2, Y4
    VMOVUPS 64(R10), Y2
    VFMADD231PS Y1, Y2, Y5

    VMOVUPS 96(DI), Y1
    VMOVUPS 96(SI), Y2
    VFMADD231PS Y1, Y2, Y6
    VMOVUPS 96(R8), Y2
    VFMADD231PS Y1, Y2, Y7
    VMOVUPS 96(R9), Y2
    VFMADD231PS Y1, Y2, Y8
    VMOVUPS 96(R10), Y2
    VFMADD231PS Y1, Y2, Y9

    ADDQ $128, DI
    ADDQ $128, SI
    ADDQ $128, R8
    ADDQ $128, R9
    ADDQ $128, R10
    DECQ AX
    JNZ  dot4_avx_loop32

dot4_avx_loop8_check:
    ANDQ $31, CX
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   dot4_avx_reduce

dot4_avx_loop8:
    VMOVUPS (DI), Y1
    VMOVUPS (SI), Y2
    VFMADD231PS Y1, Y2, Y0
    VMOVUPS (R8), Y2
    VFMADD231PS Y1, Y2, Y3
    VMOVUPS (R9), Y2
    VFMADD231PS Y1, Y2, Y4
    VMOVUPS (R10), Y2
    VFMADD231PS Y1, Y2, Y5
    ADDQ $32, DI
    ADDQ $32, SI
    ADDQ $32, R8
    ADDQ $32, R9
    ADDQ $32, R10
    DECQ AX
    JNZ  dot4_avx_loop8

dot4_avx_reduce:
    VADDPS Y6, Y0, Y0
    VADDPS Y7, Y3, Y3
    VADDPS Y8, Y4, Y4
    VADDPS Y9, Y5, Y5

    // Reduce acc0 into X0.
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    // Reduce acc1 into X3.
    VEXTRACTF128 $1, Y3, X1
    VADDPS X1, X3, X3
    VHADDPS X3, X3, X3
    VHADDPS X3, X3, X3

    // Reduce acc2 into X4.
    VEXTRACTF128 $1, Y4, X1
    VADDPS X1, X4, X4
    VHADDPS X4, X4, X4
    VHADDPS X4, X4, X4

    // Reduce acc3 into X5.
    VEXTRACTF128 $1, Y5, X1
    VADDPS X1, X5, X5
    VHADDPS X5, X5, X5
    VHADDPS X5, X5, X5

    ANDQ $7, CX
    JZ   dot4_avx_done

dot4_avx_scalar:
    VMOVSS (DI), X1
    VMOVSS (SI), X2
    VFMADD231SS X1, X2, X0
    VMOVSS (R8), X2
    VFMADD231SS X1, X2, X3
    VMOVSS (R9), X2
    VFMADD231SS X1, X2, X4
    VMOVSS (R10), X2
    VFMADD231SS X1, X2, X5
    ADDQ $4, DI
    ADDQ $4, SI
    ADDQ $4, R8
    ADDQ $4, R9
    ADDQ $4, R10
    DECQ CX
    JNZ  dot4_avx_scalar

dot4_avx_done:
    VMOVSS X0, (DX)
    VMOVSS X3, 4(DX)
    VMOVSS X4, 8(DX)
    VMOVSS X5, 12(DX)
    VZEROUPPER
    RET

// func dotProduct4AVX512(results, row0, row1, row2, row3, vec *float32, n int)
// Computes four dot products against the same vec, reusing each vec load.
TEXT ·dotProduct4AVX512(SB), NOSPLIT, $0-56
    MOVQ results+0(FP), DX
    MOVQ row0+8(FP), SI
    MOVQ row1+16(FP), R8
    MOVQ row2+24(FP), R9
    MOVQ row3+32(FP), R10
    MOVQ vec+40(FP), DI
    MOVQ n+48(FP), CX

    VPXORD Z0, Z0, Z0          // acc0a
    VPXORD Z3, Z3, Z3          // acc1a
    VPXORD Z4, Z4, Z4          // acc2a
    VPXORD Z5, Z5, Z5          // acc3a
    VPXORD Z6, Z6, Z6          // acc0b
    VPXORD Z7, Z7, Z7          // acc1b
    VPXORD Z8, Z8, Z8          // acc2b
    VPXORD Z9, Z9, Z9          // acc3b

    MOVQ CX, AX
    SHRQ $6, AX                // n / 64
    JZ   dot4_512_loop16_check

dot4_512_loop64:
    VMOVUPS (DI), Z1
    VMOVUPS (SI), Z2
    VFMADD231PS Z1, Z2, Z0
    VMOVUPS (R8), Z2
    VFMADD231PS Z1, Z2, Z3
    VMOVUPS (R9), Z2
    VFMADD231PS Z1, Z2, Z4
    VMOVUPS (R10), Z2
    VFMADD231PS Z1, Z2, Z5

    VMOVUPS 64(DI), Z1
    VMOVUPS 64(SI), Z2
    VFMADD231PS Z1, Z2, Z6
    VMOVUPS 64(R8), Z2
    VFMADD231PS Z1, Z2, Z7
    VMOVUPS 64(R9), Z2
    VFMADD231PS Z1, Z2, Z8
    VMOVUPS 64(R10), Z2
    VFMADD231PS Z1, Z2, Z9

    VMOVUPS 128(DI), Z1
    VMOVUPS 128(SI), Z2
    VFMADD231PS Z1, Z2, Z0
    VMOVUPS 128(R8), Z2
    VFMADD231PS Z1, Z2, Z3
    VMOVUPS 128(R9), Z2
    VFMADD231PS Z1, Z2, Z4
    VMOVUPS 128(R10), Z2
    VFMADD231PS Z1, Z2, Z5

    VMOVUPS 192(DI), Z1
    VMOVUPS 192(SI), Z2
    VFMADD231PS Z1, Z2, Z6
    VMOVUPS 192(R8), Z2
    VFMADD231PS Z1, Z2, Z7
    VMOVUPS 192(R9), Z2
    VFMADD231PS Z1, Z2, Z8
    VMOVUPS 192(R10), Z2
    VFMADD231PS Z1, Z2, Z9

    ADDQ $256, DI
    ADDQ $256, SI
    ADDQ $256, R8
    ADDQ $256, R9
    ADDQ $256, R10
    DECQ AX
    JNZ  dot4_512_loop64

dot4_512_loop16_check:
    ANDQ $63, CX
    MOVQ CX, AX
    SHRQ $4, AX
    JZ   dot4_512_reduce

dot4_512_loop16:
    VMOVUPS (DI), Z1
    VMOVUPS (SI), Z2
    VFMADD231PS Z1, Z2, Z0
    VMOVUPS (R8), Z2
    VFMADD231PS Z1, Z2, Z3
    VMOVUPS (R9), Z2
    VFMADD231PS Z1, Z2, Z4
    VMOVUPS (R10), Z2
    VFMADD231PS Z1, Z2, Z5
    ADDQ $64, DI
    ADDQ $64, SI
    ADDQ $64, R8
    ADDQ $64, R9
    ADDQ $64, R10
    DECQ AX
    JNZ  dot4_512_loop16

dot4_512_reduce:
    VADDPS Z6, Z0, Z0
    VADDPS Z7, Z3, Z3
    VADDPS Z8, Z4, Z4
    VADDPS Z9, Z5, Z5

    // Reduce acc0 into X0.
    // VEXTRACTF64X4 (AVX512F) extracts the same upper 256 bits as VEXTRACTF32X8
    // (AVX512DQ) but without requiring DQ, matching the AVX512F+VL dispatch gate.
    VEXTRACTF64X4 $1, Z0, Y1
    VADDPS Y1, Y0, Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    // Reduce acc1 into X3.
    VEXTRACTF64X4 $1, Z3, Y1
    VADDPS Y1, Y3, Y3
    VEXTRACTF128 $1, Y3, X1
    VADDPS X1, X3, X3
    VHADDPS X3, X3, X3
    VHADDPS X3, X3, X3

    // Reduce acc2 into X4.
    VEXTRACTF64X4 $1, Z4, Y1
    VADDPS Y1, Y4, Y4
    VEXTRACTF128 $1, Y4, X1
    VADDPS X1, X4, X4
    VHADDPS X4, X4, X4
    VHADDPS X4, X4, X4

    // Reduce acc3 into X5.
    VEXTRACTF64X4 $1, Z5, Y1
    VADDPS Y1, Y5, Y5
    VEXTRACTF128 $1, Y5, X1
    VADDPS X1, X5, X5
    VHADDPS X5, X5, X5
    VHADDPS X5, X5, X5

    ANDQ $15, CX
    JZ   dot4_512_done

dot4_512_scalar:
    VMOVSS (DI), X1
    VMOVSS (SI), X2
    VFMADD231SS X1, X2, X0
    VMOVSS (R8), X2
    VFMADD231SS X1, X2, X3
    VMOVSS (R9), X2
    VFMADD231SS X1, X2, X4
    VMOVSS (R10), X2
    VFMADD231SS X1, X2, X5
    ADDQ $4, DI
    ADDQ $4, SI
    ADDQ $4, R8
    ADDQ $4, R9
    ADDQ $4, R10
    DECQ CX
    JNZ  dot4_512_scalar

dot4_512_done:
    VMOVSS X0, (DX)
    VMOVSS X3, 4(DX)
    VMOVSS X4, 8(DX)
    VMOVSS X5, 12(DX)
    VZEROUPPER
    RET

// ============================================================================
// VARIANCE / EUCLIDEAN DISTANCE REDUCTIONS
// Ported from the f64 kernels (f64/f64_amd64.s). float32 doubles the lane count
// versus float64: 4 per XMM (SSE), 8 per YMM (AVX). Accumulation is in float32
// to match variance32Go / euclideanDistance32Go (f32/f32_go.go).
// ============================================================================

// func varianceSSE(a []float32, mean float32) float32
// Population variance: sum((a[i]-mean)^2) / len(a). 4 accumulators of 4 floats.
TEXT ·varianceSSE(SB), NOSPLIT, $0-36
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVSS mean+24(FP), X2
    SHUFPS $0, X2, X2          // broadcast mean to all 4 lanes

    XORPS X0, X0
    XORPS X3, X3
    XORPS X4, X4
    XORPS X5, X5

    // Process 16 elements (4 vectors of 4) per iteration
    MOVQ CX, AX
    SHRQ $4, AX                // len / 16
    JZ   var32_sse_loop4_check

var32_sse_loop16:
    MOVUPS 0(SI), X1
    SUBPS X2, X1
    MULPS X1, X1
    ADDPS X1, X0

    MOVUPS 16(SI), X1
    SUBPS X2, X1
    MULPS X1, X1
    ADDPS X1, X3

    MOVUPS 32(SI), X1
    SUBPS X2, X1
    MULPS X1, X1
    ADDPS X1, X4

    MOVUPS 48(SI), X1
    SUBPS X2, X1
    MULPS X1, X1
    ADDPS X1, X5

    ADDQ $64, SI
    DECQ AX
    JNZ  var32_sse_loop16

    ADDPS X3, X0
    ADDPS X5, X4
    ADDPS X4, X0

var32_sse_loop4_check:
    ANDQ $15, CX
    MOVQ CX, AX
    SHRQ $2, AX
    JZ   var32_sse_remainder

var32_sse_loop4:
    MOVUPS (SI), X1
    SUBPS X2, X1
    MULPS X1, X1
    ADDPS X1, X0
    ADDQ $16, SI
    DECQ AX
    JNZ  var32_sse_loop4

var32_sse_remainder:
    // Horizontal sum of the 4 float lanes in X0 into X0[0].
    MOVAPS X0, X1
    SHUFPS $0x0E, X1, X1
    ADDPS X1, X0
    MOVAPS X0, X1
    SHUFPS $0x01, X1, X1
    ADDSS X1, X0

    ANDQ $3, CX
    JZ   var32_sse_divide
    MOVSS mean+24(FP), X2      // scalar mean for the tail

var32_sse_scalar:
    MOVSS (SI), X1
    SUBSS X2, X1
    MULSS X1, X1
    ADDSS X1, X0
    ADDQ $4, SI
    DECQ CX
    JNZ  var32_sse_scalar

var32_sse_divide:
    MOVQ a_len+8(FP), CX
    CVTSQ2SS CX, X1
    DIVSS X1, X0
    MOVSS X0, ret+32(FP)
    RET

// func varianceAVX(a []float32, mean float32) float32
// 4 independent FMA accumulators of 8 floats to hide FMA latency.
TEXT ·varianceAVX(SB), NOSPLIT, $0-36
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    VBROADCASTSS mean+24(FP), Y2

    VXORPS Y0, Y0, Y0
    VXORPS Y3, Y3, Y3
    VXORPS Y4, Y4, Y4
    VXORPS Y5, Y5, Y5

    // Process 32 elements (4 vectors of 8) per iteration
    MOVQ CX, AX
    SHRQ $5, AX                // len / 32
    JZ   var32_avx_loop8_check

var32_avx_loop32:
    VMOVUPS 0(SI), Y1
    VSUBPS Y2, Y1, Y1
    VFMADD231PS Y1, Y1, Y0

    VMOVUPS 32(SI), Y1
    VSUBPS Y2, Y1, Y1
    VFMADD231PS Y1, Y1, Y3

    VMOVUPS 64(SI), Y1
    VSUBPS Y2, Y1, Y1
    VFMADD231PS Y1, Y1, Y4

    VMOVUPS 96(SI), Y1
    VSUBPS Y2, Y1, Y1
    VFMADD231PS Y1, Y1, Y5

    ADDQ $128, SI
    DECQ AX
    JNZ  var32_avx_loop32

    VADDPS Y3, Y0, Y0
    VADDPS Y5, Y4, Y4
    VADDPS Y4, Y0, Y0

var32_avx_loop8_check:
    ANDQ $31, CX
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   var32_avx_remainder

var32_avx_loop8:
    VMOVUPS (SI), Y1
    VSUBPS Y2, Y1, Y1
    VFMADD231PS Y1, Y1, Y0
    ADDQ $32, SI
    DECQ AX
    JNZ  var32_avx_loop8

var32_avx_remainder:
    // Reduce Y0 (8 floats) to X0[0] before any VEX scalar op zeroes the upper YMM.
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    ANDQ $7, CX
    JZ   var32_avx_divide
    VMOVSS mean+24(FP), X2

var32_avx_scalar:
    VMOVSS (SI), X1
    VSUBSS X2, X1, X1
    VFMADD231SS X1, X1, X0
    ADDQ $4, SI
    DECQ CX
    JNZ  var32_avx_scalar

var32_avx_divide:
    MOVQ a_len+8(FP), CX
    CVTSQ2SS CX, X1
    VDIVSS X1, X0, X0
    VMOVSS X0, ret+32(FP)
    VZEROUPPER
    RET

// func euclideanDistanceSSE(a, b []float32) float32
// sqrt(sum((a[i]-b[i])^2)). 4 accumulators of 4 floats. Callers pass equal-length
// slices (EuclideanDistance slices both to min length).
TEXT ·euclideanDistanceSSE(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    XORPS X0, X0
    XORPS X3, X3
    XORPS X4, X4
    XORPS X5, X5

    MOVQ CX, AX
    SHRQ $4, AX                // len / 16
    JZ   euclid32_sse_loop4_check

euclid32_sse_loop16:
    MOVUPS 0(SI), X1
    MOVUPS 0(DI), X2
    SUBPS X2, X1
    MULPS X1, X1
    ADDPS X1, X0

    MOVUPS 16(SI), X1
    MOVUPS 16(DI), X2
    SUBPS X2, X1
    MULPS X1, X1
    ADDPS X1, X3

    MOVUPS 32(SI), X1
    MOVUPS 32(DI), X2
    SUBPS X2, X1
    MULPS X1, X1
    ADDPS X1, X4

    MOVUPS 48(SI), X1
    MOVUPS 48(DI), X2
    SUBPS X2, X1
    MULPS X1, X1
    ADDPS X1, X5

    ADDQ $64, SI
    ADDQ $64, DI
    DECQ AX
    JNZ  euclid32_sse_loop16

    ADDPS X3, X0
    ADDPS X5, X4
    ADDPS X4, X0

euclid32_sse_loop4_check:
    ANDQ $15, CX
    MOVQ CX, AX
    SHRQ $2, AX
    JZ   euclid32_sse_remainder

euclid32_sse_loop4:
    MOVUPS (SI), X1
    MOVUPS (DI), X2
    SUBPS X2, X1
    MULPS X1, X1
    ADDPS X1, X0
    ADDQ $16, SI
    ADDQ $16, DI
    DECQ AX
    JNZ  euclid32_sse_loop4

euclid32_sse_remainder:
    MOVAPS X0, X1
    SHUFPS $0x0E, X1, X1
    ADDPS X1, X0
    MOVAPS X0, X1
    SHUFPS $0x01, X1, X1
    ADDSS X1, X0

    ANDQ $3, CX
    JZ   euclid32_sse_sqrt

euclid32_sse_scalar:
    MOVSS (SI), X1
    MOVSS (DI), X2
    SUBSS X2, X1
    MULSS X1, X1
    ADDSS X1, X0
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ CX
    JNZ  euclid32_sse_scalar

euclid32_sse_sqrt:
    SQRTSS X0, X0
    MOVSS X0, ret+48(FP)
    RET

// func euclideanDistanceAVX(a, b []float32) float32
// 4 independent FMA accumulators of 8 floats, then horizontal reduce and sqrt.
TEXT ·euclideanDistanceAVX(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    VXORPS Y0, Y0, Y0
    VXORPS Y3, Y3, Y3
    VXORPS Y4, Y4, Y4
    VXORPS Y5, Y5, Y5

    MOVQ CX, AX
    SHRQ $5, AX                // len / 32
    JZ   euclid32_avx_loop8_check

euclid32_avx_loop32:
    VMOVUPS (SI), Y1
    VMOVUPS (DI), Y2
    VSUBPS Y2, Y1, Y1
    VFMADD231PS Y1, Y1, Y0

    VMOVUPS 32(SI), Y1
    VMOVUPS 32(DI), Y2
    VSUBPS Y2, Y1, Y1
    VFMADD231PS Y1, Y1, Y3

    VMOVUPS 64(SI), Y1
    VMOVUPS 64(DI), Y2
    VSUBPS Y2, Y1, Y1
    VFMADD231PS Y1, Y1, Y4

    VMOVUPS 96(SI), Y1
    VMOVUPS 96(DI), Y2
    VSUBPS Y2, Y1, Y1
    VFMADD231PS Y1, Y1, Y5

    ADDQ $128, SI
    ADDQ $128, DI
    DECQ AX
    JNZ  euclid32_avx_loop32

    VADDPS Y3, Y0, Y0
    VADDPS Y5, Y4, Y4
    VADDPS Y4, Y0, Y0

euclid32_avx_loop8_check:
    ANDQ $31, CX
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   euclid32_avx_remainder

euclid32_avx_loop8:
    VMOVUPS (SI), Y1
    VMOVUPS (DI), Y2
    VSUBPS Y2, Y1, Y1
    VFMADD231PS Y1, Y1, Y0
    ADDQ $32, SI
    ADDQ $32, DI
    DECQ AX
    JNZ  euclid32_avx_loop8

euclid32_avx_remainder:
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    ANDQ $7, CX
    JZ   euclid32_avx_sqrt

euclid32_avx_scalar:
    VMOVSS (SI), X1
    VMOVSS (DI), X2
    VSUBSS X2, X1, X1
    VFMADD231SS X1, X1, X0
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ CX
    JNZ  euclid32_avx_scalar

euclid32_avx_sqrt:
    VSQRTSS X0, X0, X0
    VMOVSS X0, ret+48(FP)
    VZEROUPPER
    RET

// ============================================================================
// logAVX / powAVX / powElemAVX (f32): vectorized natural log core (issue #109)
// ============================================================================

// Mantissa reduction offset: bits(x) - log32_off puts the biased exponent of
// m = x / 2^e in the tmp exponent field such that m in [sqrt(2)/2, sqrt(2)),
// with e = tmp >> 23 (arithmetic) and bits(m) = bits(x) - (tmp & 0xff800000).
DATA log32_off<>+0x00(SB)/4, $0x3f350000
DATA log32_off<>+0x04(SB)/4, $0x3f350000
DATA log32_off<>+0x08(SB)/4, $0x3f350000
DATA log32_off<>+0x0c(SB)/4, $0x3f350000
DATA log32_off<>+0x10(SB)/4, $0x3f350000
DATA log32_off<>+0x14(SB)/4, $0x3f350000
DATA log32_off<>+0x18(SB)/4, $0x3f350000
DATA log32_off<>+0x1c(SB)/4, $0x3f350000
GLOBL log32_off<>(SB), RODATA|NOPTR, $32

DATA log32_expmask<>+0x00(SB)/4, $0xff800000
DATA log32_expmask<>+0x04(SB)/4, $0xff800000
DATA log32_expmask<>+0x08(SB)/4, $0xff800000
DATA log32_expmask<>+0x0c(SB)/4, $0xff800000
DATA log32_expmask<>+0x10(SB)/4, $0xff800000
DATA log32_expmask<>+0x14(SB)/4, $0xff800000
DATA log32_expmask<>+0x18(SB)/4, $0xff800000
DATA log32_expmask<>+0x1c(SB)/4, $0xff800000
GLOBL log32_expmask<>(SB), RODATA|NOPTR, $32

// FLT_MIN = 1.1754944e-38: positive inputs below this are subnormal and
// pre-scaled by 2^32 (exponent bias -32) before the reduction.
DATA log32_fltmin<>+0x00(SB)/4, $0x00800000
DATA log32_fltmin<>+0x04(SB)/4, $0x00800000
DATA log32_fltmin<>+0x08(SB)/4, $0x00800000
DATA log32_fltmin<>+0x0c(SB)/4, $0x00800000
DATA log32_fltmin<>+0x10(SB)/4, $0x00800000
DATA log32_fltmin<>+0x14(SB)/4, $0x00800000
DATA log32_fltmin<>+0x18(SB)/4, $0x00800000
DATA log32_fltmin<>+0x1c(SB)/4, $0x00800000
GLOBL log32_fltmin<>(SB), RODATA|NOPTR, $32

DATA log32_two32<>+0x00(SB)/4, $0x4f800000  // 2^32
DATA log32_two32<>+0x04(SB)/4, $0x4f800000
DATA log32_two32<>+0x08(SB)/4, $0x4f800000
DATA log32_two32<>+0x0c(SB)/4, $0x4f800000
DATA log32_two32<>+0x10(SB)/4, $0x4f800000
DATA log32_two32<>+0x14(SB)/4, $0x4f800000
DATA log32_two32<>+0x18(SB)/4, $0x4f800000
DATA log32_two32<>+0x1c(SB)/4, $0x4f800000
GLOBL log32_two32<>(SB), RODATA|NOPTR, $32

DATA log32_negsc<>+0x00(SB)/4, $0xc2000000  // -32.0 (exponent bias)
DATA log32_negsc<>+0x04(SB)/4, $0xc2000000
DATA log32_negsc<>+0x08(SB)/4, $0xc2000000
DATA log32_negsc<>+0x0c(SB)/4, $0xc2000000
DATA log32_negsc<>+0x10(SB)/4, $0xc2000000
DATA log32_negsc<>+0x14(SB)/4, $0xc2000000
DATA log32_negsc<>+0x18(SB)/4, $0xc2000000
DATA log32_negsc<>+0x1c(SB)/4, $0xc2000000
GLOBL log32_negsc<>(SB), RODATA|NOPTR, $32

// Cephes logf minimax polynomial for ln(m), m in [sqrt(2)/2, sqrt(2)):
// with z = m-1, ln(m) = z - 0.5*z^2 + z^3*P(z),
// P(z) = ((((((((p0*z + p1)*z + p2)*z + p3)*z + p4)*z + p5)*z + p6)*z
// + p7)*z + p8). Worst-case relative error of the full ln(x) is ~1.2 ulps.
DATA log32_p0<>+0x00(SB)/4, $0x3d9021bb  // +7.0376836292e-2
DATA log32_p0<>+0x04(SB)/4, $0x3d9021bb
DATA log32_p0<>+0x08(SB)/4, $0x3d9021bb
DATA log32_p0<>+0x0c(SB)/4, $0x3d9021bb
DATA log32_p0<>+0x10(SB)/4, $0x3d9021bb
DATA log32_p0<>+0x14(SB)/4, $0x3d9021bb
DATA log32_p0<>+0x18(SB)/4, $0x3d9021bb
DATA log32_p0<>+0x1c(SB)/4, $0x3d9021bb
GLOBL log32_p0<>(SB), RODATA|NOPTR, $32

DATA log32_p1<>+0x00(SB)/4, $0xbdebd1b8  // -1.1514610310e-1
DATA log32_p1<>+0x04(SB)/4, $0xbdebd1b8
DATA log32_p1<>+0x08(SB)/4, $0xbdebd1b8
DATA log32_p1<>+0x0c(SB)/4, $0xbdebd1b8
DATA log32_p1<>+0x10(SB)/4, $0xbdebd1b8
DATA log32_p1<>+0x14(SB)/4, $0xbdebd1b8
DATA log32_p1<>+0x18(SB)/4, $0xbdebd1b8
DATA log32_p1<>+0x1c(SB)/4, $0xbdebd1b8
GLOBL log32_p1<>(SB), RODATA|NOPTR, $32

DATA log32_p2<>+0x00(SB)/4, $0x3def251a  // +1.1676998740e-1
DATA log32_p2<>+0x04(SB)/4, $0x3def251a
DATA log32_p2<>+0x08(SB)/4, $0x3def251a
DATA log32_p2<>+0x0c(SB)/4, $0x3def251a
DATA log32_p2<>+0x10(SB)/4, $0x3def251a
DATA log32_p2<>+0x14(SB)/4, $0x3def251a
DATA log32_p2<>+0x18(SB)/4, $0x3def251a
DATA log32_p2<>+0x1c(SB)/4, $0x3def251a
GLOBL log32_p2<>(SB), RODATA|NOPTR, $32

DATA log32_p3<>+0x00(SB)/4, $0xbdfe5d4f  // -1.2420140846e-1
DATA log32_p3<>+0x04(SB)/4, $0xbdfe5d4f
DATA log32_p3<>+0x08(SB)/4, $0xbdfe5d4f
DATA log32_p3<>+0x0c(SB)/4, $0xbdfe5d4f
DATA log32_p3<>+0x10(SB)/4, $0xbdfe5d4f
DATA log32_p3<>+0x14(SB)/4, $0xbdfe5d4f
DATA log32_p3<>+0x18(SB)/4, $0xbdfe5d4f
DATA log32_p3<>+0x1c(SB)/4, $0xbdfe5d4f
GLOBL log32_p3<>(SB), RODATA|NOPTR, $32

DATA log32_p4<>+0x00(SB)/4, $0x3e11e9bf  // +1.4249322787e-1
DATA log32_p4<>+0x04(SB)/4, $0x3e11e9bf
DATA log32_p4<>+0x08(SB)/4, $0x3e11e9bf
DATA log32_p4<>+0x0c(SB)/4, $0x3e11e9bf
DATA log32_p4<>+0x10(SB)/4, $0x3e11e9bf
DATA log32_p4<>+0x14(SB)/4, $0x3e11e9bf
DATA log32_p4<>+0x18(SB)/4, $0x3e11e9bf
DATA log32_p4<>+0x1c(SB)/4, $0x3e11e9bf
GLOBL log32_p4<>(SB), RODATA|NOPTR, $32

DATA log32_p5<>+0x00(SB)/4, $0xbe2aae50  // -1.6668057665e-1
DATA log32_p5<>+0x04(SB)/4, $0xbe2aae50
DATA log32_p5<>+0x08(SB)/4, $0xbe2aae50
DATA log32_p5<>+0x0c(SB)/4, $0xbe2aae50
DATA log32_p5<>+0x10(SB)/4, $0xbe2aae50
DATA log32_p5<>+0x14(SB)/4, $0xbe2aae50
DATA log32_p5<>+0x18(SB)/4, $0xbe2aae50
DATA log32_p5<>+0x1c(SB)/4, $0xbe2aae50
GLOBL log32_p5<>(SB), RODATA|NOPTR, $32

DATA log32_p6<>+0x00(SB)/4, $0x3e4cceac  // +2.0000714765e-1
DATA log32_p6<>+0x04(SB)/4, $0x3e4cceac
DATA log32_p6<>+0x08(SB)/4, $0x3e4cceac
DATA log32_p6<>+0x0c(SB)/4, $0x3e4cceac
DATA log32_p6<>+0x10(SB)/4, $0x3e4cceac
DATA log32_p6<>+0x14(SB)/4, $0x3e4cceac
DATA log32_p6<>+0x18(SB)/4, $0x3e4cceac
DATA log32_p6<>+0x1c(SB)/4, $0x3e4cceac
GLOBL log32_p6<>(SB), RODATA|NOPTR, $32

DATA log32_p7<>+0x00(SB)/4, $0xbe7ffffc  // -2.4999993993e-1
DATA log32_p7<>+0x04(SB)/4, $0xbe7ffffc
DATA log32_p7<>+0x08(SB)/4, $0xbe7ffffc
DATA log32_p7<>+0x0c(SB)/4, $0xbe7ffffc
DATA log32_p7<>+0x10(SB)/4, $0xbe7ffffc
DATA log32_p7<>+0x14(SB)/4, $0xbe7ffffc
DATA log32_p7<>+0x18(SB)/4, $0xbe7ffffc
DATA log32_p7<>+0x1c(SB)/4, $0xbe7ffffc
GLOBL log32_p7<>(SB), RODATA|NOPTR, $32

DATA log32_p8<>+0x00(SB)/4, $0x3eaaaaaa  // +3.3333331174e-1
DATA log32_p8<>+0x04(SB)/4, $0x3eaaaaaa
DATA log32_p8<>+0x08(SB)/4, $0x3eaaaaaa
DATA log32_p8<>+0x0c(SB)/4, $0x3eaaaaaa
DATA log32_p8<>+0x10(SB)/4, $0x3eaaaaaa
DATA log32_p8<>+0x14(SB)/4, $0x3eaaaaaa
DATA log32_p8<>+0x18(SB)/4, $0x3eaaaaaa
DATA log32_p8<>+0x1c(SB)/4, $0x3eaaaaaa
GLOBL log32_p8<>(SB), RODATA|NOPTR, $32

// Cephes logf ln(2) hi/lo split for the pow kernels' fixed natural-log
// reconstruction (logAVX takes its split via arguments instead).
DATA log32_ln2hi<>+0x00(SB)/4, $0x3f318000  // 0.693359375
DATA log32_ln2hi<>+0x04(SB)/4, $0x3f318000
DATA log32_ln2hi<>+0x08(SB)/4, $0x3f318000
DATA log32_ln2hi<>+0x0c(SB)/4, $0x3f318000
DATA log32_ln2hi<>+0x10(SB)/4, $0x3f318000
DATA log32_ln2hi<>+0x14(SB)/4, $0x3f318000
DATA log32_ln2hi<>+0x18(SB)/4, $0x3f318000
DATA log32_ln2hi<>+0x1c(SB)/4, $0x3f318000
GLOBL log32_ln2hi<>(SB), RODATA|NOPTR, $32

DATA log32_ln2lo<>+0x00(SB)/4, $0xb95e8083  // -2.12194440e-4
DATA log32_ln2lo<>+0x04(SB)/4, $0xb95e8083
DATA log32_ln2lo<>+0x08(SB)/4, $0xb95e8083
DATA log32_ln2lo<>+0x0c(SB)/4, $0xb95e8083
DATA log32_ln2lo<>+0x10(SB)/4, $0xb95e8083
DATA log32_ln2lo<>+0x14(SB)/4, $0xb95e8083
DATA log32_ln2lo<>+0x18(SB)/4, $0xb95e8083
DATA log32_ln2lo<>+0x1c(SB)/4, $0xb95e8083
GLOBL log32_ln2lo<>(SB), RODATA|NOPTR, $32

DATA log32_posinf<>+0x00(SB)/4, $0x7f800000
DATA log32_posinf<>+0x04(SB)/4, $0x7f800000
DATA log32_posinf<>+0x08(SB)/4, $0x7f800000
DATA log32_posinf<>+0x0c(SB)/4, $0x7f800000
DATA log32_posinf<>+0x10(SB)/4, $0x7f800000
DATA log32_posinf<>+0x14(SB)/4, $0x7f800000
DATA log32_posinf<>+0x18(SB)/4, $0x7f800000
DATA log32_posinf<>+0x1c(SB)/4, $0x7f800000
GLOBL log32_posinf<>(SB), RODATA|NOPTR, $32

DATA log32_neginf<>+0x00(SB)/4, $0xff800000
DATA log32_neginf<>+0x04(SB)/4, $0xff800000
DATA log32_neginf<>+0x08(SB)/4, $0xff800000
DATA log32_neginf<>+0x0c(SB)/4, $0xff800000
DATA log32_neginf<>+0x10(SB)/4, $0xff800000
DATA log32_neginf<>+0x14(SB)/4, $0xff800000
DATA log32_neginf<>+0x18(SB)/4, $0xff800000
DATA log32_neginf<>+0x1c(SB)/4, $0xff800000
GLOBL log32_neginf<>(SB), RODATA|NOPTR, $32

DATA log32_nan<>+0x00(SB)/4, $0x7fc00000
DATA log32_nan<>+0x04(SB)/4, $0x7fc00000
DATA log32_nan<>+0x08(SB)/4, $0x7fc00000
DATA log32_nan<>+0x0c(SB)/4, $0x7fc00000
DATA log32_nan<>+0x10(SB)/4, $0x7fc00000
DATA log32_nan<>+0x14(SB)/4, $0x7fc00000
DATA log32_nan<>+0x18(SB)/4, $0x7fc00000
DATA log32_nan<>+0x1c(SB)/4, $0x7fc00000
GLOBL log32_nan<>(SB), RODATA|NOPTR, $32

// func logAVX(dst, src []float32, k1hi, k1lo, k2 float32)
// Shared kernel for Log, Log2, and Log10: per lane it computes
// result = e*k1hi + (lnm*k2 + e*k1lo), with x = m*2^e, m in
// [sqrt(2)/2, sqrt(2)) and lnm = ln(m) = z - 0.5*z^2 + z^3*P(z) for
// z = m-1 (Cephes logf degree-8 minimax polynomial). Positive subnormal
// inputs are pre-scaled by 2^32 (exponent bias -32). Special lanes are fixed
// up with blends from the original input: +Inf -> +Inf, +-0 -> -Inf,
// x < 0 or NaN -> NaN, matching math.Log. Requires AVX2 (YMM integer ops in
// the exponent extraction) and FMA. Processes 8 elements per iteration; the
// 0-7 element tail uses the scalar path below.
TEXT ·logAVX(SB), NOSPLIT, $0-60
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    // Loop-invariant constants. The low 128 bits (X7-X15) are reused by the
    // scalar remainder path.
    VMOVUPS log32_negsc<>(SB), Y7      // Y7 = -32.0 (subnormal exponent bias)
    VMOVUPS log32_two32<>(SB), Y8      // Y8 = 2^32 (subnormal pre-scale)
    VMOVUPS log32_fltmin<>(SB), Y9     // Y9 = FLT_MIN
    VMOVUPS exp_one<>(SB), Y10         // Y10 = 1.0
    VMOVUPS log32_expmask<>(SB), Y11   // Y11 = 0xff800000
    VMOVUPS log32_off<>(SB), Y12       // Y12 = reduction offset
    VBROADCASTSS k2+56(FP), Y13        // Y13 = k2
    VBROADCASTSS k1lo+52(FP), Y14      // Y14 = k1lo
    VBROADCASTSS k1hi+48(FP), Y15      // Y15 = k1hi

    MOVQ CX, R8
    SHRQ $3, R8                        // len / 8
    JZ   log32_remainder

log32_loop8:
    VMOVUPS (SI), Y0                   // Y0 = x (kept for the special-lane blends)

    // Subnormal pre-scale: lanes with 0 < x < FLT_MIN are scaled by 2^32 and
    // carry an exponent bias of -32. (Negative/NaN lanes fail the compare or
    // produce garbage that the final blends overwrite.)
    VCMPPS $17, Y9, Y0, Y1             // Y1 = mask: x < FLT_MIN (LT_OQ)
    VMULPS Y8, Y0, Y2                  // Y2 = x * 2^32
    VBLENDVPS Y1, Y2, Y0, Y2           // Y2 = xs
    VANDPS Y7, Y1, Y1                  // Y1 = ebias = subnormal ? -32.0 : 0.0

    // Exponent/mantissa split: tmp = bits(xs) - OFF; e = tmp >> 23
    // (arithmetic, directly on the 32-bit lanes); bits(m) = bits(xs) -
    // (tmp & 0xff800000), leaving m in [sqrt(2)/2, sqrt(2)).
    VPSUBD Y12, Y2, Y3                 // Y3 = tmp
    VPAND Y11, Y3, Y4                  // Y4 = tmp & expmask
    VPSUBD Y4, Y2, Y4                  // Y4 = m
    VPSRAD $23, Y3, Y3                 // Y3 = e (int32)
    VCVTDQ2PS Y3, Y3                   // Y3 = e as float32
    VADDPS Y1, Y3, Y3                  // Y3 = e + ebias

    // z = m - 1, zz = z^2
    VSUBPS Y10, Y4, Y5                 // Y5 = z
    VMULPS Y5, Y5, Y4                  // Y4 = zz

    // P(z), Horner with memory-operand FMAs
    VMOVUPS log32_p0<>(SB), Y2
    VFMADD213PS log32_p1<>(SB), Y5, Y2 // Y2 = Y2*z + p1
    VFMADD213PS log32_p2<>(SB), Y5, Y2
    VFMADD213PS log32_p3<>(SB), Y5, Y2
    VFMADD213PS log32_p4<>(SB), Y5, Y2
    VFMADD213PS log32_p5<>(SB), Y5, Y2
    VFMADD213PS log32_p6<>(SB), Y5, Y2
    VFMADD213PS log32_p7<>(SB), Y5, Y2
    VFMADD213PS log32_p8<>(SB), Y5, Y2 // Y2 = P(z)

    // lnm = z + (z^3*P(z) - 0.5*zz)
    VMULPS Y5, Y4, Y6                  // Y6 = z^3
    VMULPS Y6, Y2, Y2                  // Y2 = z^3 * P(z)
    VFNMADD231PS exp_half<>(SB), Y4, Y2 // Y2 -= 0.5*zz
    VADDPS Y5, Y2, Y2                  // Y2 = lnm

    // result = e*k1hi + (lnm*k2 + e*k1lo)
    VMULPS Y14, Y3, Y4                 // Y4 = e * k1lo
    VFMADD231PS Y13, Y2, Y4            // Y4 += lnm * k2
    VFMADD231PS Y15, Y3, Y4            // Y4 += e * k1hi

    // Special lanes from the original x: +Inf -> +Inf, +-0 -> -Inf,
    // x < 0 or NaN -> NaN (canonical quiet NaN, like math.Log).
    VMOVUPS log32_posinf<>(SB), Y2
    VCMPPS $0, Y2, Y0, Y1              // Y1 = mask: x == +Inf (EQ_OQ)
    VBLENDVPS Y1, Y2, Y4, Y4
    VXORPS Y2, Y2, Y2
    VCMPPS $0, Y2, Y0, Y1              // Y1 = mask: x == +-0
    VMOVUPS log32_neginf<>(SB), Y3
    VBLENDVPS Y1, Y3, Y4, Y4
    VCMPPS $17, Y2, Y0, Y1             // Y1 = mask: x < 0
    VCMPPS $3, Y0, Y0, Y2              // Y2 = mask: x unordered (NaN)
    VORPS Y2, Y1, Y1
    VMOVUPS log32_nan<>(SB), Y3
    VBLENDVPS Y1, Y3, Y4, Y4

    VMOVUPS Y4, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ R8
    JNZ  log32_loop8

log32_remainder:
    ANDQ $7, CX
    JZ   log32_done

log32_scalar:
    MOVL (SI), AX                      // AX = bits(x)
    VMOVSS (SI), X0                    // X0 = x

    // Specials first (JP before JB/JE: unordered sets ZF=PF=CF=1)
    VXORPS X1, X1, X1
    VUCOMISS X1, X0
    JP   log32_scalar_nan
    JB   log32_scalar_nan              // x < 0
    JE   log32_scalar_neginf           // x == +-0
    MOVL $0x7F800000, BX
    CMPL AX, BX
    JEQ  log32_scalar_posinf

    // Subnormal pre-scale (x positive finite; bits compare as ints)
    XORL R9, R9
    MOVL $0x00800000, BX
    CMPL AX, BX
    JGE  log32_scalar_normal
    VMOVSS log32_two32<>(SB), X2       // 2^32
    VMULSS X2, X0, X0
    VMOVD X0, AX
    MOVL $-32, R9

log32_scalar_normal:
    MOVL $0x3F350000, BX
    MOVL AX, R10
    SUBL BX, R10                       // R10 = tmp = bits - OFF
    MOVL R10, R11
    SARL $23, R11                      // R11 = e
    ADDL R9, R11                       // e += bias
    MOVL $0xFF800000, BX
    ANDL BX, R10
    SUBL R10, AX                       // AX = bits(m)
    VMOVD AX, X2                       // X2 = m
    VCVTSI2SSL R11, X3, X3             // X3 = e as float32

    // z = m - 1, zz = z^2 (X10 = 1.0 from the vector constants)
    VSUBSS X10, X2, X4                 // X4 = z
    VMULSS X4, X4, X5                  // X5 = zz

    VMOVSS log32_p0<>(SB), X1
    VFMADD213SS log32_p1<>(SB), X4, X1
    VFMADD213SS log32_p2<>(SB), X4, X1
    VFMADD213SS log32_p3<>(SB), X4, X1
    VFMADD213SS log32_p4<>(SB), X4, X1
    VFMADD213SS log32_p5<>(SB), X4, X1
    VFMADD213SS log32_p6<>(SB), X4, X1
    VFMADD213SS log32_p7<>(SB), X4, X1
    VFMADD213SS log32_p8<>(SB), X4, X1 // X1 = P(z)

    VMULSS X4, X5, X6                  // X6 = z^3
    VMULSS X6, X1, X1                  // X1 = z^3*P(z)
    VFNMADD231SS exp_half<>(SB), X5, X1 // X1 -= 0.5*zz
    VADDSS X4, X1, X1                  // X1 = lnm

    VMULSS X14, X3, X5                 // e * k1lo
    VFMADD231SS X13, X1, X5            // += lnm * k2
    VFMADD231SS X15, X3, X5            // += e * k1hi
    VMOVSS X5, (DX)
    JMP  log32_scalar_next

log32_scalar_nan:
    MOVL $0x7FC00000, (DX)
    JMP  log32_scalar_next

log32_scalar_neginf:
    MOVL $0xFF800000, (DX)
    JMP  log32_scalar_next

log32_scalar_posinf:
    MOVL $0x7F800000, (DX)

log32_scalar_next:
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  log32_scalar

log32_done:
    VZEROUPPER
    RET

// func powAVX(dst, src []float32, exp float32)
// Fused pow(x, p) = exp(p*ln(x)) for slices whose elements are all positive
// and finite (the dispatcher guarantees this, see powSIMDOK32). The log core
// matches logAVX (constants from memory instead of registers); the exp core
// matches expAVX except y = p*ln(x) is clamped to [-104, 89] (past
// ln(MaxFloat32) and ln of the smallest subnormal) and the 2^k
// reconstruction is split into 2^(k>>1) * 2^(k-(k>>1)), so overflow goes to
// +Inf and underflow degrades gradually through subnormals to 0, matching
// math.Pow's result classes. Accuracy is ~1.4e-5 relative (log error
// amplified by |y|, then the exp polynomial). Requires AVX2 and FMA.
TEXT ·powAVX(SB), NOSPLIT, $0-52
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    // Exp-core constants in registers (the log core uses memory operands).
    // The low 128 bits (X7-X15) are reused by the scalar remainder path.
    VBROADCASTSS exp+48(FP), Y7        // Y7 = p
    VMOVUPS exp_c5<>(SB), Y8           // Y8 = 1/120
    VMOVUPS exp_c4<>(SB), Y9           // Y9 = 1/24
    VMOVUPS exp_c3<>(SB), Y10          // Y10 = 1/6
    VMOVUPS exp_c2<>(SB), Y11          // Y11 = 0.5
    VMOVUPS exp_one<>(SB), Y12         // Y12 = 1.0
    VMOVUPS exp_magic<>(SB), Y13       // Y13 = rounding magic
    VMOVUPS exp_ln2<>(SB), Y14         // Y14 = ln(2)
    VMOVUPS exp_log2e<>(SB), Y15       // Y15 = log2(e)

    MOVQ CX, R8
    SHRQ $3, R8
    JZ   pow32_remainder

pow32_loop8:
    VMOVUPS (SI), Y0                   // Y0 = x (positive finite)

    // --- log core (see logAVX) ---
    VCMPPS $17, log32_fltmin<>(SB), Y0, Y1
    VMULPS log32_two32<>(SB), Y0, Y2
    VBLENDVPS Y1, Y2, Y0, Y0           // x, subnormals pre-scaled
    VANDPS log32_negsc<>(SB), Y1, Y1   // ebias
    VPSUBD log32_off<>(SB), Y0, Y2     // tmp
    VPAND log32_expmask<>(SB), Y2, Y3
    VPSUBD Y3, Y0, Y3                  // m
    VPSRAD $23, Y2, Y2
    VCVTDQ2PS Y2, Y2
    VADDPS Y1, Y2, Y2                  // e
    VSUBPS Y12, Y3, Y5                 // z = m - 1
    VMULPS Y5, Y5, Y4                  // zz
    VMOVUPS log32_p0<>(SB), Y3
    VFMADD213PS log32_p1<>(SB), Y5, Y3
    VFMADD213PS log32_p2<>(SB), Y5, Y3
    VFMADD213PS log32_p3<>(SB), Y5, Y3
    VFMADD213PS log32_p4<>(SB), Y5, Y3
    VFMADD213PS log32_p5<>(SB), Y5, Y3
    VFMADD213PS log32_p6<>(SB), Y5, Y3
    VFMADD213PS log32_p7<>(SB), Y5, Y3
    VFMADD213PS log32_p8<>(SB), Y5, Y3 // P(z)
    VMULPS Y5, Y4, Y6                  // z^3
    VMULPS Y6, Y3, Y3                  // z^3*P(z)
    VFNMADD231PS exp_half<>(SB), Y4, Y3 // -= 0.5*zz
    VADDPS Y5, Y3, Y3                  // lnm
    VMULPS log32_ln2lo<>(SB), Y2, Y4   // e*ln2lo
    VADDPS Y3, Y4, Y4                  // + lnm
    VFMADD231PS log32_ln2hi<>(SB), Y2, Y4 // Y4 = ln(x)

    // y = p*ln(x); keep the pre-clamp y in Y6 for the overflow blends, then
    // clamp to [-88, 88] for the exp core
    VMULPS Y7, Y4, Y0
    VMINPS log32_powclamp_hi<>(SB), Y0, Y0
    VMAXPS log32_powclamp_lo<>(SB), Y0, Y0

    // --- exp core (see expAVX) ---
    VMULPS Y15, Y0, Y1                 // y * log2e
    VADDPS Y13, Y1, Y2                 // + magic
    VSUBPS Y13, Y2, Y3                 // Y3 = k = round(y * log2e)
    VMULPS Y14, Y3, Y4                 // k * ln2
    VSUBPS Y4, Y0, Y0                  // r
    VMULPS Y0, Y8, Y1                  // r*c5
    VADDPS Y9, Y1, Y1
    VMULPS Y0, Y1, Y1
    VADDPS Y10, Y1, Y1
    VMULPS Y0, Y1, Y1
    VADDPS Y11, Y1, Y1
    VMULPS Y0, Y1, Y1
    VADDPS Y12, Y1, Y1
    VMULPS Y0, Y1, Y1
    VADDPS Y12, Y1, Y1                 // exp(r)
    // Split 2^k reconstruction: k can reach +-150 here, past the biased
    // exponent range, so build 2^(k>>1) and 2^(k-(k>>1)) separately. The
    // double multiply overflows to +Inf / underflows through subnormals to 0
    // exactly where math.Pow does.
    VCVTPS2DQ Y3, Y4                   // int(k)
    VPSRAD $1, Y4, Y3                  // k1 = k >> 1
    VPSUBD Y3, Y4, Y4                  // k2 = k - k1
    VPSLLD $23, Y3, Y3
    VPADDD Y12, Y3, Y3                 // 2^k1 bits
    VPSLLD $23, Y4, Y4
    VPADDD Y12, Y4, Y4                 // 2^k2 bits
    VMULPS Y3, Y1, Y1                  // exp(r) * 2^k1
    VMULPS Y4, Y1, Y1                  // * 2^k2 = exp(y)

    VMOVUPS Y1, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ R8
    JNZ  pow32_loop8

pow32_remainder:
    ANDQ $7, CX
    JZ   pow32_done

pow32_scalar:
    MOVL (SI), AX
    VMOVSS (SI), X0

    // log core, scalar (x positive finite; subnormal pre-scale via GPR)
    XORL R9, R9
    MOVL $0x00800000, BX
    CMPL AX, BX
    JGE  pow32_scalar_normal
    VMOVSS log32_two32<>(SB), X2       // 2^32
    VMULSS X2, X0, X0
    VMOVD X0, AX
    MOVL $-32, R9

pow32_scalar_normal:
    MOVL $0x3F350000, BX
    MOVL AX, R10
    SUBL BX, R10                       // tmp
    MOVL R10, R11
    SARL $23, R11                      // e
    ADDL R9, R11
    MOVL $0xFF800000, BX
    ANDL BX, R10
    SUBL R10, AX                       // bits(m)
    VMOVD AX, X2                       // m
    VCVTSI2SSL R11, X3, X3             // e

    VSUBSS X12, X2, X4                 // z = m - 1 (X12 = 1.0)
    VMULSS X4, X4, X5                  // zz
    VMOVSS log32_p0<>(SB), X1
    VFMADD213SS log32_p1<>(SB), X4, X1
    VFMADD213SS log32_p2<>(SB), X4, X1
    VFMADD213SS log32_p3<>(SB), X4, X1
    VFMADD213SS log32_p4<>(SB), X4, X1
    VFMADD213SS log32_p5<>(SB), X4, X1
    VFMADD213SS log32_p6<>(SB), X4, X1
    VFMADD213SS log32_p7<>(SB), X4, X1
    VFMADD213SS log32_p8<>(SB), X4, X1 // P(z)
    VMULSS X4, X5, X6                  // z^3
    VMULSS X6, X1, X1
    VFNMADD231SS exp_half<>(SB), X5, X1 // -= 0.5*zz
    VADDSS X4, X1, X1                  // lnm
    VMULSS log32_ln2lo<>(SB), X3, X5
    VADDSS X1, X5, X5
    VFMADD231SS log32_ln2hi<>(SB), X3, X5 // X5 = ln(x)

    // y = p*ln(x); pre-clamp copy in X6, clamp (X7 = p)
    VMULSS X7, X5, X0
    VMINSS log32_powclamp_hi<>(SB), X0, X0
    VMAXSS log32_powclamp_lo<>(SB), X0, X0

    // exp core, scalar (X8-X15 = exp constants)
    VMULSS X15, X0, X1
    VADDSS X13, X1, X2
    VSUBSS X13, X2, X3                 // k
    VMULSS X14, X3, X4
    VSUBSS X4, X0, X0                  // r
    VMULSS X0, X8, X1
    VADDSS X9, X1, X1
    VMULSS X0, X1, X1
    VADDSS X10, X1, X1
    VMULSS X0, X1, X1
    VADDSS X11, X1, X1
    VMULSS X0, X1, X1
    VADDSS X12, X1, X1
    VMULSS X0, X1, X1
    VADDSS X12, X1, X1                 // exp(r)
    // Split 2^k reconstruction (see the vector body)
    VCVTSS2SI X3, AX                   // k
    MOVL AX, R10
    SARL $1, R10                       // k1 = k >> 1
    SUBL R10, AX                       // k2 = k - k1
    MOVL $0x3F800000, BX
    SHLL $23, R10
    ADDL BX, R10
    VMOVD R10, X4
    VMULSS X4, X1, X1                  // exp(r) * 2^k1
    SHLL $23, AX
    ADDL BX, AX
    VMOVD AX, X4
    VMULSS X4, X1, X1                  // * 2^k2 = exp(y)

    VMOVSS X1, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  pow32_scalar

pow32_done:
    VZEROUPPER
    RET

// func powElemAVX(dst, base, exp []float32)
// Elementwise pow(base[i], exp[i]) = exp(exp[i]*ln(base[i])). Same cores and
// preconditions as powAVX (all bases positive finite, all exponents finite),
// with the exponent loaded per lane instead of broadcast.
TEXT ·powElemAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ base_base+24(FP), SI
    MOVQ exp_base+48(FP), DI

    VMOVUPS exp_c5<>(SB), Y8           // Y8 = 1/120
    VMOVUPS exp_c4<>(SB), Y9           // Y9 = 1/24
    VMOVUPS exp_c3<>(SB), Y10          // Y10 = 1/6
    VMOVUPS exp_c2<>(SB), Y11          // Y11 = 0.5
    VMOVUPS exp_one<>(SB), Y12         // Y12 = 1.0
    VMOVUPS exp_magic<>(SB), Y13       // Y13 = rounding magic
    VMOVUPS exp_ln2<>(SB), Y14         // Y14 = ln(2)
    VMOVUPS exp_log2e<>(SB), Y15       // Y15 = log2(e)

    MOVQ CX, R8
    SHRQ $3, R8
    JZ   powelem32_remainder

powelem32_loop8:
    VMOVUPS (SI), Y0                   // Y0 = base (positive finite)

    // --- log core (see logAVX) ---
    VCMPPS $17, log32_fltmin<>(SB), Y0, Y1
    VMULPS log32_two32<>(SB), Y0, Y2
    VBLENDVPS Y1, Y2, Y0, Y0
    VANDPS log32_negsc<>(SB), Y1, Y1   // ebias
    VPSUBD log32_off<>(SB), Y0, Y2     // tmp
    VPAND log32_expmask<>(SB), Y2, Y3
    VPSUBD Y3, Y0, Y3                  // m
    VPSRAD $23, Y2, Y2
    VCVTDQ2PS Y2, Y2
    VADDPS Y1, Y2, Y2                  // e
    VSUBPS Y12, Y3, Y5                 // z = m - 1
    VMULPS Y5, Y5, Y4                  // zz
    VMOVUPS log32_p0<>(SB), Y3
    VFMADD213PS log32_p1<>(SB), Y5, Y3
    VFMADD213PS log32_p2<>(SB), Y5, Y3
    VFMADD213PS log32_p3<>(SB), Y5, Y3
    VFMADD213PS log32_p4<>(SB), Y5, Y3
    VFMADD213PS log32_p5<>(SB), Y5, Y3
    VFMADD213PS log32_p6<>(SB), Y5, Y3
    VFMADD213PS log32_p7<>(SB), Y5, Y3
    VFMADD213PS log32_p8<>(SB), Y5, Y3 // P(z)
    VMULPS Y5, Y4, Y6                  // z^3
    VMULPS Y6, Y3, Y3
    VFNMADD231PS exp_half<>(SB), Y4, Y3 // -= 0.5*zz
    VADDPS Y5, Y3, Y3                  // lnm
    VMULPS log32_ln2lo<>(SB), Y2, Y4
    VADDPS Y3, Y4, Y4
    VFMADD231PS log32_ln2hi<>(SB), Y2, Y4 // Y4 = ln(base)

    // y = exp[i]*ln(base[i]); pre-clamp copy in Y6, clamp for the exp core
    VMOVUPS (DI), Y7                   // Y7 = exponents (finite)
    VMULPS Y7, Y4, Y0
    VMINPS log32_powclamp_hi<>(SB), Y0, Y0
    VMAXPS log32_powclamp_lo<>(SB), Y0, Y0

    // --- exp core (see expAVX) ---
    VMULPS Y15, Y0, Y1
    VADDPS Y13, Y1, Y2
    VSUBPS Y13, Y2, Y3                 // k
    VMULPS Y14, Y3, Y4
    VSUBPS Y4, Y0, Y0                  // r
    VMULPS Y0, Y8, Y1
    VADDPS Y9, Y1, Y1
    VMULPS Y0, Y1, Y1
    VADDPS Y10, Y1, Y1
    VMULPS Y0, Y1, Y1
    VADDPS Y11, Y1, Y1
    VMULPS Y0, Y1, Y1
    VADDPS Y12, Y1, Y1
    VMULPS Y0, Y1, Y1
    VADDPS Y12, Y1, Y1                 // exp(r)
    // Split 2^k reconstruction (see powAVX)
    VCVTPS2DQ Y3, Y4                   // int(k)
    VPSRAD $1, Y4, Y3                  // k1 = k >> 1
    VPSUBD Y3, Y4, Y4                  // k2 = k - k1
    VPSLLD $23, Y3, Y3
    VPADDD Y12, Y3, Y3                 // 2^k1 bits
    VPSLLD $23, Y4, Y4
    VPADDD Y12, Y4, Y4                 // 2^k2 bits
    VMULPS Y3, Y1, Y1                  // exp(r) * 2^k1
    VMULPS Y4, Y1, Y1                  // * 2^k2 = exp(y)

    VMOVUPS Y1, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ R8
    JNZ  powelem32_loop8

powelem32_remainder:
    ANDQ $7, CX
    JZ   powelem32_done

powelem32_scalar:
    MOVL (SI), AX
    VMOVSS (SI), X0

    XORL R9, R9
    MOVL $0x00800000, BX
    CMPL AX, BX
    JGE  powelem32_scalar_normal
    VMOVSS log32_two32<>(SB), X2       // 2^32
    VMULSS X2, X0, X0
    VMOVD X0, AX
    MOVL $-32, R9

powelem32_scalar_normal:
    MOVL $0x3F350000, BX
    MOVL AX, R10
    SUBL BX, R10                       // tmp
    MOVL R10, R11
    SARL $23, R11                      // e
    ADDL R9, R11
    MOVL $0xFF800000, BX
    ANDL BX, R10
    SUBL R10, AX                       // bits(m)
    VMOVD AX, X2
    VCVTSI2SSL R11, X3, X3             // e

    VSUBSS X12, X2, X4                 // z
    VMULSS X4, X4, X5                  // zz
    VMOVSS log32_p0<>(SB), X1
    VFMADD213SS log32_p1<>(SB), X4, X1
    VFMADD213SS log32_p2<>(SB), X4, X1
    VFMADD213SS log32_p3<>(SB), X4, X1
    VFMADD213SS log32_p4<>(SB), X4, X1
    VFMADD213SS log32_p5<>(SB), X4, X1
    VFMADD213SS log32_p6<>(SB), X4, X1
    VFMADD213SS log32_p7<>(SB), X4, X1
    VFMADD213SS log32_p8<>(SB), X4, X1
    VMULSS X4, X5, X6                  // z^3
    VMULSS X6, X1, X1
    VFNMADD231SS exp_half<>(SB), X5, X1
    VADDSS X4, X1, X1                  // lnm
    VMULSS log32_ln2lo<>(SB), X3, X5
    VADDSS X1, X5, X5
    VFMADD231SS log32_ln2hi<>(SB), X3, X5 // ln(base)

    VMOVSS (DI), X7                    // p
    VMULSS X7, X5, X0
    VMINSS log32_powclamp_hi<>(SB), X0, X0
    VMAXSS log32_powclamp_lo<>(SB), X0, X0

    VMULSS X15, X0, X1
    VADDSS X13, X1, X2
    VSUBSS X13, X2, X3                 // k
    VMULSS X14, X3, X4
    VSUBSS X4, X0, X0                  // r
    VMULSS X0, X8, X1
    VADDSS X9, X1, X1
    VMULSS X0, X1, X1
    VADDSS X10, X1, X1
    VMULSS X0, X1, X1
    VADDSS X11, X1, X1
    VMULSS X0, X1, X1
    VADDSS X12, X1, X1
    VMULSS X0, X1, X1
    VADDSS X12, X1, X1                 // exp(r)
    // Split 2^k reconstruction (see the vector body)
    VCVTSS2SI X3, AX                   // k
    MOVL AX, R10
    SARL $1, R10                       // k1 = k >> 1
    SUBL R10, AX                       // k2 = k - k1
    MOVL $0x3F800000, BX
    SHLL $23, R10
    ADDL BX, R10
    VMOVD R10, X4
    VMULSS X4, X1, X1                  // exp(r) * 2^k1
    SHLL $23, AX
    ADDL BX, AX
    VMOVD AX, X4
    VMULSS X4, X1, X1                  // * 2^k2 = exp(y)

    VMOVSS X1, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  powelem32_scalar

powelem32_done:
    VZEROUPPER
    RET

// Pow clamp bounds: wider than the Exp kernel's +-88 because the pow kernels
// split the 2^k reconstruction (2^(k>>1) * 2^(k-(k>>1))), which covers the
// full float32 result range: overflow goes to +Inf and underflow degrades
// gradually through subnormals to 0, matching math.Pow's classes.
// ln(MaxFloat32) ~ 88.72, ln(min subnormal) ~ -103.28.
DATA log32_powclamp_hi<>+0x00(SB)/4, $0x42b20000  // 89.0
DATA log32_powclamp_hi<>+0x04(SB)/4, $0x42b20000
DATA log32_powclamp_hi<>+0x08(SB)/4, $0x42b20000
DATA log32_powclamp_hi<>+0x0c(SB)/4, $0x42b20000
DATA log32_powclamp_hi<>+0x10(SB)/4, $0x42b20000
DATA log32_powclamp_hi<>+0x14(SB)/4, $0x42b20000
DATA log32_powclamp_hi<>+0x18(SB)/4, $0x42b20000
DATA log32_powclamp_hi<>+0x1c(SB)/4, $0x42b20000
GLOBL log32_powclamp_hi<>(SB), RODATA|NOPTR, $32

DATA log32_powclamp_lo<>+0x00(SB)/4, $0xc2d00000  // -104.0
DATA log32_powclamp_lo<>+0x04(SB)/4, $0xc2d00000
DATA log32_powclamp_lo<>+0x08(SB)/4, $0xc2d00000
DATA log32_powclamp_lo<>+0x0c(SB)/4, $0xc2d00000
DATA log32_powclamp_lo<>+0x10(SB)/4, $0xc2d00000
DATA log32_powclamp_lo<>+0x14(SB)/4, $0xc2d00000
DATA log32_powclamp_lo<>+0x18(SB)/4, $0xc2d00000
DATA log32_powclamp_lo<>+0x1c(SB)/4, $0xc2d00000
GLOBL log32_powclamp_lo<>(SB), RODATA|NOPTR, $32

// minidxones is dup(int32 1): the curIdx increment for VPADDD. The 8-wide kernel
// loads all 32 bytes (Y7); the 4-wide kernel loads the low 16 bytes (X7).
DATA minidxones<>+0x00(SB)/4, $0x00000001
DATA minidxones<>+0x04(SB)/4, $0x00000001
DATA minidxones<>+0x08(SB)/4, $0x00000001
DATA minidxones<>+0x0c(SB)/4, $0x00000001
DATA minidxones<>+0x10(SB)/4, $0x00000001
DATA minidxones<>+0x14(SB)/4, $0x00000001
DATA minidxones<>+0x18(SB)/4, $0x00000001
DATA minidxones<>+0x1c(SB)/4, $0x00000001
GLOBL minidxones<>(SB), RODATA|NOPTR, $32

// minidxrev8 is the VPERMPS control {7,6,5,4,3,2,1,0}: it reverses the eight
// lanes of a YMM in one cross-lane shuffle for the rev == 1 (descending-slide)
// store, so out-lane l = in-lane 7-l.
DATA minidxrev8<>+0x00(SB)/4, $7
DATA minidxrev8<>+0x04(SB)/4, $6
DATA minidxrev8<>+0x08(SB)/4, $5
DATA minidxrev8<>+0x0c(SB)/4, $4
DATA minidxrev8<>+0x10(SB)/4, $3
DATA minidxrev8<>+0x14(SB)/4, $2
DATA minidxrev8<>+0x18(SB)/4, $1
DATA minidxrev8<>+0x1c(SB)/4, $0
GLOBL minidxrev8<>(SB), RODATA|NOPTR, $32

// func minIdxOfSumRows8AVX2(vals []float32, idxs []int32, a, k []float32, rev int)
// Lane-per-row argmin-of-sum for a block of eight rows (one per YMM lane). Each
// lane replays the scalar loop exactly: candidate i (i in [0, n), n = len(a))
// broadcasts a[i], adds it to the eight rows' k values with a single VADDPS (one
// rounding, never fused), and updates (bestVal, bestIdx) only on a strict VCMPPS
// LT_OQ (cand < best). Strict compare keeps first-index-wins on ties, and LT_OQ
// yields false for any NaN operand, so a NaN candidate never displaces the
// incumbent and a NaN incumbent is never beaten; a +Inf pad never beats a finite
// value. The bits match the Go reference lane for lane.
//
// The k pointer is pre-sliced by the dispatcher so k[0] is the first address the
// kernel reads. Both slide signs load k[i:i+8] ascending (the window slides by
// one element per candidate, so k advances 4 bytes, not 32). For slide == +1 the
// dispatcher passes row r's window start, lane l reads k[i+l] = row (r+l)'s
// candidate i, and the union over i in [0,n), l in [0,8) is k indices
// [off, off+n+6] = exactly rows r..r+7's windows, all range-checked by the
// wrapper. For slide == -1 it passes row (r+7)'s window start (off-7 >= 0 because
// the wrapper validated row r+7); the ascending load lands row r+7 in lane 0 and
// row r in lane 7, so rev == 1 reverses both result vectors once at store
// (VPERMPS) to restore lane l == row r+l. Either way every read stays inside the
// union of the eight rows' validated windows, so there is no over-read.
TEXT ·minIdxOfSumRows8AVX2(SB), NOSPLIT, $0-104
    MOVQ vals_base+0(FP), AX
    MOVQ idxs_base+24(FP), BX
    MOVQ a_base+48(FP), SI
    MOVQ a_len+56(FP), CX          // CX = n (dispatcher guarantees n >= 1)
    MOVQ k_base+72(FP), DI
    MOVQ rev+96(FP), DX

    // Candidate 0: seed bestVal = a[0] + k[0:8], bestIdx = 0, curIdx = 0.
    VBROADCASTSS (SI), Y3          // dup(a[0])
    VMOVUPS (DI), Y2               // k[0:8] (explicit unaligned load)
    VADDPS Y2, Y3, Y0              // bestVal = a[0] + k[0:8]
    VPXOR Y1, Y1, Y1               // bestIdx = 0
    VPXOR Y6, Y6, Y6               // curIdx = 0
    VMOVUPS minidxones<>(SB), Y7   // Y7 = dup(1), the curIdx increment

    ADDQ $4, SI                    // a[1]
    ADDQ $4, DI                    // k[1:9]
    MOVQ CX, R9
    DECQ R9                        // R9 = n-1 remaining candidates
    JZ   minidxofsumrows8_avx2_store

minidxofsumrows8_avx2_loop:
    VBROADCASTSS (SI), Y3          // dup(a[i])
    VMOVUPS (DI), Y2               // k[i:i+8]
    VADDPS Y2, Y3, Y4              // cand = a[i] + k[i:i+8]
    VPADDD Y7, Y6, Y6              // curIdx++ (before the selects read it)
    VCMPPS $0x11, Y0, Y4, Y5       // Y5 = mask (cand LT_OQ best)
    VBLENDVPS Y5, Y4, Y0, Y0       // bestVal = mask ? cand : bestVal
    VBLENDVPS Y5, Y6, Y1, Y1       // bestIdx = mask ? curIdx : bestIdx
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ R9
    JNZ  minidxofsumrows8_avx2_loop

minidxofsumrows8_avx2_store:
    TESTQ DX, DX
    JZ    minidxofsumrows8_avx2_write
    // rev == 1: reverse all eight lanes of both results in one shuffle each.
    VMOVUPS minidxrev8<>(SB), Y8   // control {7,6,5,4,3,2,1,0}
    VPERMPS Y0, Y8, Y0             // bestVal lanes reversed
    VPERMPS Y1, Y8, Y1             // bestIdx lanes reversed

minidxofsumrows8_avx2_write:
    VMOVUPS Y0, (AX)               // vals[0:8]
    VMOVUPS Y1, (BX)               // idxs[0:8]
    VZEROUPPER
    RET

// func minIdxOfSumRows4AVX2(vals []float32, idxs []int32, a, k []float32, rev int)
// The 4-wide (XMM) sibling of minIdxOfSumRows8AVX2: same lane-per-row argmin-of-
// sum, four rows per block, used for the 4-row sub-block a row count leaves after
// the 8-wide blocks (a count near 11 to 17 can leave a 4-or-more remainder,
// e.g. 12 -> 8+4, 14 -> 8+4+2, hence this kernel).
// The k union is n+3 wide; rev == 1 reverses the four lanes with VPERMILPS imm
// [3,2,1,0]. This kernel is VEX.128-only (no YMM touched), so VZEROUPPER is not
// required before RET.
TEXT ·minIdxOfSumRows4AVX2(SB), NOSPLIT, $0-104
    MOVQ vals_base+0(FP), AX
    MOVQ idxs_base+24(FP), BX
    MOVQ a_base+48(FP), SI
    MOVQ a_len+56(FP), CX          // CX = n (dispatcher guarantees n >= 1)
    MOVQ k_base+72(FP), DI
    MOVQ rev+96(FP), DX

    // Candidate 0: seed bestVal = a[0] + k[0:4], bestIdx = 0, curIdx = 0.
    VBROADCASTSS (SI), X3          // dup(a[0])
    VMOVUPS (DI), X2               // k[0:4] (explicit unaligned load)
    VADDPS X2, X3, X0              // bestVal = a[0] + k[0:4]
    VPXOR X1, X1, X1               // bestIdx = 0
    VPXOR X6, X6, X6               // curIdx = 0
    VMOVUPS minidxones<>(SB), X7   // X7 = dup(1) (low 16 bytes), the increment

    ADDQ $4, SI                    // a[1]
    ADDQ $4, DI                    // k[1:5]
    MOVQ CX, R9
    DECQ R9                        // R9 = n-1 remaining candidates
    JZ   minidxofsumrows4_avx2_store

minidxofsumrows4_avx2_loop:
    VBROADCASTSS (SI), X3          // dup(a[i])
    VMOVUPS (DI), X2               // k[i:i+4]
    VADDPS X2, X3, X4              // cand = a[i] + k[i:i+4]
    VPADDD X7, X6, X6              // curIdx++ (before the selects read it)
    VCMPPS $0x11, X0, X4, X5       // X5 = mask (cand LT_OQ best)
    VBLENDVPS X5, X4, X0, X0       // bestVal = mask ? cand : bestVal
    VBLENDVPS X5, X6, X1, X1       // bestIdx = mask ? curIdx : bestIdx
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ R9
    JNZ  minidxofsumrows4_avx2_loop

minidxofsumrows4_avx2_store:
    TESTQ DX, DX
    JZ    minidxofsumrows4_avx2_write
    // rev == 1: reverse the four lanes of both results, [3,2,1,0].
    VPERMILPS $0x1B, X0, X0        // bestVal lanes reversed
    VPERMILPS $0x1B, X1, X1        // bestIdx lanes reversed

minidxofsumrows4_avx2_write:
    VMOVUPS X0, (AX)               // vals[0:4]
    VMOVUPS X1, (BX)               // idxs[0:4]
    RET
