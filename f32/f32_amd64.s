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
    VXORPS Z0, Z0, Z0          // acc0
    VXORPS Z3, Z3, Z3          // acc1
    VXORPS Z4, Z4, Z4          // acc2
    VXORPS Z5, Z5, Z5          // acc3

    // Process 64 elements per iteration (4 vectors × 16 floats)
    MOVQ CX, AX
    SHRQ $6, AX                // len / 64
    JZ   dot32_512_loop16_check

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
    VXORPS Z0, Z0, Z0          // acc0
    VXORPS Z3, Z3, Z3          // acc1
    VXORPS Z4, Z4, Z4          // acc2
    VXORPS Z5, Z5, Z5          // acc3

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
    // First vector (8 elements)
    VMOVUPS (R10), Y1          // Y1 = d[i:i+8]
    VMOVUPS (R9), Y2           // Y2 = c[i:i+8]
    VMOVUPS (R8), Y3           // Y3 = b[i:i+8]
    VMOVUPS (DI), Y4           // Y4 = a[i:i+8]
    VMOVUPS (SI), Y5           // Y5 = hist[i:i+8]

    // Horner's method: coef = a + x*(b + x*(c + x*d))
    VFMADD231PS Y1, Y7, Y2     // Y2 = d*x + c
    VFMADD231PS Y2, Y7, Y3     // Y3 = (d*x+c)*x + b
    VFMADD231PS Y3, Y7, Y4     // Y4 = coef
    VFMADD231PS Y5, Y4, Y0     // acc0 += hist * coef

    // Second vector (8 elements)
    VMOVUPS 32(R10), Y1        // Y1 = d[i+8:i+16]
    VMOVUPS 32(R9), Y2         // Y2 = c[i+8:i+16]
    VMOVUPS 32(R8), Y3         // Y3 = b[i+8:i+16]
    VMOVUPS 32(DI), Y4         // Y4 = a[i+8:i+16]
    VMOVUPS 32(SI), Y5         // Y5 = hist[i+8:i+16]

    VFMADD231PS Y1, Y7, Y2     // Y2 = d*x + c
    VFMADD231PS Y2, Y7, Y3     // Y3 = (d*x+c)*x + b
    VFMADD231PS Y3, Y7, Y4     // Y4 = coef
    VFMADD231PS Y5, Y4, Y6     // acc1 += hist * coef

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
