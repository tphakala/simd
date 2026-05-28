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

exp32_scalar:
    VMOVSS (SI), X0                     // X0 = x

    // Clamp
    VMOVSS exp_clamp_hi<>(SB), X1
    VMOVSS exp_clamp_lo<>(SB), X2
    VMINSS X1, X0, X0
    VMAXSS X2, X0, X0

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

    // Set up reverse pointers: zRe[n-8], zIm[n-8]
    MOVQ CX, R9
    SUBQ $8, R9                      // R9 = n - 8
    SHLQ $2, R9                      // R9 = (n-8) * 4 bytes
    MOVQ DI, R9                      // R9 = zRe base
    ADDQ CX, R9
    SUBQ $8, R9
    SHLQ $2, R9                      // Wrong, redo this

    // Recalculate: R9 = &zRe[n-8], R10 = &zIm[n-8]
    MOVQ CX, R14                     // R14 = n
    SUBQ $8, R14                     // R14 = n - 8
    SHLQ $2, R14                     // R14 = (n-8) * 4 = byte offset
    MOVQ DI, R9
    ADDQ R14, R9                     // R9 = &zRe[n-8]
    MOVQ R8, R10
    ADDQ R14, R10                    // R10 = &zIm[n-8]

    // Offset forward pointers to start at index 1
    ADDQ $4, DI                      // DI = &zRe[1]
    ADDQ $4, R8                      // R8 = &zIm[1]
    ADDQ $4, DX                      // DX = &outRe[1]
    ADDQ $4, SI                      // SI = &outIm[1]

    // Load constants
    // Reverse permutation mask: [7, 6, 5, 4, 3, 2, 1, 0]
    MOVQ $0x0001000200030004, R14
    MOVQ R14, X14
    MOVQ $0x0005000600070000, R14    // Wrong format, need 32-bit indices

    // Actually, VPERMPS uses 32-bit indices. Let me use a different approach.
    // Create reverse mask in YMM register
    VPCMPEQD Y15, Y15, Y15           // Y15 = all 1s (will use for sign flip)
    VPSRLD $1, Y15, Y15              // Y15 = 0x7FFFFFFF (clear sign bit for abs mask)

    // For reverse permutation, we'll construct it differently
    // Use VPERMPD for 64-bit permute then shuffle within lanes
    // Actually, let's load the permutation mask from memory (cleaner)

    // Broadcast 0.5 (0x3F000000 = 0.5f)
    MOVL $0x3F000000, R14
    MOVD R14, X13
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
    MOVQ AX, R14
    SHLQ $2, R14                     // R14 = k * 4 bytes
    ADDQ R14, DX                     // DX = &outRe[k]
    ADDQ R14, SI                     // SI = &outIm[k]
    ADDQ R14, DI                     // DI = &zRe[k]
    ADDQ R14, R8                     // R8 = &zIm[k]

    // Twiddle offset is (k-1)
    DECQ AX
    MOVQ AX, R14
    SHLQ $2, R14
    ADDQ R14, R11                    // R11 = &twRe[k-1]
    ADDQ R14, R12                    // R12 = &twIm[k-1]
    INCQ AX                          // Restore AX = k

realfft_scalar:
    // Calculate mirror index: nk = n - k
    MOVQ CX, R14
    SUBQ AX, R14                     // R14 = n - k = nk

    // Load Z[k]
    VMOVSS (DI), X0                  // X0 = zRe[k]
    VMOVSS (R8), X1                  // X1 = zIm[k]

    // Load conj(Z[n-k])
    MOVQ zRe_base+48(FP), R15
    MOVQ R14, R9
    SHLQ $2, R9
    ADDQ R9, R15
    VMOVSS (R15), X2                 // X2 = zRe[nk]

    MOVQ zIm_base+72(FP), R15
    ADDQ R9, R15
    VMOVSS (R15), X3                 // X3 = zIm[nk]

    // Load 0.5 constant (0x3F000000 = 0.5f)
    MOVL $0x3F000000, R14
    MOVD R14, X13

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
