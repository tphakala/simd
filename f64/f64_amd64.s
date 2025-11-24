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
// Optimized with 4 independent accumulators to hide FMA latency (4 cycles).
// Processes 16 doubles per iteration for better instruction-level parallelism.
// Handles mismatched slice lengths: uses min(len(a), len(b)).
TEXT ·dotProductAVX(SB), NOSPLIT, $0-56
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_len+32(FP), AX
    CMPQ AX, CX
    CMOVQLT AX, CX             // CX = min(len(a), len(b))
    MOVQ b_base+24(FP), DI

    // Initialize 4 independent accumulators
    VXORPD Y0, Y0, Y0          // acc0
    VXORPD Y3, Y3, Y3          // acc1
    VXORPD Y4, Y4, Y4          // acc2
    VXORPD Y5, Y5, Y5          // acc3

    // Process 16 elements per iteration (4 vectors × 4 doubles)
    MOVQ CX, AX
    SHRQ $4, AX                // len / 16
    JZ   dot_avx_loop4_check

dot_avx_loop16:
    // Load and FMA for acc0
    VMOVUPD (SI), Y1
    VMOVUPD (DI), Y2
    VFMADD231PD Y1, Y2, Y0

    // Load and FMA for acc1
    VMOVUPD 32(SI), Y1
    VMOVUPD 32(DI), Y2
    VFMADD231PD Y1, Y2, Y3

    // Load and FMA for acc2
    VMOVUPD 64(SI), Y1
    VMOVUPD 64(DI), Y2
    VFMADD231PD Y1, Y2, Y4

    // Load and FMA for acc3
    VMOVUPD 96(SI), Y1
    VMOVUPD 96(DI), Y2
    VFMADD231PD Y1, Y2, Y5

    ADDQ $128, SI
    ADDQ $128, DI
    DECQ AX
    JNZ  dot_avx_loop16

    // Combine accumulators: Y0 = Y0 + Y3 + Y4 + Y5
    VADDPD Y3, Y0, Y0
    VADDPD Y4, Y0, Y0
    VADDPD Y5, Y0, Y0

dot_avx_loop4_check:
    // Handle remaining 4-element chunks
    ANDQ $15, CX
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
    // Reduce Y0 to scalar
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0

    // Handle remaining 1-3 elements
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
//
// Division Latency Hiding via 4x Loop Unrolling
// =============================================
// VDIVPD has high latency (~13 cycles) but good throughput (~8 cycles).
// This means the CPU can have multiple divisions in-flight simultaneously.
// By issuing 4 independent VDIVPD instructions per iteration, we allow
// the CPU's out-of-order execution to overlap their execution, achieving
// closer to the theoretical throughput limit.
//
// Without unrolling: Each iteration waits for previous VDIVPD to complete
// With 4x unrolling: 4 VDIVPDs can execute in parallel across iterations
//
// Timing (Alder Lake P-core, from uops.info):
//   - VDIVPD YMM latency: ≤13 cycles
//   - VDIVPD YMM throughput: 8.00 cycles (can start new div every 8 cycles)
//
TEXT ·divAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    // Process 16 elements per iteration (4 vectors × 4 doubles)
    // This allows 4 independent VDIVPD operations to be in-flight
    MOVQ CX, AX
    SHRQ $4, AX                // len / 16
    JZ   div_avx_loop4_check

div_avx_loop16:
    // Load 4 vectors from a (128 bytes total)
    VMOVUPD 0(SI), Y0
    VMOVUPD 32(SI), Y2
    VMOVUPD 64(SI), Y4
    VMOVUPD 96(SI), Y6
    // Load 4 vectors from b (128 bytes total)
    VMOVUPD 0(DI), Y1
    VMOVUPD 32(DI), Y3
    VMOVUPD 64(DI), Y5
    VMOVUPD 96(DI), Y7
    // Issue 4 independent divisions - no data dependencies between them
    // CPU can execute these in parallel using pipelined divider unit
    VDIVPD Y1, Y0, Y0
    VDIVPD Y3, Y2, Y2
    VDIVPD Y5, Y4, Y4
    VDIVPD Y7, Y6, Y6
    // Store results (128 bytes total)
    VMOVUPD Y0, 0(DX)
    VMOVUPD Y2, 32(DX)
    VMOVUPD Y4, 64(DX)
    VMOVUPD Y6, 96(DX)
    ADDQ $128, SI
    ADDQ $128, DI
    ADDQ $128, DX
    DECQ AX
    JNZ  div_avx_loop16

div_avx_loop4_check:
    // Handle remaining 4-element chunks
    ANDQ $15, CX
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
// Optimized with 4 independent accumulators to hide ADD latency (4 cycles).
TEXT ·sumAVX(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    // Initialize 4 independent accumulators
    VXORPD Y0, Y0, Y0          // acc0
    VXORPD Y3, Y3, Y3          // acc1
    VXORPD Y4, Y4, Y4          // acc2
    VXORPD Y5, Y5, Y5          // acc3

    // Process 16 elements per iteration (4 vectors × 4 doubles)
    MOVQ CX, AX
    SHRQ $4, AX                // len / 16
    JZ   sum_avx_loop4_check

sum_avx_loop16:
    VADDPD (SI), Y0, Y0
    VADDPD 32(SI), Y3, Y3
    VADDPD 64(SI), Y4, Y4
    VADDPD 96(SI), Y5, Y5
    ADDQ $128, SI
    DECQ AX
    JNZ  sum_avx_loop16

    // Combine accumulators
    VADDPD Y3, Y0, Y0
    VADDPD Y4, Y0, Y0
    VADDPD Y5, Y0, Y0

sum_avx_loop4_check:
    // Handle remaining 4-element chunks
    ANDQ $15, CX
    MOVQ CX, AX
    SHRQ $2, AX
    JZ   sum_avx_remainder

sum_avx_loop4:
    VADDPD (SI), Y0, Y0
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
//
// Square Root Latency Hiding via 4x Loop Unrolling
// =================================================
// VSQRTPD has high latency (~13 cycles) but good throughput (~9 cycles).
// Similar to division, we can have multiple sqrt operations in-flight.
// By issuing 4 independent VSQRTPD instructions per iteration, the CPU's
// pipelined execution unit can process them concurrently.
//
// Timing (Alder Lake P-core, from uops.info):
//   - VSQRTPD YMM latency: ≤13 cycles
//   - VSQRTPD YMM throughput: 9.00 cycles (can start new sqrt every 9 cycles)
//
TEXT ·sqrtAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Process 16 elements per iteration (4 vectors × 4 doubles)
    // This allows 4 independent VSQRTPD operations to overlap
    MOVQ CX, AX
    SHRQ $4, AX                // len / 16
    JZ   sqrt_avx_loop4_check

sqrt_avx_loop16:
    // Load 4 vectors from source (128 bytes total)
    VMOVUPD 0(SI), Y0
    VMOVUPD 32(SI), Y2
    VMOVUPD 64(SI), Y4
    VMOVUPD 96(SI), Y6
    // Issue 4 independent sqrts - no data dependencies between them
    // CPU can execute these in parallel using pipelined sqrt unit
    VSQRTPD Y0, Y0
    VSQRTPD Y2, Y2
    VSQRTPD Y4, Y4
    VSQRTPD Y6, Y6
    // Store results (128 bytes total)
    VMOVUPD Y0, 0(DX)
    VMOVUPD Y2, 32(DX)
    VMOVUPD Y4, 64(DX)
    VMOVUPD Y6, 96(DX)
    ADDQ $128, SI
    ADDQ $128, DX
    DECQ AX
    JNZ  sqrt_avx_loop16

sqrt_avx_loop4_check:
    // Handle remaining 4-element chunks
    ANDQ $15, CX
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
//
// Reciprocal via Division with Latency Hiding
// ============================================
// Computes 1.0/a[i] using VDIVPD. While approximate reciprocal instructions
// exist (VRCPPS for float32), they only provide ~12-bit precision. For full
// IEEE 754 precision, we use actual division with 1.0 as the dividend.
//
// Same latency hiding strategy as divAVX: 4x unrolling allows 4 independent
// VDIVPD operations to overlap, hiding the ~13 cycle latency.
//
// Timing (Alder Lake P-core, from uops.info):
//   - VDIVPD YMM latency: ≤13 cycles
//   - VDIVPD YMM throughput: 8.00 cycles
//
TEXT ·reciprocalAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Broadcast 1.0 to Y3 (preserved across all iterations)
    // 0x3FF0000000000000 is 1.0 in IEEE 754 double precision
    MOVQ $0x3FF0000000000000, AX
    MOVQ AX, X3
    VBROADCASTSD X3, Y3

    // Process 16 elements per iteration (4 vectors × 4 doubles)
    // This allows 4 independent VDIVPD operations to be in-flight
    MOVQ CX, AX
    SHRQ $4, AX                // len / 16
    JZ   recip_avx_loop4_check

recip_avx_loop16:
    // Load 4 vectors from source (128 bytes total)
    VMOVUPD 0(SI), Y0
    VMOVUPD 32(SI), Y1
    VMOVUPD 64(SI), Y2
    VMOVUPD 96(SI), Y4
    // Issue 4 independent divisions: dst[i] = 1.0 / a[i]
    // No data dependencies - CPU can execute in parallel
    VDIVPD Y0, Y3, Y0
    VDIVPD Y1, Y3, Y1
    VDIVPD Y2, Y3, Y2
    VDIVPD Y4, Y3, Y4
    // Store results (128 bytes total)
    VMOVUPD Y0, 0(DX)
    VMOVUPD Y1, 32(DX)
    VMOVUPD Y2, 64(DX)
    VMOVUPD Y4, 96(DX)
    ADDQ $128, SI
    ADDQ $128, DX
    DECQ AX
    JNZ  recip_avx_loop16

recip_avx_loop4_check:
    // Handle remaining 4-element chunks
    ANDQ $15, CX
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
// Uses 4 accumulators to avoid FMA dependency chain stalls
TEXT ·varianceAVX(SB), NOSPLIT, $0-40
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    VBROADCASTSD mean+24(FP), Y2

    // Initialize 4 accumulators for parallel reduction
    VXORPD Y0, Y0, Y0
    VXORPD Y3, Y3, Y3
    VXORPD Y4, Y4, Y4
    VXORPD Y5, Y5, Y5

    // Process 16 elements (4 vectors) per iteration
    MOVQ CX, AX
    SHRQ $4, AX           // len / 16
    JZ   var_avx_loop4

var_avx_loop16:
    VMOVUPD 0(SI), Y1
    VSUBPD Y2, Y1, Y1
    VFMADD231PD Y1, Y1, Y0

    VMOVUPD 32(SI), Y1
    VSUBPD Y2, Y1, Y1
    VFMADD231PD Y1, Y1, Y3

    VMOVUPD 64(SI), Y1
    VSUBPD Y2, Y1, Y1
    VFMADD231PD Y1, Y1, Y4

    VMOVUPD 96(SI), Y1
    VSUBPD Y2, Y1, Y1
    VFMADD231PD Y1, Y1, Y5

    ADDQ $128, SI
    DECQ AX
    JNZ  var_avx_loop16

    // Combine accumulators
    VADDPD Y3, Y0, Y0
    VADDPD Y5, Y4, Y4
    VADDPD Y4, Y0, Y0

var_avx_loop4:
    // Handle remaining groups of 4
    MOVQ CX, AX
    ANDQ $12, AX          // (len % 16) & ~3 = remaining complete vectors
    SHRQ $2, AX
    JZ   var_avx_remainder

var_avx_loop4_inner:
    VMOVUPD (SI), Y1
    VSUBPD Y2, Y1, Y1
    VFMADD231PD Y1, Y1, Y0
    ADDQ $32, SI
    DECQ AX
    JNZ  var_avx_loop4_inner

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
// Optimized with 4 independent accumulators to hide FMA latency (4 cycles).
TEXT ·euclideanDistanceAVX(SB), NOSPLIT, $0-56
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    // Initialize 4 independent accumulators
    VXORPD Y0, Y0, Y0          // acc0
    VXORPD Y3, Y3, Y3          // acc1
    VXORPD Y4, Y4, Y4          // acc2
    VXORPD Y5, Y5, Y5          // acc3

    // Process 16 elements per iteration
    MOVQ CX, AX
    SHRQ $4, AX                // len / 16
    JZ   euclid_avx_loop4_check

euclid_avx_loop16:
    VMOVUPD (SI), Y1
    VMOVUPD (DI), Y2
    VSUBPD Y2, Y1, Y1
    VFMADD231PD Y1, Y1, Y0

    VMOVUPD 32(SI), Y1
    VMOVUPD 32(DI), Y2
    VSUBPD Y2, Y1, Y1
    VFMADD231PD Y1, Y1, Y3

    VMOVUPD 64(SI), Y1
    VMOVUPD 64(DI), Y2
    VSUBPD Y2, Y1, Y1
    VFMADD231PD Y1, Y1, Y4

    VMOVUPD 96(SI), Y1
    VMOVUPD 96(DI), Y2
    VSUBPD Y2, Y1, Y1
    VFMADD231PD Y1, Y1, Y5

    ADDQ $128, SI
    ADDQ $128, DI
    DECQ AX
    JNZ  euclid_avx_loop16

    // Combine accumulators
    VADDPD Y3, Y0, Y0
    VADDPD Y4, Y0, Y0
    VADDPD Y5, Y0, Y0

euclid_avx_loop4_check:
    ANDQ $15, CX
    MOVQ CX, AX
    SHRQ $2, AX
    JZ   euclid_avx_remainder

euclid_avx_loop4:
    VMOVUPD (SI), Y1
    VMOVUPD (DI), Y2
    VSUBPD Y2, Y1, Y1
    VFMADD231PD Y1, Y1, Y0
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
// Optimized with 4 independent accumulators to hide FMA latency.
// Processes 32 doubles per iteration (4 vectors × 8 doubles).
// Handles mismatched slice lengths: uses min(len(a), len(b)).
TEXT ·dotProductAVX512(SB), NOSPLIT, $0-56
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_len+32(FP), AX
    CMPQ AX, CX
    CMOVQLT AX, CX             // CX = min(len(a), len(b))
    MOVQ b_base+24(FP), DI

    // Initialize 4 independent accumulators
    VXORPD Z0, Z0, Z0          // acc0
    VXORPD Z3, Z3, Z3          // acc1
    VXORPD Z4, Z4, Z4          // acc2
    VXORPD Z5, Z5, Z5          // acc3

    // Process 32 elements per iteration (4 vectors × 8 doubles)
    MOVQ CX, AX
    SHRQ $5, AX                // len / 32
    JZ   dot_512_loop8_check

dot_512_loop32:
    // Load and FMA for acc0
    VMOVUPD (SI), Z1
    VMOVUPD (DI), Z2
    VFMADD231PD Z1, Z2, Z0

    // Load and FMA for acc1
    VMOVUPD 64(SI), Z1
    VMOVUPD 64(DI), Z2
    VFMADD231PD Z1, Z2, Z3

    // Load and FMA for acc2
    VMOVUPD 128(SI), Z1
    VMOVUPD 128(DI), Z2
    VFMADD231PD Z1, Z2, Z4

    // Load and FMA for acc3
    VMOVUPD 192(SI), Z1
    VMOVUPD 192(DI), Z2
    VFMADD231PD Z1, Z2, Z5

    ADDQ $256, SI
    ADDQ $256, DI
    DECQ AX
    JNZ  dot_512_loop32

    // Combine accumulators: Z0 = Z0 + Z3 + Z4 + Z5
    VADDPD Z3, Z0, Z0
    VADDPD Z4, Z0, Z0
    VADDPD Z5, Z0, Z0

dot_512_loop8_check:
    // Handle remaining 8-element chunks
    ANDQ $31, CX
    MOVQ CX, AX
    SHRQ $3, AX
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
// Optimized with 4x unrolling to hide VDIVPD latency.
TEXT ·divAVX512(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    // Process 32 elements per iteration (4 vectors × 8 doubles)
    MOVQ CX, AX
    SHRQ $5, AX                // len / 32
    JZ   div_512_loop8_check

div_512_loop32:
    // Load 4 vectors from a
    VMOVUPD 0(SI), Z0
    VMOVUPD 64(SI), Z2
    VMOVUPD 128(SI), Z4
    VMOVUPD 192(SI), Z6
    // Load 4 vectors from b
    VMOVUPD 0(DI), Z1
    VMOVUPD 64(DI), Z3
    VMOVUPD 128(DI), Z5
    VMOVUPD 192(DI), Z7
    // Issue 4 independent divisions
    VDIVPD Z1, Z0, Z0
    VDIVPD Z3, Z2, Z2
    VDIVPD Z5, Z4, Z4
    VDIVPD Z7, Z6, Z6
    // Store results
    VMOVUPD Z0, 0(DX)
    VMOVUPD Z2, 64(DX)
    VMOVUPD Z4, 128(DX)
    VMOVUPD Z6, 192(DX)
    ADDQ $256, SI
    ADDQ $256, DI
    ADDQ $256, DX
    DECQ AX
    JNZ  div_512_loop32

div_512_loop8_check:
    // Handle remaining 8-element chunks
    ANDQ $31, CX
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
// Optimized with 4 independent accumulators to hide ADD latency (4 cycles).
TEXT ·sumAVX512(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    // Initialize 4 independent accumulators
    VXORPD Z0, Z0, Z0          // acc0
    VXORPD Z3, Z3, Z3          // acc1
    VXORPD Z4, Z4, Z4          // acc2
    VXORPD Z5, Z5, Z5          // acc3

    // Process 32 elements per iteration (4 vectors × 8 doubles)
    MOVQ CX, AX
    SHRQ $5, AX                // len / 32
    JZ   sum_512_loop8_check

sum_512_loop32:
    VADDPD (SI), Z0, Z0
    VADDPD 64(SI), Z3, Z3
    VADDPD 128(SI), Z4, Z4
    VADDPD 192(SI), Z5, Z5
    ADDQ $256, SI
    DECQ AX
    JNZ  sum_512_loop32

    // Combine accumulators
    VADDPD Z3, Z0, Z0
    VADDPD Z4, Z0, Z0
    VADDPD Z5, Z0, Z0

sum_512_loop8_check:
    ANDQ $31, CX
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   sum_512_remainder

sum_512_loop8:
    VADDPD (SI), Z0, Z0
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
// Optimized with 4x unrolling to hide VSQRTPD latency.
TEXT ·sqrtAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Process 32 elements per iteration (4 vectors × 8 doubles)
    MOVQ CX, AX
    SHRQ $5, AX                // len / 32
    JZ   sqrt_512_loop8_check

sqrt_512_loop32:
    // Load 4 vectors
    VMOVUPD 0(SI), Z0
    VMOVUPD 64(SI), Z2
    VMOVUPD 128(SI), Z4
    VMOVUPD 192(SI), Z6
    // Issue 4 independent sqrts
    VSQRTPD Z0, Z0
    VSQRTPD Z2, Z2
    VSQRTPD Z4, Z4
    VSQRTPD Z6, Z6
    // Store results
    VMOVUPD Z0, 0(DX)
    VMOVUPD Z2, 64(DX)
    VMOVUPD Z4, 128(DX)
    VMOVUPD Z6, 192(DX)
    ADDQ $256, SI
    ADDQ $256, DX
    DECQ AX
    JNZ  sqrt_512_loop32

sqrt_512_loop8_check:
    // Handle remaining 8-element chunks
    ANDQ $31, CX
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
// Optimized with 4x unrolling to hide VDIVPD latency.
TEXT ·reciprocalAVX512(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    // Load 1.0 for division
    MOVQ $0x3FF0000000000000, AX
    MOVQ AX, X3
    VBROADCASTSD X3, Z3

    // Process 32 elements per iteration (4 vectors × 8 doubles)
    MOVQ CX, AX
    SHRQ $5, AX                // len / 32
    JZ   recip_512_loop8_check

recip_512_loop32:
    // Load 4 vectors
    VMOVUPD 0(SI), Z0
    VMOVUPD 64(SI), Z1
    VMOVUPD 128(SI), Z2
    VMOVUPD 192(SI), Z4
    // Issue 4 independent divisions
    VDIVPD Z0, Z3, Z0
    VDIVPD Z1, Z3, Z1
    VDIVPD Z2, Z3, Z2
    VDIVPD Z4, Z3, Z4
    // Store results
    VMOVUPD Z0, 0(DX)
    VMOVUPD Z1, 64(DX)
    VMOVUPD Z2, 128(DX)
    VMOVUPD Z4, 192(DX)
    ADDQ $256, SI
    ADDQ $256, DX
    DECQ AX
    JNZ  recip_512_loop32

recip_512_loop8_check:
    // Handle remaining 8-element chunks
    ANDQ $31, CX
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
// Uses 4 accumulators to avoid FMA dependency chain stalls
TEXT ·varianceAVX512(SB), NOSPLIT, $0-40
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    VBROADCASTSD mean+24(FP), Z2

    // Initialize 4 accumulators for parallel reduction
    VXORPD Z0, Z0, Z0
    VXORPD Z3, Z3, Z3
    VXORPD Z4, Z4, Z4
    VXORPD Z5, Z5, Z5

    // Process 32 elements (4 vectors of 8) per iteration
    MOVQ CX, AX
    SHRQ $5, AX           // len / 32
    JZ   var_512_loop8

var_512_loop32:
    VMOVUPD 0(SI), Z1
    VSUBPD Z2, Z1, Z1
    VFMADD231PD Z1, Z1, Z0

    VMOVUPD 64(SI), Z1
    VSUBPD Z2, Z1, Z1
    VFMADD231PD Z1, Z1, Z3

    VMOVUPD 128(SI), Z1
    VSUBPD Z2, Z1, Z1
    VFMADD231PD Z1, Z1, Z4

    VMOVUPD 192(SI), Z1
    VSUBPD Z2, Z1, Z1
    VFMADD231PD Z1, Z1, Z5

    ADDQ $256, SI
    DECQ AX
    JNZ  var_512_loop32

    // Combine accumulators
    VADDPD Z3, Z0, Z0
    VADDPD Z5, Z4, Z4
    VADDPD Z4, Z0, Z0

var_512_loop8:
    // Handle remaining groups of 8
    MOVQ CX, AX
    ANDQ $24, AX          // (len % 32) & ~7 = remaining complete vectors
    SHRQ $3, AX
    JZ   var_512_remainder

var_512_loop8_inner:
    VMOVUPD (SI), Z1
    VSUBPD Z2, Z1, Z1
    VFMADD231PD Z1, Z1, Z0
    ADDQ $64, SI
    DECQ AX
    JNZ  var_512_loop8_inner

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
// Optimized with 4 independent accumulators to hide FMA latency (4 cycles).
TEXT ·euclideanDistanceAVX512(SB), NOSPLIT, $0-56
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    // Initialize 4 independent accumulators
    VXORPD Z0, Z0, Z0          // acc0
    VXORPD Z3, Z3, Z3          // acc1
    VXORPD Z4, Z4, Z4          // acc2
    VXORPD Z5, Z5, Z5          // acc3

    // Process 32 elements per iteration
    MOVQ CX, AX
    SHRQ $5, AX                // len / 32
    JZ   euclid_512_loop8_check

euclid_512_loop32:
    VMOVUPD (SI), Z1
    VMOVUPD (DI), Z2
    VSUBPD Z2, Z1, Z1
    VFMADD231PD Z1, Z1, Z0

    VMOVUPD 64(SI), Z1
    VMOVUPD 64(DI), Z2
    VSUBPD Z2, Z1, Z1
    VFMADD231PD Z1, Z1, Z3

    VMOVUPD 128(SI), Z1
    VMOVUPD 128(DI), Z2
    VSUBPD Z2, Z1, Z1
    VFMADD231PD Z1, Z1, Z4

    VMOVUPD 192(SI), Z1
    VMOVUPD 192(DI), Z2
    VSUBPD Z2, Z1, Z1
    VFMADD231PD Z1, Z1, Z5

    ADDQ $256, SI
    ADDQ $256, DI
    DECQ AX
    JNZ  euclid_512_loop32

    // Combine accumulators
    VADDPD Z3, Z0, Z0
    VADDPD Z4, Z0, Z0
    VADDPD Z5, Z0, Z0

euclid_512_loop8_check:
    ANDQ $31, CX
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
// Handles mismatched slice lengths: uses min(len(a), len(b)).
TEXT ·dotProductSSE2(SB), NOSPLIT, $0-56
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_len+32(FP), AX
    CMPQ AX, CX
    CMOVQLT AX, CX             // CX = min(len(a), len(b))
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
// Uses 4 accumulators to avoid dependency chain stalls
TEXT ·varianceSSE2(SB), NOSPLIT, $0-40
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVSD mean+24(FP), X2
    SHUFPD $0, X2, X2

    // Initialize 4 accumulators for parallel reduction
    XORPD X0, X0
    XORPD X3, X3
    XORPD X4, X4
    XORPD X5, X5

    // Process 8 elements (4 vectors of 2) per iteration
    MOVQ CX, AX
    SHRQ $3, AX           // len / 8
    JZ   var_sse2_loop2

var_sse2_loop8:
    MOVUPD 0(SI), X1
    SUBPD X2, X1
    MULPD X1, X1
    ADDPD X1, X0

    MOVUPD 16(SI), X1
    SUBPD X2, X1
    MULPD X1, X1
    ADDPD X1, X3

    MOVUPD 32(SI), X1
    SUBPD X2, X1
    MULPD X1, X1
    ADDPD X1, X4

    MOVUPD 48(SI), X1
    SUBPD X2, X1
    MULPD X1, X1
    ADDPD X1, X5

    ADDQ $64, SI
    DECQ AX
    JNZ  var_sse2_loop8

    // Combine accumulators
    ADDPD X3, X0
    ADDPD X5, X4
    ADDPD X4, X0

var_sse2_loop2:
    // Handle remaining groups of 2
    MOVQ CX, AX
    ANDQ $6, AX           // (len % 8) & ~1 = remaining complete vectors
    SHRQ $1, AX
    JZ   var_sse2_remainder

var_sse2_loop2_inner:
    MOVUPD (SI), X1
    SUBPD X2, X1
    MULPD X1, X1
    ADDPD X1, X0
    ADDQ $16, SI
    DECQ AX
    JNZ  var_sse2_loop2_inner

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

// ============================================================================
// INTERLEAVE/DEINTERLEAVE IMPLEMENTATIONS
// ============================================================================

// func interleave2AVX(dst, a, b []float64)
// Interleaves two slices: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], dst[3]=b[1], ...
// Input a has n elements, b has n elements, dst has 2n elements
TEXT ·interleave2AVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX    // dst pointer
    MOVQ a_base+24(FP), SI     // a pointer
    MOVQ a_len+32(FP), CX      // n = len(a)
    MOVQ b_base+48(FP), DI     // b pointer

    // Process 4 pairs at a time (8 output elements)
    MOVQ CX, AX
    SHRQ $2, AX                // AX = n / 4
    JZ   interleave2_avx_remainder

interleave2_avx_loop4:
    // Load 4 from a: Y0 = [a0, a1, a2, a3]
    VMOVUPD (SI), Y0
    // Load 4 from b: Y1 = [b0, b1, b2, b3]
    VMOVUPD (DI), Y1

    // Unpack within 128-bit lanes
    // VUNPCKLPD: Y2 = [a0, b0, a2, b2]
    VUNPCKLPD Y1, Y0, Y2
    // VUNPCKHPD: Y3 = [a1, b1, a3, b3]
    VUNPCKHPD Y1, Y0, Y3

    // Permute to get final order
    // Y4 = [a0, b0, a1, b1] (low halves)
    VPERM2F128 $0x20, Y3, Y2, Y4
    // Y5 = [a2, b2, a3, b3] (high halves)
    VPERM2F128 $0x31, Y3, Y2, Y5

    // Store 8 elements to dst
    VMOVUPD Y4, (DX)
    VMOVUPD Y5, 32(DX)

    ADDQ $32, SI               // a += 4 * 8
    ADDQ $32, DI               // b += 4 * 8
    ADDQ $64, DX               // dst += 8 * 8
    DECQ AX
    JNZ  interleave2_avx_loop4

interleave2_avx_remainder:
    ANDQ $3, CX
    JZ   interleave2_avx_done

interleave2_avx_scalar:
    VMOVSD (SI), X0
    VMOVSD (DI), X1
    VMOVSD X0, (DX)
    VMOVSD X1, 8(DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $16, DX
    DECQ CX
    JNZ  interleave2_avx_scalar

interleave2_avx_done:
    VZEROUPPER
    RET

// func deinterleave2AVX(a, b, src []float64)
// Deinterleaves: a[0]=src[0], b[0]=src[1], a[1]=src[2], b[1]=src[3], ...
TEXT ·deinterleave2AVX(SB), NOSPLIT, $0-72
    MOVQ a_base+0(FP), DX      // a pointer
    MOVQ a_len+8(FP), CX       // n = len(a)
    MOVQ b_base+24(FP), R8     // b pointer
    MOVQ src_base+48(FP), SI   // src pointer

    // Process 4 pairs at a time
    MOVQ CX, AX
    SHRQ $2, AX
    JZ   deinterleave2_avx_remainder

deinterleave2_avx_loop4:
    // Load 8 interleaved elements
    // Y0 = [a0, b0, a1, b1]
    VMOVUPD (SI), Y0
    // Y1 = [a2, b2, a3, b3]
    VMOVUPD 32(SI), Y1

    // Permute to group a's and b's
    // Y2 = [a0, b0, a2, b2] (low halves of Y0 and Y1)
    VPERM2F128 $0x20, Y1, Y0, Y2
    // Y3 = [a1, b1, a3, b3] (high halves of Y0 and Y1)
    VPERM2F128 $0x31, Y1, Y0, Y3

    // Unpack to separate a's and b's
    // Y4 = [a0, a1, a2, a3]
    VUNPCKLPD Y3, Y2, Y4
    // Y5 = [b0, b1, b2, b3]
    VUNPCKHPD Y3, Y2, Y5

    // Store results
    VMOVUPD Y4, (DX)
    VMOVUPD Y5, (R8)

    ADDQ $64, SI               // src += 8 * 8
    ADDQ $32, DX               // a += 4 * 8
    ADDQ $32, R8               // b += 4 * 8
    DECQ AX
    JNZ  deinterleave2_avx_loop4

deinterleave2_avx_remainder:
    ANDQ $3, CX
    JZ   deinterleave2_avx_done

deinterleave2_avx_scalar:
    VMOVSD (SI), X0
    VMOVSD 8(SI), X1
    VMOVSD X0, (DX)
    VMOVSD X1, (R8)
    ADDQ $16, SI
    ADDQ $8, DX
    ADDQ $8, R8
    DECQ CX
    JNZ  deinterleave2_avx_scalar

deinterleave2_avx_done:
    VZEROUPPER
    RET

// func interleave2SSE2(dst, a, b []float64)
TEXT ·interleave2SSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ a_base+24(FP), SI
    MOVQ a_len+32(FP), CX
    MOVQ b_base+48(FP), DI

    // Process 2 pairs at a time (4 output elements)
    MOVQ CX, AX
    SHRQ $1, AX
    JZ   interleave2_sse2_remainder

interleave2_sse2_loop2:
    // X0 = [a0, a1]
    MOVUPD (SI), X0
    // X1 = [b0, b1]
    MOVUPD (DI), X1

    // X2 = [a0, b0]
    MOVAPD X0, X2
    UNPCKLPD X1, X2
    // X3 = [a1, b1]
    MOVAPD X0, X3
    UNPCKHPD X1, X3

    MOVUPD X2, (DX)
    MOVUPD X3, 16(DX)

    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  interleave2_sse2_loop2

interleave2_sse2_remainder:
    ANDQ $1, CX
    JZ   interleave2_sse2_done

    MOVSD (SI), X0
    MOVSD (DI), X1
    MOVSD X0, (DX)
    MOVSD X1, 8(DX)

interleave2_sse2_done:
    RET

// func deinterleave2SSE2(a, b, src []float64)
TEXT ·deinterleave2SSE2(SB), NOSPLIT, $0-72
    MOVQ a_base+0(FP), DX
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), R8
    MOVQ src_base+48(FP), SI

    // Process 2 pairs at a time
    MOVQ CX, AX
    SHRQ $1, AX
    JZ   deinterleave2_sse2_remainder

deinterleave2_sse2_loop2:
    // X0 = [a0, b0]
    MOVUPD (SI), X0
    // X1 = [a1, b1]
    MOVUPD 16(SI), X1

    // X2 = [a0, a1]
    MOVAPD X0, X2
    UNPCKLPD X1, X2
    // X3 = [b0, b1]
    MOVAPD X0, X3
    UNPCKHPD X1, X3

    MOVUPD X2, (DX)
    MOVUPD X3, (R8)

    ADDQ $32, SI
    ADDQ $16, DX
    ADDQ $16, R8
    DECQ AX
    JNZ  deinterleave2_sse2_loop2

deinterleave2_sse2_remainder:
    ANDQ $1, CX
    JZ   deinterleave2_sse2_done

    MOVSD (SI), X0
    MOVSD 8(SI), X1
    MOVSD X0, (DX)
    MOVSD X1, (R8)

deinterleave2_sse2_done:
    RET

// ============================================================================
// ADDSCALED - dst[i] += alpha * s[i] (AXPY operation)
// ============================================================================

// func addScaledAVX(dst []float64, alpha float64, s []float64)
TEXT ·addScaledAVX(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    VBROADCASTSD alpha+24(FP), Y2
    MOVQ s_base+32(FP), SI

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   addscaled_avx_remainder

addscaled_avx_loop4:
    VMOVUPD (DX), Y0
    VMOVUPD (SI), Y1
    VFMADD231PD Y1, Y2, Y0    // Y0 = Y0 + Y1 * Y2
    VMOVUPD Y0, (DX)
    ADDQ $32, DX
    ADDQ $32, SI
    DECQ AX
    JNZ  addscaled_avx_loop4

addscaled_avx_remainder:
    ANDQ $3, CX
    JZ   addscaled_avx_done

addscaled_avx_scalar:
    VMOVSD (DX), X0
    VMOVSD (SI), X1
    VFMADD231SD X1, X2, X0    // X0 = X0 + X1 * X2
    VMOVSD X0, (DX)
    ADDQ $8, DX
    ADDQ $8, SI
    DECQ CX
    JNZ  addscaled_avx_scalar

addscaled_avx_done:
    VZEROUPPER
    RET

// func addScaledAVX512(dst []float64, alpha float64, s []float64)
TEXT ·addScaledAVX512(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    VBROADCASTSD alpha+24(FP), Z2
    MOVQ s_base+32(FP), SI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   addscaled_avx512_remainder

addscaled_avx512_loop8:
    VMOVUPD (DX), Z0
    VMOVUPD (SI), Z1
    VFMADD231PD Z1, Z2, Z0    // Z0 = Z0 + Z1 * Z2
    VMOVUPD Z0, (DX)
    ADDQ $64, DX
    ADDQ $64, SI
    DECQ AX
    JNZ  addscaled_avx512_loop8

addscaled_avx512_remainder:
    ANDQ $7, CX
    JZ   addscaled_avx512_done

addscaled_avx512_scalar:
    VMOVSD (DX), X0
    VMOVSD (SI), X1
    VMOVSD alpha+24(FP), X2
    VFMADD231SD X1, X2, X0
    VMOVSD X0, (DX)
    ADDQ $8, DX
    ADDQ $8, SI
    DECQ CX
    JNZ  addscaled_avx512_scalar

addscaled_avx512_done:
    VZEROUPPER
    RET

// func addScaledSSE2(dst []float64, alpha float64, s []float64)
TEXT ·addScaledSSE2(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVSD alpha+24(FP), X2
    UNPCKLPD X2, X2           // Broadcast alpha to both lanes
    MOVQ s_base+32(FP), SI

    MOVQ CX, AX
    SHRQ $1, AX
    JZ   addscaled_sse2_remainder

addscaled_sse2_loop2:
    MOVUPD (DX), X0
    MOVUPD (SI), X1
    MULPD X2, X1              // X1 = s * alpha
    ADDPD X1, X0              // X0 = dst + (s * alpha)
    MOVUPD X0, (DX)
    ADDQ $16, DX
    ADDQ $16, SI
    DECQ AX
    JNZ  addscaled_sse2_loop2

addscaled_sse2_remainder:
    ANDQ $1, CX
    JZ   addscaled_sse2_done

    MOVSD (DX), X0
    MOVSD (SI), X1
    MULSD X2, X1
    ADDSD X1, X0
    MOVSD X0, (DX)

addscaled_sse2_done:
    RET

// ============================================================================
// CUBIC INTERPOLATION DOT PRODUCT
// ============================================================================

// func cubicInterpDotAVX(hist, a, b, c, d []float64, x float64) float64
// Computes: Σ hist[i] * (a[i] + x*(b[i] + x*(c[i] + x*d[i])))
// Uses Horner's method for polynomial evaluation with FMA.
//
// Frame layout (5 slices + 1 float64 + 1 return):
//   hist: base+0, len+8, cap+16
//   a:    base+24, len+32, cap+40
//   b:    base+48, len+56, cap+64
//   c:    base+72, len+80, cap+88
//   d:    base+96, len+104, cap+112
//   x:    +120
//   ret:  +128
TEXT ·cubicInterpDotAVX(SB), NOSPLIT, $0-136
    MOVQ hist_base+0(FP), SI   // SI = hist pointer
    MOVQ hist_len+8(FP), CX    // CX = length
    MOVQ a_base+24(FP), DI     // DI = a pointer
    MOVQ b_base+48(FP), R8     // R8 = b pointer
    MOVQ c_base+72(FP), R9     // R9 = c pointer
    MOVQ d_base+96(FP), R10    // R10 = d pointer

    // Broadcast x to all lanes of Y7
    VBROADCASTSD x+120(FP), Y7

    // Initialize accumulator to zero
    VXORPD Y0, Y0, Y0          // Y0 = accumulator

    // Process 4 elements per iteration (one YMM register)
    MOVQ CX, AX
    SHRQ $2, AX                // AX = len / 4
    JZ   cubic_avx_remainder

cubic_avx_loop4:
    // Load coefficient vectors
    VMOVUPD (R10), Y1          // Y1 = d[i:i+4]
    VMOVUPD (R9), Y2           // Y2 = c[i:i+4]
    VMOVUPD (R8), Y3           // Y3 = b[i:i+4]
    VMOVUPD (DI), Y4           // Y4 = a[i:i+4]
    VMOVUPD (SI), Y5           // Y5 = hist[i:i+4]

    // Horner's method: coef = a + x*(b + x*(c + x*d))
    // Step 1: Y2 = c + x*d (using VFMADD231: dst = src1*src2 + dst)
    VFMADD231PD Y1, Y7, Y2     // Y2 = d*x + c
    // Step 2: Y3 = b + x*(c + x*d)
    VFMADD231PD Y2, Y7, Y3     // Y3 = (d*x+c)*x + b
    // Step 3: Y4 = a + x*(b + x*(c + x*d))
    VFMADD231PD Y3, Y7, Y4     // Y4 = ((d*x+c)*x+b)*x + a = coef

    // Accumulate: acc += hist * coef
    VFMADD231PD Y5, Y4, Y0     // Y0 = hist * coef + Y0

    // Advance pointers
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, R8
    ADDQ $32, R9
    ADDQ $32, R10
    DECQ AX
    JNZ  cubic_avx_loop4

cubic_avx_remainder:
    // Reduce Y0 to scalar first (before scalar ops that zero upper bits)
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0         // X0[0] = sum of all 4 elements

    // Handle remaining 1-3 elements
    ANDQ $3, CX
    JZ   cubic_avx_done

cubic_avx_scalar:
    // Load single elements
    VMOVSD (R10), X1           // X1 = d[i]
    VMOVSD (R9), X2            // X2 = c[i]
    VMOVSD (R8), X3            // X3 = b[i]
    VMOVSD (DI), X4            // X4 = a[i]
    VMOVSD (SI), X5            // X5 = hist[i]
    VMOVSD x+120(FP), X6       // X6 = x

    // Horner's method for scalar
    VFMADD231SD X1, X6, X2     // X2 = d*x + c
    VFMADD231SD X2, X6, X3     // X3 = (d*x+c)*x + b
    VFMADD231SD X3, X6, X4     // X4 = coef

    // Accumulate
    VFMADD231SD X5, X4, X0     // X0 = hist * coef + X0

    // Advance pointers
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, R8
    ADDQ $8, R9
    ADDQ $8, R10
    DECQ CX
    JNZ  cubic_avx_scalar

cubic_avx_done:
    VMOVSD X0, ret+128(FP)
    VZEROUPPER
    RET

// Constants for sigmoid (float64)
DATA sigmoid_half64<>+0x00(SB)/8, $0x3FE0000000000000  // 0.5
DATA sigmoid_half64<>+0x08(SB)/8, $0x3FE0000000000000
DATA sigmoid_half64<>+0x10(SB)/8, $0x3FE0000000000000
DATA sigmoid_half64<>+0x18(SB)/8, $0x3FE0000000000000
GLOBL sigmoid_half64<>(SB), RODATA|NOPTR, $32

DATA sigmoid_one64<>+0x00(SB)/8, $0x3FF0000000000000  // 1.0
DATA sigmoid_one64<>+0x08(SB)/8, $0x3FF0000000000000
DATA sigmoid_one64<>+0x10(SB)/8, $0x3FF0000000000000
DATA sigmoid_one64<>+0x18(SB)/8, $0x3FF0000000000000
GLOBL sigmoid_one64<>(SB), RODATA|NOPTR, $32

// Note: absf64mask is already defined at top of file

// func sigmoidAVX(dst, src []float64)
// Implements fast sigmoid approximation: σ(x) ≈ 0.5 + 0.5 * x / (1 + |x|)
// This approximation is SIMD-friendly and commonly used in neural networks.
TEXT ·sigmoidAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DI
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    // Load constants (use unaligned loads to avoid alignment faults)
    VMOVUPD sigmoid_half64<>(SB), Y8   // Y8 = 0.5
    VMOVUPD sigmoid_one64<>(SB), Y9    // Y9 = 1.0
    VMOVUPD absf64mask<>(SB), Y10      // Y10 = abs mask

    // Process 4 elements per iteration
    MOVQ CX, AX
    SHRQ $2, AX
    JZ   sigmoid64_remainder

sigmoid64_loop4:
    VMOVUPD (SI), Y0               // Y0 = x
    VANDPD Y10, Y0, Y1             // Y1 = |x|
    VADDPD Y9, Y1, Y2              // Y2 = 1 + |x|
    VDIVPD Y2, Y0, Y3              // Y3 = x / (1 + |x|)
    VMULPD Y8, Y3, Y4              // Y4 = 0.5 * x / (1 + |x|)
    VADDPD Y8, Y4, Y5              // Y5 = 0.5 + 0.5 * x / (1 + |x|)
    VMOVUPD Y5, (DI)               // store result

    ADDQ $32, SI
    ADDQ $32, DI
    DECQ AX
    JNZ  sigmoid64_loop4

sigmoid64_remainder:
    ANDQ $3, CX
    JZ   sigmoid64_done

sigmoid64_scalar:
    VMOVSD (SI), X0                // X0 = x
    VANDPD X10, X0, X1             // X1 = |x|
    VADDSD X9, X1, X2              // X2 = 1 + |x|
    VDIVSD X2, X0, X3              // X3 = x / (1 + |x|)
    VMULSD X8, X3, X4              // X4 = 0.5 * x / (1 + |x|)
    VADDSD X8, X4, X5              // X5 = 0.5 + 0.5 * x / (1 + |x|)
    VMOVSD X5, (DI)                // store result

    ADDQ $8, SI
    ADDQ $8, DI
    DECQ CX
    JNZ  sigmoid64_scalar

sigmoid64_done:
    VZEROUPPER
    RET

// func clampScaleAVX(dst, src []float64, minVal, maxVal, scale float64)
// Performs fused clamp and scale: dst[i] = (clamp(src[i], minVal, maxVal) - minVal) * scale
TEXT ·clampScaleAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI
    VBROADCASTSD minVal+48(FP), Y1     // Y1 = minVal (broadcast to all lanes)
    VBROADCASTSD maxVal+56(FP), Y2     // Y2 = maxVal
    VBROADCASTSD scale+64(FP), Y3      // Y3 = scale

    // Process 4 elements per iteration
    MOVQ CX, AX
    SHRQ $2, AX                        // len / 4
    JZ   clampscale64_remainder

clampscale64_loop4:
    VMOVUPD (SI), Y0                   // Y0 = src[i]
    VMAXPD Y0, Y1, Y0                  // Y0 = max(src[i], minVal)
    VMINPD Y0, Y2, Y0                  // Y0 = min(max(src[i], minVal), maxVal)
    VSUBPD Y1, Y0, Y0                  // Y0 = clamped - minVal
    VMULPD Y3, Y0, Y0                  // Y0 = (clamped - minVal) * scale
    VMOVUPD Y0, (DX)                   // store result
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  clampscale64_loop4

clampscale64_remainder:
    ANDQ $3, CX                        // remainder = len % 4
    JZ   clampscale64_done
    VMOVSD minVal+48(FP), X1           // X1 = minVal (scalar)
    VMOVSD maxVal+56(FP), X2           // X2 = maxVal
    VMOVSD scale+64(FP), X3            // X3 = scale

clampscale64_scalar:
    VMOVSD (SI), X0                    // X0 = src[i]
    VMAXSD X0, X1, X0                  // X0 = max(src[i], minVal)
    VMINSD X0, X2, X0                  // X0 = min(max(src[i], minVal), maxVal)
    VSUBSD X1, X0, X0                  // X0 = clamped - minVal
    VMULSD X3, X0, X0                  // X0 = (clamped - minVal) * scale
    VMOVSD X0, (DX)                    // store result
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  clampscale64_scalar

clampscale64_done:
    VZEROUPPER
    RET

// func reluAVX(dst, src []float64)
// Computes ReLU: dst[i] = max(0, src[i])
TEXT ·reluAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    // Create zero vector
    VXORPD Y1, Y1, Y1                  // Y1 = 0

    // Process 4 elements per iteration
    MOVQ CX, AX
    SHRQ $2, AX                        // len / 4
    JZ   relu64_remainder

relu64_loop4:
    VMOVUPD (SI), Y0                   // Y0 = src[i]
    VMAXPD Y0, Y1, Y0                  // Y0 = max(src[i], 0)
    VMOVUPD Y0, (DX)                   // store result
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  relu64_loop4

relu64_remainder:
    ANDQ $3, CX                        // remainder = len % 4
    JZ   relu64_done
    VXORPD X1, X1, X1                  // X1 = 0 (scalar)

relu64_scalar:
    VMOVSD (SI), X0                    // X0 = src[i]
    VMAXSD X0, X1, X0                  // X0 = max(src[i], 0)
    VMOVSD X0, (DX)                    // store result
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  relu64_scalar

relu64_done:
    VZEROUPPER
    RET

// func tanhAVX(dst, src []float64)
// Computes fast tanh approximation with saturation:
// tanh(x) ≈ x / (1 + |x|) for |x| <= 2.5, else ±1.0
TEXT ·tanhAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    // Load constants (use unaligned loads to avoid alignment faults)
    VMOVUPD sigmoid_one64<>(SB), Y2    // Y2 = 1.0 (reuse existing constant)
    VMOVUPD absf64mask<>(SB), Y3       // Y3 = abs mask

    // Create 2.5 threshold
    MOVQ $0x4004000000000000, AX       // 2.5 in float64
    VMOVQ AX, X4
    VBROADCASTSD X4, Y4                // Y4 = {2.5, 2.5, 2.5, 2.5}

    // Create sign mask (0x8000000000000000)
    MOVQ $0x8000000000000000, AX
    VMOVQ AX, X5
    VBROADCASTSD X5, Y5                // Y5 = sign mask

    // Process 4 elements per iteration
    MOVQ CX, AX
    SHRQ $2, AX                        // len / 4
    JZ   tanh64_remainder

tanh64_loop4:
    VMOVUPD (SI), Y0                   // Y0 = x
    VANDPD Y3, Y0, Y1                  // Y1 = |x|

    // Compute approximation
    VADDPD Y2, Y1, Y6                  // Y6 = 1 + |x|
    VDIVPD Y6, Y0, Y7                  // Y7 = x / (1 + |x|) (approximation)

    // Create saturated value: copysign(1.0, x)
    VANDPD Y5, Y0, Y6                  // Y6 = sign bit of x
    VORPD Y2, Y6, Y6                   // Y6 = 1.0 with sign of x (saturated value)

    // Compare |x| > 2.5 and select
    VCMPPD $0x1E, Y4, Y1, Y1           // Y1 = mask (|x| > 2.5), using NLE (not less or equal)
    VBLENDVPD Y1, Y6, Y7, Y0           // Y0 = blend(approx, saturated, mask)

    VMOVUPD Y0, (DX)                   // store result
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  tanh64_loop4

tanh64_remainder:
    ANDQ $3, CX                        // remainder = len % 4
    JZ   tanh64_done

tanh64_scalar:
    VMOVSD (SI), X0                    // X0 = x
    VANDPD X3, X0, X1                  // X1 = |x|

    // Compare |x| with 2.5
    VUCOMISD X4, X1
    JBE tanh64_scalar_approx

    // Saturate: return ±1.0
    VANDPD X5, X0, X6                  // X6 = sign bit of x
    VORPD X2, X6, X0                   // X0 = 1.0 with sign of x
    JMP tanh64_scalar_store

tanh64_scalar_approx:
    VADDSD X2, X1, X1                  // X1 = 1 + |x|
    VDIVSD X1, X0, X0                  // X0 = x / (1 + |x|)

tanh64_scalar_store:
    VMOVSD X0, (DX)                    // store result
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  tanh64_scalar

tanh64_done:
    VZEROUPPER
    RET
