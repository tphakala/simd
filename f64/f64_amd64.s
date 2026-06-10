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

// Constants for Round (half away from zero): result = trunc(x) + copysign(1, x)
// when |x - trunc(x)| >= 0.5, otherwise result = trunc(x). Ties (|frac|==0.5)
// round away from zero, matching math.Round.
DATA roundf64_signmask<>+0x00(SB)/8, $0x8000000000000000
DATA roundf64_signmask<>+0x08(SB)/8, $0x8000000000000000
DATA roundf64_signmask<>+0x10(SB)/8, $0x8000000000000000
DATA roundf64_signmask<>+0x18(SB)/8, $0x8000000000000000
GLOBL roundf64_signmask<>(SB), RODATA|NOPTR, $32

DATA roundf64_absmask<>+0x00(SB)/8, $0x7fffffffffffffff
DATA roundf64_absmask<>+0x08(SB)/8, $0x7fffffffffffffff
DATA roundf64_absmask<>+0x10(SB)/8, $0x7fffffffffffffff
DATA roundf64_absmask<>+0x18(SB)/8, $0x7fffffffffffffff
GLOBL roundf64_absmask<>(SB), RODATA|NOPTR, $32

DATA roundf64_half<>+0x00(SB)/8, $0x3fe0000000000000
DATA roundf64_half<>+0x08(SB)/8, $0x3fe0000000000000
DATA roundf64_half<>+0x10(SB)/8, $0x3fe0000000000000
DATA roundf64_half<>+0x18(SB)/8, $0x3fe0000000000000
GLOBL roundf64_half<>(SB), RODATA|NOPTR, $32

DATA roundf64_one<>+0x00(SB)/8, $0x3ff0000000000000
DATA roundf64_one<>+0x08(SB)/8, $0x3ff0000000000000
DATA roundf64_one<>+0x10(SB)/8, $0x3ff0000000000000
DATA roundf64_one<>+0x18(SB)/8, $0x3ff0000000000000
GLOBL roundf64_one<>(SB), RODATA|NOPTR, $32

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
    VPXORQ Z0, Z0, Z0          // acc0
    VPXORQ Z3, Z3, Z3          // acc1
    VPXORQ Z4, Z4, Z4          // acc2
    VPXORQ Z5, Z5, Z5          // acc3

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

// func dotProduct4AVX(results, row0, row1, row2, row3, vec *float64, n int)
// Computes four dot products against the same vec, reusing each vec load.
// Two accumulator banks per row (a/b) hide FMA latency; 16 doubles per row per
// main iteration (4 chunks of 4 lanes). Tail reuses dotProductAVX's reduction.
TEXT ·dotProduct4AVX(SB), NOSPLIT, $0-56
    MOVQ results+0(FP), DX
    MOVQ row0+8(FP), SI
    MOVQ row1+16(FP), R8
    MOVQ row2+24(FP), R9
    MOVQ row3+32(FP), R10
    MOVQ vec+40(FP), DI
    MOVQ n+48(FP), CX

    VXORPD Y0, Y0, Y0          // acc0a
    VXORPD Y3, Y3, Y3          // acc1a
    VXORPD Y4, Y4, Y4          // acc2a
    VXORPD Y5, Y5, Y5          // acc3a
    VXORPD Y6, Y6, Y6          // acc0b
    VXORPD Y7, Y7, Y7          // acc1b
    VXORPD Y8, Y8, Y8          // acc2b
    VXORPD Y9, Y9, Y9          // acc3b

    MOVQ CX, AX
    SHRQ $4, AX                // n / 16
    JZ   dot4_avx_loop4_check

dot4_avx_loop16:
    VMOVUPD (DI), Y1
    VMOVUPD (SI), Y2
    VFMADD231PD Y1, Y2, Y0
    VMOVUPD (R8), Y2
    VFMADD231PD Y1, Y2, Y3
    VMOVUPD (R9), Y2
    VFMADD231PD Y1, Y2, Y4
    VMOVUPD (R10), Y2
    VFMADD231PD Y1, Y2, Y5

    VMOVUPD 32(DI), Y1
    VMOVUPD 32(SI), Y2
    VFMADD231PD Y1, Y2, Y6
    VMOVUPD 32(R8), Y2
    VFMADD231PD Y1, Y2, Y7
    VMOVUPD 32(R9), Y2
    VFMADD231PD Y1, Y2, Y8
    VMOVUPD 32(R10), Y2
    VFMADD231PD Y1, Y2, Y9

    VMOVUPD 64(DI), Y1
    VMOVUPD 64(SI), Y2
    VFMADD231PD Y1, Y2, Y0
    VMOVUPD 64(R8), Y2
    VFMADD231PD Y1, Y2, Y3
    VMOVUPD 64(R9), Y2
    VFMADD231PD Y1, Y2, Y4
    VMOVUPD 64(R10), Y2
    VFMADD231PD Y1, Y2, Y5

    VMOVUPD 96(DI), Y1
    VMOVUPD 96(SI), Y2
    VFMADD231PD Y1, Y2, Y6
    VMOVUPD 96(R8), Y2
    VFMADD231PD Y1, Y2, Y7
    VMOVUPD 96(R9), Y2
    VFMADD231PD Y1, Y2, Y8
    VMOVUPD 96(R10), Y2
    VFMADD231PD Y1, Y2, Y9

    ADDQ $128, DI
    ADDQ $128, SI
    ADDQ $128, R8
    ADDQ $128, R9
    ADDQ $128, R10
    DECQ AX
    JNZ  dot4_avx_loop16

dot4_avx_loop4_check:
    ANDQ $15, CX
    MOVQ CX, AX
    SHRQ $2, AX
    JZ   dot4_avx_reduce

dot4_avx_loop4:
    VMOVUPD (DI), Y1
    VMOVUPD (SI), Y2
    VFMADD231PD Y1, Y2, Y0
    VMOVUPD (R8), Y2
    VFMADD231PD Y1, Y2, Y3
    VMOVUPD (R9), Y2
    VFMADD231PD Y1, Y2, Y4
    VMOVUPD (R10), Y2
    VFMADD231PD Y1, Y2, Y5
    ADDQ $32, DI
    ADDQ $32, SI
    ADDQ $32, R8
    ADDQ $32, R9
    ADDQ $32, R10
    DECQ AX
    JNZ  dot4_avx_loop4

dot4_avx_reduce:
    VADDPD Y6, Y0, Y0
    VADDPD Y7, Y3, Y3
    VADDPD Y8, Y4, Y4
    VADDPD Y9, Y5, Y5

    // Reduce acc0 into X0.
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0

    // Reduce acc1 into X3.
    VEXTRACTF128 $1, Y3, X1
    VADDPD X1, X3, X3
    VHADDPD X3, X3, X3

    // Reduce acc2 into X4.
    VEXTRACTF128 $1, Y4, X1
    VADDPD X1, X4, X4
    VHADDPD X4, X4, X4

    // Reduce acc3 into X5.
    VEXTRACTF128 $1, Y5, X1
    VADDPD X1, X5, X5
    VHADDPD X5, X5, X5

    ANDQ $3, CX
    JZ   dot4_avx_done

dot4_avx_scalar:
    VMOVSD (DI), X1
    VMOVSD (SI), X2
    VFMADD231SD X1, X2, X0
    VMOVSD (R8), X2
    VFMADD231SD X1, X2, X3
    VMOVSD (R9), X2
    VFMADD231SD X1, X2, X4
    VMOVSD (R10), X2
    VFMADD231SD X1, X2, X5
    ADDQ $8, DI
    ADDQ $8, SI
    ADDQ $8, R8
    ADDQ $8, R9
    ADDQ $8, R10
    DECQ CX
    JNZ  dot4_avx_scalar

dot4_avx_done:
    VMOVSD X0, (DX)
    VMOVSD X3, 8(DX)
    VMOVSD X4, 16(DX)
    VMOVSD X5, 24(DX)
    VZEROUPPER
    RET

// func autocorrStep4AVX(acc, broadcast, window *float64, count int)
// Steady-region accumulation for four autocorrelation lags at once. Y0 holds
// the four seeded accumulators (lanes = lags base..base+3). Each iteration
// broadcasts x[i], loads the four ascending window samples, reverses them with
// VPERMPD so lane j carries x[i-(base+j)], multiplies, and adds. Separate
// VMULPD + VADDPD (no FMA) keep each lane's sum bit-identical to the scalar
// reference. broadcast and window advance one float64 (8 bytes) per iteration.
TEXT ·autocorrStep4AVX(SB), NOSPLIT, $0-32
    MOVQ acc+0(FP), DX
    MOVQ broadcast+8(FP), SI
    MOVQ window+16(FP), DI
    MOVQ count+24(FP), CX

    VMOVUPD (DX), Y0           // Y0 = seeded accumulators (lags base..base+3)
    TESTQ CX, CX
    JZ    autocorr4_done

autocorr4_loop:
    VBROADCASTSD (SI), Y1      // Y1 = x[i] in all four lanes
    VMOVUPD (DI), Y2           // Y2 = [x[i-base-3], x[i-base-2], x[i-base-1], x[i-base]]
    VPERMPD $0x1B, Y2, Y2      // reverse -> [x[i-base], x[i-base-1], x[i-base-2], x[i-base-3]]
    VMULPD Y1, Y2, Y2          // Y2 = x[i] * window         (separate multiply)
    VADDPD Y2, Y0, Y0          // Y0 += Y2                   (separate add, no FMA)
    ADDQ $8, SI
    ADDQ $8, DI
    DECQ CX
    JNZ  autocorr4_loop

    VMOVUPD Y0, (DX)           // store the four lag accumulators back

autocorr4_done:
    VZEROUPPER
    RET

// func dotProduct4AVX512(results, row0, row1, row2, row3, vec *float64, n int)
// Computes four dot products against the same vec, reusing each vec load.
// Two accumulator banks per row (a/b); 32 doubles per row per main iteration
// (4 chunks of 8 lanes). Reduction stays on AVX512F (VEXTRACTF64X4, no DQ),
// matching the AVX512F+VL dispatch gate.
TEXT ·dotProduct4AVX512(SB), NOSPLIT, $0-56
    MOVQ results+0(FP), DX
    MOVQ row0+8(FP), SI
    MOVQ row1+16(FP), R8
    MOVQ row2+24(FP), R9
    MOVQ row3+32(FP), R10
    MOVQ vec+40(FP), DI
    MOVQ n+48(FP), CX

    VPXORQ Z0, Z0, Z0          // acc0a
    VPXORQ Z3, Z3, Z3          // acc1a
    VPXORQ Z4, Z4, Z4          // acc2a
    VPXORQ Z5, Z5, Z5          // acc3a
    VPXORQ Z6, Z6, Z6          // acc0b
    VPXORQ Z7, Z7, Z7          // acc1b
    VPXORQ Z8, Z8, Z8          // acc2b
    VPXORQ Z9, Z9, Z9          // acc3b

    MOVQ CX, AX
    SHRQ $5, AX                // n / 32
    JZ   dot4_512_loop8_check

dot4_512_loop32:
    VMOVUPD (DI), Z1
    VMOVUPD (SI), Z2
    VFMADD231PD Z1, Z2, Z0
    VMOVUPD (R8), Z2
    VFMADD231PD Z1, Z2, Z3
    VMOVUPD (R9), Z2
    VFMADD231PD Z1, Z2, Z4
    VMOVUPD (R10), Z2
    VFMADD231PD Z1, Z2, Z5

    VMOVUPD 64(DI), Z1
    VMOVUPD 64(SI), Z2
    VFMADD231PD Z1, Z2, Z6
    VMOVUPD 64(R8), Z2
    VFMADD231PD Z1, Z2, Z7
    VMOVUPD 64(R9), Z2
    VFMADD231PD Z1, Z2, Z8
    VMOVUPD 64(R10), Z2
    VFMADD231PD Z1, Z2, Z9

    VMOVUPD 128(DI), Z1
    VMOVUPD 128(SI), Z2
    VFMADD231PD Z1, Z2, Z0
    VMOVUPD 128(R8), Z2
    VFMADD231PD Z1, Z2, Z3
    VMOVUPD 128(R9), Z2
    VFMADD231PD Z1, Z2, Z4
    VMOVUPD 128(R10), Z2
    VFMADD231PD Z1, Z2, Z5

    VMOVUPD 192(DI), Z1
    VMOVUPD 192(SI), Z2
    VFMADD231PD Z1, Z2, Z6
    VMOVUPD 192(R8), Z2
    VFMADD231PD Z1, Z2, Z7
    VMOVUPD 192(R9), Z2
    VFMADD231PD Z1, Z2, Z8
    VMOVUPD 192(R10), Z2
    VFMADD231PD Z1, Z2, Z9

    ADDQ $256, DI
    ADDQ $256, SI
    ADDQ $256, R8
    ADDQ $256, R9
    ADDQ $256, R10
    DECQ AX
    JNZ  dot4_512_loop32

dot4_512_loop8_check:
    ANDQ $31, CX
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   dot4_512_reduce

dot4_512_loop8:
    VMOVUPD (DI), Z1
    VMOVUPD (SI), Z2
    VFMADD231PD Z1, Z2, Z0
    VMOVUPD (R8), Z2
    VFMADD231PD Z1, Z2, Z3
    VMOVUPD (R9), Z2
    VFMADD231PD Z1, Z2, Z4
    VMOVUPD (R10), Z2
    VFMADD231PD Z1, Z2, Z5
    ADDQ $64, DI
    ADDQ $64, SI
    ADDQ $64, R8
    ADDQ $64, R9
    ADDQ $64, R10
    DECQ AX
    JNZ  dot4_512_loop8

dot4_512_reduce:
    VADDPD Z6, Z0, Z0
    VADDPD Z7, Z3, Z3
    VADDPD Z8, Z4, Z4
    VADDPD Z9, Z5, Z5

    // Reduce acc0 into X0. VEXTRACTF64X4 (AVX512F) keeps the upper-256 extract
    // off the AVX512DQ path, matching the AVX512F+VL dispatch gate.
    VEXTRACTF64X4 $1, Z0, Y1
    VADDPD Y1, Y0, Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0

    // Reduce acc1 into X3.
    VEXTRACTF64X4 $1, Z3, Y1
    VADDPD Y1, Y3, Y3
    VEXTRACTF128 $1, Y3, X1
    VADDPD X1, X3, X3
    VHADDPD X3, X3, X3

    // Reduce acc2 into X4.
    VEXTRACTF64X4 $1, Z4, Y1
    VADDPD Y1, Y4, Y4
    VEXTRACTF128 $1, Y4, X1
    VADDPD X1, X4, X4
    VHADDPD X4, X4, X4

    // Reduce acc3 into X5.
    VEXTRACTF64X4 $1, Z5, Y1
    VADDPD Y1, Y5, Y5
    VEXTRACTF128 $1, Y5, X1
    VADDPD X1, X5, X5
    VHADDPD X5, X5, X5

    ANDQ $7, CX
    JZ   dot4_512_done

dot4_512_scalar:
    VMOVSD (DI), X1
    VMOVSD (SI), X2
    VFMADD231SD X1, X2, X0
    VMOVSD (R8), X2
    VFMADD231SD X1, X2, X3
    VMOVSD (R9), X2
    VFMADD231SD X1, X2, X4
    VMOVSD (R10), X2
    VFMADD231SD X1, X2, X5
    ADDQ $8, DI
    ADDQ $8, SI
    ADDQ $8, R8
    ADDQ $8, R9
    ADDQ $8, R10
    DECQ CX
    JNZ  dot4_512_scalar

dot4_512_done:
    VMOVSD X0, (DX)
    VMOVSD X3, 8(DX)
    VMOVSD X4, 16(DX)
    VMOVSD X5, 24(DX)
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
    VPXORQ Z0, Z0, Z0          // acc0
    VPXORQ Z3, Z3, Z3          // acc1
    VPXORQ Z4, Z4, Z4          // acc2
    VPXORQ Z5, Z5, Z5          // acc3

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
    // VPANDQ (AVX512F) integer-domain AND: identical bits to VANDPD (AVX512DQ), no DQ dep.
    VPANDQ Z0, Z2, Z1
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

    VPXORQ Z2, Z2, Z2

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
    VPXORQ Z0, Z0, Z0
    VPXORQ Z3, Z3, Z3
    VPXORQ Z4, Z4, Z4
    VPXORQ Z5, Z5, Z5

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
    VPXORQ Z0, Z0, Z0          // acc0
    VPXORQ Z3, Z3, Z3          // acc1
    VPXORQ Z4, Z4, Z4          // acc2
    VPXORQ Z5, Z5, Z5          // acc3

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

// func interleave4AVX(dst, s0, s1, s2, s3 []float64, n int)
// Interleaves 4 planar streams (dst[i*4+c] = s_c[i]) via a 4x4 transpose of
// YMM registers (4 doubles each), 4 frames per iteration. n is a multiple of 4
// (the caller handles the tail).
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
    VMOVUPD (AX), Y0           // r0 = s0[i:i+4]
    VMOVUPD (BX), Y1           // r1 = s1[i:i+4]
    VMOVUPD (CX), Y2           // r2 = s2[i:i+4]
    VMOVUPD (DX), Y3           // r3 = s3[i:i+4]
    VUNPCKLPD Y1, Y0, Y4       // [r0.0,r1.0,r0.2,r1.2]
    VUNPCKHPD Y1, Y0, Y5       // [r0.1,r1.1,r0.3,r1.3]
    VUNPCKLPD Y3, Y2, Y6       // [r2.0,r3.0,r2.2,r3.2]
    VUNPCKHPD Y3, Y2, Y7       // [r2.1,r3.1,r2.3,r3.3]
    VPERM2F128 $0x20, Y6, Y4, Y8   // frame0 = [r0.0,r1.0,r2.0,r3.0]
    VPERM2F128 $0x20, Y7, Y5, Y9   // frame1 = [r0.1,r1.1,r2.1,r3.1]
    VPERM2F128 $0x31, Y6, Y4, Y10  // frame2 = [r0.2,r1.2,r2.2,r3.2]
    VPERM2F128 $0x31, Y7, Y5, Y11  // frame3 = [r0.3,r1.3,r2.3,r3.3]
    VMOVUPD Y8, (DI)
    VMOVUPD Y9, 32(DI)
    VMOVUPD Y10, 64(DI)
    VMOVUPD Y11, 96(DI)
    ADDQ $32, AX
    ADDQ $32, BX
    ADDQ $32, CX
    ADDQ $32, DX
    ADDQ $128, DI
    DECQ SI
    JNZ interleave4_avx_loop

interleave4_avx_done:
    VZEROUPPER
    RET

// func deinterleave4AVX(d0, d1, d2, d3, src []float64, n int)
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
    VMOVUPD (SI), Y0           // frame0 = src[0:4]
    VMOVUPD 32(SI), Y1         // frame1
    VMOVUPD 64(SI), Y2         // frame2
    VMOVUPD 96(SI), Y3         // frame3
    VUNPCKLPD Y1, Y0, Y4       // [f0c0,f1c0,f0c2,f1c2]
    VUNPCKHPD Y1, Y0, Y5       // [f0c1,f1c1,f0c3,f1c3]
    VUNPCKLPD Y3, Y2, Y6       // [f2c0,f3c0,f2c2,f3c2]
    VUNPCKHPD Y3, Y2, Y7       // [f2c1,f3c1,f2c3,f3c3]
    VPERM2F128 $0x20, Y6, Y4, Y8   // chan0 = [f0c0,f1c0,f2c0,f3c0]
    VPERM2F128 $0x20, Y7, Y5, Y9   // chan1
    VPERM2F128 $0x31, Y6, Y4, Y10  // chan2
    VPERM2F128 $0x31, Y7, Y5, Y11  // chan3
    VMOVUPD Y8, (AX)
    VMOVUPD Y9, (BX)
    VMOVUPD Y10, (CX)
    VMOVUPD Y11, (DX)
    ADDQ $32, AX
    ADDQ $32, BX
    ADDQ $32, CX
    ADDQ $32, DX
    ADDQ $128, SI
    DECQ DI
    JNZ deinterleave4_avx_loop

deinterleave4_avx_done:
    VZEROUPPER
    RET

// func interleave3AVX(dst, s0, s1, s2 []float64, n int)
// Interleaves 3 planar streams (dst[i*3+c] = s_c[i]) into 12 contiguous doubles
// per 4 frames via per-stream VPERMPD gathers + VBLENDPD merges (3 streams do
// not map onto a clean register transpose like N=4 does). n is a multiple of 4
// (the caller handles the tail). Requires AVX2.
//
// Per 4-frame block, with a=s0/b=s1/c=s2, the 3 output rows (4 doubles each) are
//   O0 = [a0,b0,c0,a1]  O1 = [b1,c1,a2,b2]  O2 = [c2,a3,b3,c3].
// VPERMPD imm bits [2k+1:2k] pick output lane k's source lane; VBLENDPD imm
// bit k selects the first source operand for lane k (the gathered value).
TEXT ·interleave3AVX(SB), NOSPLIT, $0-104
    MOVQ dst_base+0(FP), DI
    MOVQ s0_base+24(FP), AX
    MOVQ s1_base+48(FP), BX
    MOVQ s2_base+72(FP), CX
    MOVQ n+96(FP), SI
    SHRQ $2, SI                // SI = n/4 blocks
    TESTQ SI, SI
    JZ interleave3_avx_done

interleave3_avx_loop:
    VMOVUPD (AX), Y0           // A = s0[i:i+4]
    VMOVUPD (BX), Y1           // B = s1[i:i+4]
    VMOVUPD (CX), Y2           // C = s2[i:i+4]
    VPERMPD $0x40, Y0, Y3      // O0 base: [a0,a0,a0,a1]
    VPERMPD $0x00, Y1, Y4      // [b0,b0,b0,b0]
    VPERMPD $0x00, Y2, Y5      // [c0,c0,c0,c0]
    VBLENDPD $0x2, Y4, Y3, Y3  // lane1 <- b0
    VBLENDPD $0x4, Y5, Y3, Y3  // lane2 <- c0
    VMOVUPD Y3, (DI)           // O0 = [a0,b0,c0,a1]
    VPERMPD $0x81, Y1, Y4      // O1 base: [b1,b0,b0,b2]
    VPERMPD $0x04, Y2, Y5      // [c0,c1,c0,c0]
    VPERMPD $0x20, Y0, Y3      // [a0,a0,a2,a0]
    VBLENDPD $0x2, Y5, Y4, Y4  // lane1 <- c1
    VBLENDPD $0x4, Y3, Y4, Y4  // lane2 <- a2
    VMOVUPD Y4, 32(DI)         // O1 = [b1,c1,a2,b2]
    VPERMPD $0xC2, Y2, Y5      // O2 base: [c2,c0,c0,c3]
    VPERMPD $0x0C, Y0, Y3      // [a0,a3,a0,a0]
    VPERMPD $0x30, Y1, Y4      // [b0,b0,b3,b0]
    VBLENDPD $0x2, Y3, Y5, Y5  // lane1 <- a3
    VBLENDPD $0x4, Y4, Y5, Y5  // lane2 <- b3
    VMOVUPD Y5, 64(DI)         // O2 = [c2,a3,b3,c3]
    ADDQ $32, AX
    ADDQ $32, BX
    ADDQ $32, CX
    ADDQ $96, DI
    DECQ SI
    JNZ interleave3_avx_loop

interleave3_avx_done:
    VZEROUPPER
    RET

// func deinterleave3AVX(d0, d1, d2, src []float64, n int)
// Splits a 3-stream interleaved buffer (d_c[i] = src[i*3+c]) into planar streams,
// 4 frames per iteration, via per-stream VPERMPD gathers + VBLENDPD merges. n is
// a multiple of 4 (the caller handles the tail). Requires AVX2.
//
// Per block the 3 input rows are S0=[a0,b0,c0,a1] S1=[b1,c1,a2,b2] S2=[c2,a3,b3,c3]
// and the planar outputs are A=[a0,a1,a2,a3] B=[b0,b1,b2,b3] C=[c0,c1,c2,c3].
TEXT ·deinterleave3AVX(SB), NOSPLIT, $0-104
    MOVQ d0_base+0(FP), AX
    MOVQ d1_base+24(FP), BX
    MOVQ d2_base+48(FP), CX
    MOVQ src_base+72(FP), SI
    MOVQ n+96(FP), DI
    SHRQ $2, DI                // DI = n/4 blocks
    TESTQ DI, DI
    JZ deinterleave3_avx_done

deinterleave3_avx_loop:
    VMOVUPD (SI), Y0           // S0 = src[0:4]
    VMOVUPD 32(SI), Y1         // S1 = src[4:8]
    VMOVUPD 64(SI), Y2         // S2 = src[8:12]
    VPERMPD $0x0C, Y0, Y3      // A base: [a0,a1,a0,a0]
    VPERMPD $0x20, Y1, Y4      // [.,.,a2,.]
    VPERMPD $0x40, Y2, Y5      // [.,.,.,a3]
    VBLENDPD $0x4, Y4, Y3, Y3  // lane2 <- a2
    VBLENDPD $0x8, Y5, Y3, Y3  // lane3 <- a3
    VMOVUPD Y3, (AX)           // A = [a0,a1,a2,a3]
    VPERMPD $0x01, Y0, Y3      // B base: [b0,a0,a0,a0]
    VPERMPD $0x30, Y1, Y4      // [b1,b1,b2,.]
    VPERMPD $0x80, Y2, Y5      // [.,.,.,b3]
    VBLENDPD $0x6, Y4, Y3, Y3  // lanes1,2 <- b1,b2
    VBLENDPD $0x8, Y5, Y3, Y3  // lane3 <- b3
    VMOVUPD Y3, (BX)           // B = [b0,b1,b2,b3]
    VPERMPD $0x02, Y0, Y3      // C base: [c0,.,.,.]
    VPERMPD $0x04, Y1, Y4      // [.,c1,.,.]
    VPERMPD $0xC0, Y2, Y5      // [c2,.,c2,c3]
    VBLENDPD $0x2, Y4, Y3, Y3  // lane1 <- c1
    VBLENDPD $0xC, Y5, Y3, Y3  // lanes2,3 <- c2,c3
    VMOVUPD Y3, (CX)           // C = [c0,c1,c2,c3]
    ADDQ $96, SI
    ADDQ $32, AX
    ADDQ $32, BX
    ADDQ $32, CX
    DECQ DI
    JNZ deinterleave3_avx_loop

deinterleave3_avx_done:
    VZEROUPPER
    RET

// func interleave8AVX(dst []float64, srcs [][]float64, n int)
// Interleaves 8 planar streams (dst[i*8+c] = srcs[c][i]) into 32 doubles per 4
// frames. A YMM holds 4 doubles, so each output frame spans two YMM rows; the
// kernel runs two independent 4x4 transposes (interleave4AVX's algorithm):
// streams 0-3 produce each frame's low YMM, streams 4-7 the high YMM. n is a
// multiple of 4 (the caller handles the tail). srcs must have exactly 8 elements.
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
    SHRQ $2, SI                  // SI = n/4 blocks
    TESTQ SI, SI
    JZ interleave8_avx_done

interleave8_avx_loop:
    // streams 0-3 -> low YMM of frames 0..3
    VMOVUPD (AX), Y0             // s0[i:i+4]
    VMOVUPD (BX), Y1             // s1
    VMOVUPD (CX), Y2             // s2
    VMOVUPD (DX), Y3             // s3
    VUNPCKLPD Y1, Y0, Y4         // [s0.0,s1.0,s0.2,s1.2]
    VUNPCKHPD Y1, Y0, Y5         // [s0.1,s1.1,s0.3,s1.3]
    VUNPCKLPD Y3, Y2, Y6         // [s2.0,s3.0,s2.2,s3.2]
    VUNPCKHPD Y3, Y2, Y7         // [s2.1,s3.1,s2.3,s3.3]
    VPERM2F128 $0x20, Y6, Y4, Y8    // f0 lo = [s0.0,s1.0,s2.0,s3.0]
    VPERM2F128 $0x20, Y7, Y5, Y9    // f1 lo
    VPERM2F128 $0x31, Y6, Y4, Y10   // f2 lo
    VPERM2F128 $0x31, Y7, Y5, Y11   // f3 lo
    VMOVUPD Y8, (DI)             // dst[0:4]
    VMOVUPD Y9, 64(DI)           // dst[8:12]
    VMOVUPD Y10, 128(DI)         // dst[16:20]
    VMOVUPD Y11, 192(DI)         // dst[24:28]
    // streams 4-7 -> high YMM of frames 0..3
    VMOVUPD (R8), Y0             // s4
    VMOVUPD (R9), Y1             // s5
    VMOVUPD (R10), Y2            // s6
    VMOVUPD (R11), Y3            // s7
    VUNPCKLPD Y1, Y0, Y4
    VUNPCKHPD Y1, Y0, Y5
    VUNPCKLPD Y3, Y2, Y6
    VUNPCKHPD Y3, Y2, Y7
    VPERM2F128 $0x20, Y6, Y4, Y8    // f0 hi = [s4.0,s5.0,s6.0,s7.0]
    VPERM2F128 $0x20, Y7, Y5, Y9    // f1 hi
    VPERM2F128 $0x31, Y6, Y4, Y10   // f2 hi
    VPERM2F128 $0x31, Y7, Y5, Y11   // f3 hi
    VMOVUPD Y8, 32(DI)           // dst[4:8]
    VMOVUPD Y9, 96(DI)           // dst[12:16]
    VMOVUPD Y10, 160(DI)         // dst[20:24]
    VMOVUPD Y11, 224(DI)         // dst[28:32]
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

// func deinterleave8AVX(dsts [][]float64, src []float64, n int)
// Splits an interleaved 8-stream buffer (dsts[c][i] = src[i*8+c]) into planar
// streams, 4 frames per iteration. Each frame spans two YMM rows; the low rows
// (src[f*8:f*8+4]) transpose into streams 0-3 and the high rows
// (src[f*8+4:f*8+8]) into streams 4-7. n is a multiple of 4 (the caller handles
// the tail). dsts must have exactly 8 elements.
TEXT ·deinterleave8AVX(SB), NOSPLIT, $0-56
    MOVQ dsts_base+0(FP), R12    // R12 = &dsts[0] (array of slice headers)
    MOVQ 0(R12), AX              // dsts[0].ptr
    MOVQ 24(R12), BX             // dsts[1].ptr
    MOVQ 48(R12), CX             // dsts[2].ptr
    MOVQ 72(R12), DX             // dsts[3].ptr
    MOVQ 96(R12), R8             // dsts[4].ptr
    MOVQ 120(R12), R9            // dsts[5].ptr
    MOVQ 144(R12), R10           // dsts[6].ptr
    MOVQ 168(R12), R11           // dsts[7].ptr
    MOVQ src_base+24(FP), SI
    MOVQ n+48(FP), DI
    SHRQ $2, DI                  // DI = n/4 blocks
    TESTQ DI, DI
    JZ deinterleave8_avx_done

deinterleave8_avx_loop:
    // low rows of frames 0..3 -> streams 0-3
    VMOVUPD (SI), Y0             // f0 lo = [s0.0,s1.0,s2.0,s3.0]
    VMOVUPD 64(SI), Y1           // f1 lo
    VMOVUPD 128(SI), Y2          // f2 lo
    VMOVUPD 192(SI), Y3          // f3 lo
    VUNPCKLPD Y1, Y0, Y4
    VUNPCKHPD Y1, Y0, Y5
    VUNPCKLPD Y3, Y2, Y6
    VUNPCKHPD Y3, Y2, Y7
    VPERM2F128 $0x20, Y6, Y4, Y8    // s0 = [s0.0,s0.1,s0.2,s0.3]
    VPERM2F128 $0x20, Y7, Y5, Y9    // s1
    VPERM2F128 $0x31, Y6, Y4, Y10   // s2
    VPERM2F128 $0x31, Y7, Y5, Y11   // s3
    VMOVUPD Y8, (AX)
    VMOVUPD Y9, (BX)
    VMOVUPD Y10, (CX)
    VMOVUPD Y11, (DX)
    // high rows of frames 0..3 -> streams 4-7
    VMOVUPD 32(SI), Y0           // f0 hi = [s4.0,s5.0,s6.0,s7.0]
    VMOVUPD 96(SI), Y1           // f1 hi
    VMOVUPD 160(SI), Y2          // f2 hi
    VMOVUPD 224(SI), Y3          // f3 hi
    VUNPCKLPD Y1, Y0, Y4
    VUNPCKHPD Y1, Y0, Y5
    VUNPCKLPD Y3, Y2, Y6
    VUNPCKHPD Y3, Y2, Y7
    VPERM2F128 $0x20, Y6, Y4, Y8    // s4
    VPERM2F128 $0x20, Y7, Y5, Y9    // s5
    VPERM2F128 $0x31, Y6, Y4, Y10   // s6
    VPERM2F128 $0x31, Y7, Y5, Y11   // s7
    VMOVUPD Y8, (R8)
    VMOVUPD Y9, (R9)
    VMOVUPD Y10, (R10)
    VMOVUPD Y11, (R11)
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

// func interleave6AVX(dst, s0, s1, s2, s3, s4, s5 []float64, n int)
// Interleaves 6 planar streams (dst[i*6+c] = s_c[i]) into 24 contiguous doubles
// per 4 frames. N=6 has no clean register transpose, so the kernel zips each
// stream pair (s0,s1)(s2,s3)(s4,s5) into a 128-bit pair stream via VUNPCKLPD/
// VUNPCKHPD + VPERM2F128, then interleaves the three pair-streams at 128-bit lane
// granularity with VPERM2F128 (the f64 analogue of the f32 N=6 zip, working on
// 2-double pairs instead of 2-float pairs). n is a multiple of 4 (the caller
// handles the tail). Requires AVX2.
//
// PA pairs hold (s0[k],s1[k]); PB (s2,s3); PC (s4,s5). The 6 output rows of 2
// pairs each, [A0,B0] [C0,A1] [B1,C1] [A2,B2] [C2,A3] [B3,C3], lay the six
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
    SHRQ $2, SI                  // SI = n/4 blocks
    TESTQ SI, SI
    JZ interleave6_avx_done

interleave6_avx_loop:
    VMOVUPD (AX), Y0             // s0[i:i+4]
    VMOVUPD (BX), Y1             // s1
    VMOVUPD (CX), Y2             // s2
    VMOVUPD (DX), Y3             // s3
    VMOVUPD (R8), Y4             // s4
    VMOVUPD (R9), Y5             // s5
    VUNPCKLPD Y1, Y0, Y6         // zip (s0,s1): [a0,b0|a2,b2]
    VUNPCKHPD Y1, Y0, Y7         // [a1,b1|a3,b3]
    VPERM2F128 $0x20, Y7, Y6, Y8 // PA_lo=[a0,b0,a1,b1] pairs (frames 0,1)
    VPERM2F128 $0x31, Y7, Y6, Y9 // PA_hi=[a2,b2,a3,b3] pairs (frames 2,3)
    VUNPCKLPD Y3, Y2, Y6         // zip (s2,s3)
    VUNPCKHPD Y3, Y2, Y7
    VPERM2F128 $0x20, Y7, Y6, Y10 // PB_lo
    VPERM2F128 $0x31, Y7, Y6, Y11 // PB_hi
    VUNPCKLPD Y5, Y4, Y6         // zip (s4,s5)
    VUNPCKHPD Y5, Y4, Y7
    VPERM2F128 $0x20, Y7, Y6, Y12 // PC_lo
    VPERM2F128 $0x31, Y7, Y6, Y13 // PC_hi
    VPERM2F128 $0x20, Y10, Y8, Y0  // [A0,B0]
    VPERM2F128 $0x30, Y8, Y12, Y1  // [C0,A1]
    VPERM2F128 $0x31, Y12, Y10, Y2 // [B1,C1]
    VPERM2F128 $0x20, Y11, Y9, Y3  // [A2,B2]
    VPERM2F128 $0x30, Y9, Y13, Y4  // [C2,A3]
    VPERM2F128 $0x31, Y13, Y11, Y5 // [B3,C3]
    VMOVUPD Y0, (DI)
    VMOVUPD Y1, 32(DI)
    VMOVUPD Y2, 64(DI)
    VMOVUPD Y3, 96(DI)
    VMOVUPD Y4, 128(DI)
    VMOVUPD Y5, 160(DI)
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

// func deinterleave6AVX(d0, d1, d2, d3, d4, d5, src []float64, n int)
// Splits a 6-stream interleaved buffer (d_c[i] = src[i*6+c]) into planar streams,
// 4 frames per iteration. The inverse of interleave6AVX: VPERM2F128 regroups the
// six input rows into the three 128-bit pair-streams PA/PB/PC, then VUNPCKLPD/
// VUNPCKHPD + VPERMPD unzip each pair back into its two float64 planar streams. n
// is a multiple of 4 (the caller handles the tail). Requires AVX2.
TEXT ·deinterleave6AVX(SB), NOSPLIT, $0-176
    MOVQ d0_base+0(FP), AX
    MOVQ d1_base+24(FP), BX
    MOVQ d2_base+48(FP), CX
    MOVQ d3_base+72(FP), DX
    MOVQ d4_base+96(FP), R8
    MOVQ d5_base+120(FP), R9
    MOVQ src_base+144(FP), SI
    MOVQ n+168(FP), DI
    SHRQ $2, DI                  // DI = n/4 blocks
    TESTQ DI, DI
    JZ deinterleave6_avx_done

deinterleave6_avx_loop:
    VMOVUPD (SI), Y0             // [A0,B0]
    VMOVUPD 32(SI), Y1           // [C0,A1]
    VMOVUPD 64(SI), Y2           // [B1,C1]
    VMOVUPD 96(SI), Y3           // [A2,B2]
    VMOVUPD 128(SI), Y4          // [C2,A3]
    VMOVUPD 160(SI), Y5          // [B3,C3]
    VPERM2F128 $0x30, Y1, Y0, Y8  // PA_lo=[A0,A1]
    VPERM2F128 $0x30, Y4, Y3, Y9  // PA_hi=[A2,A3]
    VPERM2F128 $0x21, Y2, Y0, Y10 // PB_lo=[B0,B1]
    VPERM2F128 $0x21, Y5, Y3, Y11 // PB_hi=[B2,B3]
    VPERM2F128 $0x30, Y2, Y1, Y12 // PC_lo=[C0,C1]
    VPERM2F128 $0x30, Y5, Y4, Y13 // PC_hi=[C2,C3]
    // unzip pair-streams back to planar float64 (a even lanes, b odd lanes)
    VUNPCKLPD Y9, Y8, Y6         // [a0,a2,a1,a3]
    VPERMPD $0xD8, Y6, Y0        // s0=[a0,a1,a2,a3]
    VUNPCKHPD Y9, Y8, Y7         // [b0,b2,b1,b3]
    VPERMPD $0xD8, Y7, Y1        // s1=[b0,b1,b2,b3]
    VMOVUPD Y0, (AX)
    VMOVUPD Y1, (BX)
    VUNPCKLPD Y11, Y10, Y6
    VPERMPD $0xD8, Y6, Y2        // s2
    VUNPCKHPD Y11, Y10, Y7
    VPERMPD $0xD8, Y7, Y6        // s3
    VMOVUPD Y2, (CX)
    VMOVUPD Y6, (DX)
    VUNPCKLPD Y13, Y12, Y6
    VPERMPD $0xD8, Y6, Y3        // s4
    VUNPCKHPD Y13, Y12, Y7
    VPERMPD $0xD8, Y7, Y7        // s5
    VMOVUPD Y3, (R8)
    VMOVUPD Y7, (R9)
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

    // Independent accumulators for ILP.
    VXORPD Y0, Y0, Y0
    VXORPD Y11, Y11, Y11
    VXORPD Y12, Y12, Y12
    VXORPD Y13, Y13, Y13

    // Process 16 elements per iteration (4 YMM × 4 doubles)
    MOVQ CX, AX
    SHRQ $4, AX                // AX = len / 16
    JZ   cubic_avx_loop4_check

    // DX is a shared byte-offset index for all 5 streams (hist, a, b, c, d),
    // so a single increment per iteration replaces five pointer advances.
    XORQ DX, DX

cubic_avx_loop16:
    // Multi-stage Horner with four independent chains.
    // VFMADD213PD dst, mul, add: dst = dst*mul + add (Y7 = broadcast x).
    // All loads use base+DX indexing; DX is the shared byte offset.

    // Stage 1: p = c + x*d (4 parallel chains)
    VMOVUPD 0(R9)(DX*1), Y1
    VMOVUPD 0(R10)(DX*1), Y5
    VFMADD231PD Y5, Y7, Y1

    VMOVUPD 32(R9)(DX*1), Y2
    VMOVUPD 32(R10)(DX*1), Y5
    VFMADD231PD Y5, Y7, Y2

    VMOVUPD 64(R9)(DX*1), Y3
    VMOVUPD 64(R10)(DX*1), Y5
    VFMADD231PD Y5, Y7, Y3

    VMOVUPD 96(R9)(DX*1), Y4
    VMOVUPD 96(R10)(DX*1), Y5
    VFMADD231PD Y5, Y7, Y4

    // Stage 2: p = b + x*p
    VMOVUPD 0(R8)(DX*1), Y5
    VFMADD213PD Y5, Y7, Y1
    VMOVUPD 32(R8)(DX*1), Y5
    VFMADD213PD Y5, Y7, Y2
    VMOVUPD 64(R8)(DX*1), Y5
    VFMADD213PD Y5, Y7, Y3
    VMOVUPD 96(R8)(DX*1), Y5
    VFMADD213PD Y5, Y7, Y4

    // Stage 3: coef = a + x*p
    VMOVUPD 0(DI)(DX*1), Y5
    VFMADD213PD Y5, Y7, Y1
    VMOVUPD 32(DI)(DX*1), Y5
    VFMADD213PD Y5, Y7, Y2
    VMOVUPD 64(DI)(DX*1), Y5
    VFMADD213PD Y5, Y7, Y3
    VMOVUPD 96(DI)(DX*1), Y5
    VFMADD213PD Y5, Y7, Y4

    // Accumulate: Σ hist * coef with independent accumulators.
    VMOVUPD 0(SI)(DX*1), Y5
    VFMADD231PD Y5, Y1, Y0
    VMOVUPD 32(SI)(DX*1), Y5
    VFMADD231PD Y5, Y2, Y11
    VMOVUPD 64(SI)(DX*1), Y5
    VFMADD231PD Y5, Y3, Y12
    VMOVUPD 96(SI)(DX*1), Y5
    VFMADD231PD Y5, Y4, Y13

    ADDQ $128, DX             // single index advance (was 5 pointer adds)
    DECQ AX
    JNZ  cubic_avx_loop16

    // Combine accumulators.
    VADDPD Y11, Y0, Y0
    VADDPD Y12, Y0, Y0
    VADDPD Y13, Y0, Y0

    // Advance base pointers once past the bytes consumed by loop16 so the
    // loop4 and scalar tails resume at the correct offset.
    ADDQ DX, SI
    ADDQ DX, DI
    ADDQ DX, R8
    ADDQ DX, R9
    ADDQ DX, R10

cubic_avx_loop4_check:
    // Handle remaining 4-element vectors.
    ANDQ $15, CX
    MOVQ CX, AX
    SHRQ $2, AX
    JZ   cubic_avx_remainder

cubic_avx_loop4:
    VMOVUPD (R10), Y1          // d
    VMOVUPD (R9), Y2           // c
    VMOVUPD (R8), Y3           // b
    VMOVUPD (DI), Y4           // a
    VMOVUPD (SI), Y5           // hist

    // Horner's method: coef = a + x*(b + x*(c + x*d))
    VFMADD231PD Y1, Y7, Y2
    VFMADD231PD Y2, Y7, Y3
    VFMADD231PD Y3, Y7, Y4
    VFMADD231PD Y5, Y4, Y0

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

// Constants for exp-based tanh (float64)
DATA tanh64_two<>+0x00(SB)/8, $0x4000000000000000  // 2.0
DATA tanh64_two<>+0x08(SB)/8, $0x4000000000000000
DATA tanh64_two<>+0x10(SB)/8, $0x4000000000000000
DATA tanh64_two<>+0x18(SB)/8, $0x4000000000000000
GLOBL tanh64_two<>(SB), RODATA|NOPTR, $32

DATA tanh64_log2e<>+0x00(SB)/8, $0x3FF71547652B82FE  // log2(e) = 1.4426950408889634
DATA tanh64_log2e<>+0x08(SB)/8, $0x3FF71547652B82FE
DATA tanh64_log2e<>+0x10(SB)/8, $0x3FF71547652B82FE
DATA tanh64_log2e<>+0x18(SB)/8, $0x3FF71547652B82FE
GLOBL tanh64_log2e<>(SB), RODATA|NOPTR, $32

DATA tanh64_ln2<>+0x00(SB)/8, $0x3FE62E42FEFA39EF  // ln(2) = 0.6931471805599453
DATA tanh64_ln2<>+0x08(SB)/8, $0x3FE62E42FEFA39EF
DATA tanh64_ln2<>+0x10(SB)/8, $0x3FE62E42FEFA39EF
DATA tanh64_ln2<>+0x18(SB)/8, $0x3FE62E42FEFA39EF
GLOBL tanh64_ln2<>(SB), RODATA|NOPTR, $32

DATA tanh64_c3<>+0x00(SB)/8, $0x3FC5555555555555  // c3 = 1/6
DATA tanh64_c3<>+0x08(SB)/8, $0x3FC5555555555555
DATA tanh64_c3<>+0x10(SB)/8, $0x3FC5555555555555
DATA tanh64_c3<>+0x18(SB)/8, $0x3FC5555555555555
GLOBL tanh64_c3<>(SB), RODATA|NOPTR, $32

DATA tanh64_c4<>+0x00(SB)/8, $0x3FA5555555555555  // c4 = 1/24
DATA tanh64_c4<>+0x08(SB)/8, $0x3FA5555555555555
DATA tanh64_c4<>+0x10(SB)/8, $0x3FA5555555555555
DATA tanh64_c4<>+0x18(SB)/8, $0x3FA5555555555555
GLOBL tanh64_c4<>(SB), RODATA|NOPTR, $32

DATA tanh64_c5<>+0x00(SB)/8, $0x3F81111111111111  // c5 = 1/120
DATA tanh64_c5<>+0x08(SB)/8, $0x3F81111111111111
DATA tanh64_c5<>+0x10(SB)/8, $0x3F81111111111111
DATA tanh64_c5<>+0x18(SB)/8, $0x3F81111111111111
GLOBL tanh64_c5<>(SB), RODATA|NOPTR, $32

DATA tanh64_clamp_hi<>+0x00(SB)/8, $0x4034000000000000  // 20.0
DATA tanh64_clamp_hi<>+0x08(SB)/8, $0x4034000000000000
DATA tanh64_clamp_hi<>+0x10(SB)/8, $0x4034000000000000
DATA tanh64_clamp_hi<>+0x18(SB)/8, $0x4034000000000000
GLOBL tanh64_clamp_hi<>(SB), RODATA|NOPTR, $32

DATA tanh64_clamp_lo<>+0x00(SB)/8, $0xC034000000000000  // -20.0
DATA tanh64_clamp_lo<>+0x08(SB)/8, $0xC034000000000000
DATA tanh64_clamp_lo<>+0x10(SB)/8, $0xC034000000000000
DATA tanh64_clamp_lo<>+0x18(SB)/8, $0xC034000000000000
GLOBL tanh64_clamp_lo<>(SB), RODATA|NOPTR, $32

// Exp clamp thresholds: ±709.0 keeps the 2^k reconstruction inside the
// representable float64 range (exp(709) ~= 8.2e307 < MaxFloat64). Matches the
// pure-Go fallback's overflow/underflow clamp.
// Widened to 4 lanes (32 bytes) so the 4/iter YMM activation kernels can load
// the clamp directly with VMOVUPD. The scalar remainder path reads only the
// first lane via VMOVSD.
DATA exp_clamp_hi64<>+0x00(SB)/8, $0x4086280000000000  // 709.0
DATA exp_clamp_hi64<>+0x08(SB)/8, $0x4086280000000000
DATA exp_clamp_hi64<>+0x10(SB)/8, $0x4086280000000000
DATA exp_clamp_hi64<>+0x18(SB)/8, $0x4086280000000000
GLOBL exp_clamp_hi64<>(SB), RODATA|NOPTR, $32

DATA exp_clamp_lo64<>+0x00(SB)/8, $0xC086280000000000  // -709.0
DATA exp_clamp_lo64<>+0x08(SB)/8, $0xC086280000000000
DATA exp_clamp_lo64<>+0x10(SB)/8, $0xC086280000000000
DATA exp_clamp_lo64<>+0x18(SB)/8, $0xC086280000000000
GLOBL exp_clamp_lo64<>(SB), RODATA|NOPTR, $32

// Note: absf64mask is already defined at top of file

// func sigmoidAVX(dst, src []float64)
// Computes accurate sigmoid: sigmoid(x) = 1 / (1 + e^(-x)).
// Uses the same exp range reduction + degree-5 polynomial core as tanhAVX,
// replacing the previous fast rational approximation (0.5 + 0.5*x/(1+|x|)),
// which was far less accurate than the f32 kernel. Inputs are clamped via
// z = -x in [-709, 709] so the 2^k reconstruction stays in range.
// Processes 4 elements per iteration on YMM. The 2^k reconstruction is
// vectorized with AVX2 integer ops (VCVTTPD2DQ/VPMOVSXDQ/VPSLLQ/VPADDQ), so
// this kernel must be dispatched only when cpu.X86.AVX2 is set; no FMA is used.
// The 0-3 element tail uses the scalar 1/iter path below.
TEXT ·sigmoidAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DI
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    // Load constants into YMM registers (all are 32-byte / 4-lane RODATA).
    // The low 128 bits (X9-X15) are reused by the scalar remainder path.
    VMOVUPD tanh64_log2e<>(SB), Y9     // Y9 = log2(e)
    VMOVUPD tanh64_ln2<>(SB), Y10      // Y10 = ln(2)
    VMOVUPD sigmoid_one64<>(SB), Y11   // Y11 = 1.0
    VMOVUPD sigmoid_half64<>(SB), Y12  // Y12 = 0.5 (c2)
    VMOVUPD tanh64_c3<>(SB), Y13       // Y13 = 1/6 (c3)
    VMOVUPD tanh64_c4<>(SB), Y14       // Y14 = 1/24 (c4)
    VMOVUPD tanh64_c5<>(SB), Y15       // Y15 = 1/120 (c5)

    // Clamp bounds are loop-invariant; hoist into Y7/Y8 (both free in this
    // kernel) so they are not reloaded from memory on every iteration.
    VMOVUPD exp_clamp_hi64<>(SB), Y7   // Y7 = 709.0
    VMOVUPD exp_clamp_lo64<>(SB), Y8   // Y8 = -709.0

    // Process 4 elements per iteration (YMM = 256 bits = 4 x float64)
    MOVQ CX, R8
    SHRQ $2, R8                        // len / 4
    JZ   sigmoid64_remainder

sigmoid64_loop4:
    VMOVUPD (SI), Y0                   // Y0 = 4 x float64 = x

    // z = -x
    VXORPD Y1, Y1, Y1                  // Y1 = 0
    VSUBPD Y0, Y1, Y0                  // Y0 = -x = z

    // Clamp z to [-709, 709] so 2^k stays representable (Y7=709, Y8=-709)
    VMINPD Y7, Y0, Y0                  // Y0 = min(z, 709)
    VMAXPD Y8, Y0, Y0                  // Y0 = max(min(z, 709), -709)

    // Range reduction: k = round(z * log2e), r = z - k * ln2
    VMULPD Y9, Y0, Y1                  // Y1 = z * log2e
    VROUNDPD $0, Y1, Y2                // Y2 = k = round(Y1) (nearest)
    VMULPD Y10, Y2, Y3                 // Y3 = k * ln2
    VSUBPD Y3, Y0, Y3                  // Y3 = r = z - k * ln2

    // Polynomial: exp(r) ≈ 1 + r*(1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
    // Horner's method
    VMULPD Y3, Y15, Y4                 // Y4 = r * c5
    VADDPD Y14, Y4, Y4                 // Y4 = c4 + r*c5
    VMULPD Y3, Y4, Y4                  // Y4 = r*(c4 + r*c5)
    VADDPD Y13, Y4, Y4                 // Y4 = c3 + r*(...)
    VMULPD Y3, Y4, Y4                  // Y4 = r*(c3 + ...)
    VADDPD Y12, Y4, Y4                 // Y4 = c2 + r*(...)
    VMULPD Y3, Y4, Y4                  // Y4 = r*(c2 + ...)
    VADDPD Y11, Y4, Y4                 // Y4 = 1 + r*(...)
    VMULPD Y3, Y4, Y4                  // Y4 = r*(1 + ...)
    VADDPD Y11, Y4, Y4                 // Y4 = exp(r)

    // Reconstruct exp(z) = exp(r) * 2^k for all 4 lanes with AVX2 integer ops.
    // k is integer-valued (from VROUNDPD), so float64->int32 truncation is exact.
    VCVTTPD2DQY Y2, X5                 // X5 = 4 x int32(k)
    VPMOVSXDQ X5, Y5                   // Y5 = 4 x int64(k) (sign-extended; k may be < 0)
    VPSLLQ $52, Y5, Y5                 // Y5 = k << 52 (into the exponent field)
    VPADDQ sigmoid_one64<>(SB), Y5, Y5 // Y5 = (k<<52) + bits(1.0) = bits(2^k)
    VMULPD Y5, Y4, Y4                  // Y4 = exp(z) = exp(-x)

    // sigmoid(x) = 1 / (1 + exp(-x))
    VADDPD Y4, Y11, Y6                 // Y6 = 1 + exp(-x)
    VDIVPD Y6, Y11, Y0                 // Y0 = 1 / (1 + exp(-x))

    VMOVUPD Y0, (DI)                   // store 4 results
    ADDQ $32, SI
    ADDQ $32, DI
    DECQ R8
    JNZ  sigmoid64_loop4

sigmoid64_remainder:
    ANDQ $3, CX                        // remainder = len % 4
    JZ   sigmoid64_done

sigmoid64_scalar:
    VMOVSD (SI), X0                    // X0 = x

    // z = -x
    VXORPD X1, X1, X1
    VSUBSD X0, X1, X0                  // X0 = -x

    // Clamp to [-709, 709] using the hoisted bounds (X7=709, X8=-709)
    VMINSD X7, X0, X0
    VMAXSD X8, X0, X0

    // Range reduction
    VMULSD X9, X0, X1                  // X1 = z * log2e
    VROUNDSD $0, X1, X1, X2            // X2 = k = round(X1)
    VMULSD X10, X2, X3                 // X3 = k * ln2
    VSUBSD X3, X0, X3                  // X3 = r = z - k * ln2

    // Horner's polynomial
    VMULSD X3, X15, X4                 // X4 = r * c5
    VADDSD X14, X4, X4
    VMULSD X3, X4, X4
    VADDSD X13, X4, X4
    VMULSD X3, X4, X4
    VADDSD X12, X4, X4
    VMULSD X3, X4, X4
    VADDSD X11, X4, X4
    VMULSD X3, X4, X4
    VADDSD X11, X4, X4                 // X4 = exp(r)

    // Reconstruct 2^k
    VCVTTSD2SI X2, AX                  // AX = int64(k)
    SHLQ $52, AX                       // AX = k << 52
    MOVQ $0x3FF0000000000000, BX       // 1.0's bits
    ADDQ BX, AX                        // AX = 2^k bits
    VMOVQ AX, X5
    VMULSD X5, X4, X4                  // X4 = exp(-x)

    // sigmoid = 1 / (1 + exp(-x))
    VADDSD X4, X11, X6                 // X6 = 1 + exp(-x)
    VDIVSD X6, X11, X0                 // X0 = 1 / (1 + exp(-x))

    VMOVSD X0, (DI)
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
// Computes accurate tanh: tanh(x) = (1 - e^(-2x)) / (1 + e^(-2x))
// Uses range reduction and a degree-5 polynomial approximation for exp.
// Processes 4 elements per iteration on YMM. The 2^k reconstruction is
// vectorized with AVX2 integer ops (VCVTTPD2DQ/VPMOVSXDQ/VPSLLQ/VPADDQ), so
// this kernel must be dispatched only when cpu.X86.AVX2 is set; no FMA is used.
// The 0-3 element tail uses the scalar 1/iter path below.
TEXT ·tanhAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    // Load constants into YMM registers (all are 32-byte / 4-lane RODATA).
    // The low 128 bits (X8-X15) are reused by the scalar remainder path.
    VMOVUPD tanh64_two<>(SB), Y8       // Y8 = 2.0
    VMOVUPD tanh64_log2e<>(SB), Y9     // Y9 = log2(e)
    VMOVUPD tanh64_ln2<>(SB), Y10      // Y10 = ln(2)
    VMOVUPD sigmoid_one64<>(SB), Y11   // Y11 = 1.0
    VMOVUPD sigmoid_half64<>(SB), Y12  // Y12 = 0.5 (c2)
    VMOVUPD tanh64_c3<>(SB), Y13       // Y13 = 1/6 (c3)
    VMOVUPD tanh64_c4<>(SB), Y14       // Y14 = 1/24 (c4)
    VMOVUPD tanh64_c5<>(SB), Y15       // Y15 = 1/120 (c5)

    // Process 4 elements per iteration (YMM = 256 bits = 4 x float64)
    MOVQ CX, R8
    SHRQ $2, R8                        // len / 4
    JZ   tanh64_remainder

tanh64_loop4:
    VMOVUPD (SI), Y0                   // Y0 = 4 x float64

    // Compute z = -2x
    VXORPD Y1, Y1, Y1                  // Y1 = 0
    VSUBPD Y0, Y1, Y0                  // Y0 = -x
    VMULPD Y8, Y0, Y0                  // Y0 = -2x = z

    // Clamp z to [-20, 20]
    VMOVUPD tanh64_clamp_hi<>(SB), Y1  // Y1 = 20.0
    VMOVUPD tanh64_clamp_lo<>(SB), Y2  // Y2 = -20.0
    VMINPD Y1, Y0, Y0                  // Y0 = min(z, 20)
    VMAXPD Y2, Y0, Y0                  // Y0 = max(min(z, 20), -20)

    // Range reduction: k = round(z * log2e), r = z - k * ln2
    VMULPD Y9, Y0, Y1                  // Y1 = z * log2e
    VROUNDPD $0, Y1, Y2                // Y2 = k = round(Y1) (nearest)
    VMULPD Y10, Y2, Y3                 // Y3 = k * ln2
    VSUBPD Y3, Y0, Y3                  // Y3 = r = z - k * ln2

    // Polynomial: exp(r) ≈ 1 + r*(1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
    // Horner's method
    VMULPD Y3, Y15, Y4                 // Y4 = r * c5
    VADDPD Y14, Y4, Y4                 // Y4 = c4 + r*c5
    VMULPD Y3, Y4, Y4                  // Y4 = r*(c4 + r*c5)
    VADDPD Y13, Y4, Y4                 // Y4 = c3 + r*(...)
    VMULPD Y3, Y4, Y4                  // Y4 = r*(c3 + ...)
    VADDPD Y12, Y4, Y4                 // Y4 = c2 + r*(...)
    VMULPD Y3, Y4, Y4                  // Y4 = r*(c2 + ...)
    VADDPD Y11, Y4, Y4                 // Y4 = 1 + r*(...)
    VMULPD Y3, Y4, Y4                  // Y4 = r*(1 + ...)
    VADDPD Y11, Y4, Y4                 // Y4 = exp(r)

    // Reconstruct exp(z) = exp(r) * 2^k for all 4 lanes with AVX2 integer ops.
    // k is integer-valued (from VROUNDPD), so float64->int32 truncation is exact.
    VCVTTPD2DQY Y2, X5                 // X5 = 4 x int32(k)
    VPMOVSXDQ X5, Y5                   // Y5 = 4 x int64(k) (sign-extended; k may be < 0)
    VPSLLQ $52, Y5, Y5                 // Y5 = k << 52 (into the exponent field)
    VPADDQ sigmoid_one64<>(SB), Y5, Y5 // Y5 = (k<<52) + bits(1.0) = bits(2^k)
    VMULPD Y5, Y4, Y4                  // Y4 = exp(z) = exp(-2x)

    // tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
    VSUBPD Y4, Y11, Y5                 // Y5 = 1 - exp(-2x)
    VADDPD Y4, Y11, Y6                 // Y6 = 1 + exp(-2x)
    VDIVPD Y6, Y5, Y0                  // Y0 = tanh(x)

    VMOVUPD Y0, (DX)                   // store 4 results
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ R8
    JNZ  tanh64_loop4

tanh64_remainder:
    ANDQ $3, CX                        // remainder = len % 4
    JZ   tanh64_done

tanh64_scalar:
    // Scalar path for remaining element
    VMOVSD (SI), X0                    // X0 = x

    // z = -2x
    VXORPD X1, X1, X1
    VSUBSD X0, X1, X0                  // X0 = -x
    VMULSD X8, X0, X0                  // X0 = -2x

    // Clamp to [-20, 20]
    MOVQ $0x4034000000000000, AX       // 20.0
    VMOVQ AX, X1
    MOVQ $0xC034000000000000, AX       // -20.0
    VMOVQ AX, X2
    VMINSD X1, X0, X0
    VMAXSD X2, X0, X0

    // Range reduction
    VMULSD X9, X0, X1                  // X1 = z * log2e
    VROUNDSD $0, X1, X1, X2            // X2 = k = round(X1)
    VMULSD X10, X2, X3                 // X3 = k * ln2
    VSUBSD X3, X0, X3                  // X3 = r = z - k * ln2

    // Horner's polynomial
    VMULSD X3, X15, X4                 // X4 = r * c5
    VADDSD X14, X4, X4
    VMULSD X3, X4, X4
    VADDSD X13, X4, X4
    VMULSD X3, X4, X4
    VADDSD X12, X4, X4
    VMULSD X3, X4, X4
    VADDSD X11, X4, X4
    VMULSD X3, X4, X4
    VADDSD X11, X4, X4                 // X4 = exp(r)

    // Reconstruct 2^k
    VCVTTSD2SI X2, AX                  // AX = int64(k)
    SHLQ $52, AX                       // AX = k << 52
    MOVQ $0x3FF0000000000000, BX       // 1.0's bits
    ADDQ BX, AX                        // AX = 2^k bits
    VMOVQ AX, X5
    VMULSD X5, X4, X4                  // X4 = exp(-2x)

    // tanh = (1 - exp) / (1 + exp)
    VSUBSD X4, X11, X5                 // X5 = 1 - exp
    VADDSD X4, X11, X6                 // X6 = 1 + exp
    VDIVSD X6, X5, X0                  // X0 = tanh

    VMOVSD X0, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  tanh64_scalar

tanh64_done:
    VZEROUPPER
    RET

// func expAVX(dst, src []float64)
// Computes e^x using range reduction and a degree-5 polynomial, the same exp
// core as tanhAVX but without the -2x scaling and the tanh wrap. Inputs are
// clamped to [-709, 709] to match the pure-Go fallback: results stay finite
// and large-negative inputs underflow to 0. Processes 4 elements per iteration
// on YMM. The 2^k reconstruction is vectorized with AVX2 integer ops
// (VCVTTPD2DQ/VPMOVSXDQ/VPSLLQ/VPADDQ), so this kernel must be dispatched only
// when cpu.X86.AVX2 is set; no FMA is used. The 0-3 element tail uses the
// scalar 1/iter path below.
TEXT ·expAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    // Load constants into YMM registers (all are 32-byte / 4-lane RODATA).
    // The low 128 bits (X9-X15) are reused by the scalar remainder path.
    VMOVUPD tanh64_log2e<>(SB), Y9     // Y9 = log2(e)
    VMOVUPD tanh64_ln2<>(SB), Y10      // Y10 = ln(2)
    VMOVUPD sigmoid_one64<>(SB), Y11   // Y11 = 1.0
    VMOVUPD sigmoid_half64<>(SB), Y12  // Y12 = 0.5 (c2)
    VMOVUPD tanh64_c3<>(SB), Y13       // Y13 = 1/6 (c3)
    VMOVUPD tanh64_c4<>(SB), Y14       // Y14 = 1/24 (c4)
    VMOVUPD tanh64_c5<>(SB), Y15       // Y15 = 1/120 (c5)

    // Clamp bounds are loop-invariant; hoist into Y7/Y8 (both free in this
    // kernel) so they are not reloaded from memory on every iteration.
    VMOVUPD exp_clamp_hi64<>(SB), Y7   // Y7 = 709.0
    VMOVUPD exp_clamp_lo64<>(SB), Y8   // Y8 = -709.0

    // Process 4 elements per iteration (YMM = 256 bits = 4 x float64)
    MOVQ CX, R8
    SHRQ $2, R8                        // len / 4
    JZ   exp64_remainder

exp64_loop4:
    VMOVUPD (SI), Y0                   // Y0 = 4 x float64

    // Clamp x to [-709, 709] (Y7=709, Y8=-709)
    VMINPD Y7, Y0, Y0
    VMAXPD Y8, Y0, Y0

    // Range reduction: k = round(x * log2e), r = x - k * ln2
    VMULPD Y9, Y0, Y1                  // Y1 = x * log2e
    VROUNDPD $0, Y1, Y2                // Y2 = k = round(Y1)
    VMULPD Y10, Y2, Y3                 // Y3 = k * ln2
    VSUBPD Y3, Y0, Y3                  // Y3 = r = x - k * ln2

    // Polynomial: exp(r) ~= 1 + r*(1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
    VMULPD Y3, Y15, Y4                 // Y4 = r * c5
    VADDPD Y14, Y4, Y4                 // Y4 = c4 + r*c5
    VMULPD Y3, Y4, Y4                  // Y4 = r*(c4 + r*c5)
    VADDPD Y13, Y4, Y4                 // Y4 = c3 + r*(...)
    VMULPD Y3, Y4, Y4                  // Y4 = r*(c3 + ...)
    VADDPD Y12, Y4, Y4                 // Y4 = c2 + r*(...)
    VMULPD Y3, Y4, Y4                  // Y4 = r*(c2 + ...)
    VADDPD Y11, Y4, Y4                 // Y4 = 1 + r*(...)
    VMULPD Y3, Y4, Y4                  // Y4 = r*(1 + ...)
    VADDPD Y11, Y4, Y4                 // Y4 = exp(r)

    // Reconstruct exp(x) = exp(r) * 2^k for all 4 lanes with AVX2 integer ops.
    // k is integer-valued (from VROUNDPD), so float64->int32 truncation is exact.
    VCVTTPD2DQY Y2, X5                 // X5 = 4 x int32(k)
    VPMOVSXDQ X5, Y5                   // Y5 = 4 x int64(k) (sign-extended; k may be < 0)
    VPSLLQ $52, Y5, Y5                 // Y5 = k << 52 (into the exponent field)
    VPADDQ sigmoid_one64<>(SB), Y5, Y5 // Y5 = (k<<52) + bits(1.0) = bits(2^k)
    VMULPD Y5, Y4, Y4                  // Y4 = exp(x)

    VMOVUPD Y4, (DX)                   // store 4 results
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ R8
    JNZ  exp64_loop4

exp64_remainder:
    ANDQ $3, CX
    JZ   exp64_done

exp64_scalar:
    VMOVSD (SI), X0                    // X0 = x

    // Clamp to [-709, 709] using the hoisted bounds (X7=709, X8=-709)
    VMINSD X7, X0, X0
    VMAXSD X8, X0, X0

    // Range reduction
    VMULSD X9, X0, X1
    VROUNDSD $0, X1, X1, X2            // X2 = k
    VMULSD X10, X2, X3
    VSUBSD X3, X0, X3                  // X3 = r

    // Polynomial
    VMULSD X3, X15, X4
    VADDSD X14, X4, X4
    VMULSD X3, X4, X4
    VADDSD X13, X4, X4
    VMULSD X3, X4, X4
    VADDSD X12, X4, X4
    VMULSD X3, X4, X4
    VADDSD X11, X4, X4
    VMULSD X3, X4, X4
    VADDSD X11, X4, X4                 // X4 = exp(r)

    // Reconstruct 2^k
    VCVTTSD2SI X2, AX                  // AX = int64(k)
    SHLQ $52, AX
    MOVQ $0x3FF0000000000000, BX
    ADDQ BX, AX
    VMOVQ AX, X5
    VMULSD X5, X4, X4                  // X4 = exp(x)

    VMOVSD X4, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  exp64_scalar

exp64_done:
    VZEROUPPER
    RET

// ============================================================================
// roundAVX: round-half-away-from-zero
// ============================================================================

// func roundAVX(dst, src []float64)
TEXT ·roundAVX(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    // Algorithm: round-half-away-from-zero matching math.Round.
    //   t   = trunc(x)
    //   f   = x - t                     // signed fractional part in (-1, 1)
    //   inc = (|f| >= 0.5) ? sign(x)*1.0 : 0
    //   result = t + inc
    // This avoids the trunc(|x|+0.5) overcounting at |x|=nextafter(0.5,0),
    // where the FP add rounds up to exactly 1.0.
    VMOVUPD roundf64_absmask<>(SB), Y3
    VMOVUPD roundf64_signmask<>(SB), Y4
    VMOVUPD roundf64_half<>(SB), Y5
    VMOVUPD roundf64_one<>(SB), Y11

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   round_avx_remainder

round_avx_loop4:
    VMOVUPD (SI), Y0                  // Y0 = x
    VROUNDPD $3, Y0, Y1                // Y1 = trunc(x)
    VSUBPD Y1, Y0, Y2                  // Y2 = x - trunc(x) = signed frac
    VANDPD Y3, Y2, Y6                  // Y6 = |frac|
    VCMPPD $5, Y5, Y6, Y7              // Y7 = mask: |frac| NLT 0.5  (|frac| >= 0.5)
    VANDPD Y11, Y7, Y8                 // Y8 = 1.0 where mask, else 0
    VANDPD Y4, Y0, Y9                  // Y9 = signbit(x)
    VORPD Y9, Y8, Y8                   // Y8 = ±1.0 (or ±0.0 when no increment)
    VADDPD Y8, Y1, Y1                  // Y1 = trunc(x) + ±1 or +0
    VMOVUPD Y1, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  round_avx_loop4

round_avx_remainder:
    ANDQ $3, CX
    JZ   round_avx_done

    VMOVSD roundf64_absmask<>(SB), X3
    VMOVSD roundf64_signmask<>(SB), X4
    VMOVSD roundf64_half<>(SB), X5
    VMOVSD roundf64_one<>(SB), X11

round_avx_scalar:
    VMOVSD (SI), X0                   // X0 = x
    VROUNDSD $3, X0, X0, X1            // X1 = trunc(x)
    VSUBSD X1, X0, X2                  // X2 = signed frac
    VANDPD X3, X2, X6                  // X6 = |frac|
    VCMPSD $5, X5, X6, X7              // X7 = mask: |frac| NLT 0.5
    VANDPD X11, X7, X8                 // X8 = 1.0 where mask else 0
    VANDPD X4, X0, X9                  // X9 = signbit(x)
    VORPD X9, X8, X8                   // X8 = ±1 or ±0
    VADDSD X8, X1, X1                  // X1 = trunc(x) + inc
    VMOVSD X1, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  round_avx_scalar

round_avx_done:
    VZEROUPPER
    RET

// func convolveDecimateAVX(dst, signal, kernel []float64, factor, phase int)
//
// Fused decimating valid convolution. For each output k it computes the dot
// product of signal[pos:pos+kLen] with kernel, then advances pos by factor.
// The inner dot replicates dotProductAVX exactly (4 accumulators, 16/4/scalar
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
    LEAQ (R10)(BX*8), SI          // SI = &signal[pos]
    MOVQ R11, DI                   // DI = &kernel[0]

    VXORPD Y0, Y0, Y0
    VXORPD Y3, Y3, Y3
    VXORPD Y4, Y4, Y4
    VXORPD Y5, Y5, Y5

    MOVQ R12, CX                   // CX = kLen
    MOVQ CX, AX
    SHRQ $4, AX                    // kLen / 16
    JZ   cd_avx_loop4_check

cd_avx_loop16:
    VMOVUPD (SI), Y1
    VMOVUPD (DI), Y2
    VFMADD231PD Y1, Y2, Y0
    VMOVUPD 32(SI), Y1
    VMOVUPD 32(DI), Y2
    VFMADD231PD Y1, Y2, Y3
    VMOVUPD 64(SI), Y1
    VMOVUPD 64(DI), Y2
    VFMADD231PD Y1, Y2, Y4
    VMOVUPD 96(SI), Y1
    VMOVUPD 96(DI), Y2
    VFMADD231PD Y1, Y2, Y5
    ADDQ $128, SI
    ADDQ $128, DI
    DECQ AX
    JNZ  cd_avx_loop16

    VADDPD Y3, Y0, Y0
    VADDPD Y4, Y0, Y0
    VADDPD Y5, Y0, Y0

cd_avx_loop4_check:
    MOVQ R12, CX
    ANDQ $15, CX
    MOVQ CX, AX
    SHRQ $2, AX
    JZ   cd_avx_reduce

cd_avx_loop4:
    VMOVUPD (SI), Y1
    VMOVUPD (DI), Y2
    VFMADD231PD Y1, Y2, Y0
    ADDQ $32, SI
    ADDQ $32, DI
    DECQ AX
    JNZ  cd_avx_loop4

cd_avx_reduce:
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0

    MOVQ R12, CX
    ANDQ $3, CX
    JZ   cd_avx_store

cd_avx_scalar:
    VMOVSD (SI), X1
    VMOVSD (DI), X2
    VFMADD231SD X1, X2, X0
    ADDQ $8, SI
    ADDQ $8, DI
    DECQ CX
    JNZ  cd_avx_scalar

cd_avx_store:
    VMOVSD X0, (R8)
    ADDQ $8, R8
    ADDQ R13, BX                  // pos += factor
    DECQ R9
    JNZ  cd_avx_outer

cd_avx_ret:
    VZEROUPPER
    RET

// func convolveDecimateAVX512(dst, signal, kernel []float64, factor, phase int)
//
// AVX-512 fused decimating valid convolution. Inner dot replicates
// dotProductAVX512 (4 Z accumulators, 32/8/scalar reduction) for bit-identical
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
    LEAQ (R10)(BX*8), SI
    MOVQ R11, DI

    VPXORQ Z0, Z0, Z0
    VPXORQ Z3, Z3, Z3
    VPXORQ Z4, Z4, Z4
    VPXORQ Z5, Z5, Z5

    MOVQ R12, CX
    MOVQ CX, AX
    SHRQ $5, AX                    // kLen / 32
    JZ   cd_avx512_loop8_check

cd_avx512_loop32:
    VMOVUPD (SI), Z1
    VMOVUPD (DI), Z2
    VFMADD231PD Z1, Z2, Z0
    VMOVUPD 64(SI), Z1
    VMOVUPD 64(DI), Z2
    VFMADD231PD Z1, Z2, Z3
    VMOVUPD 128(SI), Z1
    VMOVUPD 128(DI), Z2
    VFMADD231PD Z1, Z2, Z4
    VMOVUPD 192(SI), Z1
    VMOVUPD 192(DI), Z2
    VFMADD231PD Z1, Z2, Z5
    ADDQ $256, SI
    ADDQ $256, DI
    DECQ AX
    JNZ  cd_avx512_loop32

    VADDPD Z3, Z0, Z0
    VADDPD Z4, Z0, Z0
    VADDPD Z5, Z0, Z0

cd_avx512_loop8_check:
    MOVQ R12, CX
    ANDQ $31, CX
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   cd_avx512_reduce

cd_avx512_loop8:
    VMOVUPD (SI), Z1
    VMOVUPD (DI), Z2
    VFMADD231PD Z1, Z2, Z0
    ADDQ $64, SI
    ADDQ $64, DI
    DECQ AX
    JNZ  cd_avx512_loop8

cd_avx512_reduce:
    VEXTRACTF64X4 $1, Z0, Y1
    VADDPD Y1, Y0, Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    VHADDPD X0, X0, X0

    MOVQ R12, CX
    ANDQ $7, CX
    JZ   cd_avx512_store

cd_avx512_scalar:
    VMOVSD (SI), X1
    VMOVSD (DI), X2
    VFMADD231SD X1, X2, X0
    ADDQ $8, SI
    ADDQ $8, DI
    DECQ CX
    JNZ  cd_avx512_scalar

cd_avx512_store:
    VMOVSD X0, (R8)
    ADDQ $8, R8
    ADDQ R13, BX
    DECQ R9
    JNZ  cd_avx512_outer

cd_avx512_ret:
    VZEROUPPER
    RET

// func convolveDecimateSSE2(dst, signal, kernel []float64, factor, phase int)
//
// SSE2 fused decimating valid convolution (also used on the AVX-without-FMA
// path, which selects the SSE2 dot). Inner dot replicates dotProductSSE2
// (single accumulator, 2-wide loop, SHUFPD horizontal sum) for bit-identical
// results vs a per-window DotProductUnsafe on those paths.
TEXT ·convolveDecimateSSE2(SB), NOSPLIT, $0-88
    MOVQ dst_base+0(FP), R8
    MOVQ dst_len+8(FP), R9
    MOVQ signal_base+24(FP), R10
    MOVQ kernel_base+48(FP), R11
    MOVQ kernel_len+56(FP), R12
    MOVQ factor+72(FP), R13
    MOVQ phase+80(FP), BX

    TESTQ R9, R9
    JZ    cd_sse2_ret

cd_sse2_outer:
    LEAQ (R10)(BX*8), SI
    MOVQ R11, DI
    XORPD X0, X0

    MOVQ R12, CX
    MOVQ CX, AX
    SHRQ $1, AX                    // kLen / 2
    JZ   cd_sse2_reduce

cd_sse2_loop2:
    MOVUPD (SI), X1
    MOVUPD (DI), X2
    MULPD X2, X1
    ADDPD X1, X0
    ADDQ $16, SI
    ADDQ $16, DI
    DECQ AX
    JNZ  cd_sse2_loop2

cd_sse2_reduce:
    MOVAPD X0, X1
    SHUFPD $1, X1, X1
    ADDSD X1, X0

    MOVQ R12, CX
    ANDQ $1, CX
    JZ   cd_sse2_store

    MOVSD (SI), X1
    MOVSD (DI), X2
    MULSD X2, X1
    ADDSD X1, X0

cd_sse2_store:
    MOVSD X0, (R8)
    ADDQ $8, R8
    ADDQ R13, BX
    DECQ R9
    JNZ  cd_sse2_outer

cd_sse2_ret:
    RET

// ============================================================================
// logAVX / powAVX / powElemAVX: vectorized natural log core (issue #109)
// ============================================================================

// Shared log-core constants. The mantissa reduction is the musl /
// ARM-optimized-routines integer trick: tmp = bits(x) - OFF puts the biased
// exponent of m = x / 2^e in tmp's exponent field such that
// m in [sqrt(2)/2, sqrt(2)), with e = tmp >> 52 (arithmetic) and
// bits(m) = bits(x) - (tmp & 0xfff0000000000000). No compares or branches.
DATA log64_off<>+0x00(SB)/8, $0x3fe6a09e00000000
DATA log64_off<>+0x08(SB)/8, $0x3fe6a09e00000000
DATA log64_off<>+0x10(SB)/8, $0x3fe6a09e00000000
DATA log64_off<>+0x18(SB)/8, $0x3fe6a09e00000000
GLOBL log64_off<>(SB), RODATA|NOPTR, $32

DATA log64_expmask<>+0x00(SB)/8, $0xfff0000000000000
DATA log64_expmask<>+0x08(SB)/8, $0xfff0000000000000
DATA log64_expmask<>+0x10(SB)/8, $0xfff0000000000000
DATA log64_expmask<>+0x18(SB)/8, $0xfff0000000000000
GLOBL log64_expmask<>(SB), RODATA|NOPTR, $32

// DBL_MIN = 2.2250738585072014e-308: positive inputs below this are
// subnormal and pre-scaled by 2^64 (exponent bias -64) before the reduction.
DATA log64_dblmin<>+0x00(SB)/8, $0x0010000000000000
DATA log64_dblmin<>+0x08(SB)/8, $0x0010000000000000
DATA log64_dblmin<>+0x10(SB)/8, $0x0010000000000000
DATA log64_dblmin<>+0x18(SB)/8, $0x0010000000000000
GLOBL log64_dblmin<>(SB), RODATA|NOPTR, $32

DATA log64_two64<>+0x00(SB)/8, $0x43f0000000000000  // 2^64
DATA log64_two64<>+0x08(SB)/8, $0x43f0000000000000
DATA log64_two64<>+0x10(SB)/8, $0x43f0000000000000
DATA log64_two64<>+0x18(SB)/8, $0x43f0000000000000
GLOBL log64_two64<>(SB), RODATA|NOPTR, $32

DATA log64_negsc<>+0x00(SB)/8, $0xc050000000000000  // -64.0 (exponent bias)
DATA log64_negsc<>+0x08(SB)/8, $0xc050000000000000
DATA log64_negsc<>+0x10(SB)/8, $0xc050000000000000
DATA log64_negsc<>+0x18(SB)/8, $0xc050000000000000
GLOBL log64_negsc<>(SB), RODATA|NOPTR, $32

// VPERMD indices gathering the high (odd) dwords of the four int64 lanes
// into the low 128 bits, for the int32 -> float64 exponent conversion.
DATA log64_permidx<>+0x00(SB)/4, $1
DATA log64_permidx<>+0x04(SB)/4, $3
DATA log64_permidx<>+0x08(SB)/4, $5
DATA log64_permidx<>+0x0c(SB)/4, $7
DATA log64_permidx<>+0x10(SB)/4, $0
DATA log64_permidx<>+0x14(SB)/4, $0
DATA log64_permidx<>+0x18(SB)/4, $0
DATA log64_permidx<>+0x1c(SB)/4, $0
GLOBL log64_permidx<>(SB), RODATA|NOPTR, $32

// Minimax polynomial for ln(m), m in [sqrt(2)/2, sqrt(2)) (SLEEF xlog_u1
// coefficients): with s = (m-1)/(m+1) and t = s^2,
// ln(m) = 2s + s*t*P(t), P(t) = ((((((c0*t + c1)*t + c2)*t + c3)*t + c4)*t
// + c5)*t + c6). Worst-case relative error of the full ln(x) is ~2 ulps.
DATA log64_c0<>+0x00(SB)/8, $0x3fc39c4f5407567e  // 0.15320769885027014
DATA log64_c0<>+0x08(SB)/8, $0x3fc39c4f5407567e
DATA log64_c0<>+0x10(SB)/8, $0x3fc39c4f5407567e
DATA log64_c0<>+0x18(SB)/8, $0x3fc39c4f5407567e
GLOBL log64_c0<>(SB), RODATA|NOPTR, $32

DATA log64_c1<>+0x00(SB)/8, $0x3fc3872e67fe8e84  // 0.15256290510034287
DATA log64_c1<>+0x08(SB)/8, $0x3fc3872e67fe8e84
DATA log64_c1<>+0x10(SB)/8, $0x3fc3872e67fe8e84
DATA log64_c1<>+0x18(SB)/8, $0x3fc3872e67fe8e84
GLOBL log64_c1<>(SB), RODATA|NOPTR, $32

DATA log64_c2<>+0x00(SB)/8, $0x3fc747353a506035  // 0.1818605932937786
DATA log64_c2<>+0x08(SB)/8, $0x3fc747353a506035
DATA log64_c2<>+0x10(SB)/8, $0x3fc747353a506035
DATA log64_c2<>+0x18(SB)/8, $0x3fc747353a506035
GLOBL log64_c2<>(SB), RODATA|NOPTR, $32

DATA log64_c3<>+0x00(SB)/8, $0x3fcc71c0a65ecd8e  // 0.222221451983938
DATA log64_c3<>+0x08(SB)/8, $0x3fcc71c0a65ecd8e
DATA log64_c3<>+0x10(SB)/8, $0x3fcc71c0a65ecd8e
DATA log64_c3<>+0x18(SB)/8, $0x3fcc71c0a65ecd8e
GLOBL log64_c3<>(SB), RODATA|NOPTR, $32

DATA log64_c4<>+0x00(SB)/8, $0x3fd249249a68a245  // 0.28571429327942993
DATA log64_c4<>+0x08(SB)/8, $0x3fd249249a68a245
DATA log64_c4<>+0x10(SB)/8, $0x3fd249249a68a245
DATA log64_c4<>+0x18(SB)/8, $0x3fd249249a68a245
GLOBL log64_c4<>(SB), RODATA|NOPTR, $32

DATA log64_c5<>+0x00(SB)/8, $0x3fd99999998f92ea  // 0.3999999999635252
DATA log64_c5<>+0x08(SB)/8, $0x3fd99999998f92ea
DATA log64_c5<>+0x10(SB)/8, $0x3fd99999998f92ea
DATA log64_c5<>+0x18(SB)/8, $0x3fd99999998f92ea
GLOBL log64_c5<>(SB), RODATA|NOPTR, $32

DATA log64_c6<>+0x00(SB)/8, $0x3fe55555555557ae  // 0.66666666666673335
DATA log64_c6<>+0x08(SB)/8, $0x3fe55555555557ae
DATA log64_c6<>+0x10(SB)/8, $0x3fe55555555557ae
DATA log64_c6<>+0x18(SB)/8, $0x3fe55555555557ae
GLOBL log64_c6<>(SB), RODATA|NOPTR, $32

// fdlibm ln(2) hi/lo split for the pow kernels' fixed natural-log
// reconstruction (logAVX takes its split via arguments instead).
DATA log64_ln2hi<>+0x00(SB)/8, $0x3fe62e42fee00000
DATA log64_ln2hi<>+0x08(SB)/8, $0x3fe62e42fee00000
DATA log64_ln2hi<>+0x10(SB)/8, $0x3fe62e42fee00000
DATA log64_ln2hi<>+0x18(SB)/8, $0x3fe62e42fee00000
GLOBL log64_ln2hi<>(SB), RODATA|NOPTR, $32

DATA log64_ln2lo<>+0x00(SB)/8, $0x3dea39ef35793c76
DATA log64_ln2lo<>+0x08(SB)/8, $0x3dea39ef35793c76
DATA log64_ln2lo<>+0x10(SB)/8, $0x3dea39ef35793c76
DATA log64_ln2lo<>+0x18(SB)/8, $0x3dea39ef35793c76
GLOBL log64_ln2lo<>(SB), RODATA|NOPTR, $32

DATA log64_posinf<>+0x00(SB)/8, $0x7ff0000000000000
DATA log64_posinf<>+0x08(SB)/8, $0x7ff0000000000000
DATA log64_posinf<>+0x10(SB)/8, $0x7ff0000000000000
DATA log64_posinf<>+0x18(SB)/8, $0x7ff0000000000000
GLOBL log64_posinf<>(SB), RODATA|NOPTR, $32

DATA log64_neginf<>+0x00(SB)/8, $0xfff0000000000000
DATA log64_neginf<>+0x08(SB)/8, $0xfff0000000000000
DATA log64_neginf<>+0x10(SB)/8, $0xfff0000000000000
DATA log64_neginf<>+0x18(SB)/8, $0xfff0000000000000
GLOBL log64_neginf<>(SB), RODATA|NOPTR, $32

DATA log64_nan<>+0x00(SB)/8, $0x7ff8000000000000
DATA log64_nan<>+0x08(SB)/8, $0x7ff8000000000000
DATA log64_nan<>+0x10(SB)/8, $0x7ff8000000000000
DATA log64_nan<>+0x18(SB)/8, $0x7ff8000000000000
GLOBL log64_nan<>(SB), RODATA|NOPTR, $32

// func logAVX(dst, src []float64, k1hi, k1lo, k2 float64)
// Shared kernel for Log, Log2, and Log10: per lane it computes
// result = e*k1hi + (lnm*k2 + e*k1lo), with x = m*2^e, m in
// [sqrt(2)/2, sqrt(2)) and lnm = ln(m) = 2s + s*t*P(t) for s = (m-1)/(m+1),
// t = s^2 (atanh form, SLEEF xlog_u1 minimax polynomial). Positive subnormal
// inputs are pre-scaled by 2^64 (exponent bias -64). Special lanes are fixed
// up with blends from the original input: +Inf -> +Inf, +-0 -> -Inf,
// x < 0 or NaN -> NaN, matching math.Log. Requires AVX2 (YMM integer ops in
// the exponent extraction) and FMA. Processes 4 elements per iteration; the
// 0-3 element tail uses the scalar path below.
TEXT ·logAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    // Loop-invariant constants. The low 128 bits (X10, X13-X15) are reused
    // by the scalar remainder path.
    VMOVUPD log64_negsc<>(SB), Y6      // Y6 = -64.0 (subnormal exponent bias)
    VMOVUPD log64_two64<>(SB), Y7      // Y7 = 2^64 (subnormal pre-scale)
    VMOVUPD log64_dblmin<>(SB), Y8     // Y8 = DBL_MIN
    VMOVUPD log64_permidx<>(SB), Y9    // Y9 = VPERMD odd-dword gather indices
    VMOVUPD sigmoid_one64<>(SB), Y10   // Y10 = 1.0
    VMOVUPD log64_expmask<>(SB), Y11   // Y11 = 0xfff0000000000000
    VMOVUPD log64_off<>(SB), Y12       // Y12 = reduction offset
    VBROADCASTSD k2+64(FP), Y13        // Y13 = k2
    VBROADCASTSD k1lo+56(FP), Y14      // Y14 = k1lo
    VBROADCASTSD k1hi+48(FP), Y15      // Y15 = k1hi

    MOVQ CX, R8
    SHRQ $2, R8                        // len / 4
    JZ   log64_remainder

log64_loop4:
    VMOVUPD (SI), Y0                   // Y0 = x (kept for the special-lane blends)

    // Subnormal pre-scale: lanes with 0 < x < DBL_MIN are scaled by 2^64 and
    // carry an exponent bias of -64. (Negative/NaN lanes fail the compare or
    // produce garbage that the final blends overwrite.)
    VCMPPD $17, Y8, Y0, Y1             // Y1 = mask: x < DBL_MIN (LT_OQ, NaN -> false)
    VMULPD Y7, Y0, Y2                  // Y2 = x * 2^64
    VBLENDVPD Y1, Y2, Y0, Y2           // Y2 = xs = subnormal ? x*2^64 : x
    VANDPD Y6, Y1, Y1                  // Y1 = ebias = subnormal ? -64.0 : 0.0

    // Exponent/mantissa split (musl/ARM-optimized-routines integer trick):
    // tmp = bits(xs) - OFF; e = tmp >> 52 (arithmetic); bits(m) = bits(xs) -
    // (tmp & 0xfff0000000000000), leaving m in [sqrt(2)/2, sqrt(2)).
    VPSUBQ Y12, Y2, Y3                 // Y3 = tmp
    VPAND Y11, Y3, Y4                  // Y4 = tmp & expmask
    VPSUBQ Y4, Y2, Y4                  // Y4 = m
    // AVX2 has no VPSRAQ: arithmetic-shift the high dwords by 20 (giving
    // bits 52..63 of each lane sign-extended), gather them with VPERMD, and
    // convert int32 -> float64.
    VPSRAD $20, Y3, Y3                 // odd dwords = e (int32)
    VPERMD Y3, Y9, Y3                  // X3 = 4 x int32 e
    VCVTDQ2PD X3, Y3                   // Y3 = e as float64
    VADDPD Y1, Y3, Y3                  // Y3 = e + ebias

    // s = (m-1)/(m+1), t = s^2
    VSUBPD Y10, Y4, Y5                 // Y5 = m - 1
    VADDPD Y10, Y4, Y4                 // Y4 = m + 1
    VDIVPD Y4, Y5, Y5                  // Y5 = s
    VMULPD Y5, Y5, Y4                  // Y4 = t

    // P(t), Horner with memory-operand FMAs
    VMOVUPD log64_c0<>(SB), Y2
    VFMADD213PD log64_c1<>(SB), Y4, Y2 // Y2 = Y2*t + c1
    VFMADD213PD log64_c2<>(SB), Y4, Y2
    VFMADD213PD log64_c3<>(SB), Y4, Y2
    VFMADD213PD log64_c4<>(SB), Y4, Y2
    VFMADD213PD log64_c5<>(SB), Y4, Y2
    VFMADD213PD log64_c6<>(SB), Y4, Y2 // Y2 = P(t)

    VMULPD Y5, Y4, Y4                  // Y4 = s*t
    VADDPD Y5, Y5, Y5                  // Y5 = 2s
    VFMADD231PD Y2, Y4, Y5             // Y5 = lnm = s*t*P(t) + 2s

    // result = e*k1hi + (lnm*k2 + e*k1lo)
    VMULPD Y14, Y3, Y4                 // Y4 = e * k1lo
    VFMADD231PD Y13, Y5, Y4            // Y4 += lnm * k2
    VFMADD231PD Y15, Y3, Y4            // Y4 += e * k1hi

    // Special lanes from the original x: +Inf -> +Inf, +-0 -> -Inf,
    // x < 0 or NaN -> NaN (canonical quiet NaN, like math.Log).
    VMOVUPD log64_posinf<>(SB), Y2
    VCMPPD $0, Y2, Y0, Y1              // Y1 = mask: x == +Inf (EQ_OQ)
    VBLENDVPD Y1, Y2, Y4, Y4
    VXORPD Y2, Y2, Y2
    VCMPPD $0, Y2, Y0, Y1              // Y1 = mask: x == +-0
    VMOVUPD log64_neginf<>(SB), Y3
    VBLENDVPD Y1, Y3, Y4, Y4
    VCMPPD $17, Y2, Y0, Y1             // Y1 = mask: x < 0
    VCMPPD $3, Y0, Y0, Y2              // Y2 = mask: x unordered (NaN)
    VORPD Y2, Y1, Y1
    VMOVUPD log64_nan<>(SB), Y3
    VBLENDVPD Y1, Y3, Y4, Y4

    VMOVUPD Y4, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ R8
    JNZ  log64_loop4

log64_remainder:
    ANDQ $3, CX
    JZ   log64_done

log64_scalar:
    MOVQ (SI), AX                      // AX = bits(x)
    VMOVSD (SI), X0                    // X0 = x

    // Specials first: NaN or x < 0 -> NaN, +-0 -> -Inf, +Inf -> +Inf.
    // VUCOMISD sets ZF=PF=CF=1 for unordered, so test JP before JB/JE.
    VXORPD X1, X1, X1
    VUCOMISD X1, X0
    JP   log64_scalar_nan
    JB   log64_scalar_nan              // x < 0
    JE   log64_scalar_neginf           // x == +-0
    MOVQ $0x7FF0000000000000, BX
    CMPQ AX, BX
    JEQ  log64_scalar_posinf

    // Subnormal pre-scale (x is positive finite here; bits compare as ints)
    XORQ R9, R9                        // R9 = exponent bias
    MOVQ $0x0010000000000000, BX       // DBL_MIN bits
    CMPQ AX, BX
    JGE  log64_scalar_normal
    VMOVSD log64_two64<>(SB), X2       // 2^64
    VMULSD X2, X0, X0                  // x *= 2^64
    VMOVQ X0, AX
    MOVQ $-64, R9

log64_scalar_normal:
    MOVQ $0x3FE6A09E00000000, BX
    MOVQ AX, R10
    SUBQ BX, R10                       // R10 = tmp = bits - OFF
    MOVQ R10, R11
    SARQ $52, R11                      // R11 = e
    ADDQ R9, R11                       // e += bias
    MOVQ $0xFFF0000000000000, BX
    ANDQ BX, R10
    SUBQ R10, AX                       // AX = bits(m)
    VMOVQ AX, X2                       // X2 = m
    VCVTSI2SDQ R11, X3, X3             // X3 = e as float64

    // s = (m-1)/(m+1), t = s^2 (X10 = 1.0 from the vector constants)
    VSUBSD X10, X2, X4                 // m - 1
    VADDSD X10, X2, X5                 // m + 1
    VDIVSD X5, X4, X4                  // X4 = s
    VMULSD X4, X4, X5                  // X5 = t

    VMOVSD log64_c0<>(SB), X1
    VFMADD213SD log64_c1<>(SB), X5, X1
    VFMADD213SD log64_c2<>(SB), X5, X1
    VFMADD213SD log64_c3<>(SB), X5, X1
    VFMADD213SD log64_c4<>(SB), X5, X1
    VFMADD213SD log64_c5<>(SB), X5, X1
    VFMADD213SD log64_c6<>(SB), X5, X1 // X1 = P(t)

    VMULSD X4, X5, X5                  // X5 = s*t
    VADDSD X4, X4, X4                  // X4 = 2s
    VFMADD231SD X1, X5, X4             // X4 = lnm

    VMULSD X14, X3, X5                 // e * k1lo
    VFMADD231SD X13, X4, X5            // += lnm * k2
    VFMADD231SD X15, X3, X5            // += e * k1hi
    VMOVSD X5, (DX)
    JMP  log64_scalar_next

log64_scalar_nan:
    MOVQ $0x7FF8000000000000, AX
    MOVQ AX, (DX)
    JMP  log64_scalar_next

log64_scalar_neginf:
    MOVQ $0xFFF0000000000000, AX
    MOVQ AX, (DX)
    JMP  log64_scalar_next

log64_scalar_posinf:
    MOVQ $0x7FF0000000000000, AX
    MOVQ AX, (DX)

log64_scalar_next:
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  log64_scalar

log64_done:
    VZEROUPPER
    RET

// func powAVX(dst, src []float64, exp float64)
// Fused pow(x, p) = exp(p*ln(x)) for slices whose elements are all positive
// and finite (the dispatcher guarantees this, see powSIMDOK64). The log core
// matches logAVX (constants from memory instead of registers); the exp core
// matches expAVX except y = p*ln(x) is clamped to [-746, 710] (past
// ln(MaxFloat64) ~ 709.78 and ln of the smallest subnormal ~ -744.44) and
// the 2^k reconstruction is split into 2^(k>>1) * 2^(k-(k>>1)), so overflow
// goes to +Inf and underflow degrades gradually through subnormals to 0,
// matching math.Pow's result classes. Accuracy is bounded by the exp core's
// degree-5 polynomial (~3e-6 relative). Requires AVX2 and FMA.
TEXT ·powAVX(SB), NOSPLIT, $0-56
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    // Exp-core constants in registers (the log core uses memory operands).
    // The low 128 bits (X8-X15) are reused by the scalar remainder path.
    VMOVUPD log64_permidx<>(SB), Y7    // Y7 = VPERMD gather indices
    VBROADCASTSD exp+48(FP), Y8        // Y8 = p
    VMOVUPD tanh64_c5<>(SB), Y9        // Y9 = 1/120
    VMOVUPD tanh64_c4<>(SB), Y10       // Y10 = 1/24
    VMOVUPD tanh64_c3<>(SB), Y11       // Y11 = 1/6
    VMOVUPD sigmoid_half64<>(SB), Y12  // Y12 = 0.5
    VMOVUPD sigmoid_one64<>(SB), Y13   // Y13 = 1.0
    VMOVUPD tanh64_ln2<>(SB), Y14      // Y14 = ln(2)
    VMOVUPD tanh64_log2e<>(SB), Y15    // Y15 = log2(e)

    MOVQ CX, R8
    SHRQ $2, R8
    JZ   pow64_remainder

pow64_loop4:
    VMOVUPD (SI), Y0                   // Y0 = x (positive finite)

    // --- log core (see logAVX) ---
    VCMPPD $17, log64_dblmin<>(SB), Y0, Y1
    VMULPD log64_two64<>(SB), Y0, Y2
    VBLENDVPD Y1, Y2, Y0, Y0           // x, subnormals pre-scaled
    VANDPD log64_negsc<>(SB), Y1, Y1   // ebias
    VPSUBQ log64_off<>(SB), Y0, Y2     // tmp
    VPAND log64_expmask<>(SB), Y2, Y3
    VPSUBQ Y3, Y0, Y3                  // m
    VPSRAD $20, Y2, Y2
    VPERMD Y2, Y7, Y2
    VCVTDQ2PD X2, Y2
    VADDPD Y1, Y2, Y2                  // e
    VSUBPD Y13, Y3, Y4                 // m - 1
    VADDPD Y13, Y3, Y3                 // m + 1
    VDIVPD Y3, Y4, Y4                  // s
    VMULPD Y4, Y4, Y3                  // t
    VMOVUPD log64_c0<>(SB), Y5
    VFMADD213PD log64_c1<>(SB), Y3, Y5
    VFMADD213PD log64_c2<>(SB), Y3, Y5
    VFMADD213PD log64_c3<>(SB), Y3, Y5
    VFMADD213PD log64_c4<>(SB), Y3, Y5
    VFMADD213PD log64_c5<>(SB), Y3, Y5
    VFMADD213PD log64_c6<>(SB), Y3, Y5 // P(t)
    VMULPD Y4, Y3, Y3                  // s*t
    VADDPD Y4, Y4, Y4                  // 2s
    VFMADD231PD Y5, Y3, Y4             // lnm
    VMULPD log64_ln2lo<>(SB), Y2, Y3   // e*ln2lo
    VADDPD Y4, Y3, Y3                  // + lnm
    VFMADD231PD log64_ln2hi<>(SB), Y2, Y3 // Y3 = ln(x)

    // y = p*ln(x), clamped to [-746, 710]; the split 2^k reconstruction
    // below covers the whole result range
    VMULPD Y8, Y3, Y0
    VMINPD log64_powclamp_hi<>(SB), Y0, Y0
    VMAXPD log64_powclamp_lo<>(SB), Y0, Y0

    // --- exp core (see expAVX) ---
    VMULPD Y15, Y0, Y1                 // y * log2e
    VROUNDPD $0, Y1, Y2                // k
    VMULPD Y14, Y2, Y3                 // k * ln2
    VSUBPD Y3, Y0, Y3                  // r
    VMULPD Y3, Y9, Y4                  // r*c5
    VADDPD Y10, Y4, Y4
    VMULPD Y3, Y4, Y4
    VADDPD Y11, Y4, Y4
    VMULPD Y3, Y4, Y4
    VADDPD Y12, Y4, Y4
    VMULPD Y3, Y4, Y4
    VADDPD Y13, Y4, Y4
    VMULPD Y3, Y4, Y4
    VADDPD Y13, Y4, Y4                 // exp(r)
    // Split 2^k reconstruction: k reaches past the biased exponent range
    // (down to -1076), so build 2^(k>>1) and 2^(k-(k>>1)) separately. The
    // double multiply overflows to +Inf / underflows through subnormals to
    // 0 exactly where math.Pow does.
    VCVTTPD2DQY Y2, X5                 // X5 = 4 x int32(k)
    VPSRAD $1, X5, X3                  // k1 = k >> 1
    VPSUBD X3, X5, X5                  // k2 = k - k1
    VPMOVSXDQ X3, Y3
    VPSLLQ $52, Y3, Y3
    VPADDQ sigmoid_one64<>(SB), Y3, Y3 // 2^k1 bits
    VMULPD Y3, Y4, Y4                  // exp(r) * 2^k1
    VPMOVSXDQ X5, Y5
    VPSLLQ $52, Y5, Y5
    VPADDQ sigmoid_one64<>(SB), Y5, Y5 // 2^k2 bits
    VMULPD Y5, Y4, Y4                  // * 2^k2 = exp(y)

    VMOVUPD Y4, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ R8
    JNZ  pow64_loop4

pow64_remainder:
    ANDQ $3, CX
    JZ   pow64_done

pow64_scalar:
    MOVQ (SI), AX
    VMOVSD (SI), X0

    // log core, scalar (x positive finite; subnormal pre-scale via GPR)
    XORQ R9, R9
    MOVQ $0x0010000000000000, BX
    CMPQ AX, BX
    JGE  pow64_scalar_normal
    VMOVSD log64_two64<>(SB), X2       // 2^64
    VMULSD X2, X0, X0
    VMOVQ X0, AX
    MOVQ $-64, R9

pow64_scalar_normal:
    MOVQ $0x3FE6A09E00000000, BX
    MOVQ AX, R10
    SUBQ BX, R10                       // tmp
    MOVQ R10, R11
    SARQ $52, R11                      // e
    ADDQ R9, R11
    MOVQ $0xFFF0000000000000, BX
    ANDQ BX, R10
    SUBQ R10, AX                       // bits(m)
    VMOVQ AX, X2                       // m
    VCVTSI2SDQ R11, X3, X3             // e

    VSUBSD X13, X2, X4                 // m - 1 (X13 = 1.0)
    VADDSD X13, X2, X5                 // m + 1
    VDIVSD X5, X4, X4                  // s
    VMULSD X4, X4, X5                  // t
    VMOVSD log64_c0<>(SB), X1
    VFMADD213SD log64_c1<>(SB), X5, X1
    VFMADD213SD log64_c2<>(SB), X5, X1
    VFMADD213SD log64_c3<>(SB), X5, X1
    VFMADD213SD log64_c4<>(SB), X5, X1
    VFMADD213SD log64_c5<>(SB), X5, X1
    VFMADD213SD log64_c6<>(SB), X5, X1 // P(t)
    VMULSD X4, X5, X5                  // s*t
    VADDSD X4, X4, X4                  // 2s
    VFMADD231SD X1, X5, X4             // lnm
    VMULSD log64_ln2lo<>(SB), X3, X5
    VADDSD X4, X5, X5
    VFMADD231SD log64_ln2hi<>(SB), X3, X5 // X5 = ln(x)

    // y = p*ln(x), clamped (X8 = p from the vector constants)
    VMULSD X8, X5, X0
    VMINSD log64_powclamp_hi<>(SB), X0, X0
    VMAXSD log64_powclamp_lo<>(SB), X0, X0

    // exp core, scalar (X9-X15 = exp constants)
    VMULSD X15, X0, X1
    VROUNDSD $0, X1, X1, X2            // k
    VMULSD X14, X2, X3
    VSUBSD X3, X0, X3                  // r
    VMULSD X3, X9, X4
    VADDSD X10, X4, X4
    VMULSD X3, X4, X4
    VADDSD X11, X4, X4
    VMULSD X3, X4, X4
    VADDSD X12, X4, X4
    VMULSD X3, X4, X4
    VADDSD X13, X4, X4
    VMULSD X3, X4, X4
    VADDSD X13, X4, X4                 // exp(r)
    // Split 2^k reconstruction (see the vector body)
    VCVTTSD2SI X2, AX                  // k
    MOVQ AX, R10
    SARQ $1, R10                       // k1 = k >> 1
    SUBQ R10, AX                       // k2 = k - k1
    MOVQ $0x3FF0000000000000, BX
    SHLQ $52, R10
    ADDQ BX, R10
    VMOVQ R10, X5
    VMULSD X5, X4, X4                  // exp(r) * 2^k1
    SHLQ $52, AX
    ADDQ BX, AX
    VMOVQ AX, X5
    VMULSD X5, X4, X4                  // * 2^k2 = exp(y)

    VMOVSD X4, (DX)
    ADDQ $8, SI
    ADDQ $8, DX
    DECQ CX
    JNZ  pow64_scalar

pow64_done:
    VZEROUPPER
    RET

// func powElemAVX(dst, base, exp []float64)
// Elementwise pow(base[i], exp[i]) = exp(exp[i]*ln(base[i])). Same cores and
// preconditions as powAVX (all bases positive finite, all exponents finite),
// with the exponent loaded per lane instead of broadcast.
TEXT ·powElemAVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ base_base+24(FP), SI
    MOVQ exp_base+48(FP), DI

    VMOVUPD log64_permidx<>(SB), Y7    // Y7 = VPERMD gather indices
    VMOVUPD tanh64_c5<>(SB), Y9        // Y9 = 1/120
    VMOVUPD tanh64_c4<>(SB), Y10       // Y10 = 1/24
    VMOVUPD tanh64_c3<>(SB), Y11       // Y11 = 1/6
    VMOVUPD sigmoid_half64<>(SB), Y12  // Y12 = 0.5
    VMOVUPD sigmoid_one64<>(SB), Y13   // Y13 = 1.0
    VMOVUPD tanh64_ln2<>(SB), Y14      // Y14 = ln(2)
    VMOVUPD tanh64_log2e<>(SB), Y15    // Y15 = log2(e)

    MOVQ CX, R8
    SHRQ $2, R8
    JZ   powelem64_remainder

powelem64_loop4:
    VMOVUPD (SI), Y0                   // Y0 = base (positive finite)

    // --- log core (see logAVX) ---
    VCMPPD $17, log64_dblmin<>(SB), Y0, Y1
    VMULPD log64_two64<>(SB), Y0, Y2
    VBLENDVPD Y1, Y2, Y0, Y0
    VANDPD log64_negsc<>(SB), Y1, Y1   // ebias
    VPSUBQ log64_off<>(SB), Y0, Y2     // tmp
    VPAND log64_expmask<>(SB), Y2, Y3
    VPSUBQ Y3, Y0, Y3                  // m
    VPSRAD $20, Y2, Y2
    VPERMD Y2, Y7, Y2
    VCVTDQ2PD X2, Y2
    VADDPD Y1, Y2, Y2                  // e
    VSUBPD Y13, Y3, Y4                 // m - 1
    VADDPD Y13, Y3, Y3                 // m + 1
    VDIVPD Y3, Y4, Y4                  // s
    VMULPD Y4, Y4, Y3                  // t
    VMOVUPD log64_c0<>(SB), Y5
    VFMADD213PD log64_c1<>(SB), Y3, Y5
    VFMADD213PD log64_c2<>(SB), Y3, Y5
    VFMADD213PD log64_c3<>(SB), Y3, Y5
    VFMADD213PD log64_c4<>(SB), Y3, Y5
    VFMADD213PD log64_c5<>(SB), Y3, Y5
    VFMADD213PD log64_c6<>(SB), Y3, Y5 // P(t)
    VMULPD Y4, Y3, Y3                  // s*t
    VADDPD Y4, Y4, Y4                  // 2s
    VFMADD231PD Y5, Y3, Y4             // lnm
    VMULPD log64_ln2lo<>(SB), Y2, Y3
    VADDPD Y4, Y3, Y3
    VFMADD231PD log64_ln2hi<>(SB), Y2, Y3 // Y3 = ln(base)

    // y = exp[i]*ln(base[i]), clamped to [-746, 710]; the split 2^k
    // reconstruction below covers the whole result range
    VMOVUPD (DI), Y8                   // Y8 = exponents (finite)
    VMULPD Y8, Y3, Y0
    VMINPD log64_powclamp_hi<>(SB), Y0, Y0
    VMAXPD log64_powclamp_lo<>(SB), Y0, Y0

    // --- exp core (see expAVX) ---
    VMULPD Y15, Y0, Y1
    VROUNDPD $0, Y1, Y2                // k
    VMULPD Y14, Y2, Y3
    VSUBPD Y3, Y0, Y3                  // r
    VMULPD Y3, Y9, Y4
    VADDPD Y10, Y4, Y4
    VMULPD Y3, Y4, Y4
    VADDPD Y11, Y4, Y4
    VMULPD Y3, Y4, Y4
    VADDPD Y12, Y4, Y4
    VMULPD Y3, Y4, Y4
    VADDPD Y13, Y4, Y4
    VMULPD Y3, Y4, Y4
    VADDPD Y13, Y4, Y4                 // exp(r)
    // Split 2^k reconstruction (see powAVX)
    VCVTTPD2DQY Y2, X5                 // X5 = 4 x int32(k)
    VPSRAD $1, X5, X3                  // k1 = k >> 1
    VPSUBD X3, X5, X5                  // k2 = k - k1
    VPMOVSXDQ X3, Y3
    VPSLLQ $52, Y3, Y3
    VPADDQ sigmoid_one64<>(SB), Y3, Y3 // 2^k1 bits
    VMULPD Y3, Y4, Y4                  // exp(r) * 2^k1
    VPMOVSXDQ X5, Y5
    VPSLLQ $52, Y5, Y5
    VPADDQ sigmoid_one64<>(SB), Y5, Y5 // 2^k2 bits
    VMULPD Y5, Y4, Y4                  // * 2^k2 = exp(y)

    VMOVUPD Y4, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ R8
    JNZ  powelem64_loop4

powelem64_remainder:
    ANDQ $3, CX
    JZ   powelem64_done

powelem64_scalar:
    MOVQ (SI), AX
    VMOVSD (SI), X0

    XORQ R9, R9
    MOVQ $0x0010000000000000, BX
    CMPQ AX, BX
    JGE  powelem64_scalar_normal
    VMOVSD log64_two64<>(SB), X2       // 2^64
    VMULSD X2, X0, X0
    VMOVQ X0, AX
    MOVQ $-64, R9

powelem64_scalar_normal:
    MOVQ $0x3FE6A09E00000000, BX
    MOVQ AX, R10
    SUBQ BX, R10                       // tmp
    MOVQ R10, R11
    SARQ $52, R11                      // e
    ADDQ R9, R11
    MOVQ $0xFFF0000000000000, BX
    ANDQ BX, R10
    SUBQ R10, AX                       // bits(m)
    VMOVQ AX, X2
    VCVTSI2SDQ R11, X3, X3             // e

    VSUBSD X13, X2, X4                 // m - 1
    VADDSD X13, X2, X5                 // m + 1
    VDIVSD X5, X4, X4                  // s
    VMULSD X4, X4, X5                  // t
    VMOVSD log64_c0<>(SB), X1
    VFMADD213SD log64_c1<>(SB), X5, X1
    VFMADD213SD log64_c2<>(SB), X5, X1
    VFMADD213SD log64_c3<>(SB), X5, X1
    VFMADD213SD log64_c4<>(SB), X5, X1
    VFMADD213SD log64_c5<>(SB), X5, X1
    VFMADD213SD log64_c6<>(SB), X5, X1
    VMULSD X4, X5, X5                  // s*t
    VADDSD X4, X4, X4                  // 2s
    VFMADD231SD X1, X5, X4             // lnm
    VMULSD log64_ln2lo<>(SB), X3, X5
    VADDSD X4, X5, X5
    VFMADD231SD log64_ln2hi<>(SB), X3, X5 // ln(base)

    VMOVSD (DI), X8                    // p
    VMULSD X8, X5, X0
    VMINSD log64_powclamp_hi<>(SB), X0, X0
    VMAXSD log64_powclamp_lo<>(SB), X0, X0

    VMULSD X15, X0, X1
    VROUNDSD $0, X1, X1, X2
    VMULSD X14, X2, X3
    VSUBSD X3, X0, X3
    VMULSD X3, X9, X4
    VADDSD X10, X4, X4
    VMULSD X3, X4, X4
    VADDSD X11, X4, X4
    VMULSD X3, X4, X4
    VADDSD X12, X4, X4
    VMULSD X3, X4, X4
    VADDSD X13, X4, X4
    VMULSD X3, X4, X4
    VADDSD X13, X4, X4
    // Split 2^k reconstruction (see powAVX)
    VCVTTSD2SI X2, AX                  // k
    MOVQ AX, R10
    SARQ $1, R10                       // k1 = k >> 1
    SUBQ R10, AX                       // k2 = k - k1
    MOVQ $0x3FF0000000000000, BX
    SHLQ $52, R10
    ADDQ BX, R10
    VMOVQ R10, X5
    VMULSD X5, X4, X4                  // exp(r) * 2^k1
    SHLQ $52, AX
    ADDQ BX, AX
    VMOVQ AX, X5
    VMULSD X5, X4, X4                  // * 2^k2 = exp(y)

    VMOVSD X4, (DX)
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  powelem64_scalar

powelem64_done:
    VZEROUPPER
    RET

// Pow clamp bounds: wider than the Exp kernel's +-709 because the pow
// kernels split the 2^k reconstruction (2^(k>>1) * 2^(k-(k>>1))), which
// covers the full float64 result range. ln(MaxFloat64) ~ 709.78,
// ln(smallest subnormal) ~ -744.44.
DATA log64_powclamp_hi<>+0x00(SB)/8, $0x4086300000000000  // 710.0
DATA log64_powclamp_hi<>+0x08(SB)/8, $0x4086300000000000
DATA log64_powclamp_hi<>+0x10(SB)/8, $0x4086300000000000
DATA log64_powclamp_hi<>+0x18(SB)/8, $0x4086300000000000
GLOBL log64_powclamp_hi<>(SB), RODATA|NOPTR, $32

DATA log64_powclamp_lo<>+0x00(SB)/8, $0xc087500000000000  // -746.0
DATA log64_powclamp_lo<>+0x08(SB)/8, $0xc087500000000000
DATA log64_powclamp_lo<>+0x10(SB)/8, $0xc087500000000000
DATA log64_powclamp_lo<>+0x18(SB)/8, $0xc087500000000000
GLOBL log64_powclamp_lo<>(SB), RODATA|NOPTR, $32
