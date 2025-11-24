//go:build arm64

#include "textflag.h"

// ARM64 NEON implementation for float64 operations
// NEON processes 2 x float64 per vector register (128-bit)
// All vector instructions use WORD opcodes since Go's ARM64 assembler
// doesn't support NEON mnemonics directly.

// Opcode formulas for float64 (2D arrangement):
// FADD Vd.2D, Vn.2D, Vm.2D: 0x4E60D400 | (Vm << 16) | (Vn << 5) | Vd
// FSUB Vd.2D, Vn.2D, Vm.2D: 0x4EE0D400 | (Vm << 16) | (Vn << 5) | Vd
// FMUL Vd.2D, Vn.2D, Vm.2D: 0x6E60DC00 | (Vm << 16) | (Vn << 5) | Vd
// FDIV Vd.2D, Vn.2D, Vm.2D: 0x6E60FC00 | (Vm << 16) | (Vn << 5) | Vd
// FMIN Vd.2D, Vn.2D, Vm.2D: 0x4EE0F400 | (Vm << 16) | (Vn << 5) | Vd
// FMAX Vd.2D, Vn.2D, Vm.2D: 0x4E60F400 | (Vm << 16) | (Vn << 5) | Vd
// FABS Vd.2D, Vn.2D:        0x4EE0F800 | (Vn << 5) | Vd
// FNEG Vd.2D, Vn.2D:        0x6EE0F800 | (Vn << 5) | Vd
// FMLA Vd.2D, Vn.2D, Vm.2D: 0x4E60CC00 | (Vm << 16) | (Vn << 5) | Vd
// FADDP Dd, Vn.2D:          0x7E70D800 | (Vn << 5) | Vd

// func dotProductNEON(a, b []float64) float64
// Handles mismatched slice lengths: uses min(len(a), len(b)).
TEXT ·dotProductNEON(SB), NOSPLIT, $0-56
    MOVD a_base+0(FP), R0      // R0 = &a[0]
    MOVD a_len+8(FP), R2       // R2 = len(a)
    MOVD b_len+32(FP), R3      // R3 = len(b)
    CMP R3, R2
    CSEL LT, R2, R3, R2        // R2 = min(len(a), len(b))
    MOVD b_base+24(FP), R1     // R1 = &b[0]

    // V0, V1 = dual accumulators for ILP
    VEOR V0.B16, V0.B16, V0.B16
    VEOR V1.B16, V1.B16, V1.B16

    // Process 4 elements (2 NEON ops) per iteration
    LSR $2, R2, R3             // R3 = len / 4
    CBZ R3, dot_remainder2

dot_loop4:
    VLD1.P 16(R0), [V2.D2]     // Load a[i:i+2]
    VLD1.P 16(R0), [V3.D2]     // Load a[i+2:i+4]
    VLD1.P 16(R1), [V4.D2]     // Load b[i:i+2]
    VLD1.P 16(R1), [V5.D2]     // Load b[i+2:i+4]
    WORD $0x4E64CC40           // FMLA V0.2D, V2.2D, V4.2D
    WORD $0x4E65CC61           // FMLA V1.2D, V3.2D, V5.2D
    SUB $1, R3
    CBNZ R3, dot_loop4

    // Combine accumulators: V0 = V0 + V1
    WORD $0x4E61D400           // FADD V0.2D, V0.2D, V1.2D

dot_remainder2:
    // Check for 2-3 remaining elements
    AND $3, R2, R3
    LSR $1, R3, R4             // R4 = remainder / 2
    CBZ R4, dot_remainder1

    VLD1.P 16(R0), [V2.D2]
    VLD1.P 16(R1), [V4.D2]
    WORD $0x4E64CC40           // FMLA V0.2D, V2.2D, V4.2D

dot_remainder1:
    // Check for final single element
    AND $1, R3, R4
    CBZ R4, dot_reduce

    // Reduce vector FIRST before scalar ops (scalar ops zero upper V bits)
    WORD $0x7E70D800           // FADDP D0, V0.2D

    FMOVD (R0), F2
    FMOVD (R1), F4
    FMADDD F4, F0, F2, F0      // F0 = F2 * F4 + F0 (Go syntax: Fm, Fa, Fn, Fd)

    FMOVD F0, ret+48(FP)
    RET

dot_reduce:
    // Horizontal sum when no scalar remainder
    WORD $0x7E70D800           // FADDP D0, V0.2D

    FMOVD F0, ret+48(FP)
    RET

// func addNEON(dst, a, b []float64)
TEXT ·addNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $1, R3, R4
    CBZ R4, add_scalar

add_loop2:
    VLD1.P 16(R1), [V0.D2]
    VLD1.P 16(R2), [V1.D2]
    WORD $0x4E61D402           // FADD V2.2D, V0.2D, V1.2D
    VST1.P [V2.D2], 16(R0)
    SUB $1, R4
    CBNZ R4, add_loop2

add_scalar:
    AND $1, R3
    CBZ R3, add_done
    FMOVD (R1), F0
    FMOVD (R2), F1
    FADDD F0, F1, F0
    FMOVD F0, (R0)

add_done:
    RET

// func subNEON(dst, a, b []float64)
TEXT ·subNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $1, R3, R4
    CBZ R4, sub_scalar

sub_loop2:
    VLD1.P 16(R1), [V0.D2]
    VLD1.P 16(R2), [V1.D2]
    WORD $0x4EE1D402           // FSUB V2.2D, V0.2D, V1.2D
    VST1.P [V2.D2], 16(R0)
    SUB $1, R4
    CBNZ R4, sub_loop2

sub_scalar:
    AND $1, R3
    CBZ R3, sub_done
    FMOVD (R1), F0
    FMOVD (R2), F1
    FSUBD F1, F0, F0
    FMOVD F0, (R0)

sub_done:
    RET

// func mulNEON(dst, a, b []float64)
TEXT ·mulNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $1, R3, R4
    CBZ R4, mul_scalar

mul_loop2:
    VLD1.P 16(R1), [V0.D2]
    VLD1.P 16(R2), [V1.D2]
    WORD $0x6E61DC02           // FMUL V2.2D, V0.2D, V1.2D
    VST1.P [V2.D2], 16(R0)
    SUB $1, R4
    CBNZ R4, mul_loop2

mul_scalar:
    AND $1, R3
    CBZ R3, mul_done
    FMOVD (R1), F0
    FMOVD (R2), F1
    FMULD F0, F1, F0
    FMOVD F0, (R0)

mul_done:
    RET

// func divNEON(dst, a, b []float64)
TEXT ·divNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $1, R3, R4
    CBZ R4, div_scalar

div_loop2:
    VLD1.P 16(R1), [V0.D2]
    VLD1.P 16(R2), [V1.D2]
    WORD $0x6E61FC02           // FDIV V2.2D, V0.2D, V1.2D
    VST1.P [V2.D2], 16(R0)
    SUB $1, R4
    CBNZ R4, div_loop2

div_scalar:
    AND $1, R3
    CBZ R3, div_done
    FMOVD (R1), F0
    FMOVD (R2), F1
    FDIVD F1, F0, F0
    FMOVD F0, (R0)

div_done:
    RET

// func scaleNEON(dst, a []float64, s float64)
TEXT ·scaleNEON(SB), NOSPLIT, $0-56
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1
    FMOVD s+48(FP), F3
    // DUP V3.2D, V3.D[0] - broadcast scalar to both lanes
    WORD $0x4E080463           // DUP V3.2D, V3.D[0]

    LSR $1, R2, R3
    CBZ R3, scale_scalar

scale_loop2:
    VLD1.P 16(R1), [V0.D2]
    WORD $0x6E63DC01           // FMUL V1.2D, V0.2D, V3.2D
    VST1.P [V1.D2], 16(R0)
    SUB $1, R3
    CBNZ R3, scale_loop2

scale_scalar:
    AND $1, R2
    CBZ R2, scale_done
    FMOVD (R1), F0
    FMULD F0, F3, F0
    FMOVD F0, (R0)

scale_done:
    RET

// func addScalarNEON(dst, a []float64, s float64)
TEXT ·addScalarNEON(SB), NOSPLIT, $0-56
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1
    FMOVD s+48(FP), F3
    // DUP V3.2D, V3.D[0] - broadcast scalar to both lanes
    WORD $0x4E080463           // DUP V3.2D, V3.D[0]

    LSR $1, R2, R3
    CBZ R3, addsc_scalar

addsc_loop2:
    VLD1.P 16(R1), [V0.D2]
    WORD $0x4E63D401           // FADD V1.2D, V0.2D, V3.2D
    VST1.P [V1.D2], 16(R0)
    SUB $1, R3
    CBNZ R3, addsc_loop2

addsc_scalar:
    AND $1, R2
    CBZ R2, addsc_done
    FMOVD (R1), F0
    FADDD F0, F3, F0
    FMOVD F0, (R0)

addsc_done:
    RET

// func sumNEON(a []float64) float64
TEXT ·sumNEON(SB), NOSPLIT, $0-32
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1

    VEOR V0.B16, V0.B16, V0.B16

    LSR $1, R1, R2
    CBZ R2, sum_scalar

sum_loop2:
    VLD1.P 16(R0), [V1.D2]
    WORD $0x4E61D400           // FADD V0.2D, V0.2D, V1.2D
    SUB $1, R2
    CBNZ R2, sum_loop2

sum_scalar:
    AND $1, R1
    CBZ R1, sum_reduce

    // Reduce vector FIRST before scalar ops
    WORD $0x7E70D800           // FADDP D0, V0.2D
    FMOVD (R0), F1
    FADDD F0, F1, F0
    FMOVD F0, ret+24(FP)
    RET

sum_reduce:
    // No scalar remainder
    WORD $0x7E70D800           // FADDP D0, V0.2D
    FMOVD F0, ret+24(FP)
    RET

// func minNEON(a []float64) float64
TEXT ·minNEON(SB), NOSPLIT, $0-32
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1

    VLD1.P 16(R0), [V0.D2]     // Initialize with first 2
    SUB $2, R1

    LSR $1, R1, R2
    CBZ R2, min_scalar

min_loop2:
    VLD1.P 16(R0), [V1.D2]
    WORD $0x4EE1F400           // FMIN V0.2D, V0.2D, V1.2D
    SUB $1, R2
    CBNZ R2, min_loop2

min_scalar:
    AND $1, R1
    CBZ R1, min_reduce

    // Reduce vector FIRST before scalar ops
    VDUP V0.D[1], V1.D2
    FMIND F0, F1, F0
    FMOVD (R0), F1
    FMIND F0, F1, F0
    FMOVD F0, ret+24(FP)
    RET

min_reduce:
    // No scalar remainder
    VDUP V0.D[1], V1.D2
    FMIND F0, F1, F0
    FMOVD F0, ret+24(FP)
    RET

// func maxNEON(a []float64) float64
TEXT ·maxNEON(SB), NOSPLIT, $0-32
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1

    VLD1.P 16(R0), [V0.D2]
    SUB $2, R1

    LSR $1, R1, R2
    CBZ R2, max_scalar

max_loop2:
    VLD1.P 16(R0), [V1.D2]
    WORD $0x4E61F400           // FMAX V0.2D, V0.2D, V1.2D
    SUB $1, R2
    CBNZ R2, max_loop2

max_scalar:
    AND $1, R1
    CBZ R1, max_reduce

    // Reduce vector FIRST before scalar ops
    VDUP V0.D[1], V1.D2
    FMAXD F0, F1, F0
    FMOVD (R0), F1
    FMAXD F0, F1, F0
    FMOVD F0, ret+24(FP)
    RET

max_reduce:
    // No scalar remainder
    VDUP V0.D[1], V1.D2
    FMAXD F0, F1, F0
    FMOVD F0, ret+24(FP)
    RET

// func absNEON(dst, a []float64)
TEXT ·absNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1

    LSR $1, R2, R3
    CBZ R3, abs_scalar

abs_loop2:
    VLD1.P 16(R1), [V0.D2]
    WORD $0x4EE0F801           // FABS V1.2D, V0.2D
    VST1.P [V1.D2], 16(R0)
    SUB $1, R3
    CBNZ R3, abs_loop2

abs_scalar:
    AND $1, R2
    CBZ R2, abs_done
    FMOVD (R1), F0
    FABSD F0, F0
    FMOVD F0, (R0)

abs_done:
    RET

// func negNEON(dst, a []float64)
TEXT ·negNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1

    LSR $1, R2, R3
    CBZ R3, neg_scalar

neg_loop2:
    VLD1.P 16(R1), [V0.D2]
    WORD $0x6EE0F801           // FNEG V1.2D, V0.2D
    VST1.P [V1.D2], 16(R0)
    SUB $1, R3
    CBNZ R3, neg_loop2

neg_scalar:
    AND $1, R2
    CBZ R2, neg_done
    FMOVD (R1), F0
    FNEGD F0, F0
    FMOVD F0, (R0)

neg_done:
    RET

// func fmaNEON(dst, a, b, c []float64)
TEXT ·fmaNEON(SB), NOSPLIT, $0-96
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R4
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2
    MOVD c_base+72(FP), R3

    LSR $1, R4, R5
    CBZ R5, fma_scalar

fma_loop2:
    VLD1.P 16(R1), [V0.D2]     // a
    VLD1.P 16(R2), [V1.D2]     // b
    VLD1.P 16(R3), [V2.D2]     // c
    WORD $0x4E61CC02           // FMLA V2.2D, V0.2D, V1.2D
    VST1.P [V2.D2], 16(R0)
    SUB $1, R5
    CBNZ R5, fma_loop2

fma_scalar:
    AND $1, R4
    CBZ R4, fma_done
    FMOVD (R1), F0              // a[i]
    FMOVD (R2), F1              // b[i]
    FMOVD (R3), F2              // c[i]
    FMADDD F1, F2, F0, F2       // F2 = F0 * F1 + F2 = a[i] * b[i] + c[i] (Go syntax: Fm, Fa, Fn, Fd)
    FMOVD F2, (R0)

fma_done:
    RET

// func clampNEON(dst, a []float64, minVal, maxVal float64)
TEXT ·clampNEON(SB), NOSPLIT, $0-64
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1
    FMOVD minVal+48(FP), F2
    FMOVD maxVal+56(FP), F3
    // DUP V2.2D, V2.D[0] and DUP V3.2D, V3.D[0]
    WORD $0x4E080442           // DUP V2.2D, V2.D[0]
    WORD $0x4E080463           // DUP V3.2D, V3.D[0]

    LSR $1, R2, R3
    CBZ R3, clamp_scalar

clamp_loop2:
    VLD1.P 16(R1), [V0.D2]
    WORD $0x4E62F400           // FMAX V0.2D, V0.2D, V2.2D (clamp to min)
    WORD $0x4EE3F400           // FMIN V0.2D, V0.2D, V3.2D (clamp to max)
    VST1.P [V0.D2], 16(R0)
    SUB $1, R3
    CBNZ R3, clamp_loop2

clamp_scalar:
    AND $1, R2
    CBZ R2, clamp_done
    FMOVD (R1), F0
    FMAXD F0, F2, F0
    FMIND F0, F3, F0
    FMOVD F0, (R0)

clamp_done:
    RET

// FSQRT Vd.2D, Vn.2D: 0x6EE1F800 | (Vn << 5) | Vd
// FRECPE Vd.2D, Vn.2D: 0x4EE1D800 | (Vn << 5) | Vd (approximate)
// FRECPS Vd.2D, Vn.2D, Vm.2D: 0x4E60FC00 | (Vm << 16) | (Vn << 5) | Vd (Newton-Raphson step)

// func sqrtNEON(dst, a []float64)
TEXT ·sqrtNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1

    LSR $1, R2, R3
    CBZ R3, sqrt64_scalar

sqrt64_loop2:
    VLD1.P 16(R1), [V0.D2]
    WORD $0x6EE1F801           // FSQRT V1.2D, V0.2D
    VST1.P [V1.D2], 16(R0)
    SUB $1, R3
    CBNZ R3, sqrt64_loop2

sqrt64_scalar:
    AND $1, R2
    CBZ R2, sqrt64_done
    FMOVD (R1), F0
    FSQRTD F0, F0
    FMOVD F0, (R0)

sqrt64_done:
    RET

// func reciprocalNEON(dst, a []float64)
// Uses division by 1.0 for full precision (FRECPE is only approximate)
TEXT ·reciprocalNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1

    // Load 1.0 into F3 and broadcast
    FMOVD $1.0, F3
    WORD $0x4E080463           // DUP V3.2D, V3.D[0]

    LSR $1, R2, R3
    CBZ R3, recip64_scalar

recip64_loop2:
    VLD1.P 16(R1), [V0.D2]
    WORD $0x6E60FC61           // FDIV V1.2D, V3.2D, V0.2D
    VST1.P [V1.D2], 16(R0)
    SUB $1, R3
    CBNZ R3, recip64_loop2

recip64_scalar:
    AND $1, R2
    CBZ R2, recip64_done
    FMOVD (R1), F0
    FDIVD F0, F3, F0           // F0 = 1.0 / a[i]
    FMOVD F0, (R0)

recip64_done:
    RET

// func varianceNEON(a []float64, mean float64) float64
TEXT ·varianceNEON(SB), NOSPLIT, $0-40
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1
    FMOVD mean+24(FP), F3
    // DUP V3.2D, V3.D[0] - broadcast mean to both lanes
    WORD $0x4E080463           // DUP V3.2D, V3.D[0]

    VEOR V0.B16, V0.B16, V0.B16     // Accumulator = 0

    LSR $1, R1, R2
    CBZ R2, var64_scalar

var64_loop2:
    VLD1.P 16(R0), [V1.D2]     // Load 2 elements
    WORD $0x4EE3D421           // FSUB V1.2D, V1.2D, V3.2D  (diff = a[i] - mean)
    WORD $0x4E61CC20           // FMLA V0.2D, V1.2D, V1.2D  (acc += diff * diff)
    SUB $1, R2
    CBNZ R2, var64_loop2

var64_scalar:
    AND $1, R1
    CBZ R1, var64_reduce

    // Reduce vector FIRST before scalar ops
    WORD $0x7E70D800           // FADDP D0, V0.2D
    FMOVD (R0), F1
    FSUBD F3, F1, F1           // diff = a[i] - mean
    FMADDD F1, F0, F1, F0      // acc += diff * diff
    B var64_div

var64_reduce:
    WORD $0x7E70D800           // FADDP D0, V0.2D

var64_div:
    // Divide by n
    MOVD a_len+8(FP), R1
    SCVTFD R1, F1              // Convert n to float64
    FDIVD F1, F0, F0           // variance = sum / n
    FMOVD F0, ret+32(FP)
    RET

// func euclideanDistanceNEON(a, b []float64) float64
TEXT ·euclideanDistanceNEON(SB), NOSPLIT, $0-56
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R2
    MOVD b_base+24(FP), R1

    VEOR V0.B16, V0.B16, V0.B16     // Accumulator = 0

    LSR $1, R2, R3
    CBZ R3, euclid64_scalar

euclid64_loop2:
    VLD1.P 16(R0), [V1.D2]     // Load a[i:i+2]
    VLD1.P 16(R1), [V2.D2]     // Load b[i:i+2]
    WORD $0x4EE2D421           // FSUB V1.2D, V1.2D, V2.2D  (diff = a[i] - b[i])
    WORD $0x4E61CC20           // FMLA V0.2D, V1.2D, V1.2D  (acc += diff * diff)
    SUB $1, R3
    CBNZ R3, euclid64_loop2

euclid64_scalar:
    AND $1, R2
    CBZ R2, euclid64_sqrt

    // Reduce vector FIRST before scalar ops
    WORD $0x7E70D800           // FADDP D0, V0.2D
    FMOVD (R0), F1
    FMOVD (R1), F2
    FSUBD F2, F1, F1           // diff = a[i] - b[i]
    FMADDD F1, F0, F1, F0      // acc += diff * diff
    FSQRTD F0, F0
    FMOVD F0, ret+48(FP)
    RET

euclid64_sqrt:
    WORD $0x7E70D800           // FADDP D0, V0.2D
    FSQRTD F0, F0
    FMOVD F0, ret+48(FP)
    RET

// func addScaledNEON(dst []float64, alpha float64, s []float64)
// dst[i] += alpha * s[i] - AXPY operation
TEXT ·addScaledNEON(SB), NOSPLIT, $0-56
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    FMOVD alpha+24(FP), F3
    MOVD s_base+32(FP), R1

    // Broadcast alpha to both lanes
    WORD $0x4E080463           // DUP V3.2D, V3.D[0]

    LSR $1, R2, R3
    CBZ R3, addscaled64_scalar

addscaled64_loop2:
    VLD1 (R0), [V0.D2]         // Load dst[i:i+2]
    VLD1.P 16(R1), [V1.D2]     // Load s[i:i+2]
    WORD $0x4E63CC20           // FMLA V0.2D, V1.2D, V3.2D  (dst += s * alpha)
    VST1.P [V0.D2], 16(R0)
    SUB $1, R3
    CBNZ R3, addscaled64_loop2

addscaled64_scalar:
    AND $1, R2
    CBZ R2, addscaled64_done
    FMOVD (R0), F0
    FMOVD (R1), F1
    FMADDD F3, F0, F1, F0      // dst += s * alpha (Go: Fm, Fa, Fn, Fd -> Fd = Fn*Fm + Fa)
    FMOVD F0, (R0)

addscaled64_done:
    RET

// Interleave/Deinterleave with NEON ZIP/UZP instructions
// ZIP1 Vd.2D, Vn.2D, Vm.2D: 0x4EC03800 | (Vm << 16) | (Vn << 5) | Vd
// ZIP2 Vd.2D, Vn.2D, Vm.2D: 0x4EC07800 | (Vm << 16) | (Vn << 5) | Vd
// UZP1 Vd.2D, Vn.2D, Vm.2D: 0x4EC01800 | (Vm << 16) | (Vn << 5) | Vd
// UZP2 Vd.2D, Vn.2D, Vm.2D: 0x4EC05800 | (Vm << 16) | (Vn << 5) | Vd

// func interleave2NEON(dst, a, b []float64)
TEXT ·interleave2NEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0    // dst pointer
    MOVD a_base+24(FP), R1     // a pointer
    MOVD a_len+32(FP), R3      // n = len(a)
    MOVD b_base+48(FP), R2     // b pointer

    // Process 2 pairs at a time
    LSR $1, R3, R4             // R4 = n / 2
    CBZ R4, interleave2_neon_remainder

interleave2_neon_loop2:
    VLD1.P 16(R1), [V0.D2]     // V0 = [a0, a1]
    VLD1.P 16(R2), [V1.D2]     // V1 = [b0, b1]
    WORD $0x4EC13802           // ZIP1 V2.2D, V0.2D, V1.2D -> [a0, b0]
    WORD $0x4EC17803           // ZIP2 V3.2D, V0.2D, V1.2D -> [a1, b1]
    VST1.P [V2.D2], 16(R0)     // Store [a0, b0]
    VST1.P [V3.D2], 16(R0)     // Store [a1, b1]
    SUB $1, R4
    CBNZ R4, interleave2_neon_loop2

interleave2_neon_remainder:
    AND $1, R3
    CBZ R3, interleave2_neon_done

    FMOVD (R1), F0
    FMOVD (R2), F1
    FMOVD F0, (R0)
    FMOVD F1, 8(R0)

interleave2_neon_done:
    RET

// func deinterleave2NEON(a, b, src []float64)
TEXT ·deinterleave2NEON(SB), NOSPLIT, $0-72
    MOVD a_base+0(FP), R0      // a pointer
    MOVD a_len+8(FP), R3       // n = len(a)
    MOVD b_base+24(FP), R1     // b pointer
    MOVD src_base+48(FP), R2   // src pointer

    // Process 2 pairs at a time
    LSR $1, R3, R4             // R4 = n / 2
    CBZ R4, deinterleave2_neon_remainder

deinterleave2_neon_loop2:
    VLD1.P 16(R2), [V0.D2]     // V0 = [a0, b0]
    VLD1.P 16(R2), [V1.D2]     // V1 = [a1, b1]
    WORD $0x4EC11802           // UZP1 V2.2D, V0.2D, V1.2D -> [a0, a1]
    WORD $0x4EC15803           // UZP2 V3.2D, V0.2D, V1.2D -> [b0, b1]
    VST1.P [V2.D2], 16(R0)     // Store a
    VST1.P [V3.D2], 16(R1)     // Store b
    SUB $1, R4
    CBNZ R4, deinterleave2_neon_loop2

deinterleave2_neon_remainder:
    AND $1, R3
    CBZ R3, deinterleave2_neon_done

    FMOVD (R2), F0
    FMOVD 8(R2), F1
    FMOVD F0, (R0)
    FMOVD F1, (R1)

deinterleave2_neon_done:
    RET

// ============================================================================
// CUBIC INTERPOLATION DOT PRODUCT
// ============================================================================

// func cubicInterpDotNEON(hist, a, b, c, d []float64, x float64) float64
// Computes: Σ hist[i] * (a[i] + x*(b[i] + x*(c[i] + x*d[i])))
// Uses Horner's method for polynomial evaluation with FMLA.
// Optimized with 2 independent accumulators for ILP.
//
// Frame layout (5 slices + 1 float64 + 1 return):
//   hist: base+0, len+8
//   a:    base+24, len+32
//   b:    base+48, len+56
//   c:    base+72, len+80
//   d:    base+96, len+104
//   x:    +120
//   ret:  +128
TEXT ·cubicInterpDotNEON(SB), NOSPLIT, $0-136
    MOVD hist_base+0(FP), R0   // R0 = hist pointer
    MOVD hist_len+8(FP), R6    // R6 = length
    MOVD a_base+24(FP), R1     // R1 = a pointer
    MOVD b_base+48(FP), R2     // R2 = b pointer
    MOVD c_base+72(FP), R3     // R3 = c pointer
    MOVD d_base+96(FP), R4     // R4 = d pointer
    FMOVD x+120(FP), F31       // F31 = x (scalar)

    // Broadcast x to both lanes of V31
    WORD $0x4E0807FF           // DUP V31.2D, V31.D[0]

    // Initialize dual accumulators to zero for ILP
    VEOR V0.B16, V0.B16, V0.B16  // acc0 = 0
    VEOR V1.B16, V1.B16, V1.B16  // acc1 = 0

    // Process 4 elements per iteration (2 NEON vectors)
    LSR $2, R6, R5             // R5 = len / 4
    CBZ R5, cubic_neon_loop2_check

cubic_neon_loop4:
    // Load first pair (2 elements)
    VLD1.P 16(R4), [V2.D2]     // V2 = d[i:i+2]
    VLD1.P 16(R3), [V3.D2]     // V3 = c[i:i+2]
    VLD1.P 16(R2), [V4.D2]     // V4 = b[i:i+2]
    VLD1.P 16(R1), [V5.D2]     // V5 = a[i:i+2]
    VLD1.P 16(R0), [V6.D2]     // V6 = hist[i:i+2]

    // Load second pair (2 elements)
    VLD1.P 16(R4), [V10.D2]    // V10 = d[i+2:i+4]
    VLD1.P 16(R3), [V11.D2]    // V11 = c[i+2:i+4]
    VLD1.P 16(R2), [V12.D2]    // V12 = b[i+2:i+4]
    VLD1.P 16(R1), [V13.D2]    // V13 = a[i+2:i+4]
    VLD1.P 16(R0), [V14.D2]    // V14 = hist[i+2:i+4]

    // Horner's method for first pair: coef = a + x*(b + x*(c + x*d))
    // FMLA Vd.2D, Vn.2D, Vm.2D: Vd = Vd + Vn * Vm
    // Step 1: V3 = c + d*x
    WORD $0x4E7FCC43           // FMLA V3.2D, V2.2D, V31.2D
    // Step 2: V4 = b + (c + d*x)*x
    WORD $0x4E7FCC64           // FMLA V4.2D, V3.2D, V31.2D
    // Step 3: V5 = a + (b + (c + d*x)*x)*x = coef
    WORD $0x4E7FCC85           // FMLA V5.2D, V4.2D, V31.2D
    // Accumulate: acc0 += hist * coef
    WORD $0x4E65CCC0           // FMLA V0.2D, V6.2D, V5.2D

    // Horner's method for second pair
    WORD $0x4E7FCD4B           // FMLA V11.2D, V10.2D, V31.2D
    WORD $0x4E7FCD6C           // FMLA V12.2D, V11.2D, V31.2D
    WORD $0x4E7FCD8D           // FMLA V13.2D, V12.2D, V31.2D
    // Accumulate: acc1 += hist * coef
    WORD $0x4E6DCDC1           // FMLA V1.2D, V14.2D, V13.2D

    SUB $1, R5
    CBNZ R5, cubic_neon_loop4

    // Combine accumulators: V0 = V0 + V1
    WORD $0x4E61D400           // FADD V0.2D, V0.2D, V1.2D

cubic_neon_loop2_check:
    // Check for 2-3 remaining elements
    AND $3, R6, R5
    LSR $1, R5, R7             // R7 = remainder / 2
    CBZ R7, cubic_neon_remainder1

    // Process 2 elements
    VLD1.P 16(R4), [V2.D2]     // V2 = d
    VLD1.P 16(R3), [V3.D2]     // V3 = c
    VLD1.P 16(R2), [V4.D2]     // V4 = b
    VLD1.P 16(R1), [V5.D2]     // V5 = a
    VLD1.P 16(R0), [V6.D2]     // V6 = hist

    // Horner's method
    WORD $0x4E7FCC43           // FMLA V3.2D, V2.2D, V31.2D
    WORD $0x4E7FCC64           // FMLA V4.2D, V3.2D, V31.2D
    WORD $0x4E7FCC85           // FMLA V5.2D, V4.2D, V31.2D
    WORD $0x4E65CCC0           // FMLA V0.2D, V6.2D, V5.2D

cubic_neon_remainder1:
    // Reduce vector FIRST before scalar ops
    WORD $0x7E70D800           // FADDP D0, V0.2D

    // Check for final single element
    AND $1, R5, R7
    CBZ R7, cubic_neon_done

    // Scalar path for single element
    FMOVD (R4), F2             // d
    FMOVD (R3), F3             // c
    FMOVD (R2), F4             // b
    FMOVD (R1), F5             // a
    FMOVD (R0), F6             // hist
    FMOVD x+120(FP), F7        // x

    // Horner's method scalar: coef = a + x*(b + x*(c + x*d))
    // Go FMADDD syntax: Fm, Fa, Fn, Fd -> Fd = Fn * Fm + Fa
    FMADDD F7, F3, F2, F3      // F3 = F2*F7 + F3 = d*x + c
    FMADDD F7, F4, F3, F4      // F4 = F3*F7 + F4 = (d*x+c)*x + b
    FMADDD F7, F5, F4, F5      // F5 = F4*F7 + F5 = coef
    FMADDD F5, F0, F6, F0      // F0 = F6*F5 + F0 = hist*coef + acc

cubic_neon_done:
    FMOVD F0, ret+128(FP)
    RET

// func sigmoidNEON64(dst, src []float64)
// Implements fast sigmoid approximation: σ(x) ≈ 0.5 + 0.5 * x / (1 + |x|)
// This approximation is SIMD-friendly and commonly used in neural networks.
TEXT ·sigmoidNEON64(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1

    // Load constants into vector registers
    FMOVD $0.5, F30
    FMOVD $1.0, F31
    VDUP V30.D[0], V30.D2         // V30 = {0.5, 0.5}
    VDUP V31.D[0], V31.D2         // V31 = {1.0, 1.0}

    // Process 2 elements per iteration
    LSR $1, R3, R4
    CBZ R4, sigmoid64_neon_scalar

sigmoid64_neon_loop2:
    VLD1.P 16(R1), [V0.D2]        // V0 = x
    WORD $0x4EE0F801              // FABS V1.2D, V0.2D -> V1 = |x|
    WORD $0x4E7FD422              // FADD V2.2D, V1.2D, V31.2D -> V2 = 1 + |x|
    WORD $0x6E62FC03              // FDIV V3.2D, V0.2D, V2.2D -> V3 = x / (1 + |x|)
    WORD $0x6E7EDC64              // FMUL V4.2D, V3.2D, V30.2D -> V4 = 0.5 * x / (1 + |x|)
    WORD $0x4E7ED485              // FADD V5.2D, V4.2D, V30.2D -> V5 = 0.5 + result
    VST1.P [V5.D2], 16(R0)        // store result

    SUB $1, R4
    CBNZ R4, sigmoid64_neon_loop2

sigmoid64_neon_scalar:
    AND $1, R3
    CBZ R3, sigmoid64_neon_done

    FMOVD (R1), F0                // F0 = x
    FABSD F0, F1                  // F1 = |x|
    FADDD F31, F1, F2             // F2 = 1 + |x|
    FDIVD F2, F0, F3              // F3 = x / (1 + |x|)
    FMULD F30, F3, F4             // F4 = 0.5 * x / (1 + |x|)
    FADDD F30, F4, F5             // F5 = 0.5 + result
    FMOVD F5, (R0)                // store result

sigmoid64_neon_done:
    RET

// func reluNEON64(dst, src []float64)
// Computes ReLU: dst[i] = max(0, src[i])
TEXT ·reluNEON64(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1

    // Create zero vector
    VEOR V30.B16, V30.B16, V30.B16    // V30 = {0, 0}

    // Process 2 elements per iteration (float64 uses 2D arrangement)
    LSR $1, R3, R4
    CBZ R4, relu64_neon_scalar

relu64_neon_loop2:
    VLD1.P 16(R1), [V0.D2]            // V0 = x (2x float64)
    WORD $0x4E7EF401                  // FMAX V1.2D, V0.2D, V30.2D -> V1 = max(x, 0)
    VST1.P [V1.D2], 16(R0)            // store result

    SUB $1, R4
    CBNZ R4, relu64_neon_loop2

relu64_neon_scalar:
    AND $1, R3
    CBZ R3, relu64_neon_done

relu64_neon_scalar_loop:
    FMOVD (R1), F0                    // F0 = x
    FMOVD $0.0, F1
    FMAXD F1, F0, F2                  // F2 = max(x, 0)
    FMOVD F2, (R0)                    // store result

    ADD $8, R0
    ADD $8, R1
    SUB $1, R3
    CBNZ R3, relu64_neon_scalar_loop

relu64_neon_done:
    RET

// func tanhNEON64(dst, src []float64)
// Computes fast tanh approximation: tanh(x) ≈ x / (1 + |x|)
TEXT ·tanhNEON64(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1

    // Load constants for scalar processing
    FMOVD $1.0, F31
    FMOVD $2.5, F30
    FMOVD $-1.0, F29

    // Use scalar processing for all elements to avoid complex bit manipulation
    // The scalar loop handles saturation correctly and is still reasonably fast

tanh64_neon_scalar:
    CBZ R3, tanh64_neon_done

tanh64_neon_scalar_loop:
    FMOVD (R1), F0                    // F0 = x
    FABSD F0, F1                      // F1 = |x|
    FCMPD F1, F30                     // compare |x| with 2.5
    BLE tanh64_neon_scalar_approx

    // Saturate: return ±1.0 based on sign of x
    FMOVD $0, F7                      // F7 = 0.0
    FCMPD F0, F7                      // compare x with 0
    BGE tanh64_neon_scalar_positive

    // x < 0: return -1.0
    FMOVD F29, F3
    B tanh64_neon_scalar_store

tanh64_neon_scalar_positive:
    // x >= 0: return 1.0
    FMOVD F31, F3
    B tanh64_neon_scalar_store

tanh64_neon_scalar_approx:
    FADDD F31, F1, F2                 // F2 = 1 + |x|
    FDIVD F2, F0, F3                  // F3 = x / (1 + |x|)

tanh64_neon_scalar_store:
    FMOVD F3, (R0)                    // store result

    ADD $8, R0
    ADD $8, R1
    SUB $1, R3
    CBNZ R3, tanh64_neon_scalar_loop

tanh64_neon_done:
    RET

// func clampScaleNEON64(dst, src []float64, minVal, maxVal, scale float64)
// Performs fused clamp and scale: dst[i] = (clamp(src[i], minVal, maxVal) - minVal) * scale
TEXT ·clampScaleNEON64(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1
    FMOVD minVal+48(FP), F4
    FMOVD maxVal+56(FP), F5
    FMOVD scale+64(FP), F6

    // Duplicate scalars to SIMD vectors
    WORD $0x4E080484                  // DUP V4.2D, V4.D[0] -> V4 = minVal
    WORD $0x4E0804A5                  // DUP V5.2D, V5.D[0] -> V5 = maxVal
    WORD $0x4E0804C6                  // DUP V6.2D, V6.D[0] -> V6 = scale

    // Process 2 elements per iteration
    LSR $1, R3, R4
    CBZ R4, clampscale64_neon_scalar

clampscale64_neon_loop2:
    VLD1.P 16(R1), [V0.D2]            // V0 = src[i]
    WORD $0x4E64F400                  // FMAX V0.2D, V0.2D, V4.2D -> clamp to min
    WORD $0x4EE5F400                  // FMIN V0.2D, V0.2D, V5.2D -> clamp to max
    WORD $0x4EE4D400                  // FSUB V0.2D, V0.2D, V4.2D -> subtract minVal
    WORD $0x6E66DC00                  // FMUL V0.2D, V0.2D, V6.2D -> multiply by scale
    VST1.P [V0.D2], 16(R0)            // store result
    SUB $1, R4
    CBNZ R4, clampscale64_neon_loop2

clampscale64_neon_scalar:
    AND $1, R3
    CBZ R3, clampscale64_neon_done

clampscale64_neon_scalar_loop:
    FMOVD (R1), F0                    // F0 = src[i]
    FMAXD F0, F4, F0                  // F0 = max(src[i], minVal)
    FMIND F0, F5, F0                  // F0 = min(max(src[i], minVal), maxVal)
    FSUBD F4, F0, F0                  // F0 = clamped - minVal
    FMULD F0, F6, F0                  // F0 = (clamped - minVal) * scale
    FMOVD F0, (R0)                    // store result
    ADD $8, R0
    ADD $8, R1
    SUB $1, R3
    CBNZ R3, clampscale64_neon_scalar_loop

clampscale64_neon_done:
    RET
