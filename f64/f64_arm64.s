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
// FRINTN Vd.2D, Vn.2D:      0x4E618800 | (Vn << 5) | Vd  (round to nearest, ties to even)
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

// func interleave3NEON(dst, s0, s1, s2 []float64, n int)
// Interleaves 3 planar streams (dst[i*3+c] = s_c[i]) with the NEON ST3
// structured store, 2 frames per iteration, then a single scalar tail frame.
TEXT ·interleave3NEON(SB), NOSPLIT, $0-104
    MOVD dst_base+0(FP), R0
    MOVD s0_base+24(FP), R1
    MOVD s1_base+48(FP), R2
    MOVD s2_base+72(FP), R3
    MOVD n+96(FP), R4

    LSR $1, R4, R5             // R5 = n / 2
    CBZ R5, interleave3_neon_tail

interleave3_neon_loop2:
    VLD1.P 16(R1), [V0.D2]     // V0 = s0[i:i+2]
    VLD1.P 16(R2), [V1.D2]     // V1 = s1[i:i+2]
    VLD1.P 16(R3), [V2.D2]     // V2 = s2[i:i+2]
    VST3.P [V0.D2, V1.D2, V2.D2], 48(R0)  // interleaved store of 6 doubles
    SUB $1, R5
    CBNZ R5, interleave3_neon_loop2

interleave3_neon_tail:
    AND $1, R4
    CBZ R4, interleave3_neon_done
    FMOVD (R1), F0
    FMOVD (R2), F1
    FMOVD (R3), F2
    FMOVD F0, (R0)
    FMOVD F1, 8(R0)
    FMOVD F2, 16(R0)

interleave3_neon_done:
    RET

// func deinterleave3NEON(d0, d1, d2, src []float64, n int)
// Splits an interleaved 3-stream buffer (d_c[i] = src[i*3+c]) with the NEON LD3
// structured load, 2 frames per iteration, then a single scalar tail frame.
TEXT ·deinterleave3NEON(SB), NOSPLIT, $0-104
    MOVD d0_base+0(FP), R0
    MOVD d1_base+24(FP), R1
    MOVD d2_base+48(FP), R2
    MOVD src_base+72(FP), R3
    MOVD n+96(FP), R4

    LSR $1, R4, R5             // R5 = n / 2
    CBZ R5, deinterleave3_neon_tail

deinterleave3_neon_loop2:
    VLD3.P 48(R3), [V0.D2, V1.D2, V2.D2]  // de-interleave 6 doubles
    VST1.P [V0.D2], 16(R0)     // d0[i:i+2]
    VST1.P [V1.D2], 16(R1)     // d1[i:i+2]
    VST1.P [V2.D2], 16(R2)     // d2[i:i+2]
    SUB $1, R5
    CBNZ R5, deinterleave3_neon_loop2

deinterleave3_neon_tail:
    AND $1, R4
    CBZ R4, deinterleave3_neon_done
    FMOVD (R3), F0
    FMOVD 8(R3), F1
    FMOVD 16(R3), F2
    FMOVD F0, (R0)
    FMOVD F1, (R1)
    FMOVD F2, (R2)

deinterleave3_neon_done:
    RET

// func interleave4NEON(dst, s0, s1, s2, s3 []float64, n int)
// Interleaves 4 planar streams (dst[i*4+c] = s_c[i]) with the NEON ST4
// structured store, 2 frames per iteration, then a single scalar tail frame.
TEXT ·interleave4NEON(SB), NOSPLIT, $0-128
    MOVD dst_base+0(FP), R0
    MOVD s0_base+24(FP), R1
    MOVD s1_base+48(FP), R2
    MOVD s2_base+72(FP), R3
    MOVD s3_base+96(FP), R4
    MOVD n+120(FP), R5

    LSR $1, R5, R6             // R6 = n / 2
    CBZ R6, interleave4_neon_tail

interleave4_neon_loop2:
    VLD1.P 16(R1), [V0.D2]
    VLD1.P 16(R2), [V1.D2]
    VLD1.P 16(R3), [V2.D2]
    VLD1.P 16(R4), [V3.D2]
    VST4.P [V0.D2, V1.D2, V2.D2, V3.D2], 64(R0)  // interleaved store of 8 doubles
    SUB $1, R6
    CBNZ R6, interleave4_neon_loop2

interleave4_neon_tail:
    AND $1, R5
    CBZ R5, interleave4_neon_done
    FMOVD (R1), F0
    FMOVD (R2), F1
    FMOVD (R3), F2
    FMOVD (R4), F3
    FMOVD F0, (R0)
    FMOVD F1, 8(R0)
    FMOVD F2, 16(R0)
    FMOVD F3, 24(R0)

interleave4_neon_done:
    RET

// func deinterleave4NEON(d0, d1, d2, d3, src []float64, n int)
// Splits an interleaved 4-stream buffer (d_c[i] = src[i*4+c]) with the NEON LD4
// structured load, 2 frames per iteration, then a single scalar tail frame.
TEXT ·deinterleave4NEON(SB), NOSPLIT, $0-128
    MOVD d0_base+0(FP), R0
    MOVD d1_base+24(FP), R1
    MOVD d2_base+48(FP), R2
    MOVD d3_base+72(FP), R3
    MOVD src_base+96(FP), R4
    MOVD n+120(FP), R5

    LSR $1, R5, R6             // R6 = n / 2
    CBZ R6, deinterleave4_neon_tail

deinterleave4_neon_loop2:
    VLD4.P 64(R4), [V0.D2, V1.D2, V2.D2, V3.D2]  // de-interleave 8 doubles
    VST1.P [V0.D2], 16(R0)
    VST1.P [V1.D2], 16(R1)
    VST1.P [V2.D2], 16(R2)
    VST1.P [V3.D2], 16(R3)
    SUB $1, R6
    CBNZ R6, deinterleave4_neon_loop2

deinterleave4_neon_tail:
    AND $1, R5
    CBZ R5, deinterleave4_neon_done
    FMOVD (R4), F0
    FMOVD 8(R4), F1
    FMOVD 16(R4), F2
    FMOVD 24(R4), F3
    FMOVD F0, (R0)
    FMOVD F1, (R1)
    FMOVD F2, (R2)
    FMOVD F3, (R3)

deinterleave4_neon_done:
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
// Computes accurate sigmoid: sigmoid(x) = 1 / (1 + e^(-x)).
// Uses the same exp range reduction + degree-5 polynomial core as tanhNEON64,
// replacing the previous fast rational approximation (0.5 + 0.5*x/(1+|x|)),
// which was far less accurate than the f32 kernel. Inputs are clamped via
// z = -x in [-709, 709] so the 2^k reconstruction stays in range.
// Processes 2 elements per iteration (float64 uses D2 arrangement).
TEXT ·sigmoidNEON64(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1

    // log2(e) = 1.4426950408889634 = 0x3FF71547652B82FE
    MOVD $0x3FF71547652B82FE, R10
    VMOV R10, V20.D[0]
    VDUP V20.D[0], V20.D2             // V20 = log2(e)

    // ln(2) = 0.6931471805599453 = 0x3FE62E42FEFA39EF
    MOVD $0x3FE62E42FEFA39EF, R10
    VMOV R10, V21.D[0]
    VDUP V21.D[0], V21.D2             // V21 = ln(2)

    // 1.0
    FMOVD $1.0, F22
    VDUP V22.D[0], V22.D2             // V22 = 1.0

    // c2 = 0.5
    FMOVD $0.5, F23
    VDUP V23.D[0], V23.D2             // V23 = c2 = 0.5

    // c3 = 1/6 = 0x3FC5555555555555
    MOVD $0x3FC5555555555555, R10
    VMOV R10, V24.D[0]
    VDUP V24.D[0], V24.D2             // V24 = c3 = 1/6

    // c4 = 1/24 = 0x3FA5555555555555
    MOVD $0x3FA5555555555555, R10
    VMOV R10, V25.D[0]
    VDUP V25.D[0], V25.D2             // V25 = c4 = 1/24

    // c5 = 1/120 = 0x3F81111111111111
    MOVD $0x3F81111111111111, R10
    VMOV R10, V26.D[0]
    VDUP V26.D[0], V26.D2             // V26 = c5 = 1/120

    // Clamp thresholds for z = -x: ±709.0 (matches exp; keeps 2^k in range)
    // 709.0 = 0x4086280000000000
    MOVD $0x4086280000000000, R10
    VMOV R10, V27.D[0]
    VDUP V27.D[0], V27.D2             // V27 = 709.0 (clamp_hi)

    // -709.0 = 0xC086280000000000
    MOVD $0xC086280000000000, R10
    VMOV R10, V28.D[0]
    VDUP V28.D[0], V28.D2             // V28 = -709.0 (clamp_lo)

    // Process 2 elements per iteration (float64 uses D2 arrangement)
    LSR $1, R3, R4
    CBZ R4, sigmoid64_neon_scalar

sigmoid64_neon_loop2:
    VLD1.P 16(R1), [V0.D2]            // V0 = x

    // Compute z = -x
    WORD $0x6EE0F800                  // FNEG V0.2D, V0.2D           V0 = -x = z

    // Clamp z to [-709, 709]
    WORD $0x4EFBF400                  // FMIN V0.2D, V0.2D, V27.2D   clamp upper to 709
    WORD $0x4E7CF400                  // FMAX V0.2D, V0.2D, V28.2D   clamp lower to -709

    // Range reduction: k = round(z * log2e), r = z - k * ln2
    WORD $0x6E74DC01                  // FMUL V1.2D, V0.2D, V20.2D   V1 = z * log2e
    WORD $0x4E618822                  // FRINTN V2.2D, V1.2D         V2 = k = round(V1)
    WORD $0x6E75DC44                  // FMUL V4.2D, V2.2D, V21.2D   V4 = k * ln2
    WORD $0x4EE4D403                  // FSUB V3.2D, V0.2D, V4.2D    V3 = r = z - k * ln2

    // Polynomial: exp(r) ≈ 1 + r*(1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
    // Horner's method
    WORD $0x6E7ADC64                  // FMUL V4.2D, V3.2D, V26.2D   V4 = r * c5
    WORD $0x4E79D484                  // FADD V4.2D, V4.2D, V25.2D   V4 = c4 + r*c5
    WORD $0x6E63DC84                  // FMUL V4.2D, V4.2D, V3.2D    V4 = r*(c4 + r*c5)
    WORD $0x4E78D484                  // FADD V4.2D, V4.2D, V24.2D   V4 = c3 + r*(...)
    WORD $0x6E63DC84                  // FMUL V4.2D, V4.2D, V3.2D    V4 = r*(c3 + ...)
    WORD $0x4E77D484                  // FADD V4.2D, V4.2D, V23.2D   V4 = c2 + r*(...)
    WORD $0x6E63DC84                  // FMUL V4.2D, V4.2D, V3.2D    V4 = r*(c2 + ...)
    WORD $0x4E76D484                  // FADD V4.2D, V4.2D, V22.2D   V4 = 1 + r*(...)
    WORD $0x6E63DC84                  // FMUL V4.2D, V4.2D, V3.2D    V4 = r*(1 + ...)
    WORD $0x4E76D484                  // FADD V4.2D, V4.2D, V22.2D   V4 = exp(r)

    // Reconstruct: exp(-x) = exp(r) * 2^k
    // For float64: shift by 52 bits (mantissa size), add 0x3FF0... (1.0's bits)
    WORD $0x4EE1B841                  // FCVTZS V1.2D, V2.2D         V1 = int(k)
    WORD $0x4F745421                  // SHL V1.2D, V1.2D, #52       V1 = k << 52
    WORD $0x4EF68421                  // ADD V1.2D, V1.2D, V22.2D    V1 = 2^k (add 1.0's bits)
    WORD $0x6E61DC84                  // FMUL V4.2D, V4.2D, V1.2D    V4 = e^(-x)

    // sigmoid(x) = 1 / (1 + e^(-x))
    WORD $0x4E64D6C6                  // FADD V6.2D, V22.2D, V4.2D   V6 = 1 + e^(-x)
    WORD $0x6E66FEC0                  // FDIV V0.2D, V22.2D, V6.2D   V0 = 1 / (1 + e^(-x))

    VST1.P [V0.D2], 16(R0)            // store result

    SUB $1, R4
    CBNZ R4, sigmoid64_neon_loop2

sigmoid64_neon_scalar:
    AND $1, R3
    CBZ R3, sigmoid64_neon_done

sigmoid64_neon_scalar_loop:
    // Scalar path for remaining element
    FMOVD (R1), F0                    // F0 = x
    FNEGD F0, F0                      // F0 = -x = z

    // Clamp to [-709, 709] using the hoisted bounds (F27=709, F28=-709)
    FMIND F27, F0, F0
    FMAXD F28, F0, F0

    // Range reduction using the hoisted constants (F20=log2e, F21=ln2)
    FMULD F20, F0, F1                 // F1 = z * log2e
    FRINTND F1, F2                    // F2 = k = round(F1)
    FMULD F21, F2, F3                 // F3 = k * ln2
    FSUBD F3, F0, F0                  // F0 = r = z - k * ln2

    // Horner's method using the hoisted coefficients
    // (F22=1.0, F23=c2, F24=c3, F25=c4, F26=c5)
    FMULD F0, F26, F4                 // F4 = r * c5
    FADDD F25, F4, F4                 // F4 = c4 + r*c5
    FMULD F0, F4, F4
    FADDD F24, F4, F4                 // F4 = c3 + r*(...)
    FMULD F0, F4, F4
    FADDD F23, F4, F4                 // F4 = c2 + r*(...)
    FMULD F0, F4, F4
    FADDD F22, F4, F4                 // F4 = 1 + r*(...)
    FMULD F0, F4, F4
    FADDD F22, F4, F4                 // F4 = exp(r)

    // Reconstruct 2^k (float64: shift by 52, add 0x3FF0000000000000)
    FCVTZSD F2, R10
    LSL $52, R10, R10
    MOVD $0x3FF0000000000000, R11
    ADD R11, R10, R10
    FMOVD R10, F5
    FMULD F5, F4, F4                  // F4 = e^(-x)

    // sigmoid(x) = 1 / (1 + e^(-x))
    FADDD F4, F22, F6                 // F6 = 1 + e^(-x)
    FDIVD F6, F22, F0                 // F0 = 1.0 / (1 + e^(-x))

    FMOVD F0, (R0)                    // store result

    ADD $8, R0
    ADD $8, R1
    SUB $1, R3
    CBNZ R3, sigmoid64_neon_scalar_loop

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
// Computes accurate tanh: tanh(x) = (1 - e^(-2x)) / (1 + e^(-2x))
// Uses range reduction and polynomial approximation for exp.
TEXT ·tanhNEON64(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1

    // Load constants into vector registers
    // 2.0 = 0x4000000000000000
    MOVD $0x4000000000000000, R10
    VMOV R10, V19.D[0]
    VDUP V19.D[0], V19.D2             // V19 = 2.0

    // log2(e) = 1.4426950408889634 = 0x3FF71547652B82FE
    MOVD $0x3FF71547652B82FE, R10
    VMOV R10, V20.D[0]
    VDUP V20.D[0], V20.D2             // V20 = log2(e)

    // ln(2) = 0.6931471805599453 = 0x3FE62E42FEFA39EF
    MOVD $0x3FE62E42FEFA39EF, R10
    VMOV R10, V21.D[0]
    VDUP V21.D[0], V21.D2             // V21 = ln(2)

    // 1.0 = 0x3FF0000000000000
    FMOVD $1.0, F22
    VDUP V22.D[0], V22.D2             // V22 = 1.0

    // Polynomial coefficients: c2=0.5, c3=1/6, c4=1/24, c5=1/120
    FMOVD $0.5, F23
    VDUP V23.D[0], V23.D2             // V23 = c2 = 0.5

    // c3 = 1/6 = 0x3FC5555555555555
    MOVD $0x3FC5555555555555, R10
    VMOV R10, V24.D[0]
    VDUP V24.D[0], V24.D2             // V24 = c3 = 1/6

    // c4 = 1/24 = 0x3FA5555555555555
    MOVD $0x3FA5555555555555, R10
    VMOV R10, V25.D[0]
    VDUP V25.D[0], V25.D2             // V25 = c4 = 1/24

    // c5 = 1/120 = 0x3F81111111111111
    MOVD $0x3F81111111111111, R10
    VMOV R10, V26.D[0]
    VDUP V26.D[0], V26.D2             // V26 = c5 = 1/120

    // Clamp thresholds for -2x: ±20.0 (tanh saturates at ~±10)
    // 20.0 = 0x4034000000000000
    MOVD $0x4034000000000000, R10
    VMOV R10, V27.D[0]
    VDUP V27.D[0], V27.D2             // V27 = 20.0 (clamp_hi)

    // -20.0 = 0xC034000000000000
    MOVD $0xC034000000000000, R10
    VMOV R10, V28.D[0]
    VDUP V28.D[0], V28.D2             // V28 = -20.0 (clamp_lo)

    // Process 2 elements per iteration (float64 uses D2 arrangement)
    LSR $1, R3, R4
    CBZ R4, tanh64_neon_scalar

tanh64_neon_loop2:
    VLD1.P 16(R1), [V0.D2]            // V0 = x

    // Compute -2x: negate then multiply by 2
    WORD $0x6EE0F800                  // FNEG V0.2D, V0.2D           V0 = -x
    WORD $0x6E73DC00                  // FMUL V0.2D, V0.2D, V19.2D   V0 = -2x

    // Clamp -2x to [-20, 20]
    WORD $0x4EFBF400                  // FMIN V0.2D, V0.2D, V27.2D   clamp upper to 20
    WORD $0x4E7CF400                  // FMAX V0.2D, V0.2D, V28.2D   clamp lower to -20

    // Range reduction: k = round(-2x * log2e), r = -2x - k * ln2
    WORD $0x6E74DC01                  // FMUL V1.2D, V0.2D, V20.2D   V1 = -2x * log2e
    WORD $0x4E618822                  // FRINTN V2.2D, V1.2D         V2 = k = round(V1)
    WORD $0x6E75DC44                  // FMUL V4.2D, V2.2D, V21.2D   V4 = k * ln2
    WORD $0x4EE4D403                  // FSUB V3.2D, V0.2D, V4.2D    V3 = r = -2x - k * ln2

    // Polynomial: exp(r) ≈ 1 + r*(1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
    // Horner's method
    WORD $0x6E7ADC64                  // FMUL V4.2D, V3.2D, V26.2D   V4 = r * c5
    WORD $0x4E79D484                  // FADD V4.2D, V4.2D, V25.2D   V4 = c4 + r*c5
    WORD $0x6E63DC84                  // FMUL V4.2D, V4.2D, V3.2D    V4 = r*(c4 + r*c5)
    WORD $0x4E78D484                  // FADD V4.2D, V4.2D, V24.2D   V4 = c3 + r*(...)
    WORD $0x6E63DC84                  // FMUL V4.2D, V4.2D, V3.2D    V4 = r*(c3 + ...)
    WORD $0x4E77D484                  // FADD V4.2D, V4.2D, V23.2D   V4 = c2 + r*(...)
    WORD $0x6E63DC84                  // FMUL V4.2D, V4.2D, V3.2D    V4 = r*(c2 + ...)
    WORD $0x4E76D484                  // FADD V4.2D, V4.2D, V22.2D   V4 = 1 + r*(...)
    WORD $0x6E63DC84                  // FMUL V4.2D, V4.2D, V3.2D    V4 = r*(1 + ...)
    WORD $0x4E76D484                  // FADD V4.2D, V4.2D, V22.2D   V4 = exp(r)

    // Reconstruct: exp(-2x) = exp(r) * 2^k
    // For float64: shift by 52 bits (mantissa size), add 0x3FF0... (1.0's bits)
    WORD $0x4EE1B841                  // FCVTZS V1.2D, V2.2D         V1 = int(k)
    WORD $0x4F745421                  // SHL V1.2D, V1.2D, #52       V1 = k << 52
    WORD $0x4EF68421                  // ADD V1.2D, V1.2D, V22.2D    V1 = 2^k (add 1.0's bits)
    WORD $0x6E61DC84                  // FMUL V4.2D, V4.2D, V1.2D    V4 = e^(-2x)

    // tanh(x) = (1 - e^(-2x)) / (1 + e^(-2x))
    WORD $0x4EE4D6C5                  // FSUB V5.2D, V22.2D, V4.2D   V5 = 1 - e^(-2x)
    WORD $0x4E64D6C6                  // FADD V6.2D, V22.2D, V4.2D   V6 = 1 + e^(-2x)
    WORD $0x6E66FCA0                  // FDIV V0.2D, V5.2D, V6.2D    V0 = tanh(x)

    VST1.P [V0.D2], 16(R0)            // store result

    SUB $1, R4
    CBNZ R4, tanh64_neon_loop2

tanh64_neon_scalar:
    AND $1, R3
    CBZ R3, tanh64_neon_done

tanh64_neon_scalar_loop:
    // Scalar path for remaining element
    FMOVD (R1), F0                    // F0 = x
    FNEGD F0, F0                      // F0 = -x

    // Multiply by 2: -2x
    FMOVD $2.0, F7
    FMULD F7, F0, F0                  // F0 = -2x

    // Clamp
    FMOVD $20.0, F1
    FMOVD $-20.0, F2
    FMIND F1, F0, F0
    FMAXD F2, F0, F0

    // Range reduction
    MOVD $0x3FF71547652B82FE, R10     // log2(e)
    FMOVD R10, F8
    FMULD F8, F0, F1                  // F1 = -2x * log2e
    FRINTND F1, F2                    // F2 = k = round(F1)

    MOVD $0x3FE62E42FEFA39EF, R10     // ln(2)
    FMOVD R10, F9
    FMULD F9, F2, F3                  // F3 = k * ln2
    FSUBD F3, F0, F0                  // F0 = r = -2x - k * ln2

    // Polynomial coefficients
    FMOVD $1.0, F10                   // c1 = 1.0
    FMOVD $0.5, F11                   // c2 = 0.5
    MOVD $0x3FC5555555555555, R10     // c3 = 1/6
    FMOVD R10, F12
    MOVD $0x3FA5555555555555, R10     // c4 = 1/24
    FMOVD R10, F13
    MOVD $0x3F81111111111111, R10     // c5 = 1/120
    FMOVD R10, F14

    // Horner's method
    FMULD F0, F14, F4                 // F4 = r * c5
    FADDD F13, F4, F4                 // F4 = c4 + r*c5
    FMULD F0, F4, F4
    FADDD F12, F4, F4                 // F4 = c3 + r*(...)
    FMULD F0, F4, F4
    FADDD F11, F4, F4                 // F4 = c2 + r*(...)
    FMULD F0, F4, F4
    FADDD F10, F4, F4                 // F4 = 1 + r*(...)
    FMULD F0, F4, F4
    FADDD F10, F4, F4                 // F4 = exp(r)

    // Reconstruct 2^k (float64: shift by 52, add 0x3FF0000000000000)
    FCVTZSD F2, R10
    LSL $52, R10, R10
    MOVD $0x3FF0000000000000, R11
    ADD R11, R10, R10
    FMOVD R10, F5
    FMULD F5, F4, F4                  // F4 = e^(-2x)

    // tanh(x) = (1 - e^(-2x)) / (1 + e^(-2x))
    FSUBD F4, F10, F5                 // F5 = 1 - e^(-2x)
    FADDD F4, F10, F6                 // F6 = 1 + e^(-2x)
    FDIVD F6, F5, F0                  // F0 = tanh(x)

    FMOVD F0, (R0)                    // store result

    ADD $8, R0
    ADD $8, R1
    SUB $1, R3
    CBNZ R3, tanh64_neon_scalar_loop

tanh64_neon_done:
    RET

// func expNEON64(dst, src []float64)
// Computes e^x using range reduction and a degree-5 polynomial, the same exp
// core as tanhNEON64 but without the -2x scaling and the tanh wrap. Inputs are
// clamped to [-709, 709] to match the pure-Go fallback: results stay finite
// and large-negative inputs underflow to 0. Processes 2 elements per iteration.
TEXT ·expNEON64(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1

    // log2(e) = 1.4426950408889634 = 0x3FF71547652B82FE
    MOVD $0x3FF71547652B82FE, R10
    VMOV R10, V20.D[0]
    VDUP V20.D[0], V20.D2             // V20 = log2(e)

    // ln(2) = 0.6931471805599453 = 0x3FE62E42FEFA39EF
    MOVD $0x3FE62E42FEFA39EF, R10
    VMOV R10, V21.D[0]
    VDUP V21.D[0], V21.D2             // V21 = ln(2)

    // 1.0
    FMOVD $1.0, F22
    VDUP V22.D[0], V22.D2             // V22 = 1.0

    // c2 = 0.5
    FMOVD $0.5, F23
    VDUP V23.D[0], V23.D2             // V23 = c2 = 0.5

    // c3 = 1/6 = 0x3FC5555555555555
    MOVD $0x3FC5555555555555, R10
    VMOV R10, V24.D[0]
    VDUP V24.D[0], V24.D2             // V24 = c3 = 1/6

    // c4 = 1/24 = 0x3FA5555555555555
    MOVD $0x3FA5555555555555, R10
    VMOV R10, V25.D[0]
    VDUP V25.D[0], V25.D2             // V25 = c4 = 1/24

    // c5 = 1/120 = 0x3F81111111111111
    MOVD $0x3F81111111111111, R10
    VMOV R10, V26.D[0]
    VDUP V26.D[0], V26.D2             // V26 = c5 = 1/120

    // Clamp thresholds: ±709.0 = 0x4086280000000000 / 0xC086280000000000
    MOVD $0x4086280000000000, R10
    VMOV R10, V27.D[0]
    VDUP V27.D[0], V27.D2             // V27 = 709.0 (clamp_hi)

    MOVD $0xC086280000000000, R10
    VMOV R10, V28.D[0]
    VDUP V28.D[0], V28.D2             // V28 = -709.0 (clamp_lo)

    // Process 2 elements per iteration
    LSR $1, R3, R4
    CBZ R4, exp64_neon_scalar

exp64_neon_loop2:
    VLD1.P 16(R1), [V0.D2]            // V0 = x

    // Clamp x to [-709, 709]
    WORD $0x4EFBF400                  // FMIN V0.2D, V0.2D, V27.2D
    WORD $0x4E7CF400                  // FMAX V0.2D, V0.2D, V28.2D

    // Range reduction: k = round(x * log2e), r = x - k * ln2
    WORD $0x6E74DC01                  // FMUL V1.2D, V0.2D, V20.2D   V1 = x * log2e
    WORD $0x4E618822                  // FRINTN V2.2D, V1.2D         V2 = k = round(V1)
    WORD $0x6E75DC44                  // FMUL V4.2D, V2.2D, V21.2D   V4 = k * ln2
    WORD $0x4EE4D403                  // FSUB V3.2D, V0.2D, V4.2D    V3 = r = x - k * ln2

    // Polynomial: exp(r) ~= 1 + r*(1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
    WORD $0x6E7ADC64                  // FMUL V4.2D, V3.2D, V26.2D   V4 = r * c5
    WORD $0x4E79D484                  // FADD V4.2D, V4.2D, V25.2D   V4 = c4 + r*c5
    WORD $0x6E63DC84                  // FMUL V4.2D, V4.2D, V3.2D    V4 = r*(c4 + r*c5)
    WORD $0x4E78D484                  // FADD V4.2D, V4.2D, V24.2D   V4 = c3 + r*(...)
    WORD $0x6E63DC84                  // FMUL V4.2D, V4.2D, V3.2D    V4 = r*(c3 + ...)
    WORD $0x4E77D484                  // FADD V4.2D, V4.2D, V23.2D   V4 = c2 + r*(...)
    WORD $0x6E63DC84                  // FMUL V4.2D, V4.2D, V3.2D    V4 = r*(c2 + ...)
    WORD $0x4E76D484                  // FADD V4.2D, V4.2D, V22.2D   V4 = 1 + r*(...)
    WORD $0x6E63DC84                  // FMUL V4.2D, V4.2D, V3.2D    V4 = r*(1 + ...)
    WORD $0x4E76D484                  // FADD V4.2D, V4.2D, V22.2D   V4 = exp(r)

    // Reconstruct: exp(x) = exp(r) * 2^k (float64: shift by 52)
    WORD $0x4EE1B841                  // FCVTZS V1.2D, V2.2D         V1 = int(k)
    WORD $0x4F745421                  // SHL V1.2D, V1.2D, #52       V1 = k << 52
    WORD $0x4EF68421                  // ADD V1.2D, V1.2D, V22.2D    V1 = 2^k (add 1.0's bits)
    WORD $0x6E61DC84                  // FMUL V4.2D, V4.2D, V1.2D    V4 = exp(x)

    VST1.P [V4.D2], 16(R0)            // store result

    SUB $1, R4
    CBNZ R4, exp64_neon_loop2

exp64_neon_scalar:
    AND $1, R3
    CBZ R3, exp64_neon_done

exp64_neon_scalar_loop:
    FMOVD (R1), F0                    // F0 = x

    // Clamp to [-709, 709]
    MOVD $0x4086280000000000, R10
    FMOVD R10, F1
    MOVD $0xC086280000000000, R10
    FMOVD R10, F2
    FMIND F1, F0, F0
    FMAXD F2, F0, F0

    // Range reduction
    MOVD $0x3FF71547652B82FE, R10     // log2(e)
    FMOVD R10, F8
    FMULD F8, F0, F1                  // F1 = x * log2e
    FRINTND F1, F2                    // F2 = k = round(F1)

    MOVD $0x3FE62E42FEFA39EF, R10     // ln(2)
    FMOVD R10, F9
    FMULD F9, F2, F3                  // F3 = k * ln2
    FSUBD F3, F0, F0                  // F0 = r = x - k * ln2

    // Polynomial coefficients
    FMOVD $1.0, F10                   // c1 = 1.0
    FMOVD $0.5, F11                   // c2 = 0.5
    MOVD $0x3FC5555555555555, R10     // c3 = 1/6
    FMOVD R10, F12
    MOVD $0x3FA5555555555555, R10     // c4 = 1/24
    FMOVD R10, F13
    MOVD $0x3F81111111111111, R10     // c5 = 1/120
    FMOVD R10, F14

    // Horner's method
    FMULD F0, F14, F4                 // F4 = r * c5
    FADDD F13, F4, F4
    FMULD F0, F4, F4
    FADDD F12, F4, F4
    FMULD F0, F4, F4
    FADDD F11, F4, F4
    FMULD F0, F4, F4
    FADDD F10, F4, F4
    FMULD F0, F4, F4
    FADDD F10, F4, F4                 // F4 = exp(r)

    // Reconstruct 2^k (float64: shift by 52)
    FCVTZSD F2, R10
    LSL $52, R10, R10
    MOVD $0x3FF0000000000000, R11
    ADD R11, R10, R10
    FMOVD R10, F5
    FMULD F5, F4, F4                  // F4 = exp(x)
    FMOVD F4, (R0)                    // store result

    ADD $8, R0
    ADD $8, R1
    SUB $1, R3
    CBNZ R3, exp64_neon_scalar_loop

exp64_neon_done:
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

// ============================================================================
// roundNEON: round-half-away-from-zero using FRINTA
// ============================================================================

// func roundNEON(dst, src []float64)
TEXT ·roundNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD src_base+24(FP), R1

    LSR $1, R2, R3
    CBZ R3, round_scalar

round_loop2:
    VLD1.P 16(R1), [V0.D2]
    WORD $0x6E618800           // FRINTA V0.2D, V0.2D
    VST1.P [V0.D2], 16(R0)
    SUB $1, R3
    CBNZ R3, round_loop2

round_scalar:
    AND $1, R2
    CBZ R2, round_done
    FMOVD (R1), F0
    WORD $0x1E664000           // FRINTA D0, D0
    FMOVD F0, (R0)

round_done:
    RET

// func convolveDecimateNEON(dst, signal, kernel []float64, factor, phase int)
//
// Fused decimating valid convolution. For each output it computes the dot
// product of signal[pos:pos+kLen] with kernel, then advances pos by factor.
// The inner dot replicates dotProductNEON exactly (dual accumulators, 4/2/scalar
// reduction) so results are bit-identical to a per-window DotProductUnsafe when
// the kernel takes the NEON path (the Go dispatcher only calls this for
// len(kernel) >= 2). Outer state lives in R9-R15; the inner dot uses R0-R4 and
// V0-V5.
TEXT ·convolveDecimateNEON(SB), NOSPLIT, $0-88
    MOVD dst_base+0(FP), R9
    MOVD dst_len+8(FP), R10
    MOVD signal_base+24(FP), R11
    MOVD kernel_base+48(FP), R12
    MOVD kernel_len+56(FP), R13
    MOVD factor+72(FP), R14
    MOVD phase+80(FP), R15

    CBZ R10, cd_neon_done

cd_neon_outer:
    ADD R15<<3, R11, R0           // R0 = &signal[pos]
    MOVD R12, R1
    MOVD R13, R2                  // R2 = kLen

    VEOR V0.B16, V0.B16, V0.B16
    VEOR V1.B16, V1.B16, V1.B16

    LSR $2, R2, R3                // kLen / 4
    CBZ R3, cd_neon_rem2

cd_neon_loop4:
    VLD1.P 16(R0), [V2.D2]
    VLD1.P 16(R0), [V3.D2]
    VLD1.P 16(R1), [V4.D2]
    VLD1.P 16(R1), [V5.D2]
    WORD $0x4E64CC40             // FMLA V0.2D, V2.2D, V4.2D
    WORD $0x4E65CC61             // FMLA V1.2D, V3.2D, V5.2D
    SUB $1, R3
    CBNZ R3, cd_neon_loop4

    WORD $0x4E61D400            // FADD V0.2D, V0.2D, V1.2D

cd_neon_rem2:
    AND $3, R2, R3
    LSR $1, R3, R4
    CBZ R4, cd_neon_rem1

    VLD1.P 16(R0), [V2.D2]
    VLD1.P 16(R1), [V4.D2]
    WORD $0x4E64CC40            // FMLA V0.2D, V2.2D, V4.2D

cd_neon_rem1:
    AND $1, R3, R4
    CBZ R4, cd_neon_reduce

    // Reduce vector FIRST before scalar op (scalar ops zero upper V bits).
    WORD $0x7E70D800           // FADDP D0, V0.2D
    FMOVD (R0), F2
    FMOVD (R1), F4
    FMADDD F4, F0, F2, F0        // F0 = F2 * F4 + F0

    B cd_neon_store

cd_neon_reduce:
    WORD $0x7E70D800           // FADDP D0, V0.2D

cd_neon_store:
    FMOVD F0, (R9)
    ADD $8, R9
    ADD R14, R15
    SUB $1, R10
    CBNZ R10, cd_neon_outer

cd_neon_done:
    RET

// func autocorrStep2NEON(acc, broadcast, window *float64, count int)
// Steady-region accumulation for two autocorrelation lags at once. V0 holds the
// two seeded accumulators (lanes = lags base..base+1). Each iteration broadcasts
// x[i] (LD1R), loads the two ascending window samples, swaps them with EXT #8 so
// lane j carries x[i-(base+j)], then fuses the multiply-add with FMLA.
//
// FMLA (one rounding) is deliberate: Go's arm64 compiler fuses the scalar
// reference's `sum += x[i]*x[i-lag]` into FMADDD, so the fused kernel is what
// reproduces autocorrelateGo bit-for-bit on arm64. (The amd64 backend does not
// fuse, so the AVX2 kernel uses separate VMULPD+VADDPD to match its own
// fallback.) broadcast (X1) and window (X2) advance one float64 per iteration.
// Registers are fixed by the hand-encoded WORDs: X0=acc, X1=broadcast,
// X2=window; V0=acc, V1=x[i], V2=window/temp.
TEXT ·autocorrStep2NEON(SB), NOSPLIT, $0-32
    MOVD acc+0(FP), R0
    MOVD broadcast+8(FP), R1
    MOVD window+16(FP), R2
    MOVD count+24(FP), R3

    WORD $0x4C407C00           // LD1 {V0.2D}, [X0]   seeded accumulators (lags base, base+1)
    CBZ R3, autocorr2_done

autocorr2_loop:
    WORD $0x4D40CC21           // LD1R {V1.2D}, [X1]  V1 = x[i] in both lanes
    WORD $0x4C407C42           // LD1 {V2.2D}, [X2]   V2 = [x[i-base-1], x[i-base]]
    WORD $0x6E024042           // EXT V2.16B, V2.16B, V2.16B, #8   swap -> [x[i-base], x[i-base-1]]
    WORD $0x4E61CC40           // FMLA V0.2D, V2.2D, V1.2D   V0 += x[i]*window (fused, matches Go FMADDD)
    ADD $8, R1
    ADD $8, R2
    SUB $1, R3
    CBNZ R3, autocorr2_loop

    WORD $0x4C007C00           // ST1 {V0.2D}, [X0]   store the two lag accumulators

autocorr2_done:
    RET

// func dotProduct4NEON(results, row0, row1, row2, row3, vec *float64, n int)
// Scores four rows against the same vec, reusing each vec load across the group
// instead of reloading the query per row (parity with the f32 kernel, ported
// from .4S to .2D: 2 float64 per vector). Two accumulator banks per row
// (V0-V3 bank a, V4-V7 bank b) hide FMLA latency over a 4-element main loop;
// V16/V17 hold the two query chunks, V18-V21 the four row chunks.
// FMLA  Vd.2D, Vn.2D, Vm.2D: 0x4E60CC00 | (Vm << 16) | (Vn << 5) | Vd
// FADD  Vd.2D, Vn.2D, Vm.2D: 0x4E60D400 | (Vm << 16) | (Vn << 5) | Vd
// FADDP Dd, Vn.2D:           0x7E70D800 | (Vn << 5) | Vd
TEXT ·dotProduct4NEON(SB), NOSPLIT, $0-56
    MOVD results+0(FP), R0
    MOVD row0+8(FP), R1
    MOVD row1+16(FP), R2
    MOVD row2+24(FP), R3
    MOVD row3+32(FP), R4
    MOVD vec+40(FP), R5
    MOVD n+48(FP), R6

    VEOR V0.B16, V0.B16, V0.B16
    VEOR V1.B16, V1.B16, V1.B16
    VEOR V2.B16, V2.B16, V2.B16
    VEOR V3.B16, V3.B16, V3.B16
    VEOR V4.B16, V4.B16, V4.B16
    VEOR V5.B16, V5.B16, V5.B16
    VEOR V6.B16, V6.B16, V6.B16
    VEOR V7.B16, V7.B16, V7.B16

    // Process 4 elements per row (2 chunks of 2 doubles) per iteration.
    LSR $2, R6, R7             // R7 = n / 4
    CBZ R7, dot4d_rem2_check

dot4d_loop4:
    // chunk a: group the vec + 4 row loads, then the FMLAs into bank a (V0-V3),
    // so the load/store unit pipelines the loads instead of stalling each FMLA on
    // its just-loaded operand.
    VLD1.P 16(R5), [V16.D2]
    VLD1.P 16(R1), [V18.D2]
    VLD1.P 16(R2), [V19.D2]
    VLD1.P 16(R3), [V20.D2]
    VLD1.P 16(R4), [V21.D2]
    WORD $0x4E70CE40           // FMLA V0.2D, V18.2D, V16.2D
    WORD $0x4E70CE61           // FMLA V1.2D, V19.2D, V16.2D
    WORD $0x4E70CE82           // FMLA V2.2D, V20.2D, V16.2D
    WORD $0x4E70CEA3           // FMLA V3.2D, V21.2D, V16.2D
    // chunk b: same for bank b (V4-V7) with vec[2:4] -> V17
    VLD1.P 16(R5), [V17.D2]
    VLD1.P 16(R1), [V18.D2]
    VLD1.P 16(R2), [V19.D2]
    VLD1.P 16(R3), [V20.D2]
    VLD1.P 16(R4), [V21.D2]
    WORD $0x4E71CE44           // FMLA V4.2D, V18.2D, V17.2D
    WORD $0x4E71CE65           // FMLA V5.2D, V19.2D, V17.2D
    WORD $0x4E71CE86           // FMLA V6.2D, V20.2D, V17.2D
    WORD $0x4E71CEA7           // FMLA V7.2D, V21.2D, V17.2D
    SUB $1, R7
    CBNZ R7, dot4d_loop4

    // Fold bank b into bank a (only reached when the main loop ran).
    WORD $0x4E64D400           // FADD V0.2D, V0.2D, V4.2D
    WORD $0x4E65D421           // FADD V1.2D, V1.2D, V5.2D
    WORD $0x4E66D442           // FADD V2.2D, V2.2D, V6.2D
    WORD $0x4E67D463           // FADD V3.2D, V3.2D, V7.2D

dot4d_rem2_check:
    AND $3, R6, R8
    LSR $1, R8, R9             // R9 = (n & 3) / 2
    CBZ R9, dot4d_reduce

    // One 2-element chunk into bank a: group the loads, then the FMLAs.
    VLD1.P 16(R5), [V16.D2]
    VLD1.P 16(R1), [V18.D2]
    VLD1.P 16(R2), [V19.D2]
    VLD1.P 16(R3), [V20.D2]
    VLD1.P 16(R4), [V21.D2]
    WORD $0x4E70CE40           // FMLA V0.2D, V18.2D, V16.2D
    WORD $0x4E70CE61           // FMLA V1.2D, V19.2D, V16.2D
    WORD $0x4E70CE82           // FMLA V2.2D, V20.2D, V16.2D
    WORD $0x4E70CEA3           // FMLA V3.2D, V21.2D, V16.2D

dot4d_reduce:
    // Reduce each bank-a accumulator to a scalar (D0..D3) BEFORE any scalar FMA,
    // since scalar ops zero the upper lane of the V register.
    WORD $0x7E70D800           // FADDP D0, V0.2D
    WORD $0x7E70D821           // FADDP D1, V1.2D
    WORD $0x7E70D842           // FADDP D2, V2.2D
    WORD $0x7E70D863           // FADDP D3, V3.2D

    AND $1, R6, R8             // R8 = n & 1 (scalar tail count, 0 or 1)
    CBZ R8, dot4d_store

    FMOVD (R5), F18
    FMOVD (R1), F19
    FMADDD F18, F0, F19, F0    // F0 = F19 * F18 + F0
    FMOVD (R2), F19
    FMADDD F18, F1, F19, F1
    FMOVD (R3), F19
    FMADDD F18, F2, F19, F2
    FMOVD (R4), F19
    FMADDD F18, F3, F19, F3

dot4d_store:
    FMOVD F0, (R0)
    FMOVD F1, 8(R0)
    FMOVD F2, 16(R0)
    FMOVD F3, 24(R0)
    RET

// ============================================================================
// logNEON64 / powNEON64 / powElemNEON64: vectorized natural log core (#109)
// ============================================================================

// Exp-core constants for the pow kernels, loaded per iteration with one VLD1
// pair into working registers because the persistent V12-V31 budget is
// exhausted by the log-core constants: [ln(2), 0.5, 1/6, 1/24, 1/120], one
// lane pair each.
DATA log64neon_expc<>+0x00(SB)/8, $0x3FE62E42FEFA39EF
DATA log64neon_expc<>+0x08(SB)/8, $0x3FE62E42FEFA39EF
DATA log64neon_expc<>+0x10(SB)/8, $0x3FE0000000000000
DATA log64neon_expc<>+0x18(SB)/8, $0x3FE0000000000000
DATA log64neon_expc<>+0x20(SB)/8, $0x3FC5555555555555
DATA log64neon_expc<>+0x28(SB)/8, $0x3FC5555555555555
DATA log64neon_expc<>+0x30(SB)/8, $0x3FA5555555555555
DATA log64neon_expc<>+0x38(SB)/8, $0x3FA5555555555555
DATA log64neon_expc<>+0x40(SB)/8, $0x3F81111111111111
DATA log64neon_expc<>+0x48(SB)/8, $0x3F81111111111111
GLOBL log64neon_expc<>(SB), RODATA|NOPTR, $80

// func logNEON64(dst, src []float64, k1hi, k1lo, k2 float64)
// Shared kernel for Log, Log2, and Log10: per lane it computes
// result = e*k1hi + (lnm*k2 + e*k1lo), with x = m*2^e, m in
// [sqrt(2)/2, sqrt(2)) and lnm = ln(m) = 2s + s*t*P(t) for s = (m-1)/(m+1),
// t = s^2 (atanh form, SLEEF xlog_u1 minimax polynomial; same algorithm as
// the amd64 logAVX). Positive subnormal inputs are pre-scaled by 2^64
// (exponent bias -64). Special lanes are fixed up with BIT blends from the
// original input: +Inf -> +Inf, +-0 -> -Inf, x < 0 or NaN -> NaN, matching
// math.Log. Processes 2 elements per iteration; the 0-1 element tail uses
// the scalar path below.
TEXT ·logNEON64(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1

    // Special-value constants for the blend fixups
    MOVD $0x7FF0000000000000, R10
    VMOV R10, V13.D[0]
    VDUP V13.D[0], V13.D2             // V13 = +Inf
    MOVD $0xFFF0000000000000, R10
    VMOV R10, V14.D[0]
    VDUP V14.D[0], V14.D2             // V14 = -Inf
    MOVD $0x7FF8000000000000, R10
    VMOV R10, V15.D[0]
    VDUP V15.D[0], V15.D2             // V15 = quiet NaN

    // Reduction constants (see logAVX for the algorithm notes)
    MOVD $0x3FE6A09E00000000, R10
    VMOV R10, V16.D[0]
    VDUP V16.D[0], V16.D2             // V16 = reduction offset
    MOVD $0xFFF0000000000000, R10
    VMOV R10, V17.D[0]
    VDUP V17.D[0], V17.D2             // V17 = exponent mask
    MOVD $0x0010000000000000, R10
    VMOV R10, V18.D[0]
    VDUP V18.D[0], V18.D2             // V18 = DBL_MIN
    MOVD $0x43F0000000000000, R10
    VMOV R10, V19.D[0]
    VDUP V19.D[0], V19.D2             // V19 = 2^64 (subnormal pre-scale)
    MOVD $0xC050000000000000, R10
    VMOV R10, V20.D[0]
    VDUP V20.D[0], V20.D2             // V20 = -64.0 (exponent bias)
    FMOVD $1.0, F21
    VDUP V21.D[0], V21.D2             // V21 = 1.0

    // Reconstruction constants from the arguments
    FMOVD k1hi+48(FP), F22
    VDUP V22.D[0], V22.D2             // V22 = k1hi
    FMOVD k1lo+56(FP), F23
    VDUP V23.D[0], V23.D2             // V23 = k1lo
    FMOVD k2+64(FP), F24
    VDUP V24.D[0], V24.D2             // V24 = k2

    // SLEEF xlog_u1 minimax coefficients c0..c6
    MOVD $0x3FC39C4F5407567E, R10
    VMOV R10, V25.D[0]
    VDUP V25.D[0], V25.D2             // V25 = c0 = 0.15320769885027014
    MOVD $0x3FC3872E67FE8E84, R10
    VMOV R10, V26.D[0]
    VDUP V26.D[0], V26.D2             // V26 = c1 = 0.15256290510034287
    MOVD $0x3FC747353A506035, R10
    VMOV R10, V27.D[0]
    VDUP V27.D[0], V27.D2             // V27 = c2 = 0.1818605932937786
    MOVD $0x3FCC71C0A65ECD8E, R10
    VMOV R10, V28.D[0]
    VDUP V28.D[0], V28.D2             // V28 = c3 = 0.222221451983938
    MOVD $0x3FD249249A68A245, R10
    VMOV R10, V29.D[0]
    VDUP V29.D[0], V29.D2             // V29 = c4 = 0.28571429327942993
    MOVD $0x3FD99999998F92EA, R10
    VMOV R10, V30.D[0]
    VDUP V30.D[0], V30.D2             // V30 = c5 = 0.3999999999635252
    MOVD $0x3FE55555555557AE, R10
    VMOV R10, V31.D[0]
    VDUP V31.D[0], V31.D2             // V31 = c6 = 0.66666666666673335

    LSR $1, R3, R4
    CBZ R4, log64_neon_scalar

log64_neon_loop2:
    VLD1.P 16(R1), [V0.D2]            // V0 = x (kept for the special blends)

    // Subnormal pre-scale: lanes with x < DBL_MIN scaled by 2^64, bias -64
    WORD $0x6ee0e641                  // FCMGT V1.2D, V18.2D, V0.2D   mask: x < DBL_MIN
    WORD $0x6e73dc02                  // FMUL V2.2D, V0.2D, V19.2D    x * 2^64
    WORD $0x4ea11c23                  // MOV V3.16B, V1.16B           copy mask for BSL
    WORD $0x6e601c43                  // BSL V3.16B, V2.16B, V0.16B   xs = mask ? x*2^64 : x
    WORD $0x4e341c24                  // AND V4.16B, V1.16B, V20.16B  ebias = mask & -64.0

    // tmp = bits(xs) - OFF; bits(m) = bits(xs) - (tmp & expmask);
    // e = (tmp >> 52) + ebias, leaving m in [sqrt(2)/2, sqrt(2))
    WORD $0x6ef08465                  // SUB V5.2D, V3.2D, V16.2D     tmp
    WORD $0x4e311ca6                  // AND V6.16B, V5.16B, V17.16B  tmp & expmask
    WORD $0x6ee68466                  // SUB V6.2D, V3.2D, V6.2D      bits(m)
    WORD $0x4f4c04a5                  // SSHR V5.2D, V5.2D, #52       e (int64, arithmetic)
    WORD $0x4e61d8a5                  // SCVTF V5.2D, V5.2D           e as float64
    WORD $0x4e64d4a5                  // FADD V5.2D, V5.2D, V4.2D     e += ebias

    // s = (m-1)/(m+1), t = s^2
    WORD $0x4ef5d4c7                  // FSUB V7.2D, V6.2D, V21.2D    m - 1
    WORD $0x4e75d4c6                  // FADD V6.2D, V6.2D, V21.2D    m + 1
    WORD $0x6e66fce7                  // FDIV V7.2D, V7.2D, V6.2D     s
    WORD $0x6e67dce6                  // FMUL V6.2D, V7.2D, V7.2D     t

    // P(t), Horner ping-pong between V8/V9 with FMLA
    WORD $0x4eb91f28                  // MOV V8.16B, V25.16B          acc = c0
    WORD $0x4eba1f49                  // MOV V9.16B, V26.16B
    WORD $0x4e66cd09                  // FMLA V9.2D, V8.2D, V6.2D     c1 + acc*t
    WORD $0x4ebb1f68                  // MOV V8.16B, V27.16B
    WORD $0x4e66cd28                  // FMLA V8.2D, V9.2D, V6.2D     c2 + acc*t
    WORD $0x4ebc1f89                  // MOV V9.16B, V28.16B
    WORD $0x4e66cd09                  // FMLA V9.2D, V8.2D, V6.2D     c3 + acc*t
    WORD $0x4ebd1fa8                  // MOV V8.16B, V29.16B
    WORD $0x4e66cd28                  // FMLA V8.2D, V9.2D, V6.2D     c4 + acc*t
    WORD $0x4ebe1fc9                  // MOV V9.16B, V30.16B
    WORD $0x4e66cd09                  // FMLA V9.2D, V8.2D, V6.2D     c5 + acc*t
    WORD $0x4ebf1fe8                  // MOV V8.16B, V31.16B
    WORD $0x4e66cd28                  // FMLA V8.2D, V9.2D, V6.2D     P(t) = c6 + acc*t

    // lnm = 2s + s*t*P(t)
    WORD $0x6e66dcea                  // FMUL V10.2D, V7.2D, V6.2D    s*t
    WORD $0x4e67d4eb                  // FADD V11.2D, V7.2D, V7.2D    2s
    WORD $0x4e68cd4b                  // FMLA V11.2D, V10.2D, V8.2D   lnm

    // result = e*k1hi + (lnm*k2 + e*k1lo)
    WORD $0x6e77dcaa                  // FMUL V10.2D, V5.2D, V23.2D   e * k1lo
    WORD $0x4e78cd6a                  // FMLA V10.2D, V11.2D, V24.2D  += lnm * k2
    WORD $0x4e76ccaa                  // FMLA V10.2D, V5.2D, V22.2D   += e * k1hi

    // Special lanes from the original x: +Inf -> +Inf, +-0 -> -Inf,
    // x < 0 or NaN -> NaN
    WORD $0x4e6de401                  // FCMEQ V1.2D, V0.2D, V13.2D   mask: x == +Inf
    WORD $0x6ea11daa                  // BIT V10.16B, V13.16B, V1.16B
    WORD $0x4ee0d801                  // FCMEQ V1.2D, V0.2D, #0       mask: x == +-0
    WORD $0x6ea11dca                  // BIT V10.16B, V14.16B, V1.16B
    WORD $0x4ee0e802                  // FCMLT V2.2D, V0.2D, #0       mask: x < 0
    WORD $0x4e60e401                  // FCMEQ V1.2D, V0.2D, V0.2D    mask: x ordered
    WORD $0x4ee11c42                  // ORN V2.16B, V2.16B, V1.16B   (x < 0) | NaN
    WORD $0x6ea21dea                  // BIT V10.16B, V15.16B, V2.16B

    VST1.P [V10.D2], 16(R0)
    SUB $1, R4
    CBNZ R4, log64_neon_loop2

log64_neon_scalar:
    AND $1, R3
    CBZ R3, log64_neon_done

log64_neon_scalar_loop:
    MOVD (R1), R5                     // bits(x)
    FMOVD (R1), F0

    // Specials: FCMPD sets V for unordered, N for negative, Z for zero
    FCMPD $(0.0), F0
    BVS log64_neon_scalar_nan         // x is NaN
    BMI log64_neon_scalar_nan         // x < 0
    BEQ log64_neon_scalar_neginf      // x == +-0
    MOVD $0x7FF0000000000000, R6
    CMP R6, R5
    BEQ log64_neon_scalar_posinf

    // Subnormal pre-scale (x positive finite; bits compare as ints)
    MOVD $0, R9
    MOVD $0x0010000000000000, R6
    CMP R6, R5
    BGE log64_neon_scalar_normal
    MOVD $0x43F0000000000000, R7
    FMOVD R7, F2
    FMULD F2, F0, F0
    FMOVD F0, R5
    MOVD $-64, R9

log64_neon_scalar_normal:
    MOVD $0x3FE6A09E00000000, R6
    SUB R6, R5, R7                    // tmp = bits - OFF
    ASR $52, R7, R8                   // e
    ADD R9, R8, R8                    // e += bias
    MOVD $0xFFF0000000000000, R6
    AND R6, R7, R7
    SUB R7, R5, R5                    // bits(m)
    FMOVD R5, F1                      // m
    SCVTFD R8, F2                     // e as float64

    // s = (m-1)/(m+1), t = s^2 (F21 = 1.0 from the vector constants)
    FSUBD F21, F1, F3                 // m - 1
    FADDD F21, F1, F4                 // m + 1
    FDIVD F4, F3, F3                  // s
    FMULD F3, F3, F4                  // t

    // P(t): FMADDD Fm, Fa, Fn, Fd computes Fd = Fa + Fn*Fm
    FMADDD F4, F26, F25, F5           // c1 + c0*t
    FMADDD F4, F27, F5, F5            // c2 + acc*t
    FMADDD F4, F28, F5, F5            // c3 + acc*t
    FMADDD F4, F29, F5, F5            // c4 + acc*t
    FMADDD F4, F30, F5, F5            // c5 + acc*t
    FMADDD F4, F31, F5, F5            // P(t) = c6 + acc*t

    FMULD F4, F3, F4                  // s*t
    FADDD F3, F3, F3                  // 2s
    FMADDD F5, F3, F4, F3             // lnm = 2s + s*t*P(t)

    FMULD F23, F2, F4                 // e * k1lo
    FMADDD F24, F4, F3, F4            // += lnm * k2
    FMADDD F22, F4, F2, F4            // += e * k1hi
    FMOVD F4, (R0)
    B log64_neon_scalar_next

log64_neon_scalar_nan:
    MOVD $0x7FF8000000000000, R6
    MOVD R6, (R0)
    B log64_neon_scalar_next

log64_neon_scalar_neginf:
    MOVD $0xFFF0000000000000, R6
    MOVD R6, (R0)
    B log64_neon_scalar_next

log64_neon_scalar_posinf:
    MOVD $0x7FF0000000000000, R6
    MOVD R6, (R0)

log64_neon_scalar_next:
    ADD $8, R1
    ADD $8, R0
    SUB $1, R3
    CBNZ R3, log64_neon_scalar_loop

log64_neon_done:
    RET

// func powNEON64(dst, src []float64, exp float64)
// Fused pow(x, p) = exp(p*ln(x)) for slices whose elements are all positive
// and finite (the dispatcher guarantees this, see powSIMDOK64). The log core
// matches logNEON64; the exp core matches expNEON64 with its [-709, 709]
// clamp on y = p*ln(x), but lanes whose pre-clamp y exceeds the clamp are
// blended to +Inf (y > 709) or 0 (y < -709) so true overflow/underflow keeps
// math.Pow's result classes (see powAVX for the band caveat). Accuracy is
// bounded by the exp core's degree-5 polynomial (~3e-6 relative).
TEXT ·powNEON64(SB), NOSPLIT, $0-56
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1
    MOVD $log64neon_expc<>(SB), R5    // exp-core constant table
    ADD $64, R5, R6                   // second VLD1 base (c5 = 1/120)

    FMOVD exp+48(FP), F12
    VDUP V12.D[0], V12.D2             // V12 = p

    // Exp clamp bounds and +Inf for the overflow blend
    MOVD $0x4086280000000000, R10
    VMOV R10, V13.D[0]
    VDUP V13.D[0], V13.D2             // V13 = 709.0
    MOVD $0xC086280000000000, R10
    VMOV R10, V14.D[0]
    VDUP V14.D[0], V14.D2             // V14 = -709.0
    MOVD $0x7FF0000000000000, R10
    VMOV R10, V15.D[0]
    VDUP V15.D[0], V15.D2             // V15 = +Inf

    // Log-core reduction constants (see logNEON64)
    MOVD $0x3FE6A09E00000000, R10
    VMOV R10, V16.D[0]
    VDUP V16.D[0], V16.D2             // V16 = reduction offset
    MOVD $0xFFF0000000000000, R10
    VMOV R10, V17.D[0]
    VDUP V17.D[0], V17.D2             // V17 = exponent mask
    MOVD $0x0010000000000000, R10
    VMOV R10, V18.D[0]
    VDUP V18.D[0], V18.D2             // V18 = DBL_MIN
    MOVD $0x43F0000000000000, R10
    VMOV R10, V19.D[0]
    VDUP V19.D[0], V19.D2             // V19 = 2^64
    MOVD $0xC050000000000000, R10
    VMOV R10, V20.D[0]
    VDUP V20.D[0], V20.D2             // V20 = -64.0
    FMOVD $1.0, F21
    VDUP V21.D[0], V21.D2             // V21 = 1.0

    // fdlibm ln(2) hi/lo split and log2(e)
    MOVD $0x3FE62E42FEE00000, R10
    VMOV R10, V22.D[0]
    VDUP V22.D[0], V22.D2             // V22 = ln2 hi
    MOVD $0x3DEA39EF35793C76, R10
    VMOV R10, V23.D[0]
    VDUP V23.D[0], V23.D2             // V23 = ln2 lo
    MOVD $0x3FF71547652B82FE, R10
    VMOV R10, V24.D[0]
    VDUP V24.D[0], V24.D2             // V24 = log2(e)

    // SLEEF xlog_u1 minimax coefficients c0..c6
    MOVD $0x3FC39C4F5407567E, R10
    VMOV R10, V25.D[0]
    VDUP V25.D[0], V25.D2             // V25 = c0
    MOVD $0x3FC3872E67FE8E84, R10
    VMOV R10, V26.D[0]
    VDUP V26.D[0], V26.D2             // V26 = c1
    MOVD $0x3FC747353A506035, R10
    VMOV R10, V27.D[0]
    VDUP V27.D[0], V27.D2             // V27 = c2
    MOVD $0x3FCC71C0A65ECD8E, R10
    VMOV R10, V28.D[0]
    VDUP V28.D[0], V28.D2             // V28 = c3
    MOVD $0x3FD249249A68A245, R10
    VMOV R10, V29.D[0]
    VDUP V29.D[0], V29.D2             // V29 = c4
    MOVD $0x3FD99999998F92EA, R10
    VMOV R10, V30.D[0]
    VDUP V30.D[0], V30.D2             // V30 = c5
    MOVD $0x3FE55555555557AE, R10
    VMOV R10, V31.D[0]
    VDUP V31.D[0], V31.D2             // V31 = c6

    LSR $1, R3, R4
    CBZ R4, pow64_neon_scalar

pow64_neon_loop2:
    VLD1.P 16(R1), [V0.D2]            // V0 = x (positive finite)

    // --- log core (see logNEON64) ---
    WORD $0x6ee0e641                  // FCMGT V1.2D, V18.2D, V0.2D   mask: x < DBL_MIN
    WORD $0x6e73dc02                  // FMUL V2.2D, V0.2D, V19.2D    x * 2^64
    WORD $0x4ea11c23                  // MOV V3.16B, V1.16B
    WORD $0x6e601c43                  // BSL V3.16B, V2.16B, V0.16B   xs
    WORD $0x4e341c24                  // AND V4.16B, V1.16B, V20.16B  ebias
    WORD $0x6ef08465                  // SUB V5.2D, V3.2D, V16.2D     tmp
    WORD $0x4e311ca6                  // AND V6.16B, V5.16B, V17.16B
    WORD $0x6ee68466                  // SUB V6.2D, V3.2D, V6.2D      bits(m)
    WORD $0x4f4c04a5                  // SSHR V5.2D, V5.2D, #52       e (int64)
    WORD $0x4e61d8a5                  // SCVTF V5.2D, V5.2D           e as float64
    WORD $0x4e64d4a5                  // FADD V5.2D, V5.2D, V4.2D     e += ebias
    WORD $0x4ef5d4c7                  // FSUB V7.2D, V6.2D, V21.2D    m - 1
    WORD $0x4e75d4c6                  // FADD V6.2D, V6.2D, V21.2D    m + 1
    WORD $0x6e66fce7                  // FDIV V7.2D, V7.2D, V6.2D     s
    WORD $0x6e67dce6                  // FMUL V6.2D, V7.2D, V7.2D     t
    WORD $0x4eb91f28                  // MOV V8.16B, V25.16B          acc = c0
    WORD $0x4eba1f49                  // MOV V9.16B, V26.16B
    WORD $0x4e66cd09                  // FMLA V9.2D, V8.2D, V6.2D     c1 + acc*t
    WORD $0x4ebb1f68                  // MOV V8.16B, V27.16B
    WORD $0x4e66cd28                  // FMLA V8.2D, V9.2D, V6.2D     c2 + acc*t
    WORD $0x4ebc1f89                  // MOV V9.16B, V28.16B
    WORD $0x4e66cd09                  // FMLA V9.2D, V8.2D, V6.2D     c3 + acc*t
    WORD $0x4ebd1fa8                  // MOV V8.16B, V29.16B
    WORD $0x4e66cd28                  // FMLA V8.2D, V9.2D, V6.2D     c4 + acc*t
    WORD $0x4ebe1fc9                  // MOV V9.16B, V30.16B
    WORD $0x4e66cd09                  // FMLA V9.2D, V8.2D, V6.2D     c5 + acc*t
    WORD $0x4ebf1fe8                  // MOV V8.16B, V31.16B
    WORD $0x4e66cd28                  // FMLA V8.2D, V9.2D, V6.2D     P(t)
    WORD $0x6e66dcea                  // FMUL V10.2D, V7.2D, V6.2D    s*t
    WORD $0x4e67d4eb                  // FADD V11.2D, V7.2D, V7.2D    2s
    WORD $0x4e68cd4b                  // FMLA V11.2D, V10.2D, V8.2D   lnm

    // ln(x) = e*ln2hi + (e*ln2lo + lnm)
    WORD $0x6e77dcaa                  // FMUL V10.2D, V5.2D, V23.2D   e * ln2lo
    WORD $0x4e6bd54a                  // FADD V10.2D, V10.2D, V11.2D  + lnm
    WORD $0x4e76ccaa                  // FMLA V10.2D, V5.2D, V22.2D   += e * ln2hi

    // y = p*ln(x); keep the pre-clamp y in V6 for the overflow blends, then
    // clamp to [-709, 709] for the exp core
    WORD $0x6e6cdd40                  // FMUL V0.2D, V10.2D, V12.2D   y
    WORD $0x4ea01c06                  // MOV V6.16B, V0.16B           pre-clamp y
    WORD $0x4eedf400                  // FMIN V0.2D, V0.2D, V13.2D
    WORD $0x4e6ef400                  // FMAX V0.2D, V0.2D, V14.2D

    // --- exp core (see expNEON64); constants from the table ---
    VLD1 (R5), [V1.D2, V2.D2, V3.D2, V4.D2] // ln2, 0.5, 1/6, 1/24
    VLD1 (R6), [V5.D2]                      // 1/120
    WORD $0x6e78dc07                  // FMUL V7.2D, V0.2D, V24.2D    y * log2e
    WORD $0x4e6188e7                  // FRINTN V7.2D, V7.2D          k
    WORD $0x6e61dce8                  // FMUL V8.2D, V7.2D, V1.2D     k * ln2
    WORD $0x4ee8d408                  // FSUB V8.2D, V0.2D, V8.2D     r
    WORD $0x6e65dd09                  // FMUL V9.2D, V8.2D, V5.2D     r * c5
    WORD $0x4e64d529                  // FADD V9.2D, V9.2D, V4.2D     + 1/24
    WORD $0x6e68dd29                  // FMUL V9.2D, V9.2D, V8.2D
    WORD $0x4e63d529                  // FADD V9.2D, V9.2D, V3.2D     + 1/6
    WORD $0x6e68dd29                  // FMUL V9.2D, V9.2D, V8.2D
    WORD $0x4e62d529                  // FADD V9.2D, V9.2D, V2.2D     + 0.5
    WORD $0x6e68dd29                  // FMUL V9.2D, V9.2D, V8.2D
    WORD $0x4e75d529                  // FADD V9.2D, V9.2D, V21.2D    + 1
    WORD $0x6e68dd29                  // FMUL V9.2D, V9.2D, V8.2D
    WORD $0x4e75d529                  // FADD V9.2D, V9.2D, V21.2D    exp(r)
    WORD $0x4ee1b8e7                  // FCVTZS V7.2D, V7.2D          int(k)
    WORD $0x4f7454e7                  // SHL V7.2D, V7.2D, #52
    WORD $0x4ef584e7                  // ADD V7.2D, V7.2D, V21.2D     2^k bits
    WORD $0x6e67dd29                  // FMUL V9.2D, V9.2D, V7.2D     exp(y)

    // Overflow/underflow classes from the pre-clamp y: y > 709 -> +Inf,
    // y < -709 -> 0 (strict compares, see powAVX)
    WORD $0x6eede4c1                  // FCMGT V1.2D, V6.2D, V13.2D   y > 709
    WORD $0x6ea11de9                  // BIT V9.16B, V15.16B, V1.16B
    WORD $0x6ee6e5c1                  // FCMGT V1.2D, V14.2D, V6.2D   y < -709
    WORD $0x4e611d29                  // BIC V9.16B, V9.16B, V1.16B

    VST1.P [V9.D2], 16(R0)
    SUB $1, R4
    CBNZ R4, pow64_neon_loop2

pow64_neon_scalar:
    AND $1, R3
    CBZ R3, pow64_neon_done

pow64_neon_scalar_loop:
    MOVD (R1), R7                     // bits(x)
    FMOVD (R1), F0

    // log core, scalar (x positive finite; see logNEON64 tail)
    MOVD $0, R9
    MOVD $0x0010000000000000, R8
    CMP R8, R7
    BGE pow64_neon_scalar_normal
    MOVD $0x43F0000000000000, R10
    FMOVD R10, F2
    FMULD F2, F0, F0
    FMOVD F0, R7
    MOVD $-64, R9

pow64_neon_scalar_normal:
    MOVD $0x3FE6A09E00000000, R8
    SUB R8, R7, R10                   // tmp
    ASR $52, R10, R11                 // e
    ADD R9, R11, R11
    MOVD $0xFFF0000000000000, R8
    AND R8, R10, R10
    SUB R10, R7, R7                   // bits(m)
    FMOVD R7, F1                      // m
    SCVTFD R11, F2                    // e

    FSUBD F21, F1, F3                 // m - 1
    FADDD F21, F1, F4                 // m + 1
    FDIVD F4, F3, F3                  // s
    FMULD F3, F3, F4                  // t
    FMADDD F4, F26, F25, F5           // c1 + c0*t
    FMADDD F4, F27, F5, F5
    FMADDD F4, F28, F5, F5
    FMADDD F4, F29, F5, F5
    FMADDD F4, F30, F5, F5
    FMADDD F4, F31, F5, F5            // P(t)
    FMULD F4, F3, F4                  // s*t
    FADDD F3, F3, F3                  // 2s
    FMADDD F5, F3, F4, F3             // lnm
    FMULD F23, F2, F4                 // e * ln2lo
    FADDD F3, F4, F4                  // + lnm
    FMADDD F22, F4, F2, F4            // += e * ln2hi -> ln(x)

    // y = p*ln(x); pre-clamp copy in F6, clamp to [-709, 709]
    FMULD F12, F4, F0
    FMOVD F0, F6
    FMIND F13, F0, F0
    FMAXD F14, F0, F0

    // exp core, scalar (see expNEON64 tail; F24 = log2e, F21 = 1.0)
    FMULD F24, F0, F1
    FRINTND F1, F2                    // k
    MOVD $0x3FE62E42FEFA39EF, R8      // ln(2)
    FMOVD R8, F3
    FMULD F3, F2, F3
    FSUBD F3, F0, F0                  // r
    MOVD $0x3F81111111111111, R8      // 1/120
    FMOVD R8, F4
    FMULD F0, F4, F4
    MOVD $0x3FA5555555555555, R8      // 1/24
    FMOVD R8, F5
    FADDD F5, F4, F4
    FMULD F0, F4, F4
    MOVD $0x3FC5555555555555, R8      // 1/6
    FMOVD R8, F5
    FADDD F5, F4, F4
    FMULD F0, F4, F4
    FMOVD $0.5, F5
    FADDD F5, F4, F4
    FMULD F0, F4, F4
    FADDD F21, F4, F4
    FMULD F0, F4, F4
    FADDD F21, F4, F4                 // exp(r)
    FCVTZSD F2, R8
    LSL $52, R8, R8
    MOVD $0x3FF0000000000000, R10
    ADD R10, R8, R8
    FMOVD R8, F5
    FMULD F5, F4, F4                  // exp(y)

    // Overflow/underflow classes from the pre-clamp y (F6)
    FCMPD F13, F6
    BGT pow64_neon_scalar_posinf      // y > 709
    FCMPD F14, F6
    BLT pow64_neon_scalar_zero        // y < -709
    FMOVD F4, (R0)
    B pow64_neon_scalar_next

pow64_neon_scalar_posinf:
    MOVD $0x7FF0000000000000, R8
    MOVD R8, (R0)
    B pow64_neon_scalar_next

pow64_neon_scalar_zero:
    MOVD ZR, (R0)

pow64_neon_scalar_next:
    ADD $8, R1
    ADD $8, R0
    SUB $1, R3
    CBNZ R3, pow64_neon_scalar_loop

pow64_neon_done:
    RET

// func powElemNEON64(dst, base, exp []float64)
// Elementwise pow(base[i], exp[i]) = exp(exp[i]*ln(base[i])). Same cores and
// preconditions as powNEON64 (all bases positive finite, all exponents
// finite), with the exponent loaded per lane instead of broadcast.
TEXT ·powElemNEON64(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD base_base+24(FP), R1
    MOVD exp_base+48(FP), R2
    MOVD $log64neon_expc<>(SB), R5    // exp-core constant table
    ADD $64, R5, R6                   // second VLD1 base (c5 = 1/120)

    // Exp clamp bounds and +Inf for the overflow blend
    MOVD $0x4086280000000000, R10
    VMOV R10, V13.D[0]
    VDUP V13.D[0], V13.D2             // V13 = 709.0
    MOVD $0xC086280000000000, R10
    VMOV R10, V14.D[0]
    VDUP V14.D[0], V14.D2             // V14 = -709.0
    MOVD $0x7FF0000000000000, R10
    VMOV R10, V15.D[0]
    VDUP V15.D[0], V15.D2             // V15 = +Inf

    // Log-core reduction constants (see logNEON64)
    MOVD $0x3FE6A09E00000000, R10
    VMOV R10, V16.D[0]
    VDUP V16.D[0], V16.D2             // V16 = reduction offset
    MOVD $0xFFF0000000000000, R10
    VMOV R10, V17.D[0]
    VDUP V17.D[0], V17.D2             // V17 = exponent mask
    MOVD $0x0010000000000000, R10
    VMOV R10, V18.D[0]
    VDUP V18.D[0], V18.D2             // V18 = DBL_MIN
    MOVD $0x43F0000000000000, R10
    VMOV R10, V19.D[0]
    VDUP V19.D[0], V19.D2             // V19 = 2^64
    MOVD $0xC050000000000000, R10
    VMOV R10, V20.D[0]
    VDUP V20.D[0], V20.D2             // V20 = -64.0
    FMOVD $1.0, F21
    VDUP V21.D[0], V21.D2             // V21 = 1.0

    // fdlibm ln(2) hi/lo split and log2(e)
    MOVD $0x3FE62E42FEE00000, R10
    VMOV R10, V22.D[0]
    VDUP V22.D[0], V22.D2             // V22 = ln2 hi
    MOVD $0x3DEA39EF35793C76, R10
    VMOV R10, V23.D[0]
    VDUP V23.D[0], V23.D2             // V23 = ln2 lo
    MOVD $0x3FF71547652B82FE, R10
    VMOV R10, V24.D[0]
    VDUP V24.D[0], V24.D2             // V24 = log2(e)

    // SLEEF xlog_u1 minimax coefficients c0..c6
    MOVD $0x3FC39C4F5407567E, R10
    VMOV R10, V25.D[0]
    VDUP V25.D[0], V25.D2             // V25 = c0
    MOVD $0x3FC3872E67FE8E84, R10
    VMOV R10, V26.D[0]
    VDUP V26.D[0], V26.D2             // V26 = c1
    MOVD $0x3FC747353A506035, R10
    VMOV R10, V27.D[0]
    VDUP V27.D[0], V27.D2             // V27 = c2
    MOVD $0x3FCC71C0A65ECD8E, R10
    VMOV R10, V28.D[0]
    VDUP V28.D[0], V28.D2             // V28 = c3
    MOVD $0x3FD249249A68A245, R10
    VMOV R10, V29.D[0]
    VDUP V29.D[0], V29.D2             // V29 = c4
    MOVD $0x3FD99999998F92EA, R10
    VMOV R10, V30.D[0]
    VDUP V30.D[0], V30.D2             // V30 = c5
    MOVD $0x3FE55555555557AE, R10
    VMOV R10, V31.D[0]
    VDUP V31.D[0], V31.D2             // V31 = c6

    LSR $1, R3, R4
    CBZ R4, powelem64_neon_scalar

powelem64_neon_loop2:
    VLD1.P 16(R1), [V0.D2]            // V0 = base (positive finite)

    // --- log core (see logNEON64) ---
    WORD $0x6ee0e641                  // FCMGT V1.2D, V18.2D, V0.2D   mask: x < DBL_MIN
    WORD $0x6e73dc02                  // FMUL V2.2D, V0.2D, V19.2D    x * 2^64
    WORD $0x4ea11c23                  // MOV V3.16B, V1.16B
    WORD $0x6e601c43                  // BSL V3.16B, V2.16B, V0.16B   xs
    WORD $0x4e341c24                  // AND V4.16B, V1.16B, V20.16B  ebias
    WORD $0x6ef08465                  // SUB V5.2D, V3.2D, V16.2D     tmp
    WORD $0x4e311ca6                  // AND V6.16B, V5.16B, V17.16B
    WORD $0x6ee68466                  // SUB V6.2D, V3.2D, V6.2D      bits(m)
    WORD $0x4f4c04a5                  // SSHR V5.2D, V5.2D, #52       e (int64)
    WORD $0x4e61d8a5                  // SCVTF V5.2D, V5.2D           e as float64
    WORD $0x4e64d4a5                  // FADD V5.2D, V5.2D, V4.2D     e += ebias
    WORD $0x4ef5d4c7                  // FSUB V7.2D, V6.2D, V21.2D    m - 1
    WORD $0x4e75d4c6                  // FADD V6.2D, V6.2D, V21.2D    m + 1
    WORD $0x6e66fce7                  // FDIV V7.2D, V7.2D, V6.2D     s
    WORD $0x6e67dce6                  // FMUL V6.2D, V7.2D, V7.2D     t
    WORD $0x4eb91f28                  // MOV V8.16B, V25.16B          acc = c0
    WORD $0x4eba1f49                  // MOV V9.16B, V26.16B
    WORD $0x4e66cd09                  // FMLA V9.2D, V8.2D, V6.2D     c1 + acc*t
    WORD $0x4ebb1f68                  // MOV V8.16B, V27.16B
    WORD $0x4e66cd28                  // FMLA V8.2D, V9.2D, V6.2D     c2 + acc*t
    WORD $0x4ebc1f89                  // MOV V9.16B, V28.16B
    WORD $0x4e66cd09                  // FMLA V9.2D, V8.2D, V6.2D     c3 + acc*t
    WORD $0x4ebd1fa8                  // MOV V8.16B, V29.16B
    WORD $0x4e66cd28                  // FMLA V8.2D, V9.2D, V6.2D     c4 + acc*t
    WORD $0x4ebe1fc9                  // MOV V9.16B, V30.16B
    WORD $0x4e66cd09                  // FMLA V9.2D, V8.2D, V6.2D     c5 + acc*t
    WORD $0x4ebf1fe8                  // MOV V8.16B, V31.16B
    WORD $0x4e66cd28                  // FMLA V8.2D, V9.2D, V6.2D     P(t)
    WORD $0x6e66dcea                  // FMUL V10.2D, V7.2D, V6.2D    s*t
    WORD $0x4e67d4eb                  // FADD V11.2D, V7.2D, V7.2D    2s
    WORD $0x4e68cd4b                  // FMLA V11.2D, V10.2D, V8.2D   lnm

    // ln(base) = e*ln2hi + (e*ln2lo + lnm)
    WORD $0x6e77dcaa                  // FMUL V10.2D, V5.2D, V23.2D   e * ln2lo
    WORD $0x4e6bd54a                  // FADD V10.2D, V10.2D, V11.2D  + lnm
    WORD $0x4e76ccaa                  // FMLA V10.2D, V5.2D, V22.2D   += e * ln2hi

    // y = exp[i]*ln(base[i]); pre-clamp copy in V6, clamp for the exp core
    VLD1.P 16(R2), [V12.D2]           // V12 = exponents (finite)
    WORD $0x6e6cdd40                  // FMUL V0.2D, V10.2D, V12.2D   y
    WORD $0x4ea01c06                  // MOV V6.16B, V0.16B           pre-clamp y
    WORD $0x4eedf400                  // FMIN V0.2D, V0.2D, V13.2D
    WORD $0x4e6ef400                  // FMAX V0.2D, V0.2D, V14.2D

    // --- exp core (see expNEON64); constants from the table ---
    VLD1 (R5), [V1.D2, V2.D2, V3.D2, V4.D2] // ln2, 0.5, 1/6, 1/24
    VLD1 (R6), [V5.D2]                      // 1/120
    WORD $0x6e78dc07                  // FMUL V7.2D, V0.2D, V24.2D    y * log2e
    WORD $0x4e6188e7                  // FRINTN V7.2D, V7.2D          k
    WORD $0x6e61dce8                  // FMUL V8.2D, V7.2D, V1.2D     k * ln2
    WORD $0x4ee8d408                  // FSUB V8.2D, V0.2D, V8.2D     r
    WORD $0x6e65dd09                  // FMUL V9.2D, V8.2D, V5.2D     r * c5
    WORD $0x4e64d529                  // FADD V9.2D, V9.2D, V4.2D     + 1/24
    WORD $0x6e68dd29                  // FMUL V9.2D, V9.2D, V8.2D
    WORD $0x4e63d529                  // FADD V9.2D, V9.2D, V3.2D     + 1/6
    WORD $0x6e68dd29                  // FMUL V9.2D, V9.2D, V8.2D
    WORD $0x4e62d529                  // FADD V9.2D, V9.2D, V2.2D     + 0.5
    WORD $0x6e68dd29                  // FMUL V9.2D, V9.2D, V8.2D
    WORD $0x4e75d529                  // FADD V9.2D, V9.2D, V21.2D    + 1
    WORD $0x6e68dd29                  // FMUL V9.2D, V9.2D, V8.2D
    WORD $0x4e75d529                  // FADD V9.2D, V9.2D, V21.2D    exp(r)
    WORD $0x4ee1b8e7                  // FCVTZS V7.2D, V7.2D          int(k)
    WORD $0x4f7454e7                  // SHL V7.2D, V7.2D, #52
    WORD $0x4ef584e7                  // ADD V7.2D, V7.2D, V21.2D     2^k bits
    WORD $0x6e67dd29                  // FMUL V9.2D, V9.2D, V7.2D     exp(y)

    // Overflow/underflow classes from the pre-clamp y (see powNEON64)
    WORD $0x6eede4c1                  // FCMGT V1.2D, V6.2D, V13.2D   y > 709
    WORD $0x6ea11de9                  // BIT V9.16B, V15.16B, V1.16B
    WORD $0x6ee6e5c1                  // FCMGT V1.2D, V14.2D, V6.2D   y < -709
    WORD $0x4e611d29                  // BIC V9.16B, V9.16B, V1.16B

    VST1.P [V9.D2], 16(R0)
    SUB $1, R4
    CBNZ R4, powelem64_neon_loop2

powelem64_neon_scalar:
    AND $1, R3
    CBZ R3, powelem64_neon_done

powelem64_neon_scalar_loop:
    MOVD (R1), R7                     // bits(base)
    FMOVD (R1), F0

    // log core, scalar (base positive finite; see logNEON64 tail)
    MOVD $0, R9
    MOVD $0x0010000000000000, R8
    CMP R8, R7
    BGE powelem64_neon_scalar_normal
    MOVD $0x43F0000000000000, R10
    FMOVD R10, F2
    FMULD F2, F0, F0
    FMOVD F0, R7
    MOVD $-64, R9

powelem64_neon_scalar_normal:
    MOVD $0x3FE6A09E00000000, R8
    SUB R8, R7, R10                   // tmp
    ASR $52, R10, R11                 // e
    ADD R9, R11, R11
    MOVD $0xFFF0000000000000, R8
    AND R8, R10, R10
    SUB R10, R7, R7                   // bits(m)
    FMOVD R7, F1                      // m
    SCVTFD R11, F2                    // e

    FSUBD F21, F1, F3                 // m - 1
    FADDD F21, F1, F4                 // m + 1
    FDIVD F4, F3, F3                  // s
    FMULD F3, F3, F4                  // t
    FMADDD F4, F26, F25, F5           // c1 + c0*t
    FMADDD F4, F27, F5, F5
    FMADDD F4, F28, F5, F5
    FMADDD F4, F29, F5, F5
    FMADDD F4, F30, F5, F5
    FMADDD F4, F31, F5, F5            // P(t)
    FMULD F4, F3, F4                  // s*t
    FADDD F3, F3, F3                  // 2s
    FMADDD F5, F3, F4, F3             // lnm
    FMULD F23, F2, F4                 // e * ln2lo
    FADDD F3, F4, F4                  // + lnm
    FMADDD F22, F4, F2, F4            // += e * ln2hi -> ln(base)

    // y = exp[i]*ln(base[i]); pre-clamp copy in F6, clamp to [-709, 709]
    FMOVD (R2), F12                   // p
    FMULD F12, F4, F0
    FMOVD F0, F6
    FMIND F13, F0, F0
    FMAXD F14, F0, F0

    // exp core, scalar (see expNEON64 tail; F24 = log2e, F21 = 1.0)
    FMULD F24, F0, F1
    FRINTND F1, F2                    // k
    MOVD $0x3FE62E42FEFA39EF, R8      // ln(2)
    FMOVD R8, F3
    FMULD F3, F2, F3
    FSUBD F3, F0, F0                  // r
    MOVD $0x3F81111111111111, R8      // 1/120
    FMOVD R8, F4
    FMULD F0, F4, F4
    MOVD $0x3FA5555555555555, R8      // 1/24
    FMOVD R8, F5
    FADDD F5, F4, F4
    FMULD F0, F4, F4
    MOVD $0x3FC5555555555555, R8      // 1/6
    FMOVD R8, F5
    FADDD F5, F4, F4
    FMULD F0, F4, F4
    FMOVD $0.5, F5
    FADDD F5, F4, F4
    FMULD F0, F4, F4
    FADDD F21, F4, F4
    FMULD F0, F4, F4
    FADDD F21, F4, F4                 // exp(r)
    FCVTZSD F2, R8
    LSL $52, R8, R8
    MOVD $0x3FF0000000000000, R10
    ADD R10, R8, R8
    FMOVD R8, F5
    FMULD F5, F4, F4                  // exp(y)

    // Overflow/underflow classes from the pre-clamp y (F6)
    FCMPD F13, F6
    BGT powelem64_neon_scalar_posinf  // y > 709
    FCMPD F14, F6
    BLT powelem64_neon_scalar_zero    // y < -709
    FMOVD F4, (R0)
    B powelem64_neon_scalar_next

powelem64_neon_scalar_posinf:
    MOVD $0x7FF0000000000000, R8
    MOVD R8, (R0)
    B powelem64_neon_scalar_next

powelem64_neon_scalar_zero:
    MOVD ZR, (R0)

powelem64_neon_scalar_next:
    ADD $8, R1
    ADD $8, R2
    ADD $8, R0
    SUB $1, R3
    CBNZ R3, powelem64_neon_scalar_loop

powelem64_neon_done:
    RET
