//go:build arm64

#include "textflag.h"

// ARM64 NEON for float32: 4 elements per 128-bit register
// All vector instructions use WORD opcodes since Go's ARM64 assembler
// doesn't support NEON mnemonics directly.

// Opcode formulas for float32 (4S arrangement):
// FADD Vd.4S, Vn.4S, Vm.4S: 0x4E20D400 | (Vm << 16) | (Vn << 5) | Vd
// FSUB Vd.4S, Vn.4S, Vm.4S: 0x4EA0D400 | (Vm << 16) | (Vn << 5) | Vd
// FMUL Vd.4S, Vn.4S, Vm.4S: 0x6E20DC00 | (Vm << 16) | (Vn << 5) | Vd
// FDIV Vd.4S, Vn.4S, Vm.4S: 0x6E20FC00 | (Vm << 16) | (Vn << 5) | Vd
// FMIN Vd.4S, Vn.4S, Vm.4S: 0x4EA0F400 | (Vm << 16) | (Vn << 5) | Vd
// FMAX Vd.4S, Vn.4S, Vm.4S: 0x4E20F400 | (Vm << 16) | (Vn << 5) | Vd
// FABS Vd.4S, Vn.4S:        0x4EA0F800 | (Vn << 5) | Vd
// FNEG Vd.4S, Vn.4S:        0x6EA0F800 | (Vn << 5) | Vd
// FMLA Vd.4S, Vn.4S, Vm.4S: 0x4E20CC00 | (Vm << 16) | (Vn << 5) | Vd
// FADDP Vd.4S, Vn.4S, Vm.4S: 0x6E20D400 | (Vm << 16) | (Vn << 5) | Vd
// FADDP Sd, Vn.2S:          0x7E30D800 | (Vn << 5) | Vd

// func dotProductNEON(a, b []float32) float32
// Handles mismatched slice lengths: uses min(len(a), len(b)).
TEXT ·dotProductNEON(SB), NOSPLIT, $0-52
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R2
    MOVD b_len+32(FP), R3
    CMP R3, R2
    CSEL LT, R2, R3, R2        // R2 = min(len(a), len(b))
    MOVD b_base+24(FP), R1

    VEOR V0.B16, V0.B16, V0.B16
    VEOR V1.B16, V1.B16, V1.B16

    // Process 8 elements (2 NEON ops) per iteration
    LSR $3, R2, R3
    CBZ R3, dot32_remainder4

dot32_loop8:
    VLD1.P 16(R0), [V2.S4]
    VLD1.P 16(R0), [V3.S4]
    VLD1.P 16(R1), [V4.S4]
    VLD1.P 16(R1), [V5.S4]
    WORD $0x4E24CC40           // FMLA V0.4S, V2.4S, V4.4S
    WORD $0x4E25CC61           // FMLA V1.4S, V3.4S, V5.4S
    SUB $1, R3
    CBNZ R3, dot32_loop8

    // Combine accumulators: V0 = V0 + V1
    WORD $0x4E21D400           // FADD V0.4S, V0.4S, V1.4S

dot32_remainder4:
    AND $7, R2, R3
    LSR $2, R3, R4
    CBZ R4, dot32_remainder

    VLD1.P 16(R0), [V2.S4]
    VLD1.P 16(R1), [V4.S4]
    WORD $0x4E24CC40           // FMLA V0.4S, V2.4S, V4.4S

dot32_remainder:
    AND $3, R3, R4
    CBZ R4, dot32_reduce

    // Must reduce vector FIRST before scalar ops (scalar ops zero upper V bits)
    WORD $0x6E20D400           // FADDP V0.4S, V0.4S, V0.4S
    WORD $0x7E30D800           // FADDP S0, V0.2S

dot32_scalar:
    FMOVS (R0), F2
    FMOVS (R1), F4
    FMADDS F4, F0, F2, F0      // F0 = F2 * F4 + F0 (Go syntax: Fm, Fa, Fn, Fd)
    ADD $4, R0
    ADD $4, R1
    SUB $1, R4
    CBNZ R4, dot32_scalar

    FMOVS F0, ret+48(FP)
    RET

dot32_reduce:
    // Horizontal sum of V0.4S -> S0 when no scalar remainder
    WORD $0x6E20D400           // FADDP V0.4S, V0.4S, V0.4S
    WORD $0x7E30D800           // FADDP S0, V0.2S

    FMOVS F0, ret+48(FP)
    RET

// func addNEON(dst, a, b []float32)
TEXT ·addNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $2, R3, R4
    CBZ R4, add32_scalar

add32_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    WORD $0x4E21D402           // FADD V2.4S, V0.4S, V1.4S
    VST1.P [V2.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, add32_loop4

add32_scalar:
    AND $3, R3
    CBZ R3, add32_done

add32_loop1:
    FMOVS (R1), F0
    FMOVS (R2), F1
    FADDS F0, F1, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    ADD $4, R2
    SUB $1, R3
    CBNZ R3, add32_loop1

add32_done:
    RET

// func subNEON(dst, a, b []float32)
TEXT ·subNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $2, R3, R4
    CBZ R4, sub32_scalar

sub32_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    WORD $0x4EA1D402           // FSUB V2.4S, V0.4S, V1.4S
    VST1.P [V2.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, sub32_loop4

sub32_scalar:
    AND $3, R3
    CBZ R3, sub32_done

sub32_loop1:
    FMOVS (R1), F0
    FMOVS (R2), F1
    FSUBS F1, F0, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    ADD $4, R2
    SUB $1, R3
    CBNZ R3, sub32_loop1

sub32_done:
    RET

// func mulNEON(dst, a, b []float32)
TEXT ·mulNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $2, R3, R4
    CBZ R4, mul32_scalar

mul32_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    WORD $0x6E21DC02           // FMUL V2.4S, V0.4S, V1.4S
    VST1.P [V2.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, mul32_loop4

mul32_scalar:
    AND $3, R3
    CBZ R3, mul32_done

mul32_loop1:
    FMOVS (R1), F0
    FMOVS (R2), F1
    FMULS F0, F1, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    ADD $4, R2
    SUB $1, R3
    CBNZ R3, mul32_loop1

mul32_done:
    RET

// func divNEON(dst, a, b []float32)
TEXT ·divNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $2, R3, R4
    CBZ R4, div32_scalar

div32_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    WORD $0x6E21FC02           // FDIV V2.4S, V0.4S, V1.4S
    VST1.P [V2.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, div32_loop4

div32_scalar:
    AND $3, R3
    CBZ R3, div32_done

div32_loop1:
    FMOVS (R1), F0
    FMOVS (R2), F1
    FDIVS F1, F0, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    ADD $4, R2
    SUB $1, R3
    CBNZ R3, div32_loop1

div32_done:
    RET

// func scaleNEON(dst, a []float32, s float32)
TEXT ·scaleNEON(SB), NOSPLIT, $0-52
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1
    FMOVS s+48(FP), F3
    // DUP V3.4S, V3.S[0] - broadcast scalar to all lanes
    WORD $0x4E040463           // DUP V3.4S, V3.S[0]

    LSR $2, R2, R3
    CBZ R3, scale32_scalar

scale32_loop4:
    VLD1.P 16(R1), [V0.S4]
    WORD $0x6E23DC01           // FMUL V1.4S, V0.4S, V3.4S
    VST1.P [V1.S4], 16(R0)
    SUB $1, R3
    CBNZ R3, scale32_loop4

scale32_scalar:
    AND $3, R2
    CBZ R2, scale32_done

scale32_loop1:
    FMOVS (R1), F0
    FMULS F0, F3, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    SUB $1, R2
    CBNZ R2, scale32_loop1

scale32_done:
    RET

// func addScalarNEON(dst, a []float32, s float32)
TEXT ·addScalarNEON(SB), NOSPLIT, $0-52
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1
    FMOVS s+48(FP), F3
    // DUP V3.4S, V3.S[0] - broadcast scalar to all lanes
    WORD $0x4E040463           // DUP V3.4S, V3.S[0]

    LSR $2, R2, R3
    CBZ R3, addsc32_scalar

addsc32_loop4:
    VLD1.P 16(R1), [V0.S4]
    WORD $0x4E23D401           // FADD V1.4S, V0.4S, V3.4S
    VST1.P [V1.S4], 16(R0)
    SUB $1, R3
    CBNZ R3, addsc32_loop4

addsc32_scalar:
    AND $3, R2
    CBZ R2, addsc32_done

addsc32_loop1:
    FMOVS (R1), F0
    FADDS F0, F3, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    SUB $1, R2
    CBNZ R2, addsc32_loop1

addsc32_done:
    RET

// func sumNEON(a []float32) float32
TEXT ·sumNEON(SB), NOSPLIT, $0-28
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1

    VEOR V0.B16, V0.B16, V0.B16

    LSR $2, R1, R2
    CBZ R2, sum32_reduce_first

sum32_loop4:
    VLD1.P 16(R0), [V1.S4]
    WORD $0x4E21D400           // FADD V0.4S, V0.4S, V1.4S
    SUB $1, R2
    CBNZ R2, sum32_loop4

sum32_reduce_first:
    // Horizontal sum of vector accumulator BEFORE scalar ops
    // (scalar ops zero upper bits of V registers)
    WORD $0x6E20D400           // FADDP V0.4S, V0.4S, V0.4S
    WORD $0x7E30D800           // FADDP S0, V0.2S

    // Now process scalar remainder
    AND $3, R1
    CBZ R1, sum32_done

sum32_loop1:
    FMOVS (R0), F1
    FADDS F0, F1, F0
    ADD $4, R0
    SUB $1, R1
    CBNZ R1, sum32_loop1

sum32_done:
    FMOVS F0, ret+24(FP)
    RET

// func minNEON(a []float32) float32
TEXT ·minNEON(SB), NOSPLIT, $0-28
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1

    VLD1.P 16(R0), [V0.S4]
    SUB $4, R1

    LSR $2, R1, R2
    CBZ R2, min32_scalar

min32_loop4:
    VLD1.P 16(R0), [V1.S4]
    WORD $0x4EA1F400           // FMIN V0.4S, V0.4S, V1.4S
    SUB $1, R2
    CBNZ R2, min32_loop4

min32_scalar:
    AND $3, R1
    CBZ R1, min32_reduce

    // Reduce vector FIRST before scalar ops
    WORD $0x6EB0F400           // FMINP V0.4S, V0.4S, V0.4S
    WORD $0x7EB0F800           // FMINP S0, V0.2S

min32_loop1:
    FMOVS (R0), F1
    FMINS F0, F1, F0
    ADD $4, R0
    SUB $1, R1
    CBNZ R1, min32_loop1

    FMOVS F0, ret+24(FP)
    RET

min32_reduce:
    // Horizontal min when no scalar remainder
    WORD $0x6EB0F400           // FMINP V0.4S, V0.4S, V0.4S
    WORD $0x7EB0F800           // FMINP S0, V0.2S
    FMOVS F0, ret+24(FP)
    RET

// func maxNEON(a []float32) float32
TEXT ·maxNEON(SB), NOSPLIT, $0-28
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1

    VLD1.P 16(R0), [V0.S4]
    SUB $4, R1

    LSR $2, R1, R2
    CBZ R2, max32_scalar

max32_loop4:
    VLD1.P 16(R0), [V1.S4]
    WORD $0x4E21F400           // FMAX V0.4S, V0.4S, V1.4S
    SUB $1, R2
    CBNZ R2, max32_loop4

max32_scalar:
    AND $3, R1
    CBZ R1, max32_reduce

    // Reduce vector FIRST before scalar ops
    WORD $0x6E30F400           // FMAXP V0.4S, V0.4S, V0.4S
    WORD $0x7E30F800           // FMAXP S0, V0.2S

max32_loop1:
    FMOVS (R0), F1
    FMAXS F0, F1, F0
    ADD $4, R0
    SUB $1, R1
    CBNZ R1, max32_loop1

    FMOVS F0, ret+24(FP)
    RET

max32_reduce:
    // Horizontal max when no scalar remainder
    WORD $0x6E30F400           // FMAXP V0.4S, V0.4S, V0.4S
    WORD $0x7E30F800           // FMAXP S0, V0.2S
    FMOVS F0, ret+24(FP)
    RET

// func absNEON(dst, a []float32)
TEXT ·absNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1

    LSR $2, R2, R3
    CBZ R3, abs32_scalar

abs32_loop4:
    VLD1.P 16(R1), [V0.S4]
    WORD $0x4EA0F801           // FABS V1.4S, V0.4S
    VST1.P [V1.S4], 16(R0)
    SUB $1, R3
    CBNZ R3, abs32_loop4

abs32_scalar:
    AND $3, R2
    CBZ R2, abs32_done

abs32_loop1:
    FMOVS (R1), F0
    FABSS F0, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    SUB $1, R2
    CBNZ R2, abs32_loop1

abs32_done:
    RET

// func negNEON(dst, a []float32)
TEXT ·negNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1

    LSR $2, R2, R3
    CBZ R3, neg32_scalar

neg32_loop4:
    VLD1.P 16(R1), [V0.S4]
    WORD $0x6EA0F801           // FNEG V1.4S, V0.4S
    VST1.P [V1.S4], 16(R0)
    SUB $1, R3
    CBNZ R3, neg32_loop4

neg32_scalar:
    AND $3, R2
    CBZ R2, neg32_done

neg32_loop1:
    FMOVS (R1), F0
    FNEGS F0, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    SUB $1, R2
    CBNZ R2, neg32_loop1

neg32_done:
    RET

// func fmaNEON(dst, a, b, c []float32)
TEXT ·fmaNEON(SB), NOSPLIT, $0-96
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R4
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2
    MOVD c_base+72(FP), R3

    LSR $2, R4, R5
    CBZ R5, fma32_scalar

fma32_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    VLD1.P 16(R3), [V2.S4]
    WORD $0x4E21CC02           // FMLA V2.4S, V0.4S, V1.4S
    VST1.P [V2.S4], 16(R0)
    SUB $1, R5
    CBNZ R5, fma32_loop4

fma32_scalar:
    AND $3, R4
    CBZ R4, fma32_done

fma32_loop1:
    FMOVS (R1), F0              // a[i]
    FMOVS (R2), F1              // b[i]
    FMOVS (R3), F2              // c[i]
    FMADDS F1, F2, F0, F2       // F2 = F0 * F1 + F2 = a[i] * b[i] + c[i] (Go syntax: Fm, Fa, Fn, Fd)
    FMOVS F2, (R0)
    ADD $4, R0
    ADD $4, R1
    ADD $4, R2
    ADD $4, R3
    SUB $1, R4
    CBNZ R4, fma32_loop1

fma32_done:
    RET

// func clampNEON(dst, a []float32, minVal, maxVal float32)
TEXT ·clampNEON(SB), NOSPLIT, $0-56
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1
    FMOVS minVal+48(FP), F2
    FMOVS maxVal+52(FP), F3
    // DUP V2.4S, V2.S[0] and DUP V3.4S, V3.S[0]
    WORD $0x4E040442           // DUP V2.4S, V2.S[0]
    WORD $0x4E040463           // DUP V3.4S, V3.S[0]

    LSR $2, R2, R3
    CBZ R3, clamp32_scalar

clamp32_loop4:
    VLD1.P 16(R1), [V0.S4]
    WORD $0x4E22F400           // FMAX V0.4S, V0.4S, V2.4S (clamp to min)
    WORD $0x4EA3F400           // FMIN V0.4S, V0.4S, V3.4S (clamp to max)
    VST1.P [V0.S4], 16(R0)
    SUB $1, R3
    CBNZ R3, clamp32_loop4

clamp32_scalar:
    AND $3, R2
    CBZ R2, clamp32_done

clamp32_loop1:
    FMOVS (R1), F0
    FMAXS F0, F2, F0
    FMINS F0, F3, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    SUB $1, R2
    CBNZ R2, clamp32_loop1

clamp32_done:
    RET

// Interleave/Deinterleave with NEON ZIP/UZP instructions for float32
// ZIP1 Vd.4S, Vn.4S, Vm.4S: 0x4E803800 | (Vm << 16) | (Vn << 5) | Vd
// ZIP2 Vd.4S, Vn.4S, Vm.4S: 0x4E807800 | (Vm << 16) | (Vn << 5) | Vd
// UZP1 Vd.4S, Vn.4S, Vm.4S: 0x4E801800 | (Vm << 16) | (Vn << 5) | Vd
// UZP2 Vd.4S, Vn.4S, Vm.4S: 0x4E805800 | (Vm << 16) | (Vn << 5) | Vd

// func interleave2NEON(dst, a, b []float32)
TEXT ·interleave2NEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0    // dst pointer
    MOVD a_base+24(FP), R1     // a pointer
    MOVD a_len+32(FP), R3      // n = len(a)
    MOVD b_base+48(FP), R2     // b pointer

    // Process 4 pairs at a time
    LSR $2, R3, R4             // R4 = n / 4
    CBZ R4, interleave2_neon32_remainder

interleave2_neon32_loop4:
    VLD1.P 16(R1), [V0.S4]     // V0 = [a0, a1, a2, a3]
    VLD1.P 16(R2), [V1.S4]     // V1 = [b0, b1, b2, b3]
    WORD $0x4E813802           // ZIP1 V2.4S, V0.4S, V1.4S -> [a0, b0, a1, b1]
    WORD $0x4E817803           // ZIP2 V3.4S, V0.4S, V1.4S -> [a2, b2, a3, b3]
    VST1.P [V2.S4], 16(R0)     // Store [a0, b0, a1, b1]
    VST1.P [V3.S4], 16(R0)     // Store [a2, b2, a3, b3]
    SUB $1, R4
    CBNZ R4, interleave2_neon32_loop4

interleave2_neon32_remainder:
    AND $3, R3
    CBZ R3, interleave2_neon32_done

interleave2_neon32_loop1:
    FMOVS (R1), F0
    FMOVS (R2), F1
    FMOVS F0, (R0)
    FMOVS F1, 4(R0)
    ADD $4, R1
    ADD $4, R2
    ADD $8, R0
    SUB $1, R3
    CBNZ R3, interleave2_neon32_loop1

interleave2_neon32_done:
    RET

// func deinterleave2NEON(a, b, src []float32)
TEXT ·deinterleave2NEON(SB), NOSPLIT, $0-72
    MOVD a_base+0(FP), R0      // a pointer
    MOVD a_len+8(FP), R3       // n = len(a)
    MOVD b_base+24(FP), R1     // b pointer
    MOVD src_base+48(FP), R2   // src pointer

    // Process 4 pairs at a time
    LSR $2, R3, R4             // R4 = n / 4
    CBZ R4, deinterleave2_neon32_remainder

deinterleave2_neon32_loop4:
    VLD1.P 16(R2), [V0.S4]     // V0 = [a0, b0, a1, b1]
    VLD1.P 16(R2), [V1.S4]     // V1 = [a2, b2, a3, b3]
    WORD $0x4E811802           // UZP1 V2.4S, V0.4S, V1.4S -> [a0, a1, a2, a3]
    WORD $0x4E815803           // UZP2 V3.4S, V0.4S, V1.4S -> [b0, b1, b2, b3]
    VST1.P [V2.S4], 16(R0)     // Store a
    VST1.P [V3.S4], 16(R1)     // Store b
    SUB $1, R4
    CBNZ R4, deinterleave2_neon32_loop4

deinterleave2_neon32_remainder:
    AND $3, R3
    CBZ R3, deinterleave2_neon32_done

deinterleave2_neon32_loop1:
    FMOVS (R2), F0
    FMOVS 4(R2), F1
    FMOVS F0, (R0)
    FMOVS F1, (R1)
    ADD $8, R2
    ADD $4, R0
    ADD $4, R1
    SUB $1, R3
    CBNZ R3, deinterleave2_neon32_loop1

deinterleave2_neon32_done:
    RET

// Additional NEON operations for f32

// FSQRT Vd.4S, Vn.4S: 0x6EA1F800 | (Vn << 5) | Vd

// func sqrtNEON(dst, a []float32)
TEXT ·sqrtNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1

    LSR $2, R2, R3
    CBZ R3, sqrt32_scalar

sqrt32_loop4:
    VLD1.P 16(R1), [V0.S4]
    WORD $0x6EA1F801           // FSQRT V1.4S, V0.4S
    VST1.P [V1.S4], 16(R0)
    SUB $1, R3
    CBNZ R3, sqrt32_loop4

sqrt32_scalar:
    AND $3, R2
    CBZ R2, sqrt32_done

sqrt32_loop1:
    FMOVS (R1), F0
    FSQRTS F0, F0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    SUB $1, R2
    CBNZ R2, sqrt32_loop1

sqrt32_done:
    RET

// func reciprocalNEON(dst, a []float32)
// Uses full precision division (FRECPE is only approximate)
TEXT ·reciprocalNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD a_base+24(FP), R1

    // Load 1.0 into F3 and broadcast
    FMOVS $1.0, F3
    WORD $0x4E040463           // DUP V3.4S, V3.S[0]

    LSR $2, R2, R3
    CBZ R3, recip32_scalar

recip32_loop4:
    VLD1.P 16(R1), [V0.S4]
    WORD $0x6E20FC61           // FDIV V1.4S, V3.4S, V0.4S
    VST1.P [V1.S4], 16(R0)
    SUB $1, R3
    CBNZ R3, recip32_loop4

recip32_scalar:
    AND $3, R2
    CBZ R2, recip32_done

recip32_loop1:
    FMOVS (R1), F0
    FDIVS F0, F3, F0           // F0 = 1.0 / a[i]
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    SUB $1, R2
    CBNZ R2, recip32_loop1

recip32_done:
    RET

// func addScaledNEON(dst []float32, alpha float32, s []float32)
// dst[i] += alpha * s[i] - AXPY operation
// Frame: dst(24) + alpha(4) + pad(4) + s(24) = 56 bytes
TEXT ·addScaledNEON(SB), NOSPLIT, $0-56
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    FMOVS alpha+24(FP), F3
    MOVD s_base+32(FP), R1

    // Broadcast alpha to all 4 lanes
    WORD $0x4E040463           // DUP V3.4S, V3.S[0]

    LSR $2, R2, R3
    CBZ R3, addscaled32_scalar

addscaled32_loop4:
    VLD1 (R0), [V0.S4]         // Load dst[i:i+4]
    VLD1.P 16(R1), [V1.S4]     // Load s[i:i+4]
    WORD $0x4E23CC20           // FMLA V0.4S, V1.4S, V3.4S  (dst += s * alpha)
    VST1.P [V0.S4], 16(R0)
    SUB $1, R3
    CBNZ R3, addscaled32_loop4

addscaled32_scalar:
    AND $3, R2
    CBZ R2, addscaled32_done

addscaled32_loop1:
    FMOVS (R0), F0
    FMOVS (R1), F1
    FMADDS F3, F0, F1, F0      // dst += s * alpha (Go: Fm, Fa, Fn, Fd -> Fd = Fn*Fm + Fa)
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    SUB $1, R2
    CBNZ R2, addscaled32_loop1

addscaled32_done:
    RET

// func varianceNEON32(a []float32, mean float32) float32
// Frame: a(24) + mean(4) + padding(4) + ret(4) = 36
TEXT ·varianceNEON32(SB), NOSPLIT, $0-36
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1
    FMOVS mean+24(FP), F3
    // DUP V3.4S, V3.S[0] - broadcast mean to all lanes
    WORD $0x4E040463           // DUP V3.4S, V3.S[0]

    VEOR V0.B16, V0.B16, V0.B16     // Accumulator = 0

    LSR $2, R1, R2
    CBZ R2, var32_scalar

var32_loop4:
    VLD1.P 16(R0), [V1.S4]     // Load 4 elements
    WORD $0x4EA3D421           // FSUB V1.4S, V1.4S, V3.4S  (diff = a[i] - mean)
    WORD $0x4E21CC20           // FMLA V0.4S, V1.4S, V1.4S  (acc += diff * diff)
    SUB $1, R2
    CBNZ R2, var32_loop4

var32_scalar:
    AND $3, R1
    CBZ R1, var32_reduce

    // Reduce vector FIRST before scalar ops
    WORD $0x6E20D400           // FADDP V0.4S, V0.4S, V0.4S
    WORD $0x7E30D800           // FADDP S0, V0.2S

var32_loop1:
    FMOVS (R0), F1
    FSUBS F3, F1, F1           // diff = a[i] - mean
    FMADDS F1, F0, F1, F0      // acc += diff * diff
    ADD $4, R0
    SUB $1, R1
    CBNZ R1, var32_loop1
    B var32_div

var32_reduce:
    WORD $0x6E20D400           // FADDP V0.4S, V0.4S, V0.4S
    WORD $0x7E30D800           // FADDP S0, V0.2S

var32_div:
    // Divide by n
    MOVD a_len+8(FP), R1
    SCVTFS R1, F1              // Convert n to float32
    FDIVS F1, F0, F0           // variance = sum / n
    FMOVS F0, ret+32(FP)
    RET

// func euclideanDistanceNEON32(a, b []float32) float32
TEXT ·euclideanDistanceNEON32(SB), NOSPLIT, $0-52
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R2
    MOVD b_base+24(FP), R1

    VEOR V0.B16, V0.B16, V0.B16     // Accumulator = 0

    LSR $2, R2, R3
    CBZ R3, euclid32_scalar

euclid32_loop4:
    VLD1.P 16(R0), [V1.S4]     // Load a[i:i+4]
    VLD1.P 16(R1), [V2.S4]     // Load b[i:i+4]
    WORD $0x4EA2D421           // FSUB V1.4S, V1.4S, V2.4S  (diff = a[i] - b[i])
    WORD $0x4E21CC20           // FMLA V0.4S, V1.4S, V1.4S  (acc += diff * diff)
    SUB $1, R3
    CBNZ R3, euclid32_loop4

euclid32_scalar:
    AND $3, R2
    CBZ R2, euclid32_sqrt

    // Reduce vector FIRST before scalar ops
    WORD $0x6E20D400           // FADDP V0.4S, V0.4S, V0.4S
    WORD $0x7E30D800           // FADDP S0, V0.2S

euclid32_loop1:
    FMOVS (R0), F1
    FMOVS (R1), F2
    FSUBS F2, F1, F1           // diff = a[i] - b[i]
    FMADDS F1, F0, F1, F0      // acc += diff * diff
    ADD $4, R0
    ADD $4, R1
    SUB $1, R2
    CBNZ R2, euclid32_loop1
    FSQRTS F0, F0
    FMOVS F0, ret+48(FP)
    RET

euclid32_sqrt:
    WORD $0x6E20D400           // FADDP V0.4S, V0.4S, V0.4S
    WORD $0x7E30D800           // FADDP S0, V0.2S
    FSQRTS F0, F0
    FMOVS F0, ret+48(FP)
    RET

// ============================================================================
// CUBIC INTERPOLATION DOT PRODUCT
// ============================================================================

// func cubicInterpDotNEON(hist, a, b, c, d []float32, x float32) float32
// Computes: Σ hist[i] * (a[i] + x*(b[i] + x*(c[i] + x*d[i])))
// Uses Horner's method for polynomial evaluation with FMLA.
// Optimized with 2 independent accumulators for ILP.
//
// Frame layout (5 slices + 1 float32 + 1 return):
//   hist: base+0, len+8
//   a:    base+24, len+32
//   b:    base+48, len+56
//   c:    base+72, len+80
//   d:    base+96, len+104
//   x:    +120 (float32)
//   pad:  +124 (4 bytes alignment)
//   ret:  +128 (float32)
// Frame: 5 slices(120) + x(4) + pad(4) + ret(4) = 132 bytes
TEXT ·cubicInterpDotNEON(SB), NOSPLIT, $0-132
    MOVD hist_base+0(FP), R0   // R0 = hist pointer
    MOVD hist_len+8(FP), R6    // R6 = length
    MOVD a_base+24(FP), R1     // R1 = a pointer
    MOVD b_base+48(FP), R2     // R2 = b pointer
    MOVD c_base+72(FP), R3     // R3 = c pointer
    MOVD d_base+96(FP), R4     // R4 = d pointer
    FMOVS x+120(FP), F31       // F31 = x (scalar)

    // Broadcast x to all 4 lanes of V31
    // DUP Vd.4S, Vn.S[0]: 0x4E040400 | (Vn << 5) | Vd
    WORD $0x4E0407FF           // DUP V31.4S, V31.S[0]

    // Initialize dual accumulators to zero for ILP
    VEOR V0.B16, V0.B16, V0.B16  // acc0 = 0
    VEOR V1.B16, V1.B16, V1.B16  // acc1 = 0

    // Process 8 elements per iteration (2 NEON vectors)
    LSR $3, R6, R5             // R5 = len / 8
    CBZ R5, cubic32_neon_loop4_check

cubic32_neon_loop8:
    // Load first vector (4 elements)
    VLD1.P 16(R4), [V2.S4]     // V2 = d[i:i+4]
    VLD1.P 16(R3), [V3.S4]     // V3 = c[i:i+4]
    VLD1.P 16(R2), [V4.S4]     // V4 = b[i:i+4]
    VLD1.P 16(R1), [V5.S4]     // V5 = a[i:i+4]
    VLD1.P 16(R0), [V6.S4]     // V6 = hist[i:i+4]

    // Load second vector (4 elements)
    VLD1.P 16(R4), [V10.S4]    // V10 = d[i+4:i+8]
    VLD1.P 16(R3), [V11.S4]    // V11 = c[i+4:i+8]
    VLD1.P 16(R2), [V12.S4]    // V12 = b[i+4:i+8]
    VLD1.P 16(R1), [V13.S4]    // V13 = a[i+4:i+8]
    VLD1.P 16(R0), [V14.S4]    // V14 = hist[i+4:i+8]

    // Horner's method for first vector: coef = a + x*(b + x*(c + x*d))
    // FMLA Vd.4S, Vn.4S, Vm.4S: 0x4E20CC00 | (Vm << 16) | (Vn << 5) | Vd
    // Step 1: V3 = c + d*x
    WORD $0x4E3FCC43           // FMLA V3.4S, V2.4S, V31.4S
    // Step 2: V4 = b + (c + d*x)*x
    WORD $0x4E3FCC64           // FMLA V4.4S, V3.4S, V31.4S
    // Step 3: V5 = a + (b + (c + d*x)*x)*x = coef
    WORD $0x4E3FCC85           // FMLA V5.4S, V4.4S, V31.4S
    // Accumulate: acc0 += hist * coef
    WORD $0x4E25CCC0           // FMLA V0.4S, V6.4S, V5.4S

    // Horner's method for second vector
    WORD $0x4E3FCD4B           // FMLA V11.4S, V10.4S, V31.4S
    WORD $0x4E3FCD6C           // FMLA V12.4S, V11.4S, V31.4S
    WORD $0x4E3FCD8D           // FMLA V13.4S, V12.4S, V31.4S
    // Accumulate: acc1 += hist * coef
    WORD $0x4E2DCDC1           // FMLA V1.4S, V14.4S, V13.4S

    SUB $1, R5
    CBNZ R5, cubic32_neon_loop8

    // Combine accumulators: V0 = V0 + V1
    WORD $0x4E21D400           // FADD V0.4S, V0.4S, V1.4S

cubic32_neon_loop4_check:
    // Check for 4-7 remaining elements
    AND $7, R6, R5
    LSR $2, R5, R7             // R7 = remainder / 4
    CBZ R7, cubic32_neon_remainder

    // Process 4 elements
    VLD1.P 16(R4), [V2.S4]     // V2 = d
    VLD1.P 16(R3), [V3.S4]     // V3 = c
    VLD1.P 16(R2), [V4.S4]     // V4 = b
    VLD1.P 16(R1), [V5.S4]     // V5 = a
    VLD1.P 16(R0), [V6.S4]     // V6 = hist

    // Horner's method
    WORD $0x4E3FCC43           // FMLA V3.4S, V2.4S, V31.4S
    WORD $0x4E3FCC64           // FMLA V4.4S, V3.4S, V31.4S
    WORD $0x4E3FCC85           // FMLA V5.4S, V4.4S, V31.4S
    WORD $0x4E25CCC0           // FMLA V0.4S, V6.4S, V5.4S

cubic32_neon_remainder:
    // Reduce vector FIRST before scalar ops
    WORD $0x6E20D400           // FADDP V0.4S, V0.4S, V0.4S
    WORD $0x7E30D800           // FADDP S0, V0.2S

    // Check for remaining 1-3 elements
    AND $3, R5, R7
    CBZ R7, cubic32_neon_done

cubic32_neon_scalar:
    // Scalar path for remaining elements
    FMOVS (R4), F2             // d
    FMOVS (R3), F3             // c
    FMOVS (R2), F4             // b
    FMOVS (R1), F5             // a
    FMOVS (R0), F6             // hist
    FMOVS x+120(FP), F7        // x

    // Horner's method scalar: coef = a + x*(b + x*(c + x*d))
    // Go FMADDS syntax: Fm, Fa, Fn, Fd -> Fd = Fn * Fm + Fa
    FMADDS F7, F3, F2, F3      // F3 = F2*F7 + F3 = d*x + c
    FMADDS F7, F4, F3, F4      // F4 = F3*F7 + F4 = (d*x+c)*x + b
    FMADDS F7, F5, F4, F5      // F5 = F4*F7 + F5 = coef
    FMADDS F5, F0, F6, F0      // F0 = F6*F5 + F0 = hist*coef + acc

    ADD $4, R0
    ADD $4, R1
    ADD $4, R2
    ADD $4, R3
    ADD $4, R4
    SUB $1, R7
    CBNZ R7, cubic32_neon_scalar

cubic32_neon_done:
    FMOVS F0, ret+128(FP)
    RET

// func sigmoidNEON(dst, src []float32)
// Computes accurate sigmoid: σ(x) = 1 / (1 + exp(-x))
// Uses range reduction and polynomial approximation for exp.
TEXT ·sigmoidNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1

    // Load constants into vector registers
    // log2(e) = 1.442695041 = 0x3fb8aa3b
    MOVW $0x3fb8aa3b, R10
    VMOV R10, V20.S[0]
    VDUP V20.S[0], V20.S4         // V20 = log2(e)

    // ln(2) = 0.693147181 = 0x3f317218
    MOVW $0x3f317218, R10
    VMOV R10, V21.S[0]
    VDUP V21.S[0], V21.S4         // V21 = ln(2)

    // 1.0 = 0x3f800000
    FMOVS $1.0, F22
    VDUP V22.S[0], V22.S4         // V22 = 1.0

    // Polynomial coefficients: c2=0.5, c3=1/6, c4=1/24, c5=1/120
    FMOVS $0.5, F23
    VDUP V23.S[0], V23.S4         // V23 = c2 = 0.5

    // c3 = 1/6 = 0x3e2aaaab
    MOVW $0x3e2aaaab, R10
    VMOV R10, V24.S[0]
    VDUP V24.S[0], V24.S4         // V24 = c3 = 1/6

    // c4 = 1/24 = 0x3d2aaaab
    MOVW $0x3d2aaaab, R10
    VMOV R10, V25.S[0]
    VDUP V25.S[0], V25.S4         // V25 = c4 = 1/24

    // c5 = 1/120 = 0x3c088889
    MOVW $0x3c088889, R10
    VMOV R10, V26.S[0]
    VDUP V26.S[0], V26.S4         // V26 = c5 = 1/120

    // Clamp thresholds: ±20.0 = 0x41a00000 / 0xc1a00000
    MOVW $0x41a00000, R10
    VMOV R10, V27.S[0]
    VDUP V27.S[0], V27.S4         // V27 = 20.0 (clamp_hi)

    MOVW $0xc1a00000, R10
    VMOV R10, V28.S[0]
    VDUP V28.S[0], V28.S4         // V28 = -20.0 (clamp_lo)

    // Process 4 elements per iteration
    LSR $2, R3, R4
    CBZ R4, sigmoid32_neon_scalar

sigmoid32_neon_loop4:
    VLD1.P 16(R1), [V0.S4]        // V0 = x

    // Negate: V0 = -x
    WORD $0x6EA07C00              // FNEG V0.4S, V0.4S

    // Clamp -x to [-20, 20]
    WORD $0x4EBBF400              // FMIN V0.4S, V0.4S, V27.4S  (clamp upper to 20)
    WORD $0x4E3CF400              // FMAX V0.4S, V0.4S, V28.4S  (clamp lower to -20)

    // Range reduction: k = round(-x * log2e), r = -x - k * ln2
    WORD $0x6E34DC01              // FMUL V1.4S, V0.4S, V20.4S   V1 = -x * log2e
    WORD $0x4E218C22              // FRINTN V2.4S, V1.4S         V2 = k = round(V1)
    // V3 = V0 - V2 * ln2
    WORD $0x6E35DC44              // FMUL V4.4S, V2.4S, V21.4S   V4 = k * ln2
    WORD $0x4EA4D403              // FSUB V3.4S, V0.4S, V4.4S    V3 = r = -x - k * ln2

    // Polynomial: exp(r) ≈ 1 + r*(1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
    // Horner's method
    WORD $0x6E3ADC64              // FMUL V4.4S, V3.4S, V26.4S   V4 = r * c5
    WORD $0x4E39D484              // FADD V4.4S, V4.4S, V25.4S   V4 = c4 + r*c5
    WORD $0x6E23DC84              // FMUL V4.4S, V4.4S, V3.4S    V4 = r*(c4 + r*c5)
    WORD $0x4E38D484              // FADD V4.4S, V4.4S, V24.4S   V4 = c3 + r*(...)
    WORD $0x6E23DC84              // FMUL V4.4S, V4.4S, V3.4S    V4 = r*(c3 + ...)
    WORD $0x4E37D484              // FADD V4.4S, V4.4S, V23.4S   V4 = c2 + r*(...)
    WORD $0x6E23DC84              // FMUL V4.4S, V4.4S, V3.4S    V4 = r*(c2 + ...)
    WORD $0x4E36D484              // FADD V4.4S, V4.4S, V22.4S   V4 = 1 + r*(...)
    WORD $0x6E23DC84              // FMUL V4.4S, V4.4S, V3.4S    V4 = r*(1 + ...)
    WORD $0x4E36D484              // FADD V4.4S, V4.4S, V22.4S   V4 = exp(r)

    // Reconstruct: exp(-x) = exp(r) * 2^k
    // Convert k to int, shift to exponent position, add to 1.0's bits
    WORD $0x4EA1B841              // FCVTZS V1.4S, V2.4S         V1 = int(k)
    WORD $0x4F375421              // SHL V1.4S, V1.4S, #23       V1 = k << 23
    WORD $0x4EB68421              // ADD V1.4S, V1.4S, V22.4S    V1 = 2^k (add 1.0's bits)
    WORD $0x6E21DC84              // FMUL V4.4S, V4.4S, V1.4S    V4 = exp(-x) = exp(r) * 2^k

    // Sigmoid: 1 / (1 + exp(-x))
    WORD $0x4E36D484              // FADD V4.4S, V4.4S, V22.4S   V4 = 1 + exp(-x)
    WORD $0x6E24FEC0              // FDIV V0.4S, V22.4S, V4.4S   V0 = 1 / (1 + exp(-x))

    VST1.P [V0.S4], 16(R0)        // store result

    SUB $1, R4
    CBNZ R4, sigmoid32_neon_loop4

sigmoid32_neon_scalar:
    AND $3, R3
    CBZ R3, sigmoid32_neon_done

sigmoid32_neon_scalar_loop:
    // Scalar path uses pure Go fallback approach
    FMOVS (R1), F0                // F0 = x
    FNEGS F0, F0                  // F0 = -x

    // Clamp
    FMOVS $20.0, F1
    FMOVS $-20.0, F2
    FMINS F1, F0, F0
    FMAXS F2, F0, F0

    // Range reduction
    // log2e = 1.442695041
    MOVW $0x3fb8aa3b, R10
    FMOVS R10, F8
    FMULS F8, F0, F1              // F1 = -x * log2e
    FRINTNS F1, F2                // F2 = k = round(F1)

    // ln2 = 0.693147181
    MOVW $0x3f317218, R10
    FMOVS R10, F9
    FMULS F9, F2, F3              // F3 = k * ln2
    FSUBS F3, F0, F0              // F0 = r = -x - k * ln2

    // Polynomial coefficients
    FMOVS $1.0, F10               // c1 = 1.0
    FMOVS $0.5, F11               // c2 = 0.5
    MOVW $0x3e2aaaab, R10
    FMOVS R10, F12                // c3 = 1/6
    MOVW $0x3d2aaaab, R10
    FMOVS R10, F13                // c4 = 1/24
    MOVW $0x3c088889, R10
    FMOVS R10, F14                // c5 = 1/120

    // Horner's method
    FMULS F0, F14, F4             // F4 = r * c5
    FADDS F13, F4, F4             // F4 = c4 + r*c5
    FMULS F0, F4, F4              // F4 = r*(c4 + r*c5)
    FADDS F12, F4, F4             // F4 = c3 + r*(...)
    FMULS F0, F4, F4
    FADDS F11, F4, F4             // F4 = c2 + r*(...)
    FMULS F0, F4, F4
    FADDS F10, F4, F4             // F4 = 1 + r*(...)
    FMULS F0, F4, F4
    FADDS F10, F4, F4             // F4 = exp(r)

    // Reconstruct 2^k
    FCVTZSS F2, R10               // R10 = int(k)
    LSL $23, R10, R10             // R10 = k << 23
    MOVW $0x3f800000, R11
    ADD R11, R10, R10             // R10 = 2^k bits
    FMOVS R10, F5
    FMULS F5, F4, F4              // F4 = exp(-x)

    // Sigmoid
    FADDS F10, F4, F4             // F4 = 1 + exp(-x)
    FDIVS F4, F10, F0             // F0 = 1 / (1 + exp(-x))
    FMOVS F0, (R0)                // store result

    ADD $4, R0
    ADD $4, R1
    SUB $1, R3
    CBNZ R3, sigmoid32_neon_scalar_loop

sigmoid32_neon_done:
    RET

// func reluNEON(dst, src []float32)
// Computes ReLU: dst[i] = max(0, src[i])
TEXT ·reluNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1

    // Create zero vector
    VEOR V30.B16, V30.B16, V30.B16    // V30 = {0, 0, 0, 0}

    // Process 4 elements per iteration
    LSR $2, R3, R4
    CBZ R4, relu32_neon_scalar

relu32_neon_loop4:
    VLD1.P 16(R1), [V0.S4]            // V0 = x
    WORD $0x4E3EF401                  // FMAX V1.4S, V0.4S, V30.4S -> V1 = max(x, 0)
    VST1.P [V1.S4], 16(R0)            // store result

    SUB $1, R4
    CBNZ R4, relu32_neon_loop4

relu32_neon_scalar:
    AND $3, R3
    CBZ R3, relu32_neon_done

relu32_neon_scalar_loop:
    FMOVS (R1), F0                    // F0 = x
    FMOVS $0.0, F1
    FMAXS F1, F0, F2                  // F2 = max(x, 0)
    FMOVS F2, (R0)                    // store result

    ADD $4, R0
    ADD $4, R1
    SUB $1, R3
    CBNZ R3, relu32_neon_scalar_loop

relu32_neon_done:
    RET

// func tanhNEON(dst, src []float32)
// Computes accurate tanh: tanh(x) = (1 - e^(-2x)) / (1 + e^(-2x))
// Uses range reduction and polynomial approximation for exp (same as sigmoid).
TEXT ·tanhNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1

    // Load constants into vector registers
    // 2.0 = 0x40000000
    MOVW $0x40000000, R10
    VMOV R10, V19.S[0]
    VDUP V19.S[0], V19.S4             // V19 = 2.0

    // log2(e) = 1.442695041 = 0x3fb8aa3b
    MOVW $0x3fb8aa3b, R10
    VMOV R10, V20.S[0]
    VDUP V20.S[0], V20.S4             // V20 = log2(e)

    // ln(2) = 0.693147181 = 0x3f317218
    MOVW $0x3f317218, R10
    VMOV R10, V21.S[0]
    VDUP V21.S[0], V21.S4             // V21 = ln(2)

    // 1.0 = 0x3f800000
    FMOVS $1.0, F22
    VDUP V22.S[0], V22.S4             // V22 = 1.0

    // Polynomial coefficients: c2=0.5, c3=1/6, c4=1/24, c5=1/120
    FMOVS $0.5, F23
    VDUP V23.S[0], V23.S4             // V23 = c2 = 0.5

    // c3 = 1/6 = 0x3e2aaaab
    MOVW $0x3e2aaaab, R10
    VMOV R10, V24.S[0]
    VDUP V24.S[0], V24.S4             // V24 = c3 = 1/6

    // c4 = 1/24 = 0x3d2aaaab
    MOVW $0x3d2aaaab, R10
    VMOV R10, V25.S[0]
    VDUP V25.S[0], V25.S4             // V25 = c4 = 1/24

    // c5 = 1/120 = 0x3c088889
    MOVW $0x3c088889, R10
    VMOV R10, V26.S[0]
    VDUP V26.S[0], V26.S4             // V26 = c5 = 1/120

    // Clamp thresholds for -2x: ±20.0 (tanh saturates at ~±10, so -2x range is ±20)
    MOVW $0x41a00000, R10
    VMOV R10, V27.S[0]
    VDUP V27.S[0], V27.S4             // V27 = 20.0 (clamp_hi)

    MOVW $0xc1a00000, R10
    VMOV R10, V28.S[0]
    VDUP V28.S[0], V28.S4             // V28 = -20.0 (clamp_lo)

    // Process 4 elements per iteration
    LSR $2, R3, R4
    CBZ R4, tanh32_neon_scalar

tanh32_neon_loop4:
    VLD1.P 16(R1), [V0.S4]            // V0 = x

    // Compute -2x: negate then multiply by 2
    WORD $0x6EA07C00                  // FNEG V0.4S, V0.4S           V0 = -x
    WORD $0x6E33DC00                  // FMUL V0.4S, V0.4S, V19.4S   V0 = -2x

    // Clamp -2x to [-20, 20]
    WORD $0x4EBBF400                  // FMIN V0.4S, V0.4S, V27.4S   clamp upper to 20
    WORD $0x4E3CF400                  // FMAX V0.4S, V0.4S, V28.4S   clamp lower to -20

    // Range reduction: k = round(-2x * log2e), r = -2x - k * ln2
    WORD $0x6E34DC01                  // FMUL V1.4S, V0.4S, V20.4S   V1 = -2x * log2e
    WORD $0x4E218C22                  // FRINTN V2.4S, V1.4S         V2 = k = round(V1)
    WORD $0x6E35DC44                  // FMUL V4.4S, V2.4S, V21.4S   V4 = k * ln2
    WORD $0x4EA4D403                  // FSUB V3.4S, V0.4S, V4.4S    V3 = r = -2x - k * ln2

    // Polynomial: exp(r) ≈ 1 + r*(1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
    // Horner's method
    WORD $0x6E3ADC64                  // FMUL V4.4S, V3.4S, V26.4S   V4 = r * c5
    WORD $0x4E39D484                  // FADD V4.4S, V4.4S, V25.4S   V4 = c4 + r*c5
    WORD $0x6E23DC84                  // FMUL V4.4S, V4.4S, V3.4S    V4 = r*(c4 + r*c5)
    WORD $0x4E38D484                  // FADD V4.4S, V4.4S, V24.4S   V4 = c3 + r*(...)
    WORD $0x6E23DC84                  // FMUL V4.4S, V4.4S, V3.4S    V4 = r*(c3 + ...)
    WORD $0x4E37D484                  // FADD V4.4S, V4.4S, V23.4S   V4 = c2 + r*(...)
    WORD $0x6E23DC84                  // FMUL V4.4S, V4.4S, V3.4S    V4 = r*(c2 + ...)
    WORD $0x4E36D484                  // FADD V4.4S, V4.4S, V22.4S   V4 = 1 + r*(...)
    WORD $0x6E23DC84                  // FMUL V4.4S, V4.4S, V3.4S    V4 = r*(1 + ...)
    WORD $0x4E36D484                  // FADD V4.4S, V4.4S, V22.4S   V4 = exp(r)

    // Reconstruct: exp(-2x) = exp(r) * 2^k
    WORD $0x4EA1B841                  // FCVTZS V1.4S, V2.4S         V1 = int(k)
    WORD $0x4F375421                  // SHL V1.4S, V1.4S, #23       V1 = k << 23
    WORD $0x4EB68421                  // ADD V1.4S, V1.4S, V22.4S    V1 = 2^k (add 1.0's bits)
    WORD $0x6E21DC84                  // FMUL V4.4S, V4.4S, V1.4S    V4 = e^(-2x)

    // tanh(x) = (1 - e^(-2x)) / (1 + e^(-2x))
    WORD $0x4EA4D6C5                  // FSUB V5.4S, V22.4S, V4.4S   V5 = 1 - e^(-2x)
    WORD $0x4E24D6C6                  // FADD V6.4S, V22.4S, V4.4S   V6 = 1 + e^(-2x)
    WORD $0x6E26FCA0                  // FDIV V0.4S, V5.4S, V6.4S    V0 = tanh(x)

    VST1.P [V0.S4], 16(R0)            // store result

    SUB $1, R4
    CBNZ R4, tanh32_neon_loop4

tanh32_neon_scalar:
    AND $3, R3
    CBZ R3, tanh32_neon_done

tanh32_neon_scalar_loop:
    // Scalar path for remaining elements
    FMOVS (R1), F0                    // F0 = x
    FNEGS F0, F0                      // F0 = -x

    // Multiply by 2: -2x
    FMOVS $2.0, F7
    FMULS F7, F0, F0                  // F0 = -2x

    // Clamp
    FMOVS $20.0, F1
    FMOVS $-20.0, F2
    FMINS F1, F0, F0
    FMAXS F2, F0, F0

    // Range reduction
    MOVW $0x3fb8aa3b, R10
    FMOVS R10, F8
    FMULS F8, F0, F1                  // F1 = -2x * log2e
    FRINTNS F1, F2                    // F2 = k = round(F1)

    MOVW $0x3f317218, R10
    FMOVS R10, F9
    FMULS F9, F2, F3                  // F3 = k * ln2
    FSUBS F3, F0, F0                  // F0 = r = -2x - k * ln2

    // Polynomial coefficients
    FMOVS $1.0, F10                   // c1 = 1.0
    FMOVS $0.5, F11                   // c2 = 0.5
    MOVW $0x3e2aaaab, R10
    FMOVS R10, F12                    // c3 = 1/6
    MOVW $0x3d2aaaab, R10
    FMOVS R10, F13                    // c4 = 1/24
    MOVW $0x3c088889, R10
    FMOVS R10, F14                    // c5 = 1/120

    // Horner's method
    FMULS F0, F14, F4                 // F4 = r * c5
    FADDS F13, F4, F4                 // F4 = c4 + r*c5
    FMULS F0, F4, F4
    FADDS F12, F4, F4                 // F4 = c3 + r*(...)
    FMULS F0, F4, F4
    FADDS F11, F4, F4                 // F4 = c2 + r*(...)
    FMULS F0, F4, F4
    FADDS F10, F4, F4                 // F4 = 1 + r*(...)
    FMULS F0, F4, F4
    FADDS F10, F4, F4                 // F4 = exp(r)

    // Reconstruct 2^k
    FCVTZSS F2, R10
    LSL $23, R10, R10
    MOVW $0x3f800000, R11
    ADD R11, R10, R10
    FMOVS R10, F5
    FMULS F5, F4, F4                  // F4 = e^(-2x)

    // tanh(x) = (1 - e^(-2x)) / (1 + e^(-2x))
    FSUBS F4, F10, F5                 // F5 = 1 - e^(-2x)
    FADDS F4, F10, F6                 // F6 = 1 + e^(-2x)
    FDIVS F6, F5, F0                  // F0 = tanh(x)

    FMOVS F0, (R0)                    // store result

    ADD $4, R0
    ADD $4, R1
    SUB $1, R3
    CBNZ R3, tanh32_neon_scalar_loop

tanh32_neon_done:
    RET

// func clampScaleNEON(dst, src []float32, minVal, maxVal, scale float32)
// Performs fused clamp and scale: dst[i] = (clamp(src[i], minVal, maxVal) - minVal) * scale
TEXT ·clampScaleNEON(SB), NOSPLIT, $0-60
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1
    FMOVS minVal+48(FP), F4
    FMOVS maxVal+52(FP), F5
    FMOVS scale+56(FP), F6

    // Duplicate scalars to SIMD vectors
    WORD $0x4E040484                  // DUP V4.4S, V4.S[0] -> V4 = minVal
    WORD $0x4E0404A5                  // DUP V5.4S, V5.S[0] -> V5 = maxVal
    WORD $0x4E0404C6                  // DUP V6.4S, V6.S[0] -> V6 = scale

    // Process 4 elements per iteration
    LSR $2, R3, R4
    CBZ R4, clampscale32_neon_scalar

clampscale32_neon_loop4:
    VLD1.P 16(R1), [V0.S4]            // V0 = src[i]
    WORD $0x4E24F400                  // FMAX V0.4S, V0.4S, V4.4S -> clamp to min
    WORD $0x4EA5F400                  // FMIN V0.4S, V0.4S, V5.4S -> clamp to max
    WORD $0x4EA4D400                  // FSUB V0.4S, V0.4S, V4.4S -> subtract minVal
    WORD $0x6E26DC00                  // FMUL V0.4S, V0.4S, V6.4S -> multiply by scale
    VST1.P [V0.S4], 16(R0)            // store result
    SUB $1, R4
    CBNZ R4, clampscale32_neon_loop4

clampscale32_neon_scalar:
    AND $3, R3
    CBZ R3, clampscale32_neon_done

clampscale32_neon_scalar_loop:
    FMOVS (R1), F0                    // F0 = src[i]
    FMAXS F0, F4, F0                  // F0 = max(src[i], minVal)
    FMINS F0, F5, F0                  // F0 = min(max(src[i], minVal), maxVal)
    FSUBS F4, F0, F0                  // F0 = clamped - minVal
    FMULS F0, F6, F0                  // F0 = (clamped - minVal) * scale
    FMOVS F0, (R0)                    // store result
    ADD $4, R0
    ADD $4, R1
    SUB $1, R3
    CBNZ R3, clampscale32_neon_scalar_loop

clampscale32_neon_done:
    RET

// func int32ToFloat32ScaleNEON(dst []float32, src []int32, scale float32)
// Converts int32 samples to float32 and multiplies by scale in one pass.
// dst[i] = float32(src[i]) * scale
// Optimized for audio PCM conversion (e.g., scale = 1.0/32768 for 16-bit).
//
// NEON opcodes:
// SCVTF Vd.4S, Vn.4S: 0x4E21D800 | (Vn << 5) | Vd  (signed int32 to float32)
// FMUL Vd.4S, Vn.4S, Vm.4S: 0x6E20DC00 | (Vm << 16) | (Vn << 5) | Vd
//
// Frame: dst(24) + src(24) + scale(4) = 52 bytes
TEXT ·int32ToFloat32ScaleNEON(SB), NOSPLIT, $0-52
    MOVD dst_base+0(FP), R0        // R0 = dst pointer
    MOVD dst_len+8(FP), R3         // R3 = length
    MOVD src_base+24(FP), R1       // R1 = src pointer
    FMOVS scale+48(FP), F2         // F2 = scale

    // Broadcast scale to all 4 lanes
    // DUP V2.4S, V2.S[0]: 0x4E040400 | (Vn << 5) | Vd
    WORD $0x4E040442               // DUP V2.4S, V2.S[0]

    // Process 4 int32 elements per iteration
    LSR $2, R3, R4                 // R4 = len / 4
    CBZ R4, i32tof32_neon_scalar

i32tof32_neon_loop4:
    VLD1.P 16(R1), [V0.S4]         // V0 = 4 x int32
    // SCVTF V1.4S, V0.4S - convert signed int32 to float32
    WORD $0x4E21D801               // SCVTF V1.4S, V0.4S
    // FMUL V1.4S, V1.4S, V2.4S - multiply by scale
    WORD $0x6E22DC21               // FMUL V1.4S, V1.4S, V2.4S
    VST1.P [V1.S4], 16(R0)         // store 4 x float32
    SUB $1, R4
    CBNZ R4, i32tof32_neon_loop4

i32tof32_neon_scalar:
    AND $3, R3                     // remainder = len % 4
    CBZ R3, i32tof32_neon_done

i32tof32_neon_scalar_loop:
    MOVW (R1), R5                  // Load int32
    SCVTFWS R5, F0                 // Convert int32 to float32
    FMULS F2, F0, F0               // F0 = F0 * scale
    FMOVS F0, (R0)                 // Store float32
    ADD $4, R0
    ADD $4, R1
    SUB $1, R3
    CBNZ R3, i32tof32_neon_scalar_loop

i32tof32_neon_done:
    RET

// ============================================================================
// SPLIT-FORMAT COMPLEX OPERATIONS
// ============================================================================
//
// These operate on split real/imag arrays - much simpler than interleaved
// because we can load real and imag values directly without shuffling.
//
// NEON opcodes used:
// FMUL Vd.4S, Vn.4S, Vm.4S: 0x6E20DC00 | (Vm << 16) | (Vn << 5) | Vd
// FSUB Vd.4S, Vn.4S, Vm.4S: 0x4EA0D400 | (Vm << 16) | (Vn << 5) | Vd
// FADD Vd.4S, Vn.4S, Vm.4S: 0x4E20D400 | (Vm << 16) | (Vn << 5) | Vd
// FMLA Vd.4S, Vn.4S, Vm.4S: 0x4E20CC00 | (Vm << 16) | (Vn << 5) | Vd (Vd += Vn * Vm)
// FMLS Vd.4S, Vn.4S, Vm.4S: 0x4EA0CC00 | (Vm << 16) | (Vn << 5) | Vd (Vd -= Vn * Vm)

// func mulComplexNEON(dstRe, dstIm, aRe, aIm, bRe, bIm []float32)
// Computes element-wise complex multiplication using split arrays:
//   dstRe[i] = aRe[i]*bRe[i] - aIm[i]*bIm[i]
//   dstIm[i] = aRe[i]*bIm[i] + aIm[i]*bRe[i]
// Frame: dstRe(24) + dstIm(24) + aRe(24) + aIm(24) + bRe(24) + bIm(24) = 144 bytes
TEXT ·mulComplexNEON(SB), NOSPLIT, $0-144
    MOVD dstRe_base+0(FP), R0      // R0 = dstRe pointer
    MOVD dstRe_len+8(FP), R3       // R3 = length
    MOVD dstIm_base+24(FP), R1     // R1 = dstIm pointer
    MOVD aRe_base+48(FP), R4       // R4 = aRe pointer
    MOVD aIm_base+72(FP), R5       // R5 = aIm pointer
    MOVD bRe_base+96(FP), R6       // R6 = bRe pointer
    MOVD bIm_base+120(FP), R7      // R7 = bIm pointer

    // Process 4 elements per iteration
    LSR $2, R3, R8                 // R8 = len / 4
    CBZ R8, mulcplx_neon_scalar

mulcplx_neon_loop4:
    // Load inputs
    VLD1.P 16(R4), [V0.S4]         // V0 = aRe[0:4]
    VLD1.P 16(R5), [V1.S4]         // V1 = aIm[0:4]
    VLD1.P 16(R6), [V2.S4]         // V2 = bRe[0:4]
    VLD1.P 16(R7), [V3.S4]         // V3 = bIm[0:4]

    // dstRe = aRe*bRe - aIm*bIm
    // V4 = aRe * bRe
    WORD $0x6E22DC04               // FMUL V4.4S, V0.4S, V2.4S
    // V4 = V4 - aIm*bIm (FMLS: Vd -= Vn*Vm)
    WORD $0x4EA3CC24               // FMLS V4.4S, V1.4S, V3.4S

    // dstIm = aRe*bIm + aIm*bRe
    // V5 = aRe * bIm
    WORD $0x6E23DC05               // FMUL V5.4S, V0.4S, V3.4S
    // V5 = V5 + aIm*bRe (FMLA: Vd += Vn*Vm)
    WORD $0x4E22CC25               // FMLA V5.4S, V1.4S, V2.4S

    // Store results
    VST1.P [V4.S4], 16(R0)         // dstRe[0:4]
    VST1.P [V5.S4], 16(R1)         // dstIm[0:4]

    SUB $1, R8
    CBNZ R8, mulcplx_neon_loop4

mulcplx_neon_scalar:
    AND $3, R3                     // remainder
    CBZ R3, mulcplx_neon_done

mulcplx_neon_scalar_loop:
    // Load single elements
    FMOVS (R4), F0                 // F0 = aRe
    FMOVS (R5), F1                 // F1 = aIm
    FMOVS (R6), F2                 // F2 = bRe
    FMOVS (R7), F3                 // F3 = bIm

    // dstRe = aRe*bRe - aIm*bIm (avoid FMA for clarity)
    // Go ARM64: FSUBS Fa, Fb, Fd → Fd = Fb - Fa
    FMULS F0, F2, F4               // F4 = aRe * bRe
    FMULS F1, F3, F5               // F5 = aIm * bIm
    FSUBS F5, F4, F4               // F4 = F4 - F5 = aRe*bRe - aIm*bIm

    // dstIm = aRe*bIm + aIm*bRe (avoid FMA for clarity)
    FMULS F0, F3, F5               // F5 = aRe * bIm
    FMULS F1, F2, F6               // F6 = aIm * bRe
    FADDS F5, F6, F5               // F5 = F5 + F6 = aRe*bIm + aIm*bRe

    // Store results
    FMOVS F4, (R0)
    FMOVS F5, (R1)

    ADD $4, R0
    ADD $4, R1
    ADD $4, R4
    ADD $4, R5
    ADD $4, R6
    ADD $4, R7
    SUB $1, R3
    CBNZ R3, mulcplx_neon_scalar_loop

mulcplx_neon_done:
    RET

// func mulConjComplexNEON(dstRe, dstIm, aRe, aIm, bRe, bIm []float32)
// Computes element-wise multiplication by conjugate using split arrays:
//   dstRe[i] = aRe[i]*bRe[i] + aIm[i]*bIm[i]
//   dstIm[i] = aIm[i]*bRe[i] - aRe[i]*bIm[i]
// Frame: dstRe(24) + dstIm(24) + aRe(24) + aIm(24) + bRe(24) + bIm(24) = 144 bytes
TEXT ·mulConjComplexNEON(SB), NOSPLIT, $0-144
    MOVD dstRe_base+0(FP), R0      // R0 = dstRe pointer
    MOVD dstRe_len+8(FP), R3       // R3 = length
    MOVD dstIm_base+24(FP), R1     // R1 = dstIm pointer
    MOVD aRe_base+48(FP), R4       // R4 = aRe pointer
    MOVD aIm_base+72(FP), R5       // R5 = aIm pointer
    MOVD bRe_base+96(FP), R6       // R6 = bRe pointer
    MOVD bIm_base+120(FP), R7      // R7 = bIm pointer

    // Process 4 elements per iteration
    LSR $2, R3, R8
    CBZ R8, mulconjcplx_neon_scalar

mulconjcplx_neon_loop4:
    // Load inputs
    VLD1.P 16(R4), [V0.S4]         // V0 = aRe[0:4]
    VLD1.P 16(R5), [V1.S4]         // V1 = aIm[0:4]
    VLD1.P 16(R6), [V2.S4]         // V2 = bRe[0:4]
    VLD1.P 16(R7), [V3.S4]         // V3 = bIm[0:4]

    // dstRe = aRe*bRe + aIm*bIm
    // V4 = aRe * bRe
    WORD $0x6E22DC04               // FMUL V4.4S, V0.4S, V2.4S
    // V4 = V4 + aIm*bIm (FMLA)
    WORD $0x4E23CC24               // FMLA V4.4S, V1.4S, V3.4S

    // dstIm = aIm*bRe - aRe*bIm
    // V5 = aIm * bRe
    WORD $0x6E22DC25               // FMUL V5.4S, V1.4S, V2.4S
    // V5 = V5 - aRe*bIm (FMLS)
    WORD $0x4EA3CC05               // FMLS V5.4S, V0.4S, V3.4S

    // Store results
    VST1.P [V4.S4], 16(R0)
    VST1.P [V5.S4], 16(R1)

    SUB $1, R8
    CBNZ R8, mulconjcplx_neon_loop4

mulconjcplx_neon_scalar:
    AND $3, R3
    CBZ R3, mulconjcplx_neon_done

mulconjcplx_neon_scalar_loop:
    FMOVS (R4), F0                 // F0 = aRe
    FMOVS (R5), F1                 // F1 = aIm
    FMOVS (R6), F2                 // F2 = bRe
    FMOVS (R7), F3                 // F3 = bIm

    // dstRe = aRe*bRe + aIm*bIm (avoid FMA for clarity)
    FMULS F0, F2, F4               // F4 = aRe * bRe
    FMULS F1, F3, F5               // F5 = aIm * bIm
    FADDS F4, F5, F4               // F4 = aRe*bRe + aIm*bIm

    // dstIm = aIm*bRe - aRe*bIm (avoid FMA for clarity)
    // Go ARM64: FSUBS Fa, Fb, Fd → Fd = Fb - Fa
    FMULS F1, F2, F5               // F5 = aIm * bRe
    FMULS F0, F3, F6               // F6 = aRe * bIm
    FSUBS F6, F5, F5               // F5 = F5 - F6 = aIm*bRe - aRe*bIm

    FMOVS F4, (R0)
    FMOVS F5, (R1)

    ADD $4, R0
    ADD $4, R1
    ADD $4, R4
    ADD $4, R5
    ADD $4, R6
    ADD $4, R7
    SUB $1, R3
    CBNZ R3, mulconjcplx_neon_scalar_loop

mulconjcplx_neon_done:
    RET

// func absSqComplexNEON(dst, aRe, aIm []float32)
// Computes element-wise magnitude squared using split arrays:
//   dst[i] = aRe[i]^2 + aIm[i]^2
// Frame: dst(24) + aRe(24) + aIm(24) = 72 bytes
TEXT ·absSqComplexNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0        // R0 = dst pointer
    MOVD dst_len+8(FP), R3         // R3 = length
    MOVD aRe_base+24(FP), R4       // R4 = aRe pointer
    MOVD aIm_base+48(FP), R5       // R5 = aIm pointer

    // Process 4 elements per iteration
    LSR $2, R3, R8
    CBZ R8, abssqcplx_neon_scalar

abssqcplx_neon_loop4:
    // Load inputs
    VLD1.P 16(R4), [V0.S4]         // V0 = aRe[0:4]
    VLD1.P 16(R5), [V1.S4]         // V1 = aIm[0:4]

    // dst = aRe^2 + aIm^2
    // V2 = aRe * aRe
    WORD $0x6E20DC02               // FMUL V2.4S, V0.4S, V0.4S
    // V2 = V2 + aIm*aIm (FMLA)
    WORD $0x4E21CC22               // FMLA V2.4S, V1.4S, V1.4S

    // Store result
    VST1.P [V2.S4], 16(R0)

    SUB $1, R8
    CBNZ R8, abssqcplx_neon_loop4

abssqcplx_neon_scalar:
    AND $3, R3
    CBZ R3, abssqcplx_neon_done

abssqcplx_neon_scalar_loop:
    FMOVS (R4), F0                 // F0 = aRe
    FMOVS (R5), F1                 // F1 = aIm

    // dst = aRe^2 + aIm^2 (avoid FMA for clarity)
    FMULS F0, F0, F2               // F2 = aRe^2
    FMULS F1, F1, F3               // F3 = aIm^2
    FADDS F2, F3, F2               // F2 = aRe^2 + aIm^2

    FMOVS F2, (R0)

    ADD $4, R0
    ADD $4, R4
    ADD $4, R5
    SUB $1, R3
    CBNZ R3, abssqcplx_neon_scalar_loop

abssqcplx_neon_done:
    RET

// func butterflyComplexNEON(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm []float32)
// Computes FFT butterfly with twiddle factor multiply using split arrays:
//   temp_re = lower_re[i]*tw_re[i] - lower_im[i]*tw_im[i]
//   temp_im = lower_re[i]*tw_im[i] + lower_im[i]*tw_re[i]
//   upper_re[i], lower_re[i] = upper_re[i]+temp_re, upper_re[i]-temp_re
//   upper_im[i], lower_im[i] = upper_im[i]+temp_im, upper_im[i]-temp_im
// Frame: upperRe(24) + upperIm(24) + lowerRe(24) + lowerIm(24) + twRe(24) + twIm(24) = 144 bytes
TEXT ·butterflyComplexNEON(SB), NOSPLIT, $0-144
    MOVD upperRe_base+0(FP), R0      // R0 = upperRe pointer
    MOVD upperRe_len+8(FP), R3       // R3 = length
    MOVD upperIm_base+24(FP), R1     // R1 = upperIm pointer
    MOVD lowerRe_base+48(FP), R4     // R4 = lowerRe pointer
    MOVD lowerIm_base+72(FP), R5     // R5 = lowerIm pointer
    MOVD twRe_base+96(FP), R6        // R6 = twRe pointer
    MOVD twIm_base+120(FP), R7       // R7 = twIm pointer

    // Process 4 elements per iteration
    LSR $2, R3, R8
    CBZ R8, butterfly_neon_scalar

butterfly_neon_loop4:
    // Load inputs
    VLD1.P 16(R0), [V0.S4]           // V0 = upperRe[0:4]
    VLD1.P 16(R1), [V1.S4]           // V1 = upperIm[0:4]
    VLD1.P 16(R4), [V2.S4]           // V2 = lowerRe[0:4]
    VLD1.P 16(R5), [V3.S4]           // V3 = lowerIm[0:4]
    VLD1.P 16(R6), [V4.S4]           // V4 = twRe[0:4]
    VLD1.P 16(R7), [V5.S4]           // V5 = twIm[0:4]

    // tempRe = lowerRe*twRe - lowerIm*twIm
    WORD $0x6E24DC46                  // FMUL V6.4S, V2.4S, V4.4S  (lowerRe * twRe)
    WORD $0x4EA5CC66                  // FMLS V6.4S, V3.4S, V5.4S  (V6 -= lowerIm * twIm)

    // tempIm = lowerRe*twIm + lowerIm*twRe
    WORD $0x6E25DC47                  // FMUL V7.4S, V2.4S, V5.4S  (lowerRe * twIm)
    WORD $0x4E24CC67                  // FMLA V7.4S, V3.4S, V4.4S  (V7 += lowerIm * twRe)

    // new upperRe = upperRe + tempRe, new lowerRe = upperRe - tempRe
    WORD $0x4E26D408                  // FADD V8.4S, V0.4S, V6.4S  (upperRe + tempRe)
    WORD $0x4EA6D40A                  // FSUB V10.4S, V0.4S, V6.4S (upperRe - tempRe)

    // new upperIm = upperIm + tempIm, new lowerIm = upperIm - tempIm
    WORD $0x4E27D429                  // FADD V9.4S, V1.4S, V7.4S  (upperIm + tempIm)
    WORD $0x4EA7D42B                  // FSUB V11.4S, V1.4S, V7.4S (upperIm - tempIm)

    // Store results (need to rewind pointers since we post-incremented during load)
    SUB $16, R0
    SUB $16, R1
    VST1.P [V8.S4], 16(R0)           // Store new upperRe
    VST1.P [V9.S4], 16(R1)           // Store new upperIm
    VST1.P [V10.S4], 16(R4)          // Store new lowerRe
    VST1.P [V11.S4], 16(R5)          // Store new lowerIm

    SUB $1, R8
    CBNZ R8, butterfly_neon_loop4

butterfly_neon_scalar:
    AND $3, R3
    CBZ R3, butterfly_neon_done

butterfly_neon_scalar_loop:
    // Load scalar values
    FMOVS (R0), F0                   // F0 = upperRe
    FMOVS (R1), F1                   // F1 = upperIm
    FMOVS (R4), F2                   // F2 = lowerRe
    FMOVS (R5), F3                   // F3 = lowerIm
    FMOVS (R6), F4                   // F4 = twRe
    FMOVS (R7), F5                   // F5 = twIm

    // tempRe = lowerRe*twRe - lowerIm*twIm
    FMULS F2, F4, F6                 // F6 = lowerRe * twRe
    FMULS F3, F5, F7                 // F7 = lowerIm * twIm
    FSUBS F7, F6, F6                 // F6 = tempRe (F6 - F7)

    // tempIm = lowerRe*twIm + lowerIm*twRe
    FMULS F2, F5, F7                 // F7 = lowerRe * twIm
    FMULS F3, F4, F8                 // F8 = lowerIm * twRe
    FADDS F7, F8, F7                 // F7 = tempIm

    // Butterfly: upper' = upper + temp, lower' = upper - temp
    FADDS F0, F6, F8                 // F8 = new upperRe
    FSUBS F6, F0, F9                 // F9 = new lowerRe (F0 - F6)
    FADDS F1, F7, F10                // F10 = new upperIm
    FSUBS F7, F1, F11                // F11 = new lowerIm (F1 - F7)

    // Store results
    FMOVS F8, (R0)
    FMOVS F10, (R1)
    FMOVS F9, (R4)
    FMOVS F11, (R5)

    ADD $4, R0
    ADD $4, R1
    ADD $4, R4
    ADD $4, R5
    ADD $4, R6
    ADD $4, R7
    SUB $1, R3
    CBNZ R3, butterfly_neon_scalar_loop

butterfly_neon_done:
    RET
