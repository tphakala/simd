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
TEXT ·dotProductNEON(SB), NOSPLIT, $0-52
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R2
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
TEXT ·addScaledNEON(SB), NOSPLIT, $0-52
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
