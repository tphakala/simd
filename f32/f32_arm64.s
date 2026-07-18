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
// FRINTN Vd.4S, Vn.4S:      0x4E218800 | (Vn << 5) | Vd  (round to nearest, ties to even)
// FMLA Vd.4S, Vn.4S, Vm.4S: 0x4E20CC00 | (Vm << 16) | (Vn << 5) | Vd
// FADDP Vd.4S, Vn.4S, Vm.4S: 0x6E20D400 | (Vm << 16) | (Vn << 5) | Vd
// FADDP Sd, Vn.2S:          0x7E30D800 | (Vn << 5) | Vd
// FCMGT Vd.4S, Vn.4S, Vm.4S: 0x6EA0E400 | (Vm << 16) | (Vn << 5) | Vd  (lane mask Vn > Vm; NaN operand yields 0)

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

// func dotProduct4NEON(results, row0, row1, row2, row3, vec *float32, n int)
// Scores four rows against the same vec, reusing each vec load across the
// group instead of reloading the query per row. Two accumulator banks per row
// (V0-V3 bank a, V4-V7 bank b) hide FMLA latency over an 8-element main loop;
// V16/V17 hold the two query chunks, V18-V21 the four row chunks.
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

    // Process 8 elements per row (2 NEON chunks) per iteration.
    LSR $3, R6, R7             // R7 = n / 8
    CBZ R7, dot4n_rem4_check

dot4n_loop8:
    // chunk a: vec[0:4] -> V16, accumulate into bank a (V0-V3)
    VLD1.P 16(R5), [V16.S4]
    VLD1.P 16(R1), [V18.S4]
    WORD $0x4E30CE40           // FMLA V0.4S, V18.4S, V16.4S
    VLD1.P 16(R2), [V19.S4]
    WORD $0x4E30CE61           // FMLA V1.4S, V19.4S, V16.4S
    VLD1.P 16(R3), [V20.S4]
    WORD $0x4E30CE82           // FMLA V2.4S, V20.4S, V16.4S
    VLD1.P 16(R4), [V21.S4]
    WORD $0x4E30CEA3           // FMLA V3.4S, V21.4S, V16.4S
    // chunk b: vec[4:8] -> V17, accumulate into bank b (V4-V7)
    VLD1.P 16(R5), [V17.S4]
    VLD1.P 16(R1), [V18.S4]
    WORD $0x4E31CE44           // FMLA V4.4S, V18.4S, V17.4S
    VLD1.P 16(R2), [V19.S4]
    WORD $0x4E31CE65           // FMLA V5.4S, V19.4S, V17.4S
    VLD1.P 16(R3), [V20.S4]
    WORD $0x4E31CE86           // FMLA V6.4S, V20.4S, V17.4S
    VLD1.P 16(R4), [V21.S4]
    WORD $0x4E31CEA7           // FMLA V7.4S, V21.4S, V17.4S
    SUB $1, R7
    CBNZ R7, dot4n_loop8

    // Fold bank b into bank a (only reached when the main loop ran).
    WORD $0x4E24D400           // FADD V0.4S, V0.4S, V4.4S
    WORD $0x4E25D421           // FADD V1.4S, V1.4S, V5.4S
    WORD $0x4E26D442           // FADD V2.4S, V2.4S, V6.4S
    WORD $0x4E27D463           // FADD V3.4S, V3.4S, V7.4S

dot4n_rem4_check:
    AND $7, R6, R8
    LSR $2, R8, R9             // R9 = (n & 7) / 4
    CBZ R9, dot4n_reduce

    // One 4-element chunk into bank a.
    VLD1.P 16(R5), [V16.S4]
    VLD1.P 16(R1), [V18.S4]
    WORD $0x4E30CE40           // FMLA V0.4S, V18.4S, V16.4S
    VLD1.P 16(R2), [V19.S4]
    WORD $0x4E30CE61           // FMLA V1.4S, V19.4S, V16.4S
    VLD1.P 16(R3), [V20.S4]
    WORD $0x4E30CE82           // FMLA V2.4S, V20.4S, V16.4S
    VLD1.P 16(R4), [V21.S4]
    WORD $0x4E30CEA3           // FMLA V3.4S, V21.4S, V16.4S

dot4n_reduce:
    // Reduce each bank-a accumulator to a scalar (S0..S3) BEFORE any scalar FMA,
    // since scalar ops zero the upper lanes of the V register.
    WORD $0x6E20D400           // FADDP V0.4S, V0.4S, V0.4S
    WORD $0x7E30D800           // FADDP S0, V0.2S
    WORD $0x6E21D421           // FADDP V1.4S, V1.4S, V1.4S
    WORD $0x7E30D821           // FADDP S1, V1.2S
    WORD $0x6E22D442           // FADDP V2.4S, V2.4S, V2.4S
    WORD $0x7E30D842           // FADDP S2, V2.2S
    WORD $0x6E23D463           // FADDP V3.4S, V3.4S, V3.4S
    WORD $0x7E30D863           // FADDP S3, V3.2S

    AND $3, R6, R8             // R8 = n & 3 (scalar tail count)
    CBZ R8, dot4n_store

dot4n_scalar:
    FMOVS (R5), F18
    FMOVS (R1), F19
    FMADDS F18, F0, F19, F0    // F0 = F19 * F18 + F0
    FMOVS (R2), F19
    FMADDS F18, F1, F19, F1
    FMOVS (R3), F19
    FMADDS F18, F2, F19, F2
    FMOVS (R4), F19
    FMADDS F18, F3, F19, F3
    ADD $4, R5
    ADD $4, R1
    ADD $4, R2
    ADD $4, R3
    ADD $4, R4
    SUB $1, R8
    CBNZ R8, dot4n_scalar

dot4n_store:
    FMOVS F0, (R0)
    FMOVS F1, 4(R0)
    FMOVS F2, 8(R0)
    FMOVS F3, 12(R0)
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
    WORD $0x6EA0F400           // FMINP V0.4S, V0.4S, V0.4S
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
    WORD $0x6EA0F400           // FMINP V0.4S, V0.4S, V0.4S
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
    WORD $0x6E20F400           // FMAXP V0.4S, V0.4S, V0.4S
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
    WORD $0x6E20F400           // FMAXP V0.4S, V0.4S, V0.4S
    WORD $0x7E30F800           // FMAXP S0, V0.2S
    FMOVS F0, ret+24(FP)
    RET

// func maxAbsNEON(a []float32) float32
// max_i |a[i]|. Mirrors maxNEON with FABS folded into each loaded vector and the
// scalar tail (4 x float32 per 128-bit register).
TEXT ·maxAbsNEON(SB), NOSPLIT, $0-28
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1

    VLD1.P 16(R0), [V0.S4]
    WORD $0x4EA0F800           // FABS V0.4S, V0.4S
    SUB $4, R1

    LSR $2, R1, R2
    CBZ R2, maxabs32_scalar

maxabs32_loop4:
    VLD1.P 16(R0), [V1.S4]
    WORD $0x4EA0F821           // FABS V1.4S, V1.4S
    WORD $0x4E21F400           // FMAX V0.4S, V0.4S, V1.4S
    SUB $1, R2
    CBNZ R2, maxabs32_loop4

maxabs32_scalar:
    AND $3, R1
    CBZ R1, maxabs32_reduce

    // Reduce vector FIRST before scalar ops
    WORD $0x6E20F400           // FMAXP V0.4S, V0.4S, V0.4S
    WORD $0x7E30F800           // FMAXP S0, V0.2S

maxabs32_loop1:
    FMOVS (R0), F1
    FABSS F1, F1
    FMAXS F0, F1, F0
    ADD $4, R0
    SUB $1, R1
    CBNZ R1, maxabs32_loop1

    FMOVS F0, ret+24(FP)
    RET

maxabs32_reduce:
    // Horizontal max when no scalar remainder
    WORD $0x6E20F400           // FMAXP V0.4S, V0.4S, V0.4S
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

// func interleave3NEON(dst, s0, s1, s2 []float32, n int)
// Interleaves 3 planar streams (dst[i*3+c] = s_c[i]) with the NEON ST3
// structured store, 4 frames per iteration, then a scalar tail.
TEXT ·interleave3NEON(SB), NOSPLIT, $0-104
    MOVD dst_base+0(FP), R0
    MOVD s0_base+24(FP), R1
    MOVD s1_base+48(FP), R2
    MOVD s2_base+72(FP), R3
    MOVD n+96(FP), R4

    LSR $2, R4, R5             // R5 = n / 4
    CBZ R5, interleave3_neon_tail

interleave3_neon_loop4:
    VLD1.P 16(R1), [V0.S4]     // V0 = s0[i:i+4]
    VLD1.P 16(R2), [V1.S4]     // V1 = s1[i:i+4]
    VLD1.P 16(R3), [V2.S4]     // V2 = s2[i:i+4]
    VST3.P [V0.S4, V1.S4, V2.S4], 48(R0)  // interleaved store of 12 floats
    SUB $1, R5
    CBNZ R5, interleave3_neon_loop4

interleave3_neon_tail:
    AND $3, R4
    CBZ R4, interleave3_neon_done

interleave3_neon_tail1:
    FMOVS (R1), F0
    FMOVS (R2), F1
    FMOVS (R3), F2
    FMOVS F0, (R0)
    FMOVS F1, 4(R0)
    FMOVS F2, 8(R0)
    ADD $4, R1
    ADD $4, R2
    ADD $4, R3
    ADD $12, R0
    SUB $1, R4
    CBNZ R4, interleave3_neon_tail1

interleave3_neon_done:
    RET

// func deinterleave3NEON(d0, d1, d2, src []float32, n int)
// Splits an interleaved 3-stream buffer (d_c[i] = src[i*3+c]) with the NEON LD3
// structured load, 4 frames per iteration, then a scalar tail.
TEXT ·deinterleave3NEON(SB), NOSPLIT, $0-104
    MOVD d0_base+0(FP), R0
    MOVD d1_base+24(FP), R1
    MOVD d2_base+48(FP), R2
    MOVD src_base+72(FP), R3
    MOVD n+96(FP), R4

    LSR $2, R4, R5             // R5 = n / 4
    CBZ R5, deinterleave3_neon_tail

deinterleave3_neon_loop4:
    VLD3.P 48(R3), [V0.S4, V1.S4, V2.S4]  // de-interleave 12 floats
    VST1.P [V0.S4], 16(R0)     // d0[i:i+4]
    VST1.P [V1.S4], 16(R1)     // d1[i:i+4]
    VST1.P [V2.S4], 16(R2)     // d2[i:i+4]
    SUB $1, R5
    CBNZ R5, deinterleave3_neon_loop4

deinterleave3_neon_tail:
    AND $3, R4
    CBZ R4, deinterleave3_neon_done

deinterleave3_neon_tail1:
    FMOVS (R3), F0
    FMOVS 4(R3), F1
    FMOVS 8(R3), F2
    FMOVS F0, (R0)
    FMOVS F1, (R1)
    FMOVS F2, (R2)
    ADD $12, R3
    ADD $4, R0
    ADD $4, R1
    ADD $4, R2
    SUB $1, R4
    CBNZ R4, deinterleave3_neon_tail1

deinterleave3_neon_done:
    RET

// func interleave4NEON(dst, s0, s1, s2, s3 []float32, n int)
// Interleaves 4 planar streams (dst[i*4+c] = s_c[i]) with the NEON ST4
// structured store, 4 frames per iteration, then a scalar tail.
TEXT ·interleave4NEON(SB), NOSPLIT, $0-128
    MOVD dst_base+0(FP), R0
    MOVD s0_base+24(FP), R1
    MOVD s1_base+48(FP), R2
    MOVD s2_base+72(FP), R3
    MOVD s3_base+96(FP), R4
    MOVD n+120(FP), R5

    LSR $2, R5, R6             // R6 = n / 4
    CBZ R6, interleave4_neon_tail

interleave4_neon_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    VLD1.P 16(R3), [V2.S4]
    VLD1.P 16(R4), [V3.S4]
    VST4.P [V0.S4, V1.S4, V2.S4, V3.S4], 64(R0)  // interleaved store of 16 floats
    SUB $1, R6
    CBNZ R6, interleave4_neon_loop4

interleave4_neon_tail:
    AND $3, R5
    CBZ R5, interleave4_neon_done

interleave4_neon_tail1:
    FMOVS (R1), F0
    FMOVS (R2), F1
    FMOVS (R3), F2
    FMOVS (R4), F3
    FMOVS F0, (R0)
    FMOVS F1, 4(R0)
    FMOVS F2, 8(R0)
    FMOVS F3, 12(R0)
    ADD $4, R1
    ADD $4, R2
    ADD $4, R3
    ADD $4, R4
    ADD $16, R0
    SUB $1, R5
    CBNZ R5, interleave4_neon_tail1

interleave4_neon_done:
    RET

// func deinterleave4NEON(d0, d1, d2, d3, src []float32, n int)
// Splits an interleaved 4-stream buffer (d_c[i] = src[i*4+c]) with the NEON LD4
// structured load, 4 frames per iteration, then a scalar tail.
TEXT ·deinterleave4NEON(SB), NOSPLIT, $0-128
    MOVD d0_base+0(FP), R0
    MOVD d1_base+24(FP), R1
    MOVD d2_base+48(FP), R2
    MOVD d3_base+72(FP), R3
    MOVD src_base+96(FP), R4
    MOVD n+120(FP), R5

    LSR $2, R5, R6             // R6 = n / 4
    CBZ R6, deinterleave4_neon_tail

deinterleave4_neon_loop4:
    VLD4.P 64(R4), [V0.S4, V1.S4, V2.S4, V3.S4]  // de-interleave 16 floats
    VST1.P [V0.S4], 16(R0)
    VST1.P [V1.S4], 16(R1)
    VST1.P [V2.S4], 16(R2)
    VST1.P [V3.S4], 16(R3)
    SUB $1, R6
    CBNZ R6, deinterleave4_neon_loop4

deinterleave4_neon_tail:
    AND $3, R5
    CBZ R5, deinterleave4_neon_done

deinterleave4_neon_tail1:
    FMOVS (R4), F0
    FMOVS 4(R4), F1
    FMOVS 8(R4), F2
    FMOVS 12(R4), F3
    FMOVS F0, (R0)
    FMOVS F1, (R1)
    FMOVS F2, (R2)
    FMOVS F3, (R3)
    ADD $16, R4
    ADD $4, R0
    ADD $4, R1
    ADD $4, R2
    ADD $4, R3
    SUB $1, R5
    CBNZ R5, deinterleave4_neon_tail1

deinterleave4_neon_done:
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

// roundNEON: round-half-away-from-zero using FRINTA (round to nearest, ties to
// away), which matches math.Round exactly. The float32 analogue of f64's
// roundNEON.
//
// func roundNEON(dst, src []float32)
TEXT ·roundNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R2
    MOVD src_base+24(FP), R1

    LSR $2, R2, R3
    CBZ R3, round32_scalar

round32_loop4:
    VLD1.P 16(R1), [V0.S4]
    WORD $0x6E218800           // FRINTA V0.4S, V0.4S
    VST1.P [V0.S4], 16(R0)
    SUB $1, R3
    CBNZ R3, round32_loop4

round32_scalar:
    AND $3, R2
    CBZ R2, round32_done

round32_loop1:
    FMOVS (R1), F0
    WORD $0x1E264000           // FRINTA S0, S0
    FMOVS F0, (R0)
    ADD $4, R0
    ADD $4, R1
    SUB $1, R2
    CBNZ R2, round32_loop1

round32_done:
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
    WORD $0x6EA0F800              // FNEG V0.4S, V0.4S

    // Clamp -x to [-20, 20]
    WORD $0x4EBBF400              // FMIN V0.4S, V0.4S, V27.4S  (clamp upper to 20)
    WORD $0x4E3CF400              // FMAX V0.4S, V0.4S, V28.4S  (clamp lower to -20)

    // Range reduction: k = round(-x * log2e), r = -x - k * ln2
    WORD $0x6E34DC01              // FMUL V1.4S, V0.4S, V20.4S   V1 = -x * log2e
    WORD $0x4E218822              // FRINTN V2.4S, V1.4S         V2 = k = round(V1)
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

// func expNEON(dst, src []float32)
// Computes e^x using range reduction and a degree-5 polynomial, the same exp
// core as sigmoidNEON but without the negation and the final 1/(1+exp) wrap.
// Inputs are clamped to [-88, 88] to match the pure-Go fallback: results stay
// finite and large-negative inputs underflow to 0.
TEXT ·expNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1

    // log2(e) = 1.442695041 = 0x3fb8aa3b
    MOVW $0x3fb8aa3b, R10
    VMOV R10, V20.S[0]
    VDUP V20.S[0], V20.S4         // V20 = log2(e)

    // ln(2) = 0.693147181 = 0x3f317218
    MOVW $0x3f317218, R10
    VMOV R10, V21.S[0]
    VDUP V21.S[0], V21.S4         // V21 = ln(2)

    // 1.0
    FMOVS $1.0, F22
    VDUP V22.S[0], V22.S4         // V22 = 1.0

    // c2 = 0.5
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

    // Clamp thresholds: ±88.0 = 0x42b00000 / 0xc2b00000
    MOVW $0x42b00000, R10
    VMOV R10, V27.S[0]
    VDUP V27.S[0], V27.S4         // V27 = 88.0 (clamp_hi)

    MOVW $0xc2b00000, R10
    VMOV R10, V28.S[0]
    VDUP V28.S[0], V28.S4         // V28 = -88.0 (clamp_lo)

    // Process 4 elements per iteration
    LSR $2, R3, R4
    CBZ R4, exp32_neon_scalar

exp32_neon_loop4:
    VLD1.P 16(R1), [V0.S4]        // V0 = x

    // Clamp x to [-88, 88]
    WORD $0x4EBBF400              // FMIN V0.4S, V0.4S, V27.4S
    WORD $0x4E3CF400              // FMAX V0.4S, V0.4S, V28.4S

    // Range reduction: k = round(x * log2e), r = x - k * ln2
    WORD $0x6E34DC01              // FMUL V1.4S, V0.4S, V20.4S   V1 = x * log2e
    WORD $0x4E218822              // FRINTN V2.4S, V1.4S         V2 = k = round(V1)
    WORD $0x6E35DC44              // FMUL V4.4S, V2.4S, V21.4S   V4 = k * ln2
    WORD $0x4EA4D403              // FSUB V3.4S, V0.4S, V4.4S    V3 = r = x - k * ln2

    // Polynomial: exp(r) ~= 1 + r*(1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
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

    // Reconstruct: exp(x) = exp(r) * 2^k
    WORD $0x4EA1B841              // FCVTZS V1.4S, V2.4S         V1 = int(k)
    WORD $0x4F375421              // SHL V1.4S, V1.4S, #23       V1 = k << 23
    WORD $0x4EB68421              // ADD V1.4S, V1.4S, V22.4S    V1 = 2^k (add 1.0's bits)
    WORD $0x6E21DC84              // FMUL V4.4S, V4.4S, V1.4S    V4 = exp(x)

    VST1.P [V4.S4], 16(R0)        // store result

    SUB $1, R4
    CBNZ R4, exp32_neon_loop4

exp32_neon_scalar:
    AND $3, R3
    CBZ R3, exp32_neon_done

    // Hoist loop-invariant constants out of the remainder loop. The clamp
    // bounds live in F6/F7 because F1/F2 are reused as temporaries below.
    MOVW $0x42b00000, R10
    FMOVS R10, F6                 // F6 = 88.0 (clamp_hi)
    MOVW $0xc2b00000, R10
    FMOVS R10, F7                 // F7 = -88.0 (clamp_lo)
    MOVW $0x3fb8aa3b, R10
    FMOVS R10, F8                 // F8 = log2(e)
    MOVW $0x3f317218, R10
    FMOVS R10, F9                 // F9 = ln(2)
    FMOVS $1.0, F10               // c1 = 1.0
    FMOVS $0.5, F11               // c2 = 0.5
    MOVW $0x3e2aaaab, R10
    FMOVS R10, F12                // c3 = 1/6
    MOVW $0x3d2aaaab, R10
    FMOVS R10, F13                // c4 = 1/24
    MOVW $0x3c088889, R10
    FMOVS R10, F14                // c5 = 1/120
    MOVW $0x3f800000, R11         // R11 = 1.0's bits (exponent bias)

exp32_neon_scalar_loop:
    FMOVS (R1), F0                // F0 = x

    // Clamp to [-88, 88]
    FMINS F6, F0, F0
    FMAXS F7, F0, F0

    // Range reduction
    FMULS F8, F0, F1              // F1 = x * log2e
    FRINTNS F1, F2                // F2 = k = round(F1)
    FMULS F9, F2, F3              // F3 = k * ln2
    FSUBS F3, F0, F0              // F0 = r = x - k * ln2

    // Horner's method
    FMULS F0, F14, F4             // F4 = r * c5
    FADDS F13, F4, F4
    FMULS F0, F4, F4
    FADDS F12, F4, F4
    FMULS F0, F4, F4
    FADDS F11, F4, F4
    FMULS F0, F4, F4
    FADDS F10, F4, F4
    FMULS F0, F4, F4
    FADDS F10, F4, F4             // F4 = exp(r)

    // Reconstruct 2^k
    FCVTZSS F2, R10               // R10 = int(k)
    LSL $23, R10, R10            // R10 = k << 23
    ADD R11, R10, R10            // R10 = 2^k bits
    FMOVS R10, F5
    FMULS F5, F4, F4             // F4 = exp(x)
    FMOVS F4, (R0)              // store result

    ADD $4, R0
    ADD $4, R1
    SUB $1, R3
    CBNZ R3, exp32_neon_scalar_loop

exp32_neon_done:
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
    WORD $0x6EA0F800                  // FNEG V0.4S, V0.4S           V0 = -x
    WORD $0x6E33DC00                  // FMUL V0.4S, V0.4S, V19.4S   V0 = -2x

    // Clamp -2x to [-20, 20]
    WORD $0x4EBBF400                  // FMIN V0.4S, V0.4S, V27.4S   clamp upper to 20
    WORD $0x4E3CF400                  // FMAX V0.4S, V0.4S, V28.4S   clamp lower to -20

    // Range reduction: k = round(-2x * log2e), r = -2x - k * ln2
    WORD $0x6E34DC01                  // FMUL V1.4S, V0.4S, V20.4S   V1 = -2x * log2e
    WORD $0x4E218822                  // FRINTN V2.4S, V1.4S         V2 = k = round(V1)
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

// func int16ToFloat32ScaleNEON(dst []float32, src []int16, scale float32)
// Converts int16 samples to float32 and multiplies by scale in one pass.
// dst[i] = float32(src[i]) * scale
// Optimized for 16-bit PCM audio (e.g., scale = 1.0/32768 to normalize to [-1, 1)).
//
// NEON opcodes (hand-encoded; the Go assembler lacks these vector forms):
// SXTL Vd.4S, Vn.4H:        widen 4 signed int16 to 4 int32
// SCVTF Vd.4S, Vn.4S:       signed int32 to float32
// FMUL Vd.4S, Vn.4S, Vm.4S: multiply
// DUP Vd.4S, Vn.S[0]:       broadcast lane 0
//
// Frame: dst(24) + src(24) + scale(4) = 52 bytes
TEXT ·int16ToFloat32ScaleNEON(SB), NOSPLIT, $0-52
    MOVD dst_base+0(FP), R0        // R0 = dst pointer (float32 out)
    MOVD dst_len+8(FP), R3         // R3 = length (len(dst) == len(src))
    MOVD src_base+24(FP), R1       // R1 = src pointer (int16 in)
    FMOVS scale+48(FP), F2         // F2 = scale

    // Broadcast scale to all 4 lanes
    WORD $0x4E040442               // DUP V2.4S, V2.S[0]

    // Process 4 int16 elements per iteration
    LSR $2, R3, R4                 // R4 = len / 4
    CBZ R4, i16tof32_neon_scalar

i16tof32_neon_loop4:
    VLD1.P 8(R1), [V0.H4]          // V0 = 4 x int16 (low 4 halfwords)
    WORD $0x0F10A401               // SXTL V1.4S, V0.4H
    WORD $0x4E21D821               // SCVTF V1.4S, V1.4S
    WORD $0x6E22DC21               // FMUL V1.4S, V1.4S, V2.4S
    VST1.P [V1.S4], 16(R0)         // store 4 x float32
    SUB $1, R4
    CBNZ R4, i16tof32_neon_loop4

i16tof32_neon_scalar:
    AND $3, R3                     // remainder = len % 4
    CBZ R3, i16tof32_neon_done

i16tof32_neon_scalar_loop:
    MOVH (R1), R5                  // load int16, sign-extended
    SCVTFWS R5, F0                 // convert int32 to float32
    FMULS F2, F0, F0               // F0 = F0 * scale
    FMOVS F0, (R0)                 // store float32
    ADD $4, R0
    ADD $2, R1
    SUB $1, R3
    CBNZ R3, i16tof32_neon_scalar_loop

i16tof32_neon_done:
    RET

// func float32ToInt16ScaleNEON(dst []int16, src []float32, scale float32)
// Scales float32 samples and converts to int16 PCM in one pass.
// dst[i] = clamp(roundTiesToEven(src[i]*scale), -32768, 32767), NaN -> 0.
// The hardware defines the semantics: FCVTNS rounds to nearest-even and
// saturates to int32 (NaN -> 0, +-Inf -> int32 min/max); SQXTN then saturates
// int32 to int16, yielding +Inf -> 32767, -Inf -> -32768.
//
// NEON opcodes (hand-encoded; the Go assembler lacks these vector forms):
// FMUL Vd.4S, Vn.4S, Vm.4S: multiply
// FCVTNS Vd.4S, Vn.4S:      float32 to signed int32, round to nearest-even, saturating
// SQXTN Vd.4H, Vn.4S:       signed saturating narrow int32 to int16
// DUP Vd.4S, Vn.S[0]:       broadcast lane 0
//
// The 1-3 element tail reprocesses the final aligned block of 4 (an overlapping
// store of identical values) so every element goes through the same vector path;
// the dispatcher guarantees len >= 4.
//
// Frame: dst(24) + src(24) + scale(4) = 52 bytes
TEXT ·float32ToInt16ScaleNEON(SB), NOSPLIT, $0-52
    MOVD dst_base+0(FP), R0        // R0 = dst pointer (int16 out)
    MOVD dst_len+8(FP), R3         // R3 = length (len(dst) == len(src))
    MOVD src_base+24(FP), R1       // R1 = src pointer (float32 in)
    FMOVS scale+48(FP), F2         // F2 = scale

    // Broadcast scale to all 4 lanes
    WORD $0x4E040442               // DUP V2.4S, V2.S[0]

    // Process 4 float32 elements per iteration
    LSR $2, R3, R4                 // R4 = len / 4
    CBZ R4, f32toi16_neon_tail

f32toi16_neon_loop4:
    VLD1.P 16(R1), [V0.S4]         // V0 = 4 x float32
    WORD $0x6E22DC01               // FMUL V1.4S, V0.4S, V2.4S
    WORD $0x4E21A821               // FCVTNS V1.4S, V1.4S
    WORD $0x0E614821               // SQXTN V1.4H, V1.4S
    VST1.P [V1.H4], 8(R0)          // store 4 x int16 (8 bytes)
    SUB $1, R4
    CBNZ R4, f32toi16_neon_loop4

f32toi16_neon_tail:
    AND $3, R3, R5                 // R5 = len % 4
    CBZ R5, f32toi16_neon_done

    // Back up to the final aligned block of 4 and reprocess it (overlap).
    MOVD $4, R6
    SUB R5, R6, R6                 // R6 = 4 - (len % 4)  (1..3)
    LSL $2, R6, R7                 // R7 = (4 - rem) * 4 src bytes
    SUB R7, R1, R1                 // R1 -= that many bytes
    LSL $1, R6, R7                 // R7 = (4 - rem) * 2 dst bytes
    SUB R7, R0, R0                 // R0 -= that many bytes
    VLD1 (R1), [V0.S4]            // V0 = final 4 x float32
    WORD $0x6E22DC01               // FMUL V1.4S, V0.4S, V2.4S
    WORD $0x4E21A821               // FCVTNS V1.4S, V1.4S
    WORD $0x0E614821               // SQXTN V1.4H, V1.4S
    VST1 [V1.H4], (R0)            // store final 4 x int16

f32toi16_neon_done:
    RET

// func float32ToInt32ScaleClampNEON(dst []int32, src []float32, scale, offset, minV, maxV float32)
// dst[i] = int32(clamp(src[i]*scale + offset, minV, maxV)), truncated toward zero.
//
// FMUL then FADD are deliberately SEPARATE (not FMLA): the product rounds to
// float32 before offset is added, matching the two-rounding Go reference and the
// AVX path bit-for-bit. NaN -> 0 falls out for free here: ARM FMAX/FMIN (not the
// FMAXNM/FMINNM forms) PROPAGATE a NaN operand, so a NaN survives the clamp, and
// FCVTZS maps NaN to 0. So no explicit NaN masking is needed (the AVX path needs
// it only because x86 MINPS/MAXPS replace a NaN with the bound). The clamp bounds
// +/-Inf to maxV/minV. FCVTZS truncates toward zero to match Go's int32(v). The
// vector float ops (.4S) have no Go-assembler spelling, so FMUL, FADD, FMAX, FMIN
// and FCVTZS are WORD-encoded (verified with clang -c -arch arm64 + objdump; see
// docs/assembly-encoding.md); VDUP and VLD1/VST1 are native. The 4-wide tail
// reprocesses the final full block with overlap.
TEXT ·float32ToInt32ScaleClampNEON(SB), NOSPLIT, $0-64
    MOVD dst_base+0(FP), R0        // R0 = dst pointer (int32 out)
    MOVD dst_len+8(FP), R3         // R3 = length (len(dst) == len(src))
    MOVD src_base+24(FP), R1       // R1 = src pointer (float32 in)
    FMOVS scale+48(FP), F2         // F2 = scale
    FMOVS offset+52(FP), F3        // F3 = offset
    FMOVS minV+56(FP), F4          // F4 = minV
    FMOVS maxV+60(FP), F5          // F5 = maxV

    // Broadcast each scalar to all 4 lanes (DUP element form, native VDUP).
    VDUP V2.S[0], V2.S4            // scale  x4
    VDUP V3.S[0], V3.S4            // offset x4
    VDUP V4.S[0], V4.S4            // minV   x4
    VDUP V5.S[0], V5.S4            // maxV   x4

    LSR $2, R3, R4                 // R4 = len / 4
    CBZ R4, f32toi32_neon_tail

f32toi32_neon_loop4:
    VLD1.P 16(R1), [V0.S4]         // V0 = 4 x float32
    WORD $0x6E22DC00              // FMUL V0.4S, V0.4S, V2.4S   (v = src * scale)
    WORD $0x4E23D400             // FADD V0.4S, V0.4S, V3.4S   (v += offset; separate, not FMLA)
    WORD $0x4E24F400            // FMAX V0.4S, V0.4S, V4.4S   (>= minV; -Inf -> minV, NaN propagates)
    WORD $0x4EA5F400           // FMIN V0.4S, V0.4S, V5.4S   (<= maxV; +Inf -> maxV, NaN propagates)
    WORD $0x4EA1B800          // FCVTZS V0.4S, V0.4S        (truncate toward zero; NaN -> 0)
    VST1.P [V0.S4], 16(R0)    // store 4 x int32 (16 bytes)
    SUB $1, R4
    CBNZ R4, f32toi32_neon_loop4

f32toi32_neon_tail:
    AND $3, R3, R5                 // R5 = len % 4
    CBZ R5, f32toi32_neon_done

    // Back up to the final aligned block of 4 and reprocess it (overlap). src and
    // dst are both 4-byte elements, so both back up by (4 - rem) * 4 bytes.
    MOVD $4, R6
    SUB R5, R6, R6                 // R6 = 4 - (len % 4)  (1..3)
    LSL $2, R6, R7                 // R7 = (4 - rem) * 4 bytes
    SUB R7, R1, R1
    SUB R7, R0, R0
    VLD1 (R1), [V0.S4]             // final 4 x float32
    WORD $0x6E22DC00               // FMUL V0.4S, V0.4S, V2.4S
    WORD $0x4E23D400               // FADD V0.4S, V0.4S, V3.4S
    WORD $0x4E24F400               // FMAX V0.4S, V0.4S, V4.4S
    WORD $0x4EA5F400               // FMIN V0.4S, V0.4S, V5.4S
    WORD $0x4EA1B800               // FCVTZS V0.4S, V0.4S
    VST1 [V0.S4], (R0)             // store final 4 x int32

f32toi32_neon_done:
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
    SUB $16, R4
    SUB $16, R5
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

// ============================================================================
// REAL FFT UNPACK - UNPACKING STEP FOR REAL-VALUED FFT
// ============================================================================

// func realFFTUnpackNEON(outRe, outIm, zRe, zIm, twRe, twIm []float32, n int)
// Performs the unpacking step of real FFT:
//   For k in [1, n-1]:
//     conj_z = conj(Z[n-k])
//     even = 0.5 * (Z[k] + conj_z)
//     diff = Z[k] - conj_z
//     odd = W[k] * (-0.5i) * diff
//     X[k] = even + odd
// Frame: 6 slices × 24 bytes + 1 int × 8 bytes = 152 bytes
TEXT ·realFFTUnpackNEON(SB), NOSPLIT, $0-152
    // Load parameters
    MOVD outRe_base+0(FP), R0        // R0 = outRe pointer
    MOVD outIm_base+24(FP), R1       // R1 = outIm pointer
    MOVD zRe_base+48(FP), R2         // R2 = zRe pointer (forward)
    MOVD zIm_base+72(FP), R3         // R3 = zIm pointer (forward)
    MOVD twRe_base+96(FP), R4        // R4 = twRe pointer
    MOVD twIm_base+120(FP), R5       // R5 = twIm pointer
    MOVD n+144(FP), R6               // R6 = n

    // Calculate number of iterations: (n-1) / 4
    SUB $1, R6, R7                   // R7 = n - 1
    MOVD R7, R14                     // R14 = n - 1 (save for remainder)
    LSR $2, R7                       // R7 = (n-1) / 4 = number of SIMD iterations
    CBZ R7, realfft_neon_remainder   // Skip SIMD loop if < 4 elements

    // Set up reverse pointers: zRe[n-4], zIm[n-4]
    SUB $4, R6, R8                   // R8 = n - 4
    LSL $2, R8                       // R8 = (n-4) * 4 = byte offset
    MOVD zRe_base+48(FP), R9
    ADD R8, R9                       // R9 = &zRe[n-4]
    MOVD zIm_base+72(FP), R10
    ADD R8, R10                      // R10 = &zIm[n-4]

    // Offset forward pointers to start at index 1
    ADD $4, R2                       // R2 = &zRe[1]
    ADD $4, R3                       // R3 = &zIm[1]
    ADD $4, R0                       // R0 = &outRe[1]
    ADD $4, R1                       // R1 = &outIm[1]

    // Load 0.5 constant into V30
    MOVW $0x3F000000, R11            // 0.5 in IEEE 754
    FMOVS R11, F30
    WORD $0x4E0407DE                 // DUP V30.4S, V30.S[0]

realfft_neon_loop4:
    // Load forward Z[k:k+4]
    VLD1.P 16(R2), [V0.S4]           // V0 = zRe[k:k+4] (forward)
    VLD1.P 16(R3), [V1.S4]           // V1 = zIm[k:k+4] (forward)

    // Load reverse Z[n-k-3:n-k+1] and reverse the order
    VLD1 (R9), [V2.S4]               // V2 = zRe[n-k-3:n-k+1] (to be reversed)
    VLD1 (R10), [V3.S4]              // V3 = zIm[n-k-3:n-k+1] (to be reversed)

    // Reverse V2 and V3: [0,1,2,3] -> [3,2,1,0]
    // Use REV64 + EXT to reverse 4 elements
    WORD $0x4EA00842                 // REV64 V2.4S, V2.4S  (swap pairs: [1,0,3,2])
    WORD $0x6E024042                 // EXT V2.16B, V2.16B, V2.16B, #8  (swap halves: [3,2,1,0])
    WORD $0x4EA00863                 // REV64 V3.4S, V3.4S
    WORD $0x6E034063                 // EXT V3.16B, V3.16B, V3.16B, #8

    // For conjugate: znkIm = -zIm[n-k]
    WORD $0x6EA0F863                 // FNEG V3.4S, V3.4S  (negate for conjugate)

    // Compute even = 0.5 * (Z[k] + conj(Z[n-k]))
    // evenRe = 0.5 * (zkRe + znkRe)
    WORD $0x4E22D404                 // FADD V4.4S, V0.4S, V2.4S  (zkRe + znkRe)
    WORD $0x6E3EDC84                 // FMUL V4.4S, V4.4S, V30.4S  (evenRe)

    // evenIm = 0.5 * (zkIm + znkIm)
    WORD $0x4E23D425                 // FADD V5.4S, V1.4S, V3.4S  (zkIm + znkIm)
    WORD $0x6E3EDCA5                 // FMUL V5.4S, V5.4S, V30.4S  (evenIm)

    // Compute diff = Z[k] - conj(Z[n-k])
    // diffRe = zkRe - znkRe
    WORD $0x4EA2D406                 // FSUB V6.4S, V0.4S, V2.4S  (diffRe)
    // diffIm = zkIm - znkIm
    WORD $0x4EA3D427                 // FSUB V7.4S, V1.4S, V3.4S  (diffIm)

    // Load twiddles W[k] (not post-increment, need them for computation first)
    VLD1.P 16(R4), [V8.S4]           // V8 = twRe (wr)
    VLD1.P 16(R5), [V9.S4]           // V9 = twIm (wi)

    // Compute odd = W[k] * (-0.5i) * diff
    // oddRe = 0.5 * (wr*diffIm + wi*diffRe)
    WORD $0x6E27DD0A                 // FMUL V10.4S, V8.4S, V7.4S   (wr * diffIm)
    WORD $0x4E26CD2A                 // FMLA V10.4S, V9.4S, V6.4S   (V10 += wi * diffRe)
    WORD $0x6E3EDD4A                 // FMUL V10.4S, V10.4S, V30.4S (oddRe)

    // oddIm = 0.5 * (wi*diffIm - wr*diffRe)
    WORD $0x6E27DD2B                 // FMUL V11.4S, V9.4S, V7.4S   (wi * diffIm)
    WORD $0x4EA6CD0B                 // FMLS V11.4S, V8.4S, V6.4S   (V11 -= wr * diffRe)
    WORD $0x6E3EDD6B                 // FMUL V11.4S, V11.4S, V30.4S (oddIm)

    // Compute output X[k] = even + odd
    WORD $0x4E2AD480                 // FADD V0.4S, V4.4S, V10.4S  (outRe)
    WORD $0x4E2BD4A1                 // FADD V1.4S, V5.4S, V11.4S  (outIm)

    // Store results
    VST1.P [V0.S4], 16(R0)           // Store outRe[k:k+4]
    VST1.P [V1.S4], 16(R1)           // Store outIm[k:k+4]

    // Move reverse pointers backward
    SUB $16, R9                      // reverse zRe -= 4
    SUB $16, R10                     // reverse zIm -= 4

    SUB $1, R7
    CBNZ R7, realfft_neon_loop4

realfft_neon_remainder:
    // Handle remaining elements (n-1) % 4
    AND $3, R14
    CBZ R14, realfft_neon_done

    // Reload base pointers for remainder
    MOVD outRe_base+0(FP), R0
    MOVD outIm_base+24(FP), R1
    MOVD zRe_base+48(FP), R2
    MOVD zIm_base+72(FP), R3
    MOVD twRe_base+96(FP), R4
    MOVD twIm_base+120(FP), R5
    MOVD n+144(FP), R6

    // Calculate starting k for remainder: 1 + 4 * num_full_iterations
    SUB $1, R6, R7                   // R7 = n - 1
    LSR $2, R7                       // R7 = num_full_iterations
    LSL $2, R7                       // R7 = 4 * num_full_iterations
    ADD $1, R7                       // R7 = starting k

    // Offset pointers to starting k
    LSL $2, R7, R8                   // R8 = k * 4 bytes
    ADD R8, R0                       // R0 = &outRe[k]
    ADD R8, R1                       // R1 = &outIm[k]
    ADD R8, R2                       // R2 = &zRe[k]
    ADD R8, R3                       // R3 = &zIm[k]

    // Twiddle offset is (k-1)
    SUB $1, R7
    LSL $2, R7, R8
    ADD R8, R4                       // R4 = &twRe[k-1]
    ADD R8, R5                       // R5 = &twIm[k-1]
    ADD $1, R7                       // Restore R7 = k

realfft_neon_scalar:
    // Calculate mirror index: nk = n - k
    SUB R7, R6, R8                   // R8 = n - k = nk

    // Load Z[k]
    FMOVS (R2), F0                   // F0 = zRe[k]
    FMOVS (R3), F1                   // F1 = zIm[k]

    // Load conj(Z[n-k])
    MOVD zRe_base+48(FP), R9
    LSL $2, R8, R10
    ADD R10, R9
    FMOVS (R9), F2                   // F2 = zRe[nk]

    MOVD zIm_base+72(FP), R9
    ADD R10, R9
    FMOVS (R9), F3                   // F3 = zIm[nk]

    // Negate F3 for conjugate: znkIm = -zIm[nk]
    FNEGS F3, F3                     // F3 = -zIm[nk] = znkIm

    // Load 0.5 constant
    MOVW $0x3F000000, R11
    FMOVS R11, F13                   // F13 = 0.5

    // evenRe = 0.5 * (zkRe + znkRe)
    FADDS F0, F2, F4                 // F4 = zkRe + znkRe
    FMULS F4, F13, F4                // F4 = evenRe

    // evenIm = 0.5 * (zkIm + znkIm)
    FADDS F1, F3, F5                 // F5 = zkIm + znkIm
    FMULS F5, F13, F5                // F5 = evenIm

    // diffRe = zkRe - znkRe
    FSUBS F2, F0, F6                 // F6 = diffRe

    // diffIm = zkIm - znkIm
    FSUBS F3, F1, F7                 // F7 = diffIm

    // Load twiddles
    FMOVS (R4), F8                   // F8 = wr
    FMOVS (R5), F9                   // F9 = wi

    // oddRe = 0.5 * (wr*diffIm + wi*diffRe)
    FMULS F8, F7, F10                // F10 = wr * diffIm
    FMULS F9, F6, F11                // F11 = wi * diffRe
    FADDS F10, F11, F10              // F10 = wr*diffIm + wi*diffRe
    FMULS F10, F13, F10              // F10 = oddRe

    // oddIm = 0.5 * (wi*diffIm - wr*diffRe)
    FMULS F9, F7, F11                // F11 = wi * diffIm
    FMULS F8, F6, F12                // F12 = wr * diffRe
    FSUBS F12, F11, F11              // F11 = wi*diffIm - wr*diffRe
    FMULS F11, F13, F11              // F11 = oddIm

    // output = even + odd
    FADDS F4, F10, F0                // F0 = outRe
    FADDS F5, F11, F1                // F1 = outIm

    // Store
    FMOVS F0, (R0)
    FMOVS F1, (R1)

    // Advance pointers
    ADD $4, R2
    ADD $4, R3
    ADD $4, R4
    ADD $4, R5
    ADD $4, R0
    ADD $4, R1
    ADD $1, R7                       // k++

    SUB $1, R14
    CBNZ R14, realfft_neon_scalar

realfft_neon_done:
    RET

// ============================================================================
// REVERSE - REVERSE SLICE ELEMENTS
// ============================================================================

// func reverseNEON(dst, src []float32)
// Reverses elements: dst[i] = src[len-1-i]
// Frame: dst(24) + src(24) = 48 bytes
TEXT ·reverseNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0        // R0 = dst pointer
    MOVD dst_len+8(FP), R1         // R1 = length
    MOVD src_base+24(FP), R2       // R2 = src pointer

    // Check for in-place reversal
    CMP R0, R2
    BEQ reverse_neon_inplace

    // Calculate src end pointer: R2 + (n-4)*4 (points to last full block)
    SUB $4, R1, R3                 // R3 = n - 4
    LSL $2, R3                     // R3 = (n-4) * 4
    ADD R2, R3                     // R3 = &src[n-4]

    // Process 4 elements per iteration
    LSR $2, R1, R4                 // R4 = n / 4
    CBZ R4, reverse_neon_remainder

reverse_neon_loop4:
    // Load 4 elements from reverse position
    VLD1 (R3), [V0.S4]             // V0 = src[n-4:n]

    // Reverse order within 4S register:
    // Step 1: REV64 reverses pairs within 64-bit elements: [0,1,2,3] -> [1,0,3,2]
    WORD $0x4EA00800               // REV64 V0.4S, V0.4S

    // Step 2: EXT swaps high/low 64-bits: [1,0,3,2] -> [3,2,1,0]
    WORD $0x6E004000               // EXT V0.16B, V0.16B, V0.16B, #8

    // Store to forward position
    VST1.P [V0.S4], 16(R0)         // dst += 4

    SUB $16, R3                    // src_rev -= 4
    SUB $1, R4
    CBNZ R4, reverse_neon_loop4

reverse_neon_remainder:
    AND $3, R1, R4                 // R4 = remainder count
    CBZ R4, reverse_neon_done

    // Handle remaining elements
    MOVD dst_len+8(FP), R5         // R5 = original length
    LSR $2, R5                     // R5 = n / 4
    LSL $2, R5                     // R5 = processed count (4 * (n/4))
    MOVD dst_len+8(FP), R6         // R6 = n
    SUB $1, R6                     // R6 = n - 1

reverse_neon_scalar:
    // Calculate src index: n - 1 - current_dst_idx
    SUB R5, R6, R7                 // R7 = n - 1 - processed
    LSL $2, R7                     // R7 = byte offset
    ADD R2, R7                     // R7 = &src[n-1-i]
    FMOVS (R7), F0                 // F0 = src[n-1-i]
    FMOVS F0, (R0)
    ADD $4, R0                     // dst++
    ADD $1, R5                     // processed++
    SUB $1, R4
    CBNZ R4, reverse_neon_scalar
    B reverse_neon_done

reverse_neon_inplace:
    // In-place reversal: swap from both ends toward middle
    SUB $1, R1, R3
    LSL $2, R3
    ADD R2, R3                     // R3 = &src[n-1]

    LSR $1, R1, R4                 // R4 = n / 2 swaps needed
    CBZ R4, reverse_neon_done

reverse_neon_inplace_loop:
    FMOVS (R2), F0                 // F0 = front element
    FMOVS (R3), F1                 // F1 = back element
    FMOVS F1, (R2)                 // store back to front
    FMOVS F0, (R3)                 // store front to back
    ADD $4, R2
    SUB $4, R3
    SUB $1, R4
    CBNZ R4, reverse_neon_inplace_loop

reverse_neon_done:
    RET

// ============================================================================
// ADD-SUB - FUSED SUM AND DIFFERENCE
// ============================================================================

// func addSubNEON(sumDst, diffDst, a, b []float32)
// Computes element-wise sum and difference:
//   sumDst[i] = a[i] + b[i]
//   diffDst[i] = a[i] - b[i]
// Frame: sumDst(24) + diffDst(24) + a(24) + b(24) = 96 bytes
TEXT ·addSubNEON(SB), NOSPLIT, $0-96
    MOVD sumDst_base+0(FP), R0     // R0 = sumDst pointer
    MOVD sumDst_len+8(FP), R1      // R1 = length
    MOVD diffDst_base+24(FP), R2   // R2 = diffDst pointer
    MOVD a_base+48(FP), R3         // R3 = a pointer
    MOVD b_base+72(FP), R4         // R4 = b pointer

    // Process 4 elements per iteration
    LSR $2, R1, R5                 // R5 = n / 4
    CBZ R5, addsub_neon_remainder

addsub_neon_loop4:
    // Load inputs
    VLD1.P 16(R3), [V0.S4]         // V0 = a[0:4]
    VLD1.P 16(R4), [V1.S4]         // V1 = b[0:4]

    // Compute sum and diff
    WORD $0x4E21D402               // FADD V2.4S, V0.4S, V1.4S  (sum)
    WORD $0x4EA1D403               // FSUB V3.4S, V0.4S, V1.4S  (diff)

    // Store results
    VST1.P [V2.S4], 16(R0)         // sumDst
    VST1.P [V3.S4], 16(R2)         // diffDst

    SUB $1, R5
    CBNZ R5, addsub_neon_loop4

addsub_neon_remainder:
    AND $3, R1, R5                 // R5 = remainder count
    CBZ R5, addsub_neon_done

addsub_neon_scalar:
    FMOVS (R3), F0                 // F0 = a
    FMOVS (R4), F1                 // F1 = b
    FADDS F0, F1, F2               // F2 = a + b
    FSUBS F1, F0, F3               // F3 = a - b
    FMOVS F2, (R0)
    FMOVS F3, (R2)

    ADD $4, R3
    ADD $4, R4
    ADD $4, R0
    ADD $4, R2
    SUB $1, R5
    CBNZ R5, addsub_neon_scalar

addsub_neon_done:
    RET

// func convolveDecimateNEON(dst, signal, kernel []float32, factor, phase int)
//
// Fused decimating valid convolution. For each output it computes the dot
// product of signal[pos:pos+kLen] with kernel, then advances pos by factor.
// The inner dot replicates dotProductNEON exactly (dual accumulators, 8/4/scalar
// reduction) so results are bit-identical to a per-window DotProductUnsafe when
// the kernel is long enough to take the NEON path (the Go dispatcher only calls
// this for len(kernel) >= 4). Outer state lives in R9-R15; the inner dot uses
// R0-R4 and V0-V5.
TEXT ·convolveDecimateNEON(SB), NOSPLIT, $0-88
    MOVD dst_base+0(FP), R9        // output pointer
    MOVD dst_len+8(FP), R10        // n outputs
    MOVD signal_base+24(FP), R11   // signal base
    MOVD kernel_base+48(FP), R12   // kernel base
    MOVD kernel_len+56(FP), R13    // kLen
    MOVD factor+72(FP), R14        // factor (elements)
    MOVD phase+80(FP), R15         // pos (elements)

    CBZ R10, cd_neon_done

cd_neon_outer:
    ADD R15<<2, R11, R0            // R0 = &signal[pos]
    MOVD R12, R1                   // R1 = &kernel[0]
    MOVD R13, R2                   // R2 = kLen (dot length)

    VEOR V0.B16, V0.B16, V0.B16
    VEOR V1.B16, V1.B16, V1.B16

    LSR $3, R2, R3                 // kLen / 8
    CBZ R3, cd_neon_rem4

cd_neon_loop8:
    VLD1.P 16(R0), [V2.S4]
    VLD1.P 16(R0), [V3.S4]
    VLD1.P 16(R1), [V4.S4]
    VLD1.P 16(R1), [V5.S4]
    WORD $0x4E24CC40              // FMLA V0.4S, V2.4S, V4.4S
    WORD $0x4E25CC61              // FMLA V1.4S, V3.4S, V5.4S
    SUB $1, R3
    CBNZ R3, cd_neon_loop8

    WORD $0x4E21D400             // FADD V0.4S, V0.4S, V1.4S

cd_neon_rem4:
    AND $7, R2, R3
    LSR $2, R3, R4
    CBZ R4, cd_neon_rem

    VLD1.P 16(R0), [V2.S4]
    VLD1.P 16(R1), [V4.S4]
    WORD $0x4E24CC40              // FMLA V0.4S, V2.4S, V4.4S

cd_neon_rem:
    AND $3, R3, R4
    CBZ R4, cd_neon_reduce

    // Reduce vector FIRST before scalar ops (scalar ops zero upper V bits).
    WORD $0x6E20D400             // FADDP V0.4S, V0.4S, V0.4S
    WORD $0x7E30D800             // FADDP S0, V0.2S

cd_neon_scalar:
    FMOVS (R0), F2
    FMOVS (R1), F4
    FMADDS F4, F0, F2, F0         // F0 = F2 * F4 + F0
    ADD $4, R0
    ADD $4, R1
    SUB $1, R4
    CBNZ R4, cd_neon_scalar

    B cd_neon_store

cd_neon_reduce:
    WORD $0x6E20D400             // FADDP V0.4S, V0.4S, V0.4S
    WORD $0x7E30D800             // FADDP S0, V0.2S

cd_neon_store:
    FMOVS F0, (R9)
    ADD $4, R9
    ADD R14, R15                  // pos += factor
    SUB $1, R10
    CBNZ R10, cd_neon_outer

cd_neon_done:
    RET

// func convolveValidMaxAbsNEON(signal, kernel []float32) float32
//
// Fused valid convolution + abs-max: returns max_i |dot(signal[i:i+kLen], kernel)|
// over the n = len(signal)-kLen+1 windows. The inner dot replicates
// convolveDecimateNEON (two FMLA accumulators, 8/4/scalar reduction, FADDP), so
// each window is bit-identical to dotProductNEON / ConvolveValid; the per-window
// store is replaced by FABS into a running max (F6). Caller guarantees kLen >= 4
// and len(signal) >= kLen (kLen < 4 takes the Go fallback), so n >= 1.
TEXT ·convolveValidMaxAbsNEON(SB), NOSPLIT, $0-52
    MOVD signal_base+0(FP), R11    // R11 = &signal[pos], advances 4 bytes/output
    MOVD signal_len+8(FP), R10
    MOVD kernel_base+24(FP), R12
    MOVD kernel_len+32(FP), R13     // R13 = kLen

    SUB R13, R10                   // signal_len - kLen
    ADD $1, R10                    // R10 = n
    VEOR V6.B16, V6.B16, V6.B16    // running max F6 = 0

cvma_neon_outer:
    MOVD R11, R0                   // R0 = &signal[pos]
    MOVD R12, R1                   // R1 = &kernel[0]
    MOVD R13, R2                   // R2 = kLen

    VEOR V0.B16, V0.B16, V0.B16
    VEOR V1.B16, V1.B16, V1.B16

    LSR $3, R2, R3                 // kLen / 8
    CBZ R3, cvma_neon_rem4

cvma_neon_loop8:
    VLD1.P 16(R0), [V2.S4]
    VLD1.P 16(R0), [V3.S4]
    VLD1.P 16(R1), [V4.S4]
    VLD1.P 16(R1), [V5.S4]
    WORD $0x4E24CC40              // FMLA V0.4S, V2.4S, V4.4S
    WORD $0x4E25CC61              // FMLA V1.4S, V3.4S, V5.4S
    SUB $1, R3
    CBNZ R3, cvma_neon_loop8

    WORD $0x4E21D400             // FADD V0.4S, V0.4S, V1.4S

cvma_neon_rem4:
    AND $7, R2, R3
    LSR $2, R3, R4
    CBZ R4, cvma_neon_rem

    VLD1.P 16(R0), [V2.S4]
    VLD1.P 16(R1), [V4.S4]
    WORD $0x4E24CC40              // FMLA V0.4S, V2.4S, V4.4S

cvma_neon_rem:
    AND $3, R3, R4
    CBZ R4, cvma_neon_reduce

    // Reduce vector FIRST before scalar ops (scalar ops zero upper V bits).
    WORD $0x6E20D400             // FADDP V0.4S, V0.4S, V0.4S
    WORD $0x7E30D800             // FADDP S0, V0.2S

cvma_neon_scalar:
    FMOVS (R0), F2
    FMOVS (R1), F4
    FMADDS F4, F0, F2, F0         // F0 = F2 * F4 + F0
    ADD $4, R0
    ADD $4, R1
    SUB $1, R4
    CBNZ R4, cvma_neon_scalar

    B cvma_neon_absmax

cvma_neon_reduce:
    WORD $0x6E20D400             // FADDP V0.4S, V0.4S, V0.4S
    WORD $0x7E30D800             // FADDP S0, V0.2S

cvma_neon_absmax:
    FABSS F0, F0
    FMAXS F0, F6, F6           // F6 = max(F6, |dot|)
    ADD $4, R11               // pos += 1
    SUB $1, R10
    CBNZ R10, cvma_neon_outer

    FMOVS F6, ret+48(FP)
    RET

// ============================================================================
// INTERLEAVE / DEINTERLEAVE N=6 and N=8 (profiling-gated, 5.1/7.1 audio)
// The pair trick: ZIP adjacent channel pairs at .4S so each 64-bit lane holds a
// [cA_i, cB_i] frame pair, then ST3/ST4 with .2D arrangement lays the pairs down
// in frame order (and the inverse for deinterleave via LD3/LD4 .2D + UZP). Four
// frames per iteration, then a scalar tail. WORD-encoded with arm64asm-checked
// comments.
// ZIP1 Vd.4S, Vn.4S, Vm.4S: 0x4E803800 | (Vm << 16) | (Vn << 5) | Vd
// ZIP2 Vd.4S, Vn.4S, Vm.4S: 0x4E807800 | (Vm << 16) | (Vn << 5) | Vd
// UZP1 Vd.4S, Vn.4S, Vm.4S: 0x4E801800 | (Vm << 16) | (Vn << 5) | Vd
// UZP2 Vd.4S, Vn.4S, Vm.4S: 0x4E805800 | (Vm << 16) | (Vn << 5) | Vd
// ============================================================================

// func interleave6NEON(dst, s0, s1, s2, s3, s4, s5 []float32, n int)
TEXT ·interleave6NEON(SB), NOSPLIT, $0-176
    MOVD dst_base+0(FP), R0
    MOVD s0_base+24(FP), R1
    MOVD s1_base+48(FP), R2
    MOVD s2_base+72(FP), R3
    MOVD s3_base+96(FP), R4
    MOVD s4_base+120(FP), R5
    MOVD s5_base+144(FP), R6
    MOVD n+168(FP), R7

    LSR $2, R7, R8             // R8 = n / 4 (4 frames per iteration)
    CBZ R8, interleave6_neon_tail

interleave6_neon_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    VLD1.P 16(R3), [V2.S4]
    VLD1.P 16(R4), [V3.S4]
    VLD1.P 16(R5), [V4.S4]
    VLD1.P 16(R6), [V5.S4]
    WORD $0x4E813806           // ZIP1 V6.4S, V0.4S, V1.4S   (frames 0,1 ch 0,1)
    WORD $0x4E833847           // ZIP1 V7.4S, V2.4S, V3.4S   (frames 0,1 ch 2,3)
    WORD $0x4E853888           // ZIP1 V8.4S, V4.4S, V5.4S   (frames 0,1 ch 4,5)
    WORD $0x4E817809           // ZIP2 V9.4S, V0.4S, V1.4S   (frames 2,3 ch 0,1)
    WORD $0x4E83784A           // ZIP2 V10.4S, V2.4S, V3.4S  (frames 2,3 ch 2,3)
    WORD $0x4E85788B           // ZIP2 V11.4S, V4.4S, V5.4S  (frames 2,3 ch 4,5)
    VST3.P [V6.D2, V7.D2, V8.D2], 48(R0)     // frames 0,1
    VST3.P [V9.D2, V10.D2, V11.D2], 48(R0)   // frames 2,3
    SUB $1, R8
    CBNZ R8, interleave6_neon_loop4

interleave6_neon_tail:
    AND $3, R7
    CBZ R7, interleave6_neon_done

interleave6_neon_tail1:
    FMOVS (R1), F0
    FMOVS (R2), F1
    FMOVS (R3), F2
    FMOVS (R4), F3
    FMOVS (R5), F4
    FMOVS (R6), F5
    FMOVS F0, (R0)
    FMOVS F1, 4(R0)
    FMOVS F2, 8(R0)
    FMOVS F3, 12(R0)
    FMOVS F4, 16(R0)
    FMOVS F5, 20(R0)
    ADD $4, R1
    ADD $4, R2
    ADD $4, R3
    ADD $4, R4
    ADD $4, R5
    ADD $4, R6
    ADD $24, R0
    SUB $1, R7
    CBNZ R7, interleave6_neon_tail1

interleave6_neon_done:
    RET

// func deinterleave6NEON(d0, d1, d2, d3, d4, d5, src []float32, n int)
TEXT ·deinterleave6NEON(SB), NOSPLIT, $0-176
    MOVD d0_base+0(FP), R0
    MOVD d1_base+24(FP), R1
    MOVD d2_base+48(FP), R2
    MOVD d3_base+72(FP), R3
    MOVD d4_base+96(FP), R4
    MOVD d5_base+120(FP), R5
    MOVD src_base+144(FP), R6
    MOVD n+168(FP), R7

    LSR $2, R7, R8             // 4 frames per iteration
    CBZ R8, deinterleave6_neon_tail

deinterleave6_neon_loop4:
    VLD3.P 48(R6), [V0.D2, V1.D2, V2.D2]     // frames 0,1: V0=ch(0,1), V1=ch(2,3), V2=ch(4,5)
    VLD3.P 48(R6), [V3.D2, V4.D2, V5.D2]     // frames 2,3
    WORD $0x4E831810           // UZP1 V16.4S, V0.4S, V3.4S  (s0)
    WORD $0x4E835811           // UZP2 V17.4S, V0.4S, V3.4S  (s1)
    WORD $0x4E841832           // UZP1 V18.4S, V1.4S, V4.4S  (s2)
    WORD $0x4E845833           // UZP2 V19.4S, V1.4S, V4.4S  (s3)
    WORD $0x4E851854           // UZP1 V20.4S, V2.4S, V5.4S  (s4)
    WORD $0x4E855855           // UZP2 V21.4S, V2.4S, V5.4S  (s5)
    VST1.P [V16.S4], 16(R0)
    VST1.P [V17.S4], 16(R1)
    VST1.P [V18.S4], 16(R2)
    VST1.P [V19.S4], 16(R3)
    VST1.P [V20.S4], 16(R4)
    VST1.P [V21.S4], 16(R5)
    SUB $1, R8
    CBNZ R8, deinterleave6_neon_loop4

deinterleave6_neon_tail:
    AND $3, R7
    CBZ R7, deinterleave6_neon_done

deinterleave6_neon_tail1:
    FMOVS (R6), F0
    FMOVS 4(R6), F1
    FMOVS 8(R6), F2
    FMOVS 12(R6), F3
    FMOVS 16(R6), F4
    FMOVS 20(R6), F5
    FMOVS F0, (R0)
    FMOVS F1, (R1)
    FMOVS F2, (R2)
    FMOVS F3, (R3)
    FMOVS F4, (R4)
    FMOVS F5, (R5)
    ADD $24, R6
    ADD $4, R0
    ADD $4, R1
    ADD $4, R2
    ADD $4, R3
    ADD $4, R4
    ADD $4, R5
    SUB $1, R7
    CBNZ R7, deinterleave6_neon_tail1

deinterleave6_neon_done:
    RET

// func interleave8NEON(dst, s0, s1, s2, s3, s4, s5, s6, s7 []float32, n int)
TEXT ·interleave8NEON(SB), NOSPLIT, $0-224
    MOVD dst_base+0(FP), R0
    MOVD s0_base+24(FP), R1
    MOVD s1_base+48(FP), R2
    MOVD s2_base+72(FP), R3
    MOVD s3_base+96(FP), R4
    MOVD s4_base+120(FP), R5
    MOVD s5_base+144(FP), R6
    MOVD s6_base+168(FP), R7
    MOVD s7_base+192(FP), R8
    MOVD n+216(FP), R9

    LSR $2, R9, R10            // 4 frames per iteration
    CBZ R10, interleave8_neon_tail

interleave8_neon_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    VLD1.P 16(R3), [V2.S4]
    VLD1.P 16(R4), [V3.S4]
    VLD1.P 16(R5), [V4.S4]
    VLD1.P 16(R6), [V5.S4]
    VLD1.P 16(R7), [V6.S4]
    VLD1.P 16(R8), [V7.S4]
    WORD $0x4E813808           // ZIP1 V8.4S, V0.4S, V1.4S   (frames 0,1 ch 0,1)
    WORD $0x4E833849           // ZIP1 V9.4S, V2.4S, V3.4S   (frames 0,1 ch 2,3)
    WORD $0x4E85388A           // ZIP1 V10.4S, V4.4S, V5.4S  (frames 0,1 ch 4,5)
    WORD $0x4E8738CB           // ZIP1 V11.4S, V6.4S, V7.4S  (frames 0,1 ch 6,7)
    WORD $0x4E81780C           // ZIP2 V12.4S, V0.4S, V1.4S  (frames 2,3 ch 0,1)
    WORD $0x4E83784D           // ZIP2 V13.4S, V2.4S, V3.4S  (frames 2,3 ch 2,3)
    WORD $0x4E85788E           // ZIP2 V14.4S, V4.4S, V5.4S  (frames 2,3 ch 4,5)
    WORD $0x4E8778CF           // ZIP2 V15.4S, V6.4S, V7.4S  (frames 2,3 ch 6,7)
    VST4.P [V8.D2, V9.D2, V10.D2, V11.D2], 64(R0)     // frames 0,1
    VST4.P [V12.D2, V13.D2, V14.D2, V15.D2], 64(R0)   // frames 2,3
    SUB $1, R10
    CBNZ R10, interleave8_neon_loop4

interleave8_neon_tail:
    AND $3, R9
    CBZ R9, interleave8_neon_done

interleave8_neon_tail1:
    FMOVS (R1), F0
    FMOVS (R2), F1
    FMOVS (R3), F2
    FMOVS (R4), F3
    FMOVS (R5), F4
    FMOVS (R6), F5
    FMOVS (R7), F6
    FMOVS (R8), F7
    FMOVS F0, (R0)
    FMOVS F1, 4(R0)
    FMOVS F2, 8(R0)
    FMOVS F3, 12(R0)
    FMOVS F4, 16(R0)
    FMOVS F5, 20(R0)
    FMOVS F6, 24(R0)
    FMOVS F7, 28(R0)
    ADD $4, R1
    ADD $4, R2
    ADD $4, R3
    ADD $4, R4
    ADD $4, R5
    ADD $4, R6
    ADD $4, R7
    ADD $4, R8
    ADD $32, R0
    SUB $1, R9
    CBNZ R9, interleave8_neon_tail1

interleave8_neon_done:
    RET

// func deinterleave8NEON(d0, d1, d2, d3, d4, d5, d6, d7, src []float32, n int)
TEXT ·deinterleave8NEON(SB), NOSPLIT, $0-224
    MOVD d0_base+0(FP), R0
    MOVD d1_base+24(FP), R1
    MOVD d2_base+48(FP), R2
    MOVD d3_base+72(FP), R3
    MOVD d4_base+96(FP), R4
    MOVD d5_base+120(FP), R5
    MOVD d6_base+144(FP), R6
    MOVD d7_base+168(FP), R7
    MOVD src_base+192(FP), R8
    MOVD n+216(FP), R9

    LSR $2, R9, R10            // 4 frames per iteration
    CBZ R10, deinterleave8_neon_tail

deinterleave8_neon_loop4:
    VLD4.P 64(R8), [V0.D2, V1.D2, V2.D2, V3.D2]   // frames 0,1: ch(0,1),(2,3),(4,5),(6,7)
    VLD4.P 64(R8), [V4.D2, V5.D2, V6.D2, V7.D2]   // frames 2,3
    WORD $0x4E841810           // UZP1 V16.4S, V0.4S, V4.4S  (s0)
    WORD $0x4E845811           // UZP2 V17.4S, V0.4S, V4.4S  (s1)
    WORD $0x4E851832           // UZP1 V18.4S, V1.4S, V5.4S  (s2)
    WORD $0x4E855833           // UZP2 V19.4S, V1.4S, V5.4S  (s3)
    WORD $0x4E861854           // UZP1 V20.4S, V2.4S, V6.4S  (s4)
    WORD $0x4E865855           // UZP2 V21.4S, V2.4S, V6.4S  (s5)
    WORD $0x4E871876           // UZP1 V22.4S, V3.4S, V7.4S  (s6)
    WORD $0x4E875877           // UZP2 V23.4S, V3.4S, V7.4S  (s7)
    VST1.P [V16.S4], 16(R0)
    VST1.P [V17.S4], 16(R1)
    VST1.P [V18.S4], 16(R2)
    VST1.P [V19.S4], 16(R3)
    VST1.P [V20.S4], 16(R4)
    VST1.P [V21.S4], 16(R5)
    VST1.P [V22.S4], 16(R6)
    VST1.P [V23.S4], 16(R7)
    SUB $1, R10
    CBNZ R10, deinterleave8_neon_loop4

deinterleave8_neon_tail:
    AND $3, R9
    CBZ R9, deinterleave8_neon_done

deinterleave8_neon_tail1:
    FMOVS (R8), F0
    FMOVS 4(R8), F1
    FMOVS 8(R8), F2
    FMOVS 12(R8), F3
    FMOVS 16(R8), F4
    FMOVS 20(R8), F5
    FMOVS 24(R8), F6
    FMOVS 28(R8), F7
    FMOVS F0, (R0)
    FMOVS F1, (R1)
    FMOVS F2, (R2)
    FMOVS F3, (R3)
    FMOVS F4, (R4)
    FMOVS F5, (R5)
    FMOVS F6, (R6)
    FMOVS F7, (R7)
    ADD $32, R8
    ADD $4, R0
    ADD $4, R1
    ADD $4, R2
    ADD $4, R3
    ADD $4, R4
    ADD $4, R5
    ADD $4, R6
    ADD $4, R7
    SUB $1, R9
    CBNZ R9, deinterleave8_neon_tail1

deinterleave8_neon_done:
    RET

// ============================================================================
// logNEON32 / powNEON32 / powElemNEON32: vectorized natural log core (#109)
// ============================================================================

// High-order Cephes logf coefficients p7/p8 plus the exp-core constants for
// the pow kernels, loaded per iteration with VLD1 into working registers
// because the persistent V12-V31 budget is exhausted: [p7, p8] and
// [ln(2), 1/120, 1/24, 1/6], one lane quad each.
DATA log32neon_p78<>+0x00(SB)/4, $0xbe7ffffc  // p7 = -2.4999993993e-1
DATA log32neon_p78<>+0x04(SB)/4, $0xbe7ffffc
DATA log32neon_p78<>+0x08(SB)/4, $0xbe7ffffc
DATA log32neon_p78<>+0x0c(SB)/4, $0xbe7ffffc
DATA log32neon_p78<>+0x10(SB)/4, $0x3eaaaaaa  // p8 = +3.3333331174e-1
DATA log32neon_p78<>+0x14(SB)/4, $0x3eaaaaaa
DATA log32neon_p78<>+0x18(SB)/4, $0x3eaaaaaa
DATA log32neon_p78<>+0x1c(SB)/4, $0x3eaaaaaa
GLOBL log32neon_p78<>(SB), RODATA|NOPTR, $32

DATA log32neon_expc<>+0x00(SB)/4, $0x3f317218  // ln(2)
DATA log32neon_expc<>+0x04(SB)/4, $0x3f317218
DATA log32neon_expc<>+0x08(SB)/4, $0x3f317218
DATA log32neon_expc<>+0x0c(SB)/4, $0x3f317218
DATA log32neon_expc<>+0x10(SB)/4, $0x3c088889  // 1/120
DATA log32neon_expc<>+0x14(SB)/4, $0x3c088889
DATA log32neon_expc<>+0x18(SB)/4, $0x3c088889
DATA log32neon_expc<>+0x1c(SB)/4, $0x3c088889
DATA log32neon_expc<>+0x20(SB)/4, $0x3d2aaaab  // 1/24
DATA log32neon_expc<>+0x24(SB)/4, $0x3d2aaaab
DATA log32neon_expc<>+0x28(SB)/4, $0x3d2aaaab
DATA log32neon_expc<>+0x2c(SB)/4, $0x3d2aaaab
DATA log32neon_expc<>+0x30(SB)/4, $0x3e2aaaab  // 1/6
DATA log32neon_expc<>+0x34(SB)/4, $0x3e2aaaab
DATA log32neon_expc<>+0x38(SB)/4, $0x3e2aaaab
DATA log32neon_expc<>+0x3c(SB)/4, $0x3e2aaaab
GLOBL log32neon_expc<>(SB), RODATA|NOPTR, $64

// func logNEON32(dst, src []float32, k1hi, k1lo, k2 float32)
// Shared kernel for Log, Log2, and Log10: per lane it computes
// result = e*k1hi + (lnm*k2 + e*k1lo), with x = m*2^e, m in
// [sqrt(2)/2, sqrt(2)) and lnm = ln(m) = z - 0.5*z^2 + z^3*P(z) for z = m-1
// (Cephes logf degree-8 minimax polynomial; same algorithm as the amd64
// logAVX). Positive subnormal inputs are pre-scaled by 2^32 (exponent bias
// -32). Special lanes are fixed up with BIT blends from the original input:
// +Inf -> +Inf, +-0 -> -Inf, x < 0 or NaN -> NaN, matching math.Log.
// Processes 4 elements per iteration; the 0-3 element tail uses the scalar
// path below.
TEXT ·logNEON32(SB), NOSPLIT, $0-60
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1
    MOVD $log32neon_p78<>(SB), R5     // p7/p8 table base
    ADD $16, R5, R6                   // p8 base

    // Special-value constants for the blend fixups
    MOVW $0x7f800000, R10
    VMOV R10, V12.S[0]
    VDUP V12.S[0], V12.S4             // V12 = +Inf
    MOVW $0xff800000, R10
    VMOV R10, V13.S[0]
    VDUP V13.S[0], V13.S4             // V13 = -Inf
    MOVW $0x7fc00000, R10
    VMOV R10, V14.S[0]
    VDUP V14.S[0], V14.S4             // V14 = quiet NaN

    // Reduction constants (see logAVX for the algorithm notes)
    MOVW $0x3f350000, R10
    VMOV R10, V15.S[0]
    VDUP V15.S[0], V15.S4             // V15 = reduction offset
    MOVW $0xff800000, R10
    VMOV R10, V16.S[0]
    VDUP V16.S[0], V16.S4             // V16 = exponent mask
    MOVW $0x00800000, R10
    VMOV R10, V17.S[0]
    VDUP V17.S[0], V17.S4             // V17 = FLT_MIN
    MOVW $0x4f800000, R10
    VMOV R10, V18.S[0]
    VDUP V18.S[0], V18.S4             // V18 = 2^32 (subnormal pre-scale)
    MOVW $0xc2000000, R10
    VMOV R10, V19.S[0]
    VDUP V19.S[0], V19.S4             // V19 = -32.0 (exponent bias)
    FMOVS $1.0, F20
    VDUP V20.S[0], V20.S4             // V20 = 1.0
    FMOVS $0.5, F21
    VDUP V21.S[0], V21.S4             // V21 = 0.5

    // Reconstruction constants from the arguments
    FMOVS k1hi+48(FP), F22
    VDUP V22.S[0], V22.S4             // V22 = k1hi
    FMOVS k1lo+52(FP), F23
    VDUP V23.S[0], V23.S4             // V23 = k1lo
    FMOVS k2+56(FP), F24
    VDUP V24.S[0], V24.S4             // V24 = k2

    // Cephes logf coefficients p0..p6 (p7/p8 come from the table)
    MOVW $0x3d9021bb, R10
    VMOV R10, V25.S[0]
    VDUP V25.S[0], V25.S4             // V25 = p0 = +7.0376836292e-2
    MOVW $0xbdebd1b8, R10
    VMOV R10, V26.S[0]
    VDUP V26.S[0], V26.S4             // V26 = p1 = -1.1514610310e-1
    MOVW $0x3def251a, R10
    VMOV R10, V27.S[0]
    VDUP V27.S[0], V27.S4             // V27 = p2 = +1.1676998740e-1
    MOVW $0xbdfe5d4f, R10
    VMOV R10, V28.S[0]
    VDUP V28.S[0], V28.S4             // V28 = p3 = -1.2420140846e-1
    MOVW $0x3e11e9bf, R10
    VMOV R10, V29.S[0]
    VDUP V29.S[0], V29.S4             // V29 = p4 = +1.4249322787e-1
    MOVW $0xbe2aae50, R10
    VMOV R10, V30.S[0]
    VDUP V30.S[0], V30.S4             // V30 = p5 = -1.6668057665e-1
    MOVW $0x3e4cceac, R10
    VMOV R10, V31.S[0]
    VDUP V31.S[0], V31.S4             // V31 = p6 = +2.0000714765e-1

    LSR $2, R3, R4
    CBZ R4, log32_neon_scalar

log32_neon_loop4:
    VLD1.P 16(R1), [V0.S4]            // V0 = x (kept for the special blends)
    VLD1 (R5), [V10.S4]               // V10 = p7
    VLD1 (R6), [V11.S4]               // V11 = p8

    // Subnormal pre-scale: lanes with x < FLT_MIN scaled by 2^32, bias -32
    WORD $0x6ea0e621                  // FCMGT V1.4S, V17.4S, V0.4S   mask: x < FLT_MIN
    WORD $0x6e32dc02                  // FMUL V2.4S, V0.4S, V18.4S    x * 2^32
    WORD $0x4ea11c23                  // MOV V3.16B, V1.16B           copy mask for BSL
    WORD $0x6e601c43                  // BSL V3.16B, V2.16B, V0.16B   xs
    WORD $0x4e331c24                  // AND V4.16B, V1.16B, V19.16B  ebias

    // tmp = bits(xs) - OFF; bits(m) = bits(xs) - (tmp & expmask);
    // e = (tmp >> 23) + ebias, leaving m in [sqrt(2)/2, sqrt(2))
    WORD $0x6eaf8465                  // SUB V5.4S, V3.4S, V15.4S     tmp
    WORD $0x4e301ca6                  // AND V6.16B, V5.16B, V16.16B  tmp & expmask
    WORD $0x6ea68466                  // SUB V6.4S, V3.4S, V6.4S      bits(m)
    WORD $0x4f2904a5                  // SSHR V5.4S, V5.4S, #23       e (int32, arithmetic)
    WORD $0x4e21d8a5                  // SCVTF V5.4S, V5.4S           e as float32
    WORD $0x4e24d4a4                  // FADD V4.4S, V5.4S, V4.4S     e = e + ebias

    // z = m - 1, zz = z^2
    WORD $0x4eb4d4c7                  // FSUB V7.4S, V6.4S, V20.4S    z
    WORD $0x6e27dce6                  // FMUL V6.4S, V7.4S, V7.4S     zz

    // P(z), Horner ping-pong between V8/V9 with FMLA
    WORD $0x4eb91f28                  // MOV V8.16B, V25.16B          acc = p0
    WORD $0x4eba1f49                  // MOV V9.16B, V26.16B
    WORD $0x4e27cd09                  // FMLA V9.4S, V8.4S, V7.4S     p1 + acc*z
    WORD $0x4ebb1f68                  // MOV V8.16B, V27.16B
    WORD $0x4e27cd28                  // FMLA V8.4S, V9.4S, V7.4S     p2 + acc*z
    WORD $0x4ebc1f89                  // MOV V9.16B, V28.16B
    WORD $0x4e27cd09                  // FMLA V9.4S, V8.4S, V7.4S     p3 + acc*z
    WORD $0x4ebd1fa8                  // MOV V8.16B, V29.16B
    WORD $0x4e27cd28                  // FMLA V8.4S, V9.4S, V7.4S     p4 + acc*z
    WORD $0x4ebe1fc9                  // MOV V9.16B, V30.16B
    WORD $0x4e27cd09                  // FMLA V9.4S, V8.4S, V7.4S     p5 + acc*z
    WORD $0x4ebf1fe8                  // MOV V8.16B, V31.16B
    WORD $0x4e27cd28                  // FMLA V8.4S, V9.4S, V7.4S     p6 + acc*z
    WORD $0x4eaa1d49                  // MOV V9.16B, V10.16B
    WORD $0x4e27cd09                  // FMLA V9.4S, V8.4S, V7.4S     p7 + acc*z
    WORD $0x4eab1d68                  // MOV V8.16B, V11.16B
    WORD $0x4e27cd28                  // FMLA V8.4S, V9.4S, V7.4S     P(z) = p8 + acc*z

    // lnm = z + (z^3*P(z) - 0.5*zz)
    WORD $0x6e26dcea                  // FMUL V10.4S, V7.4S, V6.4S    z^3
    WORD $0x6e28dd4a                  // FMUL V10.4S, V10.4S, V8.4S   z^3 * P(z)
    WORD $0x4eb5ccca                  // FMLS V10.4S, V6.4S, V21.4S   -= 0.5*zz
    WORD $0x4e27d54a                  // FADD V10.4S, V10.4S, V7.4S   lnm

    // result = e*k1hi + (lnm*k2 + e*k1lo)
    WORD $0x6e37dc8b                  // FMUL V11.4S, V4.4S, V23.4S   e * k1lo
    WORD $0x4e38cd4b                  // FMLA V11.4S, V10.4S, V24.4S  += lnm * k2
    WORD $0x4e36cc8b                  // FMLA V11.4S, V4.4S, V22.4S   += e * k1hi

    // Special lanes from the original x
    WORD $0x4e2ce401                  // FCMEQ V1.4S, V0.4S, V12.4S   mask: x == +Inf
    WORD $0x6ea11d8b                  // BIT V11.16B, V12.16B, V1.16B
    WORD $0x4ea0d801                  // FCMEQ V1.4S, V0.4S, #0       mask: x == +-0
    WORD $0x6ea11dab                  // BIT V11.16B, V13.16B, V1.16B
    WORD $0x4ea0e802                  // FCMLT V2.4S, V0.4S, #0       mask: x < 0
    WORD $0x4e20e401                  // FCMEQ V1.4S, V0.4S, V0.4S    mask: x ordered
    WORD $0x4ee11c42                  // ORN V2.16B, V2.16B, V1.16B   (x < 0) | NaN
    WORD $0x6ea21dcb                  // BIT V11.16B, V14.16B, V2.16B

    VST1.P [V11.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, log32_neon_loop4

log32_neon_scalar:
    AND $3, R3
    CBZ R3, log32_neon_done

log32_neon_scalar_loop:
    MOVWU (R1), R7                    // R7 = bits(x)
    FMOVS (R1), F0

    // Specials: FCMPS sets V for unordered, N for negative, Z for zero
    FCMPS $(0.0), F0
    BVS log32_neon_scalar_nan
    BMI log32_neon_scalar_nan         // x < 0
    BEQ log32_neon_scalar_neginf      // x == +-0
    MOVD $0x7F800000, R8
    CMP R8, R7
    BEQ log32_neon_scalar_posinf

    // Subnormal pre-scale (x positive finite)
    MOVD $0, R9
    MOVD $0x00800000, R8
    CMP R8, R7
    BGE log32_neon_scalar_normal
    FMULS F18, F0, F0                 // x *= 2^32 (F18 from the vector constants)
    FMOVS F0, R7
    MOVD $-32, R9

log32_neon_scalar_normal:
    MOVD $0x3F350000, R8
    SUB R8, R7, R10                   // tmp = bits - OFF
    MOVW R10, R10                     // sign-extend the 32-bit tmp
    ASR $23, R10, R11                 // e
    ADD R9, R11, R11
    MOVD $0xFF800000, R8
    AND R8, R10, R10
    SUB R10, R7, R7                   // bits(m)
    FMOVS R7, F1                      // m
    SCVTFWS R11, F2                   // e as float32

    // z = m - 1, zz = z^2 (F20 = 1.0, F21 = 0.5 from the vector constants)
    FSUBS F20, F1, F3                 // z
    FMULS F3, F3, F4                  // zz

    // P(z): FMADDS Fm, Fa, Fn, Fd computes Fd = Fa + Fn*Fm
    FMADDS F3, F26, F25, F5           // p1 + p0*z
    FMADDS F3, F27, F5, F5            // p2 + acc*z
    FMADDS F3, F28, F5, F5
    FMADDS F3, F29, F5, F5
    FMADDS F3, F30, F5, F5
    FMADDS F3, F31, F5, F5            // p6 + acc*z
    FMOVS log32neon_p78<>(SB), F6
    FMADDS F3, F6, F5, F5             // p7 + acc*z
    FMOVS log32neon_p78<>+16(SB), F6
    FMADDS F3, F6, F5, F5             // P(z) = p8 + acc*z

    FMULS F3, F4, F6                  // z^3
    FMULS F6, F5, F5                  // z^3*P(z)
    FMSUBS F4, F5, F21, F5            // -= 0.5*zz: F5 = F5 - F21*F4
    FADDS F3, F5, F5                  // lnm

    FMULS F23, F2, F6                 // e * k1lo
    FMADDS F24, F6, F5, F6            // += lnm * k2
    FMADDS F22, F6, F2, F6            // += e * k1hi
    FMOVS F6, (R0)
    B log32_neon_scalar_next

log32_neon_scalar_nan:
    MOVD $0x7FC00000, R8
    MOVW R8, (R0)
    B log32_neon_scalar_next

log32_neon_scalar_neginf:
    MOVD $0xFF800000, R8
    MOVW R8, (R0)
    B log32_neon_scalar_next

log32_neon_scalar_posinf:
    MOVD $0x7F800000, R8
    MOVW R8, (R0)

log32_neon_scalar_next:
    ADD $4, R1
    ADD $4, R0
    SUB $1, R3
    CBNZ R3, log32_neon_scalar_loop

log32_neon_done:
    RET

// func powNEON32(dst, src []float32, exp float32)
// Fused pow(x, p) = exp(p*ln(x)) for slices whose elements are all positive
// and finite (the dispatcher guarantees this, see powSIMDOK32). The log core
// matches logNEON32; the exp core matches expNEON except y = p*ln(x) is
// clamped to [-104, 89] (past ln(MaxFloat32) and ln of the smallest
// subnormal) and the 2^k reconstruction is split into
// 2^(k>>1) * 2^(k-(k>>1)), so overflow goes to +Inf and underflow degrades
// gradually through subnormals to 0, matching math.Pow's result classes.
// Accuracy is ~1.4e-5 relative (log error amplified by |y|, then the exp
// polynomial).
TEXT ·powNEON32(SB), NOSPLIT, $0-52
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1
    MOVD $log32neon_p78<>(SB), R5     // p7/p8 table base
    ADD $16, R5, R6                   // p8 base
    MOVD $log32neon_expc<>(SB), R7    // exp-core constant table

    FMOVS exp+48(FP), F12
    VDUP V12.S[0], V12.S4             // V12 = p

    // Pow clamp bounds (see the kernel comment)
    MOVW $0x42b20000, R10
    VMOV R10, V13.S[0]
    VDUP V13.S[0], V13.S4             // V13 = 89.0
    MOVW $0xc2d00000, R10
    VMOV R10, V14.S[0]
    VDUP V14.S[0], V14.S4             // V14 = -104.0

    // Log-core reduction constants (see logNEON32)
    MOVW $0x3f350000, R10
    VMOV R10, V15.S[0]
    VDUP V15.S[0], V15.S4             // V15 = reduction offset
    MOVW $0xff800000, R10
    VMOV R10, V16.S[0]
    VDUP V16.S[0], V16.S4             // V16 = exponent mask
    MOVW $0x00800000, R10
    VMOV R10, V17.S[0]
    VDUP V17.S[0], V17.S4             // V17 = FLT_MIN
    MOVW $0x4f800000, R10
    VMOV R10, V18.S[0]
    VDUP V18.S[0], V18.S4             // V18 = 2^32
    MOVW $0xc2000000, R10
    VMOV R10, V19.S[0]
    VDUP V19.S[0], V19.S4             // V19 = -32.0
    FMOVS $1.0, F20
    VDUP V20.S[0], V20.S4             // V20 = 1.0
    FMOVS $0.5, F21
    VDUP V21.S[0], V21.S4             // V21 = 0.5

    // Cephes logf ln(2) hi/lo split and log2(e)
    MOVW $0x3f318000, R10
    VMOV R10, V22.S[0]
    VDUP V22.S[0], V22.S4             // V22 = ln2 hi
    MOVW $0xb95e8083, R10
    VMOV R10, V23.S[0]
    VDUP V23.S[0], V23.S4             // V23 = ln2 lo
    MOVW $0x3fb8aa3b, R10
    VMOV R10, V24.S[0]
    VDUP V24.S[0], V24.S4             // V24 = log2(e)

    // Cephes logf coefficients p0..p6 (p7/p8 from the table)
    MOVW $0x3d9021bb, R10
    VMOV R10, V25.S[0]
    VDUP V25.S[0], V25.S4             // V25 = p0
    MOVW $0xbdebd1b8, R10
    VMOV R10, V26.S[0]
    VDUP V26.S[0], V26.S4             // V26 = p1
    MOVW $0x3def251a, R10
    VMOV R10, V27.S[0]
    VDUP V27.S[0], V27.S4             // V27 = p2
    MOVW $0xbdfe5d4f, R10
    VMOV R10, V28.S[0]
    VDUP V28.S[0], V28.S4             // V28 = p3
    MOVW $0x3e11e9bf, R10
    VMOV R10, V29.S[0]
    VDUP V29.S[0], V29.S4             // V29 = p4
    MOVW $0xbe2aae50, R10
    VMOV R10, V30.S[0]
    VDUP V30.S[0], V30.S4             // V30 = p5
    MOVW $0x3e4cceac, R10
    VMOV R10, V31.S[0]
    VDUP V31.S[0], V31.S4             // V31 = p6

    LSR $2, R3, R4
    CBZ R4, pow32_neon_scalar

pow32_neon_loop4:
    VLD1.P 16(R1), [V0.S4]            // V0 = x (positive finite)
    VLD1 (R5), [V10.S4]               // V10 = p7
    VLD1 (R6), [V11.S4]               // V11 = p8

    // --- log core (see logNEON32) ---
    WORD $0x6ea0e621                  // FCMGT V1.4S, V17.4S, V0.4S   mask: x < FLT_MIN
    WORD $0x6e32dc02                  // FMUL V2.4S, V0.4S, V18.4S    x * 2^32
    WORD $0x4ea11c23                  // MOV V3.16B, V1.16B
    WORD $0x6e601c43                  // BSL V3.16B, V2.16B, V0.16B   xs
    WORD $0x4e331c24                  // AND V4.16B, V1.16B, V19.16B  ebias
    WORD $0x6eaf8465                  // SUB V5.4S, V3.4S, V15.4S     tmp
    WORD $0x4e301ca6                  // AND V6.16B, V5.16B, V16.16B
    WORD $0x6ea68466                  // SUB V6.4S, V3.4S, V6.4S      bits(m)
    WORD $0x4f2904a5                  // SSHR V5.4S, V5.4S, #23       e (int32)
    WORD $0x4e21d8a5                  // SCVTF V5.4S, V5.4S           e as float32
    WORD $0x4e24d4a4                  // FADD V4.4S, V5.4S, V4.4S     e = e + ebias
    WORD $0x4eb4d4c7                  // FSUB V7.4S, V6.4S, V20.4S    z
    WORD $0x6e27dce6                  // FMUL V6.4S, V7.4S, V7.4S     zz
    WORD $0x4eb91f28                  // MOV V8.16B, V25.16B          acc = p0
    WORD $0x4eba1f49                  // MOV V9.16B, V26.16B
    WORD $0x4e27cd09                  // FMLA V9.4S, V8.4S, V7.4S     p1 + acc*z
    WORD $0x4ebb1f68                  // MOV V8.16B, V27.16B
    WORD $0x4e27cd28                  // FMLA V8.4S, V9.4S, V7.4S     p2 + acc*z
    WORD $0x4ebc1f89                  // MOV V9.16B, V28.16B
    WORD $0x4e27cd09                  // FMLA V9.4S, V8.4S, V7.4S     p3 + acc*z
    WORD $0x4ebd1fa8                  // MOV V8.16B, V29.16B
    WORD $0x4e27cd28                  // FMLA V8.4S, V9.4S, V7.4S     p4 + acc*z
    WORD $0x4ebe1fc9                  // MOV V9.16B, V30.16B
    WORD $0x4e27cd09                  // FMLA V9.4S, V8.4S, V7.4S     p5 + acc*z
    WORD $0x4ebf1fe8                  // MOV V8.16B, V31.16B
    WORD $0x4e27cd28                  // FMLA V8.4S, V9.4S, V7.4S     p6 + acc*z
    WORD $0x4eaa1d49                  // MOV V9.16B, V10.16B
    WORD $0x4e27cd09                  // FMLA V9.4S, V8.4S, V7.4S     p7 + acc*z
    WORD $0x4eab1d68                  // MOV V8.16B, V11.16B
    WORD $0x4e27cd28                  // FMLA V8.4S, V9.4S, V7.4S     P(z)
    WORD $0x6e26dcea                  // FMUL V10.4S, V7.4S, V6.4S    z^3
    WORD $0x6e28dd4a                  // FMUL V10.4S, V10.4S, V8.4S   z^3 * P(z)
    WORD $0x4eb5ccca                  // FMLS V10.4S, V6.4S, V21.4S   -= 0.5*zz
    WORD $0x4e27d54a                  // FADD V10.4S, V10.4S, V7.4S   lnm

    // ln(x) = e*ln2hi + (e*ln2lo + lnm)
    WORD $0x6e37dc8b                  // FMUL V11.4S, V4.4S, V23.4S   e * ln2lo
    WORD $0x4e2ad56b                  // FADD V11.4S, V11.4S, V10.4S  + lnm
    WORD $0x4e36cc8b                  // FMLA V11.4S, V4.4S, V22.4S   += e * ln2hi

    // y = p*ln(x), clamped to [-104, 89]
    WORD $0x6e2cdd60                  // FMUL V0.4S, V11.4S, V12.4S   y
    WORD $0x4eadf400                  // FMIN V0.4S, V0.4S, V13.4S
    WORD $0x4e2ef400                  // FMAX V0.4S, V0.4S, V14.4S

    // --- exp core (see expNEON); constants from the table ---
    VLD1 (R7), [V1.S4, V2.S4, V3.S4, V4.S4] // ln2, 1/120, 1/24, 1/6
    WORD $0x6e38dc05                  // FMUL V5.4S, V0.4S, V24.4S    y * log2e
    WORD $0x4e2188a5                  // FRINTN V5.4S, V5.4S          k
    WORD $0x6e21dca6                  // FMUL V6.4S, V5.4S, V1.4S     k * ln2
    WORD $0x4ea6d406                  // FSUB V6.4S, V0.4S, V6.4S     r
    WORD $0x6e22dcc7                  // FMUL V7.4S, V6.4S, V2.4S     r * 1/120
    WORD $0x4e23d4e7                  // FADD V7.4S, V7.4S, V3.4S     + 1/24
    WORD $0x6e26dce7                  // FMUL V7.4S, V7.4S, V6.4S
    WORD $0x4e24d4e7                  // FADD V7.4S, V7.4S, V4.4S     + 1/6
    WORD $0x6e26dce7                  // FMUL V7.4S, V7.4S, V6.4S
    WORD $0x4e35d4e7                  // FADD V7.4S, V7.4S, V21.4S    + 0.5
    WORD $0x6e26dce7                  // FMUL V7.4S, V7.4S, V6.4S
    WORD $0x4e34d4e7                  // FADD V7.4S, V7.4S, V20.4S    + 1
    WORD $0x6e26dce7                  // FMUL V7.4S, V7.4S, V6.4S
    WORD $0x4e34d4e7                  // FADD V7.4S, V7.4S, V20.4S    exp(r)
    // Split 2^k reconstruction (see powAVX in the f64 package)
    WORD $0x4ea1b8a5                  // FCVTZS V5.4S, V5.4S          int(k)
    WORD $0x4f3f04a8                  // SSHR V8.4S, V5.4S, #1        k1 = k >> 1
    WORD $0x6ea884a5                  // SUB V5.4S, V5.4S, V8.4S      k2 = k - k1
    WORD $0x4f375508                  // SHL V8.4S, V8.4S, #23
    WORD $0x4eb48508                  // ADD V8.4S, V8.4S, V20.4S     2^k1 bits
    WORD $0x6e28dce7                  // FMUL V7.4S, V7.4S, V8.4S     exp(r) * 2^k1
    WORD $0x4f3754a5                  // SHL V5.4S, V5.4S, #23
    WORD $0x4eb484a5                  // ADD V5.4S, V5.4S, V20.4S     2^k2 bits
    WORD $0x6e25dce7                  // FMUL V7.4S, V7.4S, V5.4S     * 2^k2 = exp(y)

    VST1.P [V7.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, pow32_neon_loop4

pow32_neon_scalar:
    AND $3, R3
    CBZ R3, pow32_neon_done

pow32_neon_scalar_loop:
    MOVWU (R1), R8                    // bits(x)
    FMOVS (R1), F0

    // log core, scalar (x positive finite)
    MOVD $0, R9
    MOVD $0x00800000, R10
    CMP R10, R8
    BGE pow32_neon_scalar_normal
    FMULS F18, F0, F0                 // x *= 2^32
    FMOVS F0, R8
    MOVD $-32, R9

pow32_neon_scalar_normal:
    MOVD $0x3F350000, R10
    SUB R10, R8, R11                  // tmp
    MOVW R11, R11                     // sign-extend
    ASR $23, R11, R12                 // e
    ADD R9, R12, R12
    MOVD $0xFF800000, R10
    AND R10, R11, R11
    SUB R11, R8, R8                   // bits(m)
    FMOVS R8, F1                      // m
    SCVTFWS R12, F2                   // e

    FSUBS F20, F1, F3                 // z
    FMULS F3, F3, F4                  // zz
    FMADDS F3, F26, F25, F5           // p1 + p0*z
    FMADDS F3, F27, F5, F5
    FMADDS F3, F28, F5, F5
    FMADDS F3, F29, F5, F5
    FMADDS F3, F30, F5, F5
    FMADDS F3, F31, F5, F5            // p6 + acc*z
    FMOVS log32neon_p78<>(SB), F6
    FMADDS F3, F6, F5, F5             // p7 + acc*z
    FMOVS log32neon_p78<>+16(SB), F6
    FMADDS F3, F6, F5, F5             // P(z)
    FMULS F3, F4, F6                  // z^3
    FMULS F6, F5, F5                  // z^3*P(z)
    FMSUBS F4, F5, F21, F5            // -= 0.5*zz
    FADDS F3, F5, F5                  // lnm
    FMULS F23, F2, F6                 // e * ln2lo
    FADDS F5, F6, F6                  // + lnm
    FMADDS F22, F6, F2, F6            // += e * ln2hi -> ln(x)

    // y = p*ln(x), clamped (F12 = p, F13/F14 = clamp bounds)
    FMULS F12, F6, F0
    FMINS F13, F0, F0
    FMAXS F14, F0, F0

    // exp core, scalar (F24 = log2e, F20 = 1.0, F21 = 0.5)
    FMULS F24, F0, F1
    FRINTNS F1, F2                    // k
    MOVW $0x3f317218, R10             // ln(2)
    FMOVS R10, F3
    FMULS F3, F2, F3
    FSUBS F3, F0, F0                  // r
    MOVW $0x3c088889, R10             // 1/120
    FMOVS R10, F4
    FMULS F0, F4, F4
    MOVW $0x3d2aaaab, R10             // 1/24
    FMOVS R10, F5
    FADDS F5, F4, F4
    FMULS F0, F4, F4
    MOVW $0x3e2aaaab, R10             // 1/6
    FMOVS R10, F5
    FADDS F5, F4, F4
    FMULS F0, F4, F4
    FADDS F21, F4, F4                 // + 0.5
    FMULS F0, F4, F4
    FADDS F20, F4, F4                 // + 1
    FMULS F0, F4, F4
    FADDS F20, F4, F4                 // exp(r)
    // Split 2^k reconstruction (see the vector body)
    FCVTZSSW F2, R8                   // k
    MOVW R8, R8                       // sign-extend
    ASR $1, R8, R10                   // k1 = k >> 1
    SUB R10, R8, R8                   // k2 = k - k1
    MOVD $0x3F800000, R11
    LSL $23, R10, R10
    ADD R11, R10, R10
    FMOVS R10, F5
    FMULS F5, F4, F4                  // exp(r) * 2^k1
    LSL $23, R8, R8
    ADD R11, R8, R8
    FMOVS R8, F5
    FMULS F5, F4, F4                  // * 2^k2 = exp(y)
    FMOVS F4, (R0)

    ADD $4, R1
    ADD $4, R0
    SUB $1, R3
    CBNZ R3, pow32_neon_scalar_loop

pow32_neon_done:
    RET

// func powElemNEON32(dst, base, exp []float32)
// Elementwise pow(base[i], exp[i]) = exp(exp[i]*ln(base[i])). Same cores and
// preconditions as powNEON32 (all bases positive finite, all exponents
// finite), with the exponent loaded per lane instead of broadcast.
TEXT ·powElemNEON32(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD base_base+24(FP), R1
    MOVD exp_base+48(FP), R2
    MOVD $log32neon_p78<>(SB), R5     // p7/p8 table base
    ADD $16, R5, R6                   // p8 base
    MOVD $log32neon_expc<>(SB), R7    // exp-core constant table

    // Pow clamp bounds
    MOVW $0x42b20000, R10
    VMOV R10, V13.S[0]
    VDUP V13.S[0], V13.S4             // V13 = 89.0
    MOVW $0xc2d00000, R10
    VMOV R10, V14.S[0]
    VDUP V14.S[0], V14.S4             // V14 = -104.0

    // Log-core reduction constants (see logNEON32)
    MOVW $0x3f350000, R10
    VMOV R10, V15.S[0]
    VDUP V15.S[0], V15.S4             // V15 = reduction offset
    MOVW $0xff800000, R10
    VMOV R10, V16.S[0]
    VDUP V16.S[0], V16.S4             // V16 = exponent mask
    MOVW $0x00800000, R10
    VMOV R10, V17.S[0]
    VDUP V17.S[0], V17.S4             // V17 = FLT_MIN
    MOVW $0x4f800000, R10
    VMOV R10, V18.S[0]
    VDUP V18.S[0], V18.S4             // V18 = 2^32
    MOVW $0xc2000000, R10
    VMOV R10, V19.S[0]
    VDUP V19.S[0], V19.S4             // V19 = -32.0
    FMOVS $1.0, F20
    VDUP V20.S[0], V20.S4             // V20 = 1.0
    FMOVS $0.5, F21
    VDUP V21.S[0], V21.S4             // V21 = 0.5

    // Cephes logf ln(2) hi/lo split and log2(e)
    MOVW $0x3f318000, R10
    VMOV R10, V22.S[0]
    VDUP V22.S[0], V22.S4             // V22 = ln2 hi
    MOVW $0xb95e8083, R10
    VMOV R10, V23.S[0]
    VDUP V23.S[0], V23.S4             // V23 = ln2 lo
    MOVW $0x3fb8aa3b, R10
    VMOV R10, V24.S[0]
    VDUP V24.S[0], V24.S4             // V24 = log2(e)

    // Cephes logf coefficients p0..p6 (p7/p8 from the table)
    MOVW $0x3d9021bb, R10
    VMOV R10, V25.S[0]
    VDUP V25.S[0], V25.S4             // V25 = p0
    MOVW $0xbdebd1b8, R10
    VMOV R10, V26.S[0]
    VDUP V26.S[0], V26.S4             // V26 = p1
    MOVW $0x3def251a, R10
    VMOV R10, V27.S[0]
    VDUP V27.S[0], V27.S4             // V27 = p2
    MOVW $0xbdfe5d4f, R10
    VMOV R10, V28.S[0]
    VDUP V28.S[0], V28.S4             // V28 = p3
    MOVW $0x3e11e9bf, R10
    VMOV R10, V29.S[0]
    VDUP V29.S[0], V29.S4             // V29 = p4
    MOVW $0xbe2aae50, R10
    VMOV R10, V30.S[0]
    VDUP V30.S[0], V30.S4             // V30 = p5
    MOVW $0x3e4cceac, R10
    VMOV R10, V31.S[0]
    VDUP V31.S[0], V31.S4             // V31 = p6

    LSR $2, R3, R4
    CBZ R4, powelem32_neon_scalar

powelem32_neon_loop4:
    VLD1.P 16(R1), [V0.S4]            // V0 = base (positive finite)
    VLD1 (R5), [V10.S4]               // V10 = p7
    VLD1 (R6), [V11.S4]               // V11 = p8

    // --- log core (see logNEON32) ---
    WORD $0x6ea0e621                  // FCMGT V1.4S, V17.4S, V0.4S   mask: x < FLT_MIN
    WORD $0x6e32dc02                  // FMUL V2.4S, V0.4S, V18.4S    x * 2^32
    WORD $0x4ea11c23                  // MOV V3.16B, V1.16B
    WORD $0x6e601c43                  // BSL V3.16B, V2.16B, V0.16B   xs
    WORD $0x4e331c24                  // AND V4.16B, V1.16B, V19.16B  ebias
    WORD $0x6eaf8465                  // SUB V5.4S, V3.4S, V15.4S     tmp
    WORD $0x4e301ca6                  // AND V6.16B, V5.16B, V16.16B
    WORD $0x6ea68466                  // SUB V6.4S, V3.4S, V6.4S      bits(m)
    WORD $0x4f2904a5                  // SSHR V5.4S, V5.4S, #23       e (int32)
    WORD $0x4e21d8a5                  // SCVTF V5.4S, V5.4S           e as float32
    WORD $0x4e24d4a4                  // FADD V4.4S, V5.4S, V4.4S     e = e + ebias
    WORD $0x4eb4d4c7                  // FSUB V7.4S, V6.4S, V20.4S    z
    WORD $0x6e27dce6                  // FMUL V6.4S, V7.4S, V7.4S     zz
    WORD $0x4eb91f28                  // MOV V8.16B, V25.16B          acc = p0
    WORD $0x4eba1f49                  // MOV V9.16B, V26.16B
    WORD $0x4e27cd09                  // FMLA V9.4S, V8.4S, V7.4S     p1 + acc*z
    WORD $0x4ebb1f68                  // MOV V8.16B, V27.16B
    WORD $0x4e27cd28                  // FMLA V8.4S, V9.4S, V7.4S     p2 + acc*z
    WORD $0x4ebc1f89                  // MOV V9.16B, V28.16B
    WORD $0x4e27cd09                  // FMLA V9.4S, V8.4S, V7.4S     p3 + acc*z
    WORD $0x4ebd1fa8                  // MOV V8.16B, V29.16B
    WORD $0x4e27cd28                  // FMLA V8.4S, V9.4S, V7.4S     p4 + acc*z
    WORD $0x4ebe1fc9                  // MOV V9.16B, V30.16B
    WORD $0x4e27cd09                  // FMLA V9.4S, V8.4S, V7.4S     p5 + acc*z
    WORD $0x4ebf1fe8                  // MOV V8.16B, V31.16B
    WORD $0x4e27cd28                  // FMLA V8.4S, V9.4S, V7.4S     p6 + acc*z
    WORD $0x4eaa1d49                  // MOV V9.16B, V10.16B
    WORD $0x4e27cd09                  // FMLA V9.4S, V8.4S, V7.4S     p7 + acc*z
    WORD $0x4eab1d68                  // MOV V8.16B, V11.16B
    WORD $0x4e27cd28                  // FMLA V8.4S, V9.4S, V7.4S     P(z)
    WORD $0x6e26dcea                  // FMUL V10.4S, V7.4S, V6.4S    z^3
    WORD $0x6e28dd4a                  // FMUL V10.4S, V10.4S, V8.4S   z^3 * P(z)
    WORD $0x4eb5ccca                  // FMLS V10.4S, V6.4S, V21.4S   -= 0.5*zz
    WORD $0x4e27d54a                  // FADD V10.4S, V10.4S, V7.4S   lnm

    // ln(base) = e*ln2hi + (e*ln2lo + lnm); the exponent load is issued
    // first so it overlaps the dependent FMUL/FADD/FMLA chain
    VLD1.P 16(R2), [V12.S4]           // V12 = exponents (finite)
    WORD $0x6e37dc8b                  // FMUL V11.4S, V4.4S, V23.4S   e * ln2lo
    WORD $0x4e2ad56b                  // FADD V11.4S, V11.4S, V10.4S  + lnm
    WORD $0x4e36cc8b                  // FMLA V11.4S, V4.4S, V22.4S   += e * ln2hi

    // y = exp[i]*ln(base[i]), clamped to [-104, 89]
    WORD $0x6e2cdd60                  // FMUL V0.4S, V11.4S, V12.4S   y
    WORD $0x4eadf400                  // FMIN V0.4S, V0.4S, V13.4S
    WORD $0x4e2ef400                  // FMAX V0.4S, V0.4S, V14.4S

    // --- exp core (see expNEON); constants from the table ---
    VLD1 (R7), [V1.S4, V2.S4, V3.S4, V4.S4] // ln2, 1/120, 1/24, 1/6
    WORD $0x6e38dc05                  // FMUL V5.4S, V0.4S, V24.4S    y * log2e
    WORD $0x4e2188a5                  // FRINTN V5.4S, V5.4S          k
    WORD $0x6e21dca6                  // FMUL V6.4S, V5.4S, V1.4S     k * ln2
    WORD $0x4ea6d406                  // FSUB V6.4S, V0.4S, V6.4S     r
    WORD $0x6e22dcc7                  // FMUL V7.4S, V6.4S, V2.4S     r * 1/120
    WORD $0x4e23d4e7                  // FADD V7.4S, V7.4S, V3.4S     + 1/24
    WORD $0x6e26dce7                  // FMUL V7.4S, V7.4S, V6.4S
    WORD $0x4e24d4e7                  // FADD V7.4S, V7.4S, V4.4S     + 1/6
    WORD $0x6e26dce7                  // FMUL V7.4S, V7.4S, V6.4S
    WORD $0x4e35d4e7                  // FADD V7.4S, V7.4S, V21.4S    + 0.5
    WORD $0x6e26dce7                  // FMUL V7.4S, V7.4S, V6.4S
    WORD $0x4e34d4e7                  // FADD V7.4S, V7.4S, V20.4S    + 1
    WORD $0x6e26dce7                  // FMUL V7.4S, V7.4S, V6.4S
    WORD $0x4e34d4e7                  // FADD V7.4S, V7.4S, V20.4S    exp(r)
    // Split 2^k reconstruction (see powNEON32)
    WORD $0x4ea1b8a5                  // FCVTZS V5.4S, V5.4S          int(k)
    WORD $0x4f3f04a8                  // SSHR V8.4S, V5.4S, #1        k1 = k >> 1
    WORD $0x6ea884a5                  // SUB V5.4S, V5.4S, V8.4S      k2 = k - k1
    WORD $0x4f375508                  // SHL V8.4S, V8.4S, #23
    WORD $0x4eb48508                  // ADD V8.4S, V8.4S, V20.4S     2^k1 bits
    WORD $0x6e28dce7                  // FMUL V7.4S, V7.4S, V8.4S     exp(r) * 2^k1
    WORD $0x4f3754a5                  // SHL V5.4S, V5.4S, #23
    WORD $0x4eb484a5                  // ADD V5.4S, V5.4S, V20.4S     2^k2 bits
    WORD $0x6e25dce7                  // FMUL V7.4S, V7.4S, V5.4S     * 2^k2 = exp(y)

    VST1.P [V7.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, powelem32_neon_loop4

powelem32_neon_scalar:
    AND $3, R3
    CBZ R3, powelem32_neon_done

powelem32_neon_scalar_loop:
    MOVWU (R1), R8                    // bits(base)
    FMOVS (R1), F0

    MOVD $0, R9
    MOVD $0x00800000, R10
    CMP R10, R8
    BGE powelem32_neon_scalar_normal
    FMULS F18, F0, F0                 // x *= 2^32
    FMOVS F0, R8
    MOVD $-32, R9

powelem32_neon_scalar_normal:
    MOVD $0x3F350000, R10
    SUB R10, R8, R11                  // tmp
    MOVW R11, R11                     // sign-extend
    ASR $23, R11, R12                 // e
    ADD R9, R12, R12
    MOVD $0xFF800000, R10
    AND R10, R11, R11
    SUB R11, R8, R8                   // bits(m)
    FMOVS R8, F1                      // m
    SCVTFWS R12, F2                   // e

    FSUBS F20, F1, F3                 // z
    FMULS F3, F3, F4                  // zz
    FMADDS F3, F26, F25, F5           // p1 + p0*z
    FMADDS F3, F27, F5, F5
    FMADDS F3, F28, F5, F5
    FMADDS F3, F29, F5, F5
    FMADDS F3, F30, F5, F5
    FMADDS F3, F31, F5, F5            // p6 + acc*z
    FMOVS log32neon_p78<>(SB), F6
    FMADDS F3, F6, F5, F5             // p7 + acc*z
    FMOVS log32neon_p78<>+16(SB), F6
    FMADDS F3, F6, F5, F5             // P(z)
    FMULS F3, F4, F6                  // z^3
    FMULS F6, F5, F5                  // z^3*P(z)
    FMSUBS F4, F5, F21, F5            // -= 0.5*zz
    FADDS F3, F5, F5                  // lnm
    FMULS F23, F2, F6                 // e * ln2lo
    FADDS F5, F6, F6                  // + lnm
    FMADDS F22, F6, F2, F6            // += e * ln2hi -> ln(base)

    // y = exp[i]*ln(base[i]), clamped
    FMOVS (R2), F12                   // p
    FMULS F12, F6, F0
    FMINS F13, F0, F0
    FMAXS F14, F0, F0

    // exp core, scalar (see powNEON32 tail)
    FMULS F24, F0, F1
    FRINTNS F1, F2                    // k
    MOVW $0x3f317218, R10             // ln(2)
    FMOVS R10, F3
    FMULS F3, F2, F3
    FSUBS F3, F0, F0                  // r
    MOVW $0x3c088889, R10             // 1/120
    FMOVS R10, F4
    FMULS F0, F4, F4
    MOVW $0x3d2aaaab, R10             // 1/24
    FMOVS R10, F5
    FADDS F5, F4, F4
    FMULS F0, F4, F4
    MOVW $0x3e2aaaab, R10             // 1/6
    FMOVS R10, F5
    FADDS F5, F4, F4
    FMULS F0, F4, F4
    FADDS F21, F4, F4                 // + 0.5
    FMULS F0, F4, F4
    FADDS F20, F4, F4                 // + 1
    FMULS F0, F4, F4
    FADDS F20, F4, F4                 // exp(r)
    // Split 2^k reconstruction (see the vector body)
    FCVTZSSW F2, R8                   // k
    MOVW R8, R8                       // sign-extend
    ASR $1, R8, R10                   // k1 = k >> 1
    SUB R10, R8, R8                   // k2 = k - k1
    MOVD $0x3F800000, R11
    LSL $23, R10, R10
    ADD R11, R10, R10
    FMOVS R10, F5
    FMULS F5, F4, F4                  // exp(r) * 2^k1
    LSL $23, R8, R8
    ADD R11, R8, R8
    FMOVS R8, F5
    FMULS F5, F4, F4                  // * 2^k2 = exp(y)
    FMOVS F4, (R0)

    ADD $4, R1
    ADD $4, R2
    ADD $4, R0
    SUB $1, R3
    CBNZ R3, powelem32_neon_scalar_loop

powelem32_neon_done:
    RET

// func minIdxOfSumRows4NEON(vals []float32, idxs []int32, a, k []float32, rev int)
// Lane-per-row argmin-of-sum for a block of four rows. Each of the four NEON
// lanes owns one output row and replays the scalar loop exactly: candidate i
// (i in [0, n), n = len(a)) broadcasts a[i], adds it to the four rows' k values
// with a single FADD (one rounding, never fused), and updates (bestVal, bestIdx)
// only on a strict FCMGT (bestVal > c). Strict compare keeps first-index-wins on
// ties, leaves the incumbent on a NaN candidate (FCMGT yields 0 for any NaN
// operand), and never lets a +Inf pad beat a finite value; the bits match the Go
// reference lane for lane.
//
// The k pointer is pre-sliced by the dispatcher so k[0] is the first address the
// kernel reads. Both slide signs load k[i:i+4] ascending (the window slides by
// one element per candidate, so k advances 4 bytes, not 16). For slide == +1 the
// dispatcher passes row r's window start, lane l reads k[i+l] = row (r+l)'s
// candidate i, and the union over i in [0,n), l in [0,4) is k indices
// [off, off+n+2] = exactly rows r..r+3's windows, all range-checked by the
// wrapper. For slide == -1 it passes row (r+3)'s window start (off-3 >= 0 because
// the wrapper validated row r+3); the ascending load lands row r+3 in lane 0 and
// row r in lane 3, so rev == 1 reverses the two result vectors once at store
// (REV64 + EXT) to restore lane l == row r+l. Either way every read stays inside
// the union of the four rows' validated windows, so there is no over-read.
TEXT ·minIdxOfSumRows4NEON(SB), NOSPLIT, $0-104
    MOVD vals_base+0(FP), R0
    MOVD idxs_base+24(FP), R1
    MOVD a_base+48(FP), R2
    MOVD a_len+56(FP), R3          // R3 = n (dispatcher guarantees n >= 1)
    MOVD k_base+72(FP), R4
    MOVD rev+96(FP), R5

    // Candidate 0: seed bestVal = a[0] + k[0:4], bestIdx = 0, curIdx = 0.
    FMOVS (R2), F3                 // a[0]
    VDUP V3.S[0], V3.S4            // broadcast a[0] across the four lanes
    VLD1 (R4), [V2.S4]             // k[0:4] (unaligned)
    ADD $4, R4                     // advance k by one element
    WORD $0x4E22D460              // FADD V0.4S, V3.4S, V2.4S   bestVal = a[0]+kv
    VEOR V1.B16, V1.B16, V1.B16    // bestIdx = 0
    VEOR V6.B16, V6.B16, V6.B16    // curIdx = 0
    MOVW $1, R7
    VDUP R7, V7.S4                 // V7 = dup(1), the curIdx increment

    SUB $1, R3, R6                 // R6 = n - 1 remaining candidates
    CBZ R6, minidxrows4_store      // n == 1: nothing to compare

minidxrows4_loop:
    ADD $4, R2                     // a[i]
    FMOVS (R2), F3
    VDUP V3.S[0], V3.S4            // broadcast a[i]
    VLD1 (R4), [V2.S4]             // k[i:i+4]
    ADD $4, R4
    WORD $0x4E22D462              // FADD V2.4S, V3.4S, V2.4S   c = a[i]+kv
    VADD V7.S4, V6.S4, V6.S4       // curIdx++ (before the selects read it)
    WORD $0x6EA2E404              // FCMGT V4.4S, V0.4S, V2.4S  mask = bestVal > c
    WORD $0x6EA41C40              // BIT V0.16B, V2.16B, V4.16B  bestVal = mask ? c : bestVal
    WORD $0x6EA41CC1              // BIT V1.16B, V6.16B, V4.16B  bestIdx = mask ? curIdx : bestIdx
    SUB $1, R6
    CBNZ R6, minidxrows4_loop

minidxrows4_store:
    CBZ R5, minidxrows4_write      // rev == 0: lanes already in row order
    // rev == 1: full lane reversal [0,1,2,3] -> [3,2,1,0] on both results.
    WORD $0x4EA00800              // REV64 V0.4S, V0.4S            [1,0,3,2]
    WORD $0x6E004000              // EXT V0.16B, V0.16B, V0.16B, #8  [3,2,1,0]
    WORD $0x4EA00821              // REV64 V1.4S, V1.4S
    WORD $0x6E014021              // EXT V1.16B, V1.16B, V1.16B, #8

minidxrows4_write:
    VST1 [V0.S4], (R0)             // vals[0:4]
    VST1 [V1.S4], (R1)             // idxs[0:4]
    RET
