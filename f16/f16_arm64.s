//go:build arm64

#include "textflag.h"

// ARM64 NEON FP16: 8 half-precision elements per 128-bit register
//
// FP16 SIMD requires FEAT_FP16 (available on Apple Silicon, Cortex-A55+)
//
// Key instructions used (8H = 8 half-precision):
//   FADD  Vd.8H, Vn.8H, Vm.8H  - FP16 vector add
//   FSUB  Vd.8H, Vn.8H, Vm.8H  - FP16 vector subtract
//   FMUL  Vd.8H, Vn.8H, Vm.8H  - FP16 vector multiply
//   FMLA  Vd.8H, Vn.8H, Vm.8H  - FP16 fused multiply-add
//   FMIN  Vd.8H, Vn.8H, Vm.8H  - FP16 minimum
//   FMAX  Vd.8H, Vn.8H, Vm.8H  - FP16 maximum
//   FABS  Vd.8H, Vn.8H         - FP16 absolute value
//   FNEG  Vd.8H, Vn.8H         - FP16 negate
//
// Conversion instructions:
//   FCVTL  Vd.4S, Vn.4H  - Convert lower 4 FP16 to 4 FP32
//   FCVTL2 Vd.4S, Vn.8H  - Convert upper 4 FP16 to 4 FP32
//   FCVTN  Vd.4H, Vn.4S  - Convert 4 FP32 to lower 4 FP16
//   FCVTN2 Vd.8H, Vn.4S  - Convert 4 FP32 to upper 4 FP16
//
// NOTE: All functions process only multiples of 8 elements.
//       Remainders are handled in Go by the dispatcher.

// func toFloat32SliceNEON(dst []float32, src []Float16)
// Converts FP16 slice to FP32 slice. Length must be multiple of 8.
TEXT ·toFloat32SliceNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0     // dst pointer
    MOVD dst_len+8(FP), R2      // length (multiple of 8)
    MOVD src_base+24(FP), R1    // src pointer

    LSR $3, R2, R3              // R3 = n / 8
    CBZ R3, cvt_to_f32_done

cvt_to_f32_loop8:
    VLD1 (R1), [V0.H8]          // Load 8 FP16 values
    ADD $16, R1

    // Convert lower 4 FP16 to 4 FP32
    WORD $0x0E217801            // FCVTL V1.4S, V0.4H

    // Convert upper 4 FP16 to 4 FP32
    WORD $0x4E217802            // FCVTL2 V2.4S, V0.8H

    // Store 8 FP32 values (32 bytes)
    VST1 [V1.S4], (R0)
    ADD $16, R0
    VST1 [V2.S4], (R0)
    ADD $16, R0

    SUB $1, R3
    CBNZ R3, cvt_to_f32_loop8

cvt_to_f32_done:
    RET

// func fromFloat32SliceNEON(dst []Float16, src []float32)
// Converts FP32 slice to FP16 slice. Length must be multiple of 8.
TEXT ·fromFloat32SliceNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0     // dst pointer
    MOVD dst_len+8(FP), R2      // length (multiple of 8)
    MOVD src_base+24(FP), R1    // src pointer

    LSR $3, R2, R3
    CBZ R3, cvt_from_f32_done

cvt_from_f32_loop8:
    // Load 8 FP32 values (32 bytes)
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R1), [V1.S4]

    // Convert 4 FP32 to lower 4 FP16
    WORD $0x0E216802            // FCVTN V2.4H, V0.4S

    // Convert 4 FP32 to upper 4 FP16 (appends to V2)
    WORD $0x4E216822            // FCVTN2 V2.8H, V1.4S

    // Store 8 FP16 values (16 bytes)
    VST1 [V2.H8], (R0)
    ADD $16, R0

    SUB $1, R3
    CBNZ R3, cvt_from_f32_loop8

cvt_from_f32_done:
    RET

// func dotProductNEON(a, b []Float16) float32
// DotProduct computes sum(a[i] * b[i]) with FP32 accumulation.
// Length must be multiple of 8.
TEXT ·dotProductNEON(SB), NOSPLIT, $0-52
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R2
    MOVD b_base+24(FP), R1

    // Zero accumulators (4 FP32 each)
    VEOR V4.B16, V4.B16, V4.B16
    VEOR V5.B16, V5.B16, V5.B16

    LSR $3, R2, R3
    CBZ R3, dot16_reduce

dot16_loop8:
    VLD1 (R0), [V0.H8]          // Load 8 FP16 from a
    ADD $16, R0
    VLD1 (R1), [V1.H8]          // Load 8 FP16 from b
    ADD $16, R1

    // Multiply in FP16: V2 = V0 * V1
    WORD $0x6E411C02            // FMUL V2.8H, V0.8H, V1.8H

    // Convert products to FP32 and accumulate
    WORD $0x0E217843            // FCVTL V3.4S, V2.4H (lower 4)
    WORD $0x4E217842            // FCVTL2 V2.4S, V2.8H (upper 4, reuse V2)

    // Accumulate in FP32
    WORD $0x4E23D484            // FADD V4.4S, V4.4S, V3.4S
    WORD $0x4E22D4A5            // FADD V5.4S, V5.4S, V2.4S

    SUB $1, R3
    CBNZ R3, dot16_loop8

    // Combine accumulators
    WORD $0x4E25D484            // FADD V4.4S, V4.4S, V5.4S

dot16_reduce:
    // Horizontal sum of V4.4S -> S4
    WORD $0x6E24D484            // FADDP V4.4S, V4.4S, V4.4S
    WORD $0x7E30D884            // FADDP S4, V4.2S
    FMOVS F4, ret+48(FP)
    RET

// func dotProductWideNEON(a, b []Float16) float32
// FP32-widened dot product: each FP16 lane is widened to FP32 BEFORE the
// multiply, so |a[i] * b[i]| > 65504 does not saturate. Length must be a
// multiple of 8 (caller guarantees via dispatch).
TEXT ·dotProductWideNEON(SB), NOSPLIT, $0-52
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R2
    MOVD b_base+24(FP), R1

    // Zero accumulators (4 FP32 each)
    VEOR V4.B16, V4.B16, V4.B16
    VEOR V5.B16, V5.B16, V5.B16

    LSR $3, R2, R3
    CBZ R3, dotF32_reduce

dotF32_loop8:
    VLD1 (R0), [V0.H8]          // Load 8 FP16 from a
    ADD $16, R0
    VLD1 (R1), [V1.H8]          // Load 8 FP16 from b
    ADD $16, R1

    // Widen to FP32 BEFORE multiplying. Each FCVTL/FCVTL2 expands 4 FP16
    // lanes into 4 FP32 lanes.
    WORD $0x0E217802            // FCVTL  V2.4S, V0.4H (a lower)
    WORD $0x4E217803            // FCVTL2 V3.4S, V0.8H (a upper)
    WORD $0x0E217826            // FCVTL  V6.4S, V1.4H (b lower)
    WORD $0x4E217827            // FCVTL2 V7.4S, V1.8H (b upper)

    // Fused multiply-add in FP32: V4 += V2*V6 ; V5 += V3*V7
    WORD $0x4E26CC44            // FMLA V4.4S, V2.4S, V6.4S
    WORD $0x4E27CC65            // FMLA V5.4S, V3.4S, V7.4S

    SUB $1, R3
    CBNZ R3, dotF32_loop8

    // Combine accumulators
    WORD $0x4E25D484            // FADD V4.4S, V4.4S, V5.4S

dotF32_reduce:
    // Horizontal sum of V4.4S -> S4
    WORD $0x6E24D484            // FADDP V4.4S, V4.4S, V4.4S
    WORD $0x7E30D884            // FADDP S4, V4.2S
    FMOVS F4, ret+48(FP)
    RET

// func addNEON(dst, a, b []Float16)
// Length must be multiple of 8.
TEXT ·addNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $3, R3, R4
    CBZ R4, add16_done

add16_loop8:
    VLD1 (R1), [V0.H8]
    ADD $16, R1
    VLD1 (R2), [V1.H8]
    ADD $16, R2

    // FADD V2.8H, V0.8H, V1.8H
    WORD $0x4E411402            // FADD V2.8H, V0.8H, V1.8H

    VST1 [V2.H8], (R0)
    ADD $16, R0

    SUB $1, R4
    CBNZ R4, add16_loop8

add16_done:
    RET

// func subNEON(dst, a, b []Float16)
// Length must be multiple of 8.
TEXT ·subNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $3, R3, R4
    CBZ R4, sub16_done

sub16_loop8:
    VLD1 (R1), [V0.H8]
    ADD $16, R1
    VLD1 (R2), [V1.H8]
    ADD $16, R2

    // FSUB V2.8H, V0.8H, V1.8H
    WORD $0x4EC11402            // FSUB V2.8H, V0.8H, V1.8H

    VST1 [V2.H8], (R0)
    ADD $16, R0

    SUB $1, R4
    CBNZ R4, sub16_loop8

sub16_done:
    RET

// func mulNEON(dst, a, b []Float16)
// Length must be multiple of 8.
TEXT ·mulNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $3, R3, R4
    CBZ R4, mul16_done

mul16_loop8:
    VLD1 (R1), [V0.H8]
    ADD $16, R1
    VLD1 (R2), [V1.H8]
    ADD $16, R2

    // FMUL V2.8H, V0.8H, V1.8H
    WORD $0x6E411C02            // FMUL V2.8H, V0.8H, V1.8H

    VST1 [V2.H8], (R0)
    ADD $16, R0

    SUB $1, R4
    CBNZ R4, mul16_loop8

mul16_done:
    RET

// func scaleNEON(dst, a []Float16, s Float16)
// Length must be multiple of 8.
TEXT ·scaleNEON(SB), NOSPLIT, $0-50
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVHU s+48(FP), R2          // scalar s

    // Broadcast s to V1.8H
    // Move scalar to low bits of V1, then duplicate across all lanes
    FMOVD R2, F1
    WORD $0x4E020421            // DUP V1.8H, V1.H[0] (Q=1 for 128-bit)

    LSR $3, R3, R4
    CBZ R4, scale16_done

scale16_loop8:
    VLD1 (R1), [V0.H8]
    ADD $16, R1

    // FMUL V2.8H, V0.8H, V1.8H
    WORD $0x6E411C02            // FMUL V2.8H, V0.8H, V1.8H

    VST1 [V2.H8], (R0)
    ADD $16, R0

    SUB $1, R4
    CBNZ R4, scale16_loop8

scale16_done:
    RET

// func fmaNEON(dst, a, b, c []Float16)
// FMA: dst = a * b + c. Length must be multiple of 8.
TEXT ·fmaNEON(SB), NOSPLIT, $0-96
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R4
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2
    MOVD c_base+72(FP), R3

    LSR $3, R4, R5
    CBZ R5, fma16_done

fma16_loop8:
    VLD1 (R1), [V0.H8]
    ADD $16, R1
    VLD1 (R2), [V1.H8]
    ADD $16, R2
    VLD1 (R3), [V2.H8]
    ADD $16, R3

    // FMLA V2.8H, V0.8H, V1.8H (V2 = V0*V1 + V2)
    WORD $0x4E410C02            // FMLA V2.8H, V0.8H, V1.8H

    VST1 [V2.H8], (R0)
    ADD $16, R0

    SUB $1, R5
    CBNZ R5, fma16_loop8

fma16_done:
    RET

// func sumNEON(a []Float16) float32
// Sum with FP32 accumulation. Length must be multiple of 8.
TEXT ·sumNEON(SB), NOSPLIT, $0-28
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R2

    // Zero accumulator (FP32)
    VEOR V4.B16, V4.B16, V4.B16

    LSR $3, R2, R3
    CBZ R3, sum16_reduce

sum16_loop8:
    VLD1 (R0), [V0.H8]
    ADD $16, R0

    // Convert to FP32 and accumulate
    WORD $0x0E217801            // FCVTL V1.4S, V0.4H
    WORD $0x4E217802            // FCVTL2 V2.4S, V0.8H
    WORD $0x4E21D484            // FADD V4.4S, V4.4S, V1.4S
    WORD $0x4E22D484            // FADD V4.4S, V4.4S, V2.4S

    SUB $1, R3
    CBNZ R3, sum16_loop8

sum16_reduce:
    WORD $0x6E24D484            // FADDP V4.4S, V4.4S, V4.4S
    WORD $0x7E30D884            // FADDP S4, V4.2S
    FMOVS F4, ret+24(FP)
    RET

// func absNEON(dst, a []Float16)
// Length must be multiple of 8.
TEXT ·absNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1

    LSR $3, R3, R4
    CBZ R4, abs16_done

abs16_loop8:
    VLD1 (R1), [V0.H8]
    ADD $16, R1

    // FABS V1.8H, V0.8H
    WORD $0x4EF8F801            // FABS V1.8H, V0.8H

    VST1 [V1.H8], (R0)
    ADD $16, R0

    SUB $1, R4
    CBNZ R4, abs16_loop8

abs16_done:
    RET

// func negNEON(dst, a []Float16)
// Length must be multiple of 8.
TEXT ·negNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1

    LSR $3, R3, R4
    CBZ R4, neg16_done

neg16_loop8:
    VLD1 (R1), [V0.H8]
    ADD $16, R1

    // FNEG V1.8H, V0.8H
    WORD $0x6EF8F801            // FNEG V1.8H, V0.8H

    VST1 [V1.H8], (R0)
    ADD $16, R0

    SUB $1, R4
    CBNZ R4, neg16_loop8

neg16_done:
    RET

// func reluNEON(dst, src []Float16)
// ReLU: max(0, x). Length must be multiple of 8.
TEXT ·reluNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1

    // Zero vector for FMAX
    VEOR V1.B16, V1.B16, V1.B16

    LSR $3, R3, R4
    CBZ R4, relu16_done

relu16_loop8:
    VLD1 (R1), [V0.H8]
    ADD $16, R1

    // FMAX V2.8H, V0.8H, V1.8H (V1 is zero)
    WORD $0x4E413402            // FMAX V2.8H, V0.8H, V1.8H

    VST1 [V2.H8], (R0)
    ADD $16, R0

    SUB $1, R4
    CBNZ R4, relu16_loop8

relu16_done:
    RET

// func minNEON(a []Float16) Float16
// Find minimum value. Length must be multiple of 8.
TEXT ·minNEON(SB), NOSPLIT, $0-26
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R2

    // Load first 8 elements as initial min
    VLD1 (R0), [V4.H8]
    ADD $16, R0

    LSR $3, R2, R3
    SUB $1, R3                  // Already processed first 8
    CBZ R3, min16_reduce

min16_loop8:
    VLD1 (R0), [V0.H8]
    ADD $16, R0

    // FMIN V4.8H, V4.8H, V0.8H
    WORD $0x4EC03484            // FMIN V4.8H, V4.8H, V0.8H

    SUB $1, R3
    CBNZ R3, min16_loop8

min16_reduce:
    // Reduce V4 to find minimum using pairwise min
    // FMINP V4.8H, V4.8H, V4.8H
    WORD $0x6EC43484            // FMINP V4.8H, V4.8H, V4.8H
    WORD $0x6EC43484            // FMINP V4.8H, V4.8H, V4.8H
    WORD $0x6EC43484            // FMINP V4.8H, V4.8H, V4.8H

    // Extract H0 to return
    FMOVD F4, R3
    MOVH R3, ret+24(FP)
    RET

// func maxNEON(a []Float16) Float16
// Find maximum value. Length must be multiple of 8.
TEXT ·maxNEON(SB), NOSPLIT, $0-26
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R2

    // Load first 8 elements as initial max
    VLD1 (R0), [V4.H8]
    ADD $16, R0

    LSR $3, R2, R3
    SUB $1, R3                  // Already processed first 8
    CBZ R3, max16_reduce

max16_loop8:
    VLD1 (R0), [V0.H8]
    ADD $16, R0

    // FMAX V4.8H, V4.8H, V0.8H
    WORD $0x4E403484            // FMAX V4.8H, V4.8H, V0.8H

    SUB $1, R3
    CBNZ R3, max16_loop8

max16_reduce:
    // Reduce V4 to find maximum using pairwise max
    // FMAXP V4.8H, V4.8H, V4.8H
    WORD $0x6E443484            // FMAXP V4.8H, V4.8H, V4.8H
    WORD $0x6E443484
    WORD $0x6E443484

    // Extract H0 to return
    FMOVD F4, R3
    MOVH R3, ret+24(FP)
    RET

// func divNEON(dst, a, b []Float16)
// Element-wise division. Length must be multiple of 8.
TEXT ·divNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $3, R3, R4
    CBZ R4, div16_done

div16_loop8:
    VLD1 (R1), [V0.H8]
    ADD $16, R1
    VLD1 (R2), [V1.H8]
    ADD $16, R2

    // FDIV V2.8H, V0.8H, V1.8H
    WORD $0x6E413C02            // FDIV V2.8H, V0.8H, V1.8H

    VST1 [V2.H8], (R0)
    ADD $16, R0

    SUB $1, R4
    CBNZ R4, div16_loop8

div16_done:
    RET

// func addScalarNEON(dst, a []Float16, s Float16)
// Add scalar to each element. Length must be multiple of 8.
TEXT ·addScalarNEON(SB), NOSPLIT, $0-50
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVHU s+48(FP), R2

    // Broadcast s to V1.8H
    FMOVD R2, F1
    WORD $0x4E020421            // DUP V1.8H, V1.H[0]

    LSR $3, R3, R4
    CBZ R4, addscalar16_done

addscalar16_loop8:
    VLD1 (R1), [V0.H8]
    ADD $16, R1

    // FADD V2.8H, V0.8H, V1.8H
    WORD $0x4E411402            // FADD V2.8H, V0.8H, V1.8H

    VST1 [V2.H8], (R0)
    ADD $16, R0

    SUB $1, R4
    CBNZ R4, addscalar16_loop8

addscalar16_done:
    RET

// func clampNEON(dst, a []Float16, minVal, maxVal Float16)
// Clamp each element to [minVal, maxVal]. Length must be multiple of 8.
TEXT ·clampNEON(SB), NOSPLIT, $0-52
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVHU minVal+48(FP), R4
    MOVHU maxVal+50(FP), R5

    // Broadcast minVal to V2.8H
    FMOVD R4, F2
    WORD $0x4E020442            // DUP V2.8H, V2.H[0]

    // Broadcast maxVal to V3.8H
    FMOVD R5, F3
    WORD $0x4E020463            // DUP V3.8H, V3.H[0]

    LSR $3, R3, R6
    CBZ R6, clamp16_done

clamp16_loop8:
    VLD1 (R1), [V0.H8]
    ADD $16, R1

    // FMAX V1.8H, V0.8H, V2.8H (clamp to min)
    WORD $0x4E423401            // FMAX V1.8H, V0.8H, V2.8H

    // FMIN V1.8H, V1.8H, V3.8H (clamp to max)
    WORD $0x4EC33421            // FMIN V1.8H, V1.8H, V3.8H

    VST1 [V1.H8], (R0)
    ADD $16, R0

    SUB $1, R6
    CBNZ R6, clamp16_loop8

clamp16_done:
    RET

// func sqrtNEON(dst, a []Float16)
// Element-wise square root. Length must be multiple of 8.
TEXT ·sqrtNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1

    LSR $3, R3, R4
    CBZ R4, sqrt16_done

sqrt16_loop8:
    VLD1 (R1), [V0.H8]
    ADD $16, R1

    // FSQRT V1.8H, V0.8H
    WORD $0x6EF9F801            // FSQRT V1.8H, V0.8H

    VST1 [V1.H8], (R0)
    ADD $16, R0

    SUB $1, R4
    CBNZ R4, sqrt16_loop8

sqrt16_done:
    RET

// func reciprocalNEON(dst, a []Float16)
// Element-wise reciprocal (1/x). Length must be multiple of 8.
TEXT ·reciprocalNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1

    // Create vector of 1.0 in FP16 (0x3C00)
    MOVD $0x3C003C003C003C00, R4
    FMOVD R4, F2
    MOVD $0x3C003C003C003C00, R4
    FMOVD R4, F3
    WORD $0x6E180462            // INS V2.D[1], V3.D[0]

    LSR $3, R3, R4
    CBZ R4, recip16_done

recip16_loop8:
    VLD1 (R1), [V0.H8]
    ADD $16, R1

    // FDIV V1.8H, V2.8H, V0.8H (1.0 / x)
    WORD $0x6E403C41            // FDIV V1.8H, V2.8H, V0.8H

    VST1 [V1.H8], (R0)
    ADD $16, R0

    SUB $1, R4
    CBNZ R4, recip16_loop8

recip16_done:
    RET

// func addScaledNEON(dst []Float16, alpha Float16, s []Float16)
// AXPY: dst[i] += alpha * s[i]. Length must be multiple of 8.
// Stack: dst(24) + alpha(2+6pad) + s(24) = 56 bytes
TEXT ·addScaledNEON(SB), NOSPLIT, $0-56
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVHU alpha+24(FP), R4
    MOVD s_base+32(FP), R1

    // Broadcast alpha to V3.8H
    FMOVD R4, F3
    WORD $0x4E020463            // DUP V3.8H, V3.H[0]

    LSR $3, R3, R5
    CBZ R5, axpy16_done

axpy16_loop8:
    VLD1 (R0), [V0.H8]          // Load dst
    VLD1 (R1), [V1.H8]          // Load s
    ADD $16, R1

    // V2 = alpha * s (FMUL V2.8H, V3.8H, V1.8H)
    WORD $0x6E411C62            // FMUL V2.8H, V3.8H, V1.8H

    // dst += V2 (FADD V0.8H, V0.8H, V2.8H)
    WORD $0x4E421400            // FADD V0.8H, V0.8H, V2.8H

    VST1 [V0.H8], (R0)
    ADD $16, R0

    SUB $1, R5
    CBNZ R5, axpy16_loop8

axpy16_done:
    RET

// func accumulateAddNEON(dst, src []Float16)
// dst += src. Length must be multiple of 8.
TEXT ·accumulateAddNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1

    LSR $3, R3, R4
    CBZ R4, accadd16_done

accadd16_loop8:
    VLD1 (R0), [V0.H8]          // Load dst
    VLD1 (R1), [V1.H8]          // Load src
    ADD $16, R1

    // FADD V0.8H, V0.8H, V1.8H
    WORD $0x4E411400            // FADD V0.8H, V0.8H, V1.8H

    VST1 [V0.H8], (R0)
    ADD $16, R0

    SUB $1, R4
    CBNZ R4, accadd16_loop8

accadd16_done:
    RET

// func sumSqDiffNEON(a, b []Float16) float32
// Returns sum((a[i]-b[i])^2). Each FP16 lane is widened to FP32 before the
// subtract and square, so large differences do not saturate. Length must be a
// multiple of 8 (caller guarantees via dispatch).
TEXT ·sumSqDiffNEON(SB), NOSPLIT, $0-52
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R2
    MOVD b_base+24(FP), R1

    VEOR V4.B16, V4.B16, V4.B16   // acc lower 4 lanes
    VEOR V5.B16, V5.B16, V5.B16   // acc upper 4 lanes

    LSR $3, R2, R3
    CBZ R3, ssd_reduce

ssd_loop8:
    VLD1 (R0), [V0.H8]            // 8 FP16 from a
    ADD $16, R0
    VLD1 (R1), [V1.H8]            // 8 FP16 from b
    ADD $16, R1

    WORD $0x0E217802             // FCVTL  V2.4S, V0.4H (a lower)
    WORD $0x4E217803             // FCVTL2 V3.4S, V0.8H (a upper)
    WORD $0x0E217826             // FCVTL  V6.4S, V1.4H (b lower)
    WORD $0x4E217827             // FCVTL2 V7.4S, V1.8H (b upper)

    WORD $0x4EA6D442             // FSUB V2.4S, V2.4S, V6.4S (diff lower)
    WORD $0x4EA7D463             // FSUB V3.4S, V3.4S, V7.4S (diff upper)

    WORD $0x4E22CC44             // FMLA V4.4S, V2.4S, V2.4S (acc += diff^2)
    WORD $0x4E23CC65             // FMLA V5.4S, V3.4S, V3.4S (acc += diff^2)

    SUB $1, R3
    CBNZ R3, ssd_loop8

    WORD $0x4E25D484             // FADD V4.4S, V4.4S, V5.4S

ssd_reduce:
    WORD $0x6E24D484             // FADDP V4.4S, V4.4S, V4.4S
    WORD $0x7E30D884             // FADDP S4, V4.2S
    FMOVS F4, ret+48(FP)
    RET

// func sumSqDevNEON(a []Float16, mean float32) float32
// Returns sum((a[i]-mean)^2), widened to FP32 before subtract/square.
// Length must be a multiple of 8 (caller guarantees via dispatch).
// Result is 8-byte aligned after the 28-byte args, so it lands at +32.
TEXT ·sumSqDevNEON(SB), NOSPLIT, $0-36
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R2

    FMOVS mean+24(FP), F1        // mean -> S1
    WORD $0x4E040421             // DUP V1.4S, V1.S[0] (broadcast mean)

    VEOR V4.B16, V4.B16, V4.B16  // acc lower
    VEOR V5.B16, V5.B16, V5.B16  // acc upper

    LSR $3, R2, R3
    CBZ R3, ssv_reduce

ssv_loop8:
    VLD1 (R0), [V0.H8]
    ADD $16, R0

    WORD $0x0E217802            // FCVTL  V2.4S, V0.4H (lower)
    WORD $0x4E217803            // FCVTL2 V3.4S, V0.8H (upper)

    WORD $0x4EA1D442            // FSUB V2.4S, V2.4S, V1.4S (x-mean lower)
    WORD $0x4EA1D463            // FSUB V3.4S, V3.4S, V1.4S (x-mean upper)

    WORD $0x4E22CC44           // FMLA V4.4S, V2.4S, V2.4S (acc += dev^2)
    WORD $0x4E23CC65           // FMLA V5.4S, V3.4S, V3.4S (acc += dev^2)

    SUB $1, R3
    CBNZ R3, ssv_loop8

    WORD $0x4E25D484           // FADD V4.4S, V4.4S, V5.4S

ssv_reduce:
    WORD $0x6E24D484           // FADDP V4.4S, V4.4S, V4.4S
    WORD $0x7E30D884           // FADDP S4, V4.2S
    FMOVS F4, ret+32(FP)
    RET

// func interleave2NEON(dst, a, b []Float16)
// dst[2i]=a[i], dst[2i+1]=b[i]. Processes a/b in blocks of 8 pairs (16 outputs
// per iteration). a_len must be a multiple of 8 (caller guarantees).
TEXT ·interleave2NEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD a_base+24(FP), R1
    MOVD a_len+32(FP), R3        // number of input pairs
    MOVD b_base+48(FP), R2

    LSR $3, R3, R4
    CBZ R4, il2_done

il2_loop8:
    VLD1 (R1), [V0.H8]           // 8 from a
    ADD $16, R1
    VLD1 (R2), [V1.H8]           // 8 from b
    ADD $16, R2

    WORD $0x4E413802            // ZIP1 V2.8H, V0.8H, V1.8H (a0 b0 .. a3 b3)
    WORD $0x4E417803            // ZIP2 V3.8H, V0.8H, V1.8H (a4 b4 .. a7 b7)

    VST1 [V2.H8], (R0)
    ADD $16, R0
    VST1 [V3.H8], (R0)
    ADD $16, R0

    SUB $1, R4
    CBNZ R4, il2_loop8

il2_done:
    RET

// func deinterleave2NEON(a, b, src []Float16)
// a[i]=src[2i], b[i]=src[2i+1]. Processes 8 outputs per channel per iteration
// (16 inputs). a_len must be a multiple of 8 (caller guarantees).
TEXT ·deinterleave2NEON(SB), NOSPLIT, $0-72
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R3         // outputs per channel
    MOVD b_base+24(FP), R1
    MOVD src_base+48(FP), R2

    LSR $3, R3, R4
    CBZ R4, dil2_done

dil2_loop8:
    VLD1 (R2), [V0.H8]           // src[0..7]
    ADD $16, R2
    VLD1 (R2), [V1.H8]           // src[8..15]
    ADD $16, R2

    WORD $0x4E411802            // UZP1 V2.8H, V0.8H, V1.8H (even lanes -> a)
    WORD $0x4E415803            // UZP2 V3.8H, V0.8H, V1.8H (odd lanes  -> b)

    VST1 [V2.H8], (R0)
    ADD $16, R0
    VST1 [V3.H8], (R1)
    ADD $16, R1

    SUB $1, R4
    CBNZ R4, dil2_loop8

dil2_done:
    RET

// func clampScaleNEON(dst, src []Float16, minF, maxF, scaleF float32)
// dst[i] = (clamp(src[i], minF, maxF) - minF) * scaleF, widened to FP32 for the
// full expression and rounded to FP16 once. Length must be a multiple of 8.
TEXT ·clampScaleNEON(SB), NOSPLIT, $0-60
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1

    FMOVS minF+48(FP), F1
    WORD $0x4E040421            // DUP V1.4S, V1.S[0] (minF)
    FMOVS maxF+52(FP), F2
    WORD $0x4E040442            // DUP V2.4S, V2.S[0] (maxF)
    FMOVS scaleF+56(FP), F3
    WORD $0x4E040463            // DUP V3.4S, V3.S[0] (scaleF)

    LSR $3, R3, R4
    CBZ R4, clampscale16_done

clampscale16_loop8:
    VLD1 (R1), [V0.H8]
    ADD $16, R1

    WORD $0x0E217804           // FCVTL  V4.4S, V0.4H (lower)
    WORD $0x4E217805           // FCVTL2 V5.4S, V0.8H (upper)

    WORD $0x4E21F484           // FMAX V4.4S, V4.4S, V1.4S (clamp low, lower)
    WORD $0x4EA2F484           // FMIN V4.4S, V4.4S, V2.4S (clamp high, lower)
    WORD $0x4E21F4A5           // FMAX V5.4S, V5.4S, V1.4S (clamp low, upper)
    WORD $0x4EA2F4A5           // FMIN V5.4S, V5.4S, V2.4S (clamp high, upper)

    WORD $0x4EA1D484           // FSUB V4.4S, V4.4S, V1.4S (subtract min, lower)
    WORD $0x4EA1D4A5           // FSUB V5.4S, V5.4S, V1.4S (subtract min, upper)

    WORD $0x6E23DC84           // FMUL V4.4S, V4.4S, V3.4S (scale lower)
    WORD $0x6E23DCA5           // FMUL V5.4S, V5.4S, V3.4S (scale upper)

    WORD $0x0E216886           // FCVTN  V6.4H, V4.4S (narrow lower)
    WORD $0x4E2168A6           // FCVTN2 V6.8H, V5.4S (narrow upper)

    VST1 [V6.H8], (R0)
    ADD $16, R0

    SUB $1, R4
    CBNZ R4, clampscale16_loop8

clampscale16_done:
    RET
