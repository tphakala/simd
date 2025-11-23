//go:build arm64

#include "textflag.h"

// ============================================================================
// ARM64 NEON IMPLEMENTATIONS - OPTIMIZED VERSION
// ============================================================================
//
// Optimization strategy:
// - Process 2 complex128 numbers per iteration (32 bytes each array)
// - Use UZP1/UZP2 to deinterleave real/imag into separate registers
// - Use true SIMD vector operations (FMUL.2D, FADD.2D, FSUB.2D)
// - Use ZIP1/ZIP2 to interleave results back
//
// complex128 layout in memory: [real0, imag0, real1, imag1, ...]
//
// NEON opcode formulas for float64 (2D arrangement):
// FMUL Vd.2D, Vn.2D, Vm.2D: 0x6E60DC00 | (Vm << 16) | (Vn << 5) | Vd
// FADD Vd.2D, Vn.2D, Vm.2D: 0x4E60D400 | (Vm << 16) | (Vn << 5) | Vd
// FSUB Vd.2D, Vn.2D, Vm.2D: 0x4EE0D400 | (Vm << 16) | (Vn << 5) | Vd
// UZP1 Vd.2D, Vn.2D, Vm.2D: 0x4EC01800 | (Vm << 16) | (Vn << 5) | Vd
// UZP2 Vd.2D, Vn.2D, Vm.2D: 0x4EC05800 | (Vm << 16) | (Vn << 5) | Vd
// ZIP1 Vd.2D, Vn.2D, Vm.2D: 0x4EC03800 | (Vm << 16) | (Vn << 5) | Vd
// ZIP2 Vd.2D, Vn.2D, Vm.2D: 0x4EC07800 | (Vm << 16) | (Vn << 5) | Vd

// func mulNEON(dst, a, b []complex128)
// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
TEXT ·mulNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0      // dst pointer
    MOVD dst_len+8(FP), R1       // length
    MOVD a_base+24(FP), R2       // a pointer
    MOVD b_base+48(FP), R3       // b pointer

    CBZ  R1, mul_neon_done

    // Check if we have at least 2 elements for vectorized loop
    CMP  $2, R1
    BLT  mul_neon_tail

mul_neon_loop2:
    // Load 2 complex128 from a: V0=[ar0,ai0], V1=[ar1,ai1]
    VLD1.P 16(R2), [V0.D2]
    VLD1.P 16(R2), [V1.D2]

    // Load 2 complex128 from b: V2=[br0,bi0], V3=[br1,bi1]
    VLD1.P 16(R3), [V2.D2]
    VLD1.P 16(R3), [V3.D2]

    // Deinterleave a: V4=[ar0,ar1], V5=[ai0,ai1]
    WORD $0x4EC11804             // UZP1 V4.2D, V0.2D, V1.2D
    WORD $0x4EC15805             // UZP2 V5.2D, V0.2D, V1.2D

    // Deinterleave b: V6=[br0,br1], V7=[bi0,bi1]
    WORD $0x4EC31846             // UZP1 V6.2D, V2.2D, V3.2D
    WORD $0x4EC35847             // UZP2 V7.2D, V2.2D, V3.2D

    // Compute products using vector operations
    // V8 = ar * br
    WORD $0x6E66DC88             // FMUL V8.2D, V4.2D, V6.2D
    // V9 = ai * bi
    WORD $0x6E67DCA9             // FMUL V9.2D, V5.2D, V7.2D
    // V10 = ar * bi
    WORD $0x6E67DC8A             // FMUL V10.2D, V4.2D, V7.2D
    // V11 = ai * br
    WORD $0x6E66DCAB             // FMUL V11.2D, V5.2D, V6.2D

    // result_real = ar*br - ai*bi (V12)
    WORD $0x4EE9D50C             // FSUB V12.2D, V8.2D, V9.2D
    // result_imag = ar*bi + ai*br (V13)
    WORD $0x4E6BD54D             // FADD V13.2D, V10.2D, V11.2D

    // Interleave results: V14=[r0,i0], V15=[r1,i1]
    WORD $0x4ECD398E             // ZIP1 V14.2D, V12.2D, V13.2D
    WORD $0x4ECD798F             // ZIP2 V15.2D, V12.2D, V13.2D

    // Store 2 complex128 results
    VST1.P [V14.D2], 16(R0)
    VST1.P [V15.D2], 16(R0)

    SUB  $2, R1
    CMP  $2, R1
    BGE  mul_neon_loop2

    // Handle remaining element (0 or 1)
    CBZ  R1, mul_neon_done

mul_neon_tail:
    // Process 1 element at a time (scalar fallback for remainder)
    VLD1 (R2), [V0.D2]           // V0 = [ar, ai]
    VLD1 (R3), [V1.D2]           // V1 = [br, bi]

    // Extract components
    VDUP V0.D[1], V2.D2          // V2 = [ai, ai], F2 = ai
    VDUP V1.D[1], V3.D2          // V3 = [bi, bi], F3 = bi

    // Compute products using scalar FP registers
    FMULD F0, F1, F4             // F4 = ar * br
    FMULD F2, F3, F5             // F5 = ai * bi
    FMULD F0, F3, F6             // F6 = ar * bi
    FMULD F2, F1, F7             // F7 = ai * br

    FSUBD F5, F4, F4             // F4 = ar*br - ai*bi
    FADDD F6, F7, F5             // F5 = ar*bi + ai*br

    FMOVD F4, (R0)
    FMOVD F5, 8(R0)

    ADD  $16, R2
    ADD  $16, R3
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, mul_neon_tail

mul_neon_done:
    RET

// func mulConjNEON(dst, a, b []complex128)
// a * conj(b) = (ar + ai*i)(br - bi*i) = (ar*br + ai*bi) + (ai*br - ar*bi)*i
TEXT ·mulConjNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2
    MOVD b_base+48(FP), R3

    CBZ  R1, mulconj_neon_done

    CMP  $2, R1
    BLT  mulconj_neon_tail

mulconj_neon_loop2:
    // Load 2 complex128 from a and b
    VLD1.P 16(R2), [V0.D2]
    VLD1.P 16(R2), [V1.D2]
    VLD1.P 16(R3), [V2.D2]
    VLD1.P 16(R3), [V3.D2]

    // Deinterleave a: V4=[ar0,ar1], V5=[ai0,ai1]
    WORD $0x4EC11804             // UZP1 V4.2D, V0.2D, V1.2D
    WORD $0x4EC15805             // UZP2 V5.2D, V0.2D, V1.2D

    // Deinterleave b: V6=[br0,br1], V7=[bi0,bi1]
    WORD $0x4EC31846             // UZP1 V6.2D, V2.2D, V3.2D
    WORD $0x4EC35847             // UZP2 V7.2D, V2.2D, V3.2D

    // Compute products
    WORD $0x6E66DC88             // FMUL V8.2D, V4.2D, V6.2D   (ar * br)
    WORD $0x6E67DCA9             // FMUL V9.2D, V5.2D, V7.2D   (ai * bi)
    WORD $0x6E66DCAA             // FMUL V10.2D, V5.2D, V6.2D  (ai * br)
    WORD $0x6E67DC8B             // FMUL V11.2D, V4.2D, V7.2D  (ar * bi)

    // result_real = ar*br + ai*bi (V12)
    WORD $0x4E69D50C             // FADD V12.2D, V8.2D, V9.2D
    // result_imag = ai*br - ar*bi (V13)
    WORD $0x4EEBD54D             // FSUB V13.2D, V10.2D, V11.2D

    // Interleave results
    WORD $0x4ECD398E             // ZIP1 V14.2D, V12.2D, V13.2D
    WORD $0x4ECD798F             // ZIP2 V15.2D, V12.2D, V13.2D

    VST1.P [V14.D2], 16(R0)
    VST1.P [V15.D2], 16(R0)

    SUB  $2, R1
    CMP  $2, R1
    BGE  mulconj_neon_loop2

    CBZ  R1, mulconj_neon_done

mulconj_neon_tail:
    VLD1 (R2), [V0.D2]
    VLD1 (R3), [V1.D2]

    VDUP V0.D[1], V2.D2
    VDUP V1.D[1], V3.D2

    FMULD F0, F1, F4             // ar * br
    FMULD F2, F3, F5             // ai * bi
    FMULD F2, F1, F6             // ai * br
    FMULD F0, F3, F7             // ar * bi

    FADDD F4, F5, F4             // ar*br + ai*bi
    FSUBD F7, F6, F5             // ai*br - ar*bi

    FMOVD F4, (R0)
    FMOVD F5, 8(R0)

    ADD  $16, R2
    ADD  $16, R3
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, mulconj_neon_tail

mulconj_neon_done:
    RET

// func scaleNEON(dst, a []complex128, s complex128)
// a * s = (ar + ai*i)(sr + si*i) = (ar*sr - ai*si) + (ar*si + ai*sr)*i
TEXT ·scaleNEON(SB), NOSPLIT, $0-64
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2

    // Load and broadcast scalar s to vectors
    FMOVD s+48(FP), F20          // F20 = sr
    FMOVD s+56(FP), F21          // F21 = si

    // Broadcast to vectors for SIMD ops
    VDUP V20.D[0], V20.D2        // V20 = [sr, sr]
    VDUP V21.D[0], V21.D2        // V21 = [si, si]

    CBZ  R1, scale_neon_done

    CMP  $2, R1
    BLT  scale_neon_tail

scale_neon_loop2:
    // Load 2 complex128 from a
    VLD1.P 16(R2), [V0.D2]       // V0 = [ar0, ai0]
    VLD1.P 16(R2), [V1.D2]       // V1 = [ar1, ai1]

    // Deinterleave: V4=[ar0,ar1], V5=[ai0,ai1]
    WORD $0x4EC11804             // UZP1 V4.2D, V0.2D, V1.2D
    WORD $0x4EC15805             // UZP2 V5.2D, V0.2D, V1.2D

    // Compute products with broadcast scalar
    // V8 = ar * sr
    WORD $0x6E74DC88             // FMUL V8.2D, V4.2D, V20.2D
    // V9 = ai * si
    WORD $0x6E75DCA9             // FMUL V9.2D, V5.2D, V21.2D
    // V10 = ar * si
    WORD $0x6E75DC8A             // FMUL V10.2D, V4.2D, V21.2D
    // V11 = ai * sr
    WORD $0x6E74DCAB             // FMUL V11.2D, V5.2D, V20.2D

    // result_real = ar*sr - ai*si (V12)
    WORD $0x4EE9D50C             // FSUB V12.2D, V8.2D, V9.2D
    // result_imag = ar*si + ai*sr (V13)
    WORD $0x4E6BD54D             // FADD V13.2D, V10.2D, V11.2D

    // Interleave results
    WORD $0x4ECD398E             // ZIP1 V14.2D, V12.2D, V13.2D
    WORD $0x4ECD798F             // ZIP2 V15.2D, V12.2D, V13.2D

    VST1.P [V14.D2], 16(R0)
    VST1.P [V15.D2], 16(R0)

    SUB  $2, R1
    CMP  $2, R1
    BGE  scale_neon_loop2

    CBZ  R1, scale_neon_done

scale_neon_tail:
    VLD1 (R2), [V0.D2]
    VDUP V0.D[1], V1.D2          // F1 = ai

    FMULD F0, F20, F2            // ar * sr
    FMULD F1, F21, F3            // ai * si
    FMULD F0, F21, F4            // ar * si
    FMULD F1, F20, F5            // ai * sr

    FSUBD F3, F2, F2             // ar*sr - ai*si
    FADDD F4, F5, F3             // ar*si + ai*sr

    FMOVD F2, (R0)
    FMOVD F3, 8(R0)

    ADD  $16, R2
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, scale_neon_tail

scale_neon_done:
    RET

// func addNEON(dst, a, b []complex128)
// Vector add - process 2 complex numbers per iteration
TEXT ·addNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2
    MOVD b_base+48(FP), R3

    CBZ  R1, add_neon_done

    CMP  $2, R1
    BLT  add_neon_tail

add_neon_loop2:
    // Load 2 complex128 from each (32 bytes total per array)
    VLD1.P 16(R2), [V0.D2]
    VLD1.P 16(R2), [V1.D2]
    VLD1.P 16(R3), [V2.D2]
    VLD1.P 16(R3), [V3.D2]

    // Add pairs
    WORD $0x4E62D404             // FADD V4.2D, V0.2D, V2.2D
    WORD $0x4E63D425             // FADD V5.2D, V1.2D, V3.2D

    // Store results
    VST1.P [V4.D2], 16(R0)
    VST1.P [V5.D2], 16(R0)

    SUB  $2, R1
    CMP  $2, R1
    BGE  add_neon_loop2

    CBZ  R1, add_neon_done

add_neon_tail:
    VLD1 (R2), [V0.D2]
    VLD1 (R3), [V1.D2]
    WORD $0x4E61D402             // FADD V2.2D, V0.2D, V1.2D
    VST1 [V2.D2], (R0)
    ADD  $16, R2
    ADD  $16, R3
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, add_neon_tail

add_neon_done:
    RET

// func subNEON(dst, a, b []complex128)
// Vector sub - process 2 complex numbers per iteration
TEXT ·subNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2
    MOVD b_base+48(FP), R3

    CBZ  R1, sub_neon_done

    CMP  $2, R1
    BLT  sub_neon_tail

sub_neon_loop2:
    VLD1.P 16(R2), [V0.D2]
    VLD1.P 16(R2), [V1.D2]
    VLD1.P 16(R3), [V2.D2]
    VLD1.P 16(R3), [V3.D2]

    WORD $0x4EE2D404             // FSUB V4.2D, V0.2D, V2.2D
    WORD $0x4EE3D425             // FSUB V5.2D, V1.2D, V3.2D

    VST1.P [V4.D2], 16(R0)
    VST1.P [V5.D2], 16(R0)

    SUB  $2, R1
    CMP  $2, R1
    BGE  sub_neon_loop2

    CBZ  R1, sub_neon_done

sub_neon_tail:
    VLD1 (R2), [V0.D2]
    VLD1 (R3), [V1.D2]
    WORD $0x4EE1D402             // FSUB V2.2D, V0.2D, V1.2D
    VST1 [V2.D2], (R0)
    ADD  $16, R2
    ADD  $16, R3
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, sub_neon_tail

sub_neon_done:
    RET

// ============================================================================
// ABS - COMPLEX MAGNITUDE: |a + bi| = sqrt(a² + b²)
// Process 2 elements per iteration
// ============================================================================

// func absNEON(dst []float64, a []complex128)
TEXT ·absNEON(SB), NOSPLIT, $0-56
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2

    CBZ  R1, abs_neon_done

    CMP  $2, R1
    BLT  abs_neon_tail

abs_neon_loop2:
    // Load 2 complex128: V0=[r0,i0], V1=[r1,i1]
    VLD1.P 16(R2), [V0.D2]
    VLD1.P 16(R2), [V1.D2]

    // Deinterleave: V4=[r0,r1], V5=[i0,i1]
    WORD $0x4EC11804             // UZP1 V4.2D, V0.2D, V1.2D
    WORD $0x4EC15805             // UZP2 V5.2D, V0.2D, V1.2D

    // V6 = r² (vector multiply)
    WORD $0x6E64DC86             // FMUL V6.2D, V4.2D, V4.2D
    // V7 = i² (vector multiply)
    WORD $0x6E65DCA7             // FMUL V7.2D, V5.2D, V5.2D
    // V8 = r² + i² (vector add)
    WORD $0x4E67D4C8             // FADD V8.2D, V6.2D, V7.2D

    // sqrt - need scalar since FSQRT.2D encoding is complex
    // Extract, sqrt, recombine
    VDUP V8.D[0], V9.D2
    VDUP V8.D[1], V10.D2
    FSQRTD F9, F9
    FSQRTD F10, F10

    // Store 2 float64 results
    FMOVD F9, (R0)
    FMOVD F10, 8(R0)
    ADD  $16, R0

    SUB  $2, R1
    CMP  $2, R1
    BGE  abs_neon_loop2

    CBZ  R1, abs_neon_done

abs_neon_tail:
    VLD1 (R2), [V0.D2]
    VDUP V0.D[1], V1.D2

    FMULD F0, F0, F2
    FMULD F1, F1, F3
    FADDD F2, F3, F4
    FSQRTD F4, F5

    FMOVD F5, (R0)

    ADD  $16, R2
    ADD  $8, R0
    SUB  $1, R1
    CBNZ R1, abs_neon_tail

abs_neon_done:
    RET

// ============================================================================
// ABSSQ - MAGNITUDE SQUARED: |a + bi|² = a² + b²
// Process 2 elements per iteration with true SIMD
// ============================================================================

// func absSqNEON(dst []float64, a []complex128)
TEXT ·absSqNEON(SB), NOSPLIT, $0-56
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2

    CBZ  R1, abssq_neon_done

    CMP  $2, R1
    BLT  abssq_neon_tail

abssq_neon_loop2:
    // Load 2 complex128
    VLD1.P 16(R2), [V0.D2]
    VLD1.P 16(R2), [V1.D2]

    // Deinterleave: V4=[r0,r1], V5=[i0,i1]
    WORD $0x4EC11804             // UZP1 V4.2D, V0.2D, V1.2D
    WORD $0x4EC15805             // UZP2 V5.2D, V0.2D, V1.2D

    // V6 = r² (vector)
    WORD $0x6E64DC86             // FMUL V6.2D, V4.2D, V4.2D
    // V7 = i² (vector)
    WORD $0x6E65DCA7             // FMUL V7.2D, V5.2D, V5.2D
    // V8 = r² + i² (vector)
    WORD $0x4E67D4C8             // FADD V8.2D, V6.2D, V7.2D

    // Store 2 float64 results
    VST1 [V8.D2], (R0)
    ADD  $16, R0

    SUB  $2, R1
    CMP  $2, R1
    BGE  abssq_neon_loop2

    CBZ  R1, abssq_neon_done

abssq_neon_tail:
    VLD1 (R2), [V0.D2]
    VDUP V0.D[1], V1.D2

    FMULD F0, F0, F2
    FMULD F1, F1, F3
    FADDD F2, F3, F4

    FMOVD F4, (R0)

    ADD  $16, R2
    ADD  $8, R0
    SUB  $1, R1
    CBNZ R1, abssq_neon_tail

abssq_neon_done:
    RET

// ============================================================================
// CONJ - COMPLEX CONJUGATE: conj(a + bi) = a - bi
// Use FNEG on vector for imaginary parts
// ============================================================================

// func conjNEON(dst, a []complex128)
TEXT ·conjNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2

    CBZ  R1, conj_neon_done

    CMP  $2, R1
    BLT  conj_neon_tail

conj_neon_loop2:
    // Load 2 complex128
    VLD1.P 16(R2), [V0.D2]       // V0 = [r0, i0]
    VLD1.P 16(R2), [V1.D2]       // V1 = [r1, i1]

    // Deinterleave: V4=[r0,r1], V5=[i0,i1]
    WORD $0x4EC11804             // UZP1 V4.2D, V0.2D, V1.2D
    WORD $0x4EC15805             // UZP2 V5.2D, V0.2D, V1.2D

    // Negate imaginary parts using vector FNEG
    WORD $0x6EE0F8A5             // FNEG V5.2D, V5.2D

    // Interleave back: V6=[r0,-i0], V7=[r1,-i1]
    WORD $0x4EC53886             // ZIP1 V6.2D, V4.2D, V5.2D
    WORD $0x4EC57887             // ZIP2 V7.2D, V4.2D, V5.2D

    VST1.P [V6.D2], 16(R0)
    VST1.P [V7.D2], 16(R0)

    SUB  $2, R1
    CMP  $2, R1
    BGE  conj_neon_loop2

    CBZ  R1, conj_neon_done

conj_neon_tail:
    VLD1 (R2), [V0.D2]
    VDUP V0.D[1], V1.D2

    FNEGD F1, F1

    FMOVD F0, (R0)
    FMOVD F1, 8(R0)

    ADD  $16, R2
    ADD  $16, R0
    SUB  $1, R1
    CBNZ R1, conj_neon_tail

conj_neon_done:
    RET
