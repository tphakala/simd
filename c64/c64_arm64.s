//go:build arm64

#include "textflag.h"

// ============================================================================
// ARM64 NEON IMPLEMENTATIONS FOR COMPLEX64
// ============================================================================
//
// complex64 layout: [real, imag] pairs in memory (8 bytes per complex64)
// NEON V register (128-bit) holds 4 float32 = 2 complex64
// Process 4 complex64 per iteration using 2 V registers
//
// Strategy: Deinterleave real/imag -> compute -> interleave back
//
// NEON opcodes for float32 (4S arrangement):
// FMUL Vd.4S, Vn.4S, Vm.4S: 0x6E20DC00 | (Vm << 16) | (Vn << 5) | Vd
// FADD Vd.4S, Vn.4S, Vm.4S: 0x4E20D400 | (Vm << 16) | (Vn << 5) | Vd
// FSUB Vd.4S, Vn.4S, Vm.4S: 0x4EA0D400 | (Vm << 16) | (Vn << 5) | Vd
// UZP1 Vd.4S, Vn.4S, Vm.4S: 0x4E801800 | (Vm << 16) | (Vn << 5) | Vd
// UZP2 Vd.4S, Vn.4S, Vm.4S: 0x4E805800 | (Vm << 16) | (Vn << 5) | Vd
// ZIP1 Vd.4S, Vn.4S, Vm.4S: 0x4E803800 | (Vm << 16) | (Vn << 5) | Vd
// ZIP2 Vd.4S, Vn.4S, Vm.4S: 0x4E807800 | (Vm << 16) | (Vn << 5) | Vd
// FNEG Vd.4S, Vn.4S:        0x6EA0F800 | (Vn << 5) | Vd
// FSQRT Vd.4S, Vn.4S:       0x6EA1F800 | (Vn << 5) | Vd

// func mulNEON(dst, a, b []complex64)
// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
TEXT ·mulNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0      // dst pointer
    MOVD dst_len+8(FP), R1       // length
    MOVD a_base+24(FP), R2       // a pointer
    MOVD b_base+48(FP), R3       // b pointer

    CBZ  R1, mul_neon_done

    // Check if we have at least 4 elements for vectorized loop
    CMP  $4, R1
    BLT  mul_neon_tail

mul_neon_loop4:
    // Load 4 complex64 from a: V0=[ar0,ai0,ar1,ai1], V1=[ar2,ai2,ar3,ai3]
    VLD1.P 16(R2), [V0.S4]
    VLD1.P 16(R2), [V1.S4]

    // Load 4 complex64 from b: V2=[br0,bi0,br1,bi1], V3=[br2,bi2,br3,bi3]
    VLD1.P 16(R3), [V2.S4]
    VLD1.P 16(R3), [V3.S4]

    // Deinterleave a: V4=[ar0,ar1,ar2,ar3], V5=[ai0,ai1,ai2,ai3]
    WORD $0x4E811804             // UZP1 V4.4S, V0.4S, V1.4S
    WORD $0x4E815805             // UZP2 V5.4S, V0.4S, V1.4S

    // Deinterleave b: V6=[br0,br1,br2,br3], V7=[bi0,bi1,bi2,bi3]
    WORD $0x4E831846             // UZP1 V6.4S, V2.4S, V3.4S
    WORD $0x4E835847             // UZP2 V7.4S, V2.4S, V3.4S

    // Compute products using vector operations
    // V8 = ar * br
    WORD $0x6E26DC88             // FMUL V8.4S, V4.4S, V6.4S
    // V9 = ai * bi
    WORD $0x6E27DCA9             // FMUL V9.4S, V5.4S, V7.4S
    // V10 = ar * bi
    WORD $0x6E27DC8A             // FMUL V10.4S, V4.4S, V7.4S
    // V11 = ai * br
    WORD $0x6E26DCAB             // FMUL V11.4S, V5.4S, V6.4S

    // result_real = ar*br - ai*bi (V12)
    WORD $0x4EA9D50C             // FSUB V12.4S, V8.4S, V9.4S
    // result_imag = ar*bi + ai*br (V13)
    WORD $0x4E2BD54D             // FADD V13.4S, V10.4S, V11.4S

    // Interleave results: V14=[r0,i0,r1,i1], V15=[r2,i2,r3,i3]
    WORD $0x4E8D398E             // ZIP1 V14.4S, V12.4S, V13.4S
    WORD $0x4E8D798F             // ZIP2 V15.4S, V12.4S, V13.4S

    // Store 4 complex64 results
    VST1.P [V14.S4], 16(R0)
    VST1.P [V15.S4], 16(R0)

    SUB  $4, R1
    CMP  $4, R1
    BGE  mul_neon_loop4

    // Handle remaining elements (0-3)
    CBZ  R1, mul_neon_done

mul_neon_tail:
    // Process 1 element at a time using scalar FP
    // Load complex64 as 2 float32
    FMOVS (R2), F0               // ar
    FMOVS 4(R2), F1              // ai
    FMOVS (R3), F2               // br
    FMOVS 4(R3), F3              // bi

    // Compute products
    FMULS F0, F2, F4             // ar * br
    FMULS F1, F3, F5             // ai * bi
    FMULS F0, F3, F6             // ar * bi
    FMULS F1, F2, F7             // ai * br

    FSUBS F5, F4, F4             // ar*br - ai*bi
    FADDS F6, F7, F5             // ar*bi + ai*br

    FMOVS F4, (R0)
    FMOVS F5, 4(R0)

    ADD  $8, R2
    ADD  $8, R3
    ADD  $8, R0
    SUB  $1, R1
    CBNZ R1, mul_neon_tail

mul_neon_done:
    RET

// func mulConjNEON(dst, a, b []complex64)
// a * conj(b) = (ar + ai*i)(br - bi*i) = (ar*br + ai*bi) + (ai*br - ar*bi)*i
TEXT ·mulConjNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2
    MOVD b_base+48(FP), R3

    CBZ  R1, mulconj_neon_done

    CMP  $4, R1
    BLT  mulconj_neon_tail

mulconj_neon_loop4:
    VLD1.P 16(R2), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    VLD1.P 16(R3), [V2.S4]
    VLD1.P 16(R3), [V3.S4]

    // Deinterleave a: V4=[ar], V5=[ai]
    WORD $0x4E811804             // UZP1 V4.4S, V0.4S, V1.4S
    WORD $0x4E815805             // UZP2 V5.4S, V0.4S, V1.4S

    // Deinterleave b: V6=[br], V7=[bi]
    WORD $0x4E831846             // UZP1 V6.4S, V2.4S, V3.4S
    WORD $0x4E835847             // UZP2 V7.4S, V2.4S, V3.4S

    // V8 = ar * br
    WORD $0x6E26DC88             // FMUL V8.4S, V4.4S, V6.4S
    // V9 = ai * bi
    WORD $0x6E27DCA9             // FMUL V9.4S, V5.4S, V7.4S
    // V10 = ai * br
    WORD $0x6E26DCAA             // FMUL V10.4S, V5.4S, V6.4S
    // V11 = ar * bi
    WORD $0x6E27DC8B             // FMUL V11.4S, V4.4S, V7.4S

    // result_real = ar*br + ai*bi (V12)
    WORD $0x4E29D50C             // FADD V12.4S, V8.4S, V9.4S
    // result_imag = ai*br - ar*bi (V13)
    WORD $0x4EABD54D             // FSUB V13.4S, V10.4S, V11.4S

    // Interleave results
    WORD $0x4E8D398E             // ZIP1 V14.4S, V12.4S, V13.4S
    WORD $0x4E8D798F             // ZIP2 V15.4S, V12.4S, V13.4S

    VST1.P [V14.S4], 16(R0)
    VST1.P [V15.S4], 16(R0)

    SUB  $4, R1
    CMP  $4, R1
    BGE  mulconj_neon_loop4

    CBZ  R1, mulconj_neon_done

mulconj_neon_tail:
    FMOVS (R2), F0
    FMOVS 4(R2), F1
    FMOVS (R3), F2
    FMOVS 4(R3), F3

    FMULS F0, F2, F4             // ar * br
    FMULS F1, F3, F5             // ai * bi
    FMULS F1, F2, F6             // ai * br
    FMULS F0, F3, F7             // ar * bi

    FADDS F4, F5, F4             // ar*br + ai*bi
    FSUBS F7, F6, F5             // ai*br - ar*bi

    FMOVS F4, (R0)
    FMOVS F5, 4(R0)

    ADD  $8, R2
    ADD  $8, R3
    ADD  $8, R0
    SUB  $1, R1
    CBNZ R1, mulconj_neon_tail

mulconj_neon_done:
    RET

// func scaleNEON(dst, a []complex64, s complex64)
TEXT ·scaleNEON(SB), NOSPLIT, $0-56
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2

    // Load scalar s and broadcast to vectors
    FMOVS s+48(FP), F20          // F20 = sr
    FMOVS s+52(FP), F21          // F21 = si

    // Broadcast to vectors
    VDUP V20.S[0], V20.S4        // V20 = [sr, sr, sr, sr]
    VDUP V21.S[0], V21.S4        // V21 = [si, si, si, si]

    CBZ  R1, scale_neon_done

    CMP  $4, R1
    BLT  scale_neon_tail

scale_neon_loop4:
    VLD1.P 16(R2), [V0.S4]
    VLD1.P 16(R2), [V1.S4]

    // Deinterleave: V4=[ar], V5=[ai]
    WORD $0x4E811804             // UZP1 V4.4S, V0.4S, V1.4S
    WORD $0x4E815805             // UZP2 V5.4S, V0.4S, V1.4S

    // V8 = ar * sr
    WORD $0x6E34DC88             // FMUL V8.4S, V4.4S, V20.4S
    // V9 = ai * si
    WORD $0x6E35DCA9             // FMUL V9.4S, V5.4S, V21.4S
    // V10 = ar * si
    WORD $0x6E35DC8A             // FMUL V10.4S, V4.4S, V21.4S
    // V11 = ai * sr
    WORD $0x6E34DCAB             // FMUL V11.4S, V5.4S, V20.4S

    // result_real = ar*sr - ai*si (V12)
    WORD $0x4EA9D50C             // FSUB V12.4S, V8.4S, V9.4S
    // result_imag = ar*si + ai*sr (V13)
    WORD $0x4E2BD54D             // FADD V13.4S, V10.4S, V11.4S

    // Interleave results
    WORD $0x4E8D398E             // ZIP1 V14.4S, V12.4S, V13.4S
    WORD $0x4E8D798F             // ZIP2 V15.4S, V12.4S, V13.4S

    VST1.P [V14.S4], 16(R0)
    VST1.P [V15.S4], 16(R0)

    SUB  $4, R1
    CMP  $4, R1
    BGE  scale_neon_loop4

    CBZ  R1, scale_neon_done

scale_neon_tail:
    FMOVS (R2), F0               // ar
    FMOVS 4(R2), F1              // ai

    FMULS F0, F20, F2            // ar * sr
    FMULS F1, F21, F3            // ai * si
    FMULS F0, F21, F4            // ar * si
    FMULS F1, F20, F5            // ai * sr

    FSUBS F3, F2, F2             // ar*sr - ai*si
    FADDS F4, F5, F3             // ar*si + ai*sr

    FMOVS F2, (R0)
    FMOVS F3, 4(R0)

    ADD  $8, R2
    ADD  $8, R0
    SUB  $1, R1
    CBNZ R1, scale_neon_tail

scale_neon_done:
    RET

// func addNEON(dst, a, b []complex64)
TEXT ·addNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2
    MOVD b_base+48(FP), R3

    CBZ  R1, add_neon_done

    CMP  $4, R1
    BLT  add_neon_tail

add_neon_loop4:
    VLD1.P 16(R2), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    VLD1.P 16(R3), [V2.S4]
    VLD1.P 16(R3), [V3.S4]

    // FADD V4.4S, V0.4S, V2.4S
    WORD $0x4E22D404
    // FADD V5.4S, V1.4S, V3.4S
    WORD $0x4E23D425

    VST1.P [V4.S4], 16(R0)
    VST1.P [V5.S4], 16(R0)

    SUB  $4, R1
    CMP  $4, R1
    BGE  add_neon_loop4

    CBZ  R1, add_neon_done

add_neon_tail:
    FMOVS (R2), F0
    FMOVS 4(R2), F1
    FMOVS (R3), F2
    FMOVS 4(R3), F3

    FADDS F0, F2, F4
    FADDS F1, F3, F5

    FMOVS F4, (R0)
    FMOVS F5, 4(R0)

    ADD  $8, R2
    ADD  $8, R3
    ADD  $8, R0
    SUB  $1, R1
    CBNZ R1, add_neon_tail

add_neon_done:
    RET

// func subNEON(dst, a, b []complex64)
TEXT ·subNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2
    MOVD b_base+48(FP), R3

    CBZ  R1, sub_neon_done

    CMP  $4, R1
    BLT  sub_neon_tail

sub_neon_loop4:
    VLD1.P 16(R2), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    VLD1.P 16(R3), [V2.S4]
    VLD1.P 16(R3), [V3.S4]

    // FSUB V4.4S, V0.4S, V2.4S
    WORD $0x4EA2D404
    // FSUB V5.4S, V1.4S, V3.4S
    WORD $0x4EA3D425

    VST1.P [V4.S4], 16(R0)
    VST1.P [V5.S4], 16(R0)

    SUB  $4, R1
    CMP  $4, R1
    BGE  sub_neon_loop4

    CBZ  R1, sub_neon_done

sub_neon_tail:
    FMOVS (R2), F0
    FMOVS 4(R2), F1
    FMOVS (R3), F2
    FMOVS 4(R3), F3

    FSUBS F2, F0, F4
    FSUBS F3, F1, F5

    FMOVS F4, (R0)
    FMOVS F5, 4(R0)

    ADD  $8, R2
    ADD  $8, R3
    ADD  $8, R0
    SUB  $1, R1
    CBNZ R1, sub_neon_tail

sub_neon_done:
    RET

// ============================================================================
// ABS - COMPLEX MAGNITUDE: |a + bi| = sqrt(a^2 + b^2)
// ============================================================================

// func absNEON(dst []float32, a []complex64)
TEXT ·absNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2

    CBZ  R1, abs_neon_done

    CMP  $4, R1
    BLT  abs_neon_tail

abs_neon_loop4:
    // Load 4 complex64
    VLD1.P 16(R2), [V0.S4]
    VLD1.P 16(R2), [V1.S4]

    // Deinterleave: V4=[r0,r1,r2,r3], V5=[i0,i1,i2,i3]
    WORD $0x4E811804             // UZP1 V4.4S, V0.4S, V1.4S
    WORD $0x4E815805             // UZP2 V5.4S, V0.4S, V1.4S

    // V6 = r^2
    WORD $0x6E24DC86             // FMUL V6.4S, V4.4S, V4.4S
    // V7 = i^2
    WORD $0x6E25DCA7             // FMUL V7.4S, V5.4S, V5.4S
    // V8 = r^2 + i^2
    WORD $0x4E27D4C8             // FADD V8.4S, V6.4S, V7.4S

    // V8 = sqrt(V8) using FSQRT.4S
    WORD $0x6EA1F908             // FSQRT V8.4S, V8.4S

    // Store 4 float32 results
    VST1.P [V8.S4], 16(R0)

    SUB  $4, R1
    CMP  $4, R1
    BGE  abs_neon_loop4

    CBZ  R1, abs_neon_done

abs_neon_tail:
    FMOVS (R2), F0               // r
    FMOVS 4(R2), F1              // i

    FMULS F0, F0, F2             // r^2
    FMULS F1, F1, F3             // i^2
    FADDS F2, F3, F4             // r^2 + i^2
    FSQRTS F4, F5                // sqrt

    FMOVS F5, (R0)

    ADD  $8, R2
    ADD  $4, R0
    SUB  $1, R1
    CBNZ R1, abs_neon_tail

abs_neon_done:
    RET

// ============================================================================
// ABSSQ - MAGNITUDE SQUARED: |a + bi|^2 = a^2 + b^2
// ============================================================================

// func absSqNEON(dst []float32, a []complex64)
TEXT ·absSqNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2

    CBZ  R1, abssq_neon_done

    CMP  $4, R1
    BLT  abssq_neon_tail

abssq_neon_loop4:
    VLD1.P 16(R2), [V0.S4]
    VLD1.P 16(R2), [V1.S4]

    // Deinterleave
    WORD $0x4E811804             // UZP1 V4.4S, V0.4S, V1.4S
    WORD $0x4E815805             // UZP2 V5.4S, V0.4S, V1.4S

    // V6 = r^2
    WORD $0x6E24DC86             // FMUL V6.4S, V4.4S, V4.4S
    // V7 = i^2
    WORD $0x6E25DCA7             // FMUL V7.4S, V5.4S, V5.4S
    // V8 = r^2 + i^2
    WORD $0x4E27D4C8             // FADD V8.4S, V6.4S, V7.4S

    // Store 4 float32 (no sqrt needed)
    VST1.P [V8.S4], 16(R0)

    SUB  $4, R1
    CMP  $4, R1
    BGE  abssq_neon_loop4

    CBZ  R1, abssq_neon_done

abssq_neon_tail:
    FMOVS (R2), F0
    FMOVS 4(R2), F1

    FMULS F0, F0, F2
    FMULS F1, F1, F3
    FADDS F2, F3, F4

    FMOVS F4, (R0)

    ADD  $8, R2
    ADD  $4, R0
    SUB  $1, R1
    CBNZ R1, abssq_neon_tail

abssq_neon_done:
    RET

// ============================================================================
// CONJ - COMPLEX CONJUGATE: conj(a + bi) = a - bi
// ============================================================================

// func conjNEON(dst, a []complex64)
TEXT ·conjNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD a_base+24(FP), R2

    CBZ  R1, conj_neon_done

    CMP  $4, R1
    BLT  conj_neon_tail

conj_neon_loop4:
    VLD1.P 16(R2), [V0.S4]
    VLD1.P 16(R2), [V1.S4]

    // Deinterleave: V4=[r], V5=[i]
    WORD $0x4E811804             // UZP1 V4.4S, V0.4S, V1.4S
    WORD $0x4E815805             // UZP2 V5.4S, V0.4S, V1.4S

    // Negate imaginary parts: FNEG V5.4S, V5.4S
    WORD $0x6EA0F8A5             // FNEG V5.4S, V5.4S

    // Interleave back: V6=[r0,-i0,r1,-i1], V7=[r2,-i2,r3,-i3]
    WORD $0x4E853886             // ZIP1 V6.4S, V4.4S, V5.4S
    WORD $0x4E857887             // ZIP2 V7.4S, V4.4S, V5.4S

    VST1.P [V6.S4], 16(R0)
    VST1.P [V7.S4], 16(R0)

    SUB  $4, R1
    CMP  $4, R1
    BGE  conj_neon_loop4

    CBZ  R1, conj_neon_done

conj_neon_tail:
    FMOVS (R2), F0               // r
    FMOVS 4(R2), F1              // i

    FNEGS F1, F1                 // -i

    FMOVS F0, (R0)
    FMOVS F1, 4(R0)

    ADD  $8, R2
    ADD  $8, R0
    SUB  $1, R1
    CBNZ R1, conj_neon_tail

conj_neon_done:
    RET

// ============================================================================
// FROMREAL - CONVERT REAL TO COMPLEX: complex(x, 0)
// ============================================================================

// func fromRealNEON(dst []complex64, src []float32)
TEXT ·fromRealNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R1
    MOVD src_base+24(FP), R2

    CBZ  R1, fromreal_neon_done

    CMP  $4, R1
    BLT  fromreal_neon_tail

    // Zero register for interleaving
    VEOR V15.B16, V15.B16, V15.B16

fromreal_neon_loop4:
    // Load 4 float32: V0=[r0,r1,r2,r3]
    VLD1.P 16(R2), [V0.S4]

    // Interleave with zeros: V1=[r0,0,r1,0], V2=[r2,0,r3,0]
    // ZIP1 V1.4S, V0.4S, V15.4S
    WORD $0x4E8F3801             // ZIP1 V1.4S, V0.4S, V15.4S
    // ZIP2 V2.4S, V0.4S, V15.4S
    WORD $0x4E8F7802             // ZIP2 V2.4S, V0.4S, V15.4S

    // Store 4 complex64 = 32 bytes
    VST1.P [V1.S4], 16(R0)
    VST1.P [V2.S4], 16(R0)

    SUB  $4, R1
    CMP  $4, R1
    BGE  fromreal_neon_loop4

    CBZ  R1, fromreal_neon_done

fromreal_neon_tail:
    FMOVS (R2), F0               // r
    FMOVS ZR, F1                 // 0

    FMOVS F0, (R0)
    FMOVS F1, 4(R0)

    ADD  $4, R2
    ADD  $8, R0
    SUB  $1, R1
    CBNZ R1, fromreal_neon_tail

fromreal_neon_done:
    RET
