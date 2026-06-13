//go:build arm64

#include "textflag.h"

// int8 SIMD kernels on ARM64 (NEON / ASIMD), with an SDOT (FEAT_DotProd) fast
// path for DotProduct.
//
// NEON instructions without a Go assembler mnemonic are hand-encoded as WORD
// with the decoded GNU form in the trailing comment; asmcheck_test.go
// cross-checks every WORD (arm64asm directly, or aarch64 objdump for SDOT, which
// arm64asm cannot decode). All encodings were verified with aarch64-linux-gnu-as
// + objdump. Scratch lives in R0-R10 and V0-V5; R28 (g), R18, R27, R16/R17, and
// the frame/link registers are left untouched (see CLAUDE.md).
//
// Saturating arithmetic (SQADD/SQSUB) clamps each byte lane to [-128, 127]; the
// scalar tail reproduces that with a widened add/sub and a CSEL clamp. The
// reductions widen to int16/int32 and accumulate in int32 lanes; int32 wrapping
// addition is associative, so the lane-parallel total matches the scalar
// reference modulo 2^32, and the int8 products never overflow their lane.

// func addSatNEON(dst, a, b []int8)
TEXT ·addSatNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR  $4, R3, R4               // R4 = n / 16
    CBZ  R4, addsat_remainder

addsat_loop16:
    VLD1.P 16(R1), [V0.B16]
    VLD1.P 16(R2), [V1.B16]
    WORD $0x4E210C02              // SQADD V2.16B, V0.16B, V1.16B
    VST1.P [V2.B16], 16(R0)
    SUB  $1, R4
    CBNZ R4, addsat_loop16

addsat_remainder:
    AND  $15, R3
    CBZ  R3, addsat_done
    MOVD $127, R8                 // clamp constants, hoisted out of the loop
    MOVD $-128, R9

addsat_scalar:
    MOVB (R1), R5                 // a (sign-extended)
    MOVB (R2), R6                 // b
    ADD  R6, R5, R5               // a + b (exact in 64-bit)
    CMP  R8, R5
    CSEL GT, R8, R5, R5           // clamp high
    CMP  R9, R5
    CSEL LT, R9, R5, R5           // clamp low
    MOVB R5, (R0)
    ADD  $1, R1
    ADD  $1, R2
    ADD  $1, R0
    SUB  $1, R3
    CBNZ R3, addsat_scalar

addsat_done:
    RET

// func subSatNEON(dst, a, b []int8)
TEXT ·subSatNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR  $4, R3, R4
    CBZ  R4, subsat_remainder

subsat_loop16:
    VLD1.P 16(R1), [V0.B16]
    VLD1.P 16(R2), [V1.B16]
    WORD $0x4E212C02              // SQSUB V2.16B, V0.16B, V1.16B
    VST1.P [V2.B16], 16(R0)
    SUB  $1, R4
    CBNZ R4, subsat_loop16

subsat_remainder:
    AND  $15, R3
    CBZ  R3, subsat_done
    MOVD $127, R8                 // clamp constants, hoisted out of the loop
    MOVD $-128, R9

subsat_scalar:
    MOVB (R1), R5
    MOVB (R2), R6
    SUB  R6, R5, R5               // a - b
    CMP  R8, R5
    CSEL GT, R8, R5, R5           // clamp high
    CMP  R9, R5
    CSEL LT, R9, R5, R5           // clamp low
    MOVB R5, (R0)
    ADD  $1, R1
    ADD  $1, R2
    ADD  $1, R0
    SUB  $1, R3
    CBNZ R3, subsat_scalar

subsat_done:
    RET

// func toI16NEON(dst []int16, src []int8)
// Sign-extends 16 int8 -> 16 int16 per iteration (SXTL low half, SXTL2 high half).
TEXT ·toI16NEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD src_base+24(FP), R1
    MOVD src_len+32(FP), R3

    LSR  $4, R3, R4               // R4 = n / 16
    CBZ  R4, toi16_remainder

toi16_loop16:
    VLD1.P 16(R1), [V0.B16]
    WORD $0x0F08A401             // SXTL  V1.8H, V0.8B   (low 8 -> int16)
    WORD $0x4F08A402             // SXTL2 V2.8H, V0.16B  (high 8 -> int16)
    VST1.P [V1.H8], 16(R0)
    VST1.P [V2.H8], 16(R0)
    SUB  $1, R4
    CBNZ R4, toi16_loop16

toi16_remainder:
    AND  $15, R3
    CBZ  R3, toi16_done

toi16_scalar:
    MOVB (R1), R5
    MOVH R5, (R0)
    ADD  $1, R1
    ADD  $2, R0
    SUB  $1, R3
    CBNZ R3, toi16_scalar

toi16_done:
    RET

// func toI32NEON(dst []int32, src []int8)
// Sign-extends 8 int8 -> 8 int32 per iteration: SXTL to int16, then SXTL/SXTL2
// to int32 for the low and high halves.
TEXT ·toI32NEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD src_base+24(FP), R1
    MOVD src_len+32(FP), R3

    LSR  $3, R3, R4               // R4 = n / 8
    CBZ  R4, toi32_remainder

toi32_loop8:
    VLD1.P 8(R1), [V0.B8]
    WORD $0x0F08A401             // SXTL  V1.8H, V0.8B   (8 int8 -> 8 int16)
    WORD $0x0F10A422             // SXTL  V2.4S, V1.4H   (low 4 -> int32)
    WORD $0x4F10A423             // SXTL2 V3.4S, V1.8H   (high 4 -> int32)
    VST1.P [V2.S4], 16(R0)
    VST1.P [V3.S4], 16(R0)
    SUB  $1, R4
    CBNZ R4, toi32_loop8

toi32_remainder:
    AND  $7, R3
    CBZ  R3, toi32_done

toi32_scalar:
    MOVB (R1), R5
    MOVW R5, (R0)
    ADD  $1, R1
    ADD  $4, R0
    SUB  $1, R3
    CBNZ R3, toi32_scalar

toi32_done:
    RET

// func sumNEON(a []int8) int32
// Pairwise widen-accumulate: SADDLP folds 16 int8 to 8 int16, SADALP accumulates
// those into a 4-lane int32 accumulator; ADDV folds the lanes and a scalar tail
// adds the (n mod 16) remainder.
TEXT ·sumNEON(SB), NOSPLIT, $0-28
    MOVD a_base+0(FP), R1
    MOVD a_len+8(FP), R3

    VEOR V2.B16, V2.B16, V2.B16   // int32 accumulator = 0
    LSR  $4, R3, R4               // R4 = n / 16
    CBZ  R4, sum_reduce

sum_loop16:
    VLD1.P 16(R1), [V0.B16]
    WORD $0x4E202801             // SADDLP V1.8H, V0.16B
    WORD $0x4E606822             // SADALP V2.4S, V1.8H
    SUB  $1, R4
    CBNZ R4, sum_loop16

sum_reduce:
    WORD $0x4EB1B843             // ADDV S3, V2.4S
    FMOVS F3, R5                  // R5 = vector total (low 32)

    AND  $15, R3
    CBZ  R3, sum_done

sum_scalar:
    MOVB.P 1(R1), R6
    ADDW R6, R5, R5               // 32-bit wrapping add
    SUB  $1, R3
    CBNZ R3, sum_scalar

sum_done:
    MOVW R5, ret+24(FP)
    RET

// func dotNEON(a, b []int8) int32
// Base-NEON dot product: SMULL/SMULL2 multiply 8-bit lanes into int16 products,
// SADALP pairwise-accumulates them into a 4-lane int32 accumulator; ADDV folds
// the lanes and a scalar tail adds the remaining products.
TEXT ·dotNEON(SB), NOSPLIT, $0-52
    MOVD a_base+0(FP), R1
    MOVD a_len+8(FP), R3
    MOVD b_base+24(FP), R2

    VEOR V4.B16, V4.B16, V4.B16   // int32 accumulator = 0
    LSR  $4, R3, R4
    CBZ  R4, dotn_reduce

dotn_loop16:
    VLD1.P 16(R1), [V0.B16]
    VLD1.P 16(R2), [V1.B16]
    WORD $0x0E21C002             // SMULL  V2.8H, V0.8B,  V1.8B
    WORD $0x4E21C003             // SMULL2 V3.8H, V0.16B, V1.16B
    WORD $0x4E606844             // SADALP V4.4S, V2.8H
    WORD $0x4E606864             // SADALP V4.4S, V3.8H
    SUB  $1, R4
    CBNZ R4, dotn_loop16

dotn_reduce:
    WORD $0x4EB1B885             // ADDV S5, V4.4S
    FMOVS F5, R5

    AND  $15, R3
    CBZ  R3, dotn_done

dotn_scalar:
    MOVB.P 1(R1), R6
    MOVB.P 1(R2), R7
    MUL  R7, R6, R6               // signed product (low 32 valid)
    ADDW R6, R5, R5
    SUB  $1, R3
    CBNZ R3, dotn_scalar

dotn_done:
    MOVW R5, ret+48(FP)
    RET

// func dotSDOT(a, b []int8) int32
// FEAT_DotProd fast path: SDOT accumulates 16 int8 multiply-adds into a 4-lane
// int32 accumulator per iteration; ADDV folds the lanes and a scalar tail adds
// the remainder. arm64asm cannot decode SDOT, so asmcheck cross-checks it via
// objdump.
TEXT ·dotSDOT(SB), NOSPLIT, $0-52
    MOVD a_base+0(FP), R1
    MOVD a_len+8(FP), R3
    MOVD b_base+24(FP), R2

    VEOR V2.B16, V2.B16, V2.B16   // int32 accumulator = 0
    LSR  $4, R3, R4
    CBZ  R4, dots_reduce

dots_loop16:
    VLD1.P 16(R1), [V0.B16]
    VLD1.P 16(R2), [V1.B16]
    WORD $0x4E819402             // SDOT V2.4S, V0.16B, V1.16B
    SUB  $1, R4
    CBNZ R4, dots_loop16

dots_reduce:
    WORD $0x4EB1B843             // ADDV S3, V2.4S
    FMOVS F3, R5

    AND  $15, R3
    CBZ  R3, dots_done

dots_scalar:
    MOVB.P 1(R1), R6
    MOVB.P 1(R2), R7
    MUL  R7, R6, R6
    ADDW R6, R5, R5
    SUB  $1, R3
    CBNZ R3, dots_scalar

dots_done:
    MOVW R5, ret+48(FP)
    RET

// func minMaxNEON(a []int8) (minVal, maxVal int8)
// Signed byte min and max in one pass: SMIN/SMAX fold 16-byte blocks into running
// accumulators, SMINV/SMAXV reduce each across its 16 lanes to a byte, and a
// scalar tail folds the (n mod 16) remainder. The dispatch gates n >= 16, so at
// least one full block exists.
TEXT ·minMaxNEON(SB), NOSPLIT, $0-26
    MOVD a_base+0(FP), R2
    MOVD a_len+8(FP), R3

    LSR  $4, R3, R4               // R4 = full 16-byte blocks (>=1)
    VLD1 (R2), [V0.B16]           // min acc = block 0 (no advance)
    VLD1.P 16(R2), [V1.B16]       // max acc = block 0 (advance to block 1)
    SUB  $1, R4
    CBZ  R4, mm_reduce

mm_loop:
    VLD1.P 16(R2), [V2.B16]
    WORD $0x4E226C00             // SMIN V0.16B, V0.16B, V2.16B
    WORD $0x4E226421             // SMAX V1.16B, V1.16B, V2.16B
    SUB  $1, R4
    CBNZ R4, mm_loop

mm_reduce:
    WORD $0x4E31A803             // SMINV B3, V0.16B
    WORD $0x4E30A824             // SMAXV B4, V1.16B
    FMOVS F3, R5                  // min byte (zero-extended)
    FMOVS F4, R6                  // max byte
    LSLW $24, R5, R5
    ASRW $24, R5, R5              // sign-extend low byte -> int32 min
    LSLW $24, R6, R6
    ASRW $24, R6, R6              // sign-extend low byte -> int32 max

    AND  $15, R3, R4              // tail count
    CBZ  R4, mm_done
    // R2 already points at &a[fullBlocks*16] after the loop.

mm_tail:
    MOVB.P 1(R2), R7              // r (sign-extended)
    CMPW R5, R7
    CSEL LT, R7, R5, R5           // R5 = min(r, R5)
    CMPW R6, R7
    CSEL GT, R7, R6, R6           // R6 = max(r, R6)
    SUB  $1, R4
    CBNZ R4, mm_tail

mm_done:
    MOVB R5, minVal+24(FP)
    MOVB R6, maxVal+25(FP)
    RET

// func minNEON(dst, a, b []int8)
// Element-wise signed min: SMIN folds 16-byte blocks; a signed CSEL scalar tail
// handles the (n mod 16) remainder.
TEXT ·minNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR  $4, R3, R4               // R4 = n / 16
    CBZ  R4, min_remainder

min_loop16:
    VLD1.P 16(R1), [V0.B16]
    VLD1.P 16(R2), [V1.B16]
    WORD $0x4E216C02              // SMIN V2.16B, V0.16B, V1.16B
    VST1.P [V2.B16], 16(R0)
    SUB  $1, R4
    CBNZ R4, min_loop16

min_remainder:
    AND  $15, R3
    CBZ  R3, min_done

min_scalar:
    MOVB (R1), R5                 // a (sign-extended)
    MOVB (R2), R6                 // b
    CMP  R6, R5
    CSEL LT, R5, R6, R5           // a < b ? a : b
    MOVB R5, (R0)
    ADD  $1, R1
    ADD  $1, R2
    ADD  $1, R0
    SUB  $1, R3
    CBNZ R3, min_scalar

min_done:
    RET

// func maxNEON(dst, a, b []int8)
// Element-wise signed max: SMAX folds 16-byte blocks; a signed CSEL scalar tail
// handles the (n mod 16) remainder.
TEXT ·maxNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR  $4, R3, R4
    CBZ  R4, max_remainder

max_loop16:
    VLD1.P 16(R1), [V0.B16]
    VLD1.P 16(R2), [V1.B16]
    WORD $0x4E216402              // SMAX V2.16B, V0.16B, V1.16B
    VST1.P [V2.B16], 16(R0)
    SUB  $1, R4
    CBNZ R4, max_loop16

max_remainder:
    AND  $15, R3
    CBZ  R3, max_done

max_scalar:
    MOVB (R1), R5
    MOVB (R2), R6
    CMP  R6, R5
    CSEL GT, R5, R6, R5           // a > b ? a : b
    MOVB R5, (R0)
    ADD  $1, R1
    ADD  $1, R2
    ADD  $1, R0
    SUB  $1, R3
    CBNZ R3, max_scalar

max_done:
    RET

// func clampNEON(dst, src []int8, lo, hi int8)
// Activation clip: DUP lo/hi into all 16 lanes, then SMAX(src, lo) and
// SMIN(., hi) per block. With lo > hi every element maps to hi. A signed CSEL
// scalar tail reproduces the max-then-min clamp on the (n mod 16) remainder.
TEXT ·clampNEON(SB), NOSPLIT, $0-50
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD src_base+24(FP), R1
    MOVB lo+48(FP), R5            // lo (sign-extended)
    MOVB hi+49(FP), R6            // hi
    WORD $0x4E010CA3              // DUP V3.16B, W5   (lo in all 16 lanes)
    WORD $0x4E010CC4              // DUP V4.16B, W6   (hi in all 16 lanes)

    LSR  $4, R3, R4
    CBZ  R4, clamp_remainder

clamp_loop16:
    VLD1.P 16(R1), [V0.B16]
    WORD $0x4E236400              // SMAX V0.16B, V0.16B, V3.16B   (max(src, lo))
    WORD $0x4E246C02              // SMIN V2.16B, V0.16B, V4.16B   (min(., hi))
    VST1.P [V2.B16], 16(R0)
    SUB  $1, R4
    CBNZ R4, clamp_loop16

clamp_remainder:
    AND  $15, R3
    CBZ  R3, clamp_done

clamp_scalar:
    MOVB (R1), R7
    CMP  R5, R7
    CSEL LT, R5, R7, R7           // src < lo ? lo : src   (max)
    CMP  R6, R7
    CSEL GT, R6, R7, R7           // val > hi ? hi : val   (min)
    MOVB R7, (R0)
    ADD  $1, R1
    ADD  $1, R0
    SUB  $1, R3
    CBNZ R3, clamp_scalar

clamp_done:
    RET

// func absNEON(dst, a []int8)
// Saturating absolute value: SQABS clamps abs(-128) to 127. A scalar tail negates
// if negative then clamps high on the (n mod 16) remainder.
TEXT ·absNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1

    LSR  $4, R3, R4
    CBZ  R4, abs_remainder

abs_loop16:
    VLD1.P 16(R1), [V0.B16]
    WORD $0x4E207802              // SQABS V2.16B, V0.16B
    VST1.P [V2.B16], 16(R0)
    SUB  $1, R4
    CBNZ R4, abs_loop16

abs_remainder:
    AND  $15, R3
    CBZ  R3, abs_done
    MOVD $127, R8                 // clamp constant, hoisted out of the loop

abs_scalar:
    MOVB (R1), R5                 // a (sign-extended)
    NEG  R5, R6                   // -a
    CMP  $0, R5
    CSEL LT, R6, R5, R5           // a < 0 ? -a : a   (|a|, can be 128)
    CMP  R8, R5
    CSEL GT, R8, R5, R5           // |a| > 127 ? 127 : |a|   (saturate -128 case)
    MOVB R5, (R0)
    ADD  $1, R1
    ADD  $1, R0
    SUB  $1, R3
    CBNZ R3, abs_scalar

abs_done:
    RET

// func negNEON(dst, a []int8)
// Saturating negation: SQNEG clamps neg(-128) to 127. A scalar tail negates then
// clamps high (-a is always >= -127, so the low bound never binds) on the
// (n mod 16) remainder.
TEXT ·negNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1

    LSR  $4, R3, R4
    CBZ  R4, neg_remainder

neg_loop16:
    VLD1.P 16(R1), [V0.B16]
    WORD $0x6E207802              // SQNEG V2.16B, V0.16B
    VST1.P [V2.B16], 16(R0)
    SUB  $1, R4
    CBNZ R4, neg_loop16

neg_remainder:
    AND  $15, R3
    CBZ  R3, neg_done
    MOVD $127, R8                 // clamp constant, hoisted out of the loop

neg_scalar:
    MOVB (R1), R5
    NEG  R5, R5                   // -a; -(-128) = 128
    CMP  R8, R5
    CSEL GT, R8, R5, R5           // -a > 127 ? 127 : -a   (saturate -128 case)
    MOVB R5, (R0)
    ADD  $1, R1
    ADD  $1, R0
    SUB  $1, R3
    CBNZ R3, neg_scalar

neg_done:
    RET

// func maxAbsNEON(a []int8) int
// Per-tensor abs-max for dynamic quantization: ABS maps each byte to its
// magnitude (abs(-128) -> 0x80, i.e. 128 read unsigned), UMAX folds 16-byte
// blocks into an unsigned-max accumulator, UMAXV reduces it to a byte, and a
// scalar tail folds the (n mod 16) remainder. The byte is read zero-extended,
// so the result lands in [0, 128].
TEXT ·maxAbsNEON(SB), NOSPLIT, $0-32
    MOVD a_base+0(FP), R1
    MOVD a_len+8(FP), R3

    VEOR V2.B16, V2.B16, V2.B16   // unsigned-max accumulator = 0
    LSR  $4, R3, R4               // R4 = n / 16
    CBZ  R4, maxabs_reduce

maxabs_loop16:
    VLD1.P 16(R1), [V0.B16]
    WORD $0x4E20B800             // ABS  V0.16B, V0.16B   (|a|; abs(-128)=0x80)
    WORD $0x6E206442             // UMAX V2.16B, V2.16B, V0.16B
    SUB  $1, R4
    CBNZ R4, maxabs_loop16

maxabs_reduce:
    WORD $0x6E30A843             // UMAXV B3, V2.16B
    FMOVS F3, R5                  // R5 = abs-max byte (zero-extended)
    AND  $0xFF, R5, R5            // defensively keep only the byte, [0, 128]

    AND  $15, R3
    CBZ  R3, maxabs_done

maxabs_scalar:
    MOVB (R1), R6                 // v (sign-extended)
    NEG  R6, R7                   // -v
    CMP  $0, R6
    CSEL LT, R7, R6, R6           // |v| = v < 0 ? -v : v   (can be 128)
    CMP  R5, R6
    CSEL HI, R6, R5, R5           // unsigned: |v| > max ? |v| : max
    ADD  $1, R1
    SUB  $1, R3
    CBNZ R3, maxabs_scalar

maxabs_done:
    MOVD R5, ret+24(FP)
    RET

// func absDiffNEON(dst, a, b []int8)
// Saturating absolute difference clamped to [0, 127]: SABD computes |a-b| as an
// unsigned byte in [0, 255], and an unsigned min with a broadcast 127 saturates
// it, so |127 - (-128)| = 255 maps to 127. A scalar tail subtracts, negates if
// negative, and clamps high on the (n mod 16) remainder.
TEXT ·absDiffNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    MOVD $127, R5                 // clamp constant, also the SABD-min broadcast
    WORD $0x4E010CA3             // DUP V3.16B, W5   (127 in all 16 lanes)

    LSR  $4, R3, R4
    CBZ  R4, absdiff_remainder

absdiff_loop16:
    VLD1.P 16(R1), [V0.B16]
    VLD1.P 16(R2), [V1.B16]
    WORD $0x4E217402             // SABD V2.16B, V0.16B, V1.16B   (|a-b|, 0..255)
    WORD $0x6E236C42             // UMIN V2.16B, V2.16B, V3.16B   (clamp to <= 127)
    VST1.P [V2.B16], 16(R0)
    SUB  $1, R4
    CBNZ R4, absdiff_loop16

absdiff_remainder:
    AND  $15, R3
    CBZ  R3, absdiff_done

absdiff_scalar:
    MOVB (R1), R6                 // a (sign-extended)
    MOVB (R2), R7                 // b
    SUB  R7, R6, R6               // a - b
    NEG  R6, R9                   // -(a - b)
    CMP  $0, R6
    CSEL LT, R9, R6, R6           // |a - b|  (0..255)
    CMP  R5, R6
    CSEL GT, R5, R6, R6           // |a - b| > 127 ? 127 : |a - b|
    MOVB R6, (R0)
    ADD  $1, R1
    ADD  $1, R2
    ADD  $1, R0
    SUB  $1, R3
    CBNZ R3, absdiff_scalar

absdiff_done:
    RET

// func addScalarSatNEON(dst, a []int8, s int8)
// Broadcast s into all 16 lanes (DUP) and add with signed saturation (SQADD). A
// CSEL clamp scalar tail handles the (n mod 16) remainder.
TEXT ·addScalarSatNEON(SB), NOSPLIT, $0-49
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVB s+48(FP), R5            // s (sign-extended)
    WORD $0x4E010CA3             // DUP V3.16B, W5   (s in all 16 lanes)

    LSR  $4, R3, R4
    CBZ  R4, addscalar_remainder

addscalar_loop16:
    VLD1.P 16(R1), [V0.B16]
    WORD $0x4E230C02            // SQADD V2.16B, V0.16B, V3.16B
    VST1.P [V2.B16], 16(R0)
    SUB  $1, R4
    CBNZ R4, addscalar_loop16

addscalar_remainder:
    AND  $15, R3
    CBZ  R3, addscalar_done
    MOVD $127, R8                // clamp constants, hoisted out of the loop
    MOVD $-128, R9

addscalar_scalar:
    MOVB (R1), R6                // a (sign-extended)
    ADD  R5, R6, R6              // a + s (exact in 64-bit)
    CMP  R8, R6
    CSEL GT, R8, R6, R6          // clamp high
    CMP  R9, R6
    CSEL LT, R9, R6, R6          // clamp low
    MOVB R6, (R0)
    ADD  $1, R1
    ADD  $1, R0
    SUB  $1, R3
    CBNZ R3, addscalar_scalar

addscalar_done:
    RET

// func subScalarSatNEON(dst, a []int8, s int8)
// Broadcast s into all 16 lanes (DUP) and subtract with signed saturation
// (SQSUB). A CSEL clamp scalar tail handles the (n mod 16) remainder.
TEXT ·subScalarSatNEON(SB), NOSPLIT, $0-49
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVB s+48(FP), R5            // s (sign-extended)
    WORD $0x4E010CA3             // DUP V3.16B, W5   (s in all 16 lanes)

    LSR  $4, R3, R4
    CBZ  R4, subscalar_remainder

subscalar_loop16:
    VLD1.P 16(R1), [V0.B16]
    WORD $0x4E232C02            // SQSUB V2.16B, V0.16B, V3.16B
    VST1.P [V2.B16], 16(R0)
    SUB  $1, R4
    CBNZ R4, subscalar_loop16

subscalar_remainder:
    AND  $15, R3
    CBZ  R3, subscalar_done
    MOVD $127, R8
    MOVD $-128, R9

subscalar_scalar:
    MOVB (R1), R6
    SUB  R5, R6, R6              // a - s
    CMP  R8, R6
    CSEL GT, R8, R6, R6          // clamp high
    CMP  R9, R6
    CSEL LT, R9, R6, R6          // clamp low
    MOVB R6, (R0)
    ADD  $1, R1
    ADD  $1, R0
    SUB  $1, R3
    CBNZ R3, subscalar_scalar

subscalar_done:
    RET
