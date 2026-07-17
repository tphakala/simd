//go:build arm64

#include "textflag.h"

// int16 (de)interleave on ARM64 (NEON / ASIMD).
//
// ZIP1/ZIP2/UZP1/UZP2 on .8H operands and the VLD1/VST1 .H8 loads/stores all
// move 16-bit lanes by bit pattern, so these kernels mirror the int32 .4S
// kernels in ../i32/i32_arm64.s with the lane width halved (8 lanes per 128-bit
// register instead of 4); only the scalar tails differ (16-bit MOVH). The
// ZIP/UZP instructions are hand-encoded as WORD because the Go assembler lacks
// mnemonics for them; the trailing comment is the decoded form and is
// cross-checked by asmcheck_test.go.

// func interleave2NEON(dst, a, b []int16)
// Interleaves: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], dst[3]=b[1], ...
TEXT ·interleave2NEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0    // dst pointer
    MOVD a_base+24(FP), R1     // a pointer
    MOVD a_len+32(FP), R3      // n = len(a)
    MOVD b_base+48(FP), R2     // b pointer

    // Process 8 pairs at a time
    LSR $3, R3, R4             // R4 = n / 8
    CBZ R4, interleave2_neon_remainder

interleave2_neon_loop8:
    VLD1.P 16(R1), [V0.H8]     // V0 = [a0, a1, a2, a3, a4, a5, a6, a7]
    VLD1.P 16(R2), [V1.H8]     // V1 = [b0, b1, b2, b3, b4, b5, b6, b7]
    WORD $0x4E413802           // ZIP1 V2.8H, V0.8H, V1.8H -> [a0,b0,a1,b1,a2,b2,a3,b3]
    WORD $0x4E417803           // ZIP2 V3.8H, V0.8H, V1.8H -> [a4,b4,a5,b5,a6,b6,a7,b7]
    VST1.P [V2.H8], 16(R0)     // Store [a0,b0,a1,b1,a2,b2,a3,b3]
    VST1.P [V3.H8], 16(R0)     // Store [a4,b4,a5,b5,a6,b6,a7,b7]
    SUB $1, R4
    CBNZ R4, interleave2_neon_loop8

interleave2_neon_remainder:
    AND $7, R3
    CBZ R3, interleave2_neon_done

interleave2_neon_loop1:
    MOVH (R1), R5
    MOVH (R2), R6
    MOVH R5, (R0)
    MOVH R6, 2(R0)
    ADD $2, R1
    ADD $2, R2
    ADD $4, R0
    SUB $1, R3
    CBNZ R3, interleave2_neon_loop1

interleave2_neon_done:
    RET

// func deinterleave2NEON(a, b, src []int16)
// Deinterleaves: a[0]=src[0], b[0]=src[1], a[1]=src[2], b[1]=src[3], ...
TEXT ·deinterleave2NEON(SB), NOSPLIT, $0-72
    MOVD a_base+0(FP), R0      // a pointer
    MOVD a_len+8(FP), R3       // n = len(a)
    MOVD b_base+24(FP), R1     // b pointer
    MOVD src_base+48(FP), R2   // src pointer

    // Process 8 pairs at a time
    LSR $3, R3, R4             // R4 = n / 8
    CBZ R4, deinterleave2_neon_remainder

deinterleave2_neon_loop8:
    VLD1.P 16(R2), [V0.H8]     // V0 = [a0,b0,a1,b1,a2,b2,a3,b3]
    VLD1.P 16(R2), [V1.H8]     // V1 = [a4,b4,a5,b5,a6,b6,a7,b7]
    WORD $0x4E411802           // UZP1 V2.8H, V0.8H, V1.8H -> [a0,a1,a2,a3,a4,a5,a6,a7]
    WORD $0x4E415803           // UZP2 V3.8H, V0.8H, V1.8H -> [b0,b1,b2,b3,b4,b5,b6,b7]
    VST1.P [V2.H8], 16(R0)     // Store a
    VST1.P [V3.H8], 16(R1)     // Store b
    SUB $1, R4
    CBNZ R4, deinterleave2_neon_loop8

deinterleave2_neon_remainder:
    AND $7, R3
    CBZ R3, deinterleave2_neon_done

deinterleave2_neon_loop1:
    MOVH (R2), R5
    MOVH 2(R2), R6
    MOVH R5, (R0)
    MOVH R6, (R1)
    ADD $4, R2
    ADD $2, R0
    ADD $2, R1
    SUB $1, R3
    CBNZ R3, deinterleave2_neon_loop1

deinterleave2_neon_done:
    RET

// func dotNEON(a, b []int16) int32
// Handles mismatched slice lengths: uses min(len(a), len(b)).
//
// Widening int16 dot product. SMLAL/SMLAL2 each multiply four int16 pairs into
// int32 and accumulate, so one iteration retires 16 products into four
// independent accumulators; the four chains keep the multiply-accumulate
// latency off the critical path. VADD folds them, ADDV reduces, and an 8-wide
// block plus a scalar tail finish n mod 16 (short CELT bands are common, so the
// 8-wide block earns its keep).
//
// Accumulation wraps in int32, matching dotGo bit-for-bit: SMLAL wraps per lane
// and wrapping addition is associative, so the lane split and the ADDV
// reduction cannot change the result.
//
// The Go assembler has no mnemonic for any integer vector multiply (SMLAL and
// friends are all "unrecognized instruction"), so these are hand-encoded as
// WORD; the trailing comment is the decoded form and asmcheck_test.go
// cross-checks it against arm64asm.
TEXT ·dotNEON(SB), NOSPLIT, $0-52
    MOVD a_base+0(FP), R0      // a pointer
    MOVD a_len+8(FP), R2
    MOVD b_len+32(FP), R3
    CMP  R3, R2
    CSEL LT, R2, R3, R2        // R2 = n = min(len(a), len(b))
    MOVD b_base+24(FP), R1     // b pointer

    VEOR V16.B16, V16.B16, V16.B16
    VEOR V17.B16, V17.B16, V17.B16
    VEOR V18.B16, V18.B16, V18.B16
    VEOR V19.B16, V19.B16, V19.B16

    LSR  $4, R2, R4            // R4 = n / 16
    CBZ  R4, dot_neon_block8

dot_neon_loop16:
    VLD1.P 32(R0), [V0.H8, V1.H8]
    VLD1.P 32(R1), [V2.H8, V3.H8]
    WORD $0x0E628010           // SMLAL V16.4S, V0.4H, V2.4H
    WORD $0x4E628011           // SMLAL2 V17.4S, V0.8H, V2.8H
    WORD $0x0E638032           // SMLAL V18.4S, V1.4H, V3.4H
    WORD $0x4E638033           // SMLAL2 V19.4S, V1.8H, V3.8H
    SUB  $1, R4
    CBNZ R4, dot_neon_loop16

dot_neon_block8:
    AND  $15, R2, R3           // R3 = n mod 16
    TBZ  $3, R3, dot_neon_fold // bit 3 clear => fewer than 8 left
    VLD1.P 16(R0), [V0.H8]
    VLD1.P 16(R1), [V2.H8]
    WORD $0x0E628010           // SMLAL V16.4S, V0.4H, V2.4H
    WORD $0x4E628011           // SMLAL2 V17.4S, V0.8H, V2.8H

dot_neon_fold:
    VADD V17.S4, V16.S4, V16.S4
    VADD V19.S4, V18.S4, V18.S4
    VADD V18.S4, V16.S4, V16.S4
    VADDV V16.S4, V16          // ADDV S16, V16.4S
    FMOVS F16, R5              // R5 = vector partial sum

    AND  $7, R2, R3            // R3 = n mod 8
    CBZ  R3, dot_neon_done

dot_neon_scalar:
    MOVH.P 2(R0), R6           // sign-extending 16-bit load
    MOVH.P 2(R1), R7
    MUL  R7, R6, R6
    ADDW R6, R5, R5            // 32-bit add: wraps like dotGo
    SUB  $1, R3
    CBNZ R3, dot_neon_scalar

dot_neon_done:
    MOVW R5, ret+48(FP)
    RET

// func xcorr4NEON(dst []int32, x, y []int16)
// Evaluates four consecutive correlation lags in one pass.
//
// This is the libopus xcorr_kernel shape: load eight x elements once, then
// multiply-accumulate them against four overlapping y windows at element
// offsets 0/1/2/3. Reusing the x load across four lags is the whole point of
// the op, and it is why this beats calling the dot kernel once per lag. Eight
// accumulators (two per lag) keep four independent SMLAL chains in flight.
//
// The four y loads are deliberately unaligned relative to each other: NEON
// loads have no alignment requirement, so the overlapping windows are just
// loads at +0/+2/+4/+6 bytes. That is cheaper than VEXT-ing the window along,
// and it is why no aligned-load instruction may ever be substituted here.
//
// Accumulation wraps in int32 exactly as dotNEON's does, so dst[k] is
// bit-identical to DotProduct(x, y[k:]).
//
// n = min(len(x), len(y)-3) is a safety net rather than a semantic: the
// dispatcher hands this kernel a y window of exactly len(x)+3 elements, so the
// clamp never fires in practice. It is here so that a wrapper bug becomes a
// wrong number instead of an out-of-bounds read.
TEXT ·xcorr4NEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD x_base+24(FP), R1
    MOVD x_len+32(FP), R2
    MOVD y_base+48(FP), R3
    MOVD y_len+56(FP), R4

    // Two accumulators per lag: V16/V17 lag0, V18/V19 lag1, V20/V21 lag2,
    // V22/V23 lag3.
    VEOR V16.B16, V16.B16, V16.B16
    VEOR V17.B16, V17.B16, V17.B16
    VEOR V18.B16, V18.B16, V18.B16
    VEOR V19.B16, V19.B16, V19.B16
    VEOR V20.B16, V20.B16, V20.B16
    VEOR V21.B16, V21.B16, V21.B16
    VEOR V22.B16, V22.B16, V22.B16
    VEOR V23.B16, V23.B16, V23.B16

    SUBS $3, R4, R4            // R4 = len(y) - 3
    BLE  xcorr4_neon_empty     // len(y) <= 3: no lag has a window
    CMP  R4, R2
    CSEL LT, R2, R4, R2        // R2 = n = min(len(x), len(y)-3)
    B    xcorr4_neon_blocks

xcorr4_neon_empty:
    MOVD $0, R2                // n = 0: fold zeroed accumulators, store zeros

xcorr4_neon_blocks:
    LSR  $3, R2, R5            // R5 = n / 8
    CBZ  R5, xcorr4_neon_fold

xcorr4_neon_loop8:
    VLD1.P 16(R1), [V0.H8]     // x[j..j+8), consumed by all four lags
    VLD1   (R3), [V1.H8]       // y[j+0 .. j+8)  lag 0
    ADD    $2, R3, R6
    VLD1   (R6), [V2.H8]       // y[j+1 .. j+9)  lag 1
    ADD    $2, R6, R6
    VLD1   (R6), [V3.H8]       // y[j+2 .. j+10) lag 2
    ADD    $2, R6, R6
    VLD1   (R6), [V4.H8]       // y[j+3 .. j+11) lag 3
    ADD    $16, R3             // advance y by 8 elements
    WORD $0x0E618010           // SMLAL V16.4S, V0.4H, V1.4H
    WORD $0x4E618011           // SMLAL2 V17.4S, V0.8H, V1.8H
    WORD $0x0E628012           // SMLAL V18.4S, V0.4H, V2.4H
    WORD $0x4E628013           // SMLAL2 V19.4S, V0.8H, V2.8H
    WORD $0x0E638014           // SMLAL V20.4S, V0.4H, V3.4H
    WORD $0x4E638015           // SMLAL2 V21.4S, V0.8H, V3.8H
    WORD $0x0E648016           // SMLAL V22.4S, V0.4H, V4.4H
    WORD $0x4E648017           // SMLAL2 V23.4S, V0.8H, V4.8H
    SUB  $1, R5
    CBNZ R5, xcorr4_neon_loop8

xcorr4_neon_fold:
    VADD V17.S4, V16.S4, V16.S4
    VADD V19.S4, V18.S4, V18.S4
    VADD V21.S4, V20.S4, V20.S4
    VADD V23.S4, V22.S4, V22.S4
    VADDV V16.S4, V16          // ADDV S16, V16.4S
    VADDV V18.S4, V18          // ADDV S18, V18.4S
    VADDV V20.S4, V20          // ADDV S20, V20.4S
    VADDV V22.S4, V22          // ADDV S22, V22.4S
    FMOVS F16, R7              // lag 0 partial sum
    FMOVS F18, R8              // lag 1
    FMOVS F20, R9              // lag 2
    FMOVS F22, R10             // lag 3

    AND  $7, R2, R11           // R11 = n mod 8
    CBZ  R11, xcorr4_neon_store

xcorr4_neon_scalar:
    MOVH.P 2(R1), R12          // x[j], sign-extended
    MOVH   0(R3), R13          // y[j+0]
    MUL    R12, R13, R13
    ADDW   R13, R7, R7
    MOVH   2(R3), R13          // y[j+1]
    MUL    R12, R13, R13
    ADDW   R13, R8, R8
    MOVH   4(R3), R13          // y[j+2]
    MUL    R12, R13, R13
    ADDW   R13, R9, R9
    MOVH   6(R3), R13          // y[j+3]
    MUL    R12, R13, R13
    ADDW   R13, R10, R10
    ADD    $2, R3
    SUB    $1, R11
    CBNZ   R11, xcorr4_neon_scalar

xcorr4_neon_store:
    MOVW R7, 0(R0)
    MOVW R8, 4(R0)
    MOVW R9, 8(R0)
    MOVW R10, 12(R0)
    RET

// Tier-3 element-wise and reduction kernels (NEON / ASIMD).
//
// All three receive pre-clamped slices from the public API, so unlike
// dotNEON there is no in-assembly length clamp anywhere below: dst_len (or
// a_len) is the trusted element count. SMULL/SRSHR/XTN/ABS/UMAXV have no Go
// assembler mnemonics and are hand-encoded as WORD with the decoded form in
// the trailing comment (cross-checked by asmcheck_test.go); VUMAX does have a
// mnemonic and uses it.

// func mulQ15NEON(dst, a, b []int16)
// Rounding Q15 multiply, 8 lanes per iteration: SMULL/SMULL2 widen the int16
// products to int32, SRSHR #15 is the rounding shift ((p + 2^14) >> 15,
// computed on a wider intermediate, and |p| <= 2^30 anyway, so the rounding
// add cannot overflow), and XTN/XTN2 narrow back to int16 by truncation,
// which is the wrap: the one product outside int16 range, (-32768)^2 ->
// +32768, lands as -32768, matching mulQ15Go.
//
// SQRDMULH would do all of this in one instruction and must never be used:
// it saturates exactly that pair to 32767 (see mulq15.go). The scalar tail
// does the same widen/round/narrow in a 64-bit GPR; the MOVH store keeps the
// low 16 bits, wrapping identically.
TEXT ·mulQ15NEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR  $3, R3, R4            // R4 = n / 8
    CBZ  R4, mulq15_neon_remainder

mulq15_neon_loop8:
    VLD1.P 16(R1), [V0.H8]
    VLD1.P 16(R2), [V1.H8]
    WORD $0x0E61C002           // SMULL V2.4S, V0.4H, V1.4H
    WORD $0x4E61C003           // SMULL2 V3.4S, V0.8H, V1.8H
    WORD $0x4F312442           // SRSHR V2.4S, V2.4S, #15
    WORD $0x4F312463           // SRSHR V3.4S, V3.4S, #15
    WORD $0x0E612844           // XTN V4.4H, V2.4S
    WORD $0x4E612864           // XTN2 V4.8H, V3.4S
    VST1.P [V4.H8], 16(R0)
    SUB  $1, R4
    CBNZ R4, mulq15_neon_loop8

mulq15_neon_remainder:
    AND  $7, R3
    CBZ  R3, mulq15_neon_done

mulq15_neon_scalar:
    MOVH.P 2(R1), R5           // a[i], sign-extended
    MOVH.P 2(R2), R6           // b[i], sign-extended
    MUL  R6, R5, R5            // 64-bit product, |p| <= 2^30
    ADD  $16384, R5            // + q15Round
    ASR  $15, R5               // rounding shift
    MOVH.P R5, 2(R0)           // low 16 bits: 32768 wraps to -32768
    SUB  $1, R3
    CBNZ R3, mulq15_neon_scalar

mulq15_neon_done:
    RET

// func absNEON(dst, a []int16)
// Wrapping absolute value, 8 lanes per iteration: vector ABS wraps at the
// type minimum (abs(-32768) = -32768 in a 16-bit lane), which is absGo's
// contract. The scalar tail computes |v| in a 64-bit GPR, where -(-32768) is
// +32768, and the MOVH store keeps the low 16 bits, wrapping it back to
// -32768 identically.
TEXT ·absNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1

    LSR  $3, R3, R4            // R4 = n / 8
    CBZ  R4, abs_neon_remainder

abs_neon_loop8:
    VLD1.P 16(R1), [V0.H8]
    WORD $0x4E60B800           // ABS V0.8H, V0.8H
    VST1.P [V0.H8], 16(R0)
    SUB  $1, R4
    CBNZ R4, abs_neon_loop8

abs_neon_remainder:
    AND  $7, R3
    CBZ  R3, abs_neon_done

abs_neon_scalar:
    MOVH.P 2(R1), R5           // v, sign-extended
    NEG  R5, R6                // -v
    CMP  $0, R5
    CSEL LT, R6, R5, R5        // |v| = v < 0 ? -v : v   (can be 32768)
    MOVH.P R5, 2(R0)           // low 16 bits: 32768 wraps to -32768
    SUB  $1, R3
    CBNZ R3, abs_neon_scalar

abs_neon_done:
    RET

// func maxAbsNEON(a []int16) int
// Per-frame abs-max (the headroom probe): ABS maps each lane to its magnitude
// (abs(-32768) -> 0x8000, i.e. 32768 read unsigned), VUMAX folds 8-lane
// blocks into an unsigned-max accumulator, UMAXV reduces it to one halfword,
// and a scalar tail folds the (n mod 8) remainder. FMOVS reads the S view of
// the H result with the upper 16 bits zero, which is why 0x8000 compares
// correctly as unsigned 32768; the result lands in [0, 32768].
TEXT ·maxAbsNEON(SB), NOSPLIT, $0-32
    MOVD a_base+0(FP), R1
    MOVD a_len+8(FP), R3

    VEOR V2.B16, V2.B16, V2.B16   // unsigned-max accumulator = 0
    LSR  $3, R3, R4               // R4 = n / 8
    CBZ  R4, maxabs_neon_reduce

maxabs_neon_loop8:
    VLD1.P 16(R1), [V0.H8]
    WORD $0x4E60B800             // ABS V0.8H, V0.8H
    VUMAX V0.H8, V2.H8, V2.H8    // unsigned max accumulate
    SUB  $1, R4
    CBNZ R4, maxabs_neon_loop8

maxabs_neon_reduce:
    WORD $0x6E70A841             // UMAXV H1, V2.8H
    FMOVS F1, R5                  // abs-max halfword, zero-extended

    AND  $7, R3
    CBZ  R3, maxabs_neon_done

maxabs_neon_scalar:
    MOVH.P 2(R1), R6              // v, sign-extended
    NEG  R6, R7                   // -v
    CMP  $0, R6
    CSEL LT, R7, R6, R6           // |v| = v < 0 ? -v : v   (can be 32768)
    CMP  R5, R6
    CSEL HI, R6, R5, R5           // unsigned: |v| > max ? |v| : max
    SUB  $1, R3
    CBNZ R3, maxabs_neon_scalar

maxabs_neon_done:
    MOVD R5, ret+24(FP)
    RET
