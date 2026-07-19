//go:build arm64

#include "textflag.h"

// int32 (de)interleave on ARM64 (NEON / ASIMD).
//
// ZIP1/ZIP2/UZP1/UZP2 on .4S operands and the VLD1/VST1 .S4 loads/stores all
// move 32-bit lanes by bit pattern, so these kernels are the int32 counterparts
// of interleave2NEON / deinterleave2NEON in ../f32/f32_arm64.s with identical
// vector bodies; only the scalar tails differ (integer MOVW instead of FMOVS).
// The ZIP/UZP instructions are hand-encoded as WORD because the Go assembler
// lacks mnemonics for them; the trailing comment is the decoded form and is
// cross-checked by asmcheck_test.go.

// func interleave2NEON(dst, a, b []int32)
// Interleaves: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], dst[3]=b[1], ...
TEXT ·interleave2NEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0    // dst pointer
    MOVD a_base+24(FP), R1     // a pointer
    MOVD a_len+32(FP), R3      // n = len(a)
    MOVD b_base+48(FP), R2     // b pointer

    // Process 4 pairs at a time
    LSR $2, R3, R4             // R4 = n / 4
    CBZ R4, interleave2_neon_remainder

interleave2_neon_loop4:
    VLD1.P 16(R1), [V0.S4]     // V0 = [a0, a1, a2, a3]
    VLD1.P 16(R2), [V1.S4]     // V1 = [b0, b1, b2, b3]
    WORD $0x4E813802           // ZIP1 V2.4S, V0.4S, V1.4S -> [a0, b0, a1, b1]
    WORD $0x4E817803           // ZIP2 V3.4S, V0.4S, V1.4S -> [a2, b2, a3, b3]
    VST1.P [V2.S4], 16(R0)     // Store [a0, b0, a1, b1]
    VST1.P [V3.S4], 16(R0)     // Store [a2, b2, a3, b3]
    SUB $1, R4
    CBNZ R4, interleave2_neon_loop4

interleave2_neon_remainder:
    AND $3, R3
    CBZ R3, interleave2_neon_done

interleave2_neon_loop1:
    MOVW (R1), R5
    MOVW (R2), R6
    MOVW R5, (R0)
    MOVW R6, 4(R0)
    ADD $4, R1
    ADD $4, R2
    ADD $8, R0
    SUB $1, R3
    CBNZ R3, interleave2_neon_loop1

interleave2_neon_done:
    RET

// func deinterleave2NEON(a, b, src []int32)
// Deinterleaves: a[0]=src[0], b[0]=src[1], a[1]=src[2], b[1]=src[3], ...
TEXT ·deinterleave2NEON(SB), NOSPLIT, $0-72
    MOVD a_base+0(FP), R0      // a pointer
    MOVD a_len+8(FP), R3       // n = len(a)
    MOVD b_base+24(FP), R1     // b pointer
    MOVD src_base+48(FP), R2   // src pointer

    // Process 4 pairs at a time
    LSR $2, R3, R4             // R4 = n / 4
    CBZ R4, deinterleave2_neon_remainder

deinterleave2_neon_loop4:
    VLD1.P 16(R2), [V0.S4]     // V0 = [a0, b0, a1, b1]
    VLD1.P 16(R2), [V1.S4]     // V1 = [a2, b2, a3, b3]
    WORD $0x4E811802           // UZP1 V2.4S, V0.4S, V1.4S -> [a0, a1, a2, a3]
    WORD $0x4E815803           // UZP2 V3.4S, V0.4S, V1.4S -> [b0, b1, b2, b3]
    VST1.P [V2.S4], 16(R0)     // Store a
    VST1.P [V3.S4], 16(R1)     // Store b
    SUB $1, R4
    CBNZ R4, deinterleave2_neon_loop4

deinterleave2_neon_remainder:
    AND $3, R3
    CBZ R3, deinterleave2_neon_done

deinterleave2_neon_loop1:
    MOVW (R2), R5
    MOVW 4(R2), R6
    MOVW R5, (R0)
    MOVW R6, (R1)
    ADD $8, R2
    ADD $4, R0
    ADD $4, R1
    SUB $1, R3
    CBNZ R3, deinterleave2_neon_loop1

deinterleave2_neon_done:
    RET

// Arithmetic and reduction kernels (NEON / ASIMD).
//
// These do integer ALU work on .4S vectors (4 int32/iter). Each vector lane is
// 32 bits, so the SIMD path wraps exactly like the int32 Go reference; the
// scalar tails use the W-register (32-bit) ALU forms (ADDW/SUBW/...) so they
// wrap identically. SMIN/SMAX/SMINV/SMAXV and vector ABS have no Go assembler
// mnemonics and are hand-encoded as WORD with the decoded GNU form in the
// trailing comment (cross-checked by asmcheck_test.go). The ADD/SUB vector
// WORDs below are hand-encoded too even though the assembler does accept the
// native VADD/VSUB spellings on .S4 operands (sumNEON uses one); they stay as
// verified WORD encodings rather than churn. Dispatched from i32_arm64.go
// gated on the NEON CPU feature, with the pure-Go reference as the fallback.

// func addNEON(dst, a, b []int32)
TEXT ·addNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $2, R3, R4
    CBZ R4, add_neon_remainder

add_neon_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    WORD $0x4EA18402           // ADD V2.4S, V0.4S, V1.4S
    VST1.P [V2.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, add_neon_loop4

add_neon_remainder:
    AND $3, R3
    CBZ R3, add_neon_done

add_neon_loop1:
    MOVW (R1), R5
    MOVW (R2), R6
    ADDW R6, R5, R5
    MOVW R5, (R0)
    ADD $4, R1
    ADD $4, R2
    ADD $4, R0
    SUB $1, R3
    CBNZ R3, add_neon_loop1

add_neon_done:
    RET

// func subNEON(dst, a, b []int32)
TEXT ·subNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVD b_base+48(FP), R2

    LSR $2, R3, R4
    CBZ R4, sub_neon_remainder

sub_neon_loop4:
    VLD1.P 16(R1), [V0.S4]
    VLD1.P 16(R2), [V1.S4]
    WORD $0x6EA18402           // SUB V2.4S, V0.4S, V1.4S
    VST1.P [V2.S4], 16(R0)
    SUB $1, R4
    CBNZ R4, sub_neon_loop4

sub_neon_remainder:
    AND $3, R3
    CBZ R3, sub_neon_done

sub_neon_loop1:
    MOVW (R1), R5
    MOVW (R2), R6
    SUBW R6, R5, R5
    MOVW R5, (R0)
    ADD $4, R1
    ADD $4, R2
    ADD $4, R0
    SUB $1, R3
    CBNZ R3, sub_neon_loop1

sub_neon_done:
    RET

// func minMaxNEON(res []int32) (minVal, maxVal int32)
// Signed int32 min and max over res in one pass. The dispatch gates len(res) >=
// 4, so at least one full 4-element (.4S) block exists: the min and max
// accumulators start from block 0 and fold the remaining full blocks with
// SMIN/SMAX, then SMINV/SMAXV reduce each accumulator across its 4 lanes to a
// scalar and a scalar tail folds the (n mod 4) remainder. SMIN/SMAX/SMINV/SMAXV
// have no Go assembler mnemonic, so they are hand-encoded WORD directives (the
// trailing comment is the decoded form, cross-checked by asmcheck_test.go).
// Every compare is signed, matching minMaxGo exactly.
TEXT ·minMaxNEON(SB), NOSPLIT, $0-32
    MOVD res_base+0(FP), R2
    MOVD res_len+8(FP), R3
    LSR  $2, R3, R4                  // R4 = full 4-element blocks (>=1)
    VLD1 (R2), [V0.S4]               // V0 = block 0 (min acc), no advance
    VLD1.P 16(R2), [V1.S4]           // V1 = block 0 (max acc), advance to block 1
    SUB  $1, R4                      // blocks remaining after block 0
    CBZ  R4, mm_neon_reduce          // single block: accumulators hold it; R2 at tail
mm_neon_loop:
    VLD1.P 16(R2), [V2.S4]           // load block + advance
    WORD $0x4EA26C00                 // SMIN V0.4S, V0.4S, V2.4S
    WORD $0x4EA26421                 // SMAX V1.4S, V1.4S, V2.4S
    SUB  $1, R4
    CBNZ R4, mm_neon_loop

mm_neon_reduce:
    WORD $0x4EB1A803                 // SMINV S3, V0.4S
    WORD $0x4EB0A824                 // SMAXV S4, V1.4S
    FMOVS F3, R5                     // R5 = running min (low 32 = int32)
    FMOVS F4, R6                     // R6 = running max (low 32 = int32)

    // scalar tail: (n mod 4) residuals (R2 already at &res[fullBlocks*4])
    AND  $3, R3, R4
    CBZ  R4, mm_neon_done
mm_neon_tail:
    MOVW.P 4(R2), R7                 // r (sign-extended; low 32 = int32)
    CMPW R5, R7                      // (R7 - R5), signed 32-bit
    CSEL LT, R7, R5, R5             // R5 = min(r, R5)
    CMPW R6, R7
    CSEL GT, R7, R6, R6             // R6 = max(r, R6)
    SUB  $1, R4
    CBNZ R4, mm_neon_tail

mm_neon_done:
    MOVW R5, minVal+24(FP)
    MOVW R6, maxVal+28(FP)
    RET

// func sumNEON(a []int32) int32
// Wrapping int32 sum: VADD folds 4-lane blocks into a vector accumulator,
// ADDV reduces it to a scalar, and a 32-bit scalar tail adds the (n mod 4)
// remainder. Every add wraps in a 32-bit lane, and wrapping addition is
// associative, so the lane split and reduction order are bit-identical to
// sumGo for every input, including forced overflow. The slice arrives
// pre-clamped from the public Sum, so a_len is the trusted element count.
TEXT ·sumNEON(SB), NOSPLIT, $0-28
    MOVD a_base+0(FP), R1
    MOVD a_len+8(FP), R3

    VEOR V0.B16, V0.B16, V0.B16
    LSR  $2, R3, R4            // R4 = n / 4
    CBZ  R4, sum_neon_reduce

sum_neon_loop4:
    VLD1.P 16(R1), [V1.S4]
    VADD V1.S4, V0.S4, V0.S4   // accumulate (wrapping)
    SUB  $1, R4
    CBNZ R4, sum_neon_loop4

sum_neon_reduce:
    VADDV V0.S4, V0            // ADDV S0, V0.4S
    FMOVS F0, R5               // vector partial sum (low 32 = int32)

    AND  $3, R3
    CBZ  R3, sum_neon_done

sum_neon_scalar:
    MOVW.P 4(R1), R6
    ADDW R6, R5, R5            // 32-bit add: wraps like sumGo
    SUB  $1, R3
    CBNZ R3, sum_neon_scalar

sum_neon_done:
    MOVW R5, ret+24(FP)
    RET

// func absNEON(dst, a []int32)
// Wrapping absolute value, 4 lanes per iteration: vector ABS wraps at the
// type minimum (abs(MinInt32) = MinInt32 in a 32-bit lane), which is absGo's
// contract. The scalar tail computes |v| in a 64-bit GPR, where -(MinInt32)
// is +2^31, and the MOVW store keeps the low 32 bits, wrapping it back to
// MinInt32 identically.
TEXT ·absNEON(SB), NOSPLIT, $0-48
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1

    LSR  $2, R3, R4            // R4 = n / 4
    CBZ  R4, abs_neon_remainder

abs_neon_loop4:
    VLD1.P 16(R1), [V0.S4]
    WORD $0x4EA0B801           // ABS V1.4S, V0.4S
    VST1.P [V1.S4], 16(R0)
    SUB  $1, R4
    CBNZ R4, abs_neon_loop4

abs_neon_remainder:
    AND  $3, R3
    CBZ  R3, abs_neon_done

abs_neon_scalar:
    MOVW.P 4(R1), R5           // v, sign-extended
    NEG  R5, R6                // -v
    CMP  $0, R5
    CSEL LT, R6, R5, R5        // |v| = v < 0 ? -v : v   (can be 2^31)
    MOVW.P R5, 4(R0)           // low 32 bits: 2^31 wraps to MinInt32
    SUB  $1, R3
    CBNZ R3, abs_neon_scalar

abs_neon_done:
    RET

// func negWhereNegNEON(dst, mag []int32, sign []float32)
// Branchless conditional negate, 4 lanes per iteration. SSHR V1.4S,#31 broadcasts
// each sign lane's IEEE-754 sign bit to a full-width int32 mask m (all-ones iff
// the sign bit is set, so -0.0/-Inf/-NaN negate); EOR then SUB apply (mag ^ m) -
// m, which is -mag when m = -1 (MinInt32 wraps to itself) and mag when m = 0.
// Bit-identical to negWhereNegGo. The scalar tail does the same in 64-bit GPRs:
// ASR $31 of the sign-extended sign word yields m = 0 or -1, and the MOVW store
// keeps the low 32 bits so -MinInt32 wraps back to MinInt32. The SSHR/EOR/SUB
// WORDs are cross-checked against arm64asm by TestArm64WordEncodings. Frame is 3
// slice headers: dst+0, mag+24, sign+48.
TEXT ·negWhereNegNEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD mag_base+24(FP), R1
    MOVD sign_base+48(FP), R2

    LSR  $2, R3, R4            // R4 = n / 4
    CBZ  R4, negwhereneg_neon_remainder

negwhereneg_neon_loop4:
    VLD1.P 16(R2), [V1.S4]     // sign
    WORD $0x4F210421           // SSHR V1.4S, V1.4S, #31   -> mask m
    VLD1.P 16(R1), [V0.S4]     // mag
    WORD $0x6E211C00           // EOR V0.16B, V0.16B, V1.16B  -> mag ^ m
    WORD $0x6EA18400           // SUB V0.4S, V0.4S, V1.4S     -> (mag ^ m) - m
    VST1.P [V0.S4], 16(R0)
    SUB  $1, R4
    CBNZ R4, negwhereneg_neon_loop4

negwhereneg_neon_remainder:
    AND  $3, R3
    CBZ  R3, negwhereneg_neon_done

negwhereneg_neon_scalar:
    MOVW.P 4(R2), R6           // sign bits, sign-extended
    ASR  $31, R6, R6           // m = sign >> 31 = 0 or -1
    MOVW.P 4(R1), R5           // mag, sign-extended
    EOR  R6, R5, R5            // mag ^ m
    SUB  R6, R5, R5            // (mag ^ m) - m
    MOVW.P R5, 4(R0)           // low 32 bits: -MinInt32 wraps to MinInt32
    SUB  $1, R3
    CBNZ R3, negwhereneg_neon_scalar

negwhereneg_neon_done:
    RET

// Fixed-point scale-by-scalar kernels (NEON / ASIMD).
//
// Unlike AVX2, NEON has a native 64-bit ARITHMETIC shift (SSHR .2D), so the Q31/
// Q15 scale is a clean widen-shift-narrow: SMULL/SMULL2 multiply the int32 lanes
// by the broadcast coefficient into int64 products (Q=0 takes lanes 0,1; Q=1 the
// SMULL2 form takes lanes 2,3), SSHR .2D arithmetically shifts each 64-bit product
// right by the fixed-point position, and XTN/XTN2 narrow the low 32 bits of each
// int64 back to int32 (truncation = the int32() wrap, so a=k=MinInt32's 2^62 >> 31
// = 2^31 lands as MinInt32). k is broadcast with DUP from a W register: MOVW
// sign-extends the int32 for Q31, MOVH the int16 for Q15, and both leave int64(k)
// in the register for the scalar tail (MUL then ASR then a MOVW store that keeps
// the low 32 bits). dst may alias a exactly: each block/lane reads a before it
// stores dst. SMULL/SMULL2/SSHR/XTN/XTN2/DUP have no Go assembler mnemonic and are
// hand-encoded WORD directives with the decoded form in the trailing comment
// (cross-checked against arm64asm by TestArm64WordEncodings). Frame is two slice
// headers plus the scalar k: dst+0, a+24, k+48.

// func scaleQ31NEON(dst, a []int32, k int32)
TEXT ·scaleQ31NEON(SB), NOSPLIT, $0-52
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVW k+48(FP), R2          // R2 = int64(k), sign-extended int32 (also tail k)
    WORD $0x4E040C41           // DUP V1.4S, W2   (k in all 4 int32 lanes)

    LSR  $2, R3, R4            // R4 = n / 4
    CBZ  R4, scaleq31_neon_remainder

scaleq31_neon_loop4:
    VLD1.P 16(R1), [V0.S4]     // a[i..i+3]
    WORD $0x0EA1C002           // SMULL V2.2D, V0.2S, V1.2S   (lanes 0,1 -> 2 int64)
    WORD $0x4EA1C003           // SMULL2 V3.2D, V0.4S, V1.4S  (lanes 2,3 -> 2 int64)
    WORD $0x4F610442           // SSHR V2.2D, V2.2D, #31
    WORD $0x4F610463           // SSHR V3.2D, V3.2D, #31
    WORD $0x0EA12844           // XTN V4.2S, V2.2D   (low 32 of results 0,1)
    WORD $0x4EA12864           // XTN2 V4.4S, V3.2D  (low 32 of results 2,3)
    VST1.P [V4.S4], 16(R0)
    SUB  $1, R4
    CBNZ R4, scaleq31_neon_loop4

scaleq31_neon_remainder:
    AND  $3, R3
    CBZ  R3, scaleq31_neon_done

scaleq31_neon_scalar:
    MOVW.P 4(R1), R5           // a[i], sign-extended to 64-bit
    MUL  R2, R5, R5            // a[i] * k (64-bit, |p| <= 2^62)
    ASR  $31, R5, R5           // arithmetic shift right 31
    MOVW.P R5, 4(R0)           // low 32 bits: wraps like int32()
    SUB  $1, R3
    CBNZ R3, scaleq31_neon_scalar

scaleq31_neon_done:
    RET

// func scaleQ15NEON(dst, a []int32, k int16)
// Identical widen-shift-narrow to scaleQ31NEON with a shift of 15. k is a signed
// int16, sign-extended by MOVH to int64(k); |k * a[i]| <= 2^46, well inside the
// int64 product.
TEXT ·scaleQ15NEON(SB), NOSPLIT, $0-50
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3
    MOVD a_base+24(FP), R1
    MOVH k+48(FP), R2          // R2 = int64(k), sign-extended int16 (also tail k)
    WORD $0x4E040C41           // DUP V1.4S, W2   (k in all 4 int32 lanes)

    LSR  $2, R3, R4            // R4 = n / 4
    CBZ  R4, scaleq15_neon_remainder

scaleq15_neon_loop4:
    VLD1.P 16(R1), [V0.S4]     // a[i..i+3]
    WORD $0x0EA1C002           // SMULL V2.2D, V0.2S, V1.2S   (lanes 0,1 -> 2 int64)
    WORD $0x4EA1C003           // SMULL2 V3.2D, V0.4S, V1.4S  (lanes 2,3 -> 2 int64)
    WORD $0x4F710442           // SSHR V2.2D, V2.2D, #15
    WORD $0x4F710463           // SSHR V3.2D, V3.2D, #15
    WORD $0x0EA12844           // XTN V4.2S, V2.2D   (low 32 of results 0,1)
    WORD $0x4EA12864           // XTN2 V4.4S, V3.2D  (low 32 of results 2,3)
    VST1.P [V4.S4], 16(R0)
    SUB  $1, R4
    CBNZ R4, scaleq15_neon_loop4

scaleq15_neon_remainder:
    AND  $3, R3
    CBZ  R3, scaleq15_neon_done

scaleq15_neon_scalar:
    MOVW.P 4(R1), R5           // a[i], sign-extended to 64-bit
    MUL  R2, R5, R5            // k * a[i] (64-bit, |p| <= 2^46)
    ASR  $15, R5, R5           // arithmetic shift right 15
    MOVW.P R5, 4(R0)           // low 32 bits: wraps like int32()
    SUB  $1, R3
    CBNZ R3, scaleq15_neon_scalar

scaleq15_neon_done:
    RET

// func butterflyNEON(lo, hi []int32)
// In-place radix-2 butterfly, 4 int32 per iteration: each block loads lo and hi,
// forms the wrapping sum (ADD .4S -> V2) and difference (SUB .4S -> V3) BEFORE
// storing either, then writes lo = lo+hi and hi = lo-hi. Both loads precede both
// stores, so each block of lo and hi is read in full before either is written;
// that ordering is what makes the in-place update safe (and why lo and hi must not
// overlap). The loads are plain VLD1 with a manual ADD $16 advance because each
// slice uses a single pointer register for both its load and its store, so a
// post-increment load (VLD1.P) would advance that pointer before the in-place store
// and misaddress it. The ADD/SUB .4S
// WORDs are the addNEON/subNEON encodings with the destination lane in V2 (sum)
// and V3 (diff); they are cross-checked against arm64asm by TestArm64WordEncodings.
// The scalar tail computes both results from the unmodified lo/hi in W registers
// (ADDW/SUBW) so it wraps identically. Frame is two slice headers: lo+0, hi+24.
TEXT ·butterflyNEON(SB), NOSPLIT, $0-48
    MOVD lo_base+0(FP), R0
    MOVD lo_len+8(FP), R3
    MOVD hi_base+24(FP), R1

    LSR  $2, R3, R4            // R4 = n / 4
    CBZ  R4, butterfly_neon_remainder

butterfly_neon_loop4:
    VLD1 (R0), [V0.S4]         // lo
    VLD1 (R1), [V1.S4]         // hi
    WORD $0x4EA18402           // ADD V2.4S, V0.4S, V1.4S   (sum = lo + hi)
    WORD $0x6EA18403           // SUB V3.4S, V0.4S, V1.4S   (diff = lo - hi)
    VST1 [V2.S4], (R0)         // lo = lo + hi
    VST1 [V3.S4], (R1)         // hi = lo - hi
    ADD  $16, R0
    ADD  $16, R1
    SUB  $1, R4
    CBNZ R4, butterfly_neon_loop4

butterfly_neon_remainder:
    AND  $3, R3
    CBZ  R3, butterfly_neon_done

butterfly_neon_loop1:
    MOVW (R0), R5             // lo
    MOVW (R1), R6             // hi
    ADDW R6, R5, R7           // R7 = lo + hi (wraps in 32-bit)
    SUBW R6, R5, R8           // R8 = lo - hi
    MOVW R7, (R0)             // lo = lo + hi
    MOVW R8, (R1)             // hi = lo - hi
    ADD  $4, R0
    ADD  $4, R1
    SUB  $1, R3
    CBNZ R3, butterfly_neon_loop1

butterfly_neon_done:
    RET

// func maxAbsNEON(a []int32) int32
// Peak magnitude with celtMaxabs32 semantics: the same signed int32 min/max
// reduction as minMaxNEON (VLD1 block 0 into both accumulators, WORD-encoded
// SMIN/SMAX fold, SMINV/SMAXV across-lane reduce into R5=min/R6=max via FMOVS,
// CSEL scalar tail), then a combine to max(maxVal, -minVal). NEG forms -minVal
// with the two's-complement wrap (-MinInt32 == MinInt32); the low 32 bits of the
// negate are correct regardless of the accumulator's upper bits (FMOVS
// zero-extends, the tail sign-extends), so the combine compares in 32-bit CMPW
// and the MOVW store keeps the low 32 bits. The signed CSEL GE then picks the
// larger of maxVal and -minVal. The SMIN/SMAX/SMINV/SMAXV WORDs are the
// minMaxNEON encodings verbatim (cross-checked by asmcheck_test.go). The dispatch
// gates len(a) >= 4, so at least one full 4-element block exists. Frame is one
// slice header plus the int32 return: a+0, ret+24.
TEXT ·maxAbsNEON(SB), NOSPLIT, $0-28
    MOVD a_base+0(FP), R2
    MOVD a_len+8(FP), R3
    LSR  $2, R3, R4                  // R4 = full 4-element blocks (>=1)
    VLD1 (R2), [V0.S4]               // V0 = block 0 (min acc), no advance
    VLD1.P 16(R2), [V1.S4]           // V1 = block 0 (max acc), advance to block 1
    SUB  $1, R4                      // blocks remaining after block 0
    CBZ  R4, maxabs_neon_reduce      // single block: accumulators hold it; R2 at tail
maxabs_neon_loop:
    VLD1.P 16(R2), [V2.S4]           // load block + advance
    WORD $0x4EA26C00                 // SMIN V0.4S, V0.4S, V2.4S
    WORD $0x4EA26421                 // SMAX V1.4S, V1.4S, V2.4S
    SUB  $1, R4
    CBNZ R4, maxabs_neon_loop

maxabs_neon_reduce:
    WORD $0x4EB1A803                 // SMINV S3, V0.4S
    WORD $0x4EB0A824                 // SMAXV S4, V1.4S
    FMOVS F3, R5                     // R5 = running min (low 32 = int32)
    FMOVS F4, R6                     // R6 = running max (low 32 = int32)

    // scalar tail: (n mod 4) residuals (R2 already at &a[fullBlocks*4])
    AND  $3, R3, R4
    CBZ  R4, maxabs_neon_combine
maxabs_neon_tail:
    MOVW.P 4(R2), R7                 // r (sign-extended; low 32 = int32)
    CMPW R5, R7                      // (R7 - R5), signed 32-bit
    CSEL LT, R7, R5, R5             // R5 = min(r, R5)
    CMPW R6, R7
    CSEL GT, R7, R6, R6             // R6 = max(r, R6)
    SUB  $1, R4
    CBNZ R4, maxabs_neon_tail

maxabs_neon_combine:
    NEG  R5, R7                      // R7 = -min (low 32 = wrapping int32 negate)
    CMPW R7, R6                      // (R6 - R7), signed 32-bit: max vs -min
    CSEL GE, R6, R7, R0             // R0 = max(R6, R7) = max(maxVal, -minVal)
    MOVW R0, ret+24(FP)
    RET

// func firValidQ15NEON(dst, x []int32, taps []int16)
// int32 valid convolution in correlation orientation with int16 Q15 taps,
// vectorized over the OUTPUT index: 4 outputs per iteration. For output block i
// the accumulator V16 starts at zero and, for each tap j, loads the sliding window
// x[i+j .. i+j+3] (one VLD1 that supplies the tap-j contribution for all 4 outputs
// at once, since output i+k reads lane k of the window), broadcasts taps[j] (DUP
// from W2, sign-extended by MOVH), forms the 4 Q15-TRUNCATED products with the
// exact scaleQ15NEON widen-shift-narrow (SMULL/SMULL2 to int64, SSHR .2D #15,
// XTN/XTN2 to the low 32 bits), and ADD .4S-accumulates them (wrapping int32). The
// window pointer R7 slides by 4 bytes per tap; the taps pointer R8 by 2. After all
// taps the 4 outputs are stored. The scalar-output tail runs the full inner tap
// loop per remaining output (MOVH/MOVW sign-extend, MUL, ASR #15, ADDW), so the
// per-product truncation is preserved and the result is bit-exact with
// firValidQ15Go. All SMULL/SMULL2/SSHR/XTN/XTN2/DUP WORDs are the scaleQ15NEON
// encodings verbatim (cross-checked by TestArm64WordEncodings). The dispatch gates
// n = len(dst) >= 4, and the kernel reads x only up to index n-1+len(taps)-1 <=
// len(x)-1, so there is no over-read. dst must not overlap x. Frame is dst+x+taps
// slice headers: dst+0, x+24, taps+48.
TEXT ·firValidQ15NEON(SB), NOSPLIT, $0-72
    MOVD dst_base+0(FP), R0
    MOVD dst_len+8(FP), R3         // n = number of outputs
    MOVD x_base+24(FP), R1         // x base = window base for output block 0
    MOVD taps_base+48(FP), R6      // taps base (constant)
    MOVD taps_len+56(FP), R5       // number of taps (>=1)

    LSR  $2, R3, R4               // R4 = n / 4 = full 4-output blocks
    CBZ  R4, fir_neon_tail

fir_neon_block:
    VEOR V16.B16, V16.B16, V16.B16 // acc = 0
    MOVD R1, R7                    // R7 = window pointer = block base
    MOVD R6, R8                    // R8 = taps pointer
    MOVD R5, R9                    // R9 = tap counter
fir_neon_tap:
    MOVH.P 2(R8), R2             // R2 = taps[j], sign-extended int16 (W2 for DUP)
    WORD $0x4E040C41            // DUP V1.4S, W2   (taps[j] in all 4 int32 lanes)
    VLD1 (R7), [V0.S4]          // window x[i+j .. i+j+3]
    ADD  $4, R7                  // slide window by 1 int32
    WORD $0x0EA1C002           // SMULL V2.2D, V0.2S, V1.2S   (lanes 0,1 -> 2 int64)
    WORD $0x4EA1C003           // SMULL2 V3.2D, V0.4S, V1.4S  (lanes 2,3 -> 2 int64)
    WORD $0x4F710442           // SSHR V2.2D, V2.2D, #15
    WORD $0x4F710463           // SSHR V3.2D, V3.2D, #15
    WORD $0x0EA12844           // XTN V4.2S, V2.2D   (low 32 of results 0,1)
    WORD $0x4EA12864           // XTN2 V4.4S, V3.2D  (low 32 of results 2,3)
    VADD V4.S4, V16.S4, V16.S4  // acc += 4 Q15-truncated products (wrapping)
    SUB  $1, R9
    CBNZ R9, fir_neon_tap
    VST1.P [V16.S4], 16(R0)      // store 4 outputs
    ADD  $16, R1                 // next block window base
    SUB  $1, R4
    CBNZ R4, fir_neon_block

fir_neon_tail:
    AND  $3, R3, R4              // R4 = n mod 4 = scalar-output tail count
    CBZ  R4, fir_neon_done
fir_neon_tail_out:
    MOVD $0, R11                // acc32 = 0
    MOVD R1, R7                 // R7 = window pointer for this output
    MOVD R6, R8                 // R8 = taps pointer
    MOVD R5, R9                 // R9 = tap counter
fir_neon_tail_tap:
    MOVH.P 2(R8), R10          // R10 = int64(taps[j]), sign-extended
    MOVW.P 4(R7), R12          // R12 = int64(x[i+j]), sign-extended
    MUL  R10, R12, R12         // taps[j] * x[i+j] (|p| <= 2^46)
    ASR  $15, R12, R12         // Q15 truncating arithmetic shift
    ADDW R12, R11, R11         // acc32 += low 32 (wrapping)
    SUB  $1, R9
    CBNZ R9, fir_neon_tail_tap
    MOVW.P R11, 4(R0)          // store output
    ADD  $4, R1                // next output window base
    SUB  $1, R4
    CBNZ R4, fir_neon_tail_out

fir_neon_done:
    RET
