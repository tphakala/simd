//go:build amd64

#include "textflag.h"

// int8 SIMD kernels on AMD64 (AVX2).
//
// All kernels gate on AVX2 in i8_amd64.go and run at least one full vector
// block (the dispatch guards the minimum length), with a scalar tail for the
// (n mod block) remainder. The Go assembler's 3-operand AVX order is dst-last:
// VPSUBSB a, b, c is c = b - a, and VPMADDWD a, b, c is c = madd(b, a). No
// hand-encoded directives are used; every mnemonic is one the Go assembler
// emits directly, so TestNoUncheckedAmd64Encodings stays clean.
//
// Saturating arithmetic (VPADDSB/VPSUBSB) clamps each byte lane to [-128, 127];
// the scalar tail reproduces that with a widened add/sub and an explicit clamp.
// The reductions (Sum, DotProduct) widen bytes to int16 with VPMOVSXBW and pair-
// reduce to int32 with VPMADDWD, accumulating in int32 lanes; since int32
// wrapping addition is associative, the lane-parallel total matches the scalar
// reference modulo 2^32. Intermediate products never overflow an int32 lane
// (|int8 * int8| <= 16384).

// func addSatAVX2(dst, a, b []int8)
TEXT ·addSatAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $5, AX                // AX = n / 32
    JZ   addsat_remainder

addsat_loop32:
    VMOVDQU (SI), Y0
    VMOVDQU (DI), Y1
    VPADDSB Y1, Y0, Y2         // Y2 = saturating(a + b)
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  addsat_loop32

addsat_remainder:
    ANDQ $31, CX
    JZ   addsat_done

addsat_scalar:
    MOVBLSX (SI), AX           // a (sign-extended to int32)
    MOVBLSX (DI), BX           // b
    ADDL BX, AX                // a + b in int32 (no overflow: |sum| <= 254)
    CMPL AX, $127
    JLE  addsat_chklo
    MOVL $127, AX
addsat_chklo:
    CMPL AX, $-128
    JGE  addsat_store
    MOVL $-128, AX
addsat_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DI
    INCQ DX
    DECQ CX
    JNZ  addsat_scalar

addsat_done:
    VZEROUPPER
    RET

// func subSatAVX2(dst, a, b []int8)
TEXT ·subSatAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $5, AX
    JZ   subsat_remainder

subsat_loop32:
    VMOVDQU (SI), Y0
    VMOVDQU (DI), Y1
    VPSUBSB Y1, Y0, Y2         // Y2 = saturating(a - b)
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  subsat_loop32

subsat_remainder:
    ANDQ $31, CX
    JZ   subsat_done

subsat_scalar:
    MOVBLSX (SI), AX
    MOVBLSX (DI), BX
    SUBL BX, AX                // a - b in int32 (|diff| <= 255)
    CMPL AX, $127
    JLE  subsat_chklo
    MOVL $127, AX
subsat_chklo:
    CMPL AX, $-128
    JGE  subsat_store
    MOVL $-128, AX
subsat_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DI
    INCQ DX
    DECQ CX
    JNZ  subsat_scalar

subsat_done:
    VZEROUPPER
    RET

// func toI16AVX2(dst []int16, src []int8)
// Sign-extends 16 int8 -> 16 int16 (256-bit) per iteration with VPMOVSXBW.
TEXT ·toI16AVX2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ src_base+24(FP), SI
    MOVQ src_len+32(FP), CX

    MOVQ CX, AX
    SHRQ $4, AX                // AX = n / 16
    JZ   toi16_remainder

toi16_loop16:
    VPMOVSXBW (SI), Y0         // 16 bytes -> 16 int16
    VMOVDQU Y0, (DX)
    ADDQ $16, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  toi16_loop16

toi16_remainder:
    ANDQ $15, CX
    JZ   toi16_done

toi16_scalar:
    MOVBLSX (SI), AX
    MOVW AX, (DX)
    INCQ SI
    ADDQ $2, DX
    DECQ CX
    JNZ  toi16_scalar

toi16_done:
    VZEROUPPER
    RET

// func toI32AVX2(dst []int32, src []int8)
// Sign-extends 8 int8 -> 8 int32 (256-bit) per iteration with VPMOVSXBD.
TEXT ·toI32AVX2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ src_base+24(FP), SI
    MOVQ src_len+32(FP), CX

    MOVQ CX, AX
    SHRQ $3, AX                // AX = n / 8
    JZ   toi32_remainder

toi32_loop8:
    VPMOVSXBD (SI), Y0         // 8 bytes -> 8 int32
    VMOVDQU Y0, (DX)
    ADDQ $8, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  toi32_loop8

toi32_remainder:
    ANDQ $7, CX
    JZ   toi32_done

toi32_scalar:
    MOVBLSX (SI), AX
    MOVL AX, (DX)
    INCQ SI
    ADDQ $4, DX
    DECQ CX
    JNZ  toi32_scalar

toi32_done:
    VZEROUPPER
    RET

// func sumAVX2(a []int8) int32
// Widens 16 bytes/iter to int16 (VPMOVSXBW) and pair-reduces to int32 with
// VPMADDWD against an all-ones int16 vector, accumulating in Y2; a horizontal
// add then folds the 8 int32 lanes and a scalar tail adds the remainder.
TEXT ·sumAVX2(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VPXOR Y2, Y2, Y2           // int32 accumulator = 0
    VPCMPEQW Y4, Y4, Y4        // all ones (each int16 lane = -1)
    VPXOR Y5, Y5, Y5
    VPSUBW Y4, Y5, Y3          // Y3 = 0 - (-1) = +1 per int16 lane

    MOVQ CX, AX
    SHRQ $4, AX                // AX = n / 16
    JZ   sum_reduce

sum_loop16:
    VPMOVSXBW (SI), Y0         // 16 int16
    VPMADDWD Y3, Y0, Y1        // Y1 = pairwise (x*1 + x*1) -> 8 int32
    VPADDD Y1, Y2, Y2          // accumulate
    ADDQ $16, SI
    DECQ AX
    JNZ  sum_loop16

sum_reduce:
    VEXTRACTI128 $1, Y2, X3
    VPADDD X3, X2, X2          // fold 8 -> 4 int32
    VPSHUFD $0x4E, X2, X3      // swap 64-bit halves
    VPADDD X3, X2, X2
    VPSHUFD $0xB1, X2, X3      // swap 32-bit within pairs
    VPADDD X3, X2, X2
    MOVQ X2, AX                // low int32 = vector total (in EAX)

    ANDQ $15, CX
    JZ   sum_done

sum_scalar:
    MOVBLSX (SI), BX
    ADDL BX, AX
    INCQ SI
    DECQ CX
    JNZ  sum_scalar

sum_done:
    MOVL AX, ret+24(FP)
    VZEROUPPER
    RET

// func dotAVX2(a, b []int8) int32
// Widens 16 bytes/iter of each operand to int16 (VPMOVSXBW) and reduces the
// products to int32 with VPMADDWD, accumulating in Y2; a horizontal add folds
// the lanes and a scalar tail adds the remaining products.
TEXT ·dotAVX2(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    VPXOR Y2, Y2, Y2           // int32 accumulator = 0

    MOVQ CX, AX
    SHRQ $4, AX                // AX = n / 16
    JZ   dot_reduce

dot_loop16:
    VPMOVSXBW (SI), Y0         // a -> 16 int16
    VPMOVSXBW (DI), Y1         // b -> 16 int16
    VPMADDWD Y1, Y0, Y4        // Y4 = pairwise (a*b) sums -> 8 int32
    VPADDD Y4, Y2, Y2          // accumulate
    ADDQ $16, SI
    ADDQ $16, DI
    DECQ AX
    JNZ  dot_loop16

dot_reduce:
    VEXTRACTI128 $1, Y2, X3
    VPADDD X3, X2, X2
    VPSHUFD $0x4E, X2, X3
    VPADDD X3, X2, X2
    VPSHUFD $0xB1, X2, X3
    VPADDD X3, X2, X2
    MOVQ X2, AX                // EAX = vector total

    ANDQ $15, CX
    JZ   dot_done

dot_scalar:
    MOVBLSX (SI), BX
    MOVBLSX (DI), DX
    IMULL DX, BX               // a*b (signed, fits int32)
    ADDL BX, AX
    INCQ SI
    INCQ DI
    DECQ CX
    JNZ  dot_scalar

dot_done:
    MOVL AX, ret+48(FP)
    VZEROUPPER
    RET

// func minMaxAVX2(a []int8) (minVal, maxVal int8)
// Signed byte min and max in one pass: VPMINSB/VPMAXSB fold 32-byte blocks into
// running accumulators, a per-lane cascade reduces a 128-bit lane to a single
// byte, and a scalar tail folds the (n mod 32) remainder. The dispatch gates
// n >= 32, so at least one full block exists.
TEXT ·minMaxAVX2(SB), NOSPLIT, $0-26
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VMOVDQU (SI), Y0           // min acc = block 0
    VMOVDQU (SI), Y1           // max acc = block 0
    MOVQ CX, AX
    SHRQ $5, AX                // AX = full 32-byte blocks (>=1)
    DECQ AX                    // blocks remaining after block 0
    JZ   mm_reduce
    LEAQ 32(SI), DI            // working ptr at block 1

mm_loop:
    VMOVDQU (DI), Y2
    VPMINSB Y2, Y0, Y0
    VPMAXSB Y2, Y1, Y1
    ADDQ $32, DI
    DECQ AX
    JNZ  mm_loop

mm_reduce:
    // Fold the 256-bit min accumulator to one byte.
    VEXTRACTI128 $1, Y0, X3
    VPMINSB X3, X0, X0         // 16 bytes
    VPSRLDQ $8, X0, X3
    VPMINSB X3, X0, X0         // 8 bytes
    VPSRLDQ $4, X0, X3
    VPMINSB X3, X0, X0         // 4 bytes
    VPSRLDQ $2, X0, X3
    VPMINSB X3, X0, X0         // 2 bytes
    VPSRLDQ $1, X0, X3
    VPMINSB X3, X0, X0         // 1 byte
    MOVD X0, AX                // AL = running min

    // Fold the 256-bit max accumulator to one byte.
    VEXTRACTI128 $1, Y1, X3
    VPMAXSB X3, X1, X1
    VPSRLDQ $8, X1, X3
    VPMAXSB X3, X1, X1
    VPSRLDQ $4, X1, X3
    VPMAXSB X3, X1, X1
    VPSRLDQ $2, X1, X3
    VPMAXSB X3, X1, X1
    VPSRLDQ $1, X1, X3
    VPMAXSB X3, X1, X1
    MOVD X1, DX                // DL = running max

    // scalar tail: (n mod 32) residuals at &a[fullBlocks*32]. Legacy GPRs only:
    // DI = tail ptr (SI free after), SI = loaded byte, CX = sign-extend temp.
    MOVQ CX, BX
    ANDQ $31, BX
    JZ   mm_done
    MOVQ CX, DI
    ANDQ $-32, DI              // fullBlocks*32 bytes
    ADDQ SI, DI               // tail ptr

mm_tail:
    MOVBLSX (DI), SI           // r (sign-extended)
    MOVBLSX AX, CX             // current min (sign-extend AL)
    CMPL SI, CX
    JGE  mm_tail_max
    MOVB SI, AX               // r < min -> new min (low byte)
mm_tail_max:
    MOVBLSX DX, CX             // current max (sign-extend DL)
    CMPL SI, CX
    JLE  mm_tail_next
    MOVB SI, DX               // r > max -> new max
mm_tail_next:
    INCQ DI
    DECQ BX
    JNZ  mm_tail

mm_done:
    MOVB AX, minVal+24(FP)
    MOVB DX, maxVal+25(FP)
    VZEROUPPER
    RET

// func minAVX2(dst, a, b []int8)
// Element-wise signed min: VPMINSB folds 32-byte blocks; a signed scalar tail
// handles the (n mod 32) remainder.
TEXT ·minAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $5, AX                // AX = n / 32
    JZ   min_remainder

min_loop32:
    VMOVDQU (SI), Y0
    VMOVDQU (DI), Y1
    VPMINSB Y1, Y0, Y2         // Y2 = min(a, b)
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  min_loop32

min_remainder:
    ANDQ $31, CX
    JZ   min_done

min_scalar:
    MOVBLSX (SI), AX           // a (sign-extended)
    MOVBLSX (DI), BX           // b
    CMPL AX, BX
    JLE  min_store             // a <= b -> keep a
    MOVL BX, AX
min_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DI
    INCQ DX
    DECQ CX
    JNZ  min_scalar

min_done:
    VZEROUPPER
    RET

// func maxAVX2(dst, a, b []int8)
// Element-wise signed max: VPMAXSB folds 32-byte blocks; a signed scalar tail
// handles the (n mod 32) remainder.
TEXT ·maxAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $5, AX
    JZ   max_remainder

max_loop32:
    VMOVDQU (SI), Y0
    VMOVDQU (DI), Y1
    VPMAXSB Y1, Y0, Y2         // Y2 = max(a, b)
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  max_loop32

max_remainder:
    ANDQ $31, CX
    JZ   max_done

max_scalar:
    MOVBLSX (SI), AX
    MOVBLSX (DI), BX
    CMPL AX, BX
    JGE  max_store             // a >= b -> keep a
    MOVL BX, AX
max_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DI
    INCQ DX
    DECQ CX
    JNZ  max_scalar

max_done:
    VZEROUPPER
    RET

// func clampAVX2(dst, src []int8, lo, hi int8)
// Activation clip: broadcast lo/hi to all 32 lanes, then VPMAXSB(src, lo) and
// VPMINSB(., hi) per block. With lo > hi every element maps to hi. A scalar tail
// reproduces the max-then-min clamp on the (n mod 32) remainder.
TEXT ·clampAVX2(SB), NOSPLIT, $0-50
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI
    VPBROADCASTB lo+48(FP), Y3 // loVec: lo in all 32 lanes
    VPBROADCASTB hi+49(FP), Y4 // hiVec

    MOVQ CX, AX
    SHRQ $5, AX
    JZ   clamp_remainder

clamp_loop32:
    VMOVDQU (SI), Y0
    VPMAXSB Y3, Y0, Y0         // max(src, lo)
    VPMINSB Y4, Y0, Y2         // min(., hi)
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  clamp_loop32

clamp_remainder:
    ANDQ $31, CX
    JZ   clamp_done
    MOVBLSX lo+48(FP), DI      // lo (sign-extended)
    MOVBLSX hi+49(FP), R8      // hi

clamp_scalar:
    MOVBLSX (SI), AX
    CMPL AX, DI
    JGE  clamp_chkhi
    MOVL DI, AX                // v < lo -> lo
clamp_chkhi:
    CMPL AX, R8
    JLE  clamp_store
    MOVL R8, AX                // v > hi -> hi
clamp_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DX
    DECQ CX
    JNZ  clamp_scalar

clamp_done:
    VZEROUPPER
    RET

// func absAVX2(dst, a []int8)
// Saturating absolute value: |a| = max(a, saturating(0 - a)), so abs(-128)
// clamps to 127 (VPSUBSB saturates 0-(-128)=128 to 127, and the VPMAXSB picks
// it). A scalar tail negates-if-negative then clamps the (n mod 32) remainder.
TEXT ·absAVX2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VPXOR Y3, Y3, Y3           // zero

    MOVQ CX, AX
    SHRQ $5, AX
    JZ   abs_remainder

abs_loop32:
    VMOVDQU (SI), Y0
    VPSUBSB Y0, Y3, Y1         // Y1 = saturating(0 - a)
    VPMAXSB Y1, Y0, Y2         // Y2 = max(a, -a) = |a|
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  abs_loop32

abs_remainder:
    ANDQ $31, CX
    JZ   abs_done

abs_scalar:
    MOVBLSX (SI), AX
    TESTL AX, AX
    JGE  abs_clamp
    NEGL AX                    // |a|; -(-128) = 128
abs_clamp:
    CMPL AX, $127
    JLE  abs_store
    MOVL $127, AX              // saturate 128 -> 127
abs_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DX
    DECQ CX
    JNZ  abs_scalar

abs_done:
    VZEROUPPER
    RET

// func negAVX2(dst, a []int8)
// Saturating negation: VPSUBSB(a, 0) = saturating(0 - a), so neg(-128) clamps to
// 127. A scalar tail negates then clamps high (-a is always >= -127, so the low
// bound never binds) on the (n mod 32) remainder.
TEXT ·negAVX2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VPXOR Y3, Y3, Y3           // zero

    MOVQ CX, AX
    SHRQ $5, AX
    JZ   neg_remainder

neg_loop32:
    VMOVDQU (SI), Y0
    VPSUBSB Y0, Y3, Y2         // Y2 = saturating(0 - a)
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  neg_loop32

neg_remainder:
    ANDQ $31, CX
    JZ   neg_done

neg_scalar:
    MOVBLSX (SI), AX
    NEGL AX                    // -a; -(-128) = 128
    CMPL AX, $127
    JLE  neg_store
    MOVL $127, AX              // saturate 128 -> 127
neg_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DX
    DECQ CX
    JNZ  neg_scalar

neg_done:
    VZEROUPPER
    RET

// func maxAbsAVX2(a []int8) int
// Per-tensor abs-max for dynamic quantization: VPABSB maps each byte to its
// magnitude (abs(-128) -> 0x80, i.e. 128 read unsigned), VPMAXUB folds 32-byte
// blocks into an unsigned-max accumulator, a VPMAXUB/VPSRLDQ cascade reduces a
// 128-bit lane to one byte, and a scalar tail folds the (n mod 32) remainder.
// The result is read zero-extended, so it lands in [0, 128].
TEXT ·maxAbsAVX2(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VPXOR Y0, Y0, Y0           // unsigned-max accumulator = 0

    MOVQ CX, AX
    SHRQ $5, AX                // AX = n / 32
    JZ   maxabs_reduce

maxabs_loop32:
    VPABSB (SI), Y1            // |a| as unsigned bytes
    VPMAXUB Y1, Y0, Y0         // unsigned max accumulate
    ADDQ $32, SI
    DECQ AX
    JNZ  maxabs_loop32

maxabs_reduce:
    VEXTRACTI128 $1, Y0, X1
    VPMAXUB X1, X0, X0         // fold 32 -> 16 bytes
    VPSRLDQ $8, X0, X1
    VPMAXUB X1, X0, X0         // 8 bytes
    VPSRLDQ $4, X0, X1
    VPMAXUB X1, X0, X0         // 4 bytes
    VPSRLDQ $2, X0, X1
    VPMAXUB X1, X0, X0         // 2 bytes
    VPSRLDQ $1, X0, X1
    VPMAXUB X1, X0, X0         // 1 byte
    MOVD X0, AX
    ANDQ $0xFF, AX             // running abs-max (unsigned byte) in [0, 128]

    ANDQ $31, CX
    JZ   maxabs_done

maxabs_scalar:
    MOVBLSX (SI), BX           // v (sign-extended)
    TESTL BX, BX
    JGE  maxabs_cmp
    NEGL BX                    // |v|; -(-128) = 128
maxabs_cmp:
    CMPL BX, AX                // both in [0, 128], unsigned == signed here
    JLE  maxabs_next
    MOVL BX, AX                // new running max
maxabs_next:
    INCQ SI
    DECQ CX
    JNZ  maxabs_scalar

maxabs_done:
    MOVQ AX, ret+24(FP)
    VZEROUPPER
    RET

// func absDiffAVX2(dst, a, b []int8)
// Saturating absolute difference clamped to [0, 127]: |a-b| = max(saturating
// (a-b), saturating(b-a)); the VPMAXSB picks the non-negative capped difference,
// so |127 - (-128)| = 255 saturates to 127. A scalar tail subtracts, negates if
// negative, and clamps high on the (n mod 32) remainder.
TEXT ·absDiffAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $5, AX
    JZ   absdiff_remainder

absdiff_loop32:
    VMOVDQU (SI), Y0           // a
    VMOVDQU (DI), Y1           // b
    VPSUBSB Y1, Y0, Y2         // Y2 = saturating(a - b)
    VPSUBSB Y0, Y1, Y3         // Y3 = saturating(b - a)
    VPMAXSB Y3, Y2, Y4         // Y4 = |a - b| clamped to [0, 127]
    VMOVDQU Y4, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  absdiff_loop32

absdiff_remainder:
    ANDQ $31, CX
    JZ   absdiff_done

absdiff_scalar:
    MOVBLSX (SI), AX
    MOVBLSX (DI), BX
    SUBL BX, AX                // a - b in int32 (|.| <= 255)
    TESTL AX, AX
    JGE  absdiff_clamp
    NEGL AX                    // |a - b|
absdiff_clamp:
    CMPL AX, $127
    JLE  absdiff_store
    MOVL $127, AX              // saturate to 127
absdiff_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DI
    INCQ DX
    DECQ CX
    JNZ  absdiff_scalar

absdiff_done:
    VZEROUPPER
    RET
