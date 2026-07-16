//go:build amd64

#include "textflag.h"

// int16 (de)interleave on AMD64.
//
// Interleaving is pure 16-bit-lane data movement: no arithmetic happens, so the
// bit pattern of each lane is irrelevant. Unlike the 32-bit lane case there is
// no float shuffle that moves 16-bit lanes, so these kernels use the integer
// word shuffles directly: PUNPCKLWD/PUNPCKHWD pack the lanes (SSE2 and the AVX2
// VEX form), the AVX2 deinterleave gathers even/odd words with a VPSHUFB byte
// mask, and lane-crossing fixups use VPERM2I128/VPERMQ (AVX2) or
// PUNPCKLQDQ/PUNPCKHQDQ (SSE2). The chain is load -> shuffle -> store with no
// integer ALU op in between. The scalar tails use the 16-bit MOVW.
//
// Two ISA tiers are provided; i16_amd64.go dispatches AVX2 > SSE2 > Go.

// deinterleave2Mask is the per-128-bit-lane VPSHUFB control that gathers the
// even words (channel a) into the low 8 bytes and the odd words (channel b)
// into the high 8 bytes of each lane. Each mask entry is a *source* byte offset
// within the lane; read in output order (output byte i takes source byte
// mask[i]):
//   a (even words 0,2,4,6): source bytes 0,1, 4,5, 8,9, 12,13
//   b (odd words 1,3,5,7):  source bytes 2,3, 6,7, 10,11, 14,15
// The same 16 bytes are repeated for the upper lane (VPSHUFB is lane-local).
DATA deinterleave2Mask<>+0(SB)/8, $0x0d0c090805040100
DATA deinterleave2Mask<>+8(SB)/8, $0x0f0e0b0a07060302
DATA deinterleave2Mask<>+16(SB)/8, $0x0d0c090805040100
DATA deinterleave2Mask<>+24(SB)/8, $0x0f0e0b0a07060302
GLOBL deinterleave2Mask<>(SB), RODATA|NOPTR, $32

// func interleave2AVX2(dst, a, b []int16)
// Interleaves two slices: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], dst[3]=b[1], ...
TEXT ·interleave2AVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX    // dst pointer
    MOVQ a_base+24(FP), SI     // a pointer
    MOVQ a_len+32(FP), CX      // n = len(a)
    MOVQ b_base+48(FP), DI     // b pointer

    // Process 16 pairs at a time (32 output elements)
    MOVQ CX, AX
    SHRQ $4, AX                // AX = n / 16
    JZ   interleave2_avx2_remainder

interleave2_avx2_loop16:
    VMOVDQU (SI), Y0           // Y0 = [a0..a15]
    VMOVDQU (DI), Y1           // Y1 = [b0..b15]

    // Interleave words within each 128-bit lane
    VPUNPCKLWD Y1, Y0, Y2      // [a0,b0,a1,b1,a2,b2,a3,b3 | a8,b8,a9,b9,a10,b10,a11,b11]
    VPUNPCKHWD Y1, Y0, Y3      // [a4,b4,a5,b5,a6,b6,a7,b7 | a12,b12,a13,b13,a14,b14,a15,b15]

    // Reorder the 128-bit lanes into the final stereo stream
    VPERM2I128 $0x20, Y3, Y2, Y4  // first 16 interleaved [a0,b0..a7,b7]
    VPERM2I128 $0x31, Y3, Y2, Y5  // next 16 interleaved  [a8,b8..a15,b15]

    VMOVDQU Y4, (DX)
    VMOVDQU Y5, 32(DX)

    ADDQ $32, SI               // a += 16 * 2
    ADDQ $32, DI               // b += 16 * 2
    ADDQ $64, DX               // dst += 32 * 2
    DECQ AX
    JNZ  interleave2_avx2_loop16

interleave2_avx2_remainder:
    ANDQ $15, CX
    JZ   interleave2_avx2_done

interleave2_avx2_scalar:
    MOVW (SI), AX
    MOVW (DI), BX
    MOVW AX, (DX)
    MOVW BX, 2(DX)
    ADDQ $2, SI
    ADDQ $2, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  interleave2_avx2_scalar

interleave2_avx2_done:
    VZEROUPPER
    RET

// func deinterleave2AVX2(a, b, src []int16)
// Deinterleaves: a[0]=src[0], b[0]=src[1], a[1]=src[2], b[1]=src[3], ...
// 1. VPSHUFB gathers evens (a's) to the low 8 bytes, odds (b's) to the high 8
//    bytes of each 128-bit lane.
// 2. VPUNPCKLQDQ/VPUNPCKHQDQ split the lanes into a-only and b-only registers.
// 3. VPERMQ fixes the lane interleave the two 128-bit halves introduced.
TEXT ·deinterleave2AVX2(SB), NOSPLIT, $0-72
    MOVQ a_base+0(FP), DX      // a pointer
    MOVQ a_len+8(FP), CX       // n = len(a)
    MOVQ b_base+24(FP), R8     // b pointer
    MOVQ src_base+48(FP), SI   // src pointer

    VMOVDQU deinterleave2Mask<>(SB), Y7  // even/odd word gather mask

    // Process 16 pairs at a time
    MOVQ CX, AX
    SHRQ $4, AX
    JZ   deinterleave2_avx2_remainder

deinterleave2_avx2_loop16:
    VMOVDQU (SI), Y0           // [a0,b0..a7,b7]
    VMOVDQU 32(SI), Y1         // [a8,b8..a15,b15]

    VPSHUFB Y7, Y0, Y0         // [a0,a1,a2,a3,b0,b1,b2,b3 | a4,a5,a6,a7,b4,b5,b6,b7]
    VPSHUFB Y7, Y1, Y1         // [a8..a11,b8..b11 | a12..a15,b12..b15]

    VPUNPCKLQDQ Y1, Y0, Y2     // a's, lane-interleaved: [a0..a3,a8..a11 | a4..a7,a12..a15]
    VPUNPCKHQDQ Y1, Y0, Y3     // b's, lane-interleaved: [b0..b3,b8..b11 | b4..b7,b12..b15]

    VPERMQ $0xD8, Y2, Y2       // a in order [a0..a15]
    VPERMQ $0xD8, Y3, Y3       // b in order [b0..b15]

    VMOVDQU Y2, (DX)           // store a
    VMOVDQU Y3, (R8)           // store b

    ADDQ $64, SI               // src += 32 * 2
    ADDQ $32, DX               // a += 16 * 2
    ADDQ $32, R8               // b += 16 * 2
    DECQ AX
    JNZ  deinterleave2_avx2_loop16

deinterleave2_avx2_remainder:
    ANDQ $15, CX
    JZ   deinterleave2_avx2_done

deinterleave2_avx2_scalar:
    MOVW (SI), AX
    MOVW 2(SI), BX
    MOVW AX, (DX)
    MOVW BX, (R8)
    ADDQ $4, SI
    ADDQ $2, DX
    ADDQ $2, R8
    DECQ CX
    JNZ  deinterleave2_avx2_scalar

deinterleave2_avx2_done:
    VZEROUPPER
    RET

// func interleave2SSE2(dst, a, b []int16)
// 128-bit fallback: 8 pairs per iteration using PUNPCKLWD/PUNPCKHWD.
TEXT ·interleave2SSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX    // dst pointer
    MOVQ a_base+24(FP), SI     // a pointer
    MOVQ a_len+32(FP), CX      // n = len(a)
    MOVQ b_base+48(FP), DI     // b pointer

    MOVQ CX, AX
    SHRQ $3, AX                // AX = n / 8
    JZ   interleave2_sse2_remainder

interleave2_sse2_loop8:
    MOVOU (SI), X0             // [a0..a7]
    MOVOU (DI), X1             // [b0..b7]
    MOVOU X0, X2               // copy a (PUNPCK overwrites its destination)
    PUNPCKLWL X1, X0           // [a0,b0,a1,b1,a2,b2,a3,b3] (Intel PUNPCKLWD)
    PUNPCKHWL X1, X2           // [a4,b4,a5,b5,a6,b6,a7,b7] (Intel PUNPCKHWD)
    MOVOU X0, (DX)
    MOVOU X2, 16(DX)

    ADDQ $16, SI               // a += 8 * 2
    ADDQ $16, DI               // b += 8 * 2
    ADDQ $32, DX               // dst += 16 * 2
    DECQ AX
    JNZ  interleave2_sse2_loop8

interleave2_sse2_remainder:
    ANDQ $7, CX
    JZ   interleave2_sse2_done

interleave2_sse2_scalar:
    MOVW (SI), AX
    MOVW (DI), BX
    MOVW AX, (DX)
    MOVW BX, 2(DX)
    ADDQ $2, SI
    ADDQ $2, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  interleave2_sse2_scalar

interleave2_sse2_done:
    RET

// func deinterleave2SSE2(a, b, src []int16)
// 128-bit fallback: 8 pairs per iteration. PSHUFLW/PSHUFHW/PSHUFD move each
// lane's evens to its low 64 bits and odds to its high 64 bits without a memory
// mask; PUNPCKLQDQ/PUNPCKHQDQ then merge the two source halves into a and b.
TEXT ·deinterleave2SSE2(SB), NOSPLIT, $0-72
    MOVQ a_base+0(FP), DX      // a pointer
    MOVQ a_len+8(FP), CX       // n = len(a)
    MOVQ b_base+24(FP), R8     // b pointer
    MOVQ src_base+48(FP), SI   // src pointer

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   deinterleave2_sse2_remainder

deinterleave2_sse2_loop8:
    MOVOU (SI), X0             // [a0,b0,a1,b1,a2,b2,a3,b3]
    MOVOU 16(SI), X1           // [a4,b4,a5,b5,a6,b6,a7,b7]

    // X0 -> [a0,a1,a2,a3 | b0,b1,b2,b3]
    PSHUFLW $0xD8, X0, X0      // low words  -> [a0,a1,b0,b1]
    PSHUFHW $0xD8, X0, X0      // high words -> [a2,a3,b2,b3]
    PSHUFD  $0xD8, X0, X0      // dwords     -> [a0,a1,a2,a3,b0,b1,b2,b3]

    // X1 -> [a4,a5,a6,a7 | b4,b5,b6,b7]
    PSHUFLW $0xD8, X1, X1
    PSHUFHW $0xD8, X1, X1
    PSHUFD  $0xD8, X1, X1

    MOVOU X0, X2
    PUNPCKLQDQ X1, X2          // a = [a0..a3,a4..a7]
    PUNPCKHQDQ X1, X0          // b = [b0..b3,b4..b7]

    MOVOU X2, (DX)             // store a
    MOVOU X0, (R8)             // store b

    ADDQ $32, SI               // src += 16 * 2
    ADDQ $16, DX               // a += 8 * 2
    ADDQ $16, R8               // b += 8 * 2
    DECQ AX
    JNZ  deinterleave2_sse2_loop8

deinterleave2_sse2_remainder:
    ANDQ $7, CX
    JZ   deinterleave2_sse2_done

deinterleave2_sse2_scalar:
    MOVW (SI), AX
    MOVW 2(SI), BX
    MOVW AX, (DX)
    MOVW BX, (R8)
    ADDQ $4, SI
    ADDQ $2, DX
    ADDQ $2, R8
    DECQ CX
    JNZ  deinterleave2_sse2_scalar

deinterleave2_sse2_done:
    RET

// Widening int16 dot product.
//
// PMADDWD multiplies eight int16 pairs and adds them pairwise into four int32
// lanes; the AVX2 form does twice that. Note the Plan 9 spelling of the SSE2
// form is PMADDWL ("long" for the 32-bit result); the VEX form keeps the Intel
// name VPMADDWD.
//
// PMADDWD is SSE2, so it is present on the whole GOAMD64=v1 baseline: there is
// no availability cliff below the SSE2 tier, and the AVX2 tier is a width win
// rather than an availability one. The dispatcher still gates on hasSSE2, which
// is always true on amd64; that mirrors the package's other kernels and keeps
// the pure-Go path reachable as a non-amd64 safety net.
//
// Overflow: the only input that can exceed int32 within one instruction is a
// pair of (-32768 * -32768), summing to 2^31, which PMADDWD wraps to
// 0x80000000. The scalar reference wraps identically (two wrapping int32 adds
// of 2^30), and because wrapping addition is associative the pairwise pre-add
// inside PMADDWD is invisible. So no special-casing is needed and the kernels
// are bit-exact with dotGo for every input; the tests pin this with all-MinInt16
// sweeps.

// func dotSSE2(a, b []int16) int32
// Handles mismatched slice lengths: uses min(len(a), len(b)).
TEXT ·dotSSE2(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_len+32(FP), DX
    CMPQ DX, CX
    CMOVQLT DX, CX             // CX = n = min(len(a), len(b))
    MOVQ b_base+24(FP), DI

    PXOR X0, X0                // int32 accumulator

    MOVQ CX, BX
    SHRQ $3, BX                // BX = n / 8
    JZ   dot_sse2_reduce

dot_sse2_loop8:
    MOVOU (SI), X1
    MOVOU (DI), X2
    PMADDWL X2, X1             // X1 = pairwise (a*b) sums -> 4 int32
    PADDD X1, X0               // accumulate (wrapping)
    ADDQ $16, SI
    ADDQ $16, DI
    DECQ BX
    JNZ  dot_sse2_loop8

dot_sse2_reduce:
    PSHUFD $0x4E, X0, X1       // swap 64-bit halves
    PADDD X1, X0
    PSHUFD $0xB1, X0, X1       // swap 32-bit within pairs
    PADDD X1, X0
    MOVQ X0, AX                // low int32 = vector total (in EAX)

    ANDQ $7, CX
    JZ   dot_sse2_done

dot_sse2_scalar:
    MOVWLSX (SI), BX           // sign-extending 16-bit load
    MOVWLSX (DI), DX
    IMULL DX, BX
    ADDL BX, AX                // 32-bit add: wraps like dotGo
    ADDQ $2, SI
    ADDQ $2, DI
    DECQ CX
    JNZ  dot_sse2_scalar

dot_sse2_done:
    MOVL AX, ret+48(FP)
    RET

// func dotAVX2(a, b []int16) int32
// Handles mismatched slice lengths: uses min(len(a), len(b)).
TEXT ·dotAVX2(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_len+32(FP), DX
    CMPQ DX, CX
    CMOVQLT DX, CX             // CX = n = min(len(a), len(b))
    MOVQ b_base+24(FP), DI

    VPXOR Y0, Y0, Y0           // int32 accumulator

    MOVQ CX, BX
    SHRQ $4, BX                // BX = n / 16
    JZ   dot_avx2_reduce

dot_avx2_loop16:
    VMOVDQU (SI), Y1
    VMOVDQU (DI), Y2
    VPMADDWD Y2, Y1, Y1        // 16 int16 pairs -> 8 int32
    VPADDD Y1, Y0, Y0          // accumulate (wrapping)
    ADDQ $32, SI
    ADDQ $32, DI
    DECQ BX
    JNZ  dot_avx2_loop16

dot_avx2_reduce:
    VEXTRACTI128 $1, Y0, X1
    VPADDD X1, X0, X0          // fold 8 -> 4 int32
    VPSHUFD $0x4E, X0, X1      // swap 64-bit halves
    VPADDD X1, X0, X0
    VPSHUFD $0xB1, X0, X1      // swap 32-bit within pairs
    VPADDD X1, X0, X0
    MOVQ X0, AX                // low int32 = vector total (in EAX)

    ANDQ $15, CX
    JZ   dot_avx2_done

dot_avx2_scalar:
    MOVWLSX (SI), BX           // sign-extending 16-bit load
    MOVWLSX (DI), DX
    IMULL DX, BX
    ADDL BX, AX                // 32-bit add: wraps like dotGo
    ADDQ $2, SI
    ADDQ $2, DI
    DECQ CX
    JNZ  dot_avx2_scalar

dot_avx2_done:
    MOVL AX, ret+48(FP)
    VZEROUPPER
    RET
