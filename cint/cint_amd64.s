//go:build amd64

#include "textflag.h"

// Fixed-point complex arithmetic kernels (AVX2).
//
// Add and Sub are flat wrapping int32 lane ops (VPADDD / VPSUBD), 8 int32 per
// iteration with a scalar tail. MulByScalar is the truncating Q15 scale-by-scalar
// (MULT16_32_Q15) applied in place over the flat lane view, reusing the i32
// ScaleQ15 recombine. Mul and MulConj are the C_MUL / conjugated complex multiply.
//
// The truncating Q15 product uses the same trick as the i32 ScaleQ15 kernel: AVX2
// has VPMULDQ (signed 32x32 -> 64 on the EVEN 32-bit lanes) but no 64-bit
// ARITHMETIC shift (VPSRAQ is AVX-512). For a product p with |p| <= 2^46 and
// result int32(p >> 15), the LOW 32 bits of the LOGICAL shift VPSRLQ $15 equal the
// low 32 bits of the arithmetic shift: result bit i reads p bit i+15, and
// i+15 <= 31+15 = 46 <= 63 is a real bit of p for both shift kinds; they differ
// only in bits >= 32 (sign-fill vs zero-fill), which the int32 store discards. So
// VPSRLQ $15 then keep the low 32 bits is exact, including the MinInt32 * MinInt16
// = 2^46 -> 2^31 -> MinInt32 wrap. No EVEX is emitted: VPMULDQ, VPSRLQ, VPSLLQ,
// VPBLENDD, VPADDD, VPSUBD, VPMOVSXWD and the VMOVD/VPBROADCASTD scalar splat are
// all AVX2 (the GP-register VPBROADCASTD is AVX-512 and is avoided).
//
// The complex multiply stays interleaved: for A = [ar0,ai0,ar1,ai1,...] and the
// sign-extended twiddle T = [br0,bi0,...], VPSRLQ $32 slides the imaginary lanes
// into the even positions so all four half-products ar*br, ai*bi, ar*bi, ai*br are
// EVEN-lane VPMULDQ products; each is VPSRLQ $15 truncated; the real combine lands
// in the even lanes and the imaginary combine (after VPSLLQ $32) in the odd lanes;
// and VPBLENDD $0xAA merges them back to interleaved [r0,i0,r1,i1,...]. The Go
// 3-operand order is dst-last: VPSUBD a, b, c is c = b - a. dst may alias a
// exactly: each block loads a and tw in full before it stores dst. Reserved
// registers (g in R14, GOT temp in R15) are untouched. Every kernel ends VZEROUPPER before RET.

// func addAVX2(dst, a, b []int32)
TEXT ·addAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   add_remainder

add_loop8:
    VMOVDQU (SI), Y0
    VMOVDQU (DI), Y1
    VPADDD  Y1, Y0, Y2         // Y2 = a + b
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  add_loop8

add_remainder:
    ANDQ $7, CX
    JZ   add_done

add_scalar:
    MOVL (SI), AX
    ADDL (DI), AX
    MOVL AX, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  add_scalar

add_done:
    VZEROUPPER
    RET

// func subAVX2(dst, a, b []int32)
TEXT ·subAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   sub_remainder

sub_loop8:
    VMOVDQU (SI), Y0
    VMOVDQU (DI), Y1
    VPSUBD  Y1, Y0, Y2         // Y2 = a - b
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  sub_loop8

sub_remainder:
    ANDQ $7, CX
    JZ   sub_done

sub_scalar:
    MOVL (SI), AX
    SUBL (DI), AX
    MOVL AX, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  sub_scalar

sub_done:
    VZEROUPPER
    RET

// func mulByScalarAVX2(a []int32, s int16)
// In-place truncating Q15 scale, 8 int32 per iteration: identical recombine to the
// i32 scaleQ15AVX2 with the load and store sharing one pointer (dst aliases a). s
// is a signed int16, sign-extended to int32 before the broadcast (VPMULDQ is signed
// 32x32) and to int64 for the tail; |s * a[i]| <= 2^46. Frame is one slice header
// plus the int16 scalar: a+0, s+24.
TEXT ·mulByScalarAVX2(SB), NOSPLIT, $0-26
    MOVQ    a_base+0(FP), SI
    MOVQ    a_len+8(FP), CX
    MOVWQSX s+24(FP), BX          // BX = int64(s), sign-extended int16 (also tail s)
    VMOVD   BX, X3                // low 32 bits = int32(s)
    VPBROADCASTD X3, Y3           // AVX2 xmm->ymm broadcast of int32(s)

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   mulbyscalar_avx2_tail

mulbyscalar_avx2_loop8:
    VMOVDQU  (SI), Y0             // a[i..i+7]
    VPMULDQ  Y3, Y0, Y4           // even-lane products a0,a2,a4,a6 * s (4x int64)
    VPSRLQ   $32, Y0, Y1          // slide odd lanes a1,a3,a5,a7 into even positions
    VPMULDQ  Y3, Y1, Y5           // odd-lane products a1,a3,a5,a7 * s (4x int64)
    VPSRLQ   $15, Y4, Y4          // low 32 of each int64 = even result lanes
    VPSRLQ   $15, Y5, Y5          // low 32 of each int64 = odd result lanes
    VPSLLQ   $32, Y5, Y5          // lift odd results to 32-bit positions 1,3,5,7
    VPBLENDD $0xAA, Y5, Y4, Y2    // lanes 1,3,5,7 <- Y5 (odd), 0,2,4,6 <- Y4 (even)
    VMOVDQU  Y2, (SI)             // in place: store scaled block over a
    ADDQ $32, SI
    DECQ AX
    JNZ  mulbyscalar_avx2_loop8

mulbyscalar_avx2_tail:
    ANDQ $7, CX
    JZ   mulbyscalar_avx2_done

mulbyscalar_avx2_scalar:
    MOVLQSX (SI), AX              // int64(a[i])
    IMULQ   BX, AX                // s * a[i] (64-bit, |p| <= 2^46)
    SARQ    $15, AX               // arithmetic shift right 15
    MOVL    AX, (SI)              // low 32 bits: wraps like int32()
    ADDQ $4, SI
    DECQ CX
    JNZ  mulbyscalar_avx2_scalar

mulbyscalar_avx2_done:
    VZEROUPPER
    RET

// func mulAVX2(dst, a []int32, tw []int16)
// C_MUL complex multiply, 4 complex (8 int32) per iteration. dst+0, a+24, tw+48.
TEXT ·mulAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX        // n (int32 count, even)
    MOVQ a_base+24(FP), SI
    MOVQ tw_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX                   // AX = n/8 = 4-complex blocks
    JZ   mul_avx2_tail

mul_avx2_loop8:
    VMOVDQU   (SI), Y0            // A = [ar0,ai0,ar1,ai1,ar2,ai2,ar3,ai3]
    VPMOVSXWD (DI), Y1            // T = [br0,bi0,...] sign-extended int16 -> int32
    VPSRLQ    $32, Y0, Y2         // As: even 32-lanes = ai0,ai1,ai2,ai3
    VPSRLQ    $32, Y1, Y3         // Ts: even 32-lanes = bi0,bi1,bi2,bi3
    VPMULDQ   Y1, Y0, Y4         // prr = ar*br (4x int64)
    VPMULDQ   Y3, Y2, Y5         // pii = ai*bi
    VPMULDQ   Y3, Y0, Y6         // pri = ar*bi
    VPMULDQ   Y1, Y2, Y7         // pir = ai*br
    VPSRLQ    $15, Y4, Y4         // Q15 truncate: low 32 of each int64 (even lanes)
    VPSRLQ    $15, Y5, Y5
    VPSRLQ    $15, Y6, Y6
    VPSRLQ    $15, Y7, Y7
    VPSUBD    Y5, Y4, Y8         // re = prr - pii (even 32-lanes valid; odd don't-care)
    VPADDD    Y7, Y6, Y9         // im = pri + pir (even 32-lanes valid)
    VPSLLQ    $32, Y9, Y9         // move imag results to ODD 32-lanes
    VPBLENDD  $0xAA, Y9, Y8, Y8   // even<-re (reals), odd<-im (imags): interleaved
    VMOVDQU   Y8, (DX)
    ADDQ $32, SI
    ADDQ $16, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  mul_avx2_loop8

mul_avx2_tail:
    MOVQ CX, BX
    ANDQ $7, BX                   // leftover int32 (even: 0,2,4,6)
    SHRQ $1, BX                   // leftover complex count
    JZ   mul_avx2_done

mul_avx2_scalar:
    MOVLQSX (SI), AX              // ar = int64(a[2k])
    MOVLQSX 4(SI), R8             // ai = int64(a[2k+1])
    MOVWQSX (DI), R9             // br = int64(tw[2k])
    MOVWQSX 2(DI), R10            // bi = int64(tw[2k+1])
    MOVQ  AX, R11
    IMULQ R9, R11                 // ar*br
    SARQ  $15, R11                // prr
    MOVQ  R8, R12
    IMULQ R10, R12                // ai*bi
    SARQ  $15, R12                // pii
    SUBL  R12, R11                // re = prr - pii (int32 wrap on low 32)
    MOVL  R11, (DX)               // dst[2k] = re
    MOVQ  AX, R11
    IMULQ R10, R11                // ar*bi
    SARQ  $15, R11                // pri
    IMULQ R9, R8                  // ai*br
    SARQ  $15, R8                 // pir
    ADDL  R8, R11                 // im = pri + pir (int32 wrap on low 32)
    MOVL  R11, 4(DX)              // dst[2k+1] = im
    ADDQ $8, SI
    ADDQ $4, DI
    ADDQ $8, DX
    DECQ BX
    JNZ  mul_avx2_scalar

mul_avx2_done:
    VZEROUPPER
    RET

// func mulConjAVX2(dst, a []int32, tw []int16)
// Conjugated complex multiply: the same four half-products as mulAVX2, but the
// real combine adds (prr + pii) and the imaginary combine subtracts (pir - pri).
TEXT ·mulConjAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX        // n (int32 count, even)
    MOVQ a_base+24(FP), SI
    MOVQ tw_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX                   // AX = n/8 = 4-complex blocks
    JZ   mulconj_avx2_tail

mulconj_avx2_loop8:
    VMOVDQU   (SI), Y0            // A = [ar0,ai0,ar1,ai1,ar2,ai2,ar3,ai3]
    VPMOVSXWD (DI), Y1            // T = [br0,bi0,...] sign-extended int16 -> int32
    VPSRLQ    $32, Y0, Y2         // As: even 32-lanes = ai0..ai3
    VPSRLQ    $32, Y1, Y3         // Ts: even 32-lanes = bi0..bi3
    VPMULDQ   Y1, Y0, Y4         // prr = ar*br
    VPMULDQ   Y3, Y2, Y5         // pii = ai*bi
    VPMULDQ   Y3, Y0, Y6         // pri = ar*bi
    VPMULDQ   Y1, Y2, Y7         // pir = ai*br
    VPSRLQ    $15, Y4, Y4         // Q15 truncate (even lanes)
    VPSRLQ    $15, Y5, Y5
    VPSRLQ    $15, Y6, Y6
    VPSRLQ    $15, Y7, Y7
    VPADDD    Y5, Y4, Y8         // re = prr + pii (even 32-lanes valid)
    VPSUBD    Y6, Y7, Y9         // im = pir - pri (Y7 - Y6, even 32-lanes valid)
    VPSLLQ    $32, Y9, Y9         // move imag results to ODD 32-lanes
    VPBLENDD  $0xAA, Y9, Y8, Y8   // even<-re, odd<-im: interleaved
    VMOVDQU   Y8, (DX)
    ADDQ $32, SI
    ADDQ $16, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  mulconj_avx2_loop8

mulconj_avx2_tail:
    MOVQ CX, BX
    ANDQ $7, BX                   // leftover int32 (even)
    SHRQ $1, BX                   // leftover complex count
    JZ   mulconj_avx2_done

mulconj_avx2_scalar:
    MOVLQSX (SI), AX              // ar
    MOVLQSX 4(SI), R8             // ai
    MOVWQSX (DI), R9             // br
    MOVWQSX 2(DI), R10            // bi
    MOVQ  AX, R11
    IMULQ R9, R11                 // ar*br
    SARQ  $15, R11                // prr
    MOVQ  R8, R12
    IMULQ R10, R12                // ai*bi
    SARQ  $15, R12                // pii
    ADDL  R12, R11                // re = prr + pii (int32 wrap on low 32)
    MOVL  R11, (DX)               // dst[2k] = re
    MOVQ  AX, R11
    IMULQ R10, R11                // ar*bi
    SARQ  $15, R11                // pri
    IMULQ R9, R8                  // ai*br
    SARQ  $15, R8                 // pir
    SUBL  R11, R8                 // im = pir - pri (int32 wrap on low 32)
    MOVL  R8, 4(DX)               // dst[2k+1] = im
    ADDQ $8, SI
    ADDQ $4, DI
    ADDQ $8, DX
    DECQ BX
    JNZ  mulconj_avx2_scalar

mulconj_avx2_done:
    VZEROUPPER
    RET
