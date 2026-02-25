//go:build darwin && arm64 && cgo

package accelerate

/*
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
*/
import "C"

import "unsafe"

const maxCInt = int(^uint32(0) >> 1)
const minCgoTrigLen64 = 128

func Enabled() bool {
	return true
}

func Sin64(dst, src []float64) bool {
	n := len(dst)
	if n == 0 || n < minCgoTrigLen64 || len(src) < n || n > maxCInt {
		return false
	}
	count := C.int(n)
	C.vvsin(
		(*C.double)(unsafe.Pointer(&dst[0])),
		(*C.double)(unsafe.Pointer(&src[0])),
		&count,
	)
	return true
}

func Cos64(dst, src []float64) bool {
	n := len(dst)
	if n == 0 || n < minCgoTrigLen64 || len(src) < n || n > maxCInt {
		return false
	}
	count := C.int(n)
	C.vvcos(
		(*C.double)(unsafe.Pointer(&dst[0])),
		(*C.double)(unsafe.Pointer(&src[0])),
		&count,
	)
	return true
}

func SinCos64(sinDst, cosDst, src []float64) bool {
	n := len(src)
	if n == 0 || n < minCgoTrigLen64 || len(sinDst) < n || len(cosDst) < n || n > maxCInt {
		return false
	}
	count := C.int(n)
	C.vvsincos(
		(*C.double)(unsafe.Pointer(&sinDst[0])),
		(*C.double)(unsafe.Pointer(&cosDst[0])),
		(*C.double)(unsafe.Pointer(&src[0])),
		&count,
	)
	return true
}
