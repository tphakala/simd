package i32

// Element-wise integer arithmetic and FLAC stereo decorrelation.
//
// These primitives cover all three of FLAC's inter-channel decorrelation modes
// in both directions. Add and Sub are the building blocks; MidSideEncode and
// MidSideDecode handle the mid/side mode, which needs the parity trick and so is
// not expressible as a single Add/Sub:
//
//	mode        encode (from left,right)     decode (to left,right)
//	---------   --------------------------   --------------------------
//	LEFT_SIDE   Sub(side, left, right)       Sub(right, left, side)
//	RIGHT_SIDE  Sub(side, left, right)       Add(left, right, side)
//	MID_SIDE    MidSideEncode(...)           MidSideDecode(...)
//
// All operations clamp to the shortest operand, write into caller-provided
// slices, and use int32 wraparound (two's complement) so the SIMD and pure-Go
// paths are bit-identical across the full int32 range.

// Add writes dst[i] = a[i] + b[i] for i in [0, n), n = min(len(dst), len(a),
// len(b)). It reconstructs the left channel in FLAC RIGHT_SIDE decode:
// Add(left, right, side).
func Add(dst, a, b []int32) {
	n := min(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	addI32(dst[:n], a[:n], b[:n])
}

// Sub writes dst[i] = a[i] - b[i] for i in [0, n), n = min(len(dst), len(a),
// len(b)). It computes the FLAC side channel (Sub(side, left, right)) and
// reconstructs the right channel in LEFT_SIDE decode (Sub(right, left, side)).
func Sub(dst, a, b []int32) {
	n := min(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	subI32(dst[:n], a[:n], b[:n])
}

// MidSideEncode computes FLAC mid/side decorrelation, forward direction:
//
//	mid[i]  = (left[i] + right[i]) >> 1   (arithmetic shift; the low bit is dropped)
//	side[i] = left[i] - right[i]
//
// It processes n = min over all four slices and leaves any trailing capacity
// untouched. MidSideDecode is the exact inverse for inputs within the codec's
// effective bit depth.
func MidSideEncode(mid, side, left, right []int32) {
	n := min(len(mid), len(side), len(left), len(right))
	if n == 0 {
		return
	}
	midSideEncodeI32(mid[:n], side[:n], left[:n], right[:n])
}

// MidSideDecode inverts MidSideEncode, reconstructing the original channels:
//
//	sum      = (mid[i] << 1) | (side[i] & 1)
//	left[i]  = (sum + side[i]) >> 1
//	right[i] = (sum - side[i]) >> 1
//
// The parity bit (side[i] & 1) restores the low bit of left+right that the
// encoder's >>1 discarded; left+right and left-right always share parity.
func MidSideDecode(left, right, mid, side []int32) {
	n := min(len(left), len(right), len(mid), len(side))
	if n == 0 {
		return
	}
	midSideDecodeI32(left[:n], right[:n], mid[:n], side[:n])
}
