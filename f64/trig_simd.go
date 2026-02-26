package f64

import "math"

const (
	trigChunkSize = 128

	trigInvPiOver2 = 0.63661977236758134308 // 2 / pi
	trigPiOver2Hi  = 1.5707963267948966
	trigPiOver2Lo  = 6.123233995736766e-17 // (pi/2) - trigPiOver2Hi

	// Polynomial coefficients for sin/cos on reduced range.
	trigSinC3  = -1.6666666666666665741e-1
	trigSinC5  = 8.333333333333332871e-3
	trigSinC7  = -1.9841269841269841253e-4
	trigSinC9  = 2.7557319223985890653e-6
	trigSinC11 = -2.5052108385441718775e-8
	trigSinC13 = 1.6059043836821614599e-10

	trigCosC2  = -5.0000000000000000000e-1
	trigCosC4  = 4.1666666666666592922e-2
	trigCosC6  = -1.3888888888873056412e-3
	trigCosC8  = 2.4801587289476729418e-5
	trigCosC10 = -2.7557314351390663304e-7
	trigCosC12 = 2.0875723212981748279e-9

	// For very large magnitudes, fall back to scalar libm range reduction.
	trigFallbackAbsLimit = 1.0e6
)

// quadrantMap applies the quadrant correction for sin/cos values.
func quadrantMap(q int, s, c float64) (sv, cv float64) {
	switch q & 3 {
	case 0:
		return s, c
	case 1:
		return c, -s
	case 2:
		return -s, -c
	default: // 3
		return -c, s
	}
}

func sin64SIMD(dst, src []float64) {
	sinCos64SIMDCore(dst, nil, src)
}

func cos64SIMD(dst, src []float64) {
	sinCos64SIMDCore(nil, dst, src)
}

func sinCos64SIMD(sinDst, cosDst, src []float64) {
	sinCos64SIMDCore(sinDst, cosDst, src)
}

// sinCos64SIMDCore computes sin/cos using SIMD arithmetic primitives with
// polynomial approximation and quadrant mapping.
func sinCos64SIMDCore(sinDst, cosDst, src []float64) {
	n := len(src)
	if n == 0 {
		return
	}

	var qBuf [trigChunkSize]float64
	var rBuf [trigChunkSize]float64
	var r2Buf [trigChunkSize]float64
	var pBuf [trigChunkSize]float64
	var sinBuf [trigChunkSize]float64
	var cosBuf [trigChunkSize]float64

	for off := 0; off < n; off += trigChunkSize {
		m := min(trigChunkSize, n-off)
		x := src[off : off+m]

		q := qBuf[:m]
		r := rBuf[:m]
		r2 := r2Buf[:m]
		p := pBuf[:m]
		s := sinBuf[:m]
		c := cosBuf[:m]

		// q = round(x * 2/pi), r = x - q*(pi/2)
		scale(q, x, trigInvPiOver2)
		round64(q, q)

		scale(r, q, trigPiOver2Hi)
		sub(r, x, r)
		scale(p, q, trigPiOver2Lo)
		sub(r, r, p)

		mul(r2, r, r)

		// sin(r) = r * P(r^2)
		scale(p, r2, trigSinC13)
		addScalar(p, p, trigSinC11)
		mul(p, p, r2)
		addScalar(p, p, trigSinC9)
		mul(p, p, r2)
		addScalar(p, p, trigSinC7)
		mul(p, p, r2)
		addScalar(p, p, trigSinC5)
		mul(p, p, r2)
		addScalar(p, p, trigSinC3)
		mul(p, p, r2)
		addScalar(p, p, 1.0)
		mul(s, p, r)

		// cos(r) = Q(r^2)
		scale(p, r2, trigCosC12)
		addScalar(p, p, trigCosC10)
		mul(p, p, r2)
		addScalar(p, p, trigCosC8)
		mul(p, p, r2)
		addScalar(p, p, trigCosC6)
		mul(p, p, r2)
		addScalar(p, p, trigCosC4)
		mul(p, p, r2)
		addScalar(p, p, trigCosC2)
		mul(p, p, r2)
		addScalar(c, p, 1.0)

		needsFallback := false
		for i := range m {
			v := x[i]
			if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > trigFallbackAbsLimit {
				needsFallback = true
				break
			}
		}

		if !needsFallback {
			for i := range m {
				sv, cv := quadrantMap(int(int64(q[i])), s[i], c[i])
				if sinDst != nil {
					sinDst[off+i] = sv
				}
				if cosDst != nil {
					cosDst[off+i] = cv
				}
			}
			continue
		}

		for i := range m {
			v := x[i]
			var sv, cv float64

			if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > trigFallbackAbsLimit {
				sv, cv = math.Sincos(v)
			} else {
				sv, cv = quadrantMap(int(int64(q[i])), s[i], c[i])
			}

			if sinDst != nil {
				sinDst[off+i] = sv
			}
			if cosDst != nil {
				cosDst[off+i] = cv
			}
		}
	}
}
