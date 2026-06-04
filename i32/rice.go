package i32

// Rice zigzag cost search for the FLAC codec.
//
// FLAC Rice-codes each residual by mapping it to an unsigned symbol with the
// zigzag fold zigzag(r) = (r<<1) ^ (r>>31) (0,-1,1,-2,2 -> 0,1,2,3,4) and then
// writing that symbol in a Rice code with parameter k: the high bits u>>k in
// unary (u>>k zero bits plus a stop bit) followed by the low k bits verbatim.
// One symbol therefore costs (u>>k) + 1 + k bits, so a block of n residuals
// costs
//
//	bits(k) = Σ_i (zigzag(res[i]) >> k) + n*(k+1)
//
// The encoder picks the k minimizing bits(k). The data-dependent term is the
// per-parameter unary-bit sum Σ_i (zigzag(res[i]) >> k); computing it for every
// candidate k is the hot loop RiceSums vectorizes. RiceBestParam wraps it with
// the scalar k scan, which is cheap (a handful of parameters).

const (
	// riceMaxParam is the largest Rice parameter the SIMD fast path scans,
	// matching FLAC's 4-bit Rice partition method (k = 0..14; parameter 15 is
	// the escape code). riceParamCount is the corresponding sum count.
	riceMaxParam   = 14
	riceParamCount = riceMaxParam + 1

	// riceMaxParam5 is the largest parameter FLAC's 5-bit partition method can
	// encode (k = 0..30; 31 is the escape). RiceBestParam clamps maxParam to it;
	// parameters above riceMaxParam are served by the wide SIMD path
	// (riceSumsWideI32), falling back to pure-Go only off the SIMD architectures.
	riceMaxParam5 = 30
)

// RiceSums fills sums[k] with the per-parameter unary-bit total
//
//	sums[k] = Σ_i (zigzag(res[i]) >> k)   for k in [0, len(sums))
//
// where zigzag(r) = (r<<1) ^ (r>>31) is the FLAC residual fold. Each sum is
// accumulated in uint64 (it cannot overflow for any realistic block length).
// The total Rice-coded bit cost of res with parameter k is sums[k] + n*(k+1),
// n = len(res); see RiceBestParam.
//
// sums is fully overwritten (not accumulated into). res is read-only. The SIMD
// fast path covers FLAC's full parameter range: the 4-bit method (len(sums) <=
// riceParamCount = 15) and the 5-bit method (up to riceMaxParam5+1 = 31). Wider
// requests fall back to the pure-Go reference.
func RiceSums(sums []uint64, res []int32) {
	switch m := len(sums); {
	case m == 0:
		return
	case m == riceParamCount:
		riceSumsI32(sums, res)
	case m < riceParamCount:
		// The 15-wide kernel writes exactly riceParamCount sums; compute the full
		// set on the stack and copy the requested prefix. The array does not
		// escape, so this stays allocation-free.
		var full [riceParamCount]uint64
		riceSumsI32(full[:], res)
		copy(sums, full[:m])
	case m <= riceMaxParam5+1:
		// FLAC 5-bit range (16..31 columns): vectorize all of it. The wide path
		// always writes riceMaxParam5+1 sums; compute on the stack (no escape) and
		// copy the requested prefix.
		var full [riceMaxParam5 + 1]uint64
		riceSumsWideI32(full[:], res)
		copy(sums, full[:m])
	default: // m > riceMaxParam5+1
		riceSumsGo(sums, res)
	}
}

// ZigzagSum returns the sum of the FLAC residual zigzag fold over res:
//
//	Σ_i zigzag(res[i]),   zigzag(r) = (r<<1) ^ (r>>31)
//
// Each residual is folded to its unsigned Rice symbol and accumulated in uint64
// (it cannot overflow for any realistic block length). This is exactly the k=0
// column of RiceSums, exposed separately for the estimate path that needs only
// that total and not the full per-parameter sweep. res is read-only; an empty
// res returns 0.
func ZigzagSum(res []int32) uint64 {
	return zigzagSumI32(res)
}

// RiceBestParam scans Rice parameters k in [0, maxParam] and returns the k that
// minimizes the total Rice-coded bit count of res, together with that bit count
//
//	bits(k) = Σ_i (zigzag(res[i]) >> k) + n*(k+1),   n = len(res)
//
// On a tie the smallest k wins. maxParam is clamped to riceMaxParam5 (30),
// FLAC's largest encodable parameter. An empty res returns (0, 0).
func RiceBestParam(res []int32, maxParam uint) (bestParam uint, bits uint64) {
	n := len(res)
	if n == 0 {
		return 0, 0
	}
	if maxParam > riceMaxParam5 {
		maxParam = riceMaxParam5
	}
	var sums [riceMaxParam5 + 1]uint64
	if maxParam < riceParamCount {
		// SIMD fast path: the kernel always computes the 15-wide FLAC range;
		// the scan below only reads sums[:maxParam+1].
		riceSumsI32(sums[:riceParamCount], res)
	} else {
		// 5-bit range: the wide kernel computes the full 31-column range on the
		// SIMD path; the scan below only reads sums[:maxParam+1].
		riceSumsWideI32(sums[:], res)
	}
	return riceScan(sums[:maxParam+1], n)
}

// riceScan returns the cost-minimizing Rice parameter over sums and its bit
// cost, where bits(k) = sums[k] + n*(k+1). The smallest k wins ties.
func riceScan(sums []uint64, n int) (bestParam uint, bits uint64) {
	bits = sums[0] + uint64(n) // n*(0+1)
	for k := 1; k < len(sums); k++ {
		c := sums[k] + uint64(n)*uint64(k+1)
		if c < bits {
			bits = c
			bestParam = uint(k)
		}
	}
	return bestParam, bits
}
