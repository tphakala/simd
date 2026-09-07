package i16

import (
	"math"
	"testing"
)

func TestSum(t *testing.T) {
	// Literal cases.
	if got := Sum(nil); got != 0 {
		t.Errorf("Sum(nil) = %d, want 0", got)
	}
	if got := Sum([]int16{}); got != 0 {
		t.Errorf("Sum(empty) = %d, want 0", got)
	}
	if got := Sum([]int16{32767, 32767, 1, -32768}); got != 32767 {
		t.Errorf("Sum([32767,32767,1,-32768]) = %d, want 32767", got)
	}

	// Parity across the sweep against the reference and an independent int64
	// oracle (which cannot wrap at these lengths, so it also pins that the
	// running total is not narrowed to int16 mid-reduction).
	for _, n := range tier3Lengths {
		a := genI16(n, 91)
		got := Sum(a)
		if want := sumGo(a); got != want {
			t.Errorf("Sum n=%d: got %d, want %d (reference)", n, got, want)
		}
		var oracle int64
		for _, v := range a {
			oracle += int64(v)
		}
		if int64(got) != oracle {
			t.Errorf("Sum n=%d: got %d, want %d (int64 oracle)", n, got, oracle)
		}
	}

	// int32 accumulation must not narrow before the total: 300 elements of 30000
	// sum to 9_000_000, which exceeds int16 but fits int32.
	big := make([]int16, 300)
	for i := range big {
		big[i] = 30000
	}
	if got, want := Sum(big), int32(300*30000); got != want {
		t.Errorf("Sum(300x30000) = %d, want %d", got, want)
	}

	// int32 two's-complement wraparound: 70000 elements of 32767 sum to
	// 2_293_690_000, which overflows int32. The result must wrap exactly like the
	// reference (verified here against an independent int64->int32 truncation).
	const wn = 70000
	wa := make([]int16, wn)
	for i := range wa {
		wa[i] = math.MaxInt16
	}
	wantWrap := int32(int64(len(wa)) * math.MaxInt16) // non-constant: truncates at runtime
	if wantWrap >= 0 {
		t.Fatalf("test setup: expected wraparound to a negative value, got %d", wantWrap)
	}
	if got := Sum(wa); got != wantWrap {
		t.Errorf("Sum wraparound = %d, want %d", got, wantWrap)
	}
}

// TestSum_AllocFree: buffers INSIDE the closure, see TestMaxAbs_AllocFree.
func TestSum_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(50, func() {
		var a [240]int16
		_ = Sum(a[:])
	}); n != 0 {
		t.Errorf("Sum forces %v caller allocations per run, want 0", n)
	}
}
