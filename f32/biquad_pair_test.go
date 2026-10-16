//go:build amd64 || arm64

package f32

import (
	"testing"
)

// pairPackLaneOf returns the section and channel a lane of the pair image
// holds, and whether the lane is used at all for the given section count.
func pairPackLaneOf(lane, sections int) (section, ch int, used bool) {
	return lane / biquadPairChannels, lane % biquadPairChannels, lane < biquadPairChannels*sections
}

// checkPackPairImage compares a packed pair image with the layout the kernels
// read.
func checkPackPairImage(t *testing.T, l *biquadLanes, sections int, coeffs []float32, st [biquadPairChannels][]float32, sel0, sel1 uint32) {
	t.Helper()
	for lane := range biquadMaxLanes {
		k, ch, used := pairPackLaneOf(lane, sections)
		for r := range biquadCoeffs {
			want := float32(0)
			if used {
				want = coeffs[biquadCoeffs*k+r]
			}
			if got := l.co[r*biquadMaxLanes+lane]; got != want {
				t.Errorf("sections=%d co row %d lane %d = %v, want %v", sections, r, lane, got, want)
			}
		}
		for r := range biquadStateLen {
			want := float32(0)
			if used {
				want = st[ch][biquadStateLen*k+r]
			}
			if got := l.st[r*biquadMaxLanes+lane]; got != want {
				t.Errorf("sections=%d st row %d lane %d = %v, want %v", sections, r, lane, got, want)
			}
		}
	}
	for i, w := range l.sel {
		want := sel0
		if i%2 == 1 {
			want = sel1
		}
		if w != want {
			t.Errorf("sections=%d sel[%d] = %#x, want %#x", sections, i, w, want)
		}
	}
}

// TestBiquadLanesPackPair pins the pair image layout: lane 2k+ch of row r holds
// section k of channel ch, coefficients are repeated in both channel lanes of a
// section, the selector alternates the two words, and lanes at or above
// 2*sections stay zero. Every value is distinct, so any swapped row, lane or
// channel is visible.
func TestBiquadLanesPackPair(t *testing.T) {
	const sel0, sel1 = 0xA0, 0xB1
	for _, sections := range []int{1, 2} {
		coeffs := make([]float32, biquadCoeffs*sections)
		for i := range coeffs {
			coeffs[i] = float32(100 + i)
		}
		st := [biquadPairChannels][]float32{make([]float32, biquadStateLen*sections), make([]float32, biquadStateLen*sections)}
		for ch := range st {
			for i := range st[ch] {
				st[ch][i] = float32(200 + 100*ch + i)
			}
		}
		var l biquadLanes
		l.packPair(coeffs, st[0], st[1], sel0, sel1)

		checkPackPairImage(t, &l, sections, coeffs, st, sel0, sel1)

		// unpackPair writes back lanes below 2*sections only, to the right channel.
		for i := range l.st {
			l.st[i] += 1000
		}
		got := [biquadPairChannels][]float32{make([]float32, len(st[0])), make([]float32, len(st[1]))}
		l.unpackPair(got[0], got[1])
		for ch := range got {
			for i := range got[ch] {
				if want := st[ch][i] + 1000; got[ch][i] != want {
					t.Errorf("sections=%d unpackPair ch=%d [%d] = %v, want %v", sections, ch, i, got[ch][i], want)
				}
			}
		}
	}
}
