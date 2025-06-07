package common

import "strings"

func SlicePlace(find byte, slice []byte) int {
	if len(slice) == 0 {
		return -1
	}
	for i := 0; i < len(slice); i++ {
		if find == slice[i] {
			return i
		}
	}
	return 0
}

// SplitSlice /**
func SplitSlice(num int, a []string) [][]string {
	var sli [][]string
	var slilen = 0
	alen := len(a)
	if alen%num == 0 {
		slilen = alen / num
	} else {
		slilen = int(alen/num) + 1
	}
	for i := 0; i < num; i++ {
		var start int
		var end int
		if i == 0 {
			start = 0
		} else {
			start = i * slilen
		}
		if (i+1)*slilen > alen {
			end = alen
		} else {
			end = (i + 1) * slilen
		}
		asli := a[start:end]
		if len(asli) == 0 {
			continue
		}
		sli = append(sli, asli)
	}
	return sli
}

/**
* remove the space or newline from string
 */
func trimSpace(s string) string {
	s = strings.Replace(s, " ", "", -1)
	s = strings.Replace(s, "\n"+"", "", -1)
	return s
}
