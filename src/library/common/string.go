package common

import (
	"reflect"
	"strconv"
	"strings"
	"unsafe"
)

func Str2bytes(s string) []byte {
	x := (*[2]uintptr)(unsafe.Pointer(&s))
	h := [3]uintptr{x[0], x[1], x[1]}
	return *(*[]byte)(unsafe.Pointer(&h))
}

func Bytes2str(b []byte) string {
	return *(*string)(unsafe.Pointer(&b))
}
func String2bytes(s string) []byte {
	stringHeader := (*reflect.StringHeader)(unsafe.Pointer(&s))

	bh := reflect.SliceHeader{
		Data: stringHeader.Data,
		Len:  stringHeader.Len,
		Cap:  stringHeader.Len,
	}

	return *(*[]byte)(unsafe.Pointer(&bh))
}

func Bytes2string(b []byte) string {
	sliceHeader := (*reflect.SliceHeader)(unsafe.Pointer(&b))

	sh := reflect.StringHeader{
		Data: sliceHeader.Data,
		Len:  sliceHeader.Len,
	}

	return *(*string)(unsafe.Pointer(&sh))
}

// VBEncode variable byte
func VBEncode(n uint32) string {
	var bytes []uint32
	for {
		bytes = append(bytes, (n+1)%128)
		if n < 128 {
			break
		}
		n /= 128
	}
	var by []string
	for i := len(bytes) - 1; i >= 0; i-- {
		if i < len(bytes)-1 {
			by = append(by, strconv.FormatUint(uint64(bytes[i]), 2)[1:]+" ")
		} else {
			by = append(by, strconv.FormatUint(uint64(bytes[i]), 2))
		}
	}
	return strings.Join(by, "")
}
