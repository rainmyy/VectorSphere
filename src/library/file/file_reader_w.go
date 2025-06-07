//go:build windows

package file

import (
	"VectorSphere/src/library/log"
	"fmt"
	"syscall"
	"unsafe"
)

const defaultMaxFileSize = 1 << 30 // 假设文件最大为 1G
const defaultMemMapSize = 128 * (1 << 20)

func MmapFile(fileName string, size int) ([]byte, error) {
	hFile, err := syscall.CreateFile(syscall.StringToUTF16Ptr(fileName), syscall.GENERIC_READ|syscall.GENERIC_WRITE, syscall.FILE_SHARE_READ, nil, syscall.OPEN_EXISTING, 0, 0)
	if err != nil {
		fmt.Println("Error opening file:", err)
		return nil, err
	}

	defer func(handle syscall.Handle) {
		err := syscall.CloseHandle(handle)
		if err != nil {
			log.Error("close handle error:", err)
		}
	}(hFile)

	h, err := syscall.CreateFileMapping(hFile, nil, syscall.PAGE_READWRITE, 0, 0, nil)
	if err != nil {
		return nil, err
	}

	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_READ, 0, 0, 0)
	if err != nil {
		return nil, err
	}

	defer func(handle syscall.Handle) {
		err := syscall.CloseHandle(handle)
		if err != nil {
			log.Error("close handle error:", err)
		}
	}(h)

	data := (*[defaultMaxFileSize]byte)(unsafe.Pointer(addr))

	return data[:size], nil
}

func MunmapFile(data []byte) {
	addr := (uintptr)(unsafe.Pointer(&data[0]))
	err := syscall.UnmapViewOfFile(addr)
	if err != nil {
		log.Error("UnmapViewOfFile failed:", err)
	}
}
