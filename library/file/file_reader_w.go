//go:build windows

package file

import (
	"os"
	"syscall"
	"unsafe"
)

const defaultMaxFileSize = 1 << 30 // 假设文件最大为 1G
const defaultMemMapSize = 128 * (1 << 20)

func MmapFile(fileName string, size int) ([]byte, error) {
	f, err := os.OpenFile(fileName, os.O_RDONLY, 0644)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	h, err := syscall.CreateFileMapping(syscall.Handle(f.Fd()), nil, syscall.PAGE_READWRITE, 0, defaultMemMapSize, nil)
	if err != nil {
		return nil, err
	}

	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_WRITE, 0, 0, uintptr(defaultMemMapSize))
	if err != nil {
		return nil, err
	}

	err = syscall.CloseHandle(h)
	if err != nil {
		return nil, err
	}

	// Convert to a byte array.
	data := (*[defaultMaxFileSize]byte)(unsafe.Pointer(addr))

	return data[:size], nil
}

func MunmapFile(data []byte) {
	addr := (uintptr)(unsafe.Pointer(&data[0]))
	syscall.UnmapViewOfFile(addr)
}
