//go:build linux

package file

import (
	"os"
	"syscall"
)

// MmapFile 文件读取
func MmapFile(fileName string, size int) ([]byte, error) {
	fd, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}
	defer fd.Close()
	return syscall.Mmap(int(fd.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
}

func MunmapFile(data []byte) {
	syscall.Munmap(data)
}
