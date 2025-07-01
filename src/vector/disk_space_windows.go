//go:build windows

package vector

import (
	"fmt"
	"syscall"
	"unsafe"
)

// getAvailableDiskSpaceWindowsNative 在Windows系统上使用Windows API获取真实的磁盘空间
func (c *MultiLevelCache) getAvailableDiskSpaceWindowsNative(path string) (int64, error) {
	// 使用Windows API GetDiskFreeSpaceEx
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	getDiskFreeSpaceEx := kernel32.NewProc("GetDiskFreeSpaceExW")

	// 将路径转换为UTF-16
	pathPtr, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return 0, fmt.Errorf("路径转换失败: %v", err)
	}

	var freeBytesAvailable uint64
	var totalNumberOfBytes uint64
	var totalNumberOfFreeBytes uint64

	// 调用Windows API
	ret, _, err := getDiskFreeSpaceEx.Call(
		uintptr(unsafe.Pointer(pathPtr)),
		uintptr(unsafe.Pointer(&freeBytesAvailable)),
		uintptr(unsafe.Pointer(&totalNumberOfBytes)),
		uintptr(unsafe.Pointer(&totalNumberOfFreeBytes)),
	)

	if ret == 0 {
		return 0, fmt.Errorf("GetDiskFreeSpaceEx调用失败: %v", err)
	}

	// 返回用户可用的字节数
	return int64(freeBytesAvailable), nil
}
