//go:build !windows
// +build !windows

package vector

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
)

// getAvailableDiskSpaceUnix 在Unix系统上使用系统调用获取真实的磁盘空间
// 这个函数会覆盖主文件中的同名函数（仅在Unix系统编译时）
func (c *MultiLevelCache) getAvailableDiskSpaceUnix(path string) (int64, error) {
	// 在Windows环境下，Unix函数不应该被调用
	if runtime.GOOS == "windows" {
		fmt.Printf("Warning: 在Windows系统上调用了Unix磁盘空间检测函数\n")
		return c.getAvailableDiskSpaceFallback(path)
	}
	
	// 使用syscall.Statfs获取文件系统统计信息
	var stat syscall.Statfs_t
	err := syscall.Statfs(path, &stat)
	if err != nil {
		// 如果系统调用失败，使用fallback方法
		fmt.Printf("Warning: 系统调用获取磁盘空间失败: %v，使用fallback方法\n", err)
		return c.getAvailableDiskSpaceFallbackUnix(path)
	}
	
	// 计算可用空间：可用块数 * 块大小
	availableBytes := int64(stat.Bavail) * int64(stat.Bsize)
	fmt.Printf("Info: 使用系统调用获取磁盘空间: %d bytes (%.2f GB)\n", 
		availableBytes, float64(availableBytes)/(1024*1024*1024))
	return availableBytes, nil
}

// getAvailableDiskSpaceFallbackUnix Unix系统的fallback实现
func (c *MultiLevelCache) getAvailableDiskSpaceFallbackUnix(path string) (int64, error) {
	// 使用os包的功能来估算磁盘空间
	tempFile := filepath.Join(path, ".temp_unix_space_check")
	
	// 尝试创建一个测试文件
	file, err := os.Create(tempFile)
	if err != nil {
		return 0, fmt.Errorf("无法在路径 %s 创建测试文件: %v", path, err)
	}
	file.Close()
	
	// 立即删除测试文件
	os.Remove(tempFile)
	
	// 对于Unix系统，返回一个较大的估计值
	fmt.Printf("Info: 在Unix系统上使用简化的磁盘空间检测\n")
	return 2 * 1024 * 1024 * 1024, nil // 假设有2GB可用空间
}