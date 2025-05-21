package util

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

func InIntSliceSortedFunc(stack []int) func(int) bool {
	sort.Ints(stack)
	return func(needle int) bool {
		index := sort.SearchInts(stack, needle)
		return index < len(stack) && stack[index] == needle
	}
}

func GetProjectRoot() (string, error) {
	exePath, err := os.Executable()
	if err != nil {
		return "", err
	}
	dir := filepath.Dir(exePath)
	// 向上查找 go.mod 文件，找到即为项目根目录
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return "", fmt.Errorf("未找到项目根目录")
}
