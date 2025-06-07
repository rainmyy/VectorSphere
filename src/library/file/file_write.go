package file

import (
	"io"
	"os"
)

func (f *File) BackupFile(dstFile string) (res int64, err error) {
	src, err := os.Open(f.name)
	defer func(src *os.File) {
		err := src.Close()
		if err != nil {

		}
	}(src)
	if err != nil {
		return
	}
	dst, err := os.OpenFile(dstFile, os.O_WRONLY|os.O_CREATE, 0644)
	defer dst.Close()
	if err != nil {
		return
	}
	return io.Copy(dst, src)
}
