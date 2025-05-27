//go:build linux

package strategy

import (
	"os"
	"syscall"
)

func NewMmap(fileName string, mode int) (*Mmap, error) {
	mmap := &Mmap{
		MmapBytes:   make([]byte, 0),
		FileName:    fileName,
		FileLen:     0,
		MapType:     0,
		FilePointer: 0,
		Filed:       nil,
	}
	fileMode := os.O_RDWR
	fileCreateMode := os.O_RDWR | os.O_RDWR | os.O_TRUNC
	if mode == MODE_CREATE {
		fileMode = os.O_RDWR | os.O_CREATE | os.O_TRUNC
	}
	f, err := os.OpenFile(fileName, fileMode, 0664)
	if err != nil {
		f, err = os.OpenFile(fileName, fileCreateMode, 0664)
		if err != nil {
			return nil, err
		}
	}
	fi, err := f.Stat()
	if err != nil {

	}
	mmap.FileLen = fi.Size()
	if mode == MODE_CREATE || mmap.FileLen == 0 {
		syscall.Ftruncate(int(f.Fd()), fi.Size()+APPEND_DATA)
		mmap.FileLen = APPEND_DATA
	}
	mmap.MmapBytes, err = syscall.Mmap(int(f.Fd()), 0, int(mmap.FileLen), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return nil, err
	}

	mmap.Filed = f
	return mmap, nil
}

func (this *Mmap) checkFilePointer(checkValue int64) error {
	if this.FilePointer+checkValue < this.FileLen {
		return nil
	}
	//sysType := runtime.GOOS
	err := syscall.Ftruncate(int(this.Filed.Fd()), this.FileLen+APPEND_DATA)
	if err != nil {
		return err
	}
	this.FileLen += APPEND_DATA
	syscall.Munmap(this.MmapBytes)
	this.MmapBytes, err = syscall.Mmap(int(this.Filed.Fd()), 0, int(this.FileLen), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return err
	}
	return nil
}

func (this *Mmap) checkFileCap(start, len int64) error {
	if start+len < this.FileLen {
		return nil
	}
	err := syscall.Ftruncate(int(this.Filed.Fd()), this.FileLen+APPEND_DATA)
	if err != nil {
		return err
	}
	this.FileLen += APPEND_DATA
	this.FilePointer = start + len

	return nil
}

func (this *Mmap) Unmap() error {

	syscall.Munmap(this.MmapBytes)
	this.Filed.Close()
	return nil
}

func (this *Mmap) Sync() error {
	dh := this.header()
	_, _, err := syscall.Syscall(syscall.SYS_MSYNC, dh.Data, uintptr(dh.Len), syscall.MS_SYNC)
	if err != 0 {
		fmt.Printf("Sync Error ")
		return errors.New("Sync Error")
	}
	return nil
}
