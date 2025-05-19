//go:build !linux
// +build !linux

package strategy

import (
	"encoding/binary"
	"os"
	"runtime"
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

func (m *Mmap) SetFileEnd(fileLen int64) {
	m.FilePointer = fileLen
}

func (m *Mmap) checkFilePointer(checkValue int64) error {
	if m.FilePointer+checkValue < m.FileLen {
		return nil
	}
	sysType := runtime.GOOS
	err := syscall.Ftruncate(int(m.Filed.Fd()), m.FileLen+APPEND_DATA)
	if err != nil {
		return err
	}
	m.FileLen += APPEND_DATA
	syscall.Munmap(m.MmapBytes)
	m.MmapBytes, err = syscall.Mmap(int(m.Filed.Fd()), 0, int(m.FileLen), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return err
	}
	return nil
}

func (m *Mmap) checkFileCap(start, len int64) error {
	if start+len < m.FileLen {
		return nil
	}
	err := syscall.Ftruncate(int(m.Filed.Fd()), m.FileLen+APPEND_DATA)
	if err != nil {
		return err
	}
	m.FileLen += APPEND_DATA
	m.FilePointer = start + len

	return nil
}

func (m *Mmap) isEndOfFile(start int64) bool {
	if m.FilePointer == start {
		return true
	}

	return false
}

func (m *Mmap) ReadInt64(start int64) int64 {
	return int64(binary.LittleEndian.Uint64(m.MmapBytes[start : start+8]))
}

func (m *Mmap) ReadUint64(start uint64) uint64 {
	return binary.LittleEndian.Uint64(m.MmapBytes[start : start+8])
}
