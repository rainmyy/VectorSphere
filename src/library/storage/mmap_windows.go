//go:build windows

package storage

import (
	"errors"
	"os"
	"syscall"
	"unsafe"
)

func NewMmap(fileName string, mode int) (*Mmap, error) {
	mmap := &Mmap{
		MmapBytes:   nil,
		FileName:    fileName,
		FileLen:     0,
		MapType:     0,
		FilePointer: 0,
		Filed:       nil,
	}
	fileMode := os.O_RDWR
	if mode == MODE_CREATE {
		fileMode = os.O_RDWR | os.O_CREATE | os.O_TRUNC
	}
	f, err := os.OpenFile(fileName, fileMode, 0664)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	mmap.FileLen = fi.Size()
	if mode == MODE_CREATE || mmap.FileLen == 0 {
		// 扩展文件
		err = syscall.Ftruncate(syscall.Handle(f.Fd()), mmap.FileLen+APPEND_DATA)
		if err != nil {
			_ = f.Close()
			return nil, err
		}
		mmap.FileLen = APPEND_DATA
	}
	h, err := syscall.CreateFileMapping(
		syscall.Handle(f.Fd()),
		nil,
		syscall.PAGE_READWRITE,
		0,
		uint32(mmap.FileLen),
		nil)
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_WRITE, 0, 0, uintptr(mmap.FileLen))
	_ = syscall.CloseHandle(h)
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	// 转换为[]byte
	mmap.MmapBytes = *(*[]byte)(unsafe.Pointer(&addr))
	mmap.Filed = f
	return mmap, nil
}

func (m *Mmap) checkFilePointer(checkValue int64) error {
	if m.FilePointer+checkValue < m.FileLen {
		return nil
	}

	err := syscall.Ftruncate(syscall.Handle(m.Filed.Fd()), m.FileLen+APPEND_DATA)
	if err != nil {
		return err
	}
	m.FileLen += APPEND_DATA
	h, err := syscall.CreateFileMapping(syscall.Handle(m.Filed.Fd()), nil, syscall.PAGE_READWRITE, 0, uint32(m.FileLen), nil)
	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_WRITE, 0, 0, uintptr(m.FileLen))
	err = syscall.CloseHandle(h)
	if err != nil {
		return err
	}
	m.MmapBytes = *(*[]byte)(unsafe.Pointer(&addr))

	return nil
}

func (m *Mmap) checkFileCap(start, len int64) error {
	if start+len < m.FileLen {
		return nil
	}
	err := syscall.Ftruncate(syscall.Handle(m.Filed.Fd()), m.FileLen+APPEND_DATA)
	if err != nil {
		return err
	}
	m.FileLen += APPEND_DATA
	m.FilePointer = start + len

	return nil
}

func (m *Mmap) Unmap() error {
	if m.MmapBytes != nil && len(m.MmapBytes) > 0 {
		addr := uintptr(unsafe.Pointer(&m.MmapBytes[0]))
		err := syscall.UnmapViewOfFile(addr)
		if err != nil {
			return err
		}
	}
	if m.Filed != nil {
		return m.Filed.Close()
	}
	return nil
}

func (m *Mmap) Sync() error {
	if m.MmapBytes == nil || len(m.MmapBytes) == 0 {
		return nil
	}
	addr := uintptr(unsafe.Pointer(&m.MmapBytes[0]))
	err := syscall.FlushViewOfFile(addr, uintptr(len(m.MmapBytes)))
	if err != nil {
		return errors.New("FlushViewOfFile failed:" + err.Error())
	}
	if m.Filed != nil {
		h := syscall.Handle(m.Filed.Fd())
		err := syscall.FlushFileBuffers(h)
		if err != nil {
			return err
		}
	}
	return nil
}
