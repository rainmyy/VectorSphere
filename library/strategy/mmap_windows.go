//go:build !windows
// +build !windows

package strategy

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
	err = syscall.CloseHandle(syscall.Handle(h))
	if err != nil {
		return err
	}
	m.MmapBytes = *(*[]byte)(unsafe.Pointer(addr))

	return nil
}
