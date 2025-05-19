package strategy

import (
	"encoding/binary"
	"os"
)

type Mmap struct {
	MmapBytes   []byte
	FileName    string
	FileLen     int64
	FilePointer int64
	MapType     int64
	Filed       *os.File
}

const APPEND_DATA int64 = 1024 * 1024
const (
	MODE_APPEND = iota
	MODE_CREATE
)

func (m *Mmap) SetFileEnd(fileLen int64) {
	m.FilePointer = fileLen
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
