package storage

import (
	"encoding/binary"
	"os"
	"reflect"
	"unsafe"
)

type DocIdNode struct {
	Docid  uint32
	Weight uint32
	//Pos    uint32
}
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

func (m *Mmap) ReadUInt64Array(start, len uint64) []DocIdNode {

	array := *(*[]DocIdNode)(unsafe.Pointer(&reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&m.MmapBytes[start])),
		Len:  int(len),
		Cap:  int(len),
	}))
	return array
}

func (m *Mmap) ReadDocIdsArray(start, len uint64) []DocIdNode {

	array := *(*[]DocIdNode)(unsafe.Pointer(&reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&m.MmapBytes[start])),
		Len:  int(len),
		Cap:  int(len),
	}))
	return array
}

func (m *Mmap) ReadString(start, lens int64) string {

	return string(m.MmapBytes[start : start+lens])
}

func (m *Mmap) Read(start, end int64) []byte {

	return m.MmapBytes[start:end]
}

func (m *Mmap) Write(start int64, buffer []byte) error {

	copy(m.MmapBytes[start:int(start)+len(buffer)], buffer)

	return nil //this.MmapBytes[start:end]
}

func (m *Mmap) WriteUInt64(start int64, value uint64) error {

	binary.LittleEndian.PutUint64(m.MmapBytes[start:start+8], value)

	return nil //this.Sync()
}

func (m *Mmap) WriteInt64(start, value int64) error {
	binary.LittleEndian.PutUint64(m.MmapBytes[start:start+8], uint64(value))
	return nil //this.Sync()
}

func (m *Mmap) AppendInt64(value int64) error {

	if err := m.checkFilePointer(8); err != nil {
		return err
	}
	binary.LittleEndian.PutUint64(m.MmapBytes[m.FilePointer:m.FilePointer+8], uint64(value))
	m.FilePointer += 8
	return nil //this.Sync()
}

func (m *Mmap) AppendUInt64(value uint64) error {

	if err := m.checkFilePointer(8); err != nil {
		return err
	}

	binary.LittleEndian.PutUint64(m.MmapBytes[m.FilePointer:m.FilePointer+8], value)
	m.FilePointer += 8
	return nil //this.Sync()
}

func (m *Mmap) AppendStringWithLen(value string) error {
	err := m.AppendInt64(int64(len(value)))
	if err != nil {
		return err
	}
	err = m.AppendString(value)
	if err != nil {
		return err
	}
	return nil //this.Sync()

}

func (m *Mmap) AppendDetail(shard uint64, value string) error {
	err := m.AppendUInt64(shard)
	if err != nil {
		return err
	}
	err = m.AppendInt64(int64(len(value)))
	if err != nil {
		return err
	}
	err = m.AppendString(value)
	if err != nil {
		return err
	}
	return nil //this.Sync()
}

func (m *Mmap) AppendString(value string) error {

	lens := int64(len(value))
	if err := m.checkFilePointer(lens); err != nil {
		return err
	}

	dst := m.MmapBytes[m.FilePointer : m.FilePointer+lens]
	copy(dst, value)
	m.FilePointer += lens
	return nil //this.Sync()

}

func (m *Mmap) AppendBytes(value []byte) error {
	lens := int64(len(value))
	if err := m.checkFilePointer(lens); err != nil {
		return err
	}
	dst := m.MmapBytes[m.FilePointer : m.FilePointer+lens]
	copy(dst, value)

	m.FilePointer += lens
	return nil //this.Sync()

}

func (m *Mmap) WriteBytes(start int64, value []byte) error {
	lens := int64(len(value))
	dst := m.MmapBytes[start : start+lens]
	copy(dst, value)
	return nil //this.Sync()
}

func (m *Mmap) GetPointer() int64 {
	return m.FilePointer
}

func (m *Mmap) header() *reflect.SliceHeader {
	return (*reflect.SliceHeader)(unsafe.Pointer(&m.MmapBytes))
}

func (m *Mmap) AppendStringWith32Bytes(value string, lens int64) error {

	err := m.AppendInt64(lens)
	if err != nil {
		return err
	}
	if err := m.checkFilePointer(32); err != nil {
		return err
	}
	dst := m.MmapBytes[m.FilePointer : m.FilePointer+32]
	copy(dst, value)
	m.FilePointer += 32
	return nil //this.Sync()
}

func (m *Mmap) ReadStringWith32Bytes(start int64) string {

	lens := m.ReadInt64(start)
	return m.ReadString(start+8, lens)

}

func (m *Mmap) WriteStringWith32Bytes(start int64, value string, lens int64) error {

	err := m.WriteInt64(start, lens)
	if err != nil {
		return err
	}
	err = m.WriteBytes(start+4, []byte(value))
	if err != nil {
		return err
	}
	return nil
}
