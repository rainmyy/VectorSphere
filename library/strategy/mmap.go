package strategy

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

func (this *Mmap) SetFileEnd(fileLen int64) {
	this.FilePointer = fileLen
}

func (this *Mmap) isEndOfFile(start int64) bool {
	if this.FilePointer == start {
		return true
	}

	return false
}

func (this *Mmap) ReadInt64(start int64) int64 {
	return int64(binary.LittleEndian.Uint64(this.MmapBytes[start : start+8]))
}

func (this *Mmap) ReadUint64(start uint64) uint64 {
	return binary.LittleEndian.Uint64(this.MmapBytes[start : start+8])
}

func (this *Mmap) ReadUInt64Array(start, len uint64) []DocIdNode {

	array := *(*[]DocIdNode)(unsafe.Pointer(&reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&this.MmapBytes[start])),
		Len:  int(len),
		Cap:  int(len),
	}))
	return array
}

func (this *Mmap) ReadDocIdsArray(start, len uint64) []DocIdNode {

	array := *(*[]DocIdNode)(unsafe.Pointer(&reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&this.MmapBytes[start])),
		Len:  int(len),
		Cap:  int(len),
	}))
	return array
}

func (this *Mmap) ReadString(start, lens int64) string {

	return string(this.MmapBytes[start : start+lens])
}

func (this *Mmap) Read(start, end int64) []byte {

	return this.MmapBytes[start:end]
}

func (this *Mmap) Write(start int64, buffer []byte) error {

	copy(this.MmapBytes[start:int(start)+len(buffer)], buffer)

	return nil //this.MmapBytes[start:end]
}

func (this *Mmap) WriteUInt64(start int64, value uint64) error {

	binary.LittleEndian.PutUint64(this.MmapBytes[start:start+8], value)

	return nil //this.Sync()
}

func (this *Mmap) WriteInt64(start, value int64) error {
	binary.LittleEndian.PutUint64(this.MmapBytes[start:start+8], uint64(value))
	return nil //this.Sync()
}

func (this *Mmap) AppendInt64(value int64) error {

	if err := this.checkFilePointer(8); err != nil {
		return err
	}
	binary.LittleEndian.PutUint64(this.MmapBytes[this.FilePointer:this.FilePointer+8], uint64(value))
	this.FilePointer += 8
	return nil //this.Sync()
}

func (this *Mmap) AppendUInt64(value uint64) error {

	if err := this.checkFilePointer(8); err != nil {
		return err
	}

	binary.LittleEndian.PutUint64(this.MmapBytes[this.FilePointer:this.FilePointer+8], value)
	this.FilePointer += 8
	return nil //this.Sync()
}

func (this *Mmap) AppendStringWithLen(value string) error {
	this.AppendInt64(int64(len(value)))
	this.AppendString(value)
	return nil //this.Sync()

}

func (this *Mmap) AppendDetail(shard uint64, value string) error {
	this.AppendUInt64(shard)
	this.AppendInt64(int64(len(value)))
	this.AppendString(value)
	return nil //this.Sync()
}

func (this *Mmap) AppendString(value string) error {

	lens := int64(len(value))
	if err := this.checkFilePointer(lens); err != nil {
		return err
	}

	dst := this.MmapBytes[this.FilePointer : this.FilePointer+lens]
	copy(dst, value)
	this.FilePointer += lens
	return nil //this.Sync()

}

func (this *Mmap) AppendBytes(value []byte) error {
	lens := int64(len(value))
	if err := this.checkFilePointer(lens); err != nil {
		return err
	}
	dst := this.MmapBytes[this.FilePointer : this.FilePointer+lens]
	copy(dst, value)

	this.FilePointer += lens
	return nil //this.Sync()

}

func (this *Mmap) WriteBytes(start int64, value []byte) error {
	lens := int64(len(value))
	dst := this.MmapBytes[start : start+lens]
	copy(dst, value)
	return nil //this.Sync()
}

func (this *Mmap) GetPointer() int64 {
	return this.FilePointer
}

func (this *Mmap) header() *reflect.SliceHeader {
	return (*reflect.SliceHeader)(unsafe.Pointer(&this.MmapBytes))
}

func (this *Mmap) AppendStringWith32Bytes(value string, lens int64) error {

	err := this.AppendInt64(lens)
	if err != nil {
		return err
	}
	if err := this.checkFilePointer(32); err != nil {
		return err
	}
	dst := this.MmapBytes[this.FilePointer : this.FilePointer+32]
	copy(dst, value)
	this.FilePointer += 32
	return nil //this.Sync()
}

func (this *Mmap) ReadStringWith32Bytes(start int64) string {

	lens := this.ReadInt64(start)
	return this.ReadString(start+8, lens)

}

func (this *Mmap) WriteStringWith32Bytes(start int64, value string, lens int64) error {

	this.WriteInt64(start, lens)
	this.WriteBytes(start+4, []byte(value))
	return nil
}
