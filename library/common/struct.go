package common

import (
	"fmt"
	"reflect"
	"unsafe"

	"seetaSearch/library/strategy"
)

func TreeStruct2Bytes(treeList []*strategy.TreeStruct) [][]byte {
	var byteList = make([][]byte, 0)
	for _, tree := range treeList {
		var sli reflect.SliceHeader
		sli.Len = int(unsafe.Sizeof(tree))
		sli.Cap = int(unsafe.Sizeof(tree))
		sli.Data = uintptr(unsafe.Pointer(&tree))
		bytes := *(*[]byte)(unsafe.Pointer(&sli))
		byteList = append(byteList, bytes)
	}

	return byteList
}

func Bytes2TreeStruct(b [][]byte) []*strategy.TreeStruct {
	var resList []*strategy.TreeStruct
	for _, val := range b {
		treeStruct := (*strategy.TreeStruct)(unsafe.Pointer(
			(*reflect.SliceHeader)(unsafe.Pointer(&val)).Data,
		))
		for _, val := range treeStruct.GetNode() {
			fmt.Print(val)
		}
		print(len(treeStruct.GetNode()))
		resList = append(resList, treeStruct)
	}
	return resList
}
