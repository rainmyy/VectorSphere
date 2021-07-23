package bind

import (
	. "bytes"

	. "github.com/rainmyy/easyDB/library/common"
	. "github.com/rainmyy/easyDB/library/strategy"
)

type String struct {
	value *Buffer
}

func (s *String) Bind(treeList []*TreeStruct) {
	var buffer = NewBuffer([]byte{})
	if len(treeList) > 1 {
		buffer.WriteRune(LeftBracket)
	}
	BindString(treeList, buffer)
	if len(treeList) > 1 {
		buffer.WriteRune(RightBracket)
	}
	s.value = buffer
}

func (s *String) GetValue() interface{} {
	return s.value
}
func StrigInstance() *String {
	return new(String)
}

/**
* 序列化tree:[{"test":[{"params":[{"name":"name1"},{"key":"value"},{"count":{"value":"www"}}]},{"params":[{"name":"name2"},{"key":"value"}]}]}]
 */
func BindString(treeList []*TreeStruct, buffer *Buffer) int {
	if len(treeList) == 0 {
		return 0
	}
	childrenNum := 0
	leafNum := 0
	for _, val := range treeList {
		if val.IsLeaf() {
			leafNum++
		}
	}
	for key, val := range treeList {
		nodeList := val.GetNode()
		if len(nodeList) <= 0 {
			continue
		}
		node := nodeList[0]
		buffer.WriteRune(LeftRrance)
		if val.IsLeaf() {
			buffer.WriteString(formatBytes(node.GetName()))
			buffer.WriteRune(Colon)
			buffer.WriteString(formatBytes(node.GetData()))
		} else {
			childrenNum = len(val.GetChildren())
			buffer.WriteString(formatBytes(node.GetName()))
			buffer.WriteRune(Colon)
			if childrenNum > 1 {
				buffer.WriteRune(LeftBracket)
			}
			BindString(val.GetChildren(), buffer)
			if childrenNum > 1 {
				buffer.WriteRune(RightBracket)
			}
		}
		buffer.WriteRune(RightRrance)
		if key != len(treeList)-1 {
			buffer.WriteRune(Comma)
		}
	}
	return childrenNum
}

func formatBytes(bytes []byte) string {
	str := Bytes2string(bytes)
	if str == "" {
		return str
	}
	return "\"" + str + "\""
}

/**
* 解绑 字符串，将字符串转换成tre类型数据
 */
func (s *String) UnBind() []*TreeStruct {
	return nil
}
