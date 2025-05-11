package file

import (
	"fmt"

	. "github.com/rainmyy/seetaSearch/library/common"
	. "github.com/rainmyy/seetaSearch/library/strategy"
)

// ParserIniContent /**
func ParserIniContent(data []byte) ([]*TreeStruct, error) {
	if data == nil {
		return nil, fmt.Errorf("content is nil")
	}
	var bytesList [][]byte

	hasSlash := false
	var bytes []byte
	if data[len(data)-1] != byte(LineBreak) {
		data = append(data, byte(LineBreak))
	}
	for i := 0; i < len(data); i++ {
		value := data[i]
		//filter the slash or hash or asterisk
		if value == byte(Slash) || value == byte(Hash) || value == byte(Asterisk) {
			hasSlash = true
			continue
		}
		if hasSlash {
			if value == byte(LineBreak) {
				hasSlash = false
			}
			continue
		}
		//cut out the data with linebreak or black
		if value != byte(LineBreak) && value != byte(Blank) {
			bytes = append(bytes, value)
		} else if len(bytes) > 0 {
			bytesList = append(bytesList, bytes)
			bytes = []byte{}
		}
	}
	if len(bytesList) == 0 {
		return nil, fmt.Errorf("bytes is empty")
	}
	//format the byte data with tree
	byteTreeList := initTreeFunc(bytesList)
	return byteTreeList, nil
}

/**
*实现树状结构
 */
func initTreeFunc(bytesList [][]byte) []*TreeStruct {
	currentTree := TreeInstance()
	//分隔符，91:'[' 46:'.' 58:'.'
	var segment = []int{int(LeftBracket), int(Period)}
	infunc := InIntSliceSortedFunc(segment)
	var rootTree = currentTree
	//根节点设置为1
	currentTree.SetHeight(1)
	for i := 0; i < len(bytesList); i++ {
		bytes := bytesList[i]
		bytesLen := len(bytes)
		if bytesLen == 0 {
			continue
		}
		tempNum := 0
		for j := 0; j < bytesLen; j++ {
			if infunc(int(bytes[j])) {
				tempNum++
			}
		}
		treeStruct := TreeInstance()
		currentHigh := currentTree.GetHeight()
		var nodeStruct *NodeStruct
		if tempNum > 0 && len(bytes) > tempNum {
			bytes = bytes[tempNum : bytesLen-1]

			nodeStruct = NodeInstance(bytes, []byte{})
			for tempNum < currentHigh {
				if currentTree != nil {
					currentTree = currentTree.GetParent()
					currentHigh = currentTree.GetHeight()
				} else {
					break
				}
			}
			treeStruct.SetNode(nodeStruct)
			treeStruct.SetParent(currentTree)
			if currentTree != nil {
				currentTree.SetChildren(treeStruct)
			}

			currentTree = treeStruct
		} else if tempNum == 0 {
			//type of key:value
			separatorPlace := SlicePlace(byte(Colon), bytes)
			if separatorPlace <= 0 {
				continue
			}
			key := bytes[0:separatorPlace]
			value := bytes[separatorPlace+1 : bytesLen]
			nodeStruct = NodeInstance(key, value)
			if currentTree == nil {
				continue
			}
			treeStruct.SetNode(nodeStruct)
			treeStruct.SetParent(currentTree)
			currentTree.SetChildren(treeStruct)
		}
	}
	return rootTree.GetChildren()
}
