package bind

import (
	. "VectorSphere/library/tree"
	"fmt"
)

type Array struct {
	length int
	value  []map[interface{}]interface{}
}

func ArrayInterface() *Array {
	return &Array{value: make([]map[interface{}]interface{}, 0)}
}

// Bind /**
func (a *Array) Bind(treeList []*TreeStruct) {
	var treeMapList = make([]map[interface{}]interface{}, 0)
	var getBindMap func(tree []*TreeStruct) []map[interface{}]interface{}
	/***
	* 递归方式获取
	 */
	getBindMap = func(tree []*TreeStruct) []map[interface{}]interface{} {
		if len(tree) == 0 {
			return nil
		}
		var treeMapList = make([]map[interface{}]interface{}, 0)
		for _, val := range tree {
			nodeList := val.GetNode()
			node := nodeList[0]
			var treeMap = make(map[interface{}]interface{})
			if val.IsLeaf() {
				treeMap[string(node.GetName())] = string(node.GetData())
			} else {
				childrenNum := len(val.GetChildren())
				nodeName := string(node.GetName())
				if childrenNum > 1 {
					treeSlice := getBindMap(val.GetChildren())
					if len(treeSlice) == 0 {
						continue
					}
					treeMap[nodeName] = treeSlice
				} else {
					res := getBindMap(val.GetChildren())
					treeMap[nodeName] = make(map[string]interface{})
					if len(res) == 0 {
						treeMap[nodeName] = nil
					} else {
						treeMap[nodeName] = res[0]
					}
				}
				if len(treeMap) == 0 {
					continue
				}
			}
			treeMapList = append(treeMapList, treeMap)
		}
		return treeMapList
	}
	treeMapList = getBindMap(treeList)
	a.value = treeMapList
	a.length = len(a.value)
}

func (a *Array) GetValue() interface{} {
	return a.value
}

func (a *Array) SetMap(m map[interface{}]interface{}) {
	a.value = append(a.value, m)
	a.length = len(a.value)
}

func (a *Array) UnBind() ([]*TreeStruct, error) {
	arr := a.value
	if arr == nil {
		return nil, fmt.Errorf("array value is nil")
	}
	//for _, v := range arr {
	//	for kk, _ := range v {
	//		println(kk)
	//	}
	//
	//}
	return nil, nil
}
