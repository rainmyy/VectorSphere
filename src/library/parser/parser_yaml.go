package parser

import (
	"VectorSphere/src/library/bind"
	. "VectorSphere/src/library/tree"
	"fmt"
	"gopkg.in/yaml.v2"
	"log"
)

func ParserYamlContent(data []byte) ([]*TreeStruct, error) {
	if len(data) == 0 {
		return nil, nil
	}
	var obj interface{}
	err := yaml.Unmarshal(data, &obj)
	if err != nil {
		return nil, err
	}
	return yamlToTree(obj), nil
}

func yamlToTree(obj interface{}) []*TreeStruct {
	switch v := obj.(type) {
	case map[interface{}]interface{}:
		var trees []*TreeStruct
		for key, val := range v {
			tree := TreeInstance()
			node := NodeInstance([]byte(fmt.Sprintf("%v", key)), []byte{})
			tree.SetNode(node)
			children := yamlToTree(val)
			for _, child := range children {
				tree.SetChildren(child)
			}
			trees = append(trees, tree)
		}
		return trees
	case []interface{}:
		var trees []*TreeStruct
		for _, item := range v {
			childs := yamlToTree(item)
			trees = append(trees, childs...)
		}
		return trees
	default:
		tree := TreeInstance()
		node := NodeInstance([]byte{}, []byte(fmt.Sprintf("%v", v)))
		tree.SetNode(node)
		return []*TreeStruct{tree}
	}
}

func ParserYaml(data []byte) ([]*TreeStruct, error) {
	// 存储解析数据
	result := make(map[interface{}]interface{})
	// 执行解析
	err := yaml.Unmarshal(data, &result)
	if err != nil {
		log.Fatalf("error: %v", err)
	}
	for k, _ := range result {
		println(k)
	}
	array := bind.ArrayInterface()
	array.SetMap(result)
	return array.UnBind()
}
