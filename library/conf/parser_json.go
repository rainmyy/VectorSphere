package conf

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	. "seetaSearch/library/strategy"
)

func ParserJsonContent(data []byte) ([]*TreeStruct, error) {
	if len(data) == 0 {
		return nil, nil
	}
	var obj interface{}
	err := json.Unmarshal(data, &obj)
	if err != nil {
		return nil, err
	}
	return jsonToTree(obj, nil), nil
}

func jsonToTree(obj interface{}, parent *TreeStruct) []*TreeStruct {
	switch v := obj.(type) {
	case map[string]interface{}:
		var trees []*TreeStruct
		for key, val := range v {
			tree := TreeInstance()
			node := NodeInstance([]byte(key), []byte{})
			tree.SetNode(node)
			children := jsonToTree(val, tree)
			for _, child := range children {
				tree.SetChildren(child)
			}
			trees = append(trees, tree)
		}
		return trees
	case []interface{}:
		var trees []*TreeStruct
		for _, item := range v {
			childs := jsonToTree(item, parent)
			trees = append(trees, childs...)
		}
		return trees
	default:
		// 叶子节点
		tree := TreeInstance()
		if parent != nil && len(parent.GetNode()) > 0 {
			key := parent.GetNode()[0].GetName()
			node := NodeInstance(key, []byte(fmt.Sprintf("%v", v)))
			tree.SetNode(node)
		}
		return []*TreeStruct{tree}
	}
}
func LoadJson(filePath string, model interface{}) *interface{} {
	f, err := os.Open(filePath)
	if err != nil {
		panic(err)
	}
	r := io.Reader(f)
	if err = json.NewDecoder(r).Decode(&model); err != nil {
		panic(err)
	}
	err = f.Close()
	if err != nil {
		return nil
	}
	return &model
}

func DumpJson(filePath string, model interface{}) (bool, error) {
	fp, err := os.OpenFile(filePath, os.O_RDWR|os.O_CREATE, 0600)
	defer func(fp *os.File) {
		err := fp.Close()
		if err != nil {

		}
	}(fp)
	if err != nil {
		return false, err
	}
	data, err := json.Marshal(model)
	if err != nil {
		return false, err
	}
	_, err = fp.Write(data)
	if err != nil {
		return false, err
	}

	return true, nil
}
