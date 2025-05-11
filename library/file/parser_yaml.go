package file

import (
	"log"

	. "github.com/rainmyy/seetaSearch/library/bind"
	. "github.com/rainmyy/seetaSearch/library/strategy"
	"gopkg.in/yaml.v2"
)

func ParserYamlContent(data []byte) ([]*TreeStruct, error) {
	// 存储解析数据
	result := make(map[string]interface{})
	// 执行解析
	err := yaml.Unmarshal([]byte(data), &result)
	if err != nil {
		log.Fatalf("error: %v", err)
	}
	array := ArrayInterface()
	array.SetMap(result)
	return array.UnBind()
}
