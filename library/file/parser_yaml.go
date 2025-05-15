package file

import (
	"log"

	"gopkg.in/yaml.v2"
	. "seetaSearch/library/bind"
	. "seetaSearch/library/strategy"
)

func ParserYamlContent(data []byte) ([]*TreeStruct, error) {
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
	array := ArrayInterface()
	array.SetMap(result)
	return array.UnBind()
}
