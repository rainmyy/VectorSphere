package conf

import (
	"gopkg.in/yaml.v2"
	"os"
)

// ReadYAML 读取YAML文件
func ReadYAML(path string, out interface{}) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return yaml.Unmarshal(data, out)
}

// WriteYAML 写入YAML文件
func WriteYAML(path string, in interface{}) error {
	data, err := yaml.Marshal(in)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}
