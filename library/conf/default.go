package conf

import (
	"fmt"
	"sync"

	. "seetaSearch/library/bind"
	. "seetaSearch/library/common"
	. "seetaSearch/library/file"
)

// DefaultConf /**
type DefaultConf struct {
	m     *sync.RWMutex
	Key   string         `bind:"must"`
	Name  string         `bind:"should"`
	Child []*DefaultConf `bind:"must"`
}

func (conf *DefaultConf) Init() *DefaultConf {
	confName := "./conf/idc/bj/service.yaml"
	fileObj, err := Instance(confName)
	if err != nil {
		fmt.Print(err.Error())
		return nil
	}

	err = fileObj.Parser(IniType)
	if err != nil {
		print(err.Error())
		return nil
	}

	str := StringInstance()
	//array := ArrayInterface()
	bindData := DefaultBind(fileObj.GetContent(), str)
	fmt.Print(bindData)
	//bytes, err := json.Marshal(bindData)
	//fmt.Print(string(bytes))
	return conf
}

func NewConf() *DefaultConf {
	return new(DefaultConf)
}
