package conf

import (
	"fmt"
	"sync"

	. "github.com/rainmyy/seetaSearch/library/bind"
	. "github.com/rainmyy/seetaSearch/library/common"
	. "github.com/rainmyy/seetaSearch/library/file"
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

	str := StrigInstance()
	//array := ArrayInterface()
	bindData := DefaultBind(fileObj.GetContent(), str)
	fmt.Print(bindData)
	//bytes, err := json.Marshal(bindData)
	//fmt.Print(string(bytes))
	return conf
}

func Intance() *DefaultConf {
	return new(DefaultConf)
}
