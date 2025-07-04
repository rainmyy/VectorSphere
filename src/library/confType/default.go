package conf

import (
	"sync"

	. "VectorSphere/src/library/bind"
	. "VectorSphere/src/library/file"
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
		return nil
	}

	err = fileObj.Parser()
	if err != nil {
		return nil
	}

	str := StringInstance()
	//array := ArrayInterface()
	DefaultBind(fileObj.GetContent(), str)
	//bytes, err := json.Marshal(bindData)
	//fmt.Print(string(bytes))
	return conf
}

func NewConf() *DefaultConf {
	return new(DefaultConf)
}
