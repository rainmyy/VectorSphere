package bind

import (
	. "seetaSearch/library/strategy"
)

type Binder interface {
	// Bind /**
	Bind(treeList []*TreeStruct)
	UnBind() []*TreeStruct
	GetValue() interface{}
}

// DefaultBind /**
func DefaultBind(tree []*TreeStruct, obj Binder) interface{} {
	obj.Bind(tree)
	bindData := obj.GetValue()
	return bindData
}

func DefaultUnBind(obj Binder) []*TreeStruct {
	return obj.UnBind()
}
