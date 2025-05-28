package bplus

// Key 类型，需要实现比较接口
type Key interface {
	Less(other Key) bool
	Equal(other Key) bool
}

// Value 类型，可以根据需要修改
type Value interface{}
