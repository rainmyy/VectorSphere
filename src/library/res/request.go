package res

type Request struct {
	Name        string
	RequestType int //0 短连接 1:长链接
	Func        func() *Result
}
