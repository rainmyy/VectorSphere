package res

import "fmt"

type Result struct {
	Code  int
	Error error
	Data  string
}

func (r *Result) SystemError() *Result {
	return r.SetResult(-1, fmt.Errorf("system error"), "")
}

func (r *Result) ErrorParamsResult() *Result {
	return r.SetResult(-2, fmt.Errorf("error params data"), "")
}

func (r *Result) EmptyResult() *Result {
	return r.SetResult(-1, fmt.Errorf("empty data"), "")
}
func (r *Result) SetResult(code int, msg error, data string) *Result {
	r.Code = code
	r.Error = msg
	if data != "" {
		r.Data = data
	}
	return r
}

func (r *Result) GetResult() *Result {
	return r
}

func ResultInstance() *Result {
	return new(Result)
}
