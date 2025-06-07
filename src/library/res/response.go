package res

type Response struct {
	Name     string
	Result   *Result
	Callback *Result
	//res      Result
}

func ReposeInstance() *Response {
	return new(Response)
}

func (r *Response) ErrorMsg(tag, errcode int, errmsg error) {
	result := ResultInstance().SetResult(errcode, errmsg, "")
	if tag == 1 {
		r.Result = result
	} else if tag == 2 {
		r.Callback = result
	}
}
