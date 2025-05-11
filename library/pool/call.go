package pool

import (
	"fmt"
	"reflect"

	. "github.com/rainmyy/seetaSearch/library/res"
)

func FuncCall(function interface{}, args ...interface{}) []interface{} {
	var resultList = make([]interface{}, 0)
	result := func(errMsg string) []interface{} {
		result := ResultInstance()
		err := fmt.Errorf(errMsg)
		result.SetResult(-1, err, "")
		resultList = append(resultList, result)
		return resultList
	}
	value := reflect.ValueOf(function)
	if value.Kind() != reflect.Func {
		return result("params 1 is not a function")
	}
	parameters := make([]reflect.Type, 0, value.Type().NumIn())
	for i := 0; i < value.Type().NumIn(); i++ {
		arg := value.Type().In(i)
		parameters = append(parameters, arg)
	}
	if value.Type().NumIn() != len(args) {
		errorMsg := fmt.Sprintf("argument %d length doesn't equal to provide length %d \n", value.Type().NumIn(), len(args))
		return result(errorMsg)
	}
	outs := make([]reflect.Type, 0, value.Type().NumOut())
	for i := 0; i < value.Type().NumOut(); i++ {
		arg := value.Type().Out(i)
		outs = append(outs, arg)
	}
	var argValues []reflect.Value
	for i := 0; i < len(args); i++ {
		switch parameters[i] {
		case reflect.TypeOf(0):
			argValues = append(argValues, reflect.ValueOf(args[i].(int)))
		case reflect.TypeOf(false):
			argValues = append(argValues, reflect.ValueOf(args[i].(bool)))
		case reflect.TypeOf(int16(0)):
			argValues = append(argValues, reflect.ValueOf(args[i].(int16)))
		case reflect.TypeOf(""):
			argValues = append(argValues, reflect.ValueOf(args[i].(string)))
		default:
			errMsg := fmt.Sprintf("unsupport type %s[%s] \n", parameters[i].Kind(), parameters[i].Name())
			return result(errMsg)
		}
	}
	resultValue := value.Call(argValues)
	for i := 0; i < len(resultValue); i++ {
		switch resultValue[i].Type() {
		case reflect.TypeOf(0):
			resultList = append(resultList, resultValue[i].Interface().(int))
		case reflect.TypeOf(""):
			resultList = append(resultList, resultValue[i].Interface().(string))
		default:
			resultList = append(resultList, resultValue[i].Interface().(*Result))
		}
	}
	if len(outs) > 1 && len(resultList) == 0 {
		return result("result is empty")
	}
	return resultList
}

func FormatResult(resultList []interface{}) *Response {
	var response = new(Response)
	for _, result := range resultList {
		if result == nil {
			continue
		}
		switch result.(type) {
		case *Result:
			response.Result = result.(*Result)
		case int:
			//退出标志
			if result.(int) == -1 {
				return nil
			}
		}
	}
	return response
}
