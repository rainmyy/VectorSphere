package pool

import (
	"seetaSearch/library/res"
	"sync"
)

const (
	defaultRuntineNumber = 10
	defailtTotal         = 10
)

type Pool struct {
	//mutex         sync.WaitGroup
	RuntineNumber int
	Total         int
	taskQuery     chan *Queue
	taskResult    chan map[string]*res.Response
	taskResponse  map[string]*res.Response
}

/*
*
执行队列
*/
type Queue struct {
	Name     string
	result   chan *res.Response
	Excel    *ExcelFunc
	CallBack *CallBackFunc
}
type ExcelFunc struct {
	Name     string
	Function interface{}
	Params   []interface{}
}
type CallBackFunc struct {
	name     string
	Function interface{}
	Params   []interface{}
}

func GetInstance() *Pool {
	return new(Pool)
}
func QueryInit(name string, function interface{}, params ...interface{}) *Queue {
	excelFunc := &ExcelFunc{Function: function, Params: params}
	query := &Queue{
		Name:   name,
		Excel:  excelFunc,
		result: make(chan *res.Response, 1),
	}
	return query
}

func (q *Queue) CallBackInit(name string, function interface{}, params ...interface{}) *Queue {
	callBackFunc := &CallBackFunc{name: name, Function: function, Params: params}
	q.CallBack = callBackFunc
	return q
}
func (p *Pool) Init(runtimeNumber, total int) *Pool {
	p.RuntineNumber = runtimeNumber
	p.Total = total
	p.taskQuery = make(chan *Queue, runtimeNumber)
	p.taskResult = make(chan map[string]*res.Response, runtimeNumber)
	p.taskResponse = make(map[string]*res.Response)
	return p
}
func (p *Pool) Start() {
	runtimeNumber := p.RuntineNumber
	if len(p.taskQuery) != runtimeNumber {
		runtimeNumber = len(p.taskQuery)
	}
	var mutex sync.WaitGroup
	for i := 0; i < runtimeNumber; i++ {
		mutex.Add(1)
		go func(num int) {
			defer mutex.Done()
			task, ok := <-p.taskQuery
			taskName := task.Name
			result := map[string]*res.Response{
				taskName: nil,
			}
			response := res.ReposeInstance()
			if !ok {
				res := res.ResultInstance().ErrorParamsResult()
				response.Result = res
				result[taskName] = response
				p.taskResult <- result
				return
			}
			task.excelQuery()
			taskResult, ok := <-task.result
			if !ok {
				res := res.ResultInstance().EmptyResult()
				response.Result = res
				result[taskName] = response
				p.taskResult <- result
				return
			}
			result = map[string]*res.Response{
				taskName: taskResult,
			}
			p.taskResult <- result
		}(i)
	}
	mutex.Wait()
	for i := 0; i < runtimeNumber; i++ {
		if result, ok := <-p.taskResult; ok {
			for name, value := range result {
				p.taskResponse[name] = value
			}
		}
	}
}

func (p *Pool) TaskResult() map[string]*res.Response {
	return p.taskResponse
}

func (p *Pool) Stop() {
	close(p.taskResult)
}

func (p *Pool) AddTask(task *Queue) {
	p.taskQuery <- task
}

/**
* 执行队列
 */
func (qeury *Queue) excelQuery() {
	defer close(qeury.result)
	excelFunc := qeury.Excel.Function
	if excelFunc == nil {
		return
	}
	var requestChannel = make(chan []interface{})
	go func() {
		defer close(requestChannel)
		params := qeury.Excel.Params
		result := FuncCall(excelFunc, params...)
		if result == nil {
			return
		}
		requestChannel <- result
	}()
	result, ok := <-requestChannel
	if !ok {
		return
	}
	response := FormatResult(result)
	if response == nil {
		return
	}
	var callBackChannel = make(chan []interface{})
	go func() {
		defer close(callBackChannel)
		if qeury.CallBack == nil {
			return
		}
		result := FuncCall(qeury.CallBack.Function, qeury.CallBack.Params...)
		if result == nil {
			return
		}
		callBackChannel <- result
	}()
	resultList, ok := <-callBackChannel
	if !ok {
		qeury.result <- response
		return
	}
	callBackResponse := FormatResult(resultList).Result
	if callBackResponse != nil {
		response.Callback = callBackResponse
	}
	qeury.result <- response
}
