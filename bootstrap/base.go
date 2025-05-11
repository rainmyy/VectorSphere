package bootstrap

import (
	"context"
	"fmt"
	"strconv"
	"sync"

	"github.com/rainmyy/seetaSearch/library/conf"
	PoolLib "github.com/rainmyy/seetaSearch/library/pool"
)

/**
*app执行入口
 */
type AppServer struct {
	mutex  sync.WaitGroup
	Ctx    context.Context
	Cancel context.CancelFunc
}

const (
	RPCSERVICE = iota
	TCPSERVICE
	READSERVICE
	WRITESERVICE
)

var ServiceLen = 4
var pool = PoolLib.GetInstance()

// Setup /**
func (app *AppServer) Setup() {
	_ = conf.Intance().Init()
	//注册执行函数
	pool := pool.Init(ServiceLen, ServiceLen)
	for i := 0; i < ServiceLen; i++ {
		app.mutex.Add(1)
		go func(num int) {
			defer app.mutex.Done()
			query := PoolLib.QueryInit(strconv.Itoa(num), download, 123, "wwww")
			pool.AddTask(query)
		}(i)
	}
	app.mutex.Wait()
}

func (app *AppServer) Start() {
	pool.Start()
	pool.TaskResult()
}

func download(url int, str string) {
	fmt.Print(str, "\n")
	//result := res.ResultInstance().SetResult(200, fmt.Errorf(""), "result")
	//return result
}

func GenInstance() *AppServer {
	return new(AppServer)
}
