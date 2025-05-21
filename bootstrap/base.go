package bootstrap

import (
	"context"
	"fmt"
	"path"
	"seetaSearch/library/res"
	"seetaSearch/library/util"
	"seetaSearch/server"
	"sync"

	"seetaSearch/library/conf"
	PoolLib "seetaSearch/library/pool"
)

/**
*app执行入口
 */
type AppServer struct {
	mutex        sync.WaitGroup
	Ctx          context.Context
	Cancel       context.CancelFunc
	funcRegister map[string]func()
}

const (
	RPCSERVICE = iota
	TCPSERVICE
	READSERVICE
	WRITESERVICE
)

var pool = PoolLib.GetInstance()

type Endpoint struct {
	Ip        string `yaml:"ip"`
	SetMaster bool   `yaml:"setMaster,omitempty"`
	Port      int    `yaml:"port,omitempty"`
}

// ServiceConfig 结构体映射整个 YAML 文件
type ServiceConfig struct {
	ServiceName string              `yaml:"serviceName"`
	TimeOut     int                 `yaml:"timeOut"`
	DefaultPort int                 `yaml:"defaultPort"`
	Heartbeat   int                 `yaml:"heartbeat"`
	Endpoints   map[string]Endpoint `yaml:"endpoints"`
}

func (app *AppServer) ReadServiceConf() (error, *ServiceConfig) {
	var cfg ServiceConfig
	rootPath, err := util.GetProjectRoot()
	if err != nil {
		return err, nil
	}
	err = conf.ReadYAML(path.Join(rootPath, "conf", "service.yaml"), &cfg)
	if err != nil {
		return err, nil
	}

	return nil, &cfg
}

func (app *AppServer) Register() {
	err, config := app.ReadServiceConf()
	if err != nil {
		panic(err)
	}
	for name, ep := range config.Endpoints {
		endpoints := []string{ep.Ip}
		port := ep.Port
		if port == 0 {
			port = config.DefaultPort
		}
		if ep.SetMaster {
			// master节点注册
			s := new(server.IndexServer)
			masterServiceName := config.ServiceName
			err := s.RegisterService(endpoints, port, masterServiceName)
			if err != nil {
				fmt.Println("Master注册失败:", err)
				continue
			}
			fmt.Printf("Master节点 %s 注册成功: %s:%d\n", name, ep.Ip, port)
			defer s.Close()
		} else {
			// sentinel节点注册
			sentinel := server.NewSentinel(endpoints, int64(config.Heartbeat), 100, config.ServiceName)
			err := sentinel.RegisterSentinel(int64(config.Heartbeat))
			if err != nil {
				fmt.Println("Sentinel注册失败:", err)
				continue
			}
			fmt.Printf("Sentinel节点 %s 注册成功: %s:%d\n", name, ep.Ip, port)
		}
	}
	//注册注册方法
	app.funcRegister["register_etcd"] = app.Register
}

func (app *AppServer) Discover() {
	err, config := app.ReadServiceConf()
	if err != nil {
		panic(err)
	}
	for name, ep := range config.Endpoints {
		if !ep.SetMaster {
			sentinel := server.NewSentinel([]string{ep.Ip}, int64(config.Heartbeat), 100, config.ServiceName)
			endpoints := sentinel.Hub.GetServiceEndpoints(config.ServiceName)
			fmt.Printf("Sentinel节点 %s 发现的master节点: %+v\n", name, endpoints)
		}
	}

	//注册发现方法
	app.funcRegister["discover_etcd"] = app.Discover
}

// Setup /
func (app *AppServer) Setup() {
	_ = conf.NewConf().Init()

	//注册执行函数
	serviceLen := len(app.funcRegister)
	pool := pool.Init(serviceLen, serviceLen)
	for k, v := range app.funcRegister {
		app.mutex.Add(1)
		go func(key string, value interface{}) {
			defer app.mutex.Done()
			query := PoolLib.QueryInit(key, v)
			pool.AddTask(query)
		}(k, v)
	}

	app.mutex.Wait()
}

func (app *AppServer) Start() map[string]*res.Response {
	pool.Start()
	return pool.TaskResult()
}

func GenInstance() *AppServer {
	return new(AppServer)
}
