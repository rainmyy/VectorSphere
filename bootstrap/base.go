package bootstrap

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"path"
	"seetaSearch/library/log"
	"seetaSearch/library/res"
	"seetaSearch/library/util"
	"seetaSearch/messages"
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

	err = conf.ReadYAML(path.Join(rootPath, "conf\\simple\\service.yaml"), &cfg)
	//err = conf.ReadYAML("D:\\code\\seetaSearch\\conf\\idc\\simple\\service.yaml", &cfg)

	if err != nil {
		return err, nil
	}

	return nil, &cfg
}

func (app *AppServer) RegisterService() {
	err, config := app.ReadServiceConf()
	if err != nil {
		log.Error("read service conf failed, err:%v", err)
		return
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
				log.Error("Master注册失败:", err)
				continue
			}
			log.Info("Master节点 %s 注册成功: %s:%d\n", name, ep.Ip, port)
		} else {
			// sentinel节点注册
			sentinel := server.NewSentinel(endpoints, int64(config.Heartbeat), 100, config.ServiceName)
			err := sentinel.RegisterSentinel(int64(config.Heartbeat))
			if err != nil {
				log.Error("Sentinel注册失败:", err)
				continue
			}
			log.Info("Sentinel节点 %s 注册成功: %s:%d\n", name, ep.Ip, port)
		}
	}
}

func (app *AppServer) DiscoverService() {
	err, config := app.ReadServiceConf()
	if err != nil {
		log.Error("read service conf failed, err:%v", err)
		return
	}

	if config == nil || config.Endpoints == nil {
		log.Error("endpoints is nil")
		return
	}

	for name, ep := range config.Endpoints {
		if !ep.SetMaster {
			sentinel := server.NewSentinel([]string{ep.Ip}, int64(config.Heartbeat), 100, config.ServiceName)
			endpoints := sentinel.Hub.GetServiceEndpoints(config.ServiceName)
			log.Info("Sentinel节点 %s 发现的master节点: %+v\n", name, endpoints)
		}
	}
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

func (app *AppServer) Register() {
	//注册发现方法
	app.funcRegister["discover_etcd"] = app.DiscoverService
	//注册注册方法
	app.funcRegister["register_etcd"] = app.RegisterService
	app.funcRegister["listen_api"] = app.ListenAPI
}

func (app *AppServer) ListenAPI() {
	err, config := app.ReadServiceConf()
	if err != nil {
		log.Error("read service conf failed, err:%v", err)
		return
	}

	for name, ep := range config.Endpoints {
		if ep.SetMaster {
			// 主节点监听 API 请求
			port := ep.Port
			if port == 0 {
				port = config.DefaultPort
			}
			address := fmt.Sprintf("%s:%d", ep.Ip, port)
			log.Info("Master节点 %s 开始监听 API 请求: %s\n", name, address)

			http.HandleFunc("/addDoc", func(w http.ResponseWriter, r *http.Request) {
				// 处理添加文档请求
				// 这里需要根据实际情况解析请求参数并调用相应的服务方法
				// 示例代码
				doc := &messages.Document{}
				s := new(server.IndexServer)
				affected, err := s.AddDoc(context.Background(), doc)
				if err != nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
					return
				}
				log.Info("添加文档成功，影响文档数量: %d", affected.Count)
			})

			http.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) {
				query := &messages.TermQuery{}
				onFlag := uint64(0)
				offFlag := uint64(0)
				orFlags := []uint64{}
				s := new(server.IndexServer)
				result, err := s.Search(context.Background(), &server.Request{Query: query, OnFlag: onFlag, OffFlag: offFlag, OrFlags: orFlags})
				if err != nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
					return
				}
				// 这里需要将结果序列化为 JSON 并返回给客户端
				err = json.NewEncoder(w).Encode(result)
				if err != nil {
					log.Error("encode json failed, err:%v\n", err)
				}
			})

			if err := http.ListenAndServe(address, nil); err != nil {
				log.Info("Master节点 %s 监听 API 请求失败: %v\n", name, err)
			}
		}
	}
}

func (app *AppServer) Start() map[string]*res.Response {
	pool.Start()
	return pool.TaskResult()
}

func GenInstance() *AppServer {
	app := &AppServer{funcRegister: make(map[string]func())}
	app.Register()
	return app
}
