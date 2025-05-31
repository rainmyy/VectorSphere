package bootstrap

import (
	"context"
	"encoding/json"
	"fmt"
	clientv3 "go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"net/http"
	"seetaSearch/library/log"
	"seetaSearch/library/res"
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
	server       *server.IndexServer
	sentinel     *server.Sentinel
	etcdCli      *clientv3.Client
	grpcConn     *grpc.ClientConn
	masterAddr   string
}

const (
	RPCSERVICE = iota
	TCPSERVICE
	READSERVICE
	WRITESERVICE
)

var pool = PoolLib.GetInstance()

// ServiceConfig 结构体映射整个 YAML 文件
type ServiceConfig struct {
	ServiceName string                     `yaml:"serviceName"`
	TimeOut     int                        `yaml:"timeOut"`
	DefaultPort int                        `yaml:"defaultPort"`
	Heartbeat   int                        `yaml:"heartbeat"`
	Endpoints   map[string]server.EndPoint `yaml:"endpoints"`
}

func (app *AppServer) ReadServiceConf() (error, *ServiceConfig) {
	var cfg ServiceConfig
	//rootPath, err := util.GetProjectRoot()
	//if err != nil {
	//	return err, nil
	//}

	//err = conf.ReadYAML(path.Join(rootPath, "conf", "idc", "simple", "service.yaml"), &cfg)
	err := conf.ReadYAML("D:\\code\\seetaSearch\\conf\\idc\\simple\\service.yaml", &cfg)

	if err != nil {
		return err, nil
	}

	return nil, &cfg
}
func (app *AppServer) RegisterToEtcd(serviceName, addr string) error {
	key := fmt.Sprintf("/seetasearch/services/%s/%s", serviceName, addr)
	_, err := app.etcdCli.Put(context.Background(), key, addr)
	return err
}
func (app *AppServer) DiscoverMaster(serviceName string) (string, error) {
	resp, err := app.etcdCli.Get(context.Background(), fmt.Sprintf("/seetasearch/services/%s/", serviceName), clientv3.WithPrefix())
	if err != nil {
		return "", err
	}
	for _, kv := range resp.Kvs {
		return string(kv.Value), nil // 取第一个主服务地址
	}
	return "", fmt.Errorf("no master found")
}
func (app *AppServer) ConnectToMaster(masterAddr string) error {
	conn, err := grpc.Dial(masterAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}
	app.grpcConn = conn
	return nil
}
func (app *AppServer) RegisterService() {
	err, config := app.ReadServiceConf()
	if err != nil {
		log.Error("read service conf failed, err:%v", err)
		return
	}
	var masterEndpoint []server.EndPoint
	var sentinelEndpoint []server.EndPoint

	for name, endpoint := range config.Endpoints {
		port := endpoint.Port
		endpoint.Name = name
		if port == 0 {
			endpoint.Port = config.DefaultPort
		}
		if endpoint.IsMaster {
			masterEndpoint = append(masterEndpoint, endpoint)
		} else {
			sentinelEndpoint = append(sentinelEndpoint, endpoint)
		}

	}
	if len(masterEndpoint) > 0 {
		// master节点注册
		s := new(server.IndexServer)
		masterServiceName := config.ServiceName
		err := s.RegisterService(masterEndpoint, config.DefaultPort, masterServiceName)
		if err != nil {
			log.Error("Master注册失败:", err)
			return
		}
		log.Info("Master节点 %s 注册成功\n", masterServiceName)
		app.server = s
	}

	if len(sentinelEndpoint) > 0 {
		sentinel := server.NewSentinel(sentinelEndpoint, int64(config.Heartbeat), 100, config.ServiceName)
		err := sentinel.RegisterSentinel(int64(config.Heartbeat))
		if err != nil {
			log.Error("Sentinel注册失败:", err)
			return
		}

		app.sentinel = sentinel
	}

	//for name, ep := range config.Endpoints {
	//	port := ep.Port
	//	if port == 0 {
	//		port = config.DefaultPort
	//	}
	//	ep.Ip = ep.Ip + ":" + strconv.Itoa(port)
	//	println(ep.Ip)
	//	endpoints := []string{ep.Ip}
	//	if ep.IsMaster {
	//		// master节点注册
	//		s := new(server.IndexServer)
	//		masterServiceName := config.ServiceName
	//		err := s.RegisterService(endpoints, port, masterServiceName)
	//		if err != nil {
	//			log.Error("Master注册失败:", err)
	//			continue
	//		}
	//		log.Info("Master节点 %s 注册成功: %s:%d\n", name, ep.Ip, port)
	//	} else {
	//		// sentinel节点注册
	//		sentinel := server.NewSentinel(endpoints, int64(config.Heartbeat), 100, config.ServiceName)
	//		err := sentinel.RegisterSentinel(int64(config.Heartbeat))
	//		if err != nil {
	//			log.Error("Sentinel注册失败:", err)
	//			continue
	//		}
	//		log.Info("Sentinel节点 %s 注册成功: %s:%d\n", name, ep.Ip, port)
	//	}
	//}
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
	var sentinelEndpoint []server.EndPoint

	for name, endpoint := range config.Endpoints {
		port := endpoint.Port
		endpoint.Name = name
		if port == 0 {
			port = config.DefaultPort
		}
		if !endpoint.IsMaster {
			sentinelEndpoint = append(sentinelEndpoint, endpoint)
		}

	}

	if len(sentinelEndpoint) > 0 {
		if app.sentinel == nil {
			sentinel := server.NewSentinel(sentinelEndpoint, int64(config.Heartbeat), 100, config.ServiceName)
			app.sentinel = sentinel
		}

		endpoints := app.sentinel.Hub.GetServiceEndpoints(config.ServiceName)
		log.Info("Sentinel节点 %s 发现的master节点: %+v\n", config.ServiceName, endpoints)
	}
	//for name, ep := range config.Endpoints {
	//	if !ep.IsMaster {
	//		sentinel := server.NewSentinel([]string{ep.Ip}, int64(config.Heartbeat), 100, config.ServiceName)
	//		endpoints := sentinel.Hub.GetServiceEndpoints(config.ServiceName)
	//		log.Info("Sentinel节点 %s 发现的master节点: %+v\n", name, endpoints)
	//	}
	//}
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
	//注册注册方法
	app.funcRegister["register_etcd"] = app.RegisterService
	//注册发现方法
	app.funcRegister["discover_etcd"] = app.DiscoverService
	app.funcRegister["listen_api"] = app.ListenAPI
}

func (app *AppServer) ListenAPI() {
	err, config := app.ReadServiceConf()
	if err != nil {
		log.Error("read service conf failed, err:%v", err)
		return
	}

	var serviceName string
	var serviceAddr string
	for name, ep := range config.Endpoints {
		if ep.IsMaster {
			port := ep.Port
			if port == 0 {
				port = config.DefaultPort
			}
			address := fmt.Sprintf(":%d", 8080)
			serviceName = name
			serviceAddr = address
		}
	}

	log.Info("Master节点 %s 开始监听 API 请求: %s\n", serviceName, serviceAddr)
	mux := http.NewServeMux()

	mux.HandleFunc("/", func(rw http.ResponseWriter, req *http.Request) {
		rw.Write([]byte("hello world"))
	})

	mux.HandleFunc("/addDoc", func(w http.ResponseWriter, r *http.Request) {
		doc := &messages.Document{}
		if err := json.NewDecoder(r.Body).Decode(doc); err != nil {
			http.Error(w, "Invalid request payload", http.StatusBadRequest)
			return
		}
		s := app.sentinel
		affected, err := s.AddDoc(doc)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		log.Info("添加文档成功，影响文档数量: %d", affected)
		w.WriteHeader(http.StatusOK)
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte("ok"))
	})

	mux.HandleFunc("/count", func(w http.ResponseWriter, r *http.Request) {
		s := app.sentinel
		total := s.Count()
		log.Info("更新文档成功,total:%d", total)
		w.WriteHeader(http.StatusOK)
	})

	mux.HandleFunc("/deleteDoc", func(w http.ResponseWriter, r *http.Request) {
		docID := r.URL.Query().Get("id")
		if docID == "" {
			http.Error(w, "Missing document ID", http.StatusBadRequest)
			return
		}
		s := app.sentinel
		doc := &server.DocId{Id: docID}
		count := s.DelDoc(doc)
		if count == 0 {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		log.Info("删除文档成功，文档ID: %s", docID)
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) {
		query := &messages.TermQuery{}
		if err := json.NewDecoder(r.Body).Decode(query); err != nil {
			http.Error(w, "Invalid request payload", http.StatusBadRequest)
			return
		}
		onFlag := uint64(0)
		offFlag := uint64(0)
		var orFlags []uint64
		s := app.sentinel
		err, result := s.Search(query, onFlag, offFlag, orFlags)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if err := json.NewEncoder(w).Encode(result); err != nil {
			log.Error("encode json failed, err:%v\n", err)
		}
	})

	if err := http.ListenAndServe(serviceAddr, mux); err != nil {
		log.Info("Master节点 %s 监听 API 请求失败: %v\n", serviceName, err)
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
