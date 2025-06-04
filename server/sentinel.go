package server

import (
	"context"
	"encoding/json"
	"errors"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"net/http"
	"seetaSearch/library/log"
	"seetaSearch/messages"
	"sync"
	"sync/atomic"
	"time"
)

// SentinelRole 表示节点角色
type SentinelRole int

const (
	Master SentinelRole = iota
	Slave
)

type Sentinel struct {
	Hub        ServiceHub
	connPool   sync.Map
	leaseId    int64
	ServiceKey string
	Role       SentinelRole
	Balancer   Balancer
}

var _ ServerInterface = (*Sentinel)(nil)

func NewSentinel(endPoints []EndPoint, heartBeat int64, qps int, serviceName string, role SentinelRole) *Sentinel {
	sentinel := &Sentinel{
		Hub:        GetHubProxy(endPoints, heartBeat, qps, serviceName),
		connPool:   sync.Map{},
		ServiceKey: serviceName,
		Role:       role,
	}
	go sentinel.WatchServiceChanges()
	return sentinel
}

func (s *Sentinel) RegisterSentinel(ttl int64) error {
	for {
		resp, err := s.Hub.GetClient().Grant(context.Background(), ttl)
		if err != nil {
			time.Sleep(time.Second)
			continue
		}
		s.leaseId = int64(resp.ID)
		_, err = s.Hub.GetClient().Put(context.Background(), s.ServiceKey, "alive", clientv3.WithLease(resp.ID))
		if err != nil {
			time.Sleep(time.Second)
			continue
		}
		ch, err := s.Hub.GetClient().KeepAlive(context.Background(), resp.ID)
		if err != nil {
			time.Sleep(time.Second)
			continue
		}
		go func() {
			for range ch {
				log.Info("sentinel heartbeat success")
			}
			log.Warning("Etcd 连接中断，Sentinel 租约 ID: %d\n", s.leaseId)
		}()
		break
	}
	return nil
}

func (s *Sentinel) WatchServiceChanges() {
	for {
		watcher := clientv3.NewWatcher(s.Hub.GetClient())
		ctx, cancel := context.WithCancel(context.Background())
		watchChan := watcher.Watch(ctx, s.ServiceKey, clientv3.WithPrefix())
		for resp := range watchChan {
			for _, ev := range resp.Events {
				switch ev.Type {
				case clientv3.EventTypePut:
					// 有新的服务节点注册
					log.Info("新的服务节点注册: %s\n", string(ev.Kv.Key))
				case clientv3.EventTypeDelete:
					// 有服务节点注销
					log.Info("服务节点注销: %s\n", string(ev.Kv.Key))
				}
			}
		}
		cancel()
		time.Sleep(time.Second) // 断线重连
	}
}

func (s *Sentinel) GetGrpcConn(point EndPoint) *grpc.ClientConn {
	v, ok := s.connPool.Load(point.Ip)
	if ok {
		conn := v.(*grpc.ClientConn)
		if conn.GetState() == connectivity.Ready {
			return conn
		}
		conn.Close()
		s.connPool.Delete(point.Ip)
	}
	// 新建连接
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	grpcConn, err := grpc.DialContext(ctx, point.Ip, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
	if err != nil {
		return nil
	}
	s.connPool.Store(point.Ip, grpcConn)
	return grpcConn
}

func (s *Sentinel) AddDoc(doc *messages.Document) (int, error) {
	endPoint := s.Hub.GetServiceEndpoint(s.ServiceKey)
	if len(endPoint.Ip) == 0 {
		return -1, errors.New("服务节点不存在")
	}
	conn := s.GetGrpcConn(endPoint)
	if conn == nil {
		return -1, errors.New("无法连接到" + endPoint.Ip)
	}
	client := NewIndexServiceClient(conn)
	affected, err := client.AddDoc(context.Background(), doc)
	if err != nil {
		return -1, err
	}
	return int(affected.Count), nil
}

func (s *Sentinel) DelDoc(docId *DocId) int {
	endpoints := s.Hub.GetServiceEndpoints(s.ServiceKey)
	if len(endpoints) == 0 {
		return 0
	}
	var n int32
	wg := sync.WaitGroup{}
	wg.Add(len(endpoints))
	for _, endpoint := range endpoints {
		go func(endpoint EndPoint) {
			defer wg.Done()
			conn := s.GetGrpcConn(endpoint)
			if conn == nil {
				return
			}
			client := NewIndexServiceClient(conn)
			affected, err := client.DelDoc(context.Background(), &DocId{Id: docId.Id})
			if err != nil {
				return
			}
			if affected.Count > 0 {
				atomic.AddInt32(&n, affected.Count)
			}
		}(endpoint)
	}
	wg.Wait()
	return int(atomic.LoadInt32(&n))
}

func (s *Sentinel) Search(query *messages.TermQuery, onFlag, offFlag uint64, orFlags []uint64) (error, []*messages.Document) {
	endpoints := s.Hub.GetServiceEndpoints(s.ServiceKey)
	if len(endpoints) == 0 {
		return errors.New(""), nil
	}

	docs := make([]*messages.Document, 0, 1000)
	resultChan := make(chan *messages.Document, 1000)

	var wg sync.WaitGroup
	wg.Add(len(endpoints))

	for _, endpoint := range endpoints {
		go func(endpoint EndPoint) {
			defer wg.Done()
			conn := s.GetGrpcConn(endpoint)
			if conn == nil {
				return
			}
			client := NewIndexServiceClient(conn)
			searchRequest := Request{Query: query, OnFlag: onFlag, OffFlag: offFlag, OrFlags: orFlags}
			searchResult, err := client.Search(context.Background(), &searchRequest)
			if err != nil {
				return
			}
			if len(searchResult.Results) > 0 {
				for _, doc := range searchResult.Results {
					resultChan <- doc
				}
			}
		}(endpoint)
	}

	signalChan := make(chan bool)
	go func() {
		for doc := range resultChan {
			docs = append(docs, doc)
		}

		signalChan <- true
	}()

	wg.Wait()         // 等待所有goroutine完成
	close(resultChan) // 关闭结果通道 当resultChan关闭时，上面协程range循环结束
	<-signalChan      // 等待结果处理完成
	return nil, docs
}

func (s *Sentinel) Count() int {
	endpoints := s.Hub.GetServiceEndpoints(s.ServiceKey)
	if len(endpoints) == 0 {
		return 0
	}

	var count int32
	var wg sync.WaitGroup
	wg.Add(len(endpoints))

	for _, endpoint := range endpoints {
		go func(endpoint EndPoint) {
			defer wg.Done()
			conn := s.GetGrpcConn(endpoint)
			if conn == nil {

				return
			}
			client := NewIndexServiceClient(conn)
			countResult, err := client.Count(context.Background(), &CountRequest{})
			if err != nil {
				return
			}
			if countResult.Count > 0 {
				atomic.AddInt32(&count, countResult.Count)
			}
		}(endpoint)
	}
	wg.Wait()
	return int(atomic.LoadInt32(&count))
}

func (s *Sentinel) Close() (err error) {
	s.connPool.Range(func(key, value any) bool {
		conn := value.(*grpc.ClientConn)
		err = conn.Close()
		return true
	})
	s.Hub.Close()
	return
}

func (s *Sentinel) StartMasterServer() {
	if s.Role != Master {
		return
	}
	http.HandleFunc("/search", s.handleSearch)
	http.HandleFunc("/add", s.handleAddDoc)
	go s.HealthCheckLoop()
	log.Info("Master HTTP server started on :8080")
	http.ListenAndServe(":8080", nil)
}

func (s *Sentinel) handleSearch(w http.ResponseWriter, r *http.Request) {
	var req Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", 400)
		return
	}
	resp, err := s.ForwardToSlaves("Search", &req)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	json.NewEncoder(w).Encode(resp)
}

func (s *Sentinel) handleAddDoc(w http.ResponseWriter, r *http.Request) {
	var doc messages.Document
	if err := json.NewDecoder(r.Body).Decode(&doc); err != nil {
		http.Error(w, "bad request", 400)
		return
	}
	resp, err := s.ForwardToSlaves("AddDoc", &doc)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	json.NewEncoder(w).Encode(resp)
}

func (s *Sentinel) ForwardToSlaves(method string, req interface{}) (interface{}, error) {
	endpoints := s.Hub.GetServiceEndpoints(s.ServiceKey)
	if len(endpoints) == 0 {
		return nil, errors.New("无可用从节点")
	}
	s.Balancer.Set(endpoints...)
	ep := s.Balancer.Take()
	conn := s.GetGrpcConn(ep)
	if conn == nil {
		return nil, errors.New("无法连接到从节点")
	}
	client := NewIndexServiceClient(conn)
	switch method {
	case "AddDoc":
		doc := req.(*messages.Document)
		return client.AddDoc(context.Background(), doc)
	case "Search":
		searchReq := req.(*Request)
		return client.Search(context.Background(), searchReq)
	default:
		return nil, errors.New("未知方法")
	}
}

func (s *Sentinel) HealthCheckLoop() {
	for {
		s.Hub.RefreshEndpoints(s.ServiceKey)
		s.connPool.Range(func(key, value any) bool {
			conn := value.(*grpc.ClientConn)
			if conn.GetState() == connectivity.TransientFailure || conn.GetState() == connectivity.Shutdown {
				conn.Close()
				s.connPool.Delete(key)
			}
			return true
		})
		time.Sleep(5 * time.Second)
	}
}

func (s *Sentinel) ElectMaster() {
	session, _ := concurrency.NewSession(s.Hub.GetClient())
	election := concurrency.NewElection(session, s.ServiceKey+"-leader")
	go func() {
		for {
			if err := election.Campaign(context.Background(), s.ServiceKey); err == nil {
				s.Role = Master
				s.StartMasterServer()
			}
			<-election.Observe(context.Background())
		}
	}()
}

// --- CLI入口伪代码 ---
//func main() {
//	role := os.Getenv("ROLE") // "master" or "slave"
//	if role == "master" {
//		sentinel := NewSentinel(nil, 0, 0, "", Master)
//		sentinel.ElectMaster()
//		signalChan := make(chan os.Signal, 1)
//		signal.Notify(signalChan, os.Interrupt)
//		<-signalChan
//		log.Info("Master shutdown")
//	} else {
//		sentinel := NewSentinel(nil, 0, 0, "", Slave)
//		// 启动gRPC服务
//		select {}
//	}
//}
