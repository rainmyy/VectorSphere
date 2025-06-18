package server

import (
	"VectorSphere/src/library/entity"
	"VectorSphere/src/library/logger"
	"VectorSphere/src/proto/messages"
	serverProto "VectorSphere/src/proto/serverProto"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
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
	Token      string
}

func (s *Sentinel) DelDoc(ctx context.Context, docId *serverProto.DocId) (*serverProto.ResCount, error) {
	endpoints := s.Hub.GetServiceEndpoints(s.ServiceKey)
	if len(endpoints) == 0 {
		return nil, errors.New("no endpoints")
	}
	var n int32
	wg := sync.WaitGroup{}
	wg.Add(len(endpoints))
	for _, endpoint := range endpoints {
		go func(endpoint entity.EndPoint) {
			defer wg.Done()
			conn := s.GetGrpcConn(endpoint)
			if conn == nil {
				return
			}
			client := serverProto.NewIndexServiceClient(conn)
			affected, err := client.DelDoc(context.Background(), &serverProto.DocId{Id: docId.Id})
			if err != nil {
				return
			}
			if affected.Count > 0 {
				atomic.AddInt32(&n, affected.Count)
			}
		}(endpoint)
	}
	wg.Wait()

	return &serverProto.ResCount{Count: n}, nil
}

func (s *Sentinel) AddDoc(ctx context.Context, document *messages.Document) (*serverProto.ResCount, error) {
	endPoint := s.Hub.GetServiceEndpoint(s.ServiceKey)
	if len(endPoint.Ip) == 0 {
		return nil, errors.New("服务节点不存在")
	}
	conn := s.GetGrpcConn(endPoint)
	if conn == nil {
		return nil, errors.New("无法连接到" + endPoint.Ip)
	}
	client := serverProto.NewIndexServiceClient(conn)
	affected, err := client.AddDoc(context.Background(), document)
	if err != nil {
		return nil, err
	}

	return &serverProto.ResCount{Count: affected.Count}, nil
}

func (s *Sentinel) Search(ctx context.Context, request *serverProto.Request) (*serverProto.Result, error) {
	endpoints := s.Hub.GetServiceEndpoints(s.ServiceKey)
	if len(endpoints) == 0 {
		return nil, errors.New("no endpoints")
	}

	docs := make([]*messages.Document, 0, 1000)
	resultChan := make(chan *messages.Document, 1000)
	query := request.Query
	onFlag := request.OnFlag
	offFlag := request.OffFlag
	var orFlags = request.OrFlags
	var wg sync.WaitGroup
	wg.Add(len(endpoints))

	for _, endpoint := range endpoints {
		go func(endpoint entity.EndPoint) {
			defer wg.Done()
			conn := s.GetGrpcConn(endpoint)
			if conn == nil {
				return
			}
			client := serverProto.NewIndexServiceClient(conn)
			searchRequest := serverProto.Request{Query: query, OnFlag: onFlag, OffFlag: offFlag, OrFlags: orFlags}
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

	return &serverProto.Result{Results: docs}, nil
}

func (s *Sentinel) Count(ctx context.Context, request *serverProto.CountRequest) (*serverProto.ResCount, error) {
	endpoints := s.Hub.GetServiceEndpoints(s.ServiceKey)
	if len(endpoints) == 0 {
		return nil, errors.New("no endpoints")
	}

	var count int32
	var wg sync.WaitGroup
	wg.Add(len(endpoints))

	for _, endpoint := range endpoints {
		go func(endpoint entity.EndPoint) {
			defer wg.Done()
			conn := s.GetGrpcConn(endpoint)
			if conn == nil {

				return
			}
			client := serverProto.NewIndexServiceClient(conn)
			countResult, err := client.Count(context.Background(), &serverProto.CountRequest{})
			if err != nil {
				return
			}
			if countResult.Count > 0 {
				atomic.AddInt32(&count, countResult.Count)
			}
		}(endpoint)
	}
	wg.Wait()

	return &serverProto.ResCount{Count: count}, nil
}

func (s *Sentinel) ExecuteTask(ctx context.Context, request *serverProto.TaskRequest) (*serverProto.TaskResponse, error) {
	//TODO implement me
	panic("implement me")
}

func (s *Sentinel) ReportTaskResult(ctx context.Context, response *serverProto.TaskResponse) (*serverProto.ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (s *Sentinel) CreateTable(ctx context.Context, request *serverProto.CreateTableRequest) (*serverProto.ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (s *Sentinel) DeleteTable(ctx context.Context, request *serverProto.TableRequest) (*serverProto.ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (s *Sentinel) AddDocumentToTable(ctx context.Context, request *serverProto.AddDocumentRequest) (*serverProto.ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (s *Sentinel) DeleteDocumentFromTable(ctx context.Context, request *serverProto.DeleteDocumentRequest) (*serverProto.ResCount, error) {
	//TODO implement me
	panic("implement me")
}

func (s *Sentinel) SearchTable(ctx context.Context, request *serverProto.SearchRequest) (*serverProto.SearchResult, error) {
	//TODO implement me
	panic("implement me")
}

func (s *Sentinel) HealthCheck(ctx context.Context, request *serverProto.HealthCheckRequest) (*serverProto.HealthCheckResponse, error) {
	//TODO implement me
	panic("implement me")
}

//var _ IndexServiceServer = (*Sentinel)(nil)

func NewSentinel(endPoints []entity.EndPoint, heartBeat int64, qps int, serviceName string, role SentinelRole) *Sentinel {
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
				logger.Info("sentinel heartbeat success")
			}
			logger.Warning("Etcd 连接中断，Sentinel 租约 ID: %d\n", s.leaseId)
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
					logger.Info("新的服务节点注册: %s\n", string(ev.Kv.Key))
				case clientv3.EventTypeDelete:
					// 有服务节点注销
					logger.Info("服务节点注销: %s\n", string(ev.Kv.Key))
				}
			}
		}
		cancel()
		time.Sleep(time.Second) // 断线重连
	}
}

func (s *Sentinel) GetGrpcConn(point entity.EndPoint) *grpc.ClientConn {
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

func (s *Sentinel) Close() (err error) {
	s.connPool.Range(func(key, value any) bool {
		conn := value.(*grpc.ClientConn)
		err = conn.Close()
		return true
	})
	s.Hub.Close()
	return
}

func (s *Sentinel) StartMasterServer(port int) {
	if s.Role != Master {
		return
	}
	mux := http.NewServeMux()
	mux.Handle("/search", s.authMiddleware(s.rateLimitMiddleware(http.HandlerFunc(s.handleSearch))))
	mux.Handle("/add", s.authMiddleware(s.rateLimitMiddleware(http.HandlerFunc(s.handleAddDoc))))
	mux.Handle("/delete", s.authMiddleware(s.rateLimitMiddleware(http.HandlerFunc(s.handleDeleteDoc))))
	mux.Handle("/update", s.authMiddleware(s.rateLimitMiddleware(http.HandlerFunc(s.handleUpdateDoc))))
	mux.Handle("/list", s.authMiddleware(s.rateLimitMiddleware(http.HandlerFunc(s.handleListDocs))))
	mux.Handle("/health", http.HandlerFunc(s.handleHealth))
	mux.Handle("/lb_switch", s.authMiddleware(http.HandlerFunc(s.handleLBSwitch)))
	mux.Handle("/gray_route", s.authMiddleware(http.HandlerFunc(s.handleGrayRoute)))
	go s.HealthCheckLoop()
	logger.Info("Master HTTP server started on :%d", port)
	err := http.ListenAndServe(fmt.Sprintf(":%d", port), mux)
	if err != nil {
		return
	}
}

// Token认证中间件
func (s *Sentinel) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := r.Header.Get("Authorization")
		if token != "Bearer your_token" {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Sentinel) rateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case rateLimiter <- struct{}{}:
			defer func() { <-rateLimiter }()
			next.ServeHTTP(w, r)
		default:
			http.Error(w, "Too Many Requests", http.StatusTooManyRequests)
		}
	})
}
func (s *Sentinel) handleSearch(w http.ResponseWriter, r *http.Request) {
	var req serverProto.Request
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

func (s *Sentinel) handleDeleteDoc(w http.ResponseWriter, r *http.Request) {
	var req struct{ Id string }
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", 400)
		return
	}
	resp, err := s.ForwardToSlaves("DeleteDoc", req.Id)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	json.NewEncoder(w).Encode(resp)
}

func (s *Sentinel) handleUpdateDoc(w http.ResponseWriter, r *http.Request) {
	var doc messages.Document
	if err := json.NewDecoder(r.Body).Decode(&doc); err != nil {
		http.Error(w, "bad request", 400)
		return
	}
	resp, err := s.ForwardToSlaves("UpdateDoc", &doc)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	json.NewEncoder(w).Encode(resp)
}

func (s *Sentinel) handleListDocs(w http.ResponseWriter, r *http.Request) {
	page, _ := strconv.Atoi(r.URL.Query().Get("page"))
	size, _ := strconv.Atoi(r.URL.Query().Get("size"))
	// 组装分页参数，转发到从节点
	req := struct{ Page, Size int }{page, size}
	resp, err := s.ForwardToSlaves("ListDocs", req)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	json.NewEncoder(w).Encode(resp)
}

func (s *Sentinel) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Write([]byte("ok"))
}

func (s *Sentinel) handleLBSwitch(w http.ResponseWriter, r *http.Request) {
	type req struct{ Strategy string }
	var body req
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, "Unknown strategy", 400)
		return
	}
	switch body.Strategy {
	case "roundrobin":
		s.Balancer = &RoundRobinBalancer{}
	case "weighted":
		s.Balancer = &WeightedBalancer{}
	case "leastconn":
		s.Balancer = &LeastConnBalancer{}
	default:
		http.Error(w, "Unknown strategy", 400)
		return
	}
	w.Write([]byte("ok"))
}

func (s *Sentinel) handleGrayRoute(w http.ResponseWriter, r *http.Request) {
	tag := r.URL.Query().Get("tag")
	endpoints := s.Hub.GetServiceEndpoints(s.ServiceKey)
	var filtered []entity.EndPoint
	for _, ep := range endpoints {
		if ep.Tags["env"] == tag {
			filtered = append(filtered, ep)
		}
	}
	s.Balancer.Set(filtered...)
	w.Write([]byte(fmt.Sprintf("Switched to %d endpoints with tag=%s", len(filtered), tag)))
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
	client := serverProto.NewIndexServiceClient(conn)
	switch method {
	case "AddDoc":
		doc := req.(*messages.Document)
		return client.AddDoc(context.Background(), doc)
	case "Search":
		searchReq := req.(*serverProto.Request)
		return client.Search(context.Background(), searchReq)
	case "DeleteDoc":
		id := req.(string)
		return client.DelDoc(context.Background(), &serverProto.DocId{Id: id})
	//case "UpdateDoc":
	//	doc := req.(*messages.Document)
	//	return client.UpdateDoc(context.Background(), doc)
	//case "ListDocs":
	//	p := req.(struct{ Page, Size int })
	//	return client.ListDocs(context.Background(), &ListRequest{Page: int32(p.Page), Size: int32(p.Size)})
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
				s.StartMasterServer(8080)
				s.sendMasterNotify("I am the new master")
			}
			<-election.Observe(context.Background())
		}
	}()
}

func (s *Sentinel) sendMasterNotify(msg string) {
	logger.Info("Master notify: %s", msg)
	// 可扩展为Webhook/邮件/钉钉等
}

func RunCLI() {
	role := flag.String("role", "master", "Role: master or slave")
	lb := flag.String("lb", "roundrobin", "Load balancer: roundrobin/weighted/leastconn")
	token := flag.String("token", "your_token", "API token")
	flag.Parse()
	var balancer Balancer
	switch *lb {
	case "roundrobin":
		balancer = &RoundRobinBalancer{}
	case "weighted":
		balancer = &WeightedBalancer{}
	case "leastconn":
		balancer = &LeastConnBalancer{}
	default:
		balancer = &RoundRobinBalancer{}
	}
	if *role == "master" {
		sentinel := NewSentinel(nil, 0, 0, "service", Master)
		sentinel.Balancer = balancer
		sentinel.Token = *token
		sentinel.ElectMaster()
		signalChan := make(chan os.Signal, 1)
		signal.Notify(signalChan, os.Interrupt)
		<-signalChan
		logger.Info("Master shutdown")
	} else {
		sentinel := NewSentinel(nil, 0, 0, "service", Slave)
		sentinel.Balancer = balancer
		sentinel.Token = *token
		StartGRPCServer(9090)
		select {}
	}
}

func StartGRPCServer(port int) {
	lis, _ := net.Listen("tcp", fmt.Sprintf(":%d", port))
	grpcServer := grpc.NewServer()
	serverProto.RegisterIndexServiceServer(grpcServer, &Sentinel{})
	logger.Info("gRPC server started on :%d", port)
	grpcServer.Serve(lis)
}
