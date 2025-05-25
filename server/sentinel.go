package server

import (
	"context"
	"errors"
	"go.etcd.io/etcd/client/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"seetaSearch/library/log"
	"seetaSearch/messages"
	"sync"
	"sync/atomic"
	"time"
)

type Sentinel struct {
	Hub        ServiceHub
	connPool   sync.Map
	leaseId    clientv3.LeaseID
	ServiceKey string
}

var _ ServerInterface = (*Sentinel)(nil)

func NewSentinel(endPoints []EndPoint, heartBeat int64, qps int, serviceName string) *Sentinel {
	sentinel := &Sentinel{
		Hub:        GetHubProxy(endPoints, heartBeat, qps, serviceName),
		connPool:   sync.Map{},
		ServiceKey: serviceName,
	}
	go sentinel.WatchServiceChanges()
	return sentinel
}

func (s *Sentinel) RegisterSentinel(ttl int64) error {
	resp, err := s.Hub.GetClient().Grant(context.Background(), ttl)
	if err != nil {
		return err
	}
	s.leaseId = resp.ID
	_, err = s.Hub.GetClient().Put(context.Background(), s.ServiceKey, "alive", clientv3.WithLease(resp.ID))
	if err != nil {
		return err
	}
	ch, err := s.Hub.GetClient().KeepAlive(context.Background(), resp.ID)
	if err != nil {
		return err
	}
	go func() {
		for range ch {
			log.Info("sentinel heartbeat success")
		}
		log.Warning("Etcd 连接中断，Sentinel 租约 ID: %d\n", s.leaseId)
	}()
	return nil
}

func (s *Sentinel) WatchServiceChanges() {
	watcher := clientv3.NewWatcher(s.Hub.GetClient())
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

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
}
func (s *Sentinel) GetGrpcConn(point EndPoint) *grpc.ClientConn {
	v, ok := s.connPool.Load(point.Ip)
	if ok {
		conn := v.(*grpc.ClientConn)
		state := conn.GetState()
		if state != connectivity.TransientFailure && state != connectivity.Shutdown {
			return conn
		}
		conn.Close()
		s.connPool.Delete(point.Ip)
	}
	cts, cancel := context.WithTimeout(context.Background(), 200*time.Microsecond)
	defer cancel()
	grpcConn, err := grpc.DialContext(cts, point.Ip, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
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

func (s *Sentinel) Search(query *messages.TermQuery, onFlag, offFlag uint64, orFlags []uint64) []*messages.Document {
	endpoints := s.Hub.GetServiceEndpoints(s.ServiceKey)
	if len(endpoints) == 0 {
		return nil
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
	return docs
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
