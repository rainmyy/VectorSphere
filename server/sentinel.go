package server

import (
	"context"
	"errors"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"seetaSearch/messages"
	"sync"
	"sync/atomic"
	"time"
)

type Sentinel struct {
	hub         ServiceHub
	connPool    sync.Map
	IndexServer string
}

const IndexService = "seata_search"

var _ ServerInterface = (*Sentinel)(nil)

func NewSentinel(serviceNames []string) *Sentinel {
	return &Sentinel{
		hub:         GetHubProxy(serviceNames, 3, 100),
		connPool:    sync.Map{},
		IndexServer: IndexService,
	}
}

func (s *Sentinel) GetGrpcConn(point EndPoint) *grpc.ClientConn {
	v, ok := s.connPool.Load(point.address)
	if ok {
		conn := v.(*grpc.ClientConn)
		state := conn.GetState()
		if state != connectivity.TransientFailure && state != connectivity.Shutdown {
			return conn
		}
		conn.Close()
		s.connPool.Delete(point.address)
	}
	cts, cancel := context.WithTimeout(context.Background(), 200*time.Microsecond)
	defer cancel()
	grpcConn, err := grpc.DialContext(cts, point.address, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
	if err != nil {
		return nil
	}
	s.connPool.Store(point.address, grpcConn)

	return grpcConn
}

func (s *Sentinel) AddDoc(doc *messages.Document) (int, error) {
	endPoint := s.hub.GetServiceEndpoint(s.IndexServer)
	if len(endPoint.address) == 0 {
		return -1, errors.New("服务节点不存在")
	}
	conn := s.GetGrpcConn(endPoint)
	if conn == nil {
		return -1, errors.New("无法连接到" + endPoint.address)
	}
	client := NewIndexServiceClient(conn)
	affected, err := client.AddDoc(context.Background(), doc)
	if err != nil {
		return -1, err
	}
	return int(affected.Count), nil
}

func (s *Sentinel) DelDoc(docId *DocId) int {
	endpoints := s.hub.GetServiceEndpoints(s.IndexServer)
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
	endpoints := s.hub.GetServiceEndpoints(s.IndexServer)
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
	endpoints := s.hub.GetServiceEndpoints(s.IndexServer)
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
	s.hub.Close()
	return
}
