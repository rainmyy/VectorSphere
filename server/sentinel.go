package server

import (
	"context"
	"errors"
	"github.com/rainmyy/seetaSearch/index"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"sync"
	"time"
)

type Sentinel struct {
	hub      ServiceHub
	connPool sync.Map
}

var _ IndexInterface = (*Sentinel)(nil)

func NewSentinel(serviceNames []string) *Sentinel {
	return &Sentinel{
		hub:      GetHubProxy(serviceNames, 3, 100),
		connPool: sync.Map{},
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

func (s *Sentinel) AddDoc(doc index.Document) (int, error) {
	endPoint := s.hub.GetServiceEndpoint(IndexService)
	if len(endPoint.address) == 0 {
		return -1, errors.New("服务节点不存在")
	}
	conn := s.GetGrpcConn(endPoint)
	if conn == nil {
		return -1, errors.New("无法连接到" + endPoint.address)
	}
	client := NewServiceClient(conn)
	affected, err := client.AddDoc(context.Background(), &doc)
	if err != nil {
		return -1, err
	}
	return int(affected.Count), nil
}

func (s *Sentinel) DelDoc(docId string) int {
	endpoints := s.hub.GetServiceEndpoints(IndexService)
	if len(endpoints) == 0 {
		return 0
	}
	var n int32
	wg := sync.WaitGroup{}
	wg.Add(len(endpoints))
}
