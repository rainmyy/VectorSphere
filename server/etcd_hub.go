package server

import (
	"context"
	"errors"
	"go.etcd.io/etcd/api/v3/v3rpc/rpctypes"
	"strings"
	"sync"
	"time"
)
import etcdv3 "go.etcd.io/etcd/client/v3"

type ServiceHub interface {
	RegisterService(serviceName string, endpoint string, leaseId etcdv3.LeaseID) (etcdv3.LeaseID, error)
	UnRegisterService(serviceName string, endpoint string) error
	GetServiceEndpoints(serviceName string) []string
	GetServiceEndpoint(serviceName string) string
	Close()
}
type EtcdServiceHub struct {
	client       *etcdv3.Client // etcd客户端，用于与etcd进行操作
	heartbeat    int64          // 服务续约的心跳频率，单位：秒
	watched      sync.Map       // 存储已经监视的服务，以避免重复监视
	loadBalancer Balancer       // 负载均衡策略的接口，支持多种负载均衡实现
}

var (
	etcdServiceHub *EtcdServiceHub
	hubOnce        sync.Once
)

const ServiceRootPath = "/opt/seeta_search/services"

func GetHub(etcdServices []string, heartbeat int64) *EtcdServiceHub {
	if etcdServiceHub != nil {
		return etcdServiceHub
	}
	hubOnce.Do(func() {
		client, err := etcdv3.New(etcdv3.Config{
			Endpoints:   etcdServices,
			DialTimeout: time.Duration(heartbeat) * time.Second,
		})
		if err != nil {

		}
		etcdServiceHub = &EtcdServiceHub{
			client:       client,
			heartbeat:    heartbeat,
			watched:      sync.Map{},
			loadBalancer: &RoundRobinBalancer{},
		}
	})
	return etcdServiceHub
}

func (etcd *EtcdServiceHub) RegisterService(service string, endpoint string, leaseId etcdv3.LeaseID) (etcdv3.LeaseID, error) {
	if leaseId <= 0 {
		leaseResp, err := etcd.client.Grant(context.Background(), etcd.heartbeat)
		if err != nil {
			return 0, err
		}
		key := ServiceRootPath + "/" + service + "/" + endpoint
		_, err = etcd.client.Put(context.Background(), key, endpoint, etcdv3.WithLease(leaseResp.ID))
		if err != nil {
			return 0, err
		}
		return leaseResp.ID, nil
	}
	_, err := etcd.client.KeepAliveOnce(context.Background(), leaseId)
	if errors.Is(err, rpctypes.ErrLeaseNotFound) {
		return etcd.RegisterService(service, endpoint, 0)
	}
	if err != nil {

	}

	return leaseId, nil
}

func (etcd *EtcdServiceHub) UnRegisterService(serviceName string, endpoint string) error {
	key := ServiceRootPath + "/" + serviceName + "/" + endpoint
	_, err := etcd.client.Delete(context.Background(), key)
	if err != nil {
		return err
	}

	return nil
}

func (etcd *EtcdServiceHub) GetServiceEndpoints(serviceName string) []string {
	prefix := ServiceRootPath + "/" + serviceName + "/"
	resp, err := etcd.client.Get(context.Background(), prefix, etcdv3.WithPrefix())
	if err != nil {
		return nil
	}
	endpoints := make([]string, len(resp.Kvs))
	for _, kv := range resp.Kvs {
		path := strings.Split(string(kv.Key), "/")
		endpoints = append(endpoints, path[len(path)-1])
	}

	return endpoints
}

func (etcd *EtcdServiceHub) GetServiceEndpoint(serviceName string) string {
	endpoints := etcd.GetServiceEndpoints(serviceName)
	return etcd.loadBalancer.Take(endpoints)
}
func (etcd *EtcdServiceHub) Close() {
	etcd.client.Close()
}
