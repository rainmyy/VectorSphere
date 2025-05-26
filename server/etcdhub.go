package server

import (
	"context"
	"errors"
	"go.etcd.io/etcd/api/v3/v3rpc/rpctypes"
	etcdv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
	"strconv"
	"strings"
	"sync"
	"time"
)

type ServiceHub interface {
	GetClient() *etcdv3.Client
	RegisterService(serviceName string, endpoint *EndPoint, leaseId etcdv3.LeaseID) (etcdv3.LeaseID, error)
	UnRegisterService(serviceName string, endpoint *EndPoint) error
	GetServiceEndpoints(serviceName string) []EndPoint
	GetServiceEndpoint(serviceName string) EndPoint
	Close()
}

type EtcdServiceHub struct {
	client    *etcdv3.Client // etcd客户端，用于与etcd进行操作
	heartbeat int64          // 服务续约的心跳频率，单位：秒
	leaseId   etcdv3.LeaseID
	session   *concurrency.Session
	election  *concurrency.Election

	watched      sync.Map // 存储已经监视的服务，以避免重复监视
	loadBalancer Balancer // 负载均衡策略的接口，支持多种负载均衡实现
}

var (
	etcdServiceHub *EtcdServiceHub
	hubOnce        sync.Once
)

const ServiceRootPath = "/opt/seeta_search/services"

func init() {

}

func GetHub(endPoints []EndPoint, heartbeat int64, serviceName string) (error, *EtcdServiceHub) {
	if etcdServiceHub != nil {
		return nil, etcdServiceHub
	}

	var endPointIp []string
	for _, endPoint := range endPoints {
		endPointIp = append(endPointIp, endPoint.Ip+":"+strconv.Itoa(endPoint.Port))
	}
	var er error
	hubOnce.Do(func() {
		client, err := etcdv3.New(etcdv3.Config{
			Endpoints:            endPointIp,
			DialTimeout:          time.Duration(heartbeat) * time.Second,
			DialKeepAliveTime:    3 * time.Second,
			DialKeepAliveTimeout: 3 * time.Second,
		})
		if err != nil {
			er = err
			return
		}

		session, err := concurrency.NewSession(client, concurrency.WithTTL(3))
		if err != nil {
			er = err
			return
		}
		defer session.Close()
		election := concurrency.NewElection(session, serviceName)
		etcdServiceHub = &EtcdServiceHub{
			client:       client,
			heartbeat:    heartbeat,
			watched:      sync.Map{},
			session:      session,
			election:     election,
			loadBalancer: LoadBalanceFactory(WeightedRoundRobin),
		}
	})

	return er, etcdServiceHub
}

func (etcd *EtcdServiceHub) RegisterService(service string, endpoint *EndPoint, leaseId etcdv3.LeaseID) (etcdv3.LeaseID, error) {
	if etcd.client == nil {
		return 0, errors.New("etcd client is nil")
	}

	if leaseId <= 0 {
		lease := etcdv3.NewLease(etcd.client)
		leaseResp, err := lease.Grant(context.Background(), 3)
		if err != nil {
			return 0, err
		}
		key := ServiceRootPath + "/" + service + "/" + endpoint.Ip
		_, err = etcd.client.Put(context.Background(), key, endpoint.Ip, etcdv3.WithLease(leaseResp.ID))
		if err != nil {
			return 0, err
		}
		return leaseResp.ID, nil
	}
	_, err := etcd.client.KeepAliveOnce(context.Background(), leaseId)
	if errors.Is(err, rpctypes.ErrLeaseNotFound) {
		return etcd.RegisterService(service, endpoint, 0)
	}

	etcd.leaseId = leaseId
	return leaseId, nil
}

func (etcd *EtcdServiceHub) UnRegisterService(serviceName string, endpoint *EndPoint) error {
	key := ServiceRootPath + "/" + serviceName + "/" + endpoint.Ip
	_, err := etcd.client.Delete(context.Background(), key)
	if err != nil {
		return err
	}

	_, err = etcd.client.Revoke(context.Background(), etcd.leaseId)
	return err
}

func (etcd *EtcdServiceHub) GetServiceEndpoints(serviceName string) []EndPoint {
	prefix := ServiceRootPath + "/" + serviceName + "/"
	resp, err := etcd.client.Get(context.Background(), prefix, etcdv3.WithPrefix())
	if err != nil {
		return nil
	}
	endpoints := make([]EndPoint, len(resp.Kvs))
	for _, kv := range resp.Kvs {
		path := strings.Split(string(kv.Key), "/")
		endpoint := EndPoint{Ip: path[len(path)-1]}
		endpoints = append(endpoints, endpoint)
	}

	return endpoints
}

func (etcd *EtcdServiceHub) GetServiceEndpoint(serviceName string) EndPoint {
	endpoints := etcd.GetServiceEndpoints(serviceName)
	etcd.loadBalancer.Set(endpoints...)
	return etcd.loadBalancer.Take()
}

func (etcd *EtcdServiceHub) Close() {
	etcd.client.Close()
}
