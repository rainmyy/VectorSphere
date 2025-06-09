package bootstrap

import (
	"context"
	"fmt"
	"github.com/cenkalti/backoff/v4"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
	"log"
	"sync"
	"time"
)

// LeaderElectionManager 领导者选举管理器
type LeaderElectionManager struct {
	client          *clientv3.Client
	session         *concurrency.Session
	elections       sync.Map // map[string]*concurrency.Election
	leaderStatus    sync.Map // map[string]bool
	leaderCallbacks sync.Map // map[string][]func(bool)
	mu              sync.RWMutex
}

// NewLeaderElectionManager 创建领导者选举管理器
func NewLeaderElectionManager(client *clientv3.Client, session *concurrency.Session) *LeaderElectionManager {
	return &LeaderElectionManager{
		client:  client,
		session: session,
	}
}

// CampaignForLeader 参与领导者选举（优化版）
func (lem *LeaderElectionManager) CampaignForLeader(ctx context.Context, electionName string, nodeID string) error {
	fullElectionPath := "/elections/" + electionName
	election := concurrency.NewElection(lem.session, fullElectionPath)
	lem.elections.Store(electionName, election)

	go func() {
		defer func() {
			lem.setLeaderStatus(electionName, false)
			lem.elections.Delete(electionName)
		}()

		// 使用指数退避重试机制
		bo := backoff.NewExponentialBackOff()
		bo.MaxElapsedTime = 0 // 无限重试
		bo.MaxInterval = 30 * time.Second

		backoff.RetryNotify(func() error {
			select {
			case <-ctx.Done():
				return backoff.Permanent(ctx.Err())
			default:
			}

			log.Printf("Campaigning for leadership in election '%s' with node ID '%s'", electionName, nodeID)

			// 参与选举
			if err := election.Campaign(ctx, nodeID); err != nil {
				return fmt.Errorf("campaign failed: %w", err)
			}

			log.Printf("Elected as leader in '%s'", electionName)
			lem.setLeaderStatus(electionName, true)

			// 保持领导者身份
			<-ctx.Done()

			// 主动放弃领导权
			resignCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			if err := election.Resign(resignCtx); err != nil {
				log.Printf("Failed to resign leadership in '%s': %v", electionName, err)
			} else {
				log.Printf("Resigned leadership in '%s'", electionName)
			}

			return nil
		}, backoff.WithContext(bo, ctx), func(err error, duration time.Duration) {
			log.Printf("Leadership campaign failed for '%s', retrying in %v: %v", electionName, duration, err)
			lem.setLeaderStatus(electionName, false)
		})
	}()

	return nil
}

// ObserveLeaderChanges 观察领导者变化
func (lem *LeaderElectionManager) ObserveLeaderChanges(ctx context.Context, electionName string, callback func(string, bool)) {
	// fullElectionPath := "/elections/" + electionName

	go func() {
		electionInterface, exists := lem.elections.Load(electionName)
		if !exists {
			log.Printf("Election %s not found for observation", electionName)
			return
		}

		election := electionInterface.(*concurrency.Election)
		observeChan := election.Observe(ctx)

		for resp := range observeChan {
			if len(resp.Kvs) > 0 {
				leaderID := string(resp.Kvs[0].Value)
				callback(leaderID, true)
				log.Printf("Leader changed in election '%s': %s", electionName, leaderID)
			} else {
				callback("", false)
				log.Printf("No leader in election '%s'", electionName)
			}
		}
	}()
}

// setLeaderStatus 设置领导者状态
func (lem *LeaderElectionManager) setLeaderStatus(electionName string, isLeader bool) {
	lem.leaderStatus.Store(electionName, isLeader)

	// 通知回调函数
	if callbacks, exists := lem.leaderCallbacks.Load(electionName); exists {
		for _, callback := range callbacks.([]func(bool)) {
			go callback(isLeader)
		}
	}
}

// IsLeader 检查是否为领导者
func (lem *LeaderElectionManager) IsLeader(electionName string) bool {
	if status, exists := lem.leaderStatus.Load(electionName); exists {
		return status.(bool)
	}
	return false
}
