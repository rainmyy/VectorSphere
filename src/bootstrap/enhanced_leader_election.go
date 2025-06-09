package bootstrap

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/cenkalti/backoff/v4"
	etcdv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
	"sync"
	"time"
)

// EnhancedLeaderElection 增强的领导者选举
type EnhancedLeaderElection struct {
	client          *etcdv3.Client
	session         *concurrency.Session
	election        *concurrency.Election
	leaderInfo      *LeaderInfo
	mu              sync.RWMutex
	leaderCallbacks []func(bool, *LeaderInfo)
	ctx             context.Context
	cancel          context.CancelFunc
	retryPolicy     *backoff.ExponentialBackOff
}

// LeaderInfo 领导者信息
type LeaderInfo struct {
	NodeID      string            `json:"nodeId"`
	Address     string            `json:"address"`
	ElectedTime time.Time         `json:"electedTime"`
	Term        int64             `json:"term"`
	Metadata    map[string]string `json:"metadata"`
}

// NewEnhancedLeaderElection 创建增强的领导者选举
func NewEnhancedLeaderElection(client *etcdv3.Client, electionKey string, nodeID string, address string) (*EnhancedLeaderElection, error) {
	ctx, cancel := context.WithCancel(context.Background())

	session, err := concurrency.NewSession(client, concurrency.WithTTL(30))
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	election := concurrency.NewElection(session, electionKey)

	retryPolicy := backoff.NewExponentialBackOff()
	retryPolicy.InitialInterval = 1 * time.Second
	retryPolicy.MaxInterval = 30 * time.Second
	retryPolicy.MaxElapsedTime = 0 // 无限重试

	ele := &EnhancedLeaderElection{
		client:      client,
		session:     session,
		election:    election,
		ctx:         ctx,
		cancel:      cancel,
		retryPolicy: retryPolicy,
	}

	// 启动选举
	go ele.startElection(nodeID, address)
	// 启动领导者监听
	go ele.observeLeader()

	return ele, nil
}

// startElection 开始选举
func (ele *EnhancedLeaderElection) startElection(nodeID, address string) {
	operation := func() error {
		leaderInfo := &LeaderInfo{
			NodeID:      nodeID,
			Address:     address,
			ElectedTime: time.Now(),
			Term:        time.Now().Unix(),
			Metadata:    make(map[string]string),
		}

		data, err := json.Marshal(leaderInfo)
		if err != nil {
			return fmt.Errorf("failed to marshal leader info: %w", err)
		}

		err = ele.election.Campaign(ele.ctx, string(data))
		if err != nil {
			return fmt.Errorf("failed to campaign: %w", err)
		}

		// 成为领导者
		ele.mu.Lock()
		ele.leaderInfo = leaderInfo
		ele.mu.Unlock()

		// 通知回调
		ele.notifyCallbacks(true, leaderInfo)

		// 等待失去领导权
		<-ele.ctx.Done()
		return nil
	}

	backoff.Retry(operation, backoff.WithContext(ele.retryPolicy, ele.ctx))
}

// observeLeader 观察领导者变化
func (ele *EnhancedLeaderElection) observeLeader() {
	for {
		select {
		case <-ele.ctx.Done():
			return
		default:
			resp, err := ele.election.Leader(ele.ctx)
			if err != nil {
				time.Sleep(5 * time.Second)
				continue
			}

			if len(resp.Kvs) > 0 {
				var leaderInfo LeaderInfo
				if err := json.Unmarshal(resp.Kvs[0].Value, &leaderInfo); err == nil {
					ele.mu.Lock()
					currentLeader := ele.leaderInfo
					ele.mu.Unlock()

					// 检查领导者是否变化
					if currentLeader == nil || currentLeader.NodeID != leaderInfo.NodeID {
						ele.notifyCallbacks(false, &leaderInfo)
					}
				}
			}
			time.Sleep(10 * time.Second)
		}
	}
}

// AddLeaderCallback 添加领导者变化回调
func (ele *EnhancedLeaderElection) AddLeaderCallback(callback func(bool, *LeaderInfo)) {
	ele.mu.Lock()
	ele.leaderCallbacks = append(ele.leaderCallbacks, callback)
	ele.mu.Unlock()
}

// notifyCallbacks 通知回调
func (ele *EnhancedLeaderElection) notifyCallbacks(isLeader bool, leaderInfo *LeaderInfo) {
	ele.mu.RLock()
	callbacks := make([]func(bool, *LeaderInfo), len(ele.leaderCallbacks))
	copy(callbacks, ele.leaderCallbacks)
	ele.mu.RUnlock()

	for _, callback := range callbacks {
		go callback(isLeader, leaderInfo)
	}
}

// IsLeader 检查是否为领导者
func (ele *EnhancedLeaderElection) IsLeader() bool {
	ele.mu.RLock()
	defer ele.mu.RUnlock()
	return ele.leaderInfo != nil
}

// GetLeaderInfo 获取当前领导者信息
func (ele *EnhancedLeaderElection) GetLeaderInfo() *LeaderInfo {
	ele.mu.RLock()
	defer ele.mu.RUnlock()
	if ele.leaderInfo != nil {
		info := *ele.leaderInfo
		return &info
	}
	return nil
}

// Resign 主动放弃领导权
func (ele *EnhancedLeaderElection) Resign() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := ele.election.Resign(ctx)
	if err != nil {
		return fmt.Errorf("failed to resign: %w", err)
	}

	ele.mu.Lock()
	ele.leaderInfo = nil
	ele.mu.Unlock()

	return nil
}
