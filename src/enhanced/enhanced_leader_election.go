package enhanced

import (
	"VectorSphere/src/library/log"
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
)

// LeadershipState 领导者状态
type LeadershipState int

const (
	Follower LeadershipState = iota
	Candidate
	Leader
	Observer
)

// ElectionStrategy 选举策略
type ElectionStrategy int

const (
	FirstComeFirstServe ElectionStrategy = iota
	PriorityBased
	LoadBased
	RandomSelection
	RoundRobinSelection
)

// LeaderInfo 领导者信息
type LeaderInfo struct {
	NodeID        string            `json:"node_id"`
	Address       string            `json:"address"`
	Port          int               `json:"port"`
	Priority      int               `json:"priority"`
	Load          float64           `json:"load"`
	Capacity      int               `json:"capacity"`
	Version       string            `json:"version"`
	StartTime     time.Time         `json:"start_time"`
	ElectedTime   time.Time         `json:"elected_time"`
	Term          int64             `json:"term"`
	Metadata      map[string]string `json:"metadata"`
	HealthScore   float64           `json:"health_score"`
	LastHeartbeat time.Time         `json:"last_heartbeat"`
}

// ElectionConfig 选举配置
type ElectionConfig struct {
	ElectionKey       string            `json:"election_key"`
	SessionTTL        int               `json:"session_ttl"`
	HeartbeatInterval time.Duration     `json:"heartbeat_interval"`
	ElectionTimeout   time.Duration     `json:"election_timeout"`
	Strategy          ElectionStrategy  `json:"strategy"`
	Priority          int               `json:"priority"`
	AutoReelection    bool              `json:"auto_reelection"`
	MaxTermDuration   time.Duration     `json:"max_term_duration"`
	MinNodes          int               `json:"min_nodes"`
	QuorumSize        int               `json:"quorum_size"`
	Metadata          map[string]string `json:"metadata"`
	HealthCheckFunc   func() float64    `json:"-"`
	LoadCheckFunc     func() float64    `json:"-"`
}

// ElectionEvent 选举事件
type ElectionEvent struct {
	Type       string                 `json:"type"`
	NodeID     string                 `json:"node_id"`
	LeaderInfo *LeaderInfo            `json:"leader_info,omitempty"`
	Term       int64                  `json:"term"`
	Timestamp  time.Time              `json:"timestamp"`
	Message    string                 `json:"message"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// ElectionMetrics 选举指标
type ElectionMetrics struct {
	TotalElections      int64         `json:"total_elections"`
	SuccessfulElections int64         `json:"successful_elections"`
	FailedElections     int64         `json:"failed_elections"`
	LeadershipDuration  time.Duration `json:"leadership_duration"`
	AverageElectionTime time.Duration `json:"average_election_time"`
	LastElectionTime    time.Time     `json:"last_election_time"`
	CurrentTerm         int64         `json:"current_term"`
	TermChanges         int64         `json:"term_changes"`
	HeartbeatsSent      int64         `json:"heartbeats_sent"`
	HeartbeatsReceived  int64         `json:"heartbeats_received"`
	MissedHeartbeats    int64         `json:"missed_heartbeats"`
}

// CandidateInfo 候选者信息
type CandidateInfo struct {
	NodeID      string            `json:"node_id"`
	Address     string            `json:"address"`
	Port        int               `json:"port"`
	Priority    int               `json:"priority"`
	Load        float64           `json:"load"`
	Capacity    int               `json:"capacity"`
	HealthScore float64           `json:"health_score"`
	Version     string            `json:"version"`
	Metadata    map[string]string `json:"metadata"`
	JoinTime    time.Time         `json:"join_time"`
	LastSeen    time.Time         `json:"last_seen"`
}

// ElectionCallback 选举回调函数
type ElectionCallback func(event *ElectionEvent)

// EnhancedLeaderElection 增强的领导者选举
type EnhancedLeaderElection struct {
	client          *clientv3.Client
	session         *concurrency.Session
	election        *concurrency.Election
	config          *ElectionConfig
	nodeInfo        *LeaderInfo
	state           LeadershipState
	currentLeader   *LeaderInfo
	candidates      map[string]*CandidateInfo
	term            int64
	mu              sync.RWMutex
	ctx             context.Context
	cancel          context.CancelFunc
	callbacks       []ElectionCallback
	metrics         *ElectionMetrics
	heartbeatTicker *time.Ticker
	electionTimer   *time.Timer
	isRunning       int32
	basePrefix      string
	leadershipStart time.Time
	lastHeartbeat   time.Time
	heartbeatChan   chan struct{}
	stopChan        chan struct{}
}

// NewEnhancedLeaderElection 创建增强的领导者选举
func NewEnhancedLeaderElection(client *clientv3.Client, config *ElectionConfig, nodeInfo *LeaderInfo) (*EnhancedLeaderElection, error) {
	if config == nil {
		return nil, fmt.Errorf("election config cannot be nil")
	}

	if nodeInfo == nil {
		return nil, fmt.Errorf("node info cannot be nil")
	}

	// 设置默认值
	if config.SessionTTL == 0 {
		config.SessionTTL = 30
	}
	if config.HeartbeatInterval == 0 {
		config.HeartbeatInterval = 5 * time.Second
	}
	if config.ElectionTimeout == 0 {
		config.ElectionTimeout = 15 * time.Second
	}
	if config.MaxTermDuration == 0 {
		config.MaxTermDuration = 24 * time.Hour
	}
	if config.QuorumSize == 0 {
		config.QuorumSize = 1
	}

	ctx, cancel := context.WithCancel(context.Background())

	// 创建会话
	session, err := concurrency.NewSession(client, concurrency.WithTTL(config.SessionTTL))
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create session: %v", err)
	}

	// 创建选举
	election := concurrency.NewElection(session, config.ElectionKey)

	ele := &EnhancedLeaderElection{
		client:        client,
		session:       session,
		election:      election,
		config:        config,
		nodeInfo:      nodeInfo,
		state:         Follower,
		candidates:    make(map[string]*CandidateInfo),
		ctx:           ctx,
		cancel:        cancel,
		callbacks:     make([]ElectionCallback, 0),
		metrics:       &ElectionMetrics{},
		basePrefix:    "/vector_sphere/election",
		heartbeatChan: make(chan struct{}, 1),
		stopChan:      make(chan struct{}),
	}

	// 初始化节点信息
	nodeInfo.StartTime = time.Now()
	nodeInfo.LastHeartbeat = time.Now()
	if nodeInfo.Metadata == nil {
		nodeInfo.Metadata = make(map[string]string)
	}

	log.Info("Enhanced leader election created for node %s", nodeInfo.NodeID)
	return ele, nil
}

// Start 启动选举
func (ele *EnhancedLeaderElection) Start() error {
	if !atomic.CompareAndSwapInt32(&ele.isRunning, 0, 1) {
		return fmt.Errorf("election is already running")
	}

	log.Info("Starting enhanced leader election for node %s", ele.nodeInfo.NodeID)

	// 注册候选者
	if err := ele.registerCandidate(); err != nil {
		log.Error("Failed to register candidate: %v", err)
		return err
	}

	// 启动各种协程
	go ele.runElection()
	go ele.watchLeadership()
	go ele.startHeartbeat()
	go ele.monitorCandidates()
	go ele.collectMetrics()

	log.Info("Enhanced leader election started successfully")
	return nil
}

// Stop 停止选举
func (ele *EnhancedLeaderElection) Stop() error {
	if !atomic.CompareAndSwapInt32(&ele.isRunning, 1, 0) {
		return fmt.Errorf("election is not running")
	}

	log.Info("Stopping enhanced leader election for node %s", ele.nodeInfo.NodeID)

	// 发送停止信号
	close(ele.stopChan)

	// 如果是领导者，主动放弃领导权
	if ele.IsLeader() {
		ele.resignLeadership()
	}

	// 注销候选者
	ele.unregisterCandidate()

	// 停止心跳
	if ele.heartbeatTicker != nil {
		ele.heartbeatTicker.Stop()
	}

	// 停止选举定时器
	if ele.electionTimer != nil {
		ele.electionTimer.Stop()
	}

	// 关闭会话
	if ele.session != nil {
		ele.session.Close()
	}

	// 取消上下文
	ele.cancel()

	log.Info("Enhanced leader election stopped")
	return nil
}

// Campaign 参与选举
func (ele *EnhancedLeaderElection) Campaign() error {
	log.Info("Node %s starting campaign for leadership", ele.nodeInfo.NodeID)

	ele.mu.Lock()
	ele.state = Candidate
	ele.mu.Unlock()

	// 触发选举事件
	ele.triggerEvent(&ElectionEvent{
		Type:      "campaign_started",
		NodeID:    ele.nodeInfo.NodeID,
		Term:      ele.term,
		Timestamp: time.Now(),
		Message:   "Node started campaign for leadership",
	})

	// 更新健康分数和负载
	ele.updateNodeMetrics()

	// 根据策略决定是否参与选举
	if !ele.shouldCampaign() {
		log.Info("Node %s decided not to campaign based on strategy", ele.nodeInfo.NodeID)
		return nil
	}

	// 创建候选者值
	candidateValue, err := ele.createCandidateValue()
	if err != nil {
		return fmt.Errorf("failed to create candidate value: %v", err)
	}

	// 参与选举
	start := time.Now()
	err = ele.election.Campaign(ele.ctx, candidateValue)
	electionDuration := time.Since(start)

	ele.mu.Lock()
	ele.metrics.TotalElections++
	ele.metrics.LastElectionTime = time.Now()
	if ele.metrics.AverageElectionTime == 0 {
		ele.metrics.AverageElectionTime = electionDuration
	} else {
		ele.metrics.AverageElectionTime = (ele.metrics.AverageElectionTime + electionDuration) / 2
	}
	ele.mu.Unlock()

	if err != nil {
		log.Error("Campaign failed for node %s: %v", ele.nodeInfo.NodeID, err)
		ele.mu.Lock()
		ele.state = Follower
		ele.metrics.FailedElections++
		ele.mu.Unlock()

		ele.triggerEvent(&ElectionEvent{
			Type:      "campaign_failed",
			NodeID:    ele.nodeInfo.NodeID,
			Term:      ele.term,
			Timestamp: time.Now(),
			Message:   fmt.Sprintf("Campaign failed: %v", err),
		})
		return err
	}

	// 成为领导者
	ele.becomeLeader()
	return nil
}

// Resign 主动放弃领导权
func (ele *EnhancedLeaderElection) Resign() error {
	if !ele.IsLeader() {
		return fmt.Errorf("node is not a leader")
	}

	log.Info("Node %s resigning from leadership", ele.nodeInfo.NodeID)
	return ele.resignLeadership()
}

// IsLeader 检查是否为领导者
func (ele *EnhancedLeaderElection) IsLeader() bool {
	ele.mu.RLock()
	defer ele.mu.RUnlock()
	return ele.state == Leader
}

// GetLeader 获取当前领导者信息
func (ele *EnhancedLeaderElection) GetLeader() *LeaderInfo {
	ele.mu.RLock()
	defer ele.mu.RUnlock()
	if ele.currentLeader != nil {
		// 返回副本
		leaderCopy := *ele.currentLeader
		return &leaderCopy
	}
	return nil
}

// GetState 获取当前状态
func (ele *EnhancedLeaderElection) GetState() LeadershipState {
	ele.mu.RLock()
	defer ele.mu.RUnlock()
	return ele.state
}

// GetTerm 获取当前任期
func (ele *EnhancedLeaderElection) GetTerm() int64 {
	ele.mu.RLock()
	defer ele.mu.RUnlock()
	return ele.term
}

// GetCandidates 获取候选者列表
func (ele *EnhancedLeaderElection) GetCandidates() []*CandidateInfo {
	ele.mu.RLock()
	defer ele.mu.RUnlock()

	candidates := make([]*CandidateInfo, 0, len(ele.candidates))
	for _, candidate := range ele.candidates {
		candidateCopy := *candidate
		candidates = append(candidates, &candidateCopy)
	}
	return candidates
}

// GetMetrics 获取选举指标
func (ele *EnhancedLeaderElection) GetMetrics() *ElectionMetrics {
	ele.mu.RLock()
	defer ele.mu.RUnlock()
	metricsCopy := *ele.metrics
	return &metricsCopy
}

// AddCallback 添加选举事件回调
func (ele *EnhancedLeaderElection) AddCallback(callback ElectionCallback) {
	ele.mu.Lock()
	ele.callbacks = append(ele.callbacks, callback)
	ele.mu.Unlock()
}

// UpdatePriority 更新节点优先级
func (ele *EnhancedLeaderElection) UpdatePriority(priority int) {
	ele.mu.Lock()
	ele.nodeInfo.Priority = priority
	ele.config.Priority = priority
	ele.mu.Unlock()

	// 更新候选者信息
	go ele.updateCandidateInfo()
}

// UpdateLoad 更新节点负载
func (ele *EnhancedLeaderElection) UpdateLoad(load float64) {
	ele.mu.Lock()
	ele.nodeInfo.Load = load
	ele.mu.Unlock()

	// 更新候选者信息
	go ele.updateCandidateInfo()
}

// UpdateHealthScore 更新健康分数
func (ele *EnhancedLeaderElection) UpdateHealthScore(score float64) {
	ele.mu.Lock()
	ele.nodeInfo.HealthScore = score
	ele.mu.Unlock()

	// 更新候选者信息
	go ele.updateCandidateInfo()
}

// 内部方法实现

// runElection 运行选举主循环
func (ele *EnhancedLeaderElection) runElection() {
	log.Info("Starting election main loop for node %s", ele.nodeInfo.NodeID)

	for {
		select {
		case <-ele.stopChan:
			log.Info("Election main loop stopped")
			return
		case <-ele.ctx.Done():
			log.Info("Election context cancelled")
			return
		default:
			// 检查是否需要重新选举
			if ele.shouldStartElection() {
				go func() {
					if err := ele.Campaign(); err != nil {
						log.Error("Campaign failed: %v", err)
						// 等待一段时间后重试
						time.Sleep(ele.config.ElectionTimeout)
					}
				}()
			}
			time.Sleep(1 * time.Second)
		}
	}
}

// watchLeadership 监控领导权变化
func (ele *EnhancedLeaderElection) watchLeadership() {
	log.Info("Starting leadership watch for node %s", ele.nodeInfo.NodeID)

	for {
		select {
		case <-ele.stopChan:
			log.Info("Leadership watch stopped")
			return
		case <-ele.ctx.Done():
			log.Info("Leadership watch context cancelled")
			return
		default:
			// 观察领导者变化
			ctx, cancel := context.WithTimeout(ele.ctx, 30*time.Second)
			ch := ele.election.Observe(ctx)

			for resp := range ch {
				if len(resp.Kvs) > 0 {
					leaderValue := string(resp.Kvs[0].Value)
					ele.handleLeaderChange(leaderValue)
				} else {
					// 没有领导者
					ele.handleLeaderLoss()
				}
			}
			cancel()

			// 短暂等待后重新开始观察
			time.Sleep(1 * time.Second)
		}
	}
}

// startHeartbeat 启动心跳
func (ele *EnhancedLeaderElection) startHeartbeat() {
	log.Info("Starting heartbeat for node %s", ele.nodeInfo.NodeID)

	ele.heartbeatTicker = time.NewTicker(ele.config.HeartbeatInterval)
	defer ele.heartbeatTicker.Stop()

	for {
		select {
		case <-ele.stopChan:
			log.Info("Heartbeat stopped")
			return
		case <-ele.ctx.Done():
			log.Info("Heartbeat context cancelled")
			return
		case <-ele.heartbeatTicker.C:
			ele.sendHeartbeat()
		case <-ele.heartbeatChan:
			ele.sendHeartbeat()
		}
	}
}

// monitorCandidates 监控候选者
func (ele *EnhancedLeaderElection) monitorCandidates() {
	log.Info("Starting candidate monitoring for node %s", ele.nodeInfo.NodeID)

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ele.stopChan:
			log.Info("Candidate monitoring stopped")
			return
		case <-ele.ctx.Done():
			log.Info("Candidate monitoring context cancelled")
			return
		case <-ticker.C:
			ele.updateCandidatesList()
			ele.cleanupStaleNodes()
		}
	}
}

// collectMetrics 收集指标
func (ele *EnhancedLeaderElection) collectMetrics() {
	log.Info("Starting metrics collection for node %s", ele.nodeInfo.NodeID)

	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ele.stopChan:
			log.Info("Metrics collection stopped")
			return
		case <-ele.ctx.Done():
			log.Info("Metrics collection context cancelled")
			return
		case <-ticker.C:
			ele.updateMetrics()
		}
	}
}

// becomeLeader 成为领导者
func (ele *EnhancedLeaderElection) becomeLeader() {
	log.Info("Node %s became leader", ele.nodeInfo.NodeID)

	ele.mu.Lock()
	ele.state = Leader
	ele.term++
	ele.nodeInfo.ElectedTime = time.Now()
	ele.nodeInfo.Term = ele.term
	ele.currentLeader = ele.nodeInfo
	ele.leadershipStart = time.Now()
	ele.metrics.SuccessfulElections++
	ele.metrics.CurrentTerm = ele.term
	ele.metrics.TermChanges++
	ele.mu.Unlock()

	// 触发领导者选举事件
	ele.triggerEvent(&ElectionEvent{
		Type:       "leader_elected",
		NodeID:     ele.nodeInfo.NodeID,
		LeaderInfo: ele.nodeInfo,
		Term:       ele.term,
		Timestamp:  time.Now(),
		Message:    "Node became leader",
	})

	// 启动领导者任期监控
	if ele.config.MaxTermDuration > 0 {
		go ele.monitorTermDuration()
	}

	// 立即发送心跳
	select {
	case ele.heartbeatChan <- struct{}{}:
	default:
	}
}

// resignLeadership 放弃领导权
func (ele *EnhancedLeaderElection) resignLeadership() error {
	log.Info("Node %s resigning leadership", ele.nodeInfo.NodeID)

	ele.mu.Lock()
	if ele.state == Leader {
		leadershipDuration := time.Since(ele.leadershipStart)
		ele.metrics.LeadershipDuration += leadershipDuration
	}
	ele.state = Follower
	ele.currentLeader = nil
	ele.mu.Unlock()

	// 触发领导者辞职事件
	ele.triggerEvent(&ElectionEvent{
		Type:      "leader_resigned",
		NodeID:    ele.nodeInfo.NodeID,
		Term:      ele.term,
		Timestamp: time.Now(),
		Message:   "Leader resigned",
	})

	// 放弃选举
	return ele.election.Resign(ele.ctx)
}

// handleLeaderChange 处理领导者变化
func (ele *EnhancedLeaderElection) handleLeaderChange(leaderValue string) {
	var newLeader LeaderInfo
	if err := json.Unmarshal([]byte(leaderValue), &newLeader); err != nil {
		log.Error("Failed to unmarshal leader info: %v", err)
		return
	}

	ele.mu.Lock()
	oldLeader := ele.currentLeader
	ele.currentLeader = &newLeader

	// 如果不是自己成为领导者，则设置为跟随者
	if newLeader.NodeID != ele.nodeInfo.NodeID {
		if ele.state == Leader {
			// 失去领导权
			leadershipDuration := time.Since(ele.leadershipStart)
			ele.metrics.LeadershipDuration += leadershipDuration
		}
		ele.state = Follower
	}
	ele.mu.Unlock()

	// 检查是否是新的领导者
	if oldLeader == nil || oldLeader.NodeID != newLeader.NodeID {
		log.Info("New leader detected: %s (term: %d)", newLeader.NodeID, newLeader.Term)

		ele.triggerEvent(&ElectionEvent{
			Type:       "leader_changed",
			NodeID:     newLeader.NodeID,
			LeaderInfo: &newLeader,
			Term:       newLeader.Term,
			Timestamp:  time.Now(),
			Message:    fmt.Sprintf("New leader: %s", newLeader.NodeID),
		})
	}
}

// handleLeaderLoss 处理领导者丢失
func (ele *EnhancedLeaderElection) handleLeaderLoss() {
	log.Warning("Leader lost, no current leader")

	ele.mu.Lock()
	oldLeader := ele.currentLeader
	ele.currentLeader = nil
	if ele.state == Leader {
		leadershipDuration := time.Since(ele.leadershipStart)
		ele.metrics.LeadershipDuration += leadershipDuration
		ele.state = Follower
	}
	ele.mu.Unlock()

	if oldLeader != nil {
		ele.triggerEvent(&ElectionEvent{
			Type:      "leader_lost",
			NodeID:    oldLeader.NodeID,
			Term:      ele.term,
			Timestamp: time.Now(),
			Message:   "Leader lost",
		})
	}
}

// shouldStartElection 判断是否应该开始选举
func (ele *EnhancedLeaderElection) shouldStartElection() bool {
	ele.mu.RLock()
	defer ele.mu.RUnlock()

	// 如果已经是领导者或候选者，不需要开始新选举
	if ele.state == Leader || ele.state == Candidate {
		return false
	}

	// 如果有当前领导者且心跳正常，不需要选举
	if ele.currentLeader != nil {
		timeSinceLastHeartbeat := time.Since(ele.currentLeader.LastHeartbeat)
		if timeSinceLastHeartbeat < ele.config.ElectionTimeout {
			return false
		}
	}

	// 检查最小节点数要求
	if len(ele.candidates) < ele.config.MinNodes {
		return false
	}

	return true
}

// shouldCampaign 判断是否应该参与选举
func (ele *EnhancedLeaderElection) shouldCampaign() bool {
	switch ele.config.Strategy {
	case PriorityBased:
		return ele.shouldCampaignByPriority()
	case LoadBased:
		return ele.shouldCampaignByLoad()
	case RandomSelection:
		return rand.Float64() < 0.5 // 50% 概率参与选举
	case RoundRobinSelection:
		return ele.shouldCampaignByRoundRobin()
	default:
		return true // FirstComeFirstServe 总是参与
	}
}

// shouldCampaignByPriority 基于优先级判断是否参与选举
func (ele *EnhancedLeaderElection) shouldCampaignByPriority() bool {
	ele.mu.RLock()
	defer ele.mu.RUnlock()

	myPriority := ele.nodeInfo.Priority
	for _, candidate := range ele.candidates {
		if candidate.Priority > myPriority {
			return false // 有更高优先级的候选者
		}
	}
	return true
}

// shouldCampaignByLoad 基于负载判断是否参与选举
func (ele *EnhancedLeaderElection) shouldCampaignByLoad() bool {
	ele.mu.RLock()
	defer ele.mu.RUnlock()

	myLoad := ele.nodeInfo.Load
	for _, candidate := range ele.candidates {
		if candidate.Load < myLoad {
			return false // 有更低负载的候选者
		}
	}
	return true
}

// shouldCampaignByRoundRobin 基于轮询判断是否参与选举
func (ele *EnhancedLeaderElection) shouldCampaignByRoundRobin() bool {
	// 简化实现：基于节点ID的哈希值和当前时间
	hash := 0
	for _, b := range []byte(ele.nodeInfo.NodeID) {
		hash = hash*31 + int(b)
	}
	return (hash+int(time.Now().Unix()/60))%len(ele.candidates) == 0
}

// createCandidateValue 创建候选者值
func (ele *EnhancedLeaderElection) createCandidateValue() (string, error) {
	ele.updateNodeMetrics()

	value, err := json.Marshal(ele.nodeInfo)
	if err != nil {
		return "", fmt.Errorf("failed to marshal candidate value: %v", err)
	}
	return string(value), nil
}

// updateNodeMetrics 更新节点指标
func (ele *EnhancedLeaderElection) updateNodeMetrics() {
	ele.mu.Lock()
	defer ele.mu.Unlock()

	// 更新健康分数
	if ele.config.HealthCheckFunc != nil {
		ele.nodeInfo.HealthScore = ele.config.HealthCheckFunc()
	}

	// 更新负载
	if ele.config.LoadCheckFunc != nil {
		ele.nodeInfo.Load = ele.config.LoadCheckFunc()
	}

	// 更新心跳时间
	ele.nodeInfo.LastHeartbeat = time.Now()
}

// registerCandidate 注册候选者
func (ele *EnhancedLeaderElection) registerCandidate() error {
	candidateInfo := &CandidateInfo{
		NodeID:      ele.nodeInfo.NodeID,
		Address:     ele.nodeInfo.Address,
		Port:        ele.nodeInfo.Port,
		Priority:    ele.nodeInfo.Priority,
		Load:        ele.nodeInfo.Load,
		Capacity:    ele.nodeInfo.Capacity,
		HealthScore: ele.nodeInfo.HealthScore,
		Version:     ele.nodeInfo.Version,
		Metadata:    ele.nodeInfo.Metadata,
		JoinTime:    time.Now(),
		LastSeen:    time.Now(),
	}

	value, err := json.Marshal(candidateInfo)
	if err != nil {
		return fmt.Errorf("failed to marshal candidate info: %v", err)
	}

	key := fmt.Sprintf("%s/candidates/%s", ele.basePrefix, ele.nodeInfo.NodeID)
	_, err = ele.client.Put(ele.ctx, key, string(value))
	if err != nil {
		return fmt.Errorf("failed to register candidate: %v", err)
	}

	log.Info("Candidate registered: %s", ele.nodeInfo.NodeID)
	return nil
}

// unregisterCandidate 注销候选者
func (ele *EnhancedLeaderElection) unregisterCandidate() {
	key := fmt.Sprintf("%s/candidates/%s", ele.basePrefix, ele.nodeInfo.NodeID)
	_, err := ele.client.Delete(ele.ctx, key)
	if err != nil {
		log.Error("Failed to unregister candidate: %v", err)
	} else {
		log.Info("Candidate unregistered: %s", ele.nodeInfo.NodeID)
	}
}

// updateCandidateInfo 更新候选者信息
func (ele *EnhancedLeaderElection) updateCandidateInfo() {
	if atomic.LoadInt32(&ele.isRunning) == 0 {
		return
	}

	candidateInfo := &CandidateInfo{
		NodeID:      ele.nodeInfo.NodeID,
		Address:     ele.nodeInfo.Address,
		Port:        ele.nodeInfo.Port,
		Priority:    ele.nodeInfo.Priority,
		Load:        ele.nodeInfo.Load,
		Capacity:    ele.nodeInfo.Capacity,
		HealthScore: ele.nodeInfo.HealthScore,
		Version:     ele.nodeInfo.Version,
		Metadata:    ele.nodeInfo.Metadata,
		LastSeen:    time.Now(),
	}

	value, err := json.Marshal(candidateInfo)
	if err != nil {
		log.Error("Failed to marshal candidate info: %v", err)
		return
	}

	key := fmt.Sprintf("%s/candidates/%s", ele.basePrefix, ele.nodeInfo.NodeID)
	_, err = ele.client.Put(ele.ctx, key, string(value))
	if err != nil {
		log.Error("Failed to update candidate info: %v", err)
	}
}

// updateCandidatesList 更新候选者列表
func (ele *EnhancedLeaderElection) updateCandidatesList() {
	key := fmt.Sprintf("%s/candidates/", ele.basePrefix)
	resp, err := ele.client.Get(ele.ctx, key, clientv3.WithPrefix())
	if err != nil {
		log.Error("Failed to get candidates list: %v", err)
		return
	}

	ele.mu.Lock()
	defer ele.mu.Unlock()

	// 清空现有候选者列表
	ele.candidates = make(map[string]*CandidateInfo)

	// 解析候选者信息
	for _, kv := range resp.Kvs {
		var candidate CandidateInfo
		if err := json.Unmarshal(kv.Value, &candidate); err != nil {
			log.Error("Failed to unmarshal candidate info: %v", err)
			continue
		}
		ele.candidates[candidate.NodeID] = &candidate
	}

	log.Debug("Updated candidates list: %d candidates", len(ele.candidates))
}

// cleanupStaleNodes 清理过期节点
func (ele *EnhancedLeaderElection) cleanupStaleNodes() {
	ele.mu.Lock()
	defer ele.mu.Unlock()

	now := time.Now()
	staleThreshold := 5 * ele.config.HeartbeatInterval

	for nodeID, candidate := range ele.candidates {
		if now.Sub(candidate.LastSeen) > staleThreshold {
			log.Warning("Removing stale candidate: %s", nodeID)
			delete(ele.candidates, nodeID)

			// 从etcd中删除
			key := fmt.Sprintf("%s/candidates/%s", ele.basePrefix, nodeID)
			go func(k string) {
				_, err := ele.client.Delete(ele.ctx, k)
				if err != nil {
					log.Error("Failed to delete stale candidate: %v", err)
				}
			}(key)
		}
	}
}

// sendHeartbeat 发送心跳
func (ele *EnhancedLeaderElection) sendHeartbeat() {
	if atomic.LoadInt32(&ele.isRunning) == 0 {
		return
	}

	ele.mu.Lock()
	ele.lastHeartbeat = time.Now()
	ele.nodeInfo.LastHeartbeat = ele.lastHeartbeat
	ele.metrics.HeartbeatsSent++
	ele.mu.Unlock()

	// 更新候选者信息（包含心跳时间）
	go ele.updateCandidateInfo()

	log.Debug("Heartbeat sent by node %s", ele.nodeInfo.NodeID)
}

// monitorTermDuration 监控任期持续时间
func (ele *EnhancedLeaderElection) monitorTermDuration() {
	timer := time.NewTimer(ele.config.MaxTermDuration)
	defer timer.Stop()

	select {
	case <-timer.C:
		if ele.IsLeader() {
			log.Info("Maximum term duration reached, resigning leadership")
			ele.resignLeadership()
		}
	case <-ele.stopChan:
		return
	case <-ele.ctx.Done():
		return
	}
}

// updateMetrics 更新指标
func (ele *EnhancedLeaderElection) updateMetrics() {
	ele.mu.Lock()
	defer ele.mu.Unlock()

	// 更新当前任期
	ele.metrics.CurrentTerm = ele.term

	// 如果是领导者，更新领导时间
	if ele.state == Leader {
		currentLeadershipDuration := time.Since(ele.leadershipStart)
		log.Debug("Current leadership duration: %v", currentLeadershipDuration)
	}

	log.Debug("Election metrics updated: total_elections=%d, successful=%d, failed=%d, current_term=%d",
		ele.metrics.TotalElections, ele.metrics.SuccessfulElections, ele.metrics.FailedElections, ele.metrics.CurrentTerm)
}

// triggerEvent 触发事件
func (ele *EnhancedLeaderElection) triggerEvent(event *ElectionEvent) {
	ele.mu.RLock()
	callbacks := make([]ElectionCallback, len(ele.callbacks))
	copy(callbacks, ele.callbacks)
	ele.mu.RUnlock()

	// 异步调用回调函数
	go func() {
		for _, callback := range callbacks {
			if callback != nil {
				try := func() {
					defer func() {
						if r := recover(); r != nil {
							log.Error("Election callback panic: %v", r)
						}
					}()
					callback(event)
				}
				try()
			}
		}
	}()

	log.Info("Election event triggered: type=%s, node=%s, term=%d, message=%s",
		event.Type, event.NodeID, event.Term, event.Message)
}
