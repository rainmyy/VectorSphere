package enhanced

import (
	"VectorSphere/src/library/log"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"strings"
	"sync"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
)

// LockType 锁类型
type LockType int

const (
	MutexLock LockType = iota
	ReadWriteLock
	ReentrantLock
	TimedLock
)

// LockInfo 锁信息
type LockInfo struct {
	LockID     string            `json:"lock_id"`
	LockKey    string            `json:"lock_key"`
	LockType   LockType          `json:"lock_type"`
	Owner      string            `json:"owner"`
	OwnerID    string            `json:"owner_id"`
	AcquiredAt time.Time         `json:"acquired_at"`
	ExpiresAt  time.Time         `json:"expires_at"`
	TTL        int64             `json:"ttl"`
	Reentrant  int               `json:"reentrant"`   // 重入次数
	ReadCount  int               `json:"read_count"`  // 读锁计数
	WriteCount int               `json:"write_count"` // 写锁计数
	Metadata   map[string]string `json:"metadata"`
	StackTrace string            `json:"stack_trace,omitempty"`
}

// LockRequest 锁请求
type LockRequest struct {
	LockKey    string            `json:"lock_key"`
	LockType   LockType          `json:"lock_type"`
	Owner      string            `json:"owner"`
	OwnerID    string            `json:"owner_id"`
	TTL        int64             `json:"ttl"`
	Timeout    time.Duration     `json:"timeout"`
	RetryDelay time.Duration     `json:"retry_delay"`
	MaxRetries int               `json:"max_retries"`
	Metadata   map[string]string `json:"metadata"`
	Blocking   bool              `json:"blocking"`
	ReadOnly   bool              `json:"read_only"`
}

// LockStats 锁统计信息
type LockStats struct {
	LockKey         string        `json:"lock_key"`
	TotalAcquires   int64         `json:"total_acquires"`
	TotalReleases   int64         `json:"total_releases"`
	CurrentHolders  int           `json:"current_holders"`
	WaitingCount    int           `json:"waiting_count"`
	AverageHoldTime time.Duration `json:"average_hold_time"`
	MaxHoldTime     time.Duration `json:"max_hold_time"`
	LastAccessed    time.Time     `json:"last_accessed"`
	DeadlockCount   int64         `json:"deadlock_count"`
}

// LockEvent 锁事件
type LockEvent struct {
	Type      string        `json:"type"` // "acquire", "release", "timeout", "deadlock"
	LockKey   string        `json:"lock_key"`
	Owner     string        `json:"owner"`
	OwnerID   string        `json:"owner_id"`
	Timestamp time.Time     `json:"timestamp"`
	Duration  time.Duration `json:"duration,omitempty"`
	Error     string        `json:"error,omitempty"`
}

// DeadlockDetector 死锁检测器
type DeadlockDetector struct {
	WaitGraph   map[string][]string // owner -> waiting for owners
	LockOwners  map[string]string   // lockKey -> owner
	Mu          sync.RWMutex
	detectCycle bool
}

// EnhancedDistributedLock 增强的分布式锁
type EnhancedDistributedLock struct {
	client           *clientv3.Client
	session          *concurrency.Session
	locks            map[string]*LockInfo      // lockID -> lockInfo
	lockStats        map[string]*LockStats     // lockKey -> stats
	waitingQueues    map[string][]*LockRequest // lockKey -> waiting requests
	deadlockDetector *DeadlockDetector
	eventListeners   map[string]chan *LockEvent
	mu               sync.RWMutex
	listenersMu      sync.RWMutex
	ctx              context.Context
	cancel           context.CancelFunc
	basePrefix       string
	defaultTTL       int64
	monitorEnabled   bool
	auditEnabled     bool
	cleanupInterval  time.Duration
}

// EnhancedDistributedLockConfig 增强分布式锁配置
type EnhancedDistributedLockConfig struct {
	BasePrefix      string        `json:"base_prefix"`
	DefaultTTL      int64         `json:"default_ttl"`
	MonitorEnabled  bool          `json:"monitor_enabled"`
	AuditEnabled    bool          `json:"audit_enabled"`
	CleanupInterval time.Duration `json:"cleanup_interval"`
	SessionTTL      int           `json:"session_ttl"`
}

// NewEnhancedDistributedLock 创建增强的分布式锁
func NewEnhancedDistributedLock(client *clientv3.Client, config *EnhancedDistributedLockConfig) (*EnhancedDistributedLock, error) {
	ctx, cancel := context.WithCancel(context.Background())

	if config == nil {
		config = &EnhancedDistributedLockConfig{
			BasePrefix:      "/vector_sphere/locks",
			DefaultTTL:      30,
			MonitorEnabled:  true,
			AuditEnabled:    true,
			CleanupInterval: 60 * time.Second,
			SessionTTL:      60,
		}
	}

	// 创建会话
	session, err := concurrency.NewSession(client, concurrency.WithTTL(config.SessionTTL))
	if err != nil {
		cancel()
		return nil, fmt.Errorf("创建etcd会话失败: %v", err)
	}

	edl := &EnhancedDistributedLock{
		client:        client,
		session:       session,
		locks:         make(map[string]*LockInfo),
		lockStats:     make(map[string]*LockStats),
		waitingQueues: make(map[string][]*LockRequest),
		deadlockDetector: &DeadlockDetector{
			WaitGraph:   make(map[string][]string),
			LockOwners:  make(map[string]string),
			detectCycle: true,
		},
		eventListeners:  make(map[string]chan *LockEvent),
		ctx:             ctx,
		cancel:          cancel,
		basePrefix:      config.BasePrefix,
		defaultTTL:      config.DefaultTTL,
		monitorEnabled:  config.MonitorEnabled,
		auditEnabled:    config.AuditEnabled,
		cleanupInterval: config.CleanupInterval,
	}

	// 启动监控和清理
	if config.MonitorEnabled {
		go edl.startMonitoring()
	}
	go edl.startCleanup()
	go edl.StartDeadlockDetection()

	return edl, nil
}

// AcquireLock 获取锁
func (edl *EnhancedDistributedLock) AcquireLock(ctx context.Context, request *LockRequest) (string, error) {
	log.Info("Acquiring lock: %s by %s", request.LockKey, request.Owner)

	// 验证请求
	if err := edl.validateLockRequest(request); err != nil {
		return "", fmt.Errorf("无效的锁请求: %v", err)
	}

	// 生成锁ID
	lockID := edl.generateLockID(request)

	// 检查重入锁
	if request.LockType == ReentrantLock {
		if existingLockID := edl.checkReentrantLock(request); existingLockID != "" {
			return edl.handleReentrantAcquire(existingLockID, request)
		}
	}

	// 检查读写锁
	if request.LockType == ReadWriteLock {
		return edl.acquireReadWriteLock(ctx, request, lockID)
	}

	// 普通互斥锁获取
	return edl.acquireMutexLock(ctx, request, lockID)
}

// ReleaseLock 释放锁
func (edl *EnhancedDistributedLock) ReleaseLock(ctx context.Context, lockID string) error {
	log.Info("Releasing lock: %s", lockID)

	edl.mu.Lock()
	lockInfo, exists := edl.locks[lockID]
	if !exists {
		edl.mu.Unlock()
		return fmt.Errorf("锁不存在: %s", lockID)
	}

	// 处理重入锁
	if lockInfo.LockType == ReentrantLock && lockInfo.Reentrant > 1 {
		lockInfo.Reentrant--
		edl.mu.Unlock()
		log.Debug("Reentrant lock count decreased to %d for %s", lockInfo.Reentrant, lockID)
		return nil
	}

	// 处理读写锁
	if lockInfo.LockType == ReadWriteLock {
		edl.mu.Unlock()
		return edl.releaseReadWriteLock(ctx, lockID, lockInfo)
	}

	// 删除锁信息
	delete(edl.locks, lockID)
	lockKey := lockInfo.LockKey
	edl.mu.Unlock()

	// 从etcd删除锁
	if err := edl.deleteLockFromEtcd(ctx, lockInfo); err != nil {
		log.Error("从etcd删除锁失败: %v", err)
	}

	// 更新统计信息
	edl.updateLockStats(lockKey, "release", time.Since(lockInfo.AcquiredAt))

	// 更新死锁检测器
	edl.deadlockDetector.removeLockOwner(lockKey)

	// 处理等待队列
	go edl.processWaitingQueue(lockKey)

	// 发送事件
	edl.sendLockEvent(&LockEvent{
		Type:      "release",
		LockKey:   lockKey,
		Owner:     lockInfo.Owner,
		OwnerID:   lockInfo.OwnerID,
		Timestamp: time.Now(),
		Duration:  time.Since(lockInfo.AcquiredAt),
	})

	log.Info("Lock released successfully: %s", lockID)
	return nil
}

// TryLock 尝试获取锁（非阻塞）
func (edl *EnhancedDistributedLock) TryLock(ctx context.Context, request *LockRequest) (string, bool, error) {
	request.Blocking = false
	request.Timeout = 0
	request.MaxRetries = 0

	lockID, err := edl.AcquireLock(ctx, request)
	if err != nil {
		if strings.Contains(err.Error(), "锁已被占用") {
			return "", false, nil
		}
		return "", false, err
	}

	return lockID, true, nil
}

// IsLocked 检查锁是否被占用
func (edl *EnhancedDistributedLock) IsLocked(ctx context.Context, lockKey string) (bool, *LockInfo, error) {
	log.Debug("Checking if lock is held: %s", lockKey)

	// 从etcd查询锁信息
	lockPath := edl.buildLockPath(lockKey)
	resp, err := edl.client.Get(ctx, lockPath, clientv3.WithPrefix())
	if err != nil {
		return false, nil, fmt.Errorf("查询锁状态失败: %v", err)
	}

	if len(resp.Kvs) == 0 {
		return false, nil, nil
	}

	// 解析锁信息
	var lockInfo LockInfo
	if err := json.Unmarshal(resp.Kvs[0].Value, &lockInfo); err != nil {
		return false, nil, fmt.Errorf("解析锁信息失败: %v", err)
	}

	// 检查锁是否过期
	if time.Now().After(lockInfo.ExpiresAt) {
		// 锁已过期，清理
		go edl.cleanupExpiredLock(ctx, &lockInfo)
		return false, nil, nil
	}

	return true, &lockInfo, nil
}

// GetLockInfo 获取锁信息
func (edl *EnhancedDistributedLock) GetLockInfo(lockID string) (*LockInfo, error) {
	edl.mu.RLock()
	defer edl.mu.RUnlock()

	lockInfo, exists := edl.locks[lockID]
	if !exists {
		return nil, fmt.Errorf("锁不存在: %s", lockID)
	}

	return lockInfo, nil
}

// GetLockStats 获取锁统计信息
func (edl *EnhancedDistributedLock) GetLockStats(lockKey string) (*LockStats, error) {
	edl.mu.RLock()
	defer edl.mu.RUnlock()

	stats, exists := edl.lockStats[lockKey]
	if !exists {
		return nil, fmt.Errorf("锁统计不存在: %s", lockKey)
	}

	return stats, nil
}

// ListLocks 列出所有锁
func (edl *EnhancedDistributedLock) ListLocks() map[string]*LockInfo {
	edl.mu.RLock()
	defer edl.mu.RUnlock()

	locks := make(map[string]*LockInfo)
	for lockID, lockInfo := range edl.locks {
		locks[lockID] = lockInfo
	}

	return locks
}

// AddLockEventListener 添加锁事件监听器
func (edl *EnhancedDistributedLock) AddLockEventListener(listenerID string) chan *LockEvent {
	edl.listenersMu.Lock()
	defer edl.listenersMu.Unlock()

	ch := make(chan *LockEvent, 100)
	edl.eventListeners[listenerID] = ch

	log.Info("Added lock event listener: %s", listenerID)
	return ch
}

// RemoveLockEventListener 移除锁事件监听器
func (edl *EnhancedDistributedLock) RemoveLockEventListener(listenerID string) {
	edl.listenersMu.Lock()
	defer edl.listenersMu.Unlock()

	if ch, exists := edl.eventListeners[listenerID]; exists {
		close(ch)
		delete(edl.eventListeners, listenerID)
		log.Info("Removed lock event listener: %s", listenerID)
	}
}

// 内部方法实现

// validateLockRequest 验证锁请求
func (edl *EnhancedDistributedLock) validateLockRequest(request *LockRequest) error {
	if request.LockKey == "" {
		return fmt.Errorf("锁键不能为空")
	}
	if request.Owner == "" {
		return fmt.Errorf("锁拥有者不能为空")
	}
	if request.OwnerID == "" {
		return fmt.Errorf("锁拥有者ID不能为空")
	}
	if request.TTL <= 0 {
		request.TTL = edl.defaultTTL
	}
	if request.Timeout < 0 {
		request.Timeout = 0
	}
	if request.RetryDelay <= 0 {
		request.RetryDelay = 100 * time.Millisecond
	}
	return nil
}

// generateLockID 生成锁ID
func (edl *EnhancedDistributedLock) generateLockID(request *LockRequest) string {
	return fmt.Sprintf("%s_%s_%d", request.LockKey, request.OwnerID, time.Now().UnixNano())
}

// checkReentrantLock 检查重入锁
func (edl *EnhancedDistributedLock) checkReentrantLock(request *LockRequest) string {
	edl.mu.RLock()
	defer edl.mu.RUnlock()

	for lockID, lockInfo := range edl.locks {
		if lockInfo.LockKey == request.LockKey &&
			lockInfo.OwnerID == request.OwnerID &&
			lockInfo.LockType == ReentrantLock {
			return lockID
		}
	}
	return ""
}

// handleReentrantAcquire 处理重入锁获取
func (edl *EnhancedDistributedLock) handleReentrantAcquire(lockID string, request *LockRequest) (string, error) {
	edl.mu.Lock()
	defer edl.mu.Unlock()

	lockInfo, exists := edl.locks[lockID]
	if !exists {
		return "", fmt.Errorf("重入锁不存在: %s", lockID)
	}

	lockInfo.Reentrant++
	log.Debug("Reentrant lock count increased to %d for %s", lockInfo.Reentrant, lockID)

	return lockID, nil
}

// acquireReadWriteLock 获取读写锁
func (edl *EnhancedDistributedLock) acquireReadWriteLock(ctx context.Context, request *LockRequest, lockID string) (string, error) {
	if request.ReadOnly {
		return edl.acquireReadLock(ctx, request, lockID)
	}
	return edl.acquireWriteLock(ctx, request, lockID)
}

// acquireReadLock 获取读锁
func (edl *EnhancedDistributedLock) acquireReadLock(ctx context.Context, request *LockRequest, lockID string) (string, error) {
	// 检查是否有写锁
	if edl.hasWriteLock(request.LockKey) {
		if !request.Blocking {
			return "", fmt.Errorf("锁已被写锁占用: %s", request.LockKey)
		}
		// 加入等待队列
		return edl.addToWaitingQueue(ctx, request, lockID)
	}

	// 获取读锁
	return edl.createReadLock(ctx, request, lockID)
}

// acquireWriteLock 获取写锁
func (edl *EnhancedDistributedLock) acquireWriteLock(ctx context.Context, request *LockRequest, lockID string) (string, error) {
	// 检查是否有任何锁
	if edl.hasAnyLock(request.LockKey) {
		if !request.Blocking {
			return "", fmt.Errorf("锁已被占用: %s", request.LockKey)
		}
		// 加入等待队列
		return edl.addToWaitingQueue(ctx, request, lockID)
	}

	// 获取写锁
	return edl.createWriteLock(ctx, request, lockID)
}

// acquireMutexLock 获取互斥锁
func (edl *EnhancedDistributedLock) acquireMutexLock(ctx context.Context, request *LockRequest, lockID string) (string, error) {
	// 使用etcd的分布式锁
	mutex := concurrency.NewMutex(edl.session, edl.buildLockPath(request.LockKey))

	// 设置超时上下文
	lockCtx := ctx
	if request.Timeout > 0 {
		var cancel context.CancelFunc
		lockCtx, cancel = context.WithTimeout(ctx, request.Timeout)
		defer cancel()
	}

	// 尝试获取锁
	start := time.Now()
	err := mutex.Lock(lockCtx)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return "", fmt.Errorf("获取锁超时: %s", request.LockKey)
		}
		return "", fmt.Errorf("获取锁失败: %v", err)
	}

	// 创建锁信息
	lockInfo := &LockInfo{
		LockID:     lockID,
		LockKey:    request.LockKey,
		LockType:   request.LockType,
		Owner:      request.Owner,
		OwnerID:    request.OwnerID,
		AcquiredAt: time.Now(),
		ExpiresAt:  time.Now().Add(time.Duration(request.TTL) * time.Second),
		TTL:        request.TTL,
		Reentrant:  1,
		Metadata:   request.Metadata,
	}

	// 存储锁信息
	edl.mu.Lock()
	edl.locks[lockID] = lockInfo
	edl.mu.Unlock()

	// 存储到etcd
	if err := edl.storeLockToEtcd(ctx, lockInfo); err != nil {
		log.Error("存储锁信息到etcd失败: %v", err)
	}

	// 更新统计信息
	edl.updateLockStats(request.LockKey, "acquire", time.Since(start))

	// 更新死锁检测器
	edl.deadlockDetector.AddLockOwner(request.LockKey, request.OwnerID)

	// 发送事件
	edl.sendLockEvent(&LockEvent{
		Type:      "acquire",
		LockKey:   request.LockKey,
		Owner:     request.Owner,
		OwnerID:   request.OwnerID,
		Timestamp: time.Now(),
		Duration:  time.Since(start),
	})

	log.Info("Mutex lock acquired successfully: %s", lockID)
	return lockID, nil
}

// hasWriteLock 检查是否有写锁
func (edl *EnhancedDistributedLock) hasWriteLock(lockKey string) bool {
	edl.mu.RLock()
	defer edl.mu.RUnlock()

	for _, lockInfo := range edl.locks {
		if lockInfo.LockKey == lockKey &&
			lockInfo.LockType == ReadWriteLock &&
			lockInfo.WriteCount > 0 {
			return true
		}
	}
	return false
}

// hasAnyLock 检查是否有任何锁
func (edl *EnhancedDistributedLock) hasAnyLock(lockKey string) bool {
	edl.mu.RLock()
	defer edl.mu.RUnlock()

	for _, lockInfo := range edl.locks {
		if lockInfo.LockKey == lockKey {
			return true
		}
	}
	return false
}

// createReadLock 创建读锁
func (edl *EnhancedDistributedLock) createReadLock(ctx context.Context, request *LockRequest, lockID string) (string, error) {
	lockInfo := &LockInfo{
		LockID:     lockID,
		LockKey:    request.LockKey,
		LockType:   ReadWriteLock,
		Owner:      request.Owner,
		OwnerID:    request.OwnerID,
		AcquiredAt: time.Now(),
		ExpiresAt:  time.Now().Add(time.Duration(request.TTL) * time.Second),
		TTL:        request.TTL,
		ReadCount:  1,
		WriteCount: 0,
		Metadata:   request.Metadata,
	}

	edl.mu.Lock()
	edl.locks[lockID] = lockInfo
	edl.mu.Unlock()

	log.Info("Read lock acquired: %s", lockID)
	return lockID, nil
}

// createWriteLock 创建写锁
func (edl *EnhancedDistributedLock) createWriteLock(ctx context.Context, request *LockRequest, lockID string) (string, error) {
	lockInfo := &LockInfo{
		LockID:     lockID,
		LockKey:    request.LockKey,
		LockType:   ReadWriteLock,
		Owner:      request.Owner,
		OwnerID:    request.OwnerID,
		AcquiredAt: time.Now(),
		ExpiresAt:  time.Now().Add(time.Duration(request.TTL) * time.Second),
		TTL:        request.TTL,
		ReadCount:  0,
		WriteCount: 1,
		Metadata:   request.Metadata,
	}

	edl.mu.Lock()
	edl.locks[lockID] = lockInfo
	edl.mu.Unlock()

	log.Info("Write lock acquired: %s", lockID)
	return lockID, nil
}

// addToWaitingQueue 添加到等待队列
func (edl *EnhancedDistributedLock) addToWaitingQueue(ctx context.Context, request *LockRequest, lockID string) (string, error) {
	edl.mu.Lock()
	if _, exists := edl.waitingQueues[request.LockKey]; !exists {
		edl.waitingQueues[request.LockKey] = make([]*LockRequest, 0)
	}
	edl.waitingQueues[request.LockKey] = append(edl.waitingQueues[request.LockKey], request)
	edl.mu.Unlock()

	// 更新死锁检测器
	edl.deadlockDetector.AddWaitingRelation(request.OwnerID, edl.getLockOwner(request.LockKey))

	// 检查死锁
	if edl.deadlockDetector.detectDeadlock() {
		return "", fmt.Errorf("检测到死锁: %s", request.LockKey)
	}

	// 等待锁释放
	return edl.waitForLock(ctx, request, lockID)
}

// waitForLock 等待锁释放
func (edl *EnhancedDistributedLock) waitForLock(ctx context.Context, request *LockRequest, lockID string) (string, error) {
	ticker := time.NewTicker(request.RetryDelay)
	defer ticker.Stop()

	start := time.Now()
	retries := 0

	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-ticker.C:
			// 检查超时
			if request.Timeout > 0 && time.Since(start) > request.Timeout {
				return "", fmt.Errorf("等待锁超时: %s", request.LockKey)
			}

			// 检查重试次数
			if request.MaxRetries > 0 && retries >= request.MaxRetries {
				return "", fmt.Errorf("达到最大重试次数: %s", request.LockKey)
			}

			// 尝试获取锁
			if !edl.hasConflictingLock(request) {
				// 从等待队列移除
				edl.removeFromWaitingQueue(request)
				// 获取锁
				return edl.acquireLockDirectly(ctx, request, lockID)
			}

			retries++
		}
	}
}

// hasConflictingLock 检查是否有冲突的锁
func (edl *EnhancedDistributedLock) hasConflictingLock(request *LockRequest) bool {
	if request.LockType == ReadWriteLock && request.ReadOnly {
		// 读锁只与写锁冲突
		return edl.hasWriteLock(request.LockKey)
	}
	// 写锁和互斥锁与任何锁冲突
	return edl.hasAnyLock(request.LockKey)
}

// removeFromWaitingQueue 从等待队列移除
func (edl *EnhancedDistributedLock) removeFromWaitingQueue(request *LockRequest) {
	edl.mu.Lock()
	defer edl.mu.Unlock()

	queue, exists := edl.waitingQueues[request.LockKey]
	if !exists {
		return
	}

	for i, req := range queue {
		if req.OwnerID == request.OwnerID {
			edl.waitingQueues[request.LockKey] = append(queue[:i], queue[i+1:]...)
			break
		}
	}
}

// acquireLockDirectly 直接获取锁
func (edl *EnhancedDistributedLock) acquireLockDirectly(ctx context.Context, request *LockRequest, lockID string) (string, error) {
	switch request.LockType {
	case ReadWriteLock:
		if request.ReadOnly {
			return edl.createReadLock(ctx, request, lockID)
		}
		return edl.createWriteLock(ctx, request, lockID)
	default:
		return edl.acquireMutexLock(ctx, request, lockID)
	}
}

// releaseReadWriteLock 释放读写锁
func (edl *EnhancedDistributedLock) releaseReadWriteLock(ctx context.Context, lockID string, lockInfo *LockInfo) error {
	edl.mu.Lock()
	if lockInfo.ReadCount > 0 {
		lockInfo.ReadCount--
		if lockInfo.ReadCount == 0 {
			delete(edl.locks, lockID)
		}
	} else if lockInfo.WriteCount > 0 {
		lockInfo.WriteCount--
		if lockInfo.WriteCount == 0 {
			delete(edl.locks, lockID)
		}
	}
	edl.mu.Unlock()

	return nil
}

// processWaitingQueue 处理等待队列
func (edl *EnhancedDistributedLock) processWaitingQueue(lockKey string) {
	edl.mu.RLock()
	queue, exists := edl.waitingQueues[lockKey]
	if !exists || len(queue) == 0 {
		edl.mu.RUnlock()
		return
	}
	edl.mu.RUnlock()

	// 通知等待的请求
	log.Debug("Processing waiting queue for lock: %s, queue length: %d", lockKey, len(queue))
}

// buildLockPath 构建锁路径
func (edl *EnhancedDistributedLock) buildLockPath(lockKey string) string {
	return path.Join(edl.basePrefix, lockKey)
}

// storeLockToEtcd 存储锁信息到etcd
func (edl *EnhancedDistributedLock) storeLockToEtcd(ctx context.Context, lockInfo *LockInfo) error {
	lockBytes, err := json.Marshal(lockInfo)
	if err != nil {
		return err
	}

	lockPath := path.Join(edl.buildLockPath(lockInfo.LockKey), "info")
	_, err = edl.client.Put(ctx, lockPath, string(lockBytes))
	return err
}

// deleteLockFromEtcd 从etcd删除锁信息
func (edl *EnhancedDistributedLock) deleteLockFromEtcd(ctx context.Context, lockInfo *LockInfo) error {
	lockPath := path.Join(edl.buildLockPath(lockInfo.LockKey), "info")
	_, err := edl.client.Delete(ctx, lockPath)
	return err
}

// updateLockStats 更新锁统计信息
func (edl *EnhancedDistributedLock) updateLockStats(lockKey, operation string, duration time.Duration) {
	edl.mu.Lock()
	defer edl.mu.Unlock()

	stats, exists := edl.lockStats[lockKey]
	if !exists {
		stats = &LockStats{
			LockKey: lockKey,
		}
		edl.lockStats[lockKey] = stats
	}

	switch operation {
	case "acquire":
		stats.TotalAcquires++
		stats.CurrentHolders++
	case "release":
		stats.TotalReleases++
		stats.CurrentHolders--

		// 更新平均持有时间
		if stats.TotalReleases > 0 {
			totalTime := stats.AverageHoldTime*time.Duration(stats.TotalReleases-1) + duration
			stats.AverageHoldTime = totalTime / time.Duration(stats.TotalReleases)
		} else {
			stats.AverageHoldTime = duration
		}

		// 更新最大持有时间
		if duration > stats.MaxHoldTime {
			stats.MaxHoldTime = duration
		}
	}

	stats.LastAccessed = time.Now()
}

// getLockOwner 获取锁拥有者
func (edl *EnhancedDistributedLock) getLockOwner(lockKey string) string {
	edl.mu.RLock()
	defer edl.mu.RUnlock()

	for _, lockInfo := range edl.locks {
		if lockInfo.LockKey == lockKey {
			return lockInfo.OwnerID
		}
	}
	return ""
}

// sendLockEvent 发送锁事件
func (edl *EnhancedDistributedLock) sendLockEvent(event *LockEvent) {
	edl.listenersMu.RLock()
	defer edl.listenersMu.RUnlock()

	for listenerID, ch := range edl.eventListeners {
		select {
		case ch <- event:
			log.Debug("Sent lock event to listener %s", listenerID)
		default:
			log.Warning("Failed to send lock event to listener %s (channel full)", listenerID)
		}
	}

	// 审计日志
	if edl.auditEnabled {
		go edl.auditLockEvent(event)
	}
}

// auditLockEvent 审计锁事件
func (edl *EnhancedDistributedLock) auditLockEvent(event *LockEvent) {
	log.Info("Lock audit: type=%s, key=%s, owner=%s, ownerID=%s, duration=%v, time=%s",
		event.Type, event.LockKey, event.Owner, event.OwnerID, event.Duration, event.Timestamp.Format(time.RFC3339))
}

// cleanupExpiredLock 清理过期锁
func (edl *EnhancedDistributedLock) cleanupExpiredLock(ctx context.Context, lockInfo *LockInfo) {
	log.Info("Cleaning up expired lock: %s", lockInfo.LockID)

	// 从内存删除
	edl.mu.Lock()
	delete(edl.locks, lockInfo.LockID)
	edl.mu.Unlock()

	// 从etcd删除
	if err := edl.deleteLockFromEtcd(ctx, lockInfo); err != nil {
		log.Error("删除过期锁失败: %v", err)
	}

	// 处理等待队列
	go edl.processWaitingQueue(lockInfo.LockKey)
}

// startMonitoring 启动监控
func (edl *EnhancedDistributedLock) startMonitoring() {
	log.Info("Starting lock monitoring")

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			edl.monitorLocks()
		case <-edl.ctx.Done():
			log.Info("Lock monitoring stopped")
			return
		}
	}
}

// monitorLocks 监控锁状态
func (edl *EnhancedDistributedLock) monitorLocks() {
	edl.mu.RLock()
	lockCount := len(edl.locks)
	statsCount := len(edl.lockStats)
	waitingCount := 0
	for _, queue := range edl.waitingQueues {
		waitingCount += len(queue)
	}
	edl.mu.RUnlock()

	log.Debug("Lock monitoring: active_locks=%d, stats_entries=%d, waiting_requests=%d",
		lockCount, statsCount, waitingCount)
}

// startCleanup 启动清理
func (edl *EnhancedDistributedLock) startCleanup() {
	log.Info("Starting lock cleanup")

	ticker := time.NewTicker(edl.cleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			edl.cleanupExpiredLocks()
		case <-edl.ctx.Done():
			log.Info("Lock cleanup stopped")
			return
		}
	}
}

// cleanupExpiredLocks 清理过期锁
func (edl *EnhancedDistributedLock) cleanupExpiredLocks() {
	now := time.Now()
	expiredLocks := make([]*LockInfo, 0)

	edl.mu.RLock()
	for _, lockInfo := range edl.locks {
		if now.After(lockInfo.ExpiresAt) {
			expiredLocks = append(expiredLocks, lockInfo)
		}
	}
	edl.mu.RUnlock()

	for _, lockInfo := range expiredLocks {
		go edl.cleanupExpiredLock(context.Background(), lockInfo)
	}

	if len(expiredLocks) > 0 {
		log.Info("Cleaned up %d expired locks", len(expiredLocks))
	}
}

// StartDeadlockDetection 启动死锁检测
func (edl *EnhancedDistributedLock) StartDeadlockDetection() {
	log.Info("Starting deadlock detection")

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if edl.deadlockDetector.detectDeadlock() {
				log.Warning("Deadlock detected!")
				edl.HandleDeadlock()
			}
		case <-edl.ctx.Done():
			log.Info("Deadlock detection stopped")
			return
		}
	}
}

// handleDeadlock 处理死锁
func (edl *EnhancedDistributedLock) HandleDeadlock() {
	log.Warning("Handling deadlock situation")

	// 发送死锁事件
	edl.sendLockEvent(&LockEvent{
		Type:      "deadlock",
		Timestamp: time.Now(),
		Error:     "Deadlock detected in lock system",
	})

	// 更新统计
	edl.mu.Lock()
	for _, stats := range edl.lockStats {
		stats.DeadlockCount++
	}
	edl.mu.Unlock()
}

// 死锁检测器方法

// addLockOwner 添加锁拥有者
func (dd *DeadlockDetector) AddLockOwner(lockKey, ownerID string) {
	dd.Mu.Lock()
	defer dd.Mu.Unlock()
	dd.LockOwners[lockKey] = ownerID
}

// removeLockOwner 移除锁拥有者
func (dd *DeadlockDetector) RemoveLockOwner(lockKey string) {
	dd.Mu.Lock()
	defer dd.Mu.Unlock()
	delete(dd.LockOwners, lockKey)
}

// addWaitingRelation 添加等待关系
func (dd *DeadlockDetector) AddWaitingRelation(waiterID, ownerID string) {
	if waiterID == ownerID || ownerID == "" {
		return
	}

	dd.Mu.Lock()
	defer dd.Mu.Unlock()

	if _, exists := dd.WaitGraph[waiterID]; !exists {
		dd.WaitGraph[waiterID] = make([]string, 0)
	}

	// 避免重复添加
	for _, existing := range dd.WaitGraph[waiterID] {
		if existing == ownerID {
			return
		}
	}

	dd.WaitGraph[waiterID] = append(dd.WaitGraph[waiterID], ownerID)
}

// detectDeadlock 检测死锁
func (dd *DeadlockDetector) detectDeadlock() bool {
	if !dd.detectCycle {
		return false
	}

	dd.Mu.RLock()
	defer dd.Mu.RUnlock()

	// 使用DFS检测环
	visited := make(map[string]bool)
	recStack := make(map[string]bool)

	for node := range dd.WaitGraph {
		if !visited[node] {
			if dd.dfsHasCycle(node, visited, recStack) {
				return true
			}
		}
	}

	return false
}

// dfsHasCycle DFS检测环
func (dd *DeadlockDetector) dfsHasCycle(node string, visited, recStack map[string]bool) bool {
	visited[node] = true
	recStack[node] = true

	for _, neighbor := range dd.WaitGraph[node] {
		if !visited[neighbor] {
			if dd.dfsHasCycle(neighbor, visited, recStack) {
				return true
			}
		} else if recStack[neighbor] {
			return true
		}
	}

	recStack[node] = false
	return false
}

// Close 关闭增强分布式锁
func (edl *EnhancedDistributedLock) Close() error {
	log.Info("Closing enhanced distributed lock")

	edl.cancel()

	// 释放所有锁
	edl.mu.Lock()
	lockIDs := make([]string, 0, len(edl.locks))
	for lockID := range edl.locks {
		lockIDs = append(lockIDs, lockID)
	}
	edl.mu.Unlock()

	for _, lockID := range lockIDs {
		if err := edl.ReleaseLock(context.Background(), lockID); err != nil {
			log.Error("释放锁失败 %s: %v", lockID, err)
		}
	}

	// 关闭会话
	if edl.session != nil {
		if err := edl.session.Close(); err != nil {
			log.Error("关闭etcd会话失败: %v", err)
		}
	}

	// 关闭事件监听器
	edl.listenersMu.Lock()
	for listenerID, ch := range edl.eventListeners {
		close(ch)
		delete(edl.eventListeners, listenerID)
	}
	edl.listenersMu.Unlock()

	return nil
}
