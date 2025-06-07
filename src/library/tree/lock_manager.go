package tree

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

type LockType int

const (
	LockShared LockType = iota
	LockExclusive
)

// LockRequest 锁请求
type LockRequest struct {
	txID    uint64
	key     Key
	lockTy  LockType
	granted bool
	waiters chan struct{}
}

// LockManager 锁管理器(实现2PL)
type LockManager struct {
	locks          map[Key]*LockRequest
	txLocks        map[uint64][]Key // 事务持有的锁
	mu             sync.Mutex
	waitGraph      map[uint64]map[uint64]struct{}   // 等待图(用于死锁检测)
	predicateLocks map[KeyRange]map[uint64]LockType // 谓词锁存储 key: key range, value: map of txID to lock mode
	predicateMu    sync.RWMutex                     // 保护predicateLocks
}

func NewLockManager() *LockManager {
	return &LockManager{
		locks:     make(map[Key]*LockRequest),
		txLocks:   make(map[uint64][]Key),
		waitGraph: make(map[uint64]map[uint64]struct{}),
	}
}

// KeyRange 辅助类型和函数
type KeyRange struct {
	Start Key
	End   Key
}

func rangesOverlap(a, b KeyRange) bool {
	return compareKeys(a.Start, b.End) <= 0 && compareKeys(a.End, b.Start) >= 0
}

func isCompatible(requested, existing LockType) bool {
	if requested == LockShared || existing == LockShared {
		return true
	}
	return false
}

// AcquirePredicateLock 实现谓词锁获取逻辑
func (lm *LockManager) AcquirePredicateLock(txID uint64, startKey Key, endKey Key, mode LockType, timeout time.Duration) error {
	keyRange := KeyRange{Start: startKey, End: endKey}

	// 1. 检查锁冲突
	lm.predicateMu.Lock()
	defer lm.predicateMu.Unlock()

	// 2. 检查当前事务是否已经持有相同范围的锁
	if holders, exists := lm.predicateLocks[keyRange]; exists {
		if currentMode, held := holders[txID]; held {
			if currentMode == mode {
				return nil // 已持有相同模式的锁
			}
			// 锁升级逻辑
			return lm.upgradePredicateLock(txID, keyRange, currentMode, mode)
		}
	}

	// 3. 检查范围重叠的现有锁
	for existingRange, holders := range lm.predicateLocks {
		if rangesOverlap(keyRange, existingRange) {
			for holderID, holderMode := range holders {
				if holderID != txID && !isCompatible(mode, holderMode) {
					return fmt.Errorf("predicate lock conflict on range %v", existingRange)
				}
			}
		}
	}

	// 4. 获取新锁
	if lm.predicateLocks == nil {
		lm.predicateLocks = make(map[KeyRange]map[uint64]LockType)
	}
	if _, exists := lm.predicateLocks[keyRange]; !exists {
		lm.predicateLocks[keyRange] = make(map[uint64]LockType)
	}
	lm.predicateLocks[keyRange][txID] = mode
	return nil
}

func (lm *LockManager) upgradePredicateLock(txID uint64, keyRange KeyRange, currentMode LockType, newMode LockType) error {
	// 1. 检查锁升级是否允许（仅允许Shared -> Exclusive）
	if currentMode == LockShared && newMode == LockExclusive {
		// 2. 检查是否有其他事务持有冲突锁
		for existingRange, holders := range lm.predicateLocks {
			if rangesOverlap(keyRange, existingRange) {
				for holderID, holderMode := range holders {
					if holderID != txID && holderMode == LockExclusive {
						return fmt.Errorf("cannot upgrade lock due to existing exclusive lock on range %v", existingRange)
					}
				}
			}
		}

		// 3. 执行升级操作
		lm.predicateLocks[keyRange][txID] = newMode
		return nil
	}

	return fmt.Errorf("invalid lock upgrade from %v to %v", currentMode, newMode)
}

// Acquire 获取锁
func (lm *LockManager) Acquire(txID uint64, key Key, lockTy LockType) (bool, error) {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	// 检查死锁
	if lm.detectDeadlock(txID) {
		return false, errors.New("error dead lock")
	}

	lr, exists := lm.locks[key]
	if !exists {
		// 无竞争，直接获取锁
		lr = &LockRequest{
			txID:    txID,
			key:     key,
			lockTy:  lockTy,
			granted: true,
		}
		lm.locks[key] = lr
		lm.txLocks[txID] = append(lm.txLocks[txID], key)
		return true, nil
	}

	// 锁兼容性检查
	if lr.granted && lm.isCompatible(lr.txID, txID, lr.lockTy, lockTy) {
		// 兼容锁(如多个共享锁)
		if lockTy == LockShared && lr.lockTy == LockShared {
			lm.txLocks[txID] = append(lm.txLocks[txID], key)
			return true, nil
		}
	}

	// 不兼容，需要等待
	waitChan := make(chan struct{})
	lm.addToWaitGraph(txID, lr.txID)

	// 创建新的锁请求
	newLR := &LockRequest{
		txID:    txID,
		key:     key,
		lockTy:  lockTy,
		granted: false,
		waiters: waitChan,
	}
	lm.locks[key] = newLR

	lm.mu.Unlock()
	<-waitChan // 等待锁释放
	lm.mu.Lock()

	// 被唤醒后检查是否获得锁
	if !newLR.granted {
		return false, errors.New("error lock timeout")
	}
	return true, nil
}

func (lm *LockManager) addToWaitGraph(requestingTxID, holdingTxID uint64) {
	if _, exists := lm.waitGraph[requestingTxID]; !exists {
		lm.waitGraph[requestingTxID] = make(map[uint64]struct{})
	}
	lm.waitGraph[requestingTxID][holdingTxID] = struct{}{}
}

func (lm *LockManager) isCompatible(existingTxID, requestingTxID uint64, existingLockType, requestingLockType LockType) bool {
	// 如果是同一个事务，请求总是兼容的
	if existingTxID == requestingTxID {
		return true
	}

	// 共享锁之间是兼容的
	if existingLockType == LockShared && requestingLockType == LockShared {
		return true
	}

	// 其他情况不兼容
	return false
}

func (lm *LockManager) detectDeadlock(txID uint64) bool {
	visited := make(map[uint64]bool)
	var visit func(uint64) bool

	visit = func(id uint64) bool {
		if visited[id] {
			return true // 发现环，存在死锁
		}
		visited[id] = true

		if waitFor, exists := lm.waitGraph[id]; exists {
			for otherTxID := range waitFor {
				if visit(otherTxID) {
					return true
				}
			}
		}

		visited[id] = false
		return false
	}

	return visit(txID)
}

// Release 释放锁
func (lm *LockManager) Release(txID uint64, key Key) {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	if lr, exists := lm.locks[key]; exists && lr.txID == txID {
		delete(lm.locks, key)
		// 从事务锁集合中移除
		if keys, ok := lm.txLocks[txID]; ok {
			for i, k := range keys {
				if k == key {
					lm.txLocks[txID] = append(keys[:i], keys[i+1:]...)
					break
				}
			}
		}

		// 唤醒等待者
		if lr.waiters != nil {
			close(lr.waiters)
		}
	}
}

// ReleaseAll 在事务提交时添加谓词锁清理逻辑
func (lm *LockManager) ReleaseAll(txID uint64) {
	// 清理谓词锁
	lm.predicateMu.Lock()
	for r, holders := range lm.predicateLocks {
		delete(holders, txID)
		if len(holders) == 0 {
			delete(lm.predicateLocks, r)
		}
	}
	lm.predicateMu.Unlock()
}
