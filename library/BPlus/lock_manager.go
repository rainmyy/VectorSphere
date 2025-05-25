package bplus

import "sync"

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
	locks     map[Key]*LockRequest
	txLocks   map[uint64][]Key // 事务持有的锁
	mu        sync.Mutex
	waitGraph map[uint64]map[uint64]struct{} // 等待图(用于死锁检测)
}

func NewLockManager() *LockManager {
	return &LockManager{
		locks:     make(map[Key]*LockRequest),
		txLocks:   make(map[uint64][]Key),
		waitGraph: make(map[uint64]map[uint64]struct{}),
	}
}

// Acquire 获取锁
func (lm *LockManager) Acquire(txID uint64, key Key, lockTy LockType) (bool, error) {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	// 检查死锁
	if lm.detectDeadlock(txID) {
		return false, ErrDeadlock
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
		return false, ErrLockTimeout
	}
	return true, nil
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
