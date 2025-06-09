package bootstrap

import (
	"context"
	"encoding/json"
	"fmt"
	clientv3 "go.etcd.io/etcd/client/v3"
	etcdv3 "go.etcd.io/etcd/client/v3"
	"log"
	"sync"
	"time"
)

// ConfigManager 配置管理器
type ConfigManager struct {
	client       *etcdv3.Client
	configCache  sync.Map
	watchers     sync.Map
	mu           sync.RWMutex
	ctx          context.Context
	cancel       context.CancelFunc
	versionCache map[string]int64

	configPrefix string
	versionMap   sync.Map // map[string]int64 存储配置版本
}

// ConfigItem 配置项
type ConfigItem struct {
	Key         string      `json:"key"`
	Value       interface{} `json:"value"`
	Version     int64       `json:"version"`
	Environment string      `json:"environment"`
	UpdateTime  time.Time   `json:"updateTime"`
	Description string      `json:"description"`
}

// NewConfigManager 创建配置管理器
func NewConfigManager(client *etcdv3.Client, configPrefix string) *ConfigManager {
	ctx, cancel := context.WithCancel(context.Background())
	return &ConfigManager{
		client:       client,
		ctx:          ctx,
		cancel:       cancel,
		versionCache: make(map[string]int64),
		configPrefix: configPrefix,
	}
}

// SetConfig 设置配置（支持版本控制）
func (cm *ConfigManager) SetConfig(env, key string, value interface{}, description string) error {
	configKey := cm.buildConfigKey(env, key)

	// 获取当前版本
	currentVersion := cm.getCurrentVersion(configKey)
	newVersion := currentVersion + 1

	configItem := &ConfigItem{
		Key:         key,
		Value:       value,
		Version:     newVersion,
		Environment: env,
		UpdateTime:  time.Now(),
		Description: description,
	}

	data, err := json.Marshal(configItem)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	// 使用事务确保原子性
	txn := cm.client.Txn(cm.ctx)
	txnResp, err := txn.Then(
		etcdv3.OpPut(configKey, string(data)),
		etcdv3.OpPut(cm.buildVersionKey(configKey), fmt.Sprintf("%d", newVersion)),
	).Commit()

	if err != nil {
		return fmt.Errorf("failed to set config: %w", err)
	}

	if !txnResp.Succeeded {
		return fmt.Errorf("config transaction failed")
	}

	// 更新本地缓存
	cm.configCache.Store(configKey, configItem)
	cm.versionCache[configKey] = newVersion

	return nil
}

// GetConfig 获取配置
func (cm *ConfigManager) GetConfig(env, key string) (*ConfigItem, error) {
	configKey := cm.buildConfigKey(env, key)

	// 先检查缓存
	if cached, ok := cm.configCache.Load(configKey); ok {
		return cached.(*ConfigItem), nil
	}

	// 从etcd获取
	resp, err := cm.client.Get(cm.ctx, configKey)
	if err != nil {
		return nil, fmt.Errorf("failed to get config: %w", err)
	}

	if len(resp.Kvs) == 0 {
		return nil, fmt.Errorf("config not found: %s", configKey)
	}

	var configItem ConfigItem
	if err := json.Unmarshal(resp.Kvs[0].Value, &configItem); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	// 更新缓存
	cm.configCache.Store(configKey, &configItem)

	return &configItem, nil
}

// WatchConfig 监听配置变化（支持热更新）
func (cm *ConfigManager) WatchConfig(env, key string, callback func(*ConfigItem)) error {
	configKey := cm.buildConfigKey(env, key)

	// 检查是否已经在监听
	if _, exists := cm.watchers.Load(configKey); exists {
		return nil
	}

	watchChan := cm.client.Watch(cm.ctx, configKey)
	cm.watchers.Store(configKey, watchChan)

	go func() {
		defer cm.watchers.Delete(configKey)

		for {
			select {
			case <-cm.ctx.Done():
				return
			case watchResp, ok := <-watchChan:
				if !ok {
					// 重新建立监听
					time.Sleep(time.Second)
					watchChan = cm.client.Watch(cm.ctx, configKey)
					cm.watchers.Store(configKey, watchChan)
					continue
				}

				if watchResp.Err() != nil {
					continue
				}

				for _, event := range watchResp.Events {
					if event.Type == etcdv3.EventTypePut {
						var configItem ConfigItem
						if err := json.Unmarshal(event.Kv.Value, &configItem); err == nil {
							// 更新缓存
							cm.configCache.Store(configKey, &configItem)
							// 触发回调
							callback(&configItem)
						}
					}
				}
			}
		}
	}()

	return nil
}

// GetConfigHistory 获取配置历史版本
func (cm *ConfigManager) GetConfigHistory(env, key string, limit int) ([]*ConfigItem, error) {
	configKey := cm.buildConfigKey(env, key)
	historyPrefix := cm.buildHistoryKey(configKey)

	resp, err := cm.client.Get(cm.ctx, historyPrefix,
		etcdv3.WithPrefix(),
		etcdv3.WithSort(etcdv3.SortByKey, etcdv3.SortDescend),
		etcdv3.WithLimit(int64(limit)),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to get config history: %w", err)
	}

	var history []*ConfigItem
	for _, kv := range resp.Kvs {
		var configItem ConfigItem
		if err := json.Unmarshal(kv.Value, &configItem); err == nil {
			history = append(history, &configItem)
		}
	}

	return history, nil
}

// buildConfigKey 构建配置键
func (cm *ConfigManager) buildConfigKey(env, key string) string {
	return fmt.Sprintf("/config/%s/%s", env, key)
}

// buildVersionKey 构建版本键
func (cm *ConfigManager) buildVersionKey(configKey string) string {
	return fmt.Sprintf("%s/version", configKey)
}

// buildHistoryKey 构建历史键
func (cm *ConfigManager) buildHistoryKey(configKey string) string {
	return fmt.Sprintf("%s/history/", configKey)
}

// getCurrentVersion 获取当前版本
func (cm *ConfigManager) getCurrentVersion(configKey string) int64 {
	if version, exists := cm.versionCache[configKey]; exists {
		return version
	}

	versionKey := cm.buildVersionKey(configKey)
	resp, err := cm.client.Get(cm.ctx, versionKey)
	if err != nil || len(resp.Kvs) == 0 {
		return 0
	}

	var version int64
	fmt.Sscanf(string(resp.Kvs[0].Value), "%d", &version)
	cm.versionCache[configKey] = version
	return version
}

// Close 关闭配置管理器
func (cm *ConfigManager) Close() {
	cm.cancel()
}

// RollbackConfig 回滚配置到指定版本
func (cm *ConfigManager) RollbackConfig(ctx context.Context, key string, version int64) error {
	versionKey := cm.configPrefix + key + "/versions"
	versionDataKey := fmt.Sprintf("%s/%d", versionKey, version)

	// 获取指定版本的配置
	resp, err := cm.client.Get(ctx, versionDataKey)
	if err != nil {
		return fmt.Errorf("failed to get version data: %w", err)
	}

	if len(resp.Kvs) == 0 {
		return fmt.Errorf("version %d not found for config %s", version, key)
	}

	var versionInfo ConfigVersion
	if err := json.Unmarshal(resp.Kvs[0].Value, &versionInfo); err != nil {
		return fmt.Errorf("failed to unmarshal version info: %w", err)
	}

	// 回滚配置
	fullKey := cm.configPrefix + key
	_, err = cm.client.Put(ctx, fullKey, string(versionInfo.Data))
	if err != nil {
		return fmt.Errorf("failed to rollback config: %w", err)
	}

	// 更新当前版本标记
	_, err = cm.client.Put(ctx, versionKey+"/current", fmt.Sprintf("%d", version))
	if err != nil {
		return fmt.Errorf("failed to update current version: %w", err)
	}

	// 更新本地缓存
	cm.configCache.Store(key, versionInfo.Data)
	cm.versionMap.Store(key, version)

	return nil
}

// WatchConfigWithHotReload 监听配置变化并支持热更新
func (cm *ConfigManager) WatchConfigWithHotReload(ctx context.Context, key string, updateFunc func([]byte) error, validator func([]byte) error) {
	fullKey := cm.configPrefix + key

	// 如果已经在监听，先停止
	if cancel, exists := cm.watchers.Load(key); exists {
		cancel.(context.CancelFunc)()
	}

	watchCtx, cancel := context.WithCancel(ctx)
	cm.watchers.Store(key, cancel)

	go func() {
		defer cancel()

		watchChan := cm.client.Watch(watchCtx, fullKey)

		for watchResp := range watchChan {
			if watchResp.Err() != nil {
				log.Printf("Config watch error for %s: %v", key, watchResp.Err())
				continue
			}

			for _, event := range watchResp.Events {
				if event.Type == clientv3.EventTypePut {
					newData := event.Kv.Value

					// 验证新配置
					if validator != nil {
						if err := validator(newData); err != nil {
							log.Printf("Config validation failed for %s: %v", key, err)
							continue
						}
					}

					// 应用新配置
					if err := updateFunc(newData); err != nil {
						log.Printf("Failed to apply config update for %s: %v", key, err)
						continue
					}

					// 更新本地缓存
					cm.configCache.Store(key, newData)
					log.Printf("Config hot-reloaded successfully for %s", key)
				}
			}
		}
	}()
}
