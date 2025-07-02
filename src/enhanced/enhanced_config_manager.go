package enhanced

import (
	"VectorSphere/src/library/logger"
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
)

// ConfigVersion 配置版本信息
type ConfigVersion struct {
	Version     string            `json:"version"`
	Timestamp   time.Time         `json:"timestamp"`
	Author      string            `json:"author"`
	Description string            `json:"description"`
	Checksum    string            `json:"checksum"`
	Tags        []string          `json:"tags"`
	Metadata    map[string]string `json:"metadata"`
}

// ConfigItem 配置项
type ConfigItem struct {
	Key         string            `json:"key"`
	Value       interface{}       `json:"value"`
	Type        string            `json:"type"`        // "string", "int", "bool", "json", "encrypted"
	Environment string            `json:"environment"` // "dev", "test", "prod", "*"
	Namespace   string            `json:"namespace"`
	Version     string            `json:"version"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
	Tags        []string          `json:"tags"`
	Metadata    map[string]string `json:"metadata"`
	Encrypted   bool              `json:"encrypted"`
	Sensitive   bool              `json:"sensitive"`
}

// ConfigGroup 配置组
type ConfigGroup struct {
	Name        string                 `json:"name"`
	Namespace   string                 `json:"namespace"`
	Environment string                 `json:"environment"`
	Items       map[string]*ConfigItem `json:"items"`
	Version     *ConfigVersion         `json:"version"`
	Parent      string                 `json:"parent,omitempty"`   // 继承的父配置组
	Inherits    []string               `json:"inherits,omitempty"` // 继承的配置组列表
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// ConfigChangeEvent 配置变更事件
type ConfigChangeEvent struct {
	Type        string      `json:"type"` // "create", "update", "delete", "rollback"
	Key         string      `json:"key"`
	OldValue    interface{} `json:"old_value,omitempty"`
	NewValue    interface{} `json:"new_value,omitempty"`
	Version     string      `json:"version"`
	Environment string      `json:"environment"`
	Namespace   string      `json:"namespace"`
	Timestamp   time.Time   `json:"timestamp"`
	Author      string      `json:"author"`
}

// ConfigWatcher 配置监听器
type ConfigWatcher struct {
	ID          string                         `json:"id"`
	Pattern     string                         `json:"pattern"`     // 监听的配置键模式
	Environment string                         `json:"environment"` // 监听的环境
	Namespace   string                         `json:"namespace"`   // 监听的命名空间
	Callback    func(*ConfigChangeEvent) error `json:"-"`
	Filter      func(*ConfigChangeEvent) bool  `json:"-"`
	CreatedAt   time.Time                      `json:"created_at"`
	Active      bool                           `json:"active"`
}

// EnhancedConfigManager 增强的配置管理器
type EnhancedConfigManager struct {
	client          *clientv3.Client
	encryptionKey   []byte
	gcm             cipher.AEAD
	configCache     map[string]*ConfigItem      // key -> config item
	groupCache      map[string]*ConfigGroup     // groupKey -> config group
	versionHistory  map[string][]*ConfigVersion // key -> versions
	watchers        map[string]*ConfigWatcher
	changeListeners map[string]chan *ConfigChangeEvent
	mu              sync.RWMutex
	watchersMu      sync.RWMutex
	ctx             context.Context
	cancel          context.CancelFunc
	basePrefix      string
	currentEnv      string
	currentNS       string
	hotReload       bool
	backupEnabled   bool
	auditEnabled    bool

	configPath    string
	lastModTime   time.Time
	mutex         sync.RWMutex
	watchersMutex sync.RWMutex
}

// EnhancedConfigManagerConfig 配置管理器配置
type EnhancedConfigManagerConfig struct {
	EncryptionKey string `json:"encryption_key"`
	BasePrefix    string `json:"base_prefix"`
	CurrentEnv    string `json:"current_env"`
	CurrentNS     string `json:"current_namespace"`
	HotReload     bool   `json:"hot_reload"`
	BackupEnabled bool   `json:"backup_enabled"`
	AuditEnabled  bool   `json:"audit_enabled"`
}

// NewEnhancedConfigManager 创建增强的配置管理器
func NewEnhancedConfigManager(client *clientv3.Client, config *EnhancedConfigManagerConfig) (*EnhancedConfigManager, error) {
	ctx, cancel := context.WithCancel(context.Background())

	if config == nil {
		config = &EnhancedConfigManagerConfig{
			BasePrefix:    "/vector_sphere/config",
			CurrentEnv:    "dev",
			CurrentNS:     "default",
			HotReload:     true,
			BackupEnabled: true,
			AuditEnabled:  true,
		}
	}

	ecm := &EnhancedConfigManager{
		client:          client,
		configCache:     make(map[string]*ConfigItem),
		groupCache:      make(map[string]*ConfigGroup),
		versionHistory:  make(map[string][]*ConfigVersion),
		watchers:        make(map[string]*ConfigWatcher),
		changeListeners: make(map[string]chan *ConfigChangeEvent),
		ctx:             ctx,
		cancel:          cancel,
		basePrefix:      config.BasePrefix,
		currentEnv:      config.CurrentEnv,
		currentNS:       config.CurrentNS,
		hotReload:       config.HotReload,
		backupEnabled:   config.BackupEnabled,
		auditEnabled:    config.AuditEnabled,
	}

	// 初始化加密
	if config.EncryptionKey != "" {
		if err := ecm.initEncryption(config.EncryptionKey); err != nil {
			return nil, fmt.Errorf("初始化加密失败: %v", err)
		}
	}

	// 启动热重载监听
	if config.HotReload {
		go ecm.startHotReload()
	}

	return ecm, nil
}

// SetConfig 设置配置项
func (ecm *EnhancedConfigManager) SetConfig(ctx context.Context, key string, value interface{}, options ...ConfigOption) error {
	logger.Info("Setting config: %s", key)

	// 应用选项
	item := &ConfigItem{
		Key:         key,
		Value:       value,
		Type:        ecm.inferType(value),
		Environment: ecm.currentEnv,
		Namespace:   ecm.currentNS,
		Version:     ecm.generateVersion(),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		Tags:        []string{},
		Metadata:    make(map[string]string),
	}

	for _, option := range options {
		option(item)
	}

	// 获取旧值用于审计
	oldItem := ecm.getFromCache(ecm.buildConfigKey(key, item.Environment, item.Namespace))

	// 加密敏感数据
	if item.Sensitive && ecm.gcm != nil {
		encryptedValue, err := ecm.encryptValue(value)
		if err != nil {
			return fmt.Errorf("加密配置值失败: %v", err)
		}
		item.Value = encryptedValue
		item.Encrypted = true
		item.Type = "encrypted"
	}

	// 序列化配置项
	itemBytes, err := json.Marshal(item)
	if err != nil {
		return fmt.Errorf("序列化配置项失败: %v", err)
	}

	// 存储到etcd
	configKey := ecm.buildConfigKey(key, item.Environment, item.Namespace)
	_, err = ecm.client.Put(ctx, configKey, string(itemBytes))
	if err != nil {
		return fmt.Errorf("存储配置失败: %v", err)
	}

	// 更新缓存
	ecm.updateCache(configKey, item)

	// 保存版本历史
	ecm.saveVersionHistory(key, item)

	// 创建备份
	if ecm.backupEnabled {
		go ecm.createBackup(ctx, key, item)
	}

	// 发送变更事件
	var oldValue interface{}
	if oldItem != nil {
		oldValue = oldItem.Value
	}

	event := &ConfigChangeEvent{
		Type:        "update",
		Key:         key,
		OldValue:    oldValue,
		NewValue:    value,
		Version:     item.Version,
		Environment: item.Environment,
		Namespace:   item.Namespace,
		Timestamp:   time.Now(),
		Author:      "system",
	}

	ecm.notifyWatchers(event)

	// 审计日志
	if ecm.auditEnabled {
		go ecm.auditLog(event)
	}

	logger.Info("Config set successfully: %s", key)
	return nil
}

// GetConfig 获取配置项
func (ecm *EnhancedConfigManager) GetConfig(ctx context.Context, key string, options ...GetConfigOption) (interface{}, error) {
	logger.Debug("Getting config: %s", key)

	// 应用获取选项
	getOpts := &GetConfigOptions{
		Environment: ecm.currentEnv,
		Namespace:   ecm.currentNS,
		UseCache:    true,
	}

	for _, option := range options {
		option(getOpts)
	}

	configKey := ecm.buildConfigKey(key, getOpts.Environment, getOpts.Namespace)

	// 尝试从缓存获取
	if getOpts.UseCache {
		if item := ecm.getFromCache(configKey); item != nil {
			return ecm.decryptIfNeeded(item)
		}
	}

	// 从etcd获取
	resp, err := ecm.client.Get(ctx, configKey)
	if err != nil {
		return nil, fmt.Errorf("获取配置失败: %v", err)
	}

	if len(resp.Kvs) == 0 {
		// 尝试继承配置
		if inheritedValue, err := ecm.getInheritedConfig(ctx, key, getOpts); err == nil {
			return inheritedValue, nil
		}
		return nil, fmt.Errorf("配置不存在: %s", key)
	}

	// 反序列化配置项
	var item ConfigItem
	if err := json.Unmarshal(resp.Kvs[0].Value, &item); err != nil {
		return nil, fmt.Errorf("反序列化配置项失败: %v", err)
	}

	// 更新缓存
	ecm.updateCache(configKey, &item)

	// 解密并返回值
	return ecm.decryptIfNeeded(&item)
}

// DeleteConfig 删除配置项
func (ecm *EnhancedConfigManager) DeleteConfig(ctx context.Context, key string, options ...ConfigOption) error {
	logger.Info("Deleting config: %s", key)

	// 构建配置键
	configKey := ecm.buildConfigKey(key, ecm.currentEnv, ecm.currentNS)

	// 获取旧值用于审计
	oldItem := ecm.getFromCache(configKey)

	// 从etcd删除
	_, err := ecm.client.Delete(ctx, configKey)
	if err != nil {
		return fmt.Errorf("删除配置失败: %v", err)
	}

	// 从缓存删除
	ecm.removeFromCache(configKey)

	// 发送变更事件
	var oldValue interface{}
	if oldItem != nil {
		oldValue = oldItem.Value
	}

	event := &ConfigChangeEvent{
		Type:        "delete",
		Key:         key,
		OldValue:    oldValue,
		Environment: ecm.currentEnv,
		Namespace:   ecm.currentNS,
		Timestamp:   time.Now(),
		Author:      "system",
	}

	ecm.notifyWatchers(event)

	// 审计日志
	if ecm.auditEnabled {
		go ecm.auditLog(event)
	}

	logger.Info("Config deleted successfully: %s", key)
	return nil
}

// SetConfigGroup 设置配置组
func (ecm *EnhancedConfigManager) SetConfigGroup(ctx context.Context, group *ConfigGroup) error {
	logger.Info("Setting config group: %s", group.Name)

	// 处理继承
	if err := ecm.processInheritance(ctx, group); err != nil {
		return fmt.Errorf("处理配置继承失败: %v", err)
	}

	// 生成版本
	if group.Version == nil {
		group.Version = &ConfigVersion{
			Version:     ecm.generateVersion(),
			Timestamp:   time.Now(),
			Author:      "system",
			Description: fmt.Sprintf("Config group %s update", group.Name),
		}
	}

	// 计算校验和
	group.Version.Checksum = ecm.calculateGroupChecksum(group)
	group.UpdatedAt = time.Now()

	// 序列化配置组
	groupBytes, err := json.Marshal(group)
	if err != nil {
		return fmt.Errorf("序列化配置组失败: %v", err)
	}

	// 存储到etcd
	groupKey := ecm.buildGroupKey(group.Name, group.Environment, group.Namespace)
	_, err = ecm.client.Put(ctx, groupKey, string(groupBytes))
	if err != nil {
		return fmt.Errorf("存储配置组失败: %v", err)
	}

	// 更新缓存
	ecm.updateGroupCache(groupKey, group)

	// 存储各个配置项
	for _, item := range group.Items {
		item.Environment = group.Environment
		item.Namespace = group.Namespace
		item.Version = group.Version.Version
		if err := ecm.SetConfig(ctx, item.Key, item.Value, WithEnvironment(item.Environment), WithNamespace(item.Namespace)); err != nil {
			logger.Warning("设置配置项失败 %s: %v", item.Key, err)
		}
	}

	logger.Info("Config group set successfully: %s", group.Name)
	return nil
}

// GetConfigGroup 获取配置组
func (ecm *EnhancedConfigManager) GetConfigGroup(ctx context.Context, name, environment, namespace string) (*ConfigGroup, error) {
	logger.Debug("Getting config group: %s", name)

	groupKey := ecm.buildGroupKey(name, environment, namespace)

	// 尝试从缓存获取
	if group := ecm.getGroupFromCache(groupKey); group != nil {
		return group, nil
	}

	// 从etcd获取
	resp, err := ecm.client.Get(ctx, groupKey)
	if err != nil {
		return nil, fmt.Errorf("获取配置组失败: %v", err)
	}

	if len(resp.Kvs) == 0 {
		return nil, fmt.Errorf("配置组不存在: %s", name)
	}

	// 反序列化配置组
	var group ConfigGroup
	if err := json.Unmarshal(resp.Kvs[0].Value, &group); err != nil {
		return nil, fmt.Errorf("反序列化配置组失败: %v", err)
	}

	// 更新缓存
	ecm.updateGroupCache(groupKey, &group)

	return &group, nil
}

// RollbackConfig 回滚配置
func (ecm *EnhancedConfigManager) RollbackConfig(ctx context.Context, key, targetVersion string) error {
	logger.Info("Rolling back config %s to version %s", key, targetVersion)

	// 获取版本历史
	ecm.mu.RLock()
	versions, exists := ecm.versionHistory[key]
	ecm.mu.RUnlock()

	if !exists || len(versions) == 0 {
		return fmt.Errorf("配置 %s 没有版本历史", key)
	}

	// 查找目标版本
	var targetVersionInfo *ConfigVersion
	for _, version := range versions {
		if version.Version == targetVersion {
			targetVersionInfo = version
			break
		}
	}

	if targetVersionInfo == nil {
		return fmt.Errorf("未找到目标版本 %s", targetVersion)
	}

	// 从备份恢复配置
	backupKey := ecm.buildBackupKey(key, targetVersion)
	resp, err := ecm.client.Get(ctx, backupKey)
	if err != nil {
		return fmt.Errorf("获取备份配置失败: %v", err)
	}

	if len(resp.Kvs) == 0 {
		return fmt.Errorf("备份配置不存在: %s", backupKey)
	}

	// 恢复配置
	configKey := ecm.buildConfigKey(key, ecm.currentEnv, ecm.currentNS)
	_, err = ecm.client.Put(ctx, configKey, string(resp.Kvs[0].Value))
	if err != nil {
		return fmt.Errorf("恢复配置失败: %v", err)
	}

	// 反序列化并更新缓存
	var item ConfigItem
	if err := json.Unmarshal(resp.Kvs[0].Value, &item); err != nil {
		return fmt.Errorf("反序列化配置项失败: %v", err)
	}

	ecm.updateCache(configKey, &item)

	// 发送回滚事件
	event := &ConfigChangeEvent{
		Type:        "rollback",
		Key:         key,
		NewValue:    item.Value,
		Version:     targetVersion,
		Environment: ecm.currentEnv,
		Namespace:   ecm.currentNS,
		Timestamp:   time.Now(),
		Author:      "system",
	}

	ecm.notifyWatchers(event)

	// 审计日志
	if ecm.auditEnabled {
		go ecm.auditLog(event)
	}

	logger.Info("Config rolled back successfully: %s to version %s", key, targetVersion)
	return nil
}

// WatchConfig 监听配置变更
func (ecm *EnhancedConfigManager) WatchConfig(pattern string, callback func(*ConfigChangeEvent) error, options ...WatchOption) (string, error) {
	watcherID := ecm.generateWatcherID()
	logger.Info("Creating config watcher: %s for pattern: %s", watcherID, pattern)

	watcher := &ConfigWatcher{
		ID:          watcherID,
		Pattern:     pattern,
		Environment: ecm.currentEnv,
		Namespace:   ecm.currentNS,
		Callback:    callback,
		CreatedAt:   time.Now(),
		Active:      true,
	}

	// 应用选项
	for _, option := range options {
		option(watcher)
	}

	// 注册监听器
	ecm.watchersMu.Lock()
	ecm.watchers[watcherID] = watcher
	ecm.watchersMu.Unlock()

	// 创建变更通知通道
	changeCh := make(chan *ConfigChangeEvent, 100)
	ecm.mu.Lock()
	ecm.changeListeners[watcherID] = changeCh
	ecm.mu.Unlock()

	// 启动监听协程
	go ecm.runWatcher(watcherID, changeCh)

	logger.Info("Config watcher created successfully: %s", watcherID)
	return watcherID, nil
}

// UnwatchConfig 取消配置监听
func (ecm *EnhancedConfigManager) UnwatchConfig(watcherID string) error {
	logger.Info("Removing config watcher: %s", watcherID)

	ecm.watchersMu.Lock()
	delete(ecm.watchers, watcherID)
	ecm.watchersMu.Unlock()

	ecm.mu.Lock()
	if ch, exists := ecm.changeListeners[watcherID]; exists {
		close(ch)
		delete(ecm.changeListeners, watcherID)
	}
	ecm.mu.Unlock()

	logger.Info("Config watcher removed successfully: %s", watcherID)
	return nil
}

// 辅助方法

// initEncryption 初始化加密
func (ecm *EnhancedConfigManager) initEncryption(key string) error {
	// 使用SHA256生成32字节密钥
	hash := sha256.Sum256([]byte(key))
	ecm.encryptionKey = hash[:]

	// 创建AES cipher
	block, err := aes.NewCipher(ecm.encryptionKey)
	if err != nil {
		return err
	}

	// 创建GCM
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return err
	}

	ecm.gcm = gcm
	return nil
}

// encryptValue 加密值
func (ecm *EnhancedConfigManager) encryptValue(value interface{}) (string, error) {
	if ecm.gcm == nil {
		return "", fmt.Errorf("加密未初始化")
	}

	// 序列化值
	valueBytes, err := json.Marshal(value)
	if err != nil {
		return "", err
	}

	// 生成随机nonce
	nonce := make([]byte, ecm.gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return "", err
	}

	// 加密
	ciphertext := ecm.gcm.Seal(nonce, nonce, valueBytes, nil)

	// Base64编码
	return base64.StdEncoding.EncodeToString(ciphertext), nil
}

// decryptValue 解密值
func (ecm *EnhancedConfigManager) decryptValue(encryptedValue string) (interface{}, error) {
	if ecm.gcm == nil {
		return nil, fmt.Errorf("加密未初始化")
	}

	// Base64解码
	ciphertext, err := base64.StdEncoding.DecodeString(encryptedValue)
	if err != nil {
		return nil, err
	}

	// 提取nonce
	nonceSize := ecm.gcm.NonceSize()
	if len(ciphertext) < nonceSize {
		return nil, fmt.Errorf("密文太短")
	}

	nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]

	// 解密
	plaintext, err := ecm.gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, err
	}

	// 反序列化
	var value interface{}
	err = json.Unmarshal(plaintext, &value)
	return value, err
}

// decryptIfNeeded 如果需要则解密
func (ecm *EnhancedConfigManager) decryptIfNeeded(item *ConfigItem) (interface{}, error) {
	if item.Encrypted {
		return ecm.decryptValue(item.Value.(string))
	}
	return item.Value, nil
}

// buildConfigKey 构建配置键
func (ecm *EnhancedConfigManager) buildConfigKey(key, environment, namespace string) string {
	return fmt.Sprintf("%s/%s/%s/%s", ecm.basePrefix, namespace, environment, key)
}

// buildGroupKey 构建配置组键
func (ecm *EnhancedConfigManager) buildGroupKey(name, environment, namespace string) string {
	return fmt.Sprintf("%s/groups/%s/%s/%s", ecm.basePrefix, namespace, environment, name)
}

// buildBackupKey 构建备份键
func (ecm *EnhancedConfigManager) buildBackupKey(key, version string) string {
	return fmt.Sprintf("%s/backups/%s/%s", ecm.basePrefix, key, version)
}

// inferType 推断类型
func (ecm *EnhancedConfigManager) inferType(value interface{}) string {
	switch value.(type) {
	case string:
		return "string"
	case int, int32, int64:
		return "int"
	case float32, float64:
		return "float"
	case bool:
		return "bool"
	default:
		return "json"
	}
}

// generateVersion 生成版本号
func (ecm *EnhancedConfigManager) generateVersion() string {
	return fmt.Sprintf("v%d", time.Now().Unix())
}

// generateWatcherID 生成监听器ID
func (ecm *EnhancedConfigManager) generateWatcherID() string {
	return fmt.Sprintf("watcher_%d", time.Now().UnixNano())
}

// calculateGroupChecksum 计算配置组校验和
func (ecm *EnhancedConfigManager) calculateGroupChecksum(group *ConfigGroup) string {
	hash := sha256.New()

	// 按键排序以确保一致性
	keys := make([]string, 0, len(group.Items))
	for key := range group.Items {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	for _, key := range keys {
		item := group.Items[key]
		hash.Write([]byte(fmt.Sprintf("%s:%v", key, item.Value)))
	}

	return fmt.Sprintf("%x", hash.Sum(nil))
}

// 缓存相关方法

// updateCache 更新缓存
func (ecm *EnhancedConfigManager) updateCache(key string, item *ConfigItem) {
	ecm.mu.Lock()
	defer ecm.mu.Unlock()
	ecm.configCache[key] = item
}

// getFromCache 从缓存获取
func (ecm *EnhancedConfigManager) getFromCache(key string) *ConfigItem {
	ecm.mu.RLock()
	defer ecm.mu.RUnlock()
	return ecm.configCache[key]
}

// removeFromCache 从缓存移除
func (ecm *EnhancedConfigManager) removeFromCache(key string) {
	ecm.mu.Lock()
	defer ecm.mu.Unlock()
	delete(ecm.configCache, key)
}

// updateGroupCache 更新配置组缓存
func (ecm *EnhancedConfigManager) updateGroupCache(key string, group *ConfigGroup) {
	ecm.mu.Lock()
	defer ecm.mu.Unlock()
	ecm.groupCache[key] = group
}

// getGroupFromCache 从缓存获取配置组
func (ecm *EnhancedConfigManager) getGroupFromCache(key string) *ConfigGroup {
	ecm.mu.RLock()
	defer ecm.mu.RUnlock()
	return ecm.groupCache[key]
}

// 配置选项

type ConfigOption func(*ConfigItem)
type GetConfigOption func(*GetConfigOptions)
type WatchOption func(*ConfigWatcher)

type GetConfigOptions struct {
	Environment string
	Namespace   string
	UseCache    bool
}

// WithEnvironment 设置环境
func WithEnvironment(env string) ConfigOption {
	return func(item *ConfigItem) {
		item.Environment = env
	}
}

// WithNamespace 设置命名空间
func WithNamespace(ns string) ConfigOption {
	return func(item *ConfigItem) {
		item.Namespace = ns
	}
}

// WithSensitive 设置为敏感数据
func WithSensitive() ConfigOption {
	return func(item *ConfigItem) {
		item.Sensitive = true
	}
}

// WithTags 设置标签
func WithTags(tags ...string) ConfigOption {
	return func(item *ConfigItem) {
		item.Tags = tags
	}
}

// WithMetadata 设置元数据
func WithMetadata(metadata map[string]string) ConfigOption {
	return func(item *ConfigItem) {
		item.Metadata = metadata
	}
}

// 其他辅助方法将在下一部分实现...

// saveVersionHistory 保存版本历史
func (ecm *EnhancedConfigManager) saveVersionHistory(key string, item *ConfigItem) {
	ecm.mu.Lock()
	defer ecm.mu.Unlock()

	version := &ConfigVersion{
		Version:     item.Version,
		Timestamp:   item.UpdatedAt,
		Author:      "system",
		Description: fmt.Sprintf("Update config %s", key),
		Checksum:    ecm.calculateItemChecksum(item),
	}

	if _, exists := ecm.versionHistory[key]; !exists {
		ecm.versionHistory[key] = make([]*ConfigVersion, 0)
	}

	ecm.versionHistory[key] = append(ecm.versionHistory[key], version)

	// 限制版本历史数量
	if len(ecm.versionHistory[key]) > 50 {
		ecm.versionHistory[key] = ecm.versionHistory[key][1:]
	}
}

// calculateItemChecksum 计算配置项校验和
func (ecm *EnhancedConfigManager) calculateItemChecksum(item *ConfigItem) string {
	hash := sha256.New()
	hash.Write([]byte(fmt.Sprintf("%s:%v:%s", item.Key, item.Value, item.Type)))
	return fmt.Sprintf("%x", hash.Sum(nil))
}

// createBackup 创建备份
func (ecm *EnhancedConfigManager) createBackup(ctx context.Context, key string, item *ConfigItem) {
	backupKey := ecm.buildBackupKey(key, item.Version)
	itemBytes, err := json.Marshal(item)
	if err != nil {
		logger.Error("序列化备份配置失败: %v", err)
		return
	}

	_, err = ecm.client.Put(ctx, backupKey, string(itemBytes))
	if err != nil {
		logger.Error("创建配置备份失败: %v", err)
	}
}

// processInheritance 处理配置继承
func (ecm *EnhancedConfigManager) processInheritance(ctx context.Context, group *ConfigGroup) error {
	if len(group.Inherits) == 0 && group.Parent == "" {
		return nil
	}

	// 处理父配置组
	if group.Parent != "" {
		parentGroup, err := ecm.GetConfigGroup(ctx, group.Parent, group.Environment, group.Namespace)
		if err != nil {
			return fmt.Errorf("获取父配置组失败: %v", err)
		}

		// 合并父配置组的配置项
		for key, item := range parentGroup.Items {
			if _, exists := group.Items[key]; !exists {
				group.Items[key] = item
			}
		}
	}

	// 处理继承的配置组列表
	for _, inheritName := range group.Inherits {
		inheritGroup, err := ecm.GetConfigGroup(ctx, inheritName, group.Environment, group.Namespace)
		if err != nil {
			logger.Warning("获取继承配置组失败 %s: %v", inheritName, err)
			continue
		}

		// 合并继承配置组的配置项
		for key, item := range inheritGroup.Items {
			if _, exists := group.Items[key]; !exists {
				group.Items[key] = item
			}
		}
	}

	return nil
}

// getInheritedConfig 获取继承的配置
func (ecm *EnhancedConfigManager) getInheritedConfig(ctx context.Context, key string, opts *GetConfigOptions) (interface{}, error) {
	// 尝试从全局环境获取
	if opts.Environment != "*" {
		globalKey := ecm.buildConfigKey(key, "*", opts.Namespace)
		if item := ecm.getFromCache(globalKey); item != nil {
			return ecm.decryptIfNeeded(item)
		}

		// 从etcd获取全局配置
		resp, err := ecm.client.Get(ctx, globalKey)
		if err == nil && len(resp.Kvs) > 0 {
			var item ConfigItem
			if err := json.Unmarshal(resp.Kvs[0].Value, &item); err == nil {
				ecm.updateCache(globalKey, &item)
				return ecm.decryptIfNeeded(&item)
			}
		}
	}

	return nil, fmt.Errorf("未找到继承配置")
}

// notifyWatchers 通知监听器
func (ecm *EnhancedConfigManager) notifyWatchers(event *ConfigChangeEvent) {
	ecm.mu.RLock()
	defer ecm.mu.RUnlock()

	for watcherID, ch := range ecm.changeListeners {
		select {
		case ch <- event:
			logger.Debug("Sent config change event to watcher %s", watcherID)
		default:
			logger.Warning("Failed to send config change event to watcher %s (channel full)", watcherID)
		}
	}
}

// runWatcher 运行监听器
func (ecm *EnhancedConfigManager) runWatcher(watcherID string, changeCh chan *ConfigChangeEvent) {
	for {
		select {
		case event := <-changeCh:
			ecm.watchersMu.RLock()
			watcher, exists := ecm.watchers[watcherID]
			ecm.watchersMu.RUnlock()

			if !exists || !watcher.Active {
				return
			}

			// 检查模式匹配
			if !ecm.matchPattern(event.Key, watcher.Pattern) {
				continue
			}

			// 检查环境和命名空间
			if watcher.Environment != "*" && event.Environment != watcher.Environment {
				continue
			}
			if watcher.Namespace != "*" && event.Namespace != watcher.Namespace {
				continue
			}

			// 应用过滤器
			if watcher.Filter != nil && !watcher.Filter(event) {
				continue
			}

			// 调用回调函数
			if err := watcher.Callback(event); err != nil {
				logger.Error("Config watcher callback failed for %s: %v", watcherID, err)
			}

		case <-ecm.ctx.Done():
			return
		}
	}
}

// matchPattern 匹配模式
func (ecm *EnhancedConfigManager) matchPattern(key, pattern string) bool {
	// 简单的通配符匹配实现
	if pattern == "*" {
		return true
	}
	if strings.Contains(pattern, "*") {
		// 支持前缀和后缀通配符
		if strings.HasPrefix(pattern, "*") {
			suffix := pattern[1:]
			return strings.HasSuffix(key, suffix)
		}
		if strings.HasSuffix(pattern, "*") {
			prefix := pattern[:len(pattern)-1]
			return strings.HasPrefix(key, prefix)
		}
	}
	return key == pattern
}

// startHotReload 启动热重载
func (ecm *EnhancedConfigManager) startHotReload() {
	logger.Info("Starting config hot reload watcher")

	watchCh := ecm.client.Watch(ecm.ctx, ecm.basePrefix, clientv3.WithPrefix())

	for {
		select {
		case watchResp := <-watchCh:
			for _, event := range watchResp.Events {
				ecm.handleEtcdEvent(event)
			}
		case <-ecm.ctx.Done():
			logger.Info("Config hot reload watcher stopped")
			return
		}
	}
}

// handleEtcdEvent 处理etcd事件
func (ecm *EnhancedConfigManager) handleEtcdEvent(event *clientv3.Event) {
	key := string(event.Kv.Key)

	// 解析配置键
	if !strings.HasPrefix(key, ecm.basePrefix) {
		return
	}

	// 跳过组和备份配置
	if strings.Contains(key, "/groups/") || strings.Contains(key, "/backups/") {
		return
	}

	var changeEvent *ConfigChangeEvent

	switch event.Type {
	case clientv3.EventTypePut:
		// 配置更新或创建
		var item ConfigItem
		if err := json.Unmarshal(event.Kv.Value, &item); err != nil {
			logger.Error("反序列化配置项失败: %v", err)
			return
		}

		// 更新缓存
		ecm.updateCache(key, &item)

		changeEvent = &ConfigChangeEvent{
			Type:        "update",
			Key:         item.Key,
			NewValue:    item.Value,
			Version:     item.Version,
			Environment: item.Environment,
			Namespace:   item.Namespace,
			Timestamp:   time.Now(),
			Author:      "external",
		}

	case clientv3.EventTypeDelete:
		// 配置删除
		ecm.removeFromCache(key)

		changeEvent = &ConfigChangeEvent{
			Type:      "delete",
			Key:       ecm.extractConfigKey(key),
			Timestamp: time.Now(),
			Author:    "external",
		}
	}

	if changeEvent != nil {
		ecm.notifyWatchers(changeEvent)
	}
}

// extractConfigKey 从完整路径提取配置键
func (ecm *EnhancedConfigManager) extractConfigKey(fullKey string) string {
	// 从 /vector_sphere/config/namespace/environment/key 提取 key
	parts := strings.Split(fullKey, "/")
	if len(parts) >= 5 {
		return parts[len(parts)-1]
	}
	return fullKey
}

// auditLog 审计日志
func (ecm *EnhancedConfigManager) auditLog(event *ConfigChangeEvent) {
	logger.Info("Config audit: type=%s, key=%s, env=%s, ns=%s, author=%s, time=%s",
		event.Type, event.Key, event.Environment, event.Namespace, event.Author, event.Timestamp.Format(time.RFC3339))

	// 这里可以集成更复杂的审计系统
	// 例如发送到日志收集系统、数据库等
}

// Close 关闭配置管理器
func (ecm *EnhancedConfigManager) Close() error {
	logger.Info("Closing enhanced config manager")

	ecm.cancel()

	// 关闭所有监听器
	ecm.watchersMu.Lock()
	for watcherID := range ecm.watchers {
		delete(ecm.watchers, watcherID)
	}
	ecm.watchersMu.Unlock()

	// 关闭所有通知通道
	ecm.mu.Lock()
	for _, ch := range ecm.changeListeners {
		close(ch)
	}
	ecm.changeListeners = make(map[string]chan *ConfigChangeEvent)
	ecm.mu.Unlock()

	return nil
}

// InitNamespace 初始化配置命名空间
// 创建命名空间的基础结构和默认配置
func (ecm *EnhancedConfigManager) InitNamespace(ctx context.Context, namespace string, environment string) error {
	logger.Info("Initializing config namespace: %s, environment: %s", namespace, environment)

	// 设置当前环境和命名空间
	ecm.currentNS = namespace
	ecm.currentEnv = environment

	// 检查命名空间是否已存在
	namespaceKey := fmt.Sprintf("%s/%s", ecm.basePrefix, namespace)
	resp, err := ecm.client.Get(ctx, namespaceKey, clientv3.WithPrefix())
	if err != nil {
		return fmt.Errorf("failed to check namespace existence: %v", err)
	}

	// 如果命名空间已存在且有配置项，则不需要初始化
	if len(resp.Kvs) > 0 {
		logger.Info("Namespace %s already exists with %d config items", namespace, len(resp.Kvs))
		return nil
	}

	// 创建命名空间元数据
	metadata := map[string]interface{}{
		"created_at":   time.Now(),
		"created_by":   "system",
		"environments": []string{environment},
		"description":  fmt.Sprintf("Configuration namespace for %s", namespace),
	}

	// 存储命名空间元数据
	metadataKey := fmt.Sprintf("%s/%s/_metadata", ecm.basePrefix, namespace)
	metadataBytes, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal namespace metadata: %v", err)
	}

	_, err = ecm.client.Put(ctx, metadataKey, string(metadataBytes))
	if err != nil {
		return fmt.Errorf("failed to store namespace metadata: %v", err)
	}

	// 创建默认配置组
	defaultGroup := &ConfigGroup{
		Name:        "default",
		Namespace:   namespace,
		Environment: environment,
		Items:       make(map[string]*ConfigItem),
		Version: &ConfigVersion{
			Version:     ecm.generateVersion(),
			Timestamp:   time.Now(),
			Author:      "system",
			Description: "Initial namespace configuration",
			Tags:        []string{"default", "initial"},
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	// 添加一些默认配置项
	defaultGroup.Items["app.name"] = &ConfigItem{
		Key:         "app.name",
		Value:       "VectorSphere",
		Type:        "string",
		Environment: environment,
		Namespace:   namespace,
		Version:     defaultGroup.Version.Version,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		Tags:        []string{"default"},
	}

	defaultGroup.Items["app.version"] = &ConfigItem{
		Key:         "app.version",
		Value:       "1.0.0",
		Type:        "string",
		Environment: environment,
		Namespace:   namespace,
		Version:     defaultGroup.Version.Version,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		Tags:        []string{"default"},
	}

	defaultGroup.Items["app.environment"] = &ConfigItem{
		Key:         "app.environment",
		Value:       environment,
		Type:        "string",
		Environment: environment,
		Namespace:   namespace,
		Version:     defaultGroup.Version.Version,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		Tags:        []string{"default"},
	}

	// 存储默认配置组
	err = ecm.SetConfigGroup(ctx, defaultGroup)
	if err != nil {
		return fmt.Errorf("failed to create default config group: %v", err)
	}

	// 创建环境特定的目录结构
	environmentKey := fmt.Sprintf("%s/%s/%s/_metadata", ecm.basePrefix, namespace, environment)
	environmentMetadata := map[string]interface{}{
		"created_at":  time.Now(),
		"description": fmt.Sprintf("Environment %s for namespace %s", environment, namespace),
	}

	environmentMetadataBytes, err := json.Marshal(environmentMetadata)
	if err != nil {
		return fmt.Errorf("failed to marshal environment metadata: %v", err)
	}

	_, err = ecm.client.Put(ctx, environmentKey, string(environmentMetadataBytes))
	if err != nil {
		return fmt.Errorf("failed to store environment metadata: %v", err)
	}

	logger.Info("Namespace %s initialized successfully with environment %s", namespace, environment)
	return nil
}
