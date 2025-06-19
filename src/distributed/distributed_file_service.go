package distributed

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"VectorSphere/src/library/logger"
	"VectorSphere/src/proto/serverProto"
)

// FileMetadata 文件元数据
type FileMetadata struct {
	Filename     string      `json:"filename"`
	Size         int64       `json:"size"`
	LastModified time.Time   `json:"last_modified"`
	Shards       []ShardInfo `json:"shards"`
	Checksum     string      `json:"checksum"`
	OwnerNode    string      `json:"owner_node"`
	Replicas     []string    `json:"replicas"`
	Version      int64       `json:"version"`
}

// ShardInfo 分片信息
type ShardInfo struct {
	ShardID  string   `json:"shard_id"`
	Offset   int64    `json:"offset"`
	Size     int64    `json:"size"`
	NodeIDs  []string `json:"node_ids"` // 存储该分片的节点列表
	Checksum string   `json:"checksum"`
}

// DistributedFileService 分布式文件服务
type DistributedFileService struct {
	distributedManager   *DistributedManager
	serviceDiscovery     *ServiceDiscovery
	communicationService *CommunicationService

	baseDir       string // 文件存储基础目录
	metadataDir   string // 元数据存储目录
	shardSize     int64  // 分片大小，默认为 64MB
	replicaFactor int    // 副本因子，默认为 2

	metadataCache map[string]*FileMetadata // 文件名 -> 元数据
	metadataMutex sync.RWMutex

	fileOperationLocks map[string]*sync.RWMutex // 文件级别的锁
	locksMutex         sync.Mutex

	ctx    context.Context
	cancel context.CancelFunc
}

// NewDistributedFileService 创建分布式文件服务
func NewDistributedFileService(dm *DistributedManager, sd *ServiceDiscovery, cs *CommunicationService, baseDir string) *DistributedFileService {
	ctx, cancel := context.WithCancel(context.Background())

	// 确保目录存在
	metadataDir := filepath.Join(baseDir, "metadata")
	err := os.MkdirAll(metadataDir, 0755)
	if err != nil {
		cancel()
		return nil
	}
	err = os.MkdirAll(filepath.Join(baseDir, "shards"), 0755)
	if err != nil {
		cancel()
		return nil
	}

	return &DistributedFileService{
		distributedManager:   dm,
		serviceDiscovery:     sd,
		communicationService: cs,
		baseDir:              baseDir,
		metadataDir:          metadataDir,
		shardSize:            64 * 1024 * 1024, // 64MB
		replicaFactor:        2,
		metadataCache:        make(map[string]*FileMetadata),
		fileOperationLocks:   make(map[string]*sync.RWMutex),
		ctx:                  ctx,
		cancel:               cancel,
	}
}

// Start 启动分布式文件服务
func (dfs *DistributedFileService) Start() error {
	logger.Info("Starting distributed file service...")

	// 加载所有元数据到缓存
	if err := dfs.loadAllMetadata(); err != nil {
		return fmt.Errorf("加载元数据失败: %v", err)
	}

	// 启动后台任务
	go dfs.startBackgroundTasks()

	return nil
}

// =============================================================================
// gRPC 服务接口实现
// =============================================================================

// CreateFile 创建文件
func (dfs *DistributedFileService) CreateFile(ctx context.Context, req *serverProto.CreateFileRequest) (*serverProto.CreateFileResponse, error) {
	// 获取文件锁
	fileLock := dfs.getFileLock(req.Filename)
	fileLock.Lock()
	defer fileLock.Unlock()

	// 检查文件是否已存在
	dfs.metadataMutex.RLock()
	_, exists := dfs.metadataCache[req.Filename]
	dfs.metadataMutex.RUnlock()

	if exists {
		return &serverProto.CreateFileResponse{
			ErrorMessage: fmt.Sprintf("文件 %s 已存在", req.Filename),
		}, nil
	}

	// 创建文件元数据
	metadata := &FileMetadata{
		Filename:     req.Filename,
		Size:         req.TotalSize,
		LastModified: time.Now(),
		Checksum:     req.ChecksumMd5,
		OwnerNode:    dfs.distributedManager.localhost,
		Replicas:     []string{},
		Version:      1,
		Shards:       []ShardInfo{},
	}

	// 保存元数据
	dfs.metadataMutex.Lock()
	err := dfs.saveMetadata(metadata)
	dfs.metadataMutex.Unlock()

	if err != nil {
		return &serverProto.CreateFileResponse{
			ErrorMessage: fmt.Sprintf("保存元数据失败: %v", err),
		}, nil
	}

	// 构建响应
	fileInfo := &serverProto.FileInfo{
		Filename:          metadata.Filename,
		Size_:             metadata.Size,
		ChecksumMd5:       metadata.Checksum,
		CreatedAt:         metadata.LastModified.Unix(),
		UpdatedAt:         metadata.LastModified.Unix(),
		ReplicationFactor: int32(dfs.replicaFactor),
		Status:            "CREATED",
		Metadata:          req.Metadata,
	}

	return &serverProto.CreateFileResponse{
		FileInfo: fileInfo,
	}, nil
}

// DeleteFile 删除文件 (gRPC接口实现)
func (dfs *DistributedFileService) DeleteFile(ctx context.Context, req *serverProto.DeleteFileRequest) (*serverProto.DeleteFileResponse, error) {
	err := dfs.deleteFileInternal(req.Filename)
	if err != nil {
		return &serverProto.DeleteFileResponse{
			Success:      false,
			ErrorMessage: err.Error(),
		}, nil
	}

	return &serverProto.DeleteFileResponse{
		Success: true,
	}, nil
}

// GetFileInfo 获取文件信息
func (dfs *DistributedFileService) GetFileInfo(ctx context.Context, req *serverProto.GetFileInfoRequest) (*serverProto.GetFileInfoResponse, error) {
	metadata, err := dfs.getFileMetadataInternal(req.Filename)
	if err != nil {
		return &serverProto.GetFileInfoResponse{
			ErrorMessage: err.Error(),
		}, nil
	}

	// 构建分片信息
	shards := make([]*serverProto.ShardInfo, len(metadata.Shards))
	for i, shard := range metadata.Shards {
		shards[i] = &serverProto.ShardInfo{
			ShardId:      shard.ShardID,
			Size_:        shard.Size,
			ChecksumMd5:  shard.Checksum,
			StorageNodes: shard.NodeIDs,
			Status:       "COMPLETE",
			Order:        int32(i),
		}
	}

	// 构建文件信息
	fileInfo := &serverProto.FileInfo{
		Filename:          metadata.Filename,
		Size_:             metadata.Size,
		ChecksumMd5:       metadata.Checksum,
		CreatedAt:         metadata.LastModified.Unix(),
		UpdatedAt:         metadata.LastModified.Unix(),
		Shards:            shards,
		ReplicationFactor: int32(dfs.replicaFactor),
		Status:            "COMPLETE",
	}

	return &serverProto.GetFileInfoResponse{
		FileInfo: fileInfo,
	}, nil
}

// ListFiles 列出文件 (gRPC接口实现)
func (dfs *DistributedFileService) ListFiles(ctx context.Context, req *serverProto.ListFilesRequest) (*serverProto.ListFilesResponse, error) {
	files, err := dfs.listFilesInternal()
	if err != nil {
		return &serverProto.ListFilesResponse{
			ErrorMessage: err.Error(),
		}, nil
	}

	// 过滤和分页
	filteredFiles := make([]FileMetadata, 0)
	for _, file := range files {
		if req.Prefix == "" || strings.HasPrefix(file.Filename, req.Prefix) {
			filteredFiles = append(filteredFiles, file)
		}
	}

	// 简单分页实现
	pageSize := int(req.PageSize)
	if pageSize <= 0 {
		pageSize = 100 // 默认页大小
	}

	startIndex := 0
	if req.PageToken != "" {
		// 简单的页面令牌实现（实际应用中应该更安全）
		if idx, parseErr := fmt.Sscanf(req.PageToken, "page_%d", &startIndex); parseErr != nil || idx != 1 {
			startIndex = 0
		}
	}

	endIndex := startIndex + pageSize
	if endIndex > len(filteredFiles) {
		endIndex = len(filteredFiles)
	}

	// 构建响应文件列表
	responseFiles := make([]*serverProto.FileInfo, 0)
	for i := startIndex; i < endIndex; i++ {
		file := filteredFiles[i]
		
		// 构建分片信息
		shards := make([]*serverProto.ShardInfo, len(file.Shards))
		for j, shard := range file.Shards {
			shards[j] = &serverProto.ShardInfo{
				ShardId:      shard.ShardID,
				Size_:        shard.Size,
				ChecksumMd5:  shard.Checksum,
				StorageNodes: shard.NodeIDs,
				Status:       "COMPLETE",
				Order:        int32(j),
			}
		}

		fileInfo := &serverProto.FileInfo{
			Filename:          file.Filename,
			Size_:             file.Size,
			ChecksumMd5:       file.Checksum,
			CreatedAt:         file.LastModified.Unix(),
			UpdatedAt:         file.LastModified.Unix(),
			Shards:            shards,
			ReplicationFactor: int32(dfs.replicaFactor),
			Status:            "COMPLETE",
		}
		responseFiles = append(responseFiles, fileInfo)
	}

	// 构建下一页令牌
	nextPageToken := ""
	if endIndex < len(filteredFiles) {
		nextPageToken = fmt.Sprintf("page_%d", endIndex)
	}

	return &serverProto.ListFilesResponse{
		Files:         responseFiles,
		NextPageToken: nextPageToken,
	}, nil
}

// UploadShard 上传分片（流式）
func (dfs *DistributedFileService) UploadShard(stream serverProto.DistributedStorageService_UploadShardServer) error {
	// 接收第一个消息（元数据）
	req, err := stream.Recv()
	if err != nil {
		return fmt.Errorf("接收元数据失败: %v", err)
	}

	metadata := req.GetMetadata()
	if metadata == nil {
		return fmt.Errorf("第一个消息必须包含元数据")
	}

	// 创建临时文件存储分片数据
	tempFile, err := os.CreateTemp("", "shard_*")
	if err != nil {
		return fmt.Errorf("创建临时文件失败: %v", err)
	}
	defer func() {
		tempFile.Close()
		os.Remove(tempFile.Name())
	}()

	// 接收分片数据
	hasher := md5.New()
	var totalSize int64

	for {
		req, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("接收分片数据失败: %v", err)
		}

		chunk := req.GetChunk()
		if chunk == nil {
			continue
		}

		// 写入临时文件
		n, err := tempFile.Write(chunk)
		if err != nil {
			return fmt.Errorf("写入分片数据失败: %v", err)
		}

		// 更新校验和
		hasher.Write(chunk)
		totalSize += int64(n)
	}

	// 验证校验和
	calculatedChecksum := fmt.Sprintf("%x", hasher.Sum(nil))
	if calculatedChecksum != metadata.ChecksumMd5 {
		return fmt.Errorf("分片校验和不匹配")
	}

	// 验证大小
	if totalSize != metadata.ShardSize {
		return fmt.Errorf("分片大小不匹配")
	}

	// 保存分片到正确位置
	shardsDir := filepath.Join(dfs.baseDir, "shards", metadata.Filename)
	if err := os.MkdirAll(shardsDir, 0755); err != nil {
		return fmt.Errorf("创建分片目录失败: %v", err)
	}

	shardPath := filepath.Join(shardsDir, metadata.ShardId)
	tempFile.Seek(0, 0)
	shardFile, err := os.Create(shardPath)
	if err != nil {
		return fmt.Errorf("创建分片文件失败: %v", err)
	}
	defer shardFile.Close()

	if _, err := io.Copy(shardFile, tempFile); err != nil {
		return fmt.Errorf("保存分片文件失败: %v", err)
	}

	// 更新文件元数据
	dfs.metadataMutex.Lock()
	fileMetadata, exists := dfs.metadataCache[metadata.Filename]
	if exists {
		// 添加或更新分片信息
		shardInfo := ShardInfo{
			ShardID:  metadata.ShardId,
			Offset:   int64(metadata.Order) * dfs.shardSize,
			Size:     totalSize,
			NodeIDs:  []string{dfs.distributedManager.localhost},
			Checksum: calculatedChecksum,
		}

		// 查找是否已存在该分片
		found := false
		for i, shard := range fileMetadata.Shards {
			if shard.ShardID == metadata.ShardId {
				fileMetadata.Shards[i] = shardInfo
				found = true
				break
			}
		}

		if !found {
			fileMetadata.Shards = append(fileMetadata.Shards, shardInfo)
		}

		// 保存更新的元数据
		dfs.saveMetadata(fileMetadata)
	}
	dfs.metadataMutex.Unlock()

	// 发送响应
	return stream.SendAndClose(&serverProto.UploadShardResponse{
		Filename:     metadata.Filename,
		ShardId:      metadata.ShardId,
		Success:      true,
		ErrorMessage: "",
	})
}

// DownloadShard 下载分片（流式）
func (dfs *DistributedFileService) DownloadShard(req *serverProto.DownloadShardRequest, stream serverProto.DistributedStorageService_DownloadShardServer) error {
	// 构建分片文件路径
	shardPath := filepath.Join(dfs.baseDir, "shards", req.Filename, req.ShardId)

	// 打开分片文件
	file, err := os.Open(shardPath)
	if err != nil {
		return fmt.Errorf("打开分片文件失败: %v", err)
	}
	defer file.Close()

	// 获取文件信息
	fileInfo, err := file.Stat()
	if err != nil {
		return fmt.Errorf("获取分片文件信息失败: %v", err)
	}

	// 处理偏移量和长度
	offset := req.Offset
	length := req.Length
	if length <= 0 {
		length = fileInfo.Size() - offset
	}

	// 设置文件偏移量
	if offset > 0 {
		if _, err := file.Seek(offset, 0); err != nil {
			return fmt.Errorf("设置文件偏移量失败: %v", err)
		}
	}

	// 流式发送分片数据
	buffer := make([]byte, 32*1024) // 32KB 缓冲区
	remaining := length

	for remaining > 0 {
		// 计算本次读取大小
		readSize := int64(len(buffer))
		if readSize > remaining {
			readSize = remaining
		}

		// 读取数据
		n, err := file.Read(buffer[:readSize])
		if err != nil && err != io.EOF {
			return fmt.Errorf("读取分片数据失败: %v", err)
		}

		if n == 0 {
			break
		}

		// 发送数据块
		if err := stream.Send(&serverProto.DownloadShardResponse{
			Chunk: buffer[:n],
		}); err != nil {
			return fmt.Errorf("发送分片数据失败: %v", err)
		}

		remaining -= int64(n)
	}

	return nil
}

// RegisterNode 注册节点
func (dfs *DistributedFileService) RegisterNode(ctx context.Context, req *serverProto.RegisterNodeRequest) (*serverProto.RegisterNodeResponse, error) {
	// 这里应该与服务发现组件集成
	// 目前简单实现
	logger.Info("节点注册请求: ID=%s, Address=%s, Capacity=%d", req.NodeId, req.Address, req.Capacity)

	// 在实际实现中，这里应该:
	// 1. 验证节点信息
	// 2. 将节点信息注册到服务发现系统
	// 3. 更新节点状态

	return &serverProto.RegisterNodeResponse{
		Success: true,
	}, nil
}

// GetNodeStatus 获取节点状态
func (dfs *DistributedFileService) GetNodeStatus(ctx context.Context, req *serverProto.GetNodeStatusRequest) (*serverProto.GetNodeStatusResponse, error) {
	// 获取本地节点状态
	if req.NodeId == dfs.distributedManager.localhost || req.NodeId == "" {
		// 计算存储使用情况
		var usedSpace int64
		var capacity int64 = 100 * 1024 * 1024 * 1024 // 100GB 默认容量

		// 计算已使用空间
		filepath.Walk(dfs.baseDir, func(path string, info os.FileInfo, err error) error {
			if err == nil && !info.IsDir() {
				usedSpace += info.Size()
			}
			return nil
		})

		loadFactor := float32(usedSpace) / float32(capacity)

		return &serverProto.GetNodeStatusResponse{
			NodeId:        dfs.distributedManager.localhost,
			Status:        "ONLINE",
			Capacity:      capacity,
			UsedSpace:     usedSpace,
			LoadFactor:    loadFactor,
			LastHeartbeat: time.Now().Unix(),
		}, nil
	}

	// 获取其他节点状态（通过服务发现）
	_, err := dfs.serviceDiscovery.GetSlaveInfo(req.NodeId)
	if err != nil {
		return &serverProto.GetNodeStatusResponse{
			ErrorMessage: fmt.Sprintf("获取节点信息失败: %v", err),
		}, nil
	}

	// 这里应该通过gRPC调用获取远程节点状态
	// 目前返回基本信息
	return &serverProto.GetNodeStatusResponse{
		NodeId:        req.NodeId,
		Status:        "ONLINE",
		LastHeartbeat: time.Now().Unix(),
	}, nil
}

// StoreShard 存储分片
func (dfs *DistributedFileService) StoreShard(ctx context.Context, req *serverProto.StoreShardRequest) (*serverProto.StoreShardResponse, error) {
	// 创建分片目录
	shardsDir := filepath.Join(dfs.baseDir, "shards", req.Filename)
	if err := os.MkdirAll(shardsDir, 0755); err != nil {
		return &serverProto.StoreShardResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("创建分片目录失败: %v", err),
		}, nil
	}

	// 验证校验和
	hasher := md5.New()
	hasher.Write(req.Data)
	calculatedChecksum := fmt.Sprintf("%x", hasher.Sum(nil))

	if calculatedChecksum != req.Checksum {
		return &serverProto.StoreShardResponse{
			Success:      false,
			ErrorMessage: "分片校验和不匹配",
		}, nil
	}

	// 保存分片文件
	shardPath := filepath.Join(shardsDir, req.ShardId)
	if err := os.WriteFile(shardPath, req.Data, 0644); err != nil {
		return &serverProto.StoreShardResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("保存分片文件失败: %v", err),
		}, nil
	}

	logger.Info("分片 %s 已成功存储到本地", req.ShardId)
	return &serverProto.StoreShardResponse{
		Success: true,
	}, nil
}

// GetShard 获取分片
func (dfs *DistributedFileService) GetShard(ctx context.Context, req *serverProto.GetShardRequest) (*serverProto.GetShardResponse, error) {
	// 构建分片文件路径
	shardPath := filepath.Join(dfs.baseDir, "shards", req.Filename, req.ShardId)

	// 读取分片文件
	data, err := os.ReadFile(shardPath)
	if err != nil {
		return &serverProto.GetShardResponse{
			ErrorMessage: fmt.Sprintf("读取分片文件失败: %v", err),
		}, nil
	}

	return &serverProto.GetShardResponse{
		Data: data,
	}, nil
}

// DeleteShard 删除分片
func (dfs *DistributedFileService) DeleteShard(ctx context.Context, req *serverProto.DeleteShardRequest) (*serverProto.DeleteShardResponse, error) {
	// 构建分片文件路径
	shardPath := filepath.Join(dfs.baseDir, "shards", req.Filename, req.ShardId)

	// 删除分片文件
	if err := os.Remove(shardPath); err != nil {
		if os.IsNotExist(err) {
			// 文件不存在，认为删除成功
			return &serverProto.DeleteShardResponse{
				Success: true,
			}, nil
		}
		return &serverProto.DeleteShardResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("删除分片文件失败: %v", err),
		}, nil
	}

	logger.Info("分片 %s 已成功删除", req.ShardId)
	return &serverProto.DeleteShardResponse{
		Success: true,
	}, nil
}

// SyncMetadata 同步元数据
func (dfs *DistributedFileService) SyncMetadata(ctx context.Context, req *serverProto.SyncMetadataRequest) (*serverProto.SyncMetadataResponse, error) {
	// 反序列化元数据
	var metadata FileMetadata
	if err := json.Unmarshal(req.Metadata, &metadata); err != nil {
		return &serverProto.SyncMetadataResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("反序列化元数据失败: %v", err),
		}, nil
	}

	// 检查版本号
	dfs.metadataMutex.Lock()
	existingMetadata, exists := dfs.metadataCache[req.Filename]
	if exists && existingMetadata.Version >= req.Version {
		// 本地版本更新或相同，不需要同步
		dfs.metadataMutex.Unlock()
		return &serverProto.SyncMetadataResponse{
			Success: true,
		}, nil
	}

	// 更新元数据
	metadata.Version = req.Version
	if err := dfs.saveMetadata(&metadata); err != nil {
		dfs.metadataMutex.Unlock()
		return &serverProto.SyncMetadataResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("保存元数据失败: %v", err),
		}, nil
	}
	dfs.metadataMutex.Unlock()

	logger.Info("文件 %s 的元数据已同步，版本: %d", req.Filename, req.Version)
	return &serverProto.SyncMetadataResponse{
		Success: true,
	}, nil
}

// Stop 停止分布式文件服务
func (dfs *DistributedFileService) Stop() {
	logger.Info("Stopping distributed file service...")
	dfs.cancel()
}

// 启动后台任务
func (dfs *DistributedFileService) startBackgroundTasks() {
	// 定期检查分片健康状态
	go dfs.periodicHealthCheck()

	// 定期同步元数据
	go dfs.periodicMetadataSync()
}

// 定期检查分片健康状态
func (dfs *DistributedFileService) periodicHealthCheck() {
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			dfs.checkShardsHealth()
		case <-dfs.ctx.Done():
			return
		}
	}
}

// 检查所有分片的健康状态
func (dfs *DistributedFileService) checkShardsHealth() {
	dfs.metadataMutex.RLock()
	files := make([]string, 0, len(dfs.metadataCache))
	for filename := range dfs.metadataCache {
		files = append(files, filename)
	}
	dfs.metadataMutex.RUnlock()

	for _, filename := range files {
		dfs.checkFileShards(filename)
	}
}

// 检查单个文件的分片健康状态
func (dfs *DistributedFileService) checkFileShards(filename string) {
	dfs.metadataMutex.RLock()
	metadata, exists := dfs.metadataCache[filename]
	dfs.metadataMutex.RUnlock()

	if !exists {
		return
	}

	// 获取文件锁
	fileLock := dfs.getFileLock(filename)
	fileLock.Lock()
	defer fileLock.Unlock()

	// 重新检查元数据（可能在获取锁的过程中已经被修改）
	dfs.metadataMutex.RLock()
	metadata, exists = dfs.metadataCache[filename]
	dfs.metadataMutex.RUnlock()

	if !exists {
		return
	}

	// 检查每个分片
	for i, shard := range metadata.Shards {
		// 检查分片副本数量是否满足要求
		if len(shard.NodeIDs) < dfs.replicaFactor {
			logger.Warning("文件 %s 的分片 %s 副本数量不足: %d/%d",
				filename, shard.ShardID, len(shard.NodeIDs), dfs.replicaFactor)

			// 尝试复制分片到其他节点
			dfs.replicateShard(filename, i)
		}
	}
}

// 复制分片到其他节点
func (dfs *DistributedFileService) replicateShard(filename string, shardIndex int) {
	// 获取健康的slave节点列表
	healthySlaves := dfs.serviceDiscovery.GetHealthySlaveAddresses()
	if len(healthySlaves) == 0 {
		logger.Error("没有可用的slave节点进行分片复制")
		return
	}

	dfs.metadataMutex.RLock()
	metadata, exists := dfs.metadataCache[filename]
	dfs.metadataMutex.RUnlock()

	if !exists || shardIndex >= len(metadata.Shards) {
		return
	}

	shard := metadata.Shards[shardIndex]

	// 找出不包含该分片的节点
	availableNodes := make([]string, 0)
	for _, slave := range healthySlaves {
		containsShard := false
		for _, nodeID := range shard.NodeIDs {
			if slave == nodeID {
				containsShard = true
				break
			}
		}
		if !containsShard {
			availableNodes = append(availableNodes, slave)
		}
	}

	if len(availableNodes) == 0 {
		logger.Warning("没有可用节点进行分片复制")
		return
	}

	// 从现有节点获取分片数据并复制到新节点
	if len(shard.NodeIDs) > 0 {
		sourceNode := shard.NodeIDs[0]
		targetNode := availableNodes[0]

		// 从源节点获取分片数据
		logger.Info("正在从节点 %s 获取分片 %s", sourceNode, shard.ShardID)
		shardData, err := dfs.fetchShardFromNode(filename, shard.ShardID, sourceNode)
		if err != nil {
			logger.Error("从节点 %s 获取分片 %s 失败: %v", sourceNode, shard.ShardID, err)
			return
		}

		// 获取目标节点信息
		slaveInfo, err := dfs.serviceDiscovery.GetSlaveInfo(targetNode)
		if err != nil {
			logger.Error("获取目标节点 %s 信息失败: %v", targetNode, err)
			return
		}

		// 获取gRPC客户端
		client, err := dfs.communicationService.GetSShardStoreSlaveClient(slaveInfo.Address)
		if err != nil {
			logger.Error("获取目标节点 %s 的gRPC客户端失败: %v", targetNode, err)
			return
		}

		// 创建请求
		req := &serverProto.StoreShardRequest{
			Filename: filename,
			ShardId:  shard.ShardID,
			Data:     shardData,
			Checksum: shard.Checksum,
		}

		// 发送请求
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		_, err = client.StoreShard(ctx, req)
		if err != nil {
			logger.Error("复制分片到目标节点 %s 失败: %v", targetNode, err)
			return
		}

		logger.Info("分片 %s 成功从节点 %s 复制到节点 %s", shard.ShardID, sourceNode, targetNode)

		// 更新元数据
		dfs.metadataMutex.Lock()
		if metadata, exists = dfs.metadataCache[filename]; exists && shardIndex < len(metadata.Shards) {
			metadata.Shards[shardIndex].NodeIDs = append(metadata.Shards[shardIndex].NodeIDs, targetNode)
			err = dfs.saveMetadata(metadata)
			if err != nil {
				return
			}
		}
		dfs.metadataMutex.Unlock()
	}
}

// 定期同步元数据
func (dfs *DistributedFileService) periodicMetadataSync() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			dfs.syncMetadata()
		case <-dfs.ctx.Done():
			return
		}
	}
}

// 同步元数据到所有节点
func (dfs *DistributedFileService) syncMetadata() {
	// 只有master节点执行同步操作
	if !dfs.distributedManager.IsMaster() {
		return
	}

	dfs.metadataMutex.RLock()
	for _, metadata := range dfs.metadataCache {
		// 将元数据同步到所有slave节点
		dfs.broadcastMetadata(metadata)
	}
	dfs.metadataMutex.RUnlock()
}

// 广播元数据到所有slave节点
func (dfs *DistributedFileService) broadcastMetadata(metadata *FileMetadata) {
	// 获取所有健康的slave节点
	healthySlaves := dfs.serviceDiscovery.GetHealthySlaveAddresses()
	if len(healthySlaves) == 0 {
		return
	}

	// 序列化元数据
	metadataBytes, err := json.Marshal(metadata)
	if err != nil {
		logger.Error("序列化元数据失败: %v", err)
		return
	}

	// 并发广播到所有slave节点
	var wg sync.WaitGroup
	for _, nodeID := range healthySlaves {
		wg.Add(1)
		go func(nodeID string) {
			defer wg.Done()

			// 获取节点信息
			slaveInfo, err := dfs.serviceDiscovery.GetSlaveInfo(nodeID)
			if err != nil {
				logger.Error("获取节点信息失败: %v", err)
				return
			}

			// 获取gRPC客户端
			client, err := dfs.communicationService.GetSShardStoreSlaveClient(slaveInfo.Address)
			if err != nil {
				logger.Error("获取gRPC客户端失败: %v", err)
				return
			}

			// 创建请求
			req := &serverProto.SyncMetadataRequest{
				Filename: metadata.Filename,
				Metadata: metadataBytes,
				Version:  metadata.Version,
			}

			// 发送请求
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()

			_, err = client.SyncMetadata(ctx, req)
			if err != nil {
				logger.Error("同步元数据到节点 %s 失败: %v", nodeID, err)
				return
			}

			logger.Debug("成功同步文件 %s 的元数据到节点 %s", metadata.Filename, nodeID)
		}(nodeID)
	}

	// 等待所有广播操作完成
	wg.Wait()
}

// 加载所有元数据到缓存
func (dfs *DistributedFileService) loadAllMetadata() error {
	dfs.metadataMutex.Lock()
	defer dfs.metadataMutex.Unlock()

	// 清空缓存
	dfs.metadataCache = make(map[string]*FileMetadata)

	// 读取元数据目录中的所有文件
	files, err := os.ReadDir(dfs.metadataDir)
	if err != nil {
		if os.IsNotExist(err) {
			// 目录不存在，创建它
			err = os.MkdirAll(dfs.metadataDir, 0755)
			if err != nil {
				return fmt.Errorf("创建元数据目录失败: %v", err)
			}
			return nil
		}
		return fmt.Errorf("读取元数据目录失败: %v", err)
	}

	// 加载每个元数据文件
	for _, file := range files {
		if file.IsDir() {
			continue
		}

		filePath := filepath.Join(dfs.metadataDir, file.Name())
		metadata, err := dfs.loadMetadataFromFile(filePath)
		if err != nil {
			logger.Error("加载元数据文件 %s 失败: %v", filePath, err)
			continue
		}

		dfs.metadataCache[metadata.Filename] = metadata
	}

	logger.Info("已加载 %d 个文件的元数据", len(dfs.metadataCache))
	return nil
}

// 从文件加载元数据
func (dfs *DistributedFileService) loadMetadataFromFile(filePath string) (*FileMetadata, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("读取元数据文件失败: %v", err)
	}

	var metadata FileMetadata
	if err := json.Unmarshal(data, &metadata); err != nil {
		return nil, fmt.Errorf("解析元数据失败: %v", err)
	}

	return &metadata, nil
}

// 保存元数据到文件
func (dfs *DistributedFileService) saveMetadata(metadata *FileMetadata) error {
	// 更新版本号和修改时间
	metadata.Version++
	metadata.LastModified = time.Now()

	// 序列化元数据
	data, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return fmt.Errorf("序列化元数据失败: %v", err)
	}

	// 保存到文件
	filePath := filepath.Join(dfs.metadataDir, fmt.Sprintf("%s.json", metadata.Filename))
	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("写入元数据文件失败: %v", err)
	}

	// 更新缓存
	dfs.metadataCache[metadata.Filename] = metadata

	return nil
}

// 获取文件锁
func (dfs *DistributedFileService) getFileLock(filename string) *sync.RWMutex {
	dfs.locksMutex.Lock()
	defer dfs.locksMutex.Unlock()

	lock, exists := dfs.fileOperationLocks[filename]
	if !exists {
		lock = &sync.RWMutex{}
		dfs.fileOperationLocks[filename] = lock
	}

	return lock
}

// UploadFile 上传文件到分布式存储系统
func (dfs *DistributedFileService) UploadFile(localFilePath, remoteFilename string) error {
	// 打开本地文件
	file, err := os.Open(localFilePath)
	if err != nil {
		return fmt.Errorf("打开本地文件失败: %v", err)
	}
	defer func(file *os.File) {
		err := file.Close()
		if err != nil {
			logger.Error("close file has error: %v", err)
		}
	}(file)

	// 获取文件信息
	fileInfo, err := file.Stat()
	if err != nil {
		return fmt.Errorf("获取文件信息失败: %v", err)
	}

	// 获取文件锁
	fileLock := dfs.getFileLock(remoteFilename)
	fileLock.Lock()
	defer fileLock.Unlock()

	// 计算文件MD5校验和
	hasher := md5.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return fmt.Errorf("计算文件校验和失败: %v", err)
	}
	checksum := fmt.Sprintf("%x", hasher.Sum(nil))

	// 重置文件指针到开头
	if _, err := file.Seek(0, 0); err != nil {
		return fmt.Errorf("重置文件指针失败: %v", err)
	}

	// 创建元数据
	metadata := &FileMetadata{
		Filename:     remoteFilename,
		Size:         fileInfo.Size(),
		LastModified: time.Now(),
		Checksum:     checksum,
		OwnerNode:    dfs.distributedManager.localhost,
		Replicas:     []string{},
		Version:      1,
		Shards:       []ShardInfo{},
	}

	// 计算分片数量
	numShards := (fileInfo.Size() + dfs.shardSize - 1) / dfs.shardSize
	shards := make([]ShardInfo, numShards)

	// 获取健康的slave节点列表
	healthySlaves := dfs.serviceDiscovery.GetHealthySlaveAddresses()
	if len(healthySlaves) == 0 && !dfs.distributedManager.IsMaster() {
		return fmt.Errorf("没有可用的节点存储文件")
	}

	// 创建分片目录
	shardsDir := filepath.Join(dfs.baseDir, "shards", remoteFilename)
	if err := os.MkdirAll(shardsDir, 0755); err != nil {
		return fmt.Errorf("创建分片目录失败: %v", err)
	}

	// 分片处理
	buffer := make([]byte, dfs.shardSize)
	for i := int64(0); i < numShards; i++ {
		// 读取分片数据
		n, err := file.Read(buffer)
		if err != nil && err != io.EOF {
			return fmt.Errorf("读取文件分片失败: %v", err)
		}

		// 计算分片校验和
		shardHasher := md5.New()
		shardHasher.Write(buffer[:n])
		shardChecksum := fmt.Sprintf("%x", shardHasher.Sum(nil))

		// 创建分片ID
		shardID := fmt.Sprintf("%s_shard_%d", remoteFilename, i)

		// 保存分片到本地
		shardPath := filepath.Join(shardsDir, shardID)
		if err := os.WriteFile(shardPath, buffer[:n], 0644); err != nil {
			return fmt.Errorf("保存分片文件失败: %v", err)
		}

		// 更新分片信息
		shards[i] = ShardInfo{
			ShardID:  shardID,
			Offset:   i * dfs.shardSize,
			Size:     int64(n),
			NodeIDs:  []string{dfs.distributedManager.localhost},
			Checksum: shardChecksum,
		}

		// 复制分片到其他节点
		if len(healthySlaves) > 0 {
			dfs.replicateShardToNodes(shardPath, shardID, shards[i].Checksum, healthySlaves, dfs.replicaFactor-1)
		}
	}

	// 更新元数据
	metadata.Shards = shards

	// 保存元数据
	dfs.metadataMutex.Lock()
	if err := dfs.saveMetadata(metadata); err != nil {
		dfs.metadataMutex.Unlock()
		return fmt.Errorf("保存元数据失败: %v", err)
	}
	dfs.metadataMutex.Unlock()

	logger.Info("文件 %s 上传成功，大小: %d 字节，分片数: %d", remoteFilename, fileInfo.Size(), numShards)
	return nil
}

// 复制分片到其他节点
func (dfs *DistributedFileService) replicateShardToNodes(shardPath, shardID, checksum string, nodes []string, count int) {
	if count <= 0 || len(nodes) == 0 {
		return
	}

	// 读取分片数据
	shardData, err := os.ReadFile(shardPath)
	if err != nil {
		logger.Error("读取分片文件失败: %v", err)
		return
	}

	// 随机选择节点进行复制，避免所有分片都复制到同一节点
	selectedNodes := make([]string, 0, count)
	if len(nodes) <= count {
		selectedNodes = nodes
	} else {
		// 随机选择count个节点
		indices := rand.Perm(len(nodes))
		for i := 0; i < count; i++ {
			selectedNodes = append(selectedNodes, nodes[indices[i]])
		}
	}

	// 并发复制到选定的节点
	var wg sync.WaitGroup
	for _, nodeID := range selectedNodes {
		wg.Add(1)
		go func(nodeID string) {
			defer wg.Done()

			// 获取节点信息
			slaveInfo, err := dfs.serviceDiscovery.GetSlaveInfo(nodeID)
			if err != nil {
				logger.Error("获取节点信息失败: %v", err)
				return
			}

			// 获取gRPC客户端
			client, err := dfs.communicationService.GetSShardStoreSlaveClient(slaveInfo.Address)
			if err != nil {
				logger.Error("获取gRPC客户端失败: %v", err)
				return
			}

			// 创建请求
			req := &serverProto.StoreShardRequest{
				Filename: filepath.Base(filepath.Dir(shardPath)),
				ShardId:  shardID,
				Data:     shardData,
				Checksum: checksum,
			}

			// 发送请求
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			_, err = client.StoreShard(ctx, req)
			if err != nil {
				logger.Error("复制分片到节点 %s 失败: %v", nodeID, err)
				return
			}

			logger.Info("分片 %s 成功复制到节点 %s", shardID, nodeID)
		}(nodeID)
	}

	// 等待所有复制操作完成
	wg.Wait()
}

// DownloadFile 从分布式存储系统下载文件
func (dfs *DistributedFileService) DownloadFile(remoteFilename, localFilePath string) error {
	// 获取文件锁
	fileLock := dfs.getFileLock(remoteFilename)
	fileLock.RLock()
	defer fileLock.RUnlock()

	// 获取元数据
	dfs.metadataMutex.RLock()
	metadata, exists := dfs.metadataCache[remoteFilename]
	dfs.metadataMutex.RUnlock()

	if !exists {
		return fmt.Errorf("文件 %s 不存在", remoteFilename)
	}

	// 创建本地文件
	file, err := os.Create(localFilePath)
	if err != nil {
		return fmt.Errorf("创建本地文件失败: %v", err)
	}
	defer func(file *os.File) {
		err = file.Close()
		if err != nil {
			logger.Error("close file failed: %v", err)
		}
	}(file)

	// 按顺序下载并合并所有分片
	for _, shard := range metadata.Shards {
		// 尝试从本地获取分片
		shardPath := filepath.Join(dfs.baseDir, "shards", remoteFilename, shard.ShardID)
		shardData, err := os.ReadFile(shardPath)

		// 如果本地没有，则从其他节点获取
		if err != nil {
			var fetchErr error
			for _, nodeID := range shard.NodeIDs {
				if nodeID == dfs.distributedManager.localhost {
					continue // 跳过本地节点，因为已经尝试过了
				}

				// 从远程节点获取分片
				shardData, fetchErr = dfs.fetchShardFromNode(remoteFilename, shard.ShardID, nodeID)
				if fetchErr == nil {
					break // 成功获取分片
				}
			}

			if fetchErr != nil {
				return fmt.Errorf("无法获取分片 %s: %v", shard.ShardID, fetchErr)
			}
		}

		// 验证分片校验和
		shardHasher := md5.New()
		shardHasher.Write(shardData)
		shardChecksum := fmt.Sprintf("%x", shardHasher.Sum(nil))

		if shardChecksum != shard.Checksum {
			return fmt.Errorf("分片 %s 校验和不匹配", shard.ShardID)
		}

		// 写入分片数据到本地文件
		if _, err := file.Write(shardData); err != nil {
			return fmt.Errorf("写入分片数据失败: %v", err)
		}
	}

	// 验证整个文件的校验和
	_, err = file.Seek(0, 0)
	if err != nil {
		return err
	}
	hasher := md5.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return fmt.Errorf("计算文件校验和失败: %v", err)
	}
	checksum := fmt.Sprintf("%x", hasher.Sum(nil))

	if checksum != metadata.Checksum {
		return fmt.Errorf("文件校验和不匹配")
	}

	logger.Info("文件 %s 下载成功，大小: %d 字节", remoteFilename, metadata.Size)
	return nil
}

// 从远程节点获取分片
func (dfs *DistributedFileService) fetchShardFromNode(filename, shardID, nodeID string) ([]byte, error) {
	// 获取节点地址
	slaveInfo, err := dfs.serviceDiscovery.GetSlaveInfo(nodeID)
	if err != nil {
		return nil, fmt.Errorf("获取节点信息失败: %v", err)
	}

	// 获取gRPC客户端
	client, err := dfs.communicationService.GetSShardStoreSlaveClient(slaveInfo.Address)
	if err != nil {
		return nil, fmt.Errorf("获取gRPC客户端失败: %v", err)
	}

	// 创建请求
	req := &serverProto.GetShardRequest{
		Filename: filename,
		ShardId:  shardID,
	}

	// 发送请求
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := client.GetShard(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("获取分片失败: %v", err)
	}

	return resp.Data, nil
}

// DeleteFile 从分布式存储系统删除文件
func (dfs *DistributedFileService) deleteFileInternal(filename string) error {
	// 获取文件锁
	fileLock := dfs.getFileLock(filename)
	fileLock.Lock()
	defer fileLock.Unlock()

	// 检查文件是否存在
	dfs.metadataMutex.RLock()
	metadata, exists := dfs.metadataCache[filename]
	dfs.metadataMutex.RUnlock()

	if !exists {
		return fmt.Errorf("文件 %s 不存在", filename)
	}

	// 删除本地分片
	shardsDir := filepath.Join(dfs.baseDir, "shards", filename)
	if err := os.RemoveAll(shardsDir); err != nil {
		logger.Warning("删除本地分片目录失败: %v", err)
	}

	// 通知其他节点删除分片
	for _, shard := range metadata.Shards {
		for _, nodeID := range shard.NodeIDs {
			if nodeID == dfs.distributedManager.localhost {
				continue // 跳过本地节点
			}

			// 通知远程节点删除分片
			go func(nodeID, filename string, shardID string) {
				// 获取节点信息
				slaveInfo, err := dfs.serviceDiscovery.GetSlaveInfo(nodeID)
				if err != nil {
					logger.Error("获取节点信息失败: %v", err)
					return
				}

				// 获取gRPC客户端
				client, err := dfs.communicationService.GetSShardStoreSlaveClient(slaveInfo.Address)
				if err != nil {
					logger.Error("获取gRPC客户端失败: %v", err)
					return
				}

				// 创建请求
				req := &serverProto.DeleteShardRequest{
					Filename: filename,
					ShardId:  shardID,
				}

				// 发送请求
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()

				_, err = client.DeleteShard(ctx, req)
				if err != nil {
					logger.Error("通知节点 %s 删除分片失败: %v", nodeID, err)
					return
				}

				logger.Debug("成功通知节点 %s 删除文件 %s 的分片 %d", nodeID, filename, shardID)
			}(nodeID, filename, shard.ShardID)
		}
	}

	// 删除元数据文件
	metadataPath := filepath.Join(dfs.metadataDir, fmt.Sprintf("%s.json", filename))
	if err := os.Remove(metadataPath); err != nil {
		logger.Warning("删除元数据文件失败: %v", err)
	}

	// 从缓存中删除
	dfs.metadataMutex.Lock()
	delete(dfs.metadataCache, filename)
	dfs.metadataMutex.Unlock()

	logger.Info("文件 %s 删除成功", filename)
	return nil
}

// ListFiles 列出分布式存储系统中的所有文件
func (dfs *DistributedFileService) listFilesInternal() ([]FileMetadata, error) {
	dfs.metadataMutex.RLock()
	defer dfs.metadataMutex.RUnlock()

	files := make([]FileMetadata, 0, len(dfs.metadataCache))
	for _, metadata := range dfs.metadataCache {
		files = append(files, *metadata)
	}

	return files, nil
}

// GetFileMetadata 获取文件元数据
func (dfs *DistributedFileService) getFileMetadataInternal(filename string) (*FileMetadata, error) {
	dfs.metadataMutex.RLock()
	defer dfs.metadataMutex.RUnlock()

	metadata, exists := dfs.metadataCache[filename]
	if !exists {
		return nil, fmt.Errorf("文件 %s 不存在", filename)
	}

	return metadata, nil
}

// 以下是与向量数据库集成的功能

// VectorDBFile 表示向量数据库文件
type VectorDBFile struct {
	TableName   string    // 表名
	IndexType   string    // 索引类型 (如 "hnsw")
	Dimension   int       // 向量维度
	CreatedAt   time.Time // 创建时间
	LastUpdated time.Time // 最后更新时间
	FileSize    int64     // 文件大小
	DocCount    int       // 文档数量
}

// StoreVectorDBTable 存储向量数据库表文件
func (dfs *DistributedFileService) StoreVectorDBTable(tableName string, tableDir string) error {
	// 检查表目录是否存在
	if _, err := os.Stat(tableDir); os.IsNotExist(err) {
		return fmt.Errorf("表目录 %s 不存在", tableDir)
	}

	// 获取表目录下的所有文件
	files, err := os.ReadDir(tableDir)
	if err != nil {
		return fmt.Errorf("读取表目录失败: %v", err)
	}

	// 创建临时目录用于打包
	tempDir, err := os.MkdirTemp("", "vectordb_table_*")
	if err != nil {
		return fmt.Errorf("创建临时目录失败: %v", err)
	}
	defer func(path string) {
		err = os.RemoveAll(path)
		if err != nil {
			logger.Error("remove has error:%v", err)
		}
	}(tempDir)

	// 创建表元数据文件
	tableMetadata := VectorDBFile{
		TableName:   tableName,
		CreatedAt:   time.Now(),
		LastUpdated: time.Now(),
	}

	// 读取表配置文件获取更多信息
	configPath := filepath.Join(tableDir, "config.json")
	if configData, err := os.ReadFile(configPath); err == nil {
		var config map[string]interface{}
		if err := json.Unmarshal(configData, &config); err == nil {
			if dim, ok := config["dimension"].(float64); ok {
				tableMetadata.Dimension = int(dim)
			}
			if indexType, ok := config["index_type"].(string); ok {
				tableMetadata.IndexType = indexType
			}
			if docCount, ok := config["doc_count"].(float64); ok {
				tableMetadata.DocCount = int(docCount)
			}
		}
	}

	// 将表元数据写入临时文件
	metadataPath := filepath.Join(tempDir, "table_metadata.json")
	metadataBytes, err := json.Marshal(tableMetadata)
	if err != nil {
		return fmt.Errorf("序列化表元数据失败: %v", err)
	}
	if err := os.WriteFile(metadataPath, metadataBytes, 0644); err != nil {
		return fmt.Errorf("写入表元数据失败: %v", err)
	}

	// 复制表文件到临时目录
	for _, file := range files {
		srcPath := filepath.Join(tableDir, file.Name())
		dstPath := filepath.Join(tempDir, file.Name())

		// 如果是目录，则递归复制
		if file.IsDir() {
			if err := copyDir(srcPath, dstPath); err != nil {
				return fmt.Errorf("复制目录 %s 失败: %v", file.Name(), err)
			}
		} else {
			if err := copyFile(srcPath, dstPath); err != nil {
				return fmt.Errorf("复制文件 %s 失败: %v", file.Name(), err)
			}
		}
	}

	// 创建表压缩包
	tableTarPath := filepath.Join(tempDir, tableName+".tar.gz")
	if err := createTarGz(tempDir, tableTarPath); err != nil {
		return fmt.Errorf("创建表压缩包失败: %v", err)
	}

	// 上传表压缩包到分布式存储
	remoteFilename := fmt.Sprintf("vectordb_table_%s.tar.gz", tableName)
	if err := dfs.UploadFile(tableTarPath, remoteFilename); err != nil {
		return fmt.Errorf("上传表文件失败: %v", err)
	}

	logger.Info("向量数据库表 %s 已成功存储到分布式系统", tableName)
	return nil
}

// RestoreVectorDBTable 从分布式存储恢复向量数据库表
func (dfs *DistributedFileService) RestoreVectorDBTable(tableName string, destDir string) error {
	// 构建远程文件名
	remoteFilename := fmt.Sprintf("vectordb_table_%s.tar.gz", tableName)

	// 创建临时目录
	tempDir, err := os.MkdirTemp("", "vectordb_restore_*")
	if err != nil {
		return fmt.Errorf("创建临时目录失败: %v", err)
	}
	defer func(path string) {
		err = os.RemoveAll(path)
		if err != nil {
			logger.Error("remove has error:%v", err)
		}
	}(tempDir)

	// 下载表压缩包
	tableTarPath := filepath.Join(tempDir, tableName+".tar.gz")
	if err := dfs.DownloadFile(remoteFilename, tableTarPath); err != nil {
		return fmt.Errorf("下载表文件失败: %v", err)
	}

	// 解压表文件
	extractDir := filepath.Join(tempDir, "extract")
	if err := os.MkdirAll(extractDir, 0755); err != nil {
		return fmt.Errorf("创建解压目录失败: %v", err)
	}

	if err := extractTarGz(tableTarPath, extractDir); err != nil {
		return fmt.Errorf("解压表文件失败: %v", err)
	}

	// 确保目标目录存在
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("创建目标目录失败: %v", err)
	}

	// 复制解压后的文件到目标目录
	files, err := os.ReadDir(extractDir)
	if err != nil {
		return fmt.Errorf("读取解压目录失败: %v", err)
	}

	for _, file := range files {
		// 跳过表元数据文件
		if file.Name() == "table_metadata.json" {
			continue
		}

		srcPath := filepath.Join(extractDir, file.Name())
		dstPath := filepath.Join(destDir, file.Name())

		// 如果是目录，则递归复制
		if file.IsDir() {
			if err := copyDir(srcPath, dstPath); err != nil {
				return fmt.Errorf("复制目录 %s 失败: %v", file.Name(), err)
			}
		} else {
			if err := copyFile(srcPath, dstPath); err != nil {
				return fmt.Errorf("复制文件 %s 失败: %v", file.Name(), err)
			}
		}
	}

	logger.Info("向量数据库表 %s 已成功从分布式系统恢复到 %s", tableName, destDir)
	return nil
}

// ListVectorDBTables 列出所有向量数据库表
func (dfs *DistributedFileService) ListVectorDBTables() ([]VectorDBFile, error) {
	// 获取所有文件
	files, err := dfs.listFilesInternal()
	if err != nil {
		return nil, fmt.Errorf("获取文件列表失败: %v", err)
	}

	// 筛选向量数据库表文件
	tables := make([]VectorDBFile, 0)
	prefix := "vectordb_table_"
	suffix := ".tar.gz"

	for _, file := range files {
		if strings.HasPrefix(file.Filename, prefix) && strings.HasSuffix(file.Filename, suffix) {
			// 提取表名
			tableName := file.Filename[len(prefix) : len(file.Filename)-len(suffix)]

			// 创建表信息
			table := VectorDBFile{
				TableName:   tableName,
				CreatedAt:   file.LastModified,
				LastUpdated: file.LastModified,
				FileSize:    file.Size,
			}

			tables = append(tables, table)
		}
	}

	return tables, nil
}

// 辅助函数：复制文件
func copyFile(src, dst string) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer func(srcFile *os.File) {
		err = srcFile.Close()
		if err != nil {
			logger.Error("close has error:%v", err)
		}
	}(srcFile)

	dstFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer func(dstFile *os.File) {
		err := dstFile.Close()
		if err != nil {
			logger.Error("close has error:%v", err)
		}
	}(dstFile)

	_, err = io.Copy(dstFile, srcFile)
	return err
}

// 辅助函数：复制目录
func copyDir(src, dst string) error {
	// 创建目标目录
	if err := os.MkdirAll(dst, 0755); err != nil {
		return err
	}

	// 读取源目录
	entries, err := os.ReadDir(src)
	if err != nil {
		return err
	}

	// 复制每个条目
	for _, entry := range entries {
		srcPath := filepath.Join(src, entry.Name())
		dstPath := filepath.Join(dst, entry.Name())

		if entry.IsDir() {
			if err := copyDir(srcPath, dstPath); err != nil {
				return err
			}
		} else {
			if err := copyFile(srcPath, dstPath); err != nil {
				return err
			}
		}
	}

	return nil
}

// 辅助函数：创建tar.gz压缩包
func createTarGz(srcDir, tarPath string) error {
	// 创建tar文件
	tarFile, err := os.Create(tarPath)
	if err != nil {
		return err
	}
	defer func(tarFile *os.File) {
		err := tarFile.Close()
		if err != nil {
			logger.Error("close has error:%v", err)
		}
	}(tarFile)

	// 创建gzip writer
	gw := gzip.NewWriter(tarFile)
	defer func(gw *gzip.Writer) {
		err := gw.Close()
		if err != nil {
			logger.Error("close has error:%v", err)
		}
	}(gw)

	// 创建tar writer
	tw := tar.NewWriter(gw)
	defer func(tw *tar.Writer) {
		err := tw.Close()
		if err != nil {
			logger.Error("close has error:%v", err)
		}
	}(tw)

	// 遍历源目录
	return filepath.Walk(srcDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// 获取相对路径
		relPath, err := filepath.Rel(srcDir, path)
		if err != nil {
			return err
		}

		// 跳过根目录
		if relPath == "." {
			return nil
		}

		// 创建tar头
		header, err := tar.FileInfoHeader(info, "")
		if err != nil {
			return err
		}
		header.Name = relPath

		// 写入tar头
		if err := tw.WriteHeader(header); err != nil {
			return err
		}

		// 如果是文件，写入文件内容
		if !info.IsDir() {
			file, err := os.Open(path)
			if err != nil {
				return err
			}
			defer func(file *os.File) {
				err := file.Close()
				if err != nil {
					logger.Error("close has error:%v", err)
				}
			}(file)

			_, err = io.Copy(tw, file)
			if err != nil {
				return err
			}
		}

		return nil
	})
}

// 辅助函数：解压tar.gz文件
func extractTarGz(tarPath, destDir string) error {
	// 打开tar文件
	tarFile, err := os.Open(tarPath)
	if err != nil {
		return err
	}
	defer func(tarFile *os.File) {
		err := tarFile.Close()
		if err != nil {
			logger.Error("close has error:%v", err)
		}
	}(tarFile)

	// 创建gzip reader
	gr, err := gzip.NewReader(tarFile)
	if err != nil {
		return err
	}
	defer func(gr *gzip.Reader) {
		err := gr.Close()
		if err != nil {
			logger.Error("close has error:%v", err)
		}
	}(gr)

	// 创建tar reader
	tr := tar.NewReader(gr)

	// 遍历tar文件
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		// 构建目标路径
		target := filepath.Join(destDir, header.Name)

		switch header.Typeflag {
		case tar.TypeDir:
			// 创建目录
			if err := os.MkdirAll(target, 0755); err != nil {
				return err
			}
		case tar.TypeReg:
			// 创建文件
			dir := filepath.Dir(target)
			if err := os.MkdirAll(dir, 0755); err != nil {
				return err
			}

			file, err := os.Create(target)
			if err != nil {
				return err
			}
			_, err = io.Copy(file, tr)
			closeErr := file.Close()
			if closeErr != nil {
				logger.Error("close has error:%v", err)
			}
			// 写入文件内容
			if err != nil {
				return err
			}
		}
	}

	return nil
}
