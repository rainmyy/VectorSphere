// versions:
// 	protoc-gen-go v1.33.0
// 	protoc        v3.10.0
// source: distributed_storage.proto

package server

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

// 文件信息
type FileInfo struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Filename          string            `protobuf:"bytes,1,opt,name=filename,proto3" json:"filename,omitempty"`
	Size              int64             `protobuf:"varint,2,opt,name=size,proto3" json:"size,omitempty"`
	ChecksumMd5       string            `protobuf:"bytes,3,opt,name=checksum_md5,json=checksumMd5,proto3" json:"checksum_md5,omitempty"`
	CreatedAt         int64             `protobuf:"varint,4,opt,name=created_at,json=createdAt,proto3" json:"created_at,omitempty"`                                                                     // Unix timestamp
	UpdatedAt         int64             `protobuf:"varint,5,opt,name=updated_at,json=updatedAt,proto3" json:"updated_at,omitempty"`                                                                     // Unix timestamp
	Metadata          map[string]string `protobuf:"bytes,6,rep,name=metadata,proto3" json:"metadata,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"` // 其他元数据，例如文件类型、所有者等
	Shards            []*ShardInfo      `protobuf:"bytes,7,rep,name=shards,proto3" json:"shards,omitempty"`                                                                                             // 文件包含的分片信息
	ReplicationFactor int32             `protobuf:"varint,8,opt,name=replication_factor,json=replicationFactor,proto3" json:"replication_factor,omitempty"`                                             // 副本因子
	Status            string            `protobuf:"bytes,9,opt,name=status,proto3" json:"status,omitempty"`                                                                                             // 文件状态 (e.g., UPLOADING, COMPLETE, DELETED)
}

func (x *FileInfo) Reset() {
	*x = FileInfo{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *FileInfo) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*FileInfo) ProtoMessage() {}

func (x *FileInfo) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use FileInfo.ProtoReflect.Descriptor instead.
func (*FileInfo) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{0}
}

func (x *FileInfo) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

func (x *FileInfo) GetSize() int64 {
	if x != nil {
		return x.Size
	}
	return 0
}

func (x *FileInfo) GetChecksumMd5() string {
	if x != nil {
		return x.ChecksumMd5
	}
	return ""
}

func (x *FileInfo) GetCreatedAt() int64 {
	if x != nil {
		return x.CreatedAt
	}
	return 0
}

func (x *FileInfo) GetUpdatedAt() int64 {
	if x != nil {
		return x.UpdatedAt
	}
	return 0
}

func (x *FileInfo) GetMetadata() map[string]string {
	if x != nil {
		return x.Metadata
	}
	return nil
}

func (x *FileInfo) GetShards() []*ShardInfo {
	if x != nil {
		return x.Shards
	}
	return nil
}

func (x *FileInfo) GetReplicationFactor() int32 {
	if x != nil {
		return x.ReplicationFactor
	}
	return 0
}

func (x *FileInfo) GetStatus() string {
	if x != nil {
		return x.Status
	}
	return ""
}

// 分片信息
type ShardInfo struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	ShardId      string   `protobuf:"bytes,1,opt,name=shard_id,json=shardId,proto3" json:"shard_id,omitempty"`
	Size         int64    `protobuf:"varint,2,opt,name=size,proto3" json:"size,omitempty"`
	ChecksumMd5  string   `protobuf:"bytes,3,opt,name=checksum_md5,json=checksumMd5,proto3" json:"checksum_md5,omitempty"`
	StorageNodes []string `protobuf:"bytes,4,rep,name=storage_nodes,json=storageNodes,proto3" json:"storage_nodes,omitempty"` // 存储该分片的节点列表
	Status       string   `protobuf:"bytes,5,opt,name=status,proto3" json:"status,omitempty"`                                 // 分片状态 (e.g., UPLOADING, COMPLETE, CORRUPTED)
	Order        int32    `protobuf:"varint,6,opt,name=order,proto3" json:"order,omitempty"`                                  // 分片在文件中的顺序
}

func (x *ShardInfo) Reset() {
	*x = ShardInfo{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ShardInfo) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ShardInfo) ProtoMessage() {}

func (x *ShardInfo) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ShardInfo.ProtoReflect.Descriptor instead.
func (*ShardInfo) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{1}
}

func (x *ShardInfo) GetShardId() string {
	if x != nil {
		return x.ShardId
	}
	return ""
}

func (x *ShardInfo) GetSize() int64 {
	if x != nil {
		return x.Size
	}
	return 0
}

func (x *ShardInfo) GetChecksumMd5() string {
	if x != nil {
		return x.ChecksumMd5
	}
	return ""
}

func (x *ShardInfo) GetStorageNodes() []string {
	if x != nil {
		return x.StorageNodes
	}
	return nil
}

func (x *ShardInfo) GetStatus() string {
	if x != nil {
		return x.Status
	}
	return ""
}

func (x *ShardInfo) GetOrder() int32 {
	if x != nil {
		return x.Order
	}
	return 0
}

// 创建文件请求
type CreateFileRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Filename          string            `protobuf:"bytes,1,opt,name=filename,proto3" json:"filename,omitempty"`
	TotalSize         int64             `protobuf:"varint,2,opt,name=total_size,json=totalSize,proto3" json:"total_size,omitempty"`                                                                     // 预期总大小
	ChecksumMd5       string            `protobuf:"bytes,3,opt,name=checksum_md5,json=checksumMd5,proto3" json:"checksum_md5,omitempty"`                                                                // 整个文件的预期校验和 (可选)
	ReplicationFactor int32             `protobuf:"varint,4,opt,name=replication_factor,json=replicationFactor,proto3" json:"replication_factor,omitempty"`                                             // 副本因子
	Metadata          map[string]string `protobuf:"bytes,5,rep,name=metadata,proto3" json:"metadata,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"` // 文件元数据
}

func (x *CreateFileRequest) Reset() {
	*x = CreateFileRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *CreateFileRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CreateFileRequest) ProtoMessage() {}

func (x *CreateFileRequest) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CreateFileRequest.ProtoReflect.Descriptor instead.
func (*CreateFileRequest) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{2}
}

func (x *CreateFileRequest) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

func (x *CreateFileRequest) GetTotalSize() int64 {
	if x != nil {
		return x.TotalSize
	}
	return 0
}

func (x *CreateFileRequest) GetChecksumMd5() string {
	if x != nil {
		return x.ChecksumMd5
	}
	return ""
}

func (x *CreateFileRequest) GetReplicationFactor() int32 {
	if x != nil {
		return x.ReplicationFactor
	}
	return 0
}

func (x *CreateFileRequest) GetMetadata() map[string]string {
	if x != nil {
		return x.Metadata
	}
	return nil
}

// 创建文件响应
type CreateFileResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	FileInfo     *FileInfo `protobuf:"bytes,1,opt,name=file_info,json=fileInfo,proto3" json:"file_info,omitempty"`
	ErrorMessage string    `protobuf:"bytes,2,opt,name=error_message,json=errorMessage,proto3" json:"error_message,omitempty"`
}

func (x *CreateFileResponse) Reset() {
	*x = CreateFileResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *CreateFileResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CreateFileResponse) ProtoMessage() {}

func (x *CreateFileResponse) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CreateFileResponse.ProtoReflect.Descriptor instead.
func (*CreateFileResponse) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{3}
}

func (x *CreateFileResponse) GetFileInfo() *FileInfo {
	if x != nil {
		return x.FileInfo
	}
	return nil
}

func (x *CreateFileResponse) GetErrorMessage() string {
	if x != nil {
		return x.ErrorMessage
	}
	return ""
}

// 删除文件请求
type DeleteFileRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Filename string `protobuf:"bytes,1,opt,name=filename,proto3" json:"filename,omitempty"`
}

func (x *DeleteFileRequest) Reset() {
	*x = DeleteFileRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[4]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *DeleteFileRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeleteFileRequest) ProtoMessage() {}

func (x *DeleteFileRequest) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[4]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DeleteFileRequest.ProtoReflect.Descriptor instead.
func (*DeleteFileRequest) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{4}
}

func (x *DeleteFileRequest) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

// 删除文件响应
type DeleteFileResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Success      bool   `protobuf:"varint,1,opt,name=success,proto3" json:"success,omitempty"`
	ErrorMessage string `protobuf:"bytes,2,opt,name=error_message,json=errorMessage,proto3" json:"error_message,omitempty"`
}

func (x *DeleteFileResponse) Reset() {
	*x = DeleteFileResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[5]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *DeleteFileResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeleteFileResponse) ProtoMessage() {}

func (x *DeleteFileResponse) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[5]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DeleteFileResponse.ProtoReflect.Descriptor instead.
func (*DeleteFileResponse) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{5}
}

func (x *DeleteFileResponse) GetSuccess() bool {
	if x != nil {
		return x.Success
	}
	return false
}

func (x *DeleteFileResponse) GetErrorMessage() string {
	if x != nil {
		return x.ErrorMessage
	}
	return ""
}

// 获取文件信息请求
type GetFileInfoRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Filename string `protobuf:"bytes,1,opt,name=filename,proto3" json:"filename,omitempty"`
}

func (x *GetFileInfoRequest) Reset() {
	*x = GetFileInfoRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[6]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *GetFileInfoRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GetFileInfoRequest) ProtoMessage() {}

func (x *GetFileInfoRequest) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[6]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GetFileInfoRequest.ProtoReflect.Descriptor instead.
func (*GetFileInfoRequest) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{6}
}

func (x *GetFileInfoRequest) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

// 获取文件信息响应
type GetFileInfoResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	FileInfo     *FileInfo `protobuf:"bytes,1,opt,name=file_info,json=fileInfo,proto3" json:"file_info,omitempty"`
	ErrorMessage string    `protobuf:"bytes,2,opt,name=error_message,json=errorMessage,proto3" json:"error_message,omitempty"`
}

func (x *GetFileInfoResponse) Reset() {
	*x = GetFileInfoResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[7]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *GetFileInfoResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GetFileInfoResponse) ProtoMessage() {}

func (x *GetFileInfoResponse) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[7]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GetFileInfoResponse.ProtoReflect.Descriptor instead.
func (*GetFileInfoResponse) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{7}
}

func (x *GetFileInfoResponse) GetFileInfo() *FileInfo {
	if x != nil {
		return x.FileInfo
	}
	return nil
}

func (x *GetFileInfoResponse) GetErrorMessage() string {
	if x != nil {
		return x.ErrorMessage
	}
	return ""
}

// 列出文件请求
type ListFilesRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Prefix    string `protobuf:"bytes,1,opt,name=prefix,proto3" json:"prefix,omitempty"`                        // 可选，用于按前缀过滤
	PageSize  int32  `protobuf:"varint,2,opt,name=page_size,json=pageSize,proto3" json:"page_size,omitempty"`   // 分页大小
	PageToken string `protobuf:"bytes,3,opt,name=page_token,json=pageToken,proto3" json:"page_token,omitempty"` // 分页令牌
}

func (x *ListFilesRequest) Reset() {
	*x = ListFilesRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[8]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ListFilesRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ListFilesRequest) ProtoMessage() {}

func (x *ListFilesRequest) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[8]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ListFilesRequest.ProtoReflect.Descriptor instead.
func (*ListFilesRequest) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{8}
}

func (x *ListFilesRequest) GetPrefix() string {
	if x != nil {
		return x.Prefix
	}
	return ""
}

func (x *ListFilesRequest) GetPageSize() int32 {
	if x != nil {
		return x.PageSize
	}
	return 0
}

func (x *ListFilesRequest) GetPageToken() string {
	if x != nil {
		return x.PageToken
	}
	return ""
}

// 列出文件响应
type ListFilesResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Files         []*FileInfo `protobuf:"bytes,1,rep,name=files,proto3" json:"files,omitempty"`
	NextPageToken string      `protobuf:"bytes,2,opt,name=next_page_token,json=nextPageToken,proto3" json:"next_page_token,omitempty"`
	ErrorMessage  string      `protobuf:"bytes,3,opt,name=error_message,json=errorMessage,proto3" json:"error_message,omitempty"`
}

func (x *ListFilesResponse) Reset() {
	*x = ListFilesResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[9]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ListFilesResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ListFilesResponse) ProtoMessage() {}

func (x *ListFilesResponse) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[9]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ListFilesResponse.ProtoReflect.Descriptor instead.
func (*ListFilesResponse) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{9}
}

func (x *ListFilesResponse) GetFiles() []*FileInfo {
	if x != nil {
		return x.Files
	}
	return nil
}

func (x *ListFilesResponse) GetNextPageToken() string {
	if x != nil {
		return x.NextPageToken
	}
	return ""
}

func (x *ListFilesResponse) GetErrorMessage() string {
	if x != nil {
		return x.ErrorMessage
	}
	return ""
}

// 上传分片请求
type UploadShardRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Types that are assignable to DataOneof:
	//
	//	*UploadShardRequest_Metadata_
	//	*UploadShardRequest_Chunk
	DataOneof isUploadShardRequest_DataOneof `protobuf_oneof:"data_oneof"`
}

func (x *UploadShardRequest) Reset() {
	*x = UploadShardRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[10]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *UploadShardRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*UploadShardRequest) ProtoMessage() {}

func (x *UploadShardRequest) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[10]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use UploadShardRequest.ProtoReflect.Descriptor instead.
func (*UploadShardRequest) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{10}
}

func (m *UploadShardRequest) GetDataOneof() isUploadShardRequest_DataOneof {
	if m != nil {
		return m.DataOneof
	}
	return nil
}

func (x *UploadShardRequest) GetMetadata() *UploadShardRequest_Metadata {
	if x, ok := x.GetDataOneof().(*UploadShardRequest_Metadata_); ok {
		return x.Metadata
	}
	return nil
}

func (x *UploadShardRequest) GetChunk() []byte {
	if x, ok := x.GetDataOneof().(*UploadShardRequest_Chunk); ok {
		return x.Chunk
	}
	return nil
}

type isUploadShardRequest_DataOneof interface {
	isUploadShardRequest_DataOneof()
}

type UploadShardRequest_Metadata_ struct {
	Metadata *UploadShardRequest_Metadata `protobuf:"bytes,1,opt,name=metadata,proto3,oneof"`
}

type UploadShardRequest_Chunk struct {
	Chunk []byte `protobuf:"bytes,2,opt,name=chunk,proto3,oneof"` // 分片数据块
}

func (*UploadShardRequest_Metadata_) isUploadShardRequest_DataOneof() {}

func (*UploadShardRequest_Chunk) isUploadShardRequest_DataOneof() {}

// 上传分片响应
type UploadShardResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Filename     string `protobuf:"bytes,1,opt,name=filename,proto3" json:"filename,omitempty"`
	ShardId      string `protobuf:"bytes,2,opt,name=shard_id,json=shardId,proto3" json:"shard_id,omitempty"`
	Success      bool   `protobuf:"varint,3,opt,name=success,proto3" json:"success,omitempty"`
	ErrorMessage string `protobuf:"bytes,4,opt,name=error_message,json=errorMessage,proto3" json:"error_message,omitempty"`
}

func (x *UploadShardResponse) Reset() {
	*x = UploadShardResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[11]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *UploadShardResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*UploadShardResponse) ProtoMessage() {}

func (x *UploadShardResponse) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[11]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use UploadShardResponse.ProtoReflect.Descriptor instead.
func (*UploadShardResponse) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{11}
}

func (x *UploadShardResponse) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

func (x *UploadShardResponse) GetShardId() string {
	if x != nil {
		return x.ShardId
	}
	return ""
}

func (x *UploadShardResponse) GetSuccess() bool {
	if x != nil {
		return x.Success
	}
	return false
}

func (x *UploadShardResponse) GetErrorMessage() string {
	if x != nil {
		return x.ErrorMessage
	}
	return ""
}

// 下载分片请求
type DownloadShardRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Filename string `protobuf:"bytes,1,opt,name=filename,proto3" json:"filename,omitempty"`
	ShardId  string `protobuf:"bytes,2,opt,name=shard_id,json=shardId,proto3" json:"shard_id,omitempty"`
	Offset   int64  `protobuf:"varint,3,opt,name=offset,proto3" json:"offset,omitempty"` // 下载偏移量 (可选, 用于断点续传)
	Length   int64  `protobuf:"varint,4,opt,name=length,proto3" json:"length,omitempty"` // 下载长度 (可选, 用于部分下载)
}

func (x *DownloadShardRequest) Reset() {
	*x = DownloadShardRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[12]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *DownloadShardRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DownloadShardRequest) ProtoMessage() {}

func (x *DownloadShardRequest) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[12]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DownloadShardRequest.ProtoReflect.Descriptor instead.
func (*DownloadShardRequest) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{12}
}

func (x *DownloadShardRequest) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

func (x *DownloadShardRequest) GetShardId() string {
	if x != nil {
		return x.ShardId
	}
	return ""
}

func (x *DownloadShardRequest) GetOffset() int64 {
	if x != nil {
		return x.Offset
	}
	return 0
}

func (x *DownloadShardRequest) GetLength() int64 {
	if x != nil {
		return x.Length
	}
	return 0
}

// 下载分片响应
type DownloadShardResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Chunk        []byte `protobuf:"bytes,1,opt,name=chunk,proto3" json:"chunk,omitempty"`                                   // 分片数据块
	ErrorMessage string `protobuf:"bytes,2,opt,name=error_message,json=errorMessage,proto3" json:"error_message,omitempty"` // 如果在流传输过程中发生错误
}

func (x *DownloadShardResponse) Reset() {
	*x = DownloadShardResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[13]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *DownloadShardResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DownloadShardResponse) ProtoMessage() {}

func (x *DownloadShardResponse) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[13]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DownloadShardResponse.ProtoReflect.Descriptor instead.
func (*DownloadShardResponse) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{13}
}

func (x *DownloadShardResponse) GetChunk() []byte {
	if x != nil {
		return x.Chunk
	}
	return nil
}

func (x *DownloadShardResponse) GetErrorMessage() string {
	if x != nil {
		return x.ErrorMessage
	}
	return ""
}

// 删除分片请求 (针对特定存储节点上的分片，与IndexService中的DeleteShard区分)
type DeleteShardStorageRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Filename string `protobuf:"bytes,1,opt,name=filename,proto3" json:"filename,omitempty"`
	ShardId  string `protobuf:"bytes,2,opt,name=shard_id,json=shardId,proto3" json:"shard_id,omitempty"`
	NodeId   string `protobuf:"bytes,3,opt,name=node_id,json=nodeId,proto3" json:"node_id,omitempty"` // 可选，如果需要指定从哪个节点删除
}

func (x *DeleteShardStorageRequest) Reset() {
	*x = DeleteShardStorageRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[14]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *DeleteShardStorageRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeleteShardStorageRequest) ProtoMessage() {}

func (x *DeleteShardStorageRequest) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[14]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DeleteShardStorageRequest.ProtoReflect.Descriptor instead.
func (*DeleteShardStorageRequest) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{14}
}

func (x *DeleteShardStorageRequest) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

func (x *DeleteShardStorageRequest) GetShardId() string {
	if x != nil {
		return x.ShardId
	}
	return ""
}

func (x *DeleteShardStorageRequest) GetNodeId() string {
	if x != nil {
		return x.NodeId
	}
	return ""
}

// 删除分片响应
type DeleteShardStorageResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Success      bool   `protobuf:"varint,1,opt,name=success,proto3" json:"success,omitempty"`
	ErrorMessage string `protobuf:"bytes,2,opt,name=error_message,json=errorMessage,proto3" json:"error_message,omitempty"`
}

func (x *DeleteShardStorageResponse) Reset() {
	*x = DeleteShardStorageResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[15]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *DeleteShardStorageResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DeleteShardStorageResponse) ProtoMessage() {}

func (x *DeleteShardStorageResponse) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[15]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DeleteShardStorageResponse.ProtoReflect.Descriptor instead.
func (*DeleteShardStorageResponse) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{15}
}

func (x *DeleteShardStorageResponse) GetSuccess() bool {
	if x != nil {
		return x.Success
	}
	return false
}

func (x *DeleteShardStorageResponse) GetErrorMessage() string {
	if x != nil {
		return x.ErrorMessage
	}
	return ""
}

// 节点注册请求 (可选)
type RegisterNodeRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	NodeId    string   `protobuf:"bytes,1,opt,name=node_id,json=nodeId,proto3" json:"node_id,omitempty"`
	Address   string   `protobuf:"bytes,2,opt,name=address,proto3" json:"address,omitempty"`                       // 节点地址 (ip:port)
	Capacity  int64    `protobuf:"varint,3,opt,name=capacity,proto3" json:"capacity,omitempty"`                    // 存储容量
	UsedSpace int64    `protobuf:"varint,4,opt,name=used_space,json=usedSpace,proto3" json:"used_space,omitempty"` // 已用空间
	Tags      []string `protobuf:"bytes,5,rep,name=tags,proto3" json:"tags,omitempty"`                             // 节点标签，例如区域、机架等
}

func (x *RegisterNodeRequest) Reset() {
	*x = RegisterNodeRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[16]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *RegisterNodeRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RegisterNodeRequest) ProtoMessage() {}

func (x *RegisterNodeRequest) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[16]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use RegisterNodeRequest.ProtoReflect.Descriptor instead.
func (*RegisterNodeRequest) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{16}
}

func (x *RegisterNodeRequest) GetNodeId() string {
	if x != nil {
		return x.NodeId
	}
	return ""
}

func (x *RegisterNodeRequest) GetAddress() string {
	if x != nil {
		return x.Address
	}
	return ""
}

func (x *RegisterNodeRequest) GetCapacity() int64 {
	if x != nil {
		return x.Capacity
	}
	return 0
}

func (x *RegisterNodeRequest) GetUsedSpace() int64 {
	if x != nil {
		return x.UsedSpace
	}
	return 0
}

func (x *RegisterNodeRequest) GetTags() []string {
	if x != nil {
		return x.Tags
	}
	return nil
}

// 节点注册响应 (可选)
type RegisterNodeResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Success      bool   `protobuf:"varint,1,opt,name=success,proto3" json:"success,omitempty"`
	ErrorMessage string `protobuf:"bytes,2,opt,name=error_message,json=errorMessage,proto3" json:"error_message,omitempty"`
}

func (x *RegisterNodeResponse) Reset() {
	*x = RegisterNodeResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[17]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *RegisterNodeResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RegisterNodeResponse) ProtoMessage() {}

func (x *RegisterNodeResponse) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[17]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use RegisterNodeResponse.ProtoReflect.Descriptor instead.
func (*RegisterNodeResponse) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{17}
}

func (x *RegisterNodeResponse) GetSuccess() bool {
	if x != nil {
		return x.Success
	}
	return false
}

func (x *RegisterNodeResponse) GetErrorMessage() string {
	if x != nil {
		return x.ErrorMessage
	}
	return ""
}

// 获取节点状态请求 (可选)
type GetNodeStatusRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	NodeId string `protobuf:"bytes,1,opt,name=node_id,json=nodeId,proto3" json:"node_id,omitempty"`
}

func (x *GetNodeStatusRequest) Reset() {
	*x = GetNodeStatusRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[18]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *GetNodeStatusRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GetNodeStatusRequest) ProtoMessage() {}

func (x *GetNodeStatusRequest) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[18]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GetNodeStatusRequest.ProtoReflect.Descriptor instead.
func (*GetNodeStatusRequest) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{18}
}

func (x *GetNodeStatusRequest) GetNodeId() string {
	if x != nil {
		return x.NodeId
	}
	return ""
}

// 获取节点状态响应 (可选)
type GetNodeStatusResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	NodeId        string  `protobuf:"bytes,1,opt,name=node_id,json=nodeId,proto3" json:"node_id,omitempty"`
	Status        string  `protobuf:"bytes,2,opt,name=status,proto3" json:"status,omitempty"` // 例如: ONLINE, OFFLINE, DEGRADED
	Capacity      int64   `protobuf:"varint,3,opt,name=capacity,proto3" json:"capacity,omitempty"`
	UsedSpace     int64   `protobuf:"varint,4,opt,name=used_space,json=usedSpace,proto3" json:"used_space,omitempty"`
	LoadFactor    float32 `protobuf:"fixed32,5,opt,name=load_factor,json=loadFactor,proto3" json:"load_factor,omitempty"`         // 负载因子
	LastHeartbeat int64   `protobuf:"varint,6,opt,name=last_heartbeat,json=lastHeartbeat,proto3" json:"last_heartbeat,omitempty"` // Unix timestamp
	ErrorMessage  string  `protobuf:"bytes,7,opt,name=error_message,json=errorMessage,proto3" json:"error_message,omitempty"`
}

func (x *GetNodeStatusResponse) Reset() {
	*x = GetNodeStatusResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[19]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *GetNodeStatusResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GetNodeStatusResponse) ProtoMessage() {}

func (x *GetNodeStatusResponse) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[19]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GetNodeStatusResponse.ProtoReflect.Descriptor instead.
func (*GetNodeStatusResponse) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{19}
}

func (x *GetNodeStatusResponse) GetNodeId() string {
	if x != nil {
		return x.NodeId
	}
	return ""
}

func (x *GetNodeStatusResponse) GetStatus() string {
	if x != nil {
		return x.Status
	}
	return ""
}

func (x *GetNodeStatusResponse) GetCapacity() int64 {
	if x != nil {
		return x.Capacity
	}
	return 0
}

func (x *GetNodeStatusResponse) GetUsedSpace() int64 {
	if x != nil {
		return x.UsedSpace
	}
	return 0
}

func (x *GetNodeStatusResponse) GetLoadFactor() float32 {
	if x != nil {
		return x.LoadFactor
	}
	return 0
}

func (x *GetNodeStatusResponse) GetLastHeartbeat() int64 {
	if x != nil {
		return x.LastHeartbeat
	}
	return 0
}

func (x *GetNodeStatusResponse) GetErrorMessage() string {
	if x != nil {
		return x.ErrorMessage
	}
	return ""
}

// 第一个消息必须是元数据
type UploadShardRequest_Metadata struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Filename    string `protobuf:"bytes,1,opt,name=filename,proto3" json:"filename,omitempty"`
	ShardId     string `protobuf:"bytes,2,opt,name=shard_id,json=shardId,proto3" json:"shard_id,omitempty"`
	ShardSize   int64  `protobuf:"varint,3,opt,name=shard_size,json=shardSize,proto3" json:"shard_size,omitempty"`      // 当前分片的大小
	ChecksumMd5 string `protobuf:"bytes,4,opt,name=checksum_md5,json=checksumMd5,proto3" json:"checksum_md5,omitempty"` // 当前分片的校验和
	Order       int32  `protobuf:"varint,5,opt,name=order,proto3" json:"order,omitempty"`                               // 分片在文件中的顺序
}

func (x *UploadShardRequest_Metadata) Reset() {
	*x = UploadShardRequest_Metadata{}
	if protoimpl.UnsafeEnabled {
		mi := &file_distributed_storage_proto_msgTypes[22]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *UploadShardRequest_Metadata) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*UploadShardRequest_Metadata) ProtoMessage() {}

func (x *UploadShardRequest_Metadata) ProtoReflect() protoreflect.Message {
	mi := &file_distributed_storage_proto_msgTypes[22]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use UploadShardRequest_Metadata.ProtoReflect.Descriptor instead.
func (*UploadShardRequest_Metadata) Descriptor() ([]byte, []int) {
	return file_distributed_storage_proto_rawDescGZIP(), []int{10, 0}
}

func (x *UploadShardRequest_Metadata) GetFilename() string {
	if x != nil {
		return x.Filename
	}
	return ""
}

func (x *UploadShardRequest_Metadata) GetShardId() string {
	if x != nil {
		return x.ShardId
	}
	return ""
}

func (x *UploadShardRequest_Metadata) GetShardSize() int64 {
	if x != nil {
		return x.ShardSize
	}
	return 0
}

func (x *UploadShardRequest_Metadata) GetChecksumMd5() string {
	if x != nil {
		return x.ChecksumMd5
	}
	return ""
}

func (x *UploadShardRequest_Metadata) GetOrder() int32 {
	if x != nil {
		return x.Order
	}
	return 0
}

var File_distributed_storage_proto protoreflect.FileDescriptor

var file_distributed_storage_proto_rawDesc = []byte{
	0x0a, 0x19, 0x64, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64, 0x5f, 0x73, 0x74,
	0x6f, 0x72, 0x61, 0x67, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x06, 0x73, 0x65, 0x72,
	0x76, 0x65, 0x72, 0x22, 0x86, 0x03, 0x0a, 0x08, 0x46, 0x69, 0x6c, 0x65, 0x49, 0x6e, 0x66, 0x6f,
	0x12, 0x1a, 0x0a, 0x08, 0x66, 0x69, 0x6c, 0x65, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x01,
	0x28, 0x09, 0x52, 0x08, 0x66, 0x69, 0x6c, 0x65, 0x6e, 0x61, 0x6d, 0x65, 0x12, 0x12, 0x0a, 0x04,
	0x73, 0x69, 0x7a, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x03, 0x52, 0x04, 0x73, 0x69, 0x7a, 0x65,
	0x12, 0x21, 0x0a, 0x0c, 0x63, 0x68, 0x65, 0x63, 0x6b, 0x73, 0x75, 0x6d, 0x5f, 0x6d, 0x64, 0x35,
	0x18, 0x03, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0b, 0x63, 0x68, 0x65, 0x63, 0x6b, 0x73, 0x75, 0x6d,
	0x4d, 0x64, 0x35, 0x12, 0x1d, 0x0a, 0x0a, 0x63, 0x72, 0x65, 0x61, 0x74, 0x65, 0x64, 0x5f, 0x61,
	0x74, 0x18, 0x04, 0x20, 0x01, 0x28, 0x03, 0x52, 0x09, 0x63, 0x72, 0x65, 0x61, 0x74, 0x65, 0x64,
	0x41, 0x74, 0x12, 0x1d, 0x0a, 0x0a, 0x75, 0x70, 0x64, 0x61, 0x74, 0x65, 0x64, 0x5f, 0x61, 0x74,
	0x18, 0x05, 0x20, 0x01, 0x28, 0x03, 0x52, 0x09, 0x75, 0x70, 0x64, 0x61, 0x74, 0x65, 0x64, 0x41,
	0x74, 0x12, 0x3a, 0x0a, 0x08, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x18, 0x06, 0x20,
	0x03, 0x28, 0x0b, 0x32, 0x1e, 0x2e, 0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2e, 0x46, 0x69, 0x6c,
	0x65, 0x49, 0x6e, 0x66, 0x6f, 0x2e, 0x4d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x45, 0x6e,
	0x74, 0x72, 0x79, 0x52, 0x08, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x12, 0x29, 0x0a,
	0x06, 0x73, 0x68, 0x61, 0x72, 0x64, 0x73, 0x18, 0x07, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x11, 0x2e,
	0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2e, 0x53, 0x68, 0x61, 0x72, 0x64, 0x49, 0x6e, 0x66, 0x6f,
	0x52, 0x06, 0x73, 0x68, 0x61, 0x72, 0x64, 0x73, 0x12, 0x2d, 0x0a, 0x12, 0x72, 0x65, 0x70, 0x6c,
	0x69, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x66, 0x61, 0x63, 0x74, 0x6f, 0x72, 0x18, 0x08,
	0x20, 0x01, 0x28, 0x05, 0x52, 0x11, 0x72, 0x65, 0x70, 0x6c, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6f,
	0x6e, 0x46, 0x61, 0x63, 0x74, 0x6f, 0x72, 0x12, 0x16, 0x0a, 0x06, 0x73, 0x74, 0x61, 0x74, 0x75,
	0x73, 0x18, 0x09, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06, 0x73, 0x74, 0x61, 0x74, 0x75, 0x73, 0x1a,
	0x3b, 0x0a, 0x0d, 0x4d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x45, 0x6e, 0x74, 0x72, 0x79,
	0x12, 0x10, 0x0a, 0x03, 0x6b, 0x65, 0x79, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x03, 0x6b,
	0x65, 0x79, 0x12, 0x14, 0x0a, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28,
	0x09, 0x52, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x3a, 0x02, 0x38, 0x01, 0x22, 0xb0, 0x01, 0x0a,
	0x09, 0x53, 0x68, 0x61, 0x72, 0x64, 0x49, 0x6e, 0x66, 0x6f, 0x12, 0x19, 0x0a, 0x08, 0x73, 0x68,
	0x61, 0x72, 0x64, 0x5f, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x07, 0x73, 0x68,
	0x61, 0x72, 0x64, 0x49, 0x64, 0x12, 0x12, 0x0a, 0x04, 0x73, 0x69, 0x7a, 0x65, 0x18, 0x02, 0x20,
	0x01, 0x28, 0x03, 0x52, 0x04, 0x73, 0x69, 0x7a, 0x65, 0x12, 0x21, 0x0a, 0x0c, 0x63, 0x68, 0x65,
	0x63, 0x6b, 0x73, 0x75, 0x6d, 0x5f, 0x6d, 0x64, 0x35, 0x18, 0x03, 0x20, 0x01, 0x28, 0x09, 0x52,
	0x0b, 0x63, 0x68, 0x65, 0x63, 0x6b, 0x73, 0x75, 0x6d, 0x4d, 0x64, 0x35, 0x12, 0x23, 0x0a, 0x0d,
	0x73, 0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x5f, 0x6e, 0x6f, 0x64, 0x65, 0x73, 0x18, 0x04, 0x20,
	0x03, 0x28, 0x09, 0x52, 0x0c, 0x73, 0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x4e, 0x6f, 0x64, 0x65,
	0x73, 0x12, 0x16, 0x0a, 0x06, 0x73, 0x74, 0x61, 0x74, 0x75, 0x73, 0x18, 0x05, 0x20, 0x01, 0x28,
	0x09, 0x52, 0x06, 0x73, 0x74, 0x61, 0x74, 0x75, 0x73, 0x12, 0x14, 0x0a, 0x05, 0x6f, 0x72, 0x64,
	0x65, 0x72, 0x18, 0x06, 0x20, 0x01, 0x28, 0x05, 0x52, 0x05, 0x6f, 0x72, 0x64, 0x65, 0x72, 0x22,
	0xa2, 0x02, 0x0a, 0x11, 0x43, 0x72, 0x65, 0x61, 0x74, 0x65, 0x46, 0x69, 0x6c, 0x65, 0x52, 0x65,
	0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x1a, 0x0a, 0x08, 0x66, 0x69, 0x6c, 0x65, 0x6e, 0x61, 0x6d,
	0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x08, 0x66, 0x69, 0x6c, 0x65, 0x6e, 0x61, 0x6d,
	0x65, 0x12, 0x1d, 0x0a, 0x0a, 0x74, 0x6f, 0x74, 0x61, 0x6c, 0x5f, 0x73, 0x69, 0x7a, 0x65, 0x18,
	0x02, 0x20, 0x01, 0x28, 0x03, 0x52, 0x09, 0x74, 0x6f, 0x74, 0x61, 0x6c, 0x53, 0x69, 0x7a, 0x65,
	0x12, 0x21, 0x0a, 0x0c, 0x63, 0x68, 0x65, 0x63, 0x6b, 0x73, 0x75, 0x6d, 0x5f, 0x6d, 0x64, 0x35,
	0x18, 0x03, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0b, 0x63, 0x68, 0x65, 0x63, 0x6b, 0x73, 0x75, 0x6d,
	0x4d, 0x64, 0x35, 0x12, 0x2d, 0x0a, 0x12, 0x72, 0x65, 0x70, 0x6c, 0x69, 0x63, 0x61, 0x74, 0x69,
	0x6f, 0x6e, 0x5f, 0x66, 0x61, 0x63, 0x74, 0x6f, 0x72, 0x18, 0x04, 0x20, 0x01, 0x28, 0x05, 0x52,
	0x11, 0x72, 0x65, 0x70, 0x6c, 0x69, 0x63, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x46, 0x61, 0x63, 0x74,
	0x6f, 0x72, 0x12, 0x43, 0x0a, 0x08, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x18, 0x05,
	0x20, 0x03, 0x28, 0x0b, 0x32, 0x27, 0x2e, 0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2e, 0x43, 0x72,
	0x65, 0x61, 0x74, 0x65, 0x46, 0x69, 0x6c, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x2e,
	0x4d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x45, 0x6e, 0x74, 0x72, 0x79, 0x52, 0x08, 0x6d,
	0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x1a, 0x3b, 0x0a, 0x0d, 0x4d, 0x65, 0x74, 0x61, 0x64,
	0x61, 0x74, 0x61, 0x45, 0x6e, 0x74, 0x72, 0x79, 0x12, 0x10, 0x0a, 0x03, 0x6b, 0x65, 0x79, 0x18,
	0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x03, 0x6b, 0x65, 0x79, 0x12, 0x14, 0x0a, 0x05, 0x76, 0x61,
	0x6c, 0x75, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65,
	0x3a, 0x02, 0x38, 0x01, 0x22, 0x68, 0x0a, 0x12, 0x43, 0x72, 0x65, 0x61, 0x74, 0x65, 0x46, 0x69,
	0x6c, 0x65, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x2d, 0x0a, 0x09, 0x66, 0x69,
	0x6c, 0x65, 0x5f, 0x69, 0x6e, 0x66, 0x6f, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x10, 0x2e,
	0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2e, 0x46, 0x69, 0x6c, 0x65, 0x49, 0x6e, 0x66, 0x6f, 0x52,
	0x08, 0x66, 0x69, 0x6c, 0x65, 0x49, 0x6e, 0x66, 0x6f, 0x12, 0x23, 0x0a, 0x0d, 0x65, 0x72, 0x72,
	0x6f, 0x72, 0x5f, 0x6d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09,
	0x52, 0x0c, 0x65, 0x72, 0x72, 0x6f, 0x72, 0x4d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x22, 0x2f,
	0x0a, 0x11, 0x44, 0x65, 0x6c, 0x65, 0x74, 0x65, 0x46, 0x69, 0x6c, 0x65, 0x52, 0x65, 0x71, 0x75,
	0x65, 0x73, 0x74, 0x12, 0x1a, 0x0a, 0x08, 0x66, 0x69, 0x6c, 0x65, 0x6e, 0x61, 0x6d, 0x65, 0x18,
	0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x08, 0x66, 0x69, 0x6c, 0x65, 0x6e, 0x61, 0x6d, 0x65, 0x22,
	0x53, 0x0a, 0x12, 0x44, 0x65, 0x6c, 0x65, 0x74, 0x65, 0x46, 0x69, 0x6c, 0x65, 0x52, 0x65, 0x73,
	0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x18, 0x0a, 0x07, 0x73, 0x75, 0x63, 0x63, 0x65, 0x73, 0x73,
	0x18, 0x01, 0x20, 0x01, 0x28, 0x08, 0x52, 0x07, 0x73, 0x75, 0x63, 0x63, 0x65, 0x73, 0x73, 0x12,
	0x23, 0x0a, 0x0d, 0x65, 0x72, 0x72, 0x6f, 0x72, 0x5f, 0x6d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65,
	0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0c, 0x65, 0x72, 0x72, 0x6f, 0x72, 0x4d, 0x65, 0x73,
	0x73, 0x61, 0x67, 0x65, 0x22, 0x30, 0x0a, 0x12, 0x47, 0x65, 0x74, 0x46, 0x69, 0x6c, 0x65, 0x49,
	0x6e, 0x66, 0x6f, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x1a, 0x0a, 0x08, 0x66, 0x69,
	0x6c, 0x65, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x08, 0x66, 0x69,
	0x6c, 0x65, 0x6e, 0x61, 0x6d, 0x65, 0x22, 0x69, 0x0a, 0x13, 0x47, 0x65, 0x74, 0x46, 0x69, 0x6c,
	0x65, 0x49, 0x6e, 0x66, 0x6f, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x2d, 0x0a,
	0x09, 0x66, 0x69, 0x6c, 0x65, 0x5f, 0x69, 0x6e, 0x66, 0x6f, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b,
	0x32, 0x10, 0x2e, 0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2e, 0x46, 0x69, 0x6c, 0x65, 0x49, 0x6e,
	0x66, 0x6f, 0x52, 0x08, 0x66, 0x69, 0x6c, 0x65, 0x49, 0x6e, 0x66, 0x6f, 0x12, 0x23, 0x0a, 0x0d,
	0x65, 0x72, 0x72, 0x6f, 0x72, 0x5f, 0x6d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x18, 0x02, 0x20,
	0x01, 0x28, 0x09, 0x52, 0x0c, 0x65, 0x72, 0x72, 0x6f, 0x72, 0x4d, 0x65, 0x73, 0x73, 0x61, 0x67,
	0x65, 0x22, 0x66, 0x0a, 0x10, 0x4c, 0x69, 0x73, 0x74, 0x46, 0x69, 0x6c, 0x65, 0x73, 0x52, 0x65,
	0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x16, 0x0a, 0x06, 0x70, 0x72, 0x65, 0x66, 0x69, 0x78, 0x18,
	0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06, 0x70, 0x72, 0x65, 0x66, 0x69, 0x78, 0x12, 0x1b, 0x0a,
	0x09, 0x70, 0x61, 0x67, 0x65, 0x5f, 0x73, 0x69, 0x7a, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x05,
	0x52, 0x08, 0x70, 0x61, 0x67, 0x65, 0x53, 0x69, 0x7a, 0x65, 0x12, 0x1d, 0x0a, 0x0a, 0x70, 0x61,
	0x67, 0x65, 0x5f, 0x74, 0x6f, 0x6b, 0x65, 0x6e, 0x18, 0x03, 0x20, 0x01, 0x28, 0x09, 0x52, 0x09,
	0x70, 0x61, 0x67, 0x65, 0x54, 0x6f, 0x6b, 0x65, 0x6e, 0x22, 0x88, 0x01, 0x0a, 0x11, 0x4c, 0x69,
	0x73, 0x74, 0x46, 0x69, 0x6c, 0x65, 0x73, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12,
	0x26, 0x0a, 0x05, 0x66, 0x69, 0x6c, 0x65, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x10,
	0x2e, 0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2e, 0x46, 0x69, 0x6c, 0x65, 0x49, 0x6e, 0x66, 0x6f,
	0x52, 0x05, 0x66, 0x69, 0x6c, 0x65, 0x73, 0x12, 0x26, 0x0a, 0x0f, 0x6e, 0x65, 0x78, 0x74, 0x5f,
	0x70, 0x61, 0x67, 0x65, 0x5f, 0x74, 0x6f, 0x6b, 0x65, 0x6e, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09,
	0x52, 0x0d, 0x6e, 0x65, 0x78, 0x74, 0x50, 0x61, 0x67, 0x65, 0x54, 0x6f, 0x6b, 0x65, 0x6e, 0x12,
	0x23, 0x0a, 0x0d, 0x65, 0x72, 0x72, 0x6f, 0x72, 0x5f, 0x6d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65,
	0x18, 0x03, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0c, 0x65, 0x72, 0x72, 0x6f, 0x72, 0x4d, 0x65, 0x73,
	0x73, 0x61, 0x67, 0x65, 0x22, 0x99, 0x02, 0x0a, 0x12, 0x55, 0x70, 0x6c, 0x6f, 0x61, 0x64, 0x53,
	0x68, 0x61, 0x72, 0x64, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x41, 0x0a, 0x08, 0x6d,
	0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x23, 0x2e,
	0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2e, 0x55, 0x70, 0x6c, 0x6f, 0x61, 0x64, 0x53, 0x68, 0x61,
	0x72, 0x64, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x2e, 0x4d, 0x65, 0x74, 0x61, 0x64, 0x61,
	0x74, 0x61, 0x48, 0x00, 0x52, 0x08, 0x6d, 0x65, 0x74, 0x61, 0x64, 0x61, 0x74, 0x61, 0x12, 0x16,
	0x0a, 0x05, 0x63, 0x68, 0x75, 0x6e, 0x6b, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0c, 0x48, 0x00, 0x52,
	0x05, 0x63, 0x68, 0x75, 0x6e, 0x6b, 0x1a, 0x99, 0x01, 0x0a, 0x08, 0x4d, 0x65, 0x74, 0x61, 0x64,
	0x61, 0x74, 0x61, 0x12, 0x1a, 0x0a, 0x08, 0x66, 0x69, 0x6c, 0x65, 0x6e, 0x61, 0x6d, 0x65, 0x18,
	0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x08, 0x66, 0x69, 0x6c, 0x65, 0x6e, 0x61, 0x6d, 0x65, 0x12,
	0x19, 0x0a, 0x08, 0x73, 0x68, 0x61, 0x72, 0x64, 0x5f, 0x69, 0x64, 0x18, 0x02, 0x20, 0x01, 0x28,
	0x09, 0x52, 0x07, 0x73, 0x68, 0x61, 0x72, 0x64, 0x49, 0x64, 0x12, 0x1d, 0x0a, 0x0a, 0x73, 0x68,
	0x61, 0x72, 0x64, 0x5f, 0x73, 0x69, 0x7a, 0x65, 0x18, 0x03, 0x20, 0x01, 0x28, 0x03, 0x52, 0x09,
	0x73, 0x68, 0x61, 0x72, 0x64, 0x53, 0x69, 0x7a, 0x65, 0x12, 0x21, 0x0a, 0x0c, 0x63, 0x68, 0x65,
	0x63, 0x6b, 0x73, 0x75, 0x6d, 0x5f, 0x6d, 0x64, 0x35, 0x18, 0x04, 0x20, 0x01, 0x28, 0x09, 0x52,
	0x0b, 0x63, 0x68, 0x65, 0x63, 0x6b, 0x73, 0x75, 0x6d, 0x4d, 0x64, 0x35, 0x12, 0x14, 0x0a, 0x05,
	0x6f, 0x72, 0x64, 0x65, 0x72, 0x18, 0x05, 0x20, 0x01, 0x28, 0x05, 0x52, 0x05, 0x6f, 0x72, 0x64,
	0x65, 0x72, 0x42, 0x0c, 0x0a, 0x0a, 0x64, 0x61, 0x74, 0x61, 0x5f, 0x6f, 0x6e, 0x65, 0x6f, 0x66,
	0x22, 0x8b, 0x01, 0x0a, 0x13, 0x55, 0x70, 0x6c, 0x6f, 0x61, 0x64, 0x53, 0x68, 0x61, 0x72, 0x64,
	0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x1a, 0x0a, 0x08, 0x66, 0x69, 0x6c, 0x65,
	0x6e, 0x61, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x08, 0x66, 0x69, 0x6c, 0x65,
	0x6e, 0x61, 0x6d, 0x65, 0x12, 0x19, 0x0a, 0x08, 0x73, 0x68, 0x61, 0x72, 0x64, 0x5f, 0x69, 0x64,
	0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x07, 0x73, 0x68, 0x61, 0x72, 0x64, 0x49, 0x64, 0x12,
	0x18, 0x0a, 0x07, 0x73, 0x75, 0x63, 0x63, 0x65, 0x73, 0x73, 0x18, 0x03, 0x20, 0x01, 0x28, 0x08,
	0x52, 0x07, 0x73, 0x75, 0x63, 0x63, 0x65, 0x73, 0x73, 0x12, 0x23, 0x0a, 0x0d, 0x65, 0x72, 0x72,
	0x6f, 0x72, 0x5f, 0x6d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x18, 0x04, 0x20, 0x01, 0x28, 0x09,
	0x52, 0x0c, 0x65, 0x72, 0x72, 0x6f, 0x72, 0x4d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x22, 0x7d,
	0x0a, 0x14, 0x44, 0x6f, 0x77, 0x6e, 0x6c, 0x6f, 0x61, 0x64, 0x53, 0x68, 0x61, 0x72, 0x64, 0x52,
	0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x1a, 0x0a, 0x08, 0x66, 0x69, 0x6c, 0x65, 0x6e, 0x61,
	0x6d, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x08, 0x66, 0x69, 0x6c, 0x65, 0x6e, 0x61,
	0x6d, 0x65, 0x12, 0x19, 0x0a, 0x08, 0x73, 0x68, 0x61, 0x72, 0x64, 0x5f, 0x69, 0x64, 0x18, 0x02,
	0x20, 0x01, 0x28, 0x09, 0x52, 0x07, 0x73, 0x68, 0x61, 0x72, 0x64, 0x49, 0x64, 0x12, 0x16, 0x0a,
	0x06, 0x6f, 0x66, 0x66, 0x73, 0x65, 0x74, 0x18, 0x03, 0x20, 0x01, 0x28, 0x03, 0x52, 0x06, 0x6f,
	0x66, 0x66, 0x73, 0x65, 0x74, 0x12, 0x16, 0x0a, 0x06, 0x6c, 0x65, 0x6e, 0x67, 0x74, 0x68, 0x18,
	0x04, 0x20, 0x01, 0x28, 0x03, 0x52, 0x06, 0x6c, 0x65, 0x6e, 0x67, 0x74, 0x68, 0x22, 0x52, 0x0a,
	0x15, 0x44, 0x6f, 0x77, 0x6e, 0x6c, 0x6f, 0x61, 0x64, 0x53, 0x68, 0x61, 0x72, 0x64, 0x52, 0x65,
	0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x14, 0x0a, 0x05, 0x63, 0x68, 0x75, 0x6e, 0x6b, 0x18,
	0x01, 0x20, 0x01, 0x28, 0x0c, 0x52, 0x05, 0x63, 0x68, 0x75, 0x6e, 0x6b, 0x12, 0x23, 0x0a, 0x0d,
	0x65, 0x72, 0x72, 0x6f, 0x72, 0x5f, 0x6d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x18, 0x02, 0x20,
	0x01, 0x28, 0x09, 0x52, 0x0c, 0x65, 0x72, 0x72, 0x6f, 0x72, 0x4d, 0x65, 0x73, 0x73, 0x61, 0x67,
	0x65, 0x22, 0x6b, 0x0a, 0x19, 0x44, 0x65, 0x6c, 0x65, 0x74, 0x65, 0x53, 0x68, 0x61, 0x72, 0x64,
	0x53, 0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x1a,
	0x0a, 0x08, 0x66, 0x69, 0x6c, 0x65, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09,
	0x52, 0x08, 0x66, 0x69, 0x6c, 0x65, 0x6e, 0x61, 0x6d, 0x65, 0x12, 0x19, 0x0a, 0x08, 0x73, 0x68,
	0x61, 0x72, 0x64, 0x5f, 0x69, 0x64, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x07, 0x73, 0x68,
	0x61, 0x72, 0x64, 0x49, 0x64, 0x12, 0x17, 0x0a, 0x07, 0x6e, 0x6f, 0x64, 0x65, 0x5f, 0x69, 0x64,
	0x18, 0x03, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06, 0x6e, 0x6f, 0x64, 0x65, 0x49, 0x64, 0x22, 0x5b,
	0x0a, 0x1a, 0x44, 0x65, 0x6c, 0x65, 0x74, 0x65, 0x53, 0x68, 0x61, 0x72, 0x64, 0x53, 0x74, 0x6f,
	0x72, 0x61, 0x67, 0x65, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x18, 0x0a, 0x07,
	0x73, 0x75, 0x63, 0x63, 0x65, 0x73, 0x73, 0x18, 0x01, 0x20, 0x01, 0x28, 0x08, 0x52, 0x07, 0x73,
	0x75, 0x63, 0x63, 0x65, 0x73, 0x73, 0x12, 0x23, 0x0a, 0x0d, 0x65, 0x72, 0x72, 0x6f, 0x72, 0x5f,
	0x6d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0c, 0x65,
	0x72, 0x72, 0x6f, 0x72, 0x4d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x22, 0x97, 0x01, 0x0a, 0x13,
	0x52, 0x65, 0x67, 0x69, 0x73, 0x74, 0x65, 0x72, 0x4e, 0x6f, 0x64, 0x65, 0x52, 0x65, 0x71, 0x75,
	0x65, 0x73, 0x74, 0x12, 0x17, 0x0a, 0x07, 0x6e, 0x6f, 0x64, 0x65, 0x5f, 0x69, 0x64, 0x18, 0x01,
	0x20, 0x01, 0x28, 0x09, 0x52, 0x06, 0x6e, 0x6f, 0x64, 0x65, 0x49, 0x64, 0x12, 0x18, 0x0a, 0x07,
	0x61, 0x64, 0x64, 0x72, 0x65, 0x73, 0x73, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x07, 0x61,
	0x64, 0x64, 0x72, 0x65, 0x73, 0x73, 0x12, 0x1a, 0x0a, 0x08, 0x63, 0x61, 0x70, 0x61, 0x63, 0x69,
	0x74, 0x79, 0x18, 0x03, 0x20, 0x01, 0x28, 0x03, 0x52, 0x08, 0x63, 0x61, 0x70, 0x61, 0x63, 0x69,
	0x74, 0x79, 0x12, 0x1d, 0x0a, 0x0a, 0x75, 0x73, 0x65, 0x64, 0x5f, 0x73, 0x70, 0x61, 0x63, 0x65,
	0x18, 0x04, 0x20, 0x01, 0x28, 0x03, 0x52, 0x09, 0x75, 0x73, 0x65, 0x64, 0x53, 0x70, 0x61, 0x63,
	0x65, 0x12, 0x12, 0x0a, 0x04, 0x74, 0x61, 0x67, 0x73, 0x18, 0x05, 0x20, 0x03, 0x28, 0x09, 0x52,
	0x04, 0x74, 0x61, 0x67, 0x73, 0x22, 0x55, 0x0a, 0x14, 0x52, 0x65, 0x67, 0x69, 0x73, 0x74, 0x65,
	0x72, 0x4e, 0x6f, 0x64, 0x65, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x18, 0x0a,
	0x07, 0x73, 0x75, 0x63, 0x63, 0x65, 0x73, 0x73, 0x18, 0x01, 0x20, 0x01, 0x28, 0x08, 0x52, 0x07,
	0x73, 0x75, 0x63, 0x63, 0x65, 0x73, 0x73, 0x12, 0x23, 0x0a, 0x0d, 0x65, 0x72, 0x72, 0x6f, 0x72,
	0x5f, 0x6d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0c,
	0x65, 0x72, 0x72, 0x6f, 0x72, 0x4d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x22, 0x2f, 0x0a, 0x14,
	0x47, 0x65, 0x74, 0x4e, 0x6f, 0x64, 0x65, 0x53, 0x74, 0x61, 0x74, 0x75, 0x73, 0x52, 0x65, 0x71,
	0x75, 0x65, 0x73, 0x74, 0x12, 0x17, 0x0a, 0x07, 0x6e, 0x6f, 0x64, 0x65, 0x5f, 0x69, 0x64, 0x18,
	0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06, 0x6e, 0x6f, 0x64, 0x65, 0x49, 0x64, 0x22, 0xf0, 0x01,
	0x0a, 0x15, 0x47, 0x65, 0x74, 0x4e, 0x6f, 0x64, 0x65, 0x53, 0x74, 0x61, 0x74, 0x75, 0x73, 0x52,
	0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x17, 0x0a, 0x07, 0x6e, 0x6f, 0x64, 0x65, 0x5f,
	0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06, 0x6e, 0x6f, 0x64, 0x65, 0x49, 0x64,
	0x12, 0x16, 0x0a, 0x06, 0x73, 0x74, 0x61, 0x74, 0x75, 0x73, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09,
	0x52, 0x06, 0x73, 0x74, 0x61, 0x74, 0x75, 0x73, 0x12, 0x1a, 0x0a, 0x08, 0x63, 0x61, 0x70, 0x61,
	0x63, 0x69, 0x74, 0x79, 0x18, 0x03, 0x20, 0x01, 0x28, 0x03, 0x52, 0x08, 0x63, 0x61, 0x70, 0x61,
	0x63, 0x69, 0x74, 0x79, 0x12, 0x1d, 0x0a, 0x0a, 0x75, 0x73, 0x65, 0x64, 0x5f, 0x73, 0x70, 0x61,
	0x63, 0x65, 0x18, 0x04, 0x20, 0x01, 0x28, 0x03, 0x52, 0x09, 0x75, 0x73, 0x65, 0x64, 0x53, 0x70,
	0x61, 0x63, 0x65, 0x12, 0x1f, 0x0a, 0x0b, 0x6c, 0x6f, 0x61, 0x64, 0x5f, 0x66, 0x61, 0x63, 0x74,
	0x6f, 0x72, 0x18, 0x05, 0x20, 0x01, 0x28, 0x02, 0x52, 0x0a, 0x6c, 0x6f, 0x61, 0x64, 0x46, 0x61,
	0x63, 0x74, 0x6f, 0x72, 0x12, 0x25, 0x0a, 0x0e, 0x6c, 0x61, 0x73, 0x74, 0x5f, 0x68, 0x65, 0x61,
	0x72, 0x74, 0x62, 0x65, 0x61, 0x74, 0x18, 0x06, 0x20, 0x01, 0x28, 0x03, 0x52, 0x0d, 0x6c, 0x61,
	0x73, 0x74, 0x48, 0x65, 0x61, 0x72, 0x74, 0x62, 0x65, 0x61, 0x74, 0x12, 0x23, 0x0a, 0x0d, 0x65,
	0x72, 0x72, 0x6f, 0x72, 0x5f, 0x6d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x18, 0x07, 0x20, 0x01,
	0x28, 0x09, 0x52, 0x0c, 0x65, 0x72, 0x72, 0x6f, 0x72, 0x4d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65,
	0x32, 0x9f, 0x04, 0x0a, 0x19, 0x44, 0x69, 0x73, 0x74, 0x72, 0x69, 0x62, 0x75, 0x74, 0x65, 0x64,
	0x53, 0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x53, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x12, 0x43,
	0x0a, 0x0a, 0x43, 0x72, 0x65, 0x61, 0x74, 0x65, 0x46, 0x69, 0x6c, 0x65, 0x12, 0x19, 0x2e, 0x73,
	0x65, 0x72, 0x76, 0x65, 0x72, 0x2e, 0x43, 0x72, 0x65, 0x61, 0x74, 0x65, 0x46, 0x69, 0x6c, 0x65,
	0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1a, 0x2e, 0x73, 0x65, 0x72, 0x76, 0x65, 0x72,
	0x2e, 0x43, 0x72, 0x65, 0x61, 0x74, 0x65, 0x46, 0x69, 0x6c, 0x65, 0x52, 0x65, 0x73, 0x70, 0x6f,
	0x6e, 0x73, 0x65, 0x12, 0x43, 0x0a, 0x0a, 0x44, 0x65, 0x6c, 0x65, 0x74, 0x65, 0x46, 0x69, 0x6c,
	0x65, 0x12, 0x19, 0x2e, 0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2e, 0x44, 0x65, 0x6c, 0x65, 0x74,
	0x65, 0x46, 0x69, 0x6c, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1a, 0x2e, 0x73,
	0x65, 0x72, 0x76, 0x65, 0x72, 0x2e, 0x44, 0x65, 0x6c, 0x65, 0x74, 0x65, 0x46, 0x69, 0x6c, 0x65,
	0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x46, 0x0a, 0x0b, 0x47, 0x65, 0x74, 0x46,
	0x69, 0x6c, 0x65, 0x49, 0x6e, 0x66, 0x6f, 0x12, 0x1a, 0x2e, 0x73, 0x65, 0x72, 0x76, 0x65, 0x72,
	0x2e, 0x47, 0x65, 0x74, 0x46, 0x69, 0x6c, 0x65, 0x49, 0x6e, 0x66, 0x6f, 0x52, 0x65, 0x71, 0x75,
	0x65, 0x73, 0x74, 0x1a, 0x1b, 0x2e, 0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2e, 0x47, 0x65, 0x74,
	0x46, 0x69, 0x6c, 0x65, 0x49, 0x6e, 0x66, 0x6f, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65,
	0x12, 0x40, 0x0a, 0x09, 0x4c, 0x69, 0x73, 0x74, 0x46, 0x69, 0x6c, 0x65, 0x73, 0x12, 0x18, 0x2e,
	0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2e, 0x4c, 0x69, 0x73, 0x74, 0x46, 0x69, 0x6c, 0x65, 0x73,
	0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x19, 0x2e, 0x73, 0x65, 0x72, 0x76, 0x65, 0x72,
	0x2e, 0x4c, 0x69, 0x73, 0x74, 0x46, 0x69, 0x6c, 0x65, 0x73, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e,
	0x73, 0x65, 0x12, 0x48, 0x0a, 0x0b, 0x55, 0x70, 0x6c, 0x6f, 0x61, 0x64, 0x53, 0x68, 0x61, 0x72,
	0x64, 0x12, 0x1a, 0x2e, 0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2e, 0x55, 0x70, 0x6c, 0x6f, 0x61,
	0x64, 0x53, 0x68, 0x61, 0x72, 0x64, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1b, 0x2e,
	0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2e, 0x55, 0x70, 0x6c, 0x6f, 0x61, 0x64, 0x53, 0x68, 0x61,
	0x72, 0x64, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x28, 0x01, 0x12, 0x4e, 0x0a, 0x0d,
	0x44, 0x6f, 0x77, 0x6e, 0x6c, 0x6f, 0x61, 0x64, 0x53, 0x68, 0x61, 0x72, 0x64, 0x12, 0x1c, 0x2e,
	0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2e, 0x44, 0x6f, 0x77, 0x6e, 0x6c, 0x6f, 0x61, 0x64, 0x53,
	0x68, 0x61, 0x72, 0x64, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1d, 0x2e, 0x73, 0x65,
	0x72, 0x76, 0x65, 0x72, 0x2e, 0x44, 0x6f, 0x77, 0x6e, 0x6c, 0x6f, 0x61, 0x64, 0x53, 0x68, 0x61,
	0x72, 0x64, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x30, 0x01, 0x12, 0x54, 0x0a, 0x0b,
	0x44, 0x65, 0x6c, 0x65, 0x74, 0x65, 0x53, 0x68, 0x61, 0x72, 0x64, 0x12, 0x21, 0x2e, 0x73, 0x65,
	0x72, 0x76, 0x65, 0x72, 0x2e, 0x44, 0x65, 0x6c, 0x65, 0x74, 0x65, 0x53, 0x68, 0x61, 0x72, 0x64,
	0x53, 0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x22,
	0x2e, 0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2e, 0x44, 0x65, 0x6c, 0x65, 0x74, 0x65, 0x53, 0x68,
	0x61, 0x72, 0x64, 0x53, 0x74, 0x6f, 0x72, 0x61, 0x67, 0x65, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e,
	0x73, 0x65, 0x42, 0x1f, 0x5a, 0x1d, 0x56, 0x65, 0x63, 0x74, 0x6f, 0x72, 0x53, 0x70, 0x68, 0x65,
	0x72, 0x65, 0x2f, 0x73, 0x72, 0x63, 0x2f, 0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2f, 0x70, 0x72,
	0x6f, 0x74, 0x6f, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_distributed_storage_proto_rawDescOnce sync.Once
	file_distributed_storage_proto_rawDescData = file_distributed_storage_proto_rawDesc
)

func file_distributed_storage_proto_rawDescGZIP() []byte {
	file_distributed_storage_proto_rawDescOnce.Do(func() {
		file_distributed_storage_proto_rawDescData = protoimpl.X.CompressGZIP(file_distributed_storage_proto_rawDescData)
	})
	return file_distributed_storage_proto_rawDescData
}

var file_distributed_storage_proto_msgTypes = make([]protoimpl.MessageInfo, 23)
var file_distributed_storage_proto_goTypes = []interface{}{
	(*FileInfo)(nil),                    // 0: server.FileInfo
	(*ShardInfo)(nil),                   // 1: server.ShardInfo
	(*CreateFileRequest)(nil),           // 2: server.CreateFileRequest
	(*CreateFileResponse)(nil),          // 3: server.CreateFileResponse
	(*DeleteFileRequest)(nil),           // 4: server.DeleteFileRequest
	(*DeleteFileResponse)(nil),          // 5: server.DeleteFileResponse
	(*GetFileInfoRequest)(nil),          // 6: server.GetFileInfoRequest
	(*GetFileInfoResponse)(nil),         // 7: server.GetFileInfoResponse
	(*ListFilesRequest)(nil),            // 8: server.ListFilesRequest
	(*ListFilesResponse)(nil),           // 9: server.ListFilesResponse
	(*UploadShardRequest)(nil),          // 10: server.UploadShardRequest
	(*UploadShardResponse)(nil),         // 11: server.UploadShardResponse
	(*DownloadShardRequest)(nil),        // 12: server.DownloadShardRequest
	(*DownloadShardResponse)(nil),       // 13: server.DownloadShardResponse
	(*DeleteShardStorageRequest)(nil),   // 14: server.DeleteShardStorageRequest
	(*DeleteShardStorageResponse)(nil),  // 15: server.DeleteShardStorageResponse
	(*RegisterNodeRequest)(nil),         // 16: server.RegisterNodeRequest
	(*RegisterNodeResponse)(nil),        // 17: server.RegisterNodeResponse
	(*GetNodeStatusRequest)(nil),        // 18: server.GetNodeStatusRequest
	(*GetNodeStatusResponse)(nil),       // 19: server.GetNodeStatusResponse
	nil,                                 // 20: server.FileInfo.MetadataEntry
	nil,                                 // 21: server.CreateFileRequest.MetadataEntry
	(*UploadShardRequest_Metadata)(nil), // 22: server.UploadShardRequest.Metadata
}
var file_distributed_storage_proto_depIdxs = []int32{
	20, // 0: server.FileInfo.metadata:type_name -> server.FileInfo.MetadataEntry
	1,  // 1: server.FileInfo.shards:type_name -> server.ShardInfo
	21, // 2: server.CreateFileRequest.metadata:type_name -> server.CreateFileRequest.MetadataEntry
	0,  // 3: server.CreateFileResponse.file_info:type_name -> server.FileInfo
	0,  // 4: server.GetFileInfoResponse.file_info:type_name -> server.FileInfo
	0,  // 5: server.ListFilesResponse.files:type_name -> server.FileInfo
	22, // 6: server.UploadShardRequest.metadata:type_name -> server.UploadShardRequest.Metadata
	2,  // 7: server.DistributedStorageService.CreateFile:input_type -> server.CreateFileRequest
	4,  // 8: server.DistributedStorageService.DeleteFile:input_type -> server.DeleteFileRequest
	6,  // 9: server.DistributedStorageService.GetFileInfo:input_type -> server.GetFileInfoRequest
	8,  // 10: server.DistributedStorageService.ListFiles:input_type -> server.ListFilesRequest
	10, // 11: server.DistributedStorageService.UploadShard:input_type -> server.UploadShardRequest
	12, // 12: server.DistributedStorageService.DownloadShard:input_type -> server.DownloadShardRequest
	14, // 13: server.DistributedStorageService.DeleteShard:input_type -> server.DeleteShardStorageRequest
	3,  // 14: server.DistributedStorageService.CreateFile:output_type -> server.CreateFileResponse
	5,  // 15: server.DistributedStorageService.DeleteFile:output_type -> server.DeleteFileResponse
	7,  // 16: server.DistributedStorageService.GetFileInfo:output_type -> server.GetFileInfoResponse
	9,  // 17: server.DistributedStorageService.ListFiles:output_type -> server.ListFilesResponse
	11, // 18: server.DistributedStorageService.UploadShard:output_type -> server.UploadShardResponse
	13, // 19: server.DistributedStorageService.DownloadShard:output_type -> server.DownloadShardResponse
	15, // 20: server.DistributedStorageService.DeleteShard:output_type -> server.DeleteShardStorageResponse
	14, // [14:21] is the sub-list for method output_type
	7,  // [7:14] is the sub-list for method input_type
	7,  // [7:7] is the sub-list for extension type_name
	7,  // [7:7] is the sub-list for extension extendee
	0,  // [0:7] is the sub-list for field type_name
}

func init() { fileDistributedStorageProtoInit() }
func fileDistributedStorageProtoInit() {
	if File_distributed_storage_proto != nil {
		return
	}

	file_distributed_storage_proto_msgTypes[10].OneofWrappers = []interface{}{
		(*UploadShardRequest_Metadata_)(nil),
		(*UploadShardRequest_Chunk)(nil),
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_distributed_storage_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   23,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_distributed_storage_proto_goTypes,
		DependencyIndexes: file_distributed_storage_proto_depIdxs,
		MessageInfos:      file_distributed_storage_proto_msgTypes,
	}.Build()
	File_distributed_storage_proto = out.File
	file_distributed_storage_proto_rawDesc = nil
	file_distributed_storage_proto_goTypes = nil
	file_distributed_storage_proto_depIdxs = nil
}

// Reference imports to suppress errors if they are not otherwise used.
var _ context.Context
var _ grpc.ClientConnInterface

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
const _ = grpc.SupportPackageIsVersion6

// DistributedStorageServiceClient is the client API for DistributedStorageService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type DistributedStorageServiceClient interface {
	// 文件操作
	CreateFile(ctx context.Context, in *CreateFileRequest, opts ...grpc.CallOption) (*CreateFileResponse, error)
	DeleteFile(ctx context.Context, in *DeleteFileRequest, opts ...grpc.CallOption) (*DeleteFileResponse, error)
	GetFileInfo(ctx context.Context, in *GetFileInfoRequest, opts ...grpc.CallOption) (*GetFileInfoResponse, error)
	ListFiles(ctx context.Context, in *ListFilesRequest, opts ...grpc.CallOption) (*ListFilesResponse, error)
	// 分片操作
	UploadShard(ctx context.Context, opts ...grpc.CallOption) (DistributedStorageService_UploadShardClient, error)
	DownloadShard(ctx context.Context, in *DownloadShardRequest, opts ...grpc.CallOption) (DistributedStorageService_DownloadShardClient, error)
	DeleteShard(ctx context.Context, in *DeleteShardStorageRequest, opts ...grpc.CallOption) (*DeleteShardStorageResponse, error)
}

type distributedStorageServiceClient struct {
	cc grpc.ClientConnInterface
}

func NewDistributedStorageServiceClient(cc grpc.ClientConnInterface) DistributedStorageServiceClient {
	return &distributedStorageServiceClient{cc}
}

func (c *distributedStorageServiceClient) CreateFile(ctx context.Context, in *CreateFileRequest, opts ...grpc.CallOption) (*CreateFileResponse, error) {
	out := new(CreateFileResponse)
	err := c.cc.Invoke(ctx, "/server.DistributedStorageService/CreateFile", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *distributedStorageServiceClient) DeleteFile(ctx context.Context, in *DeleteFileRequest, opts ...grpc.CallOption) (*DeleteFileResponse, error) {
	out := new(DeleteFileResponse)
	err := c.cc.Invoke(ctx, "/server.DistributedStorageService/DeleteFile", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *distributedStorageServiceClient) GetFileInfo(ctx context.Context, in *GetFileInfoRequest, opts ...grpc.CallOption) (*GetFileInfoResponse, error) {
	out := new(GetFileInfoResponse)
	err := c.cc.Invoke(ctx, "/server.DistributedStorageService/GetFileInfo", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *distributedStorageServiceClient) ListFiles(ctx context.Context, in *ListFilesRequest, opts ...grpc.CallOption) (*ListFilesResponse, error) {
	out := new(ListFilesResponse)
	err := c.cc.Invoke(ctx, "/server.DistributedStorageService/ListFiles", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *distributedStorageServiceClient) UploadShard(ctx context.Context, opts ...grpc.CallOption) (DistributedStorageService_UploadShardClient, error) {
	stream, err := c.cc.NewStream(ctx, &_DistributedStorageService_serviceDesc.Streams[0], "/server.DistributedStorageService/UploadShard", opts...)
	if err != nil {
		return nil, err
	}
	x := &distributedStorageServiceUploadShardClient{stream}
	return x, nil
}

type DistributedStorageService_UploadShardClient interface {
	Send(*UploadShardRequest) error
	CloseAndRecv() (*UploadShardResponse, error)
	grpc.ClientStream
}

type distributedStorageServiceUploadShardClient struct {
	grpc.ClientStream
}

func (x *distributedStorageServiceUploadShardClient) Send(m *UploadShardRequest) error {
	return x.ClientStream.SendMsg(m)
}

func (x *distributedStorageServiceUploadShardClient) CloseAndRecv() (*UploadShardResponse, error) {
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	m := new(UploadShardResponse)
	if err := x.ClientStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

func (c *distributedStorageServiceClient) DownloadShard(ctx context.Context, in *DownloadShardRequest, opts ...grpc.CallOption) (DistributedStorageService_DownloadShardClient, error) {
	stream, err := c.cc.NewStream(ctx, &_DistributedStorageService_serviceDesc.Streams[1], "/server.DistributedStorageService/DownloadShard", opts...)
	if err != nil {
		return nil, err
	}
	x := &distributedStorageServiceDownloadShardClient{stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

type DistributedStorageService_DownloadShardClient interface {
	Recv() (*DownloadShardResponse, error)
	grpc.ClientStream
}

type distributedStorageServiceDownloadShardClient struct {
	grpc.ClientStream
}

func (x *distributedStorageServiceDownloadShardClient) Recv() (*DownloadShardResponse, error) {
	m := new(DownloadShardResponse)
	if err := x.ClientStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

func (c *distributedStorageServiceClient) DeleteShard(ctx context.Context, in *DeleteShardStorageRequest, opts ...grpc.CallOption) (*DeleteShardStorageResponse, error) {
	out := new(DeleteShardStorageResponse)
	err := c.cc.Invoke(ctx, "/server.DistributedStorageService/DeleteShard", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// DistributedStorageServiceServer is the server API for DistributedStorageService service.
type DistributedStorageServiceServer interface {
	// 文件操作
	CreateFile(context.Context, *CreateFileRequest) (*CreateFileResponse, error)
	DeleteFile(context.Context, *DeleteFileRequest) (*DeleteFileResponse, error)
	GetFileInfo(context.Context, *GetFileInfoRequest) (*GetFileInfoResponse, error)
	ListFiles(context.Context, *ListFilesRequest) (*ListFilesResponse, error)
	// 分片操作
	UploadShard(DistributedStorageService_UploadShardServer) error
	DownloadShard(*DownloadShardRequest, DistributedStorageService_DownloadShardServer) error
	DeleteShard(context.Context, *DeleteShardStorageRequest) (*DeleteShardStorageResponse, error)
}

// UnimplementedDistributedStorageServiceServer can be embedded to have forward compatible implementations.
type UnimplementedDistributedStorageServiceServer struct {
}

func (*UnimplementedDistributedStorageServiceServer) CreateFile(context.Context, *CreateFileRequest) (*CreateFileResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method CreateFile not implemented")
}
func (*UnimplementedDistributedStorageServiceServer) DeleteFile(context.Context, *DeleteFileRequest) (*DeleteFileResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method DeleteFile not implemented")
}
func (*UnimplementedDistributedStorageServiceServer) GetFileInfo(context.Context, *GetFileInfoRequest) (*GetFileInfoResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetFileInfo not implemented")
}
func (*UnimplementedDistributedStorageServiceServer) ListFiles(context.Context, *ListFilesRequest) (*ListFilesResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method ListFiles not implemented")
}
func (*UnimplementedDistributedStorageServiceServer) UploadShard(DistributedStorageService_UploadShardServer) error {
	return status.Errorf(codes.Unimplemented, "method UploadShard not implemented")
}
func (*UnimplementedDistributedStorageServiceServer) DownloadShard(*DownloadShardRequest, DistributedStorageService_DownloadShardServer) error {
	return status.Errorf(codes.Unimplemented, "method DownloadShard not implemented")
}
func (*UnimplementedDistributedStorageServiceServer) DeleteShard(context.Context, *DeleteShardStorageRequest) (*DeleteShardStorageResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method DeleteShard not implemented")
}

func RegisterDistributedStorageServiceServer(s *grpc.Server, srv DistributedStorageServiceServer) {
	s.RegisterService(&_DistributedStorageService_serviceDesc, srv)
}

func _DistributedStorageService_CreateFile_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(CreateFileRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(DistributedStorageServiceServer).CreateFile(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/server.DistributedStorageService/CreateFile",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(DistributedStorageServiceServer).CreateFile(ctx, req.(*CreateFileRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _DistributedStorageService_DeleteFile_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DeleteFileRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(DistributedStorageServiceServer).DeleteFile(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/server.DistributedStorageService/DeleteFile",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(DistributedStorageServiceServer).DeleteFile(ctx, req.(*DeleteFileRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _DistributedStorageService_GetFileInfo_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GetFileInfoRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(DistributedStorageServiceServer).GetFileInfo(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/server.DistributedStorageService/GetFileInfo",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(DistributedStorageServiceServer).GetFileInfo(ctx, req.(*GetFileInfoRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _DistributedStorageService_ListFiles_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ListFilesRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(DistributedStorageServiceServer).ListFiles(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/server.DistributedStorageService/ListFiles",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(DistributedStorageServiceServer).ListFiles(ctx, req.(*ListFilesRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _DistributedStorageService_UploadShard_Handler(srv interface{}, stream grpc.ServerStream) error {
	return srv.(DistributedStorageServiceServer).UploadShard(&distributedStorageServiceUploadShardServer{stream})
}

type DistributedStorageService_UploadShardServer interface {
	SendAndClose(*UploadShardResponse) error
	Recv() (*UploadShardRequest, error)
	grpc.ServerStream
}

type distributedStorageServiceUploadShardServer struct {
	grpc.ServerStream
}

func (x *distributedStorageServiceUploadShardServer) SendAndClose(m *UploadShardResponse) error {
	return x.ServerStream.SendMsg(m)
}

func (x *distributedStorageServiceUploadShardServer) Recv() (*UploadShardRequest, error) {
	m := new(UploadShardRequest)
	if err := x.ServerStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

func _DistributedStorageService_DownloadShard_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(DownloadShardRequest)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(DistributedStorageServiceServer).DownloadShard(m, &distributedStorageServiceDownloadShardServer{stream})
}

type DistributedStorageService_DownloadShardServer interface {
	Send(*DownloadShardResponse) error
	grpc.ServerStream
}

type distributedStorageServiceDownloadShardServer struct {
	grpc.ServerStream
}

func (x *distributedStorageServiceDownloadShardServer) Send(m *DownloadShardResponse) error {
	return x.ServerStream.SendMsg(m)
}

func _DistributedStorageService_DeleteShard_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DeleteShardStorageRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(DistributedStorageServiceServer).DeleteShard(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/server.DistributedStorageService/DeleteShard",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(DistributedStorageServiceServer).DeleteShard(ctx, req.(*DeleteShardStorageRequest))
	}
	return interceptor(ctx, in, info, handler)
}

var _DistributedStorageService_serviceDesc = grpc.ServiceDesc{
	ServiceName: "server.DistributedStorageService",
	HandlerType: (*DistributedStorageServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "CreateFile",
			Handler:    _DistributedStorageService_CreateFile_Handler,
		},
		{
			MethodName: "DeleteFile",
			Handler:    _DistributedStorageService_DeleteFile_Handler,
		},
		{
			MethodName: "GetFileInfo",
			Handler:    _DistributedStorageService_GetFileInfo_Handler,
		},
		{
			MethodName: "ListFiles",
			Handler:    _DistributedStorageService_ListFiles_Handler,
		},
		{
			MethodName: "DeleteShard",
			Handler:    _DistributedStorageService_DeleteShard_Handler,
		},
	},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "UploadShard",
			Handler:       _DistributedStorageService_UploadShard_Handler,
			ClientStreams: true,
		},
		{
			StreamName:    "DownloadShard",
			Handler:       _DistributedStorageService_DownloadShard_Handler,
			ServerStreams: true,
		},
	},
	Metadata: "distributed_storage.proto",
}
