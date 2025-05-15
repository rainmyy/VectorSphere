package index

import (
	"context"
	"google.golang.org/grpc"
	"seetaSearch/entity"
)

var _ context.Context
var _ grpc.ClientConn

const _ = grpc.SupportPackageIsVersion4

type ClientInterface interface {
	DelDoc(ctx context.Context, in *entity.DocId, opts ...grpc.CallOption) (*entity.Count, error)
	AddDoc(ctx context.Context, in *entity.Document, opts ...grpc.CallOption) (*entity.Count, error)
	Search(ctx context.Context, in *entity.SearchRequest, opts ...grpc.CallOption) (*entity.SearchResult, error)
	Count(ctx context.Context, in *entity.CountRequest, opts ...grpc.CallOption) (*entity.Count, error)
}

type IndexClient struct {
	cc *grpc.ClientConn
}

func NewIndexClient(cc *grpc.ClientConn) *IndexClient {
	return &IndexClient{cc}
}

func (c *IndexClient) DeleteDoc(ctx context.Context, in *entity.DocId, opts ...grpc.CallOption) (*entity.Count, error) {
	out := new(entity.Count)
	err := c.cc.Invoke(ctx, "/IndexService/DeleteDoc", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *IndexClient) AddDoc(ctx context.Context, in *entity.Document, opts ...grpc.CallOption) (*entity.Count, error) {
	out := new(entity.Count)
	err := c.cc.Invoke(ctx, "/IndexService/AddDoc", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *IndexClient) Search(ctx context.Context, in *entity.SearchRequest, opts ...grpc.CallOption) (*entity.SearchResult, error) {
	out := new(entity.SearchResult)
	err := c.cc.Invoke(ctx, "/IndexService/Search", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *IndexClient) Count(ctx context.Context, in *entity.CountRequest, opts ...grpc.CallOption) (*entity.Count, error) {
	out := new(entity.Count)
	err := c.cc.Invoke(ctx, "/IndexService/Count", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}
