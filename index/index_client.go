package index

import (
	"context"
	"google.golang.org/grpc"
)

var _ context.Context
var _ grpc.ClientConn

const _ = grpc.SupportPackageIsVersion4

type ClientInterface interface {
	DelDoc(ctx context.Context, in *DocId, opts ...grpc.CallOption) (*Count, error)
	AddDoc(ctx context.Context, in *Document, opts ...grpc.CallOption) (*Count, error)
	Search(ctx context.Context, in *SearchRequest, opts ...grpc.CallOption) (*SearchResult, error)
	Count(ctx context.Context, in *CountRequest, opts ...grpc.CallOption) (*Count, error)
}

type IndexClient struct {
	cc *grpc.ClientConn
}

func NewIndexClient(cc *grpc.ClientConn) *IndexClient {
	return &IndexClient{cc}
}

func (c *IndexClient) DeleteDoc(ctx context.Context, in *DocId, opts ...grpc.CallOption) (*Count, error) {
	out := new(Count)
	err := c.cc.Invoke(ctx, "/IndexService/DeleteDoc", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *IndexClient) AddDoc(ctx context.Context, in *Document, opts ...grpc.CallOption) (*Count, error) {
	out := new(Count)
	err := c.cc.Invoke(ctx, "/IndexService/AddDoc", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *IndexClient) Search(ctx context.Context, in *SearchRequest, opts ...grpc.CallOption) (*SearchResult, error) {
	out := new(SearchResult)
	err := c.cc.Invoke(ctx, "/IndexService/Search", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *IndexClient) Count(ctx context.Context, in *CountRequest, opts ...grpc.CallOption) (*Count, error) {
	out := new(Count)
	err := c.cc.Invoke(ctx, "/IndexService/Count", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}
