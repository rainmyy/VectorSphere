package index

import (
	"context"
	"google.golang.org/grpc"
	"seetaSearch/messages"
)

var _ context.Context
var _ grpc.ClientConn

const _ = grpc.SupportPackageIsVersion4

type ClientInterface interface {
	DelDoc(ctx context.Context, in *messages.DocId, opts ...grpc.CallOption) (*messages.ReqCount, error)
	AddDoc(ctx context.Context, in *messages.Document, opts ...grpc.CallOption) (*messages.ReqCount, error)
	Search(ctx context.Context, in *messages.Request, opts ...grpc.CallOption) (*messages.Result, error)
	Count(ctx context.Context, in *messages.CountRequest, opts ...grpc.CallOption) (*messages.ReqCount, error)
}

type IndexClient struct {
	cc *grpc.ClientConn
}

func NewIndexClient(cc *grpc.ClientConn) *IndexClient {
	return &IndexClient{cc}
}

func (c *IndexClient) DeleteDoc(ctx context.Context, in *messages.DocId, opts ...grpc.CallOption) (*messages.ReqCount, error) {
	out := new(messages.ReqCount)
	err := c.cc.Invoke(ctx, "/IndexService/DeleteDoc", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *IndexClient) AddDoc(ctx context.Context, in *messages.Document, opts ...grpc.CallOption) (*messages.ReqCount, error) {
	out := new(messages.ReqCount)
	err := c.cc.Invoke(ctx, "/IndexService/AddDoc", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *IndexClient) Search(ctx context.Context, in *messages.Request, opts ...grpc.CallOption) (*messages.Result, error) {
	out := new(messages.Result)
	err := c.cc.Invoke(ctx, "/IndexService/Search", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *IndexClient) Count(ctx context.Context, in *messages.CountRequest, opts ...grpc.CallOption) (*messages.ReqCount, error) {
	out := new(messages.ReqCount)
	err := c.cc.Invoke(ctx, "/IndexService/Count", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}
