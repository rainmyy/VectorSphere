package index

import (
	"context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type ServerInterface interface {
	DelDoc(ctx context.Context, id *DocId) (*Count, error)
	AddDoc(ctx context.Context, document *Document) (*Count, error)
	Search(ctx context.Context, request *SearchRequest) (*SearchResult, error)
	Count(ctx context.Context, request *CountRequest) (*Count, error)
}

type DefaultServiceServer struct {
}

func (*DefaultServiceServer) DelDoc(ctx context.Context, id *DocId) (*Count, error) {
	return nil, status.Errorf(codes.Unavailable, "default service del doc")
}

func (*DefaultServiceServer) AddDoc(ctx context.Context, req *Document) (*Count, error) {
	return nil, status.Errorf(codes.Unavailable, "default service add doc")
}

func (*DefaultServiceServer) Search(ctx context.Context, req *SearchRequest) (*Count, error) {
	return nil, status.Errorf(codes.Unavailable, "default service search")
}

func (*DefaultServiceServer) Count(ctx context.Context, req *SearchRequest) (*Count, error) {
	return nil, status.Errorf(codes.Unavailable, "default service get couunt")
}

func RegisterIndexServer(s *grpc.Server, srv IndexInterface) {
	s.RegisterService()
}
