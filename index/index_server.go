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
	indexServiceServiceDesc := grpc.ServiceDesc{
		ServiceName: "IndexService",
		HandlerType: (*IndexInterface)(nil),
		Methods: []grpc.MethodDesc{
			{
				MethodName: "DelDoc",
				Handler:    DelDocHandler,
			},
			{
				MethodName: "AddDoc",
				Handler:    AddDocHandler,
			},
			{
				MethodName: "Search",
				Handler:    SearchHandler,
			},
			{
				MethodName: "Count",
				Handler:    CountHandler,
			},
		},
		Streams:  []grpc.StreamDesc{},
		Metadata: "index.proto",
	}
	s.RegisterService(&indexServiceServiceDesc, srv)
}
func CountHandler(
	srv interface{},
	ctx context.Context,
	dec func(interface{}) error,
	interceptor grpc.UnaryServerInterceptor,
) (interface{}, error) {
	in := new(CountRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ServerInterface).Count(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/IndexService/Count",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ServerInterface).Count(ctx, req.(*CountRequest))
	}
	return interceptor(ctx, in, info, handler)
}
func SearchHandler(
	srv interface{},
	ctx context.Context,
	dec func(interface{}) error,
	interceptor grpc.UnaryServerInterceptor,
) (interface{}, error) {
	in := new(SearchRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ServerInterface).Search(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/IndexService/Search",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ServerInterface).Search(ctx, req.(*SearchRequest))
	}

	return interceptor(ctx, in, info, handler)
}
func AddDocHandler(
	srv interface{},
	ctx context.Context,
	dec func(interface{}) error,
	interceptor grpc.UnaryServerInterceptor,
) (interface{}, error) {
	in := new(Document)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ServerInterface).AddDoc(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/IndexService/AddDoc",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ServerInterface).AddDoc(ctx, req.(*Document))
	}
	return interceptor(ctx, in, info, handler)
}
func DelDocHandler(
	srv interface{},
	ctx context.Context,
	dec func(interface{}) error,
	interceptor grpc.UnaryServerInterceptor,
) (interface{}, error) {
	in := new(DocId)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ServerInterface).DelDoc(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/IndexService/DelDoc",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ServerInterface).DelDoc(ctx, req.(*DocId))
	}
	return interceptor(ctx, in, info, handler)
}
