package index

import (
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type IndexServiceServer interface {
	DelDoc(context.Context, *DocId) (*Count, error)
	AddDoc(context.Context, *Document) (*Count, error)
	Search(context.Context, *SearchRequest) (*SearchResult, error)
	Count(context.Context, *CountRequest) (*Count, error)
}

type DefaultIndexServiceServer struct {
}

func (*DefaultIndexServiceServer) DelDoc(ctx context.Context, req *DocId) (*Count, error) {
	return nil, status.Errorf(codes.Unimplemented, "default method")
}
func (*DefaultIndexServiceServer) AddDoc(ctx context.Context, req *Document) (*Count, error) {
	return nil, status.Errorf(codes.Unimplemented, "default method")
}
func (*DefaultIndexServiceServer) Search(ctx context.Context, req *SearchRequest) (*Count, error) {
	return nil, status.Errorf(codes.Unimplemented, "default method")
}
func (*DefaultIndexServiceServer) Count(ctx context.Context, req *CountRequest) (*Count, error) {
	return nil, status.Errorf(codes.Unimplemented, "default method")
}

func RegisterIndexServiceServer(s *grpc.Server, srv IndexServiceServer) {
	var indexServiceDesc = grpc.ServiceDesc{
		ServiceName: "IndexService",
		HandlerType: (*IndexServiceServer)(nil),
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
	s.RegisterService(&indexServiceDesc, srv)
}

func DelDocHandler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DocId)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(IndexServiceServer).DelDoc(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/IndexService/DeleteDoc",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(IndexServiceServer).DelDoc(ctx, req.(*DocId))
	}
	return interceptor(ctx, in, info, handler)
}

func AddDocHandler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(Document)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(IndexServiceServer).AddDoc(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/IndexService/AddDoc",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(IndexServiceServer).AddDoc(ctx, req.(*Document))
	}
	return interceptor(ctx, in, info, handler)
}

func SearchHandler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(SearchRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(IndexServiceServer).Search(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/IndexService/Search",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(IndexServiceServer).Search(ctx, req.(*SearchRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func CountHandler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(CountRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(IndexServiceServer).Count(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/IndexService/Count",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(IndexServiceServer).Count(ctx, req.(*CountRequest))
	}
	return interceptor(ctx, in, info, handler)
}
