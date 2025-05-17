package index

import (
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"seetaSearch/messages"
)

type IndexServiceServer interface {
	DelDoc(context.Context, *messages.DocId) (*messages.ReqCount, error)
	AddDoc(context.Context, *messages.Document) (*messages.ReqCount, error)
	Search(context.Context, *messages.Request) (*messages.Result, error)
	Count(context.Context, *messages.CountRequest) (*messages.ReqCount, error)
}

type DefaultIndexServiceServer struct {
}

func (*DefaultIndexServiceServer) DelDoc(ctx context.Context, req *messages.DocId) (*messages.ReqCount, error) {
	return nil, status.Errorf(codes.Unimplemented, "default method")
}
func (*DefaultIndexServiceServer) AddDoc(ctx context.Context, req *messages.Document) (*messages.ReqCount, error) {
	return nil, status.Errorf(codes.Unimplemented, "default method")
}
func (*DefaultIndexServiceServer) Search(ctx context.Context, req *messages.Request) (*messages.ReqCount, error) {
	return nil, status.Errorf(codes.Unimplemented, "default method")
}
func (*DefaultIndexServiceServer) Count(ctx context.Context, req *messages.CountRequest) (*messages.ReqCount, error) {
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
		Metadata: "request.proto",
	}
	s.RegisterService(&indexServiceDesc, srv)
}

func DelDocHandler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(messages.DocId)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(IndexServiceServer).DelDoc(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/IndexService/DelDoc",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(IndexServiceServer).DelDoc(ctx, req.(*messages.DocId))
	}
	return interceptor(ctx, in, info, handler)
}

func AddDocHandler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(messages.Document)
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
		return srv.(IndexServiceServer).AddDoc(ctx, req.(*messages.Document))
	}
	return interceptor(ctx, in, info, handler)
}

func SearchHandler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(messages.Request)
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
		return srv.(IndexServiceServer).Search(ctx, req.(*messages.Request))
	}
	return interceptor(ctx, in, info, handler)
}

func CountHandler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(messages.CountRequest)
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
		return srv.(IndexServiceServer).Count(ctx, req.(*messages.CountRequest))
	}
	return interceptor(ctx, in, info, handler)
}
