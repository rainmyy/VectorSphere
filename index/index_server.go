package index

import "context"

type ServerInterface interface {
	DelDoc(ctx context.Context, id *DocId) (*Count, error)
	AddDoc(ctx context.Context, document *Document) (*Count, error)
	Search(ctx context.Context, request *SearchRequest) (*SearchResult, error)
	Count(ctx context.Context, request *CountRequest) (*Count, error)
}
