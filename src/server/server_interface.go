package server

import (
	messages2 "VectorSphere/src/proto/messages"
	serverProto "VectorSphere/src/proto/serverProto"
)

type ServerInterface interface {
	DelDoc(id *serverProto.DocId) int
	AddDoc(document *messages2.Document) (int, error)
	Search(query *messages2.TermQuery, onFlag, offFlag uint64, orFlags []uint64) (error, []*messages2.Document)
	Count() int
}
