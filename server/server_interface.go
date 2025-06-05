package server

import "VectorSphere/messages"

type ServerInterface interface {
	DelDoc(id *DocId) int
	AddDoc(document *messages.Document) (int, error)
	Search(query *messages.TermQuery, onFlag, offFlag uint64, orFlags []uint64) (error, []*messages.Document)
	Count() int
}
