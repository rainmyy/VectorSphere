package index

import (
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	"math"
)

var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf
var messageInfoDocId proto.InternalMessageInfo

type DocId struct {
	Id string `protobuf:"bytes,1,opt,name=Id,proto3" json:"id,omitempty"`
}

func (d *DocId) ProtoMessage() {
	//TODO implement me
}

func (d *DocId) Reset() {
	*d = DocId{}
}

func (d *DocId) String() string {
	return proto.CompactTextString(d)
}

func (d *DocId) DoUnmarshal(b []byte) error {
	return d.Unmarshal(b)
}

func (d *DocId) Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return messageInfoDocId.Marshal(b, d, deterministic)
	}
	b = b[:cap(b)]
	n, err := d.MarshalToBuffer(b)
	if err != nil {
		return nil, err
	}
	return b[:n], nil
}

func (d *DocId) Merge(src proto.Message) {
	messageInfoDocId.Merge(d, src)
}

func (d *DocId) Size() int {
	return d.Size()
}

func (d *DocId) DiscardUnknown() {
	messageInfoDocId.DiscardUnknown(d)
}

func (d *DocId) GetDocId() string {
	if d == nil {
		return ""
	}

	return d.Id
}

type AffectedCount struct {
	Count int32 `protobuf:"varint,1,opt,name=Count,proto3" json:"Count,omitempty"`
}

func (a *AffectedCount) ProtoMessage() {
	//TODO implement me
}

func (a *AffectedCount) Reset() {
	*a = AffectedCount{}
}

func (a *AffectedCount) String() string {
	return proto.CompactTextString(a)
}

func (a *AffectedCount) Unmarshal(b []byte) error {

}
