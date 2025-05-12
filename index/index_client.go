package index

import (
	"errors"
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	"io"
	"math"
)

var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf
var messageInfoDocId proto.InternalMessageInfo

type DocId struct {
	Id string `protobuf:"bytes,1,opt,name=Id,proto3" json:"id,omitempty"`
}

func (*DocId) ProtoMessage() {
	//TODO implement me
}

func (d *DocId) Reset() {
	*d = DocId{}
}
func (*DocId) Descriptor() ([]byte, []int) {
	return []byte{}, []int{0}
}
func (d *DocId) String() string {
	return proto.CompactTextString(d)
}

func (d *DocId) Unmarshal(data []byte) error {
	l := len(data)
	seed := 0
	for l > 0 {
		preIndex := seed
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowIndex
			}
			if seed >= l {
				return io.ErrUnexpectedEOF
			}
			b := data[seed]
			seed++
			wire |= uint64(b&0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return errors.New("proto:AffectedCount: wiretype end group")
		}
		if fieldNum <= 0 {
			return errors.New("")
		}
	}
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
