package index

import (
	"errors"
	"fmt"
	"github.com/gogo/protobuf/proto"
	"io"
	"math"
)

var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

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
				return errors.New("integer overflow")
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
			return errors.New("wireType end group for non-group")
		}
		if fieldNum <= 0 {
			return errors.New("illegal tag %d (wire type %d)")
		}
		if fieldNum == 1 {
			if wireType != 2 {
				return fmt.Errorf("wrong wireType = %d for field DocId", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return errors.New("interface overflow")
				}
				if seed >= l {
					return io.ErrUnexpectedEOF
				}
				b := data[seed]
				seed++
				stringLen |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return errors.New("negative length found during unmarshalling")
			}
			postIndex := seed + intStringLen
			if postIndex < 0 {
				return errors.New("negative length found during unmarshalling")
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			d.Id = string(data[seed:postIndex])
			seed = postIndex
		} else {
			seed = preIndex
			skippy, err := skipIndex(data[seed:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (seed+skippy) < 0 {
				return errors.New("negative length found during unmarshalling")
			}
			if (seed + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			seed += skippy
		}
	}

	if seed > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}

func (d *DocId) MarshalWithDeterministic(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return messageInfoDocId.Marshal(b, d, deterministic)
	}
	b = b[:cap(b)]
	n, err := d.MarshalToSizedBuffer(b)
	if err != nil {
		return nil, err
	}
	return b[:n], nil
}

func (d *DocId) Marshal() (dAtA []byte, err error) {
	size := d.Size()
	dAtA = make([]byte, size)
	n, err := d.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (d *DocId) MarshalTo(dAtA []byte) (int, error) {
	size := d.Size()
	return d.MarshalToSizedBuffer(dAtA[:size])
}

func (d *DocId) MarshalToSizedBuffer(data []byte) (int, error) {
	i := len(data)
	_ = i
	var l int
	_ = l
	if len(d.Id) > 0 {
		i -= len(d.Id)
		copy(data[i:], d.Id)
		i = encodeIndex(data, i, uint64(len(d.Id)))
		i--
		data[i] = 0xa
	}
	return len(data) - i, nil
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
