package entity

import (
	"errors"
	"fmt"
	"github.com/gogo/protobuf/proto"
	"io"
)

type CountRequest struct {
}

func (m *CountRequest) Reset()         { *m = CountRequest{} }
func (m *CountRequest) String() string { return proto.CompactTextString(m) }
func (*CountRequest) ProtoMessage()    {}

func (m *CountRequest) Unmarshal(data []byte) error {
	l := len(data)
	index := 0
	for index < l {
		preIndex := index
		wire, err := CalculateIntId(index, l, data)
		if err != nil {
			return err
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("illegal tag %d (wire type %d)", fieldNum, wire)
		}

		index = preIndex
		skippy, err := SkipIndex(data[index:])
		if err != nil {
			return err
		}
		if (skippy < 0) || (index+skippy) < 0 {
			return errors.New("negative length found during unmarshalling")
		}

		if (index + skippy) > l {
			return io.ErrUnexpectedEOF
		}
		index += skippy

	}

	if index > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func (m *CountRequest) MarshalWithDeterministic(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return MessageInfoDocId.Marshal(b, m, deterministic)
	}

	b = b[:cap(b)]
	n, err := m.MarshalToSizedBuffer(b)
	if err != nil {
		return nil, err
	}

	return b[:n], nil
}
func (m *CountRequest) Merge(src proto.Message) {
	MessageInfoDocId.Merge(m, src)
}

func (m *CountRequest) DiscardUnknown() {
	MessageInfoDocId.DiscardUnknown(m)
}

func (m *CountRequest) Marshal() (data []byte, err error) {
	size := m.Size()
	data = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(data[:size])
	if err != nil {
		return nil, err
	}
	return data[:n], nil
}

func (m *CountRequest) MarshalTo(data []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(data[:size])
}

func (m *CountRequest) MarshalToSizedBuffer(data []byte) (int, error) {
	i := len(data)
	return len(data) - i, nil
}
func (m *CountRequest) Size() (n int) {
	if m == nil {
		return 0
	}
	return n
}
