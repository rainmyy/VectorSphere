package entity

import (
	"errors"
	"fmt"
	"github.com/gogo/protobuf/proto"
	"io"
)

type SearchResult struct {
	Results []*Document `protobuf:"bytes,1,rep,name=Results,proto3" json:"Results,omitempty"`
}

func (m *SearchResult) Reset()         { *m = SearchResult{} }
func (m *SearchResult) String() string { return proto.CompactTextString(m) }
func (*SearchResult) ProtoMessage()    {}
func (m *SearchResult) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *SearchResult) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
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
func (m *SearchResult) Merge(src proto.Message) {
	MessageInfoDocId.Merge(m, src)
}

func (m *SearchResult) DiscardUnknown() {
	MessageInfoDocId.DiscardUnknown(m)
}

func (m *SearchResult) GetResults() []*Document {
	if m != nil {
		return m.Results
	}
	return nil
}

func (m *SearchResult) Marshal() (data []byte, err error) {
	size := m.Size()
	data = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(data[:size])
	if err != nil {
		return nil, err
	}
	return data[:n], nil
}

func (m *SearchResult) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *SearchResult) MarshalToSizedBuffer(data []byte) (int, error) {
	i := len(data)
	if len(m.Results) > 0 {
		for index := len(m.Results) - 1; index >= 0; index-- {
			{
				size, err := m.Results[index].MarshalToSizedBuffer(data[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarIntIndex(data, i, uint64(size))
			}
			i--
			data[i] = 0xa
		}
	}
	return len(data) - i, nil
}

func (m *SearchResult) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	if len(m.Results) > 0 {
		for _, e := range m.Results {
			l = e.Size()
			n += 1 + l + SovIndex(uint64(l))
		}
	}
	return n
}

func (m *SearchResult) Unmarshal(data []byte) error {
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
			return fmt.Errorf("SearchResult: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("SearchResult: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("wrong wireType = %d for field Results", wireType)
			}
			msgLen, err := CalculateIntId(index, l, data)
			if err != nil {
				return err
			}
			if msgLen < 0 {
				return errors.New("negative length found during unmarshalling")
			}
			postIndex := index + int(msgLen)
			if postIndex < 0 {
				return errors.New("negative length found during unmarshalling")
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Results = append(m.Results, &Document{})
			if err := m.Results[len(m.Results)-1].Unmarshal(data[index:postIndex]); err != nil {
				return err
			}
			index = postIndex
		default:
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
	}

	if index > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
