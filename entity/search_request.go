package entity

import (
	"errors"
	"fmt"
	"github.com/gogo/protobuf/proto"
	"io"
)

type SearchRequest struct {
	Query   *TermQuery `protobuf:"bytes,1,opt,name=Query,proto3" json:"Query,omitempty"`
	OnFlag  uint64     `protobuf:"varint,2,opt,name=OnFlag,proto3" json:"OnFlag,omitempty"`
	OffFlag uint64     `protobuf:"varint,3,opt,name=OffFlag,proto3" json:"OffFlag,omitempty"`
	OrFlags []uint64   `protobuf:"varint,4,rep,packed,name=OrFlags,proto3" json:"OrFlags,omitempty"`
}

func (m *SearchRequest) Reset()         { *m = SearchRequest{} }
func (m *SearchRequest) String() string { return proto.CompactTextString(m) }
func (*SearchRequest) ProtoMessage()    {}
func (m *SearchRequest) Unmarshal(data []byte) error {
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
			return fmt.Errorf("proto: SearchRequest: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: SearchRequest: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Query", wireType)
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
			if m.Query == nil {
				m.Query = &TermQuery{}
			}
			if err := m.Query.Unmarshal(data[index:postIndex]); err != nil {
				return err
			}
			index = postIndex
		case 2:
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field OnFlag", wireType)
			}
			flag, err := CalculateIntId(index, l, data)
			if err != nil {
				return err
			}
			m.OnFlag = flag
		case 4:
			if wireType == 0 {
				v, err := CalculateIntId(index, l, data)
				if err != nil {
					return err
				}
				m.OrFlags = append(m.OrFlags, v)
			} else if wireType == 2 {
				packedLen, err := CalculateIntId(index, l, data)
				if err != nil {
					return err
				}
				if packedLen < 0 {
					return errors.New("negative length found during unmarshalling")
				}
				postIndex := index + int(packedLen)
				if postIndex < 0 {
					return errors.New("negative length found during unmarshalling")
				}
				if postIndex > l {
					return io.ErrUnexpectedEOF
				}
				var elementCount int
				var count int
				for _, integer := range data[index:postIndex] {
					if integer < 128 {
						count++
					}
				}
				elementCount = count
				if elementCount != 0 && len(m.OrFlags) == 0 {
					m.OrFlags = make([]uint64, 0, elementCount)
				}
				for index < postIndex {
					v, err := CalculateIntId(index, l, data)
					if err != nil {
						return err
					}
					m.OrFlags = append(m.OrFlags, v)
				}
			} else {
				return fmt.Errorf("wrong wireType = %d for field OrFlags", wireType)
			}
		default:
			index = preIndex
			skippy, err := SkipIndex(data[index:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (index+skippy) < 0 {
				return errors.New("integer overflow")
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
func (m *SearchRequest) Marshal(b []byte, deterministic bool) ([]byte, error) {
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
func (m *SearchRequest) Merge(src proto.Message) {
	MessageInfoDocId.Merge(m, src)
}

func (m *SearchRequest) DiscardUnknown() {
	MessageInfoDocId.DiscardUnknown(m)
}

func (m *SearchRequest) GetQuery() *TermQuery {
	if m != nil {
		return m.Query
	}
	return nil
}

func (m *SearchRequest) GetOnFlag() uint64 {
	if m != nil {
		return m.OnFlag
	}
	return 0
}

func (m *SearchRequest) GetOffFlag() uint64 {
	if m != nil {
		return m.OffFlag
	}
	return 0
}

func (m *SearchRequest) GetOrFlags() []uint64 {
	if m != nil {
		return m.OrFlags
	}
	return nil
}

func (m *SearchRequest) MarshalTo(data []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(data[:size])
}

func (m *SearchRequest) MarshalToSizedBuffer(data []byte) (int, error) {
	i := len(data)
	_ = i
	var l int
	_ = l
	if len(m.OrFlags) > 0 {
		data2 := make([]byte, len(m.OrFlags)*10)
		var j1 int
		for _, num := range m.OrFlags {
			for num >= 1<<7 {
				data2[j1] = uint8(uint64(num)&0x7f | 0x80)
				num >>= 7
				j1++
			}
			data2[j1] = uint8(num)
			j1++
		}
		i -= j1
		copy(data[i:], data2[:j1])
		i = encodeVarIntIndex(data, i, uint64(j1))
		i--
		data[i] = 0x22
	}
	if m.OffFlag != 0 {
		i = encodeVarIntIndex(data, i, m.OffFlag)
		i--
		data[i] = 0x18
	}
	if m.OnFlag != 0 {
		i = encodeVarIntIndex(data, i, m.OnFlag)
		i--
		data[i] = 0x10
	}
	if m.Query != nil {
		{
			size, err := m.Query.MarshalToSizedBuffer(data[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarIntIndex(data, i, uint64(size))
		}
		i--
		data[i] = 0xa
	}
	return len(data) - i, nil
}

func (m *SearchRequest) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.Query != nil {
		l = m.Query.Size()
		n += 1 + l + SovIndex(uint64(l))
	}
	if m.OnFlag != 0 {
		n += 1 + SovIndex(m.OnFlag)
	}
	if m.OffFlag != 0 {
		n += 1 + SovIndex(m.OffFlag)
	}
	if len(m.OrFlags) > 0 {
		l = 0
		for _, e := range m.OrFlags {
			l += SovIndex(e)
		}
		n += 1 + SovIndex(uint64(l)) + l
	}
	return n
}
