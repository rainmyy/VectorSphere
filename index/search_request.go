package index

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
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return errors.New("integer overflow")
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := data[iNdEx]
			iNdEx++
			wire |= uint64(b&0x7F) << shift
			if b < 0x80 {
				break
			}
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
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return errors.New("integer overflow")
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := data[iNdEx]
				iNdEx++
				msglen |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if msglen < 0 {
				return errors.New("negative length found during unmarshalling")
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return errors.New("negative length found during unmarshalling")
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Query == nil {
				m.Query = &TermQuery{}
			}
			if err := m.Query.Unmarshal(data[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field OnFlag", wireType)
			}
			m.OnFlag = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return errors.New("integer overflow")
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := data[iNdEx]
				iNdEx++
				m.OnFlag |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field OffFlag", wireType)
			}
			m.OffFlag = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return errors.New("integer overflow")
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := data[iNdEx]
				iNdEx++
				m.OffFlag |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 4:
			if wireType == 0 {
				var v uint64
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return errors.New("integer overflow")
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := data[iNdEx]
					iNdEx++
					v |= uint64(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				m.OrFlags = append(m.OrFlags, v)
			} else if wireType == 2 {
				var packedLen int
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return errors.New("integer overflow")
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := data[iNdEx]
					iNdEx++
					packedLen |= int(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				if packedLen < 0 {
					return errors.New("negative length found during unmarshalling")
				}
				postIndex := iNdEx + packedLen
				if postIndex < 0 {
					return errors.New("negative length found during unmarshalling")
				}
				if postIndex > l {
					return io.ErrUnexpectedEOF
				}
				var elementCount int
				var count int
				for _, integer := range data[iNdEx:postIndex] {
					if integer < 128 {
						count++
					}
				}
				elementCount = count
				if elementCount != 0 && len(m.OrFlags) == 0 {
					m.OrFlags = make([]uint64, 0, elementCount)
				}
				for iNdEx < postIndex {
					var v uint64
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return errors.New("integer overflow")
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := data[iNdEx]
						iNdEx++
						v |= uint64(b&0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					m.OrFlags = append(m.OrFlags, v)
				}
			} else {
				return fmt.Errorf("wrong wireType = %d for field OrFlags", wireType)
			}
		default:
			iNdEx = preIndex
			skippy, err := skipIndex(data[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return errors.New("integer overflow")
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func (m *SearchRequest) Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return messageInfoDocId.Marshal(b, m, deterministic)
	}

	b = b[:cap(b)]
	n, err := m.MarshalToSizedBuffer(b)
	if err != nil {
		return nil, err
	}

	return b[:n], nil
}
func (m *SearchRequest) Merge(src proto.Message) {
	messageInfoDocId.Merge(m, src)
}
func (m *SearchRequest) XXX_Size() int {
	return m.Size()
}
func (m *SearchRequest) DiscardUnknown() {
	messageInfoDocId.DiscardUnknown(m)
}

func (m *SearchRequest) XXX_DiscardUnknown() {
	messageInfoDocId.DiscardUnknown(m)
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

func (m *SearchRequest) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *SearchRequest) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.OrFlags) > 0 {
		dAtA2 := make([]byte, len(m.OrFlags)*10)
		var j1 int
		for _, num := range m.OrFlags {
			for num >= 1<<7 {
				dAtA2[j1] = uint8(uint64(num)&0x7f | 0x80)
				num >>= 7
				j1++
			}
			dAtA2[j1] = uint8(num)
			j1++
		}
		i -= j1
		copy(dAtA[i:], dAtA2[:j1])
		i = encodeVarIntIndex(dAtA, i, uint64(j1))
		i--
		dAtA[i] = 0x22
	}
	if m.OffFlag != 0 {
		i = encodeVarIntIndex(dAtA, i, m.OffFlag)
		i--
		dAtA[i] = 0x18
	}
	if m.OnFlag != 0 {
		i = encodeVarIntIndex(dAtA, i, m.OnFlag)
		i--
		dAtA[i] = 0x10
	}
	if m.Query != nil {
		{
			size, err := m.Query.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarIntIndex(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func (m *SearchRequest) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.Query != nil {
		l = m.Query.Size()
		n += 1 + l + sovIndex(uint64(l))
	}
	if m.OnFlag != 0 {
		n += 1 + sovIndex(m.OnFlag)
	}
	if m.OffFlag != 0 {
		n += 1 + sovIndex(m.OffFlag)
	}
	if len(m.OrFlags) > 0 {
		l = 0
		for _, e := range m.OrFlags {
			l += sovIndex(uint64(e))
		}
		n += 1 + sovIndex(uint64(l)) + l
	}
	return n
}
