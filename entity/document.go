package entity

import (
	"errors"
	"fmt"
	"github.com/gogo/protobuf/proto"
	"io"
)

type Document struct {
	Id          string     `protobuf:"bytes,1,opt,name=Id,proto3" json:"id,omitempty"`
	FloatId     float64    `protobuf:"variant,2,opt,name=FloatId,proto3" json:"floatId,omitempty"`
	BitsFeature uint64     `protobuf:"variant,3,opt,name=BitsFeature,proto3" json:"bitsFeature,omitempty"`
	KeyWords    []*Keyword `protobuf:"bytes,4,rep,name=KeyWords,proto3" json:"keyWords,omitempty"`
	Bytes       []byte     `protobuf:"byte,5,opt,name=Bytes,proto3" json:"bytes,omitempty"`
}

func (m *Document) Reset()         { *m = Document{} }
func (m *Document) String() string { return proto.CompactTextString(m) }
func (*Document) ProtoMessage()    {}
func (m *Document) Unmarshal(data []byte) error {
	l := len(data)
	index := 0
	for index < l {
		preIndex := index
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return errors.New("integer overflow")
			}
			if index >= l {
				return io.ErrUnexpectedEOF
			}
			b := data[index]
			index++
			wire |= uint64(b&0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("wrong wireType = %d for field Id", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return errors.New("integer overflow")
				}
				if index >= l {
					return io.ErrUnexpectedEOF
				}
				b := data[index]
				index++
				stringLen |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return errors.New("negative length found during unmarshalling")
			}
			postIndex := index + intStringLen
			if postIndex < 0 {
				return errors.New("negative length found during unmarshalling")
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Id = string(data[index:postIndex])
			index = postIndex
		case 2:
			if wireType != 0 {
				return fmt.Errorf("wrong wireType = %d for field IntId", wireType)
			}
			var intId uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return errors.New("integer overflow")
				}
				if index >= l {
					return io.ErrUnexpectedEOF
				}
				b := data[index]
				index++
				intId |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			m.FloatId = float64(intId)
		case 3:
			if wireType != 0 {
				return fmt.Errorf("wrong wireType = %d for field BitsFeature", wireType)
			}
			m.BitsFeature = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return errors.New("integer overflow")
				}
				if index >= l {
					return io.ErrUnexpectedEOF
				}
				b := data[index]
				index++
				m.BitsFeature |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 4:
			if wireType != 2 {
				return fmt.Errorf("wrong wireType = %d for field Keywords", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return errors.New("integer overflow")
				}
				if index >= l {
					return io.ErrUnexpectedEOF
				}
				b := data[index]
				index++
				msglen |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if msglen < 0 {
				return errors.New("negative length found during unmarshalling")
			}
			postIndex := index + msglen
			if postIndex < 0 {
				return errors.New("negative length found during unmarshalling")
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.KeyWords = append(m.KeyWords, &Keyword{})
			if err := m.KeyWords[len(m.KeyWords)-1].Unmarshal(data[index:postIndex]); err != nil {
				return err
			}
			index = postIndex
		case 5:
			if wireType != 2 {
				return fmt.Errorf("wrong wireType = %d for field Bytes", wireType)
			}
			var byteLen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return errors.New("integer overflow")
				}
				if index >= l {
					return io.ErrUnexpectedEOF
				}
				b := data[index]
				index++
				byteLen |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if byteLen < 0 {
				return errors.New("negative length found during unmarshalling")
			}
			postIndex := index + byteLen
			if postIndex < 0 {
				return errors.New("negative length found during unmarshalling")
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Bytes = append(m.Bytes[:0], data[index:postIndex]...)
			if m.Bytes == nil {
				m.Bytes = []byte{}
			}
			index = postIndex
		default:
			index = preIndex
			skippy, err := skipDoc(data[index:])
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
func (m *Document) MarshalWithDeterministic(b []byte, deterministic bool) ([]byte, error) {
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
func (m *Document) Merge(src proto.Message) {
	MessageInfoDocId.Merge(m, src)
}

func (m *Document) DiscardUnknown() {
	MessageInfoDocId.DiscardUnknown(m)
}

func (m *Document) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.Id)
	if l > 0 {
		n += 1 + l + sovDoc(uint64(l))
	}
	if m.FloatId != 0 {
		n += 1 + sovDoc(uint64(m.FloatId))
	}
	if m.BitsFeature != 0 {
		n += 1 + sovDoc(uint64(m.BitsFeature))
	}
	if len(m.KeyWords) > 0 {
		for _, e := range m.KeyWords {
			l = e.Size()
			n += 1 + l + sovDoc(uint64(l))
		}
	}
	l = len(m.Bytes)
	if l > 0 {
		n += 1 + l + sovDoc(uint64(l))
	}
	return n
}

func (m *Document) Marshal() (data []byte, err error) {
	size := m.Size()
	data = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(data[:size])
	if err != nil {
		return nil, err
	}
	return data[:n], nil
}

func (m *Document) MarshalTo(data []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(data[:size])
}

func (m *Document) MarshalToSizedBuffer(data []byte) (int, error) {
	i := len(data)
	_ = i
	var l int
	_ = l
	if len(m.Bytes) > 0 {
		i -= len(m.Bytes)
		copy(data[i:], m.Bytes)
		i = encodeVarIntDoc(data, i, uint64(len(m.Bytes)))
		i--
		data[i] = 0x2a
	}
	if len(m.KeyWords) > 0 {
		for index := len(m.KeyWords) - 1; index >= 0; index-- {
			{
				size, err := m.KeyWords[index].MarshalToSizedBuffer(data[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarIntDoc(data, i, uint64(size))
			}
			i--
			data[i] = 0x22
		}
	}
	if m.BitsFeature != 0 {
		i = encodeVarIntDoc(data, i, m.BitsFeature)
		i--
		data[i] = 0x18
	}
	if m.FloatId != 0 {
		i = encodeVarIntDoc(data, i, uint64(m.FloatId))
		i--
		data[i] = 0x10
	}
	if len(m.Id) > 0 {
		i -= len(m.Id)
		copy(data[i:], m.Id)
		i = encodeVarIntDoc(data, i, uint64(len(m.Id)))
		i--
		data[i] = 0xa
	}
	return len(data) - i, nil
}
