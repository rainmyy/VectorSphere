package entity

import (
	"errors"
	"fmt"
	"io"
	mathbits "math/bits"
)

type TermQuery struct {
	Keyword *Keyword     `protobuf:"bytes,1,opt,name=Keyword,proto3" json:"Keyword,omitempty"`
	Must    []*TermQuery `protobuf:"bytes,2,rep,name=Must,proto3" json:"Must,omitempty"`
	Should  []*TermQuery `protobuf:"bytes,3,rep,name=Should,proto3" json:"Should,omitempty"`
}

func (m *TermQuery) Marshal() (data []byte, err error) {
	size := m.Size()
	data = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(data[:size])
	if err != nil {
		return nil, err
	}
	return data[:n], nil
}

func (m *TermQuery) MarshalTo(data []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(data[:size])
}

func (m *TermQuery) MarshalToSizedBuffer(data []byte) (int, error) {
	i := len(data)
	if len(m.Should) > 0 {
		for index := len(m.Should) - 1; index >= 0; index-- {
			{
				size, err := m.Should[index].MarshalToSizedBuffer(data[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarIntTermQuery(data, i, uint64(size))
			}
			i--
			data[i] = 0x1a
		}
	}
	if len(m.Must) > 0 {
		for index := len(m.Must) - 1; index >= 0; index-- {
			{
				size, err := m.Must[index].MarshalToSizedBuffer(data[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarIntTermQuery(data, i, uint64(size))
			}
			i--
			data[i] = 0x12
		}
	}
	if m.Keyword != nil {
		{
			size, err := m.Keyword.MarshalToSizedBuffer(data[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarIntTermQuery(data, i, uint64(size))
		}
		i--
		data[i] = 0xa
	}
	return len(data) - i, nil
}

func encodeVarIntTermQuery(data []byte, offset int, v uint64) int {
	offset -= sovTermQuery(v)
	base := offset
	for v >= 1<<7 {
		data[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	data[offset] = uint8(v)
	return base
}

func (m *TermQuery) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	if m.Keyword != nil {
		l = m.Keyword.Size()
		n += 1 + l + sovTermQuery(uint64(l))
	}
	if len(m.Must) > 0 {
		for _, e := range m.Must {
			l = e.Size()
			n += 1 + l + sovTermQuery(uint64(l))
		}
	}
	if len(m.Should) > 0 {
		for _, e := range m.Should {
			l = e.Size()
			n += 1 + l + sovTermQuery(uint64(l))
		}
	}
	return n
}

func sovTermQuery(x uint64) (n int) {
	return (mathbits.Len64(x|1) + 6) / 7
}
func sozTermQuery(x uint64) (n int) {
	return sovTermQuery((x << 1) ^ uint64(int64(x)>>63))
}
func (m *TermQuery) Unmarshal(data []byte) error {
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
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("wrong wireType = %d for field Keyword", wireType)
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
			if m.Keyword == nil {
				m.Keyword = &Keyword{}
			}
			if err := m.Keyword.Unmarshal(data[index:postIndex]); err != nil {
				return err
			}
			index = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Must", wireType)
			}
			msgLen, err := CalculateIntId(index, l, data)
			if err != nil {
				return err
			}
			if msgLen < 0 {
				return errors.New("ErrInvalidLengthTermQuery")
			}
			postIndex := index + int(msgLen)
			if postIndex < 0 {
				return errors.New("ErrInvalidLengthTermQuery")
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Must = append(m.Must, &TermQuery{})
			if err := m.Must[len(m.Must)-1].Unmarshal(data[index:postIndex]); err != nil {
				return err
			}
			index = postIndex
		case 3:
			if wireType != 2 {
				return fmt.Errorf("wrong wireType = %d for field Should", wireType)
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
			m.Should = append(m.Should, &TermQuery{})
			if err := m.Should[len(m.Should)-1].Unmarshal(data[index:postIndex]); err != nil {
				return err
			}
			index = postIndex
		default:
			index = preIndex
			skippy, err := skipTermQuery(data[index:])
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
