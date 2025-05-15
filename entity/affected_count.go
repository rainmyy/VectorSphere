package entity

import (
	"errors"
	"fmt"
	"github.com/golang/protobuf/proto"
	"io"
)

type Count struct {
	Count int32 `protobuf:"varint,1,opt,name=Count,proto3" json:"Count,omitempty"`
}

func (a *Count) ProtoMessage() {
	//TODO implement me
}

func (a *Count) Reset() {
	*a = Count{}
}

func (a *Count) String() string {
	return proto.CompactTextString(a)
}

func (a *Count) Unmarshal(data []byte) error {
	l := len(data)
	seed := 0
	for seed < l {
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
			return fmt.Errorf("wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("wrong wireType = %d for field Count", wireType)
			}
			a.Count = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return errors.New("interface overflow")
				}
				if seed >= l {
					return io.ErrUnexpectedEOF
				}
				b := data[seed]
				seed++
				a.Count |= int32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			seed = preIndex
			skippy, err := SkipIndex(data[seed:])
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

func (a *Count) MarshalWithDeterministic(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return MessageInfoDocId.Marshal(b, a, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := a.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}

func (a *Count) Marshal() (data []byte, err error) {
	size := a.Size()
	data = make([]byte, size)
	n, err := a.MarshalToSizedBuffer(data[:size])
	if err != nil {
		return nil, err
	}
	return data[:n], nil
}

func (a *Count) MarshalTo(data []byte) (int, error) {
	size := a.Size()
	return a.MarshalToSizedBuffer(data[:size])
}

func (a *Count) MarshalToSizedBuffer(data []byte) (int, error) {
	i := len(data)
	_ = i
	var l int
	_ = l
	if a.Count != 0 {
		i = EncodeIndex(data, i, uint64(a.Count))
		i--
		data[i] = 0x8
	}
	return len(data) - i, nil
}

func (a *Count) Size() (n int) {
	if a == nil {
		return 0
	}
	var l int
	_ = l
	if a.Count != 0 {
		n += 1 + SovIndex(uint64(a.Count))
	}
	return n
}
