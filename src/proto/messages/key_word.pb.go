// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: key_word.proto

package messages

import (
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	io "io"
	math "math"
	math_bits "math/bits"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.GoGoProtoPackageIsVersion3 // please upgrade the proto package

type KeyWord struct {
	Field string `protobuf:"bytes,1,opt,name=Field,proto3" json:"Field,omitempty"`
	Word  string `protobuf:"bytes,2,opt,name=Word,proto3" json:"Word,omitempty"`
}

func (kw *KeyWord) Reset()         { *kw = KeyWord{} }
func (kw *KeyWord) String() string { return proto.CompactTextString(kw) }
func (*KeyWord) ProtoMessage()     {}
func (*KeyWord) Descriptor() ([]byte, []int) {
	return fileDescriptor_83a5b031d8c2f195, []int{0}
}
func (kw *KeyWord) XXX_Unmarshal(b []byte) error {
	return kw.Unmarshal(b)
}
func (kw *KeyWord) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_KeyWord.Marshal(b, kw, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := kw.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (kw *KeyWord) XXX_Merge(src proto.Message) {
	xxx_messageInfo_KeyWord.Merge(kw, src)
}
func (kw *KeyWord) XXX_Size() int {
	return kw.Size()
}
func (kw *KeyWord) XXX_DiscardUnknown() {
	xxx_messageInfo_KeyWord.DiscardUnknown(kw)
}

var xxx_messageInfo_KeyWord proto.InternalMessageInfo

func (kw *KeyWord) GetField() string {
	if kw != nil {
		return kw.Field
	}
	return ""
}

func (kw *KeyWord) GetWord() string {
	if kw != nil {
		return kw.Word
	}
	return ""
}

func init() {
	proto.RegisterType((*KeyWord)(nil), "messages.KeyWord")
}

func init() { proto.RegisterFile("key_word.proto", fileDescriptor_83a5b031d8c2f195) }

var fileDescriptor_83a5b031d8c2f195 = []byte{
	// 123 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0xe2, 0xcb, 0x4e, 0xad, 0x8c,
	0x2f, 0xcf, 0x2f, 0x4a, 0xd1, 0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17, 0xe2, 0xc8, 0x4d, 0x2d, 0x2e,
	0x4e, 0x4c, 0x4f, 0x2d, 0x56, 0x32, 0xe6, 0x62, 0xf7, 0x4e, 0xad, 0x0c, 0xcf, 0x2f, 0x4a, 0x11,
	0x12, 0xe1, 0x62, 0x75, 0xcb, 0x4c, 0xcd, 0x49, 0x91, 0x60, 0x54, 0x60, 0xd4, 0xe0, 0x0c, 0x82,
	0x70, 0x84, 0x84, 0xb8, 0x58, 0x40, 0xb2, 0x12, 0x4c, 0x60, 0x41, 0x30, 0xdb, 0x49, 0xe2, 0xc4,
	0x23, 0x39, 0xc6, 0x0b, 0x8f, 0xe4, 0x18, 0x1f, 0x3c, 0x92, 0x63, 0x9c, 0xf0, 0x58, 0x8e, 0xe1,
	0xc2, 0x63, 0x39, 0x86, 0x1b, 0x8f, 0xe5, 0x18, 0x92, 0xd8, 0xc0, 0xe6, 0x1b, 0x03, 0x02, 0x00,
	0x00, 0xff, 0xff, 0x9c, 0xa3, 0xd2, 0x67, 0x71, 0x00, 0x00, 0x00,
}

func (kw *KeyWord) Marshal() (dAtA []byte, err error) {
	size := kw.Size()
	dAtA = make([]byte, size)
	n, err := kw.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (kw *KeyWord) MarshalTo(dAtA []byte) (int, error) {
	size := kw.Size()
	return kw.MarshalToSizedBuffer(dAtA[:size])
}

func (kw *KeyWord) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(kw.Word) > 0 {
		i -= len(kw.Word)
		copy(dAtA[i:], kw.Word)
		i = encodeVarintKeyWord(dAtA, i, uint64(len(kw.Word)))
		i--
		dAtA[i] = 0x12
	}
	if len(kw.Field) > 0 {
		i -= len(kw.Field)
		copy(dAtA[i:], kw.Field)
		i = encodeVarintKeyWord(dAtA, i, uint64(len(kw.Field)))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func encodeVarintKeyWord(dAtA []byte, offset int, v uint64) int {
	offset -= sovKeyWord(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (kw *KeyWord) Size() (n int) {
	if kw == nil {
		return 0
	}
	var l int
	_ = l
	l = len(kw.Field)
	if l > 0 {
		n += 1 + l + sovKeyWord(uint64(l))
	}
	l = len(kw.Word)
	if l > 0 {
		n += 1 + l + sovKeyWord(uint64(l))
	}
	return n
}

func sovKeyWord(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozKeyWord(x uint64) (n int) {
	return sovKeyWord(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (kw *KeyWord) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowKeyWord
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= uint64(b&0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: KeyWord: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: KeyWord: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Field", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowKeyWord
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthKeyWord
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthKeyWord
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			kw.Field = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Word", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowKeyWord
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthKeyWord
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthKeyWord
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			kw.Word = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipKeyWord(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthKeyWord
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
func skipKeyWord(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowKeyWord
			}
			if iNdEx >= l {
				return 0, io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		wireType := int(wire & 0x7)
		switch wireType {
		case 0:
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowKeyWord
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				iNdEx++
				if dAtA[iNdEx-1] < 0x80 {
					break
				}
			}
		case 1:
			iNdEx += 8
		case 2:
			var length int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowKeyWord
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				length |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if length < 0 {
				return 0, ErrInvalidLengthKeyWord
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupKeyWord
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthKeyWord
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthKeyWord        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowKeyWord          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupKeyWord = fmt.Errorf("proto: unexpected end of group")
)
