package index

import (
	"errors"
	"fmt"
	"github.com/gogo/protobuf/proto"
	"io"
	mathbits "math/bits"
	"strconv"
)

var messageInfoDocId proto.InternalMessageInfo

func encodeIndex(data []byte, offset int, v uint64) int {
	offset -= sovIndex(v)
	base := offset
	for v >= 1<<7 {
		data[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	data[offset] = uint8(v)
	return base
}

func sovIndex(x uint64) (n int) {
	return (mathbits.Len64(x|1) + 6) / 7
}

func skipIndex(data []byte) (n int, err error) {
	l := len(data)
	seed := 0
	depth := 0
	for seed < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, errors.New("integer overflow")
			}
			if seed >= l {
				return 0, io.ErrUnexpectedEOF
			}
			b := data[seed]
			seed++
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
					return 0, errors.New("integer overflow")
				}
				if seed >= l {
					return 0, io.ErrUnexpectedEOF
				}
				seed++
				if data[seed-1] < 0x80 {
					break
				}
			}
		case 1:
			seed += 8
		case 2:
			var length int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, errors.New("integer overflow")
				}
				if seed >= l {
					return 0, io.ErrUnexpectedEOF
				}
				b := data[seed]
				seed++
				length |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if length < 0 {
				return 0, errors.New("negative length found during unmarshalling")
			}
			seed += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, errors.New("unexpected end of group")
			}
			depth--
		case 5:
			seed += 4
		default:
			return 0, errors.New("illegal wireType" + strconv.Itoa(wireType))
		}
		if seed < 0 {
			return 0, errors.New("negative length found during unmarshalling")
		}
		if depth == 0 {
			return seed, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

func encodeVarIntDoc(data []byte, offset int, v uint64) int {
	offset -= sovDoc(v)
	base := offset
	for v >= 1<<7 {
		data[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	data[offset] = uint8(v)
	return base
}

func sovDoc(x uint64) (n int) {
	return (mathbits.Len64(x|1) + 6) / 7
}

func encodeVarIntIndex(data []byte, offset int, v uint64) int {
	offset -= sovIndex(v)
	base := offset
	for v >= 1<<7 {
		data[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	data[offset] = uint8(v)
	return base
}

func sozDoc(x uint64) (n int) {
	return sovDoc((x << 1) ^ uint64(int64(x)>>63))
}

func skipDoc(data []byte) (n int, err error) {
	l := len(data)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, errors.New("integer overflow")
			}
			if iNdEx >= l {
				return 0, io.ErrUnexpectedEOF
			}
			b := data[iNdEx]
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
					return 0, errors.New("integer overflow")
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				iNdEx++
				if data[iNdEx-1] < 0x80 {
					break
				}
			}
		case 1:
			iNdEx += 8
		case 2:
			var length int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, errors.New("integer overflow")
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				b := data[iNdEx]
				iNdEx++
				length |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if length < 0 {
				return 0, errors.New("negative length found during unmarshalling")
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, errors.New("unexpected end of group")
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, errors.New("negative length found during unmarshalling")
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

func skipTermQuery(data []byte) (n int, err error) {
	l := len(data)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, errors.New("integer overflow")
			}
			if iNdEx >= l {
				return 0, io.ErrUnexpectedEOF
			}
			b := data[iNdEx]
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
					return 0, errors.New("integer overflow")
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				iNdEx++
				if data[iNdEx-1] < 0x80 {
					break
				}
			}
		case 1:
			iNdEx += 8
		case 2:
			var length int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, errors.New("integer overflow")
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				b := data[iNdEx]
				iNdEx++
				length |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if length < 0 {
				return 0, errors.New("negative length found during unmarshalling")
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, errors.New("unexpected end of group")
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, errors.New("negative length found during unmarshalling")
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}
