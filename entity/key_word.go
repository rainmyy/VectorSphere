package entity

import (
	"github.com/gogo/protobuf/proto"
)

type Keyword struct {
	Field string `protobuf:"bytes,1,opt,name=Field,proto3" json:"Field,omitempty"`
	Word  string `protobuf:"bytes,2,opt,name=Word,proto3" json:"Word,omitempty"`
}

func (k *Keyword) ToString() string {
	if len(k.Word) > 0 {
		return k.Field + "\001" + k.Word
	}

	return ""
}
func (k *Keyword) Reset()         { *k = Keyword{} }
func (k *Keyword) String() string { return proto.CompactTextString(k) }
func (*Keyword) ProtoMessage()    {}
func (k *Keyword) Unmarshal(b []byte) error {
	return k.Unmarshal(b)
}
func (k *Keyword) MarshalWithDeterministic(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return MessageInfoDocId.Marshal(b, k, deterministic)
	}

	b = b[:cap(b)]
	n, err := k.MarshalToSizedBuffer(b)
	if err != nil {
		return nil, err
	}

	return b[:n], nil
}
func (k *Keyword) Merge(src proto.Message) {
	MessageInfoDocId.Merge(k, src)
}

func (k *Keyword) DiscardUnknown() {
	MessageInfoDocId.DiscardUnknown(k)
}
func (k *Keyword) GetField() string {
	if k != nil {
		return k.Field
	}
	return ""
}

func (k *Keyword) GetWord() string {
	if k != nil {
		return k.Word
	}
	return ""
}
func (k *Keyword) Marshal() (data []byte, err error) {
	size := k.Size()
	data = make([]byte, size)
	n, err := k.MarshalToSizedBuffer(data[:size])
	if err != nil {
		return nil, err
	}
	return data[:n], nil
}

func (k *Keyword) MarshalTo(data []byte) (int, error) {
	size := k.Size()
	return k.MarshalToSizedBuffer(data[:size])
}

func (k *Keyword) MarshalToSizedBuffer(data []byte) (int, error) {
	i := len(data)
	_ = i
	var l int
	_ = l
	if len(k.Word) > 0 {
		i -= len(k.Word)
		copy(data[i:], k.Word)
		i = encodeVarIntDoc(data, i, uint64(len(k.Word)))
		i--
		data[i] = 0x12
	}
	if len(k.Field) > 0 {
		i -= len(k.Field)
		copy(data[i:], k.Field)
		i = encodeVarIntDoc(data, i, uint64(len(k.Field)))
		i--
		data[i] = 0xa
	}
	return len(data) - i, nil
}

func (k *Keyword) Size() (n int) {
	if k == nil {
		return 0
	}
	var l int
	_ = l
	l = len(k.Field)
	if l > 0 {
		n += 1 + l + sovDoc(uint64(l))
	}
	l = len(k.Word)
	if l > 0 {
		n += 1 + l + sovDoc(uint64(l))
	}
	return n
}
