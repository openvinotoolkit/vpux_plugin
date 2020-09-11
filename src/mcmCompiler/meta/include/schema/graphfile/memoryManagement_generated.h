// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_MEMORYMANAGEMENT_MVCNN_H_
#define FLATBUFFERS_GENERATED_MEMORYMANAGEMENT_MVCNN_H_

#include "flatbuffers/flatbuffers.h"

namespace MVCNN {

struct BinaryData;
struct BinaryDataBuilder;
struct BinaryDataT;

struct IndirectDataReference;
struct IndirectDataReferenceBuilder;
struct IndirectDataReferenceT;

struct TensorReference;
struct TensorReferenceBuilder;
struct TensorReferenceT;

enum MemoryLocation {
  MemoryLocation_NULL = 0,
  MemoryLocation_ProgrammableInput = 1,
  MemoryLocation_ProgrammableOutput = 2,
  MemoryLocation_VPU_DDR_Heap = 3,
  MemoryLocation_GraphFile = 4,
  MemoryLocation_VPU_CMX_NN = 5,
  MemoryLocation_VPU_CMX_UPA = 6,
  MemoryLocation_VPU_DDR_BSS = 7,
  MemoryLocation_VPU_CSRAM = 8,
  MemoryLocation_MIN = MemoryLocation_NULL,
  MemoryLocation_MAX = MemoryLocation_VPU_CSRAM
};

inline const MemoryLocation (&EnumValuesMemoryLocation())[9] {
  static const MemoryLocation values[] = {
    MemoryLocation_NULL,
    MemoryLocation_ProgrammableInput,
    MemoryLocation_ProgrammableOutput,
    MemoryLocation_VPU_DDR_Heap,
    MemoryLocation_GraphFile,
    MemoryLocation_VPU_CMX_NN,
    MemoryLocation_VPU_CMX_UPA,
    MemoryLocation_VPU_DDR_BSS,
    MemoryLocation_VPU_CSRAM
  };
  return values;
}

inline const char * const *EnumNamesMemoryLocation() {
  static const char * const names[10] = {
    "NULL",
    "ProgrammableInput",
    "ProgrammableOutput",
    "VPU_DDR_Heap",
    "GraphFile",
    "VPU_CMX_NN",
    "VPU_CMX_UPA",
    "VPU_DDR_BSS",
    "VPU_CSRAM",
    nullptr
  };
  return names;
}

inline const char *EnumNameMemoryLocation(MemoryLocation e) {
  if (flatbuffers::IsOutRange(e, MemoryLocation_NULL, MemoryLocation_VPU_CSRAM)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesMemoryLocation()[index];
}

enum DType {
  DType_NOT_SET = 0,
  DType_FP64 = 1,
  DType_FP32 = 2,
  DType_FP16 = 3,
  DType_FP8 = 4,
  DType_U64 = 5,
  DType_U32 = 6,
  DType_U16 = 7,
  DType_U8 = 8,
  DType_I64 = 9,
  DType_I32 = 10,
  DType_I16 = 11,
  DType_I8 = 12,
  DType_I4 = 13,
  DType_I2 = 14,
  DType_I4X = 15,
  DType_BIN = 16,
  DType_LOG = 17,
  DType_I2X = 18,
  DType_MIN = DType_NOT_SET,
  DType_MAX = DType_I2X
};

inline const DType (&EnumValuesDType())[19] {
  static const DType values[] = {
    DType_NOT_SET,
    DType_FP64,
    DType_FP32,
    DType_FP16,
    DType_FP8,
    DType_U64,
    DType_U32,
    DType_U16,
    DType_U8,
    DType_I64,
    DType_I32,
    DType_I16,
    DType_I8,
    DType_I4,
    DType_I2,
    DType_I4X,
    DType_BIN,
    DType_LOG,
    DType_I2X
  };
  return values;
}

inline const char * const *EnumNamesDType() {
  static const char * const names[20] = {
    "NOT_SET",
    "FP64",
    "FP32",
    "FP16",
    "FP8",
    "U64",
    "U32",
    "U16",
    "U8",
    "I64",
    "I32",
    "I16",
    "I8",
    "I4",
    "I2",
    "I4X",
    "BIN",
    "LOG",
    "I2X",
    nullptr
  };
  return names;
}

inline const char *EnumNameDType(DType e) {
  if (flatbuffers::IsOutRange(e, DType_NOT_SET, DType_I2X)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesDType()[index];
}

struct BinaryDataT : public flatbuffers::NativeTable {
  typedef BinaryData TableType;
  MVCNN::DType underlying_type;
  uint64_t length;
  std::vector<uint64_t> data;
  BinaryDataT()
      : underlying_type(MVCNN::DType_NOT_SET),
        length(0) {
  }
};

struct BinaryData FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef BinaryDataT NativeTableType;
  typedef BinaryDataBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_UNDERLYING_TYPE = 4,
    VT_LENGTH = 6,
    VT_DATA = 8
  };
  MVCNN::DType underlying_type() const {
    return static_cast<MVCNN::DType>(GetField<int8_t>(VT_UNDERLYING_TYPE, 0));
  }
  uint64_t length() const {
    return GetField<uint64_t>(VT_LENGTH, 0);
  }
  const flatbuffers::Vector<uint64_t> *data() const {
    return GetPointer<const flatbuffers::Vector<uint64_t> *>(VT_DATA);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int8_t>(verifier, VT_UNDERLYING_TYPE) &&
           VerifyField<uint64_t>(verifier, VT_LENGTH) &&
           VerifyOffset(verifier, VT_DATA) &&
           verifier.VerifyVector(data()) &&
           verifier.EndTable();
  }
  BinaryDataT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(BinaryDataT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<BinaryData> Pack(flatbuffers::FlatBufferBuilder &_fbb, const BinaryDataT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct BinaryDataBuilder {
  typedef BinaryData Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_underlying_type(MVCNN::DType underlying_type) {
    fbb_.AddElement<int8_t>(BinaryData::VT_UNDERLYING_TYPE, static_cast<int8_t>(underlying_type), 0);
  }
  void add_length(uint64_t length) {
    fbb_.AddElement<uint64_t>(BinaryData::VT_LENGTH, length, 0);
  }
  void add_data(flatbuffers::Offset<flatbuffers::Vector<uint64_t>> data) {
    fbb_.AddOffset(BinaryData::VT_DATA, data);
  }
  explicit BinaryDataBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  BinaryDataBuilder &operator=(const BinaryDataBuilder &);
  flatbuffers::Offset<BinaryData> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<BinaryData>(end);
    return o;
  }
};

inline flatbuffers::Offset<BinaryData> CreateBinaryData(
    flatbuffers::FlatBufferBuilder &_fbb,
    MVCNN::DType underlying_type = MVCNN::DType_NOT_SET,
    uint64_t length = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint64_t>> data = 0) {
  BinaryDataBuilder builder_(_fbb);
  builder_.add_length(length);
  builder_.add_data(data);
  builder_.add_underlying_type(underlying_type);
  return builder_.Finish();
}

inline flatbuffers::Offset<BinaryData> CreateBinaryDataDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    MVCNN::DType underlying_type = MVCNN::DType_NOT_SET,
    uint64_t length = 0,
    const std::vector<uint64_t> *data = nullptr) {
  auto data__ = data ? _fbb.CreateVector<uint64_t>(*data) : 0;
  return MVCNN::CreateBinaryData(
      _fbb,
      underlying_type,
      length,
      data__);
}

flatbuffers::Offset<BinaryData> CreateBinaryData(flatbuffers::FlatBufferBuilder &_fbb, const BinaryDataT *_o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);

struct IndirectDataReferenceT : public flatbuffers::NativeTable {
  typedef IndirectDataReference TableType;
  uint64_t data_index;
  uint64_t sparsity_index;
  uint64_t storage_element_index;
  uint32_t storage_element_size;
  IndirectDataReferenceT()
      : data_index(999999999999999999ULL),
        sparsity_index(999999999999999999ULL),
        storage_element_index(999999999999999999ULL),
        storage_element_size(0) {
  }
};

struct IndirectDataReference FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef IndirectDataReferenceT NativeTableType;
  typedef IndirectDataReferenceBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DATA_INDEX = 4,
    VT_SPARSITY_INDEX = 6,
    VT_STORAGE_ELEMENT_INDEX = 8,
    VT_STORAGE_ELEMENT_SIZE = 10
  };
  /// Index/Offsets from the start of a memory location (see MemoryLocation)
  ///
  /// The base address of a memory location is calculated via "MemoryLocation" and "Locale_Index".
  /// The Memory Location informs the device "what class" of Memory we are dealing with and the
  /// Locale Index informs "which one" when there are multiple.
  /// (e.g. Use the 2nd Programmable Input Buffer)
  /// (e.g. Use the 8th Index of the GraphFile's Binary Data)
  /// The relative offset to the data is then below as data_index.
  /// In some circumstances, you may require a larger buffer than nessicary (e.g. software overwrite)
  /// This buffer should be allocated and then the "leading_offset" and "trailing_offset" fields of TensorReference
  /// used to access the 'corrected' start of the tensor. This should be a rare circumstance.
  /// For device buffers, most of the time we are dealing with a starting pointer and a total size.
  uint64_t data_index() const {
    return GetField<uint64_t>(VT_DATA_INDEX, 999999999999999999ULL);
  }
  uint64_t sparsity_index() const {
    return GetField<uint64_t>(VT_SPARSITY_INDEX, 999999999999999999ULL);
  }
  uint64_t storage_element_index() const {
    return GetField<uint64_t>(VT_STORAGE_ELEMENT_INDEX, 999999999999999999ULL);
  }
  uint32_t storage_element_size() const {
    return GetField<uint32_t>(VT_STORAGE_ELEMENT_SIZE, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint64_t>(verifier, VT_DATA_INDEX) &&
           VerifyField<uint64_t>(verifier, VT_SPARSITY_INDEX) &&
           VerifyField<uint64_t>(verifier, VT_STORAGE_ELEMENT_INDEX) &&
           VerifyField<uint32_t>(verifier, VT_STORAGE_ELEMENT_SIZE) &&
           verifier.EndTable();
  }
  IndirectDataReferenceT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(IndirectDataReferenceT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<IndirectDataReference> Pack(flatbuffers::FlatBufferBuilder &_fbb, const IndirectDataReferenceT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct IndirectDataReferenceBuilder {
  typedef IndirectDataReference Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_data_index(uint64_t data_index) {
    fbb_.AddElement<uint64_t>(IndirectDataReference::VT_DATA_INDEX, data_index, 999999999999999999ULL);
  }
  void add_sparsity_index(uint64_t sparsity_index) {
    fbb_.AddElement<uint64_t>(IndirectDataReference::VT_SPARSITY_INDEX, sparsity_index, 999999999999999999ULL);
  }
  void add_storage_element_index(uint64_t storage_element_index) {
    fbb_.AddElement<uint64_t>(IndirectDataReference::VT_STORAGE_ELEMENT_INDEX, storage_element_index, 999999999999999999ULL);
  }
  void add_storage_element_size(uint32_t storage_element_size) {
    fbb_.AddElement<uint32_t>(IndirectDataReference::VT_STORAGE_ELEMENT_SIZE, storage_element_size, 0);
  }
  explicit IndirectDataReferenceBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  IndirectDataReferenceBuilder &operator=(const IndirectDataReferenceBuilder &);
  flatbuffers::Offset<IndirectDataReference> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<IndirectDataReference>(end);
    return o;
  }
};

inline flatbuffers::Offset<IndirectDataReference> CreateIndirectDataReference(
    flatbuffers::FlatBufferBuilder &_fbb,
    uint64_t data_index = 999999999999999999ULL,
    uint64_t sparsity_index = 999999999999999999ULL,
    uint64_t storage_element_index = 999999999999999999ULL,
    uint32_t storage_element_size = 0) {
  IndirectDataReferenceBuilder builder_(_fbb);
  builder_.add_storage_element_index(storage_element_index);
  builder_.add_sparsity_index(sparsity_index);
  builder_.add_data_index(data_index);
  builder_.add_storage_element_size(storage_element_size);
  return builder_.Finish();
}

flatbuffers::Offset<IndirectDataReference> CreateIndirectDataReference(flatbuffers::FlatBufferBuilder &_fbb, const IndirectDataReferenceT *_o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);

struct TensorReferenceT : public flatbuffers::NativeTable {
  typedef TensorReference TableType;
  std::string name;
  std::vector<uint32_t> dimensions;
  std::vector<uint32_t> strides;
  uint32_t leading_offset;
  uint32_t trailing_offset;
  std::unique_ptr<MVCNN::IndirectDataReferenceT> data;
  MVCNN::MemoryLocation locale;
  std::vector<uint32_t> locale_index;
  MVCNN::DType data_dtype;
  std::vector<uint8_t> quant_zero;
  std::vector<float> quant_scale;
  std::vector<uint16_t> quant_mult;
  std::vector<uint8_t> quant_shift;
  int8_t quant_post_shift_right;
  TensorReferenceT()
      : leading_offset(0),
        trailing_offset(0),
        locale(MVCNN::MemoryLocation_NULL),
        data_dtype(MVCNN::DType_NOT_SET),
        quant_post_shift_right(0) {
  }
};

struct TensorReference FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef TensorReferenceT NativeTableType;
  typedef TensorReferenceBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_NAME = 4,
    VT_DIMENSIONS = 6,
    VT_STRIDES = 8,
    VT_LEADING_OFFSET = 10,
    VT_TRAILING_OFFSET = 12,
    VT_DATA = 14,
    VT_LOCALE = 16,
    VT_LOCALE_INDEX = 18,
    VT_DATA_DTYPE = 20,
    VT_QUANT_ZERO = 22,
    VT_QUANT_SCALE = 24,
    VT_QUANT_MULT = 26,
    VT_QUANT_SHIFT = 28,
    VT_QUANT_POST_SHIFT_RIGHT = 30
  };
  /// Information on how to access a Tensor
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  const flatbuffers::Vector<uint32_t> *dimensions() const {
    return GetPointer<const flatbuffers::Vector<uint32_t> *>(VT_DIMENSIONS);
  }
  const flatbuffers::Vector<uint32_t> *strides() const {
    return GetPointer<const flatbuffers::Vector<uint32_t> *>(VT_STRIDES);
  }
  uint32_t leading_offset() const {
    return GetField<uint32_t>(VT_LEADING_OFFSET, 0);
  }
  uint32_t trailing_offset() const {
    return GetField<uint32_t>(VT_TRAILING_OFFSET, 0);
  }
  const MVCNN::IndirectDataReference *data() const {
    return GetPointer<const MVCNN::IndirectDataReference *>(VT_DATA);
  }
  MVCNN::MemoryLocation locale() const {
    return static_cast<MVCNN::MemoryLocation>(GetField<int8_t>(VT_LOCALE, 0));
  }
  const flatbuffers::Vector<uint32_t> *locale_index() const {
    return GetPointer<const flatbuffers::Vector<uint32_t> *>(VT_LOCALE_INDEX);
  }
  MVCNN::DType data_dtype() const {
    return static_cast<MVCNN::DType>(GetField<int8_t>(VT_DATA_DTYPE, 0));
  }
  const flatbuffers::Vector<uint8_t> *quant_zero() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_QUANT_ZERO);
  }
  const flatbuffers::Vector<float> *quant_scale() const {
    return GetPointer<const flatbuffers::Vector<float> *>(VT_QUANT_SCALE);
  }
  const flatbuffers::Vector<uint16_t> *quant_mult() const {
    return GetPointer<const flatbuffers::Vector<uint16_t> *>(VT_QUANT_MULT);
  }
  const flatbuffers::Vector<uint8_t> *quant_shift() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_QUANT_SHIFT);
  }
  int8_t quant_post_shift_right() const {
    return GetField<int8_t>(VT_QUANT_POST_SHIFT_RIGHT, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           VerifyOffset(verifier, VT_DIMENSIONS) &&
           verifier.VerifyVector(dimensions()) &&
           VerifyOffset(verifier, VT_STRIDES) &&
           verifier.VerifyVector(strides()) &&
           VerifyField<uint32_t>(verifier, VT_LEADING_OFFSET) &&
           VerifyField<uint32_t>(verifier, VT_TRAILING_OFFSET) &&
           VerifyOffset(verifier, VT_DATA) &&
           verifier.VerifyTable(data()) &&
           VerifyField<int8_t>(verifier, VT_LOCALE) &&
           VerifyOffset(verifier, VT_LOCALE_INDEX) &&
           verifier.VerifyVector(locale_index()) &&
           VerifyField<int8_t>(verifier, VT_DATA_DTYPE) &&
           VerifyOffset(verifier, VT_QUANT_ZERO) &&
           verifier.VerifyVector(quant_zero()) &&
           VerifyOffset(verifier, VT_QUANT_SCALE) &&
           verifier.VerifyVector(quant_scale()) &&
           VerifyOffset(verifier, VT_QUANT_MULT) &&
           verifier.VerifyVector(quant_mult()) &&
           VerifyOffset(verifier, VT_QUANT_SHIFT) &&
           verifier.VerifyVector(quant_shift()) &&
           VerifyField<int8_t>(verifier, VT_QUANT_POST_SHIFT_RIGHT) &&
           verifier.EndTable();
  }
  TensorReferenceT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(TensorReferenceT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<TensorReference> Pack(flatbuffers::FlatBufferBuilder &_fbb, const TensorReferenceT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct TensorReferenceBuilder {
  typedef TensorReference Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(TensorReference::VT_NAME, name);
  }
  void add_dimensions(flatbuffers::Offset<flatbuffers::Vector<uint32_t>> dimensions) {
    fbb_.AddOffset(TensorReference::VT_DIMENSIONS, dimensions);
  }
  void add_strides(flatbuffers::Offset<flatbuffers::Vector<uint32_t>> strides) {
    fbb_.AddOffset(TensorReference::VT_STRIDES, strides);
  }
  void add_leading_offset(uint32_t leading_offset) {
    fbb_.AddElement<uint32_t>(TensorReference::VT_LEADING_OFFSET, leading_offset, 0);
  }
  void add_trailing_offset(uint32_t trailing_offset) {
    fbb_.AddElement<uint32_t>(TensorReference::VT_TRAILING_OFFSET, trailing_offset, 0);
  }
  void add_data(flatbuffers::Offset<MVCNN::IndirectDataReference> data) {
    fbb_.AddOffset(TensorReference::VT_DATA, data);
  }
  void add_locale(MVCNN::MemoryLocation locale) {
    fbb_.AddElement<int8_t>(TensorReference::VT_LOCALE, static_cast<int8_t>(locale), 0);
  }
  void add_locale_index(flatbuffers::Offset<flatbuffers::Vector<uint32_t>> locale_index) {
    fbb_.AddOffset(TensorReference::VT_LOCALE_INDEX, locale_index);
  }
  void add_data_dtype(MVCNN::DType data_dtype) {
    fbb_.AddElement<int8_t>(TensorReference::VT_DATA_DTYPE, static_cast<int8_t>(data_dtype), 0);
  }
  void add_quant_zero(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> quant_zero) {
    fbb_.AddOffset(TensorReference::VT_QUANT_ZERO, quant_zero);
  }
  void add_quant_scale(flatbuffers::Offset<flatbuffers::Vector<float>> quant_scale) {
    fbb_.AddOffset(TensorReference::VT_QUANT_SCALE, quant_scale);
  }
  void add_quant_mult(flatbuffers::Offset<flatbuffers::Vector<uint16_t>> quant_mult) {
    fbb_.AddOffset(TensorReference::VT_QUANT_MULT, quant_mult);
  }
  void add_quant_shift(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> quant_shift) {
    fbb_.AddOffset(TensorReference::VT_QUANT_SHIFT, quant_shift);
  }
  void add_quant_post_shift_right(int8_t quant_post_shift_right) {
    fbb_.AddElement<int8_t>(TensorReference::VT_QUANT_POST_SHIFT_RIGHT, quant_post_shift_right, 0);
  }
  explicit TensorReferenceBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  TensorReferenceBuilder &operator=(const TensorReferenceBuilder &);
  flatbuffers::Offset<TensorReference> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<TensorReference>(end);
    return o;
  }
};

inline flatbuffers::Offset<TensorReference> CreateTensorReference(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> name = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint32_t>> dimensions = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint32_t>> strides = 0,
    uint32_t leading_offset = 0,
    uint32_t trailing_offset = 0,
    flatbuffers::Offset<MVCNN::IndirectDataReference> data = 0,
    MVCNN::MemoryLocation locale = MVCNN::MemoryLocation_NULL,
    flatbuffers::Offset<flatbuffers::Vector<uint32_t>> locale_index = 0,
    MVCNN::DType data_dtype = MVCNN::DType_NOT_SET,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> quant_zero = 0,
    flatbuffers::Offset<flatbuffers::Vector<float>> quant_scale = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint16_t>> quant_mult = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> quant_shift = 0,
    int8_t quant_post_shift_right = 0) {
  TensorReferenceBuilder builder_(_fbb);
  builder_.add_quant_shift(quant_shift);
  builder_.add_quant_mult(quant_mult);
  builder_.add_quant_scale(quant_scale);
  builder_.add_quant_zero(quant_zero);
  builder_.add_locale_index(locale_index);
  builder_.add_data(data);
  builder_.add_trailing_offset(trailing_offset);
  builder_.add_leading_offset(leading_offset);
  builder_.add_strides(strides);
  builder_.add_dimensions(dimensions);
  builder_.add_name(name);
  builder_.add_quant_post_shift_right(quant_post_shift_right);
  builder_.add_data_dtype(data_dtype);
  builder_.add_locale(locale);
  return builder_.Finish();
}

inline flatbuffers::Offset<TensorReference> CreateTensorReferenceDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *name = nullptr,
    const std::vector<uint32_t> *dimensions = nullptr,
    const std::vector<uint32_t> *strides = nullptr,
    uint32_t leading_offset = 0,
    uint32_t trailing_offset = 0,
    flatbuffers::Offset<MVCNN::IndirectDataReference> data = 0,
    MVCNN::MemoryLocation locale = MVCNN::MemoryLocation_NULL,
    const std::vector<uint32_t> *locale_index = nullptr,
    MVCNN::DType data_dtype = MVCNN::DType_NOT_SET,
    const std::vector<uint8_t> *quant_zero = nullptr,
    const std::vector<float> *quant_scale = nullptr,
    const std::vector<uint16_t> *quant_mult = nullptr,
    const std::vector<uint8_t> *quant_shift = nullptr,
    int8_t quant_post_shift_right = 0) {
  auto name__ = name ? _fbb.CreateString(name) : 0;
  auto dimensions__ = dimensions ? _fbb.CreateVector<uint32_t>(*dimensions) : 0;
  auto strides__ = strides ? _fbb.CreateVector<uint32_t>(*strides) : 0;
  auto locale_index__ = locale_index ? _fbb.CreateVector<uint32_t>(*locale_index) : 0;
  auto quant_zero__ = quant_zero ? _fbb.CreateVector<uint8_t>(*quant_zero) : 0;
  auto quant_scale__ = quant_scale ? _fbb.CreateVector<float>(*quant_scale) : 0;
  auto quant_mult__ = quant_mult ? _fbb.CreateVector<uint16_t>(*quant_mult) : 0;
  auto quant_shift__ = quant_shift ? _fbb.CreateVector<uint8_t>(*quant_shift) : 0;
  return MVCNN::CreateTensorReference(
      _fbb,
      name__,
      dimensions__,
      strides__,
      leading_offset,
      trailing_offset,
      data,
      locale,
      locale_index__,
      data_dtype,
      quant_zero__,
      quant_scale__,
      quant_mult__,
      quant_shift__,
      quant_post_shift_right);
}

flatbuffers::Offset<TensorReference> CreateTensorReference(flatbuffers::FlatBufferBuilder &_fbb, const TensorReferenceT *_o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);

inline BinaryDataT *BinaryData::UnPack(const flatbuffers::resolver_function_t *_resolver) const {
  std::unique_ptr<MVCNN::BinaryDataT> _o = std::unique_ptr<MVCNN::BinaryDataT>(new BinaryDataT());
  UnPackTo(_o.get(), _resolver);
  return _o.release();
}

inline void BinaryData::UnPackTo(BinaryDataT *_o, const flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = underlying_type(); _o->underlying_type = _e; }
  { auto _e = length(); _o->length = _e; }
  { auto _e = data(); if (_e) { _o->data.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->data[_i] = _e->Get(_i); } } }
}

inline flatbuffers::Offset<BinaryData> BinaryData::Pack(flatbuffers::FlatBufferBuilder &_fbb, const BinaryDataT* _o, const flatbuffers::rehasher_function_t *_rehasher) {
  return CreateBinaryData(_fbb, _o, _rehasher);
}

inline flatbuffers::Offset<BinaryData> CreateBinaryData(flatbuffers::FlatBufferBuilder &_fbb, const BinaryDataT *_o, const flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { flatbuffers::FlatBufferBuilder *__fbb; const BinaryDataT* __o; const flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _underlying_type = _o->underlying_type;
  auto _length = _o->length;
  auto _data = _fbb.CreateVector(_o->data);
  return MVCNN::CreateBinaryData(
      _fbb,
      _underlying_type,
      _length,
      _data);
}

inline IndirectDataReferenceT *IndirectDataReference::UnPack(const flatbuffers::resolver_function_t *_resolver) const {
  std::unique_ptr<MVCNN::IndirectDataReferenceT> _o = std::unique_ptr<MVCNN::IndirectDataReferenceT>(new IndirectDataReferenceT());
  UnPackTo(_o.get(), _resolver);
  return _o.release();
}

inline void IndirectDataReference::UnPackTo(IndirectDataReferenceT *_o, const flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = data_index(); _o->data_index = _e; }
  { auto _e = sparsity_index(); _o->sparsity_index = _e; }
  { auto _e = storage_element_index(); _o->storage_element_index = _e; }
  { auto _e = storage_element_size(); _o->storage_element_size = _e; }
}

inline flatbuffers::Offset<IndirectDataReference> IndirectDataReference::Pack(flatbuffers::FlatBufferBuilder &_fbb, const IndirectDataReferenceT* _o, const flatbuffers::rehasher_function_t *_rehasher) {
  return CreateIndirectDataReference(_fbb, _o, _rehasher);
}

inline flatbuffers::Offset<IndirectDataReference> CreateIndirectDataReference(flatbuffers::FlatBufferBuilder &_fbb, const IndirectDataReferenceT *_o, const flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { flatbuffers::FlatBufferBuilder *__fbb; const IndirectDataReferenceT* __o; const flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _data_index = _o->data_index;
  auto _sparsity_index = _o->sparsity_index;
  auto _storage_element_index = _o->storage_element_index;
  auto _storage_element_size = _o->storage_element_size;
  return MVCNN::CreateIndirectDataReference(
      _fbb,
      _data_index,
      _sparsity_index,
      _storage_element_index,
      _storage_element_size);
}

inline TensorReferenceT *TensorReference::UnPack(const flatbuffers::resolver_function_t *_resolver) const {
  std::unique_ptr<MVCNN::TensorReferenceT> _o = std::unique_ptr<MVCNN::TensorReferenceT>(new TensorReferenceT());
  UnPackTo(_o.get(), _resolver);
  return _o.release();
}

inline void TensorReference::UnPackTo(TensorReferenceT *_o, const flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = name(); if (_e) _o->name = _e->str(); }
  { auto _e = dimensions(); if (_e) { _o->dimensions.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->dimensions[_i] = _e->Get(_i); } } }
  { auto _e = strides(); if (_e) { _o->strides.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->strides[_i] = _e->Get(_i); } } }
  { auto _e = leading_offset(); _o->leading_offset = _e; }
  { auto _e = trailing_offset(); _o->trailing_offset = _e; }
  { auto _e = data(); if (_e) _o->data = std::unique_ptr<MVCNN::IndirectDataReferenceT>(_e->UnPack(_resolver)); }
  { auto _e = locale(); _o->locale = _e; }
  { auto _e = locale_index(); if (_e) { _o->locale_index.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->locale_index[_i] = _e->Get(_i); } } }
  { auto _e = data_dtype(); _o->data_dtype = _e; }
  { auto _e = quant_zero(); if (_e) { _o->quant_zero.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->quant_zero[_i] = _e->Get(_i); } } }
  { auto _e = quant_scale(); if (_e) { _o->quant_scale.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->quant_scale[_i] = _e->Get(_i); } } }
  { auto _e = quant_mult(); if (_e) { _o->quant_mult.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->quant_mult[_i] = _e->Get(_i); } } }
  { auto _e = quant_shift(); if (_e) { _o->quant_shift.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->quant_shift[_i] = _e->Get(_i); } } }
  { auto _e = quant_post_shift_right(); _o->quant_post_shift_right = _e; }
}

inline flatbuffers::Offset<TensorReference> TensorReference::Pack(flatbuffers::FlatBufferBuilder &_fbb, const TensorReferenceT* _o, const flatbuffers::rehasher_function_t *_rehasher) {
  return CreateTensorReference(_fbb, _o, _rehasher);
}

inline flatbuffers::Offset<TensorReference> CreateTensorReference(flatbuffers::FlatBufferBuilder &_fbb, const TensorReferenceT *_o, const flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { flatbuffers::FlatBufferBuilder *__fbb; const TensorReferenceT* __o; const flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _name = _o->name.empty() ? _fbb.CreateSharedString("") : _fbb.CreateString(_o->name);
  auto _dimensions = _fbb.CreateVector(_o->dimensions);
  auto _strides = _fbb.CreateVector(_o->strides);
  auto _leading_offset = _o->leading_offset;
  auto _trailing_offset = _o->trailing_offset;
  auto _data = _o->data ? CreateIndirectDataReference(_fbb, _o->data.get(), _rehasher) : 0;
  auto _locale = _o->locale;
  auto _locale_index = _fbb.CreateVector(_o->locale_index);
  auto _data_dtype = _o->data_dtype;
  auto _quant_zero = _fbb.CreateVector(_o->quant_zero);
  auto _quant_scale = _fbb.CreateVector(_o->quant_scale);
  auto _quant_mult = _fbb.CreateVector(_o->quant_mult);
  auto _quant_shift = _fbb.CreateVector(_o->quant_shift);
  auto _quant_post_shift_right = _o->quant_post_shift_right;
  return MVCNN::CreateTensorReference(
      _fbb,
      _name,
      _dimensions,
      _strides,
      _leading_offset,
      _trailing_offset,
      _data,
      _locale,
      _locale_index,
      _data_dtype,
      _quant_zero,
      _quant_scale,
      _quant_mult,
      _quant_shift,
      _quant_post_shift_right);
}

}  // namespace MVCNN

#endif  // FLATBUFFERS_GENERATED_MEMORYMANAGEMENT_MVCNN_H_
