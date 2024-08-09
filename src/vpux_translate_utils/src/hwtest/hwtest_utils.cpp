//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/hwtest/hwtest_utils.hpp"

#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/custom_float.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>

#include <numeric>

namespace vpux {
namespace hwtest {

namespace {

constexpr auto NUM_LUT_CFG_REGS = 64;
constexpr auto NUM_LUT_DATA_REGS = 256;
constexpr auto NUM_LUT_SAT_REGS = 8;
constexpr auto NUM_LUT_RESERVED_REGS = 7;

// Rsqrt
const std::vector<std::vector<uint16_t>> RSQRT_SATURATION_TABLE_LUT = {
        {1024, 0, 65535},      {31744, 30719, 0},     {64512, 32768, 65535}, {65535, 65535, 65534},
        {65535, 65535, 65534}, {65535, 65535, 65534}, {65535, 65535, 65534}, {65535, 65535, 65534}};

const std::vector<std::vector<uint16_t>> RSQRT_SLOPE_INTERCEPT_LUT = {
        {51561, 16808}, {51443, 16765}, {51342, 16725}, {51253, 16689}, {51151, 16655}, {51013, 16624}, {50890, 16595},
        {50780, 16568}, {50682, 16542}, {50593, 16519}, {50512, 16496}, {50438, 16475}, {50371, 16455}, {50310, 16436},
        {50253, 16418}, {50201, 16400}, {53158, 17408}, {52993, 17347}, {52849, 17291}, {52724, 17239}, {52613, 17192},
        {52516, 17148}, {52429, 17107}, {52351, 17068}, {52282, 17032}, {52213, 16998}, {52099, 16967}, {51995, 16937},
        {51900, 16908}, {51813, 16881}, {51733, 16856}, {51660, 16831}};

const std::vector<uint16_t> RSQRT_CONFIG_LUT = {
        137, 133, 133, 132, 132, 131, 131, 130, 130, 129, 129, 128, 128, 127, 127, 126, 126, 125, 125, 124, 124, 123,
        123, 122, 122, 121, 121, 120, 120, 119, 119, 0,   255, 134, 133, 133, 132, 132, 131, 131, 130, 130, 129, 129,
        128, 128, 127, 127, 126, 126, 125, 125, 124, 124, 123, 123, 122, 122, 121, 121, 120, 120, 119, 0};

// Sigmoid
const std::vector<std::vector<uint16_t>> SIGMOID_SATURATION_TABLE_LUT = {
        {5121, 0, 14336},      {37889, 32768, 14335}, {31744, 18651, 15360}, {64512, 51419, 0},
        {65535, 65535, 65534}, {65535, 65535, 65534}, {65535, 65535, 65534}, {65535, 65535, 65534}};

const std::vector<std::vector<uint16_t>> SIGMOID_SLOPE_INTERCEPT_LUT = {
        {13312, 14337}, {13312, 14338}, {13312, 14340}, {13312, 14344}, {13311, 14352}, {13307, 14368}, {13293, 14400},
        {13262, 14463}, {13217, 14526}, {13158, 14587}, {13088, 14646}, {13008, 14703}, {12921, 14757}, {12828, 14809},
        {12731, 14858}, {12633, 14904}, {12535, 14947}, {12438, 14986}, {12344, 15023}, {12218, 15057}, {12044, 15088},
        {11881, 15116}, {11728, 15141}, {11585, 15165}, {11452, 15186}, {11330, 15205}, {11173, 15222}, {10969, 15237},
        {10784, 15251}, {10617, 15263}, {10465, 15274}, {10329, 15284}, {10172, 15292}, {9953, 15300},  {9756, 15307},
        {9580, 15313},  {9423, 15318},  {9222, 15323},  {8784, 15331},  {8434, 15337},  {8124, 15342},  {7691, 15346},
        {7352, 15349},  {7005, 15352},  {6591, 15354},  {6267, 15355},  {5884, 15356},  {5490, 15357},  {5182, 15358},
        {4764, 15358},  {4390, 15359},  {4100, 15359},  {3648, 15359},  {704, 15359},   {13312, 14335}, {13312, 14334},
        {13312, 14332}, {13312, 14328}, {13312, 14320}, {13311, 14304}, {13307, 14272}, {13293, 14208}, {13262, 14081},
        {13217, 13956}, {13158, 13834}, {13088, 13716}, {13008, 13602}, {12921, 13493}, {12863, 13390}, {12840, 13365},
        {12816, 13340}, {12792, 13316}, {12768, 13272}, {12744, 13225}, {12719, 13179}, {12695, 13133}, {12670, 13088},
        {12646, 13044}, {12621, 13001}, {12596, 12959}, {12572, 12917}, {12547, 12876}, {12523, 12836}, {12498, 12797},
        {12474, 12758}, {12450, 12721}, {12426, 12684}, {12402, 12647}, {12379, 12612}, {12355, 12577}, {12332, 12543},
        {12309, 12509}, {12285, 12477}, {12240, 12445}, {12195, 12414}, {12151, 12383}, {12108, 12353}, {12065, 12324},
        {12023, 12296}, {11981, 12247}, {11941, 12193}, {11901, 12140}, {11861, 12088}, {11822, 12037}, {11784, 11988},
        {11746, 11940}, {11709, 11893}, {11672, 11847}, {11637, 11802}, {11602, 11759}, {11567, 11716}, {11533, 11675},
        {11500, 11634}, {11468, 11595}, {11436, 11557}, {11405, 11519}, {11374, 11483}, {11345, 11447}, {11315, 11413},
        {11287, 11379}, {11254, 11347}, {11199, 11315}, {11146, 11284}, {11094, 11244}, {11043, 11185}, {10993, 11128},
        {10945, 11072}, {10897, 11018}, {10851, 10966}, {10806, 10915}, {10762, 10865}, {10719, 10817}, {10677, 10770},
        {10636, 10724}, {10596, 10680}, {10557, 10637}, {10520, 10595}, {10483, 10554}, {10447, 10515}, {10412, 10476},
        {10378, 10439}, {10344, 10403}, {10312, 10368}, {10281, 10333}, {10250, 10300}, {10201, 10268}, {10143, 10233},
        {10086, 10172}, {10031, 10113}, {9978, 10056},  {9926, 10000},  {9875, 9946},   {9826, 9893},   {9778, 9842},
        {9732, 9793},   {9687, 9745},   {9643, 9698},   {9600, 9653},   {9559, 9609},   {9518, 9566},   {9479, 9525},
        {9441, 9484},   {9404, 9445},   {9368, 9407},   {9317, 9371},   {9251, 9301},   {9163, 9234},   {9045, 9128},
        {8935, 9011},   {8830, 8901},   {8732, 8797},   {8639, 8700},   {8552, 8608},   {8470, 8522},   {8393, 8441},
        {8320, 8364},   {8251, 8292},   {8181, 8225},   {8059, 8130},   {7945, 8011},   {7837, 7898},   {7736, 7793},
        {7641, 7694},   {7551, 7600},   {7467, 7512},   {7387, 7430},   {7313, 7352},   {7242, 7279},   {7176, 7211},
        {7060, 7125},   {6944, 7004},   {6834, 6890},   {6731, 6783},   {6634, 6683},   {6543, 6588},   {6457, 6500},
        {6376, 6416},   {6300, 6338},   {6229, 6264},   {6162, 6195},   {6055, 6116},   {5936, 5994},   {5825, 5879},
        {5721, 5771},   {5623, 5670},   {5530, 5575},   {5444, 5485},   {5362, 5401},   {5286, 5322},   {5214, 5248},
        {5146, 5178},   {5045, 5106},   {4926, 4983},   {4814, 4867},   {4709, 4758},   {4610, 4656},   {4517, 4560},
        {4429, 4470},   {4347, 4386},   {4270, 4306},   {4197, 4231},   {4129, 4161},   {4035, 4094},   {3914, 3970},
        {3801, 3854},   {3695, 3745},   {3595, 3642},   {3502, 3545},   {3416, 3455},   {3334, 3369},   {3256, 3289},
        {3183, 3214},   {3114, 3144},   {3027, 3077},   {2905, 2958},   {2791, 2841},   {2684, 2730},   {2584, 2627},
        {2489, 2530},   {2401, 2439},   {2317, 2353},   {2239, 2273},   {2165, 2197},   {2096, 2126},   {2014, 2059},
        {1892, 1945},   {1778, 1827},   {1670, 1716},   {1569, 1612},   {1474, 1515},   {1385, 1423},   {1301, 1337},
        {1222, 1256},   {1148, 1180},   {1078, 1108},   {1013, 1041}};

const std::vector<uint16_t> SIGMOID_CONFIG_LUT = {
        54,   54,   54,   54,  54,  54,  0,   1,   2,   3,   4,   5,   6,   1031, 2057, 3085,
        4117, 4133, 53,   54,  54,  54,  54,  54,  54,  54,  54,  54,  54,  54,   54,   54,
        256,  256,  256,  256, 256, 54,  55,  56,  57,  58,  59,  60,  61,  1086, 2112, 5188,
        6244, 6308, 7396, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,  256,  256};

// Sin
const std::vector<std::vector<uint16_t>> SIN_SATURATION_TABLE_LUT = {
        {10240, 0, 65535},     {43008, 32768, 65535}, {31744, 16383, 15360}, {64512, 49151, 48128},
        {65535, 65535, 65534}, {65535, 65535, 65534}, {65535, 65535, 65534}, {65535, 65535, 65534}};

const std::vector<std::vector<uint16_t>> SIN_SLOPE_INTERCEPT_LUT = {
        {15358, 10240},   {15351, 11263},  {15335, 12283}, {15311, 12791}, {15288, 13291},   {15270, 13425},
        {15251, 13547},   {15229, 13668},  {15206, 13788}, {15180, 13907}, {15153, 14023},   {15124, 14138},
        {15078, 14252},   {15009, 14404},  {14935, 14510}, {14853, 14612}, {14766, 14708},   {14673, 14799},
        {14575, 14884},   {14472, 14963},  {14392, 15035}, {14337, 15069}, {14227, 15101},   {14113, 15131},
        {13998, 15160},   {13881, 15187},  {13762, 15211}, {13642, 15234}, {13521, 15256},   {13398, 15275},
        {13236, 15292},   {12987, 15307},  {12736, 15321}, {12483, 15332}, {12171, 15342},   {11662, 15349},
        {11039, 15355},   {9791, 15358},   {40705, 15360}, {43232, 15359}, {44143, 15357},   {44653, 15353},
        {45109, 15346},   {45362, 15338},  {45614, 15327}, {45864, 15315}, {46096, 15300},   {46220, 15284},
        {46342, 15266},   {46463, 15246},  {46582, 15224}, {46700, 15200}, {0xFFFF, 0xFFFF}, {0xFFFF, 0xFFFF},
        {0xFFFF, 0xFFFF}, {0xFFFF, 0xFFFF}};  // 4 blocks of padding

const std::vector<uint16_t> SIN_CONFIG_LUT = {52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 0,  1,  1026, 3076, 3084, 5140,
                                              52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52,   52,   52,   52,
                                              0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,    0,    0,    0,
                                              0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,    0,    0,    0};

// Tanh
const std::vector<std::vector<uint16_t>> TANH_SATURATION_TABLE_LUT = {
        {10240, 0, 65535},     {43008, 32768, 65535}, {31744, 17448, 15360}, {64512, 50216, 48128},
        {65535, 65535, 65534}, {65535, 65535, 65534}, {65535, 65535, 65534}, {65535, 65535, 65534}};

const std::vector<std::vector<uint16_t>> TANH_SLOPE_INTERCEPT_LUT = {
        {15355, 10239}, {15347, 11261}, {15336, 11772}, {15320, 12277}, {15301, 12534}, {15278, 12782}, {15251, 13028},
        {15222, 13270}, {15190, 13411}, {15154, 13528}, {15117, 13643}, {15077, 13756}, {15035, 13866}, {14991, 13974},
        {14946, 14079}, {14900, 14181}, {14852, 14280}, {14804, 14356}, {14755, 14403}, {14706, 14448}, {14657, 14491},
        {14607, 14533}, {14559, 14574}, {14510, 14613}, {14462, 14650}, {14415, 14686}, {14368, 14721}, {14310, 14754},
        {14221, 14785}, {14135, 14815}, {14050, 14844}, {13929, 14872}, {13776, 14923}, {13633, 14969}, {13500, 15011},
        {13378, 15049}, {13221, 15083}, {13017, 15114}, {12832, 15141}, {12665, 15166}, {12513, 15188}, {12377, 15207},
        {12220, 15224}, {12001, 15240}, {11804, 15254}, {11628, 15266}, {11471, 15277}, {11270, 15286}, {10832, 15302},
        {10482, 15315}, {10172, 15325}, {9739, 15333},  {9400, 15339},  {9054, 15343},  {8639, 15347},  {8315, 15350},
        {7932, 15352},  {7538, 15354},  {7230, 15355},  {6812, 15356},  {6439, 15357},  {6147, 15358},  {5696, 15358},
        {2431, 15359}};

const std::vector<uint16_t> TANH_CONFIG_LUT = {
        63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 0, 1025, 2051, 3079, 4111, 4127, 4143, 63, 63, 63, 63, 63,
        63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 0, 0,    0,    0,    0,    0,    0,    0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,    0,    0,    0,    0,    0,    0,  0,  0};

mlir::Type parseType(mlir::OpBuilder builder, mlir::Type ty, const nb::QuantParams& qp) {
    if (!qp.present) {
        return ty;
    }

    auto intTy = ty.dyn_cast<mlir::IntegerType>();
    auto float8E5M2Ty = ty.dyn_cast<mlir::Float8E5M2Type>();
    auto float8E4M3Ty = ty.dyn_cast<mlir::Float8E4M3FNType>();
    if (!intTy && !float8E5M2Ty && !float8E4M3Ty) {
        return ty;
    }

    auto lowRange = qp.low_range;
    auto highRange = qp.high_range;
    auto outputType = builder.getF32Type();
    auto flags = intTy && intTy.isSigned() ? mlir::quant::QuantizationFlags::Signed : 0;
    if (float8E5M2Ty != nullptr) {
        lowRange = std::numeric_limits<vpux::type::float8_e5m2>::lowest();
        highRange = std::numeric_limits<vpux::type::float8_e5m2>::max();
        flags = mlir::quant::QuantizationFlags::Signed;
    } else if (float8E4M3Ty != nullptr) {
        lowRange = std::numeric_limits<vpux::type::float8_e4m3>::lowest();
        highRange = std::numeric_limits<vpux::type::float8_e4m3>::max();
        flags = mlir::quant::QuantizationFlags::Signed;
    }

    if (qp.scale.size() == 1) {
        return mlir::quant::UniformQuantizedType::get(flags, ty, outputType, qp.scale.front(), qp.zeropoint, lowRange,
                                                      highRange);
    }

    std::vector<int64_t> zeropoint(qp.scale.size(), qp.zeropoint);
    return mlir::quant::UniformQuantizedPerAxisType::get(flags, ty, outputType, qp.scale, zeropoint,
                                                         (DimsOrder::NCHW).dimPos(vpux::Dims4D::Act::C), lowRange,
                                                         highRange);
}

mlir::Type convertToMLIRType(mlir::OpBuilder builder, nb::DType dtype) {
    auto ctx = builder.getContext();
    switch (dtype) {
    case nb::DType::U4:
        return getUInt4Type(ctx);
    case nb::DType::U8:
        return getUInt8Type(ctx);
    case nb::DType::I4:
        return getSInt4Type(ctx);
    case nb::DType::I8:
        return getSInt8Type(ctx);
    case nb::DType::I32:
        return getSInt32Type(ctx);
    case nb::DType::BF8:
        return builder.getFloat8E5M2Type();
    case nb::DType::HF8:
        return builder.getFloat8E4M3FNType();
    case nb::DType::FP16:
        return builder.getF16Type();
    case nb::DType::FP32:
        return builder.getF32Type();
    case nb::DType::BF16:
        return builder.getBF16Type();
    default:
        throw std::domain_error{"Expected a valid data type"};
    }
}

template <class StorageType>
mlir::DenseElementsAttr generateWeights(std::ifstream& stream, mlir::RankedTensorType type, std::size_t elementsCount) {
    if (!stream) {
        auto generatedElements = std::vector<StorageType>(elementsCount);
        // have to add at least one non-zero element to make attribute non-splat. BitPack can't
        // work with splat tensors
        generatedElements[0] = 1;
        return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<StorageType>(generatedElements));
    }

    std::vector<StorageType> buffer(elementsCount);
    const auto expectedBytesCountToRead = elementsCount * sizeof(StorageType);
    // read as bytes since FP16/BFP16 are not supported by C++ standard
    stream.read(reinterpret_cast<char*>(buffer.data()), expectedBytesCountToRead);

    const auto actualBytesCountRead = static_cast<std::size_t>(stream.gcount());
    const auto state = stream.rdstate();

    if (expectedBytesCountToRead == actualBytesCountRead) {
        return mlir::DenseElementsAttr::get(type, llvm::ArrayRef<StorageType>(buffer));
    }

    VPUX_THROW_UNLESS((state & std::ifstream::eofbit) == 0,
                      "Failed to read {0} bytes from weights file, read {1} bytes before EOF has been reached",
                      expectedBytesCountToRead, actualBytesCountRead);
    VPUX_THROW_UNLESS(
            (state & std::ifstream::failbit) == 0,
            "Failed to read {0} bytes from weights file, read {1} bytes before logical error on i/o operation occurred",
            expectedBytesCountToRead, actualBytesCountRead);
    VPUX_THROW_UNLESS(
            (state & std::ifstream::badbit) == 0,
            "Failed to read {0} bytes from weights file, read {1} bytes before read error on i/o operation occurred",
            expectedBytesCountToRead, actualBytesCountRead);
    VPUX_THROW("Unexpected std::ifstream::rdstate value {}", state);
}

}  // namespace

mlir::DenseElementsAttr generateWeights(mlir::OpBuilder builder, llvm::ArrayRef<int64_t> shape, mlir::Type type,
                                        mlir::MLIRContext* context, const char* weightsFileName) {
    VPUX_THROW_UNLESS(!shape.empty(), "generateWeights: Got empty shape");
    auto wtData_ddr_valueType = mlir::RankedTensorType::get(shape, type);
    const auto vecSize = static_cast<std::size_t>(
            std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));

    if (auto qtype = type.dyn_cast_or_null<mlir::quant::QuantizedType>()) {
        type = mlir::quant::QuantizedType::castToStorageType(qtype);
        if (type.isFloat8E5M2()) {
            wtData_ddr_valueType = mlir::RankedTensorType::get(shape, builder.getFloat8E5M2Type());
        } else if (type.isFloat8E4M3FN()) {
            wtData_ddr_valueType = mlir::RankedTensorType::get(shape, builder.getFloat8E4M3FNType());
        } else if (qtype.getFlags() & mlir::quant::QuantizationFlags::Signed) {
            wtData_ddr_valueType = mlir::RankedTensorType::get(shape, getSInt8Type(context));
        } else {
            wtData_ddr_valueType = mlir::RankedTensorType::get(shape, getUInt8Type(context));
        }
    }

    std::ifstream stream{weightsFileName, std::ios::in | std::ios::binary};
    if (!stream) {
        std::cerr << "Warning: Unable to open weight data file " << weightsFileName << '\n';
    }

    if (type.isInteger(4)) {
        std::vector<int64_t> uintWrapperShape{shape.begin(), shape.end()};
        VPUX_THROW_UNLESS(!uintWrapperShape.empty(), "generateWeights: Got empty shape");
        // in NHWC tensor two int4 neighboring elements by C axis will be united into one uint8 element. So we have to
        // recalculate shape for uint wrapper tensor
        uintWrapperShape[Dims4D::Filter::OC.ind()] /= 2;

        const auto wrapperVecSize = static_cast<std::size_t>(std::accumulate(
                uintWrapperShape.begin(), uintWrapperShape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));

        auto uintWrapperValueType = mlir::RankedTensorType::get(uintWrapperShape, getUInt8Type(context));
        const auto weightsPacked = generateWeights<uint8_t>(stream, uintWrapperValueType, wrapperVecSize);
        std::vector<std::uint8_t> weightsUnpacked;
        weightsUnpacked.reserve(weightsPacked.size() * 2);
        for (const auto& elemPacked : weightsPacked.getValues<uint8_t>()) {
            const int8_t msn = (elemPacked & 0xf0) >> 4;
            const int8_t lsn = (elemPacked & 0x0f) >> 0;
            weightsUnpacked.push_back(lsn);
            weightsUnpacked.push_back(msn);
        }
        VPUX_THROW_UNLESS(
                weightsUnpacked.size() == vecSize,
                "Warning: count of elements in weights file {0} doesn't match with provided weights shape {1}",
                weightsUnpacked.size(), shape);

        if (type.isSignedInteger(4)) {
            return mlir::DenseElementsAttr::get(
                    wtData_ddr_valueType,
                    ArrayRef(reinterpret_cast<const int8_t*>(weightsUnpacked.data()), weightsUnpacked.size()));
        } else {
            return mlir::DenseElementsAttr::get(wtData_ddr_valueType, ArrayRef(weightsUnpacked));
        }
    } else if (type.isSignedInteger(8)) {
        return generateWeights<std::int8_t>(stream, wtData_ddr_valueType, vecSize);
    } else if (type.isInteger(8)) {
        return generateWeights<std::uint8_t>(stream, wtData_ddr_valueType, vecSize);
    } else if (type.isF16()) {
        return generateWeights<vpux::type::float16>(stream, wtData_ddr_valueType, vecSize);
    } else if (type.isBF16()) {
        return generateWeights<vpux::type::bfloat16>(stream, wtData_ddr_valueType, vecSize);
    } else if (type.isF32()) {
        return generateWeights<float>(stream, wtData_ddr_valueType, vecSize);
    } else if (type.isFloat8E5M2()) {
        return generateWeights<vpux::type::float8_e5m2>(stream, wtData_ddr_valueType, vecSize);
    } else if (type.isFloat8E4M3FN()) {
        return generateWeights<vpux::type::float8_e4m3>(stream, wtData_ddr_valueType, vecSize);
    } else {
        VPUX_THROW("Unexpected weights data type: {0}", type);
    }
}

vpux::Const::ContentAttr generateDefaultWeightsAttr(mlir::DenseElementsAttr weights, mlir::Type type) {
    auto weightsAttribute = vpux::Const::ContentAttr::get(weights);
    weightsAttribute = weightsAttribute.reorder(vpux::DimsOrder::OYXI);

    if (auto qty = type.dyn_cast<mlir::quant::QuantizedType>()) {
        auto storageType = mlir::quant::QuantizedType::castToStorageType(qty);
        if (!(storageType.isFloat8E5M2() || storageType.isFloat8E4M3FN())) {
            const auto quantizedType = vpux::changeStorageType(qty, weightsAttribute.getType().getElementType());
            weightsAttribute = weightsAttribute.quantCast(quantizedType);
            if (qty.getStorageType().isInteger(4)) {
                weightsAttribute = weightsAttribute.bitPack(4);
            }
        }
    }

    return weightsAttribute;
}

std::vector<uint16_t> computeSprLookupTable(nb::ActivationType activation_type) {
    // LUT Data; Following values (fp16) are synced with numerics-bench rsqrt LUT

    // Saturation/bypass regions, from l-to-r:
    // upper threshold (ru)
    // lower threshold (rl)
    // saturation value (y value to be set if x value in above interval)
    const std::vector<std::vector<uint16_t>>* saturationTableLutPtr = nullptr;

    // Linear funcion approximations, from l-to-r:
    // slope
    // intercept
    const std::vector<std::vector<uint16_t>>* slopeInterceptLutPtr = nullptr;

    // Table containing base addresses and exponents for LUT address generation
    // Each numer a single fp16 data point, represented by :|<- n->|<- base address ->|
    //                                                      |6 bits|<-   10 bits     >|
    //
    //  - base/start LUT address for each input exponent stored in LUT_CFG table
    //  - n: number of linear segments (l), a power of 2 i.e. l = 2^n^ , where n = 0, 1, 2, 3, 4,...

    // Address generation (S-sign, E-exponent, M-mantissa):
    //    fp32 input:     S    EEEEEEEE    MMMMMMMMMMMMMMMMMMMMMM
    //                         |      |    |...|- n MSB of mantissa
    //                     exp16=exp32-112    |
    //                             V          V
    //                          LUT_CFG ----> + -> LUT address
    //                                b. addr
    const std::vector<uint16_t>* configLutPtr = nullptr;

    // SYM_B = 0 (pos 128) (1 for SIN and TANH)
    // RCP = 0 (pos 129)
    // RSQRT = 0 (pos 130) (1 for RSQRT)
    uint16_t specialConfig = 0;

    switch (activation_type) {
    case nb::ActivationType::Rsqrt:
        saturationTableLutPtr = &RSQRT_SATURATION_TABLE_LUT;
        slopeInterceptLutPtr = &RSQRT_SLOPE_INTERCEPT_LUT;
        configLutPtr = &RSQRT_CONFIG_LUT;
        specialConfig = 0x0004;
        break;
    case nb::ActivationType::Sigmoid:
        saturationTableLutPtr = &SIGMOID_SATURATION_TABLE_LUT;
        slopeInterceptLutPtr = &SIGMOID_SLOPE_INTERCEPT_LUT;
        configLutPtr = &SIGMOID_CONFIG_LUT;
        specialConfig = 0x0000;
        break;
    case nb::ActivationType::Sin:
        saturationTableLutPtr = &SIN_SATURATION_TABLE_LUT;
        slopeInterceptLutPtr = &SIN_SLOPE_INTERCEPT_LUT;
        configLutPtr = &SIN_CONFIG_LUT;
        specialConfig = 0x0001;
        break;
    case nb::ActivationType::Tanh:
        saturationTableLutPtr = &TANH_SATURATION_TABLE_LUT;
        slopeInterceptLutPtr = &TANH_SLOPE_INTERCEPT_LUT;
        configLutPtr = &TANH_CONFIG_LUT;
        specialConfig = 0x0001;
        break;
    default:
        VPUX_THROW("Unexpected activation type data type");
    }

    const auto& saturationTableLut = *saturationTableLutPtr;
    const auto& slopeInterceptLut = *slopeInterceptLutPtr;
    const auto& configLut = *configLutPtr;

    // LUT creation/formatting
    std::vector<uint16_t> sprLookUpTableContent;

    // saturation regions inserted in REVERSE order (saturation value, lower threshold, upper threshold)
    for (uint32_t index = 0; index < NUM_LUT_SAT_REGS; ++index) {
        sprLookUpTableContent.insert(std::end(sprLookUpTableContent), std::rbegin(saturationTableLut[index]),
                                     std::rend(saturationTableLut[index]));
    }

    // special configuration
    sprLookUpTableContent.push_back(specialConfig);

    // reserved region
    for (uint32_t index = 0; index < NUM_LUT_RESERVED_REGS; ++index) {
        sprLookUpTableContent.push_back(0x0000);
    }

    // lut cfg region
    for (uint32_t index = 0; index < NUM_LUT_CFG_REGS; ++index) {
        sprLookUpTableContent.push_back(configLut[index]);
    }

    // lut data region inserted in normal order (slope, intercept)
    if (size(slopeInterceptLut) > NUM_LUT_DATA_REGS) {
        VPUX_THROW("Size of lut data {0} is bigger than allowed size of {1}", size(slopeInterceptLut),
                   NUM_LUT_DATA_REGS);
    }
    for (uint32_t index = 0; index < size(slopeInterceptLut); ++index) {
        sprLookUpTableContent.insert(std::end(sprLookUpTableContent), std::begin(slopeInterceptLut[index]),
                                     std::end(slopeInterceptLut[index]));
    }

    return sprLookUpTableContent;
}

std::size_t totalTensorSize(llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
    auto qType = elementType.dyn_cast_or_null<mlir::quant::UniformQuantizedType>();
    auto qPerChType = elementType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>();

    if (qType) {
        elementType = qType.getStorageType();
    } else if (qPerChType) {
        elementType = qPerChType.getStorageType();
    }

    size_t numBits = elementType.getIntOrFloatBitWidth();

    const auto totalSize =
            std::accumulate(shape.begin(), shape.end(), static_cast<std::int64_t>(1), std::multiplies<std::int64_t>());
    const auto totalBits = totalSize * numBits;
    VPUX_THROW_UNLESS(totalBits % CHAR_BIT == 0, "Tensors size is not allligned to Byte");
    return static_cast<std::size_t>(totalBits / CHAR_BIT);
}

std::size_t totalWeightsSize(llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
    auto qType = elementType.dyn_cast_or_null<mlir::quant::UniformQuantizedType>();
    auto qPerChType = elementType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>();

    if (qType) {
        elementType = qType.getStorageType();
    } else if (qPerChType) {
        elementType = qPerChType.getStorageType();
    }

    size_t numBits = elementType.getIntOrFloatBitWidth();

    const auto weightsSetSize = shape[vpux::Dims4D::Filter::IC.ind()] * shape[vpux::Dims4D::Filter::KX.ind()] *
                                shape[vpux::Dims4D::Filter::KY.ind()];
    const auto weightsSetSizeBits = weightsSetSize * numBits;
    VPUX_THROW_UNLESS(weightsSetSizeBits % CHAR_BIT == 0, "Weights tensor size is not allligned to Byte");
    const auto weightsSetSizeBytes = weightsSetSizeBits / CHAR_BIT;
    const std::uint64_t weightsAlignment = 32;
    const auto weightsSizeBytesAligned = vpux::alignValUp(weightsSetSizeBytes, weightsAlignment);
    auto weightsTotalSize = weightsSizeBytesAligned * shape[vpux::Dims4D::Filter::OC.ind()];

    return weightsTotalSize;
}

std::vector<int64_t> convertNBPadtoNCETaskPad(const std::array<int64_t, 4>& nb_pad) {
    std::vector<int64_t> ncetask_pad(nb_pad.size());

    ncetask_pad[PAD_NCETASK_LEFT] = nb_pad[PAD_NB_LEFT];
    ncetask_pad[PAD_NCETASK_RIGHT] = nb_pad[PAD_NB_RIGHT];
    ncetask_pad[PAD_NCETASK_TOP] = nb_pad[PAD_NB_TOP];
    ncetask_pad[PAD_NCETASK_BOTTOM] = nb_pad[PAD_NB_BOTTOM];

    return ncetask_pad;
}

mlir::Type parseInputType(mlir::OpBuilder builder, const nb::InputLayer& input) {
    return parseType(builder, convertToMLIRType(builder, input.dtype), input.qp);
}

mlir::Type parseOutputType(mlir::OpBuilder builder, const nb::OutputLayer& output) {
    return parseType(builder, convertToMLIRType(builder, output.dtype), output.qp);
}

mlir::Type parseWeightsType(mlir::OpBuilder builder, const nb::WeightLayer& weight) {
    return parseType(builder, convertToMLIRType(builder, weight.dtype), weight.qp);
}

void buildCNNOp(mlir::OpBuilder& builder, llvm::StringRef mainFuncName, llvm::ArrayRef<mlir::Type> inputs,
                llvm::ArrayRef<mlir::Type> outputs) {
    return buildCNNOp(builder, mainFuncName, inputs, outputs, {});
}

void buildCNNOp(mlir::OpBuilder& builder, llvm::StringRef mainFuncName, llvm::ArrayRef<mlir::Type> inputs,
                llvm::ArrayRef<mlir::Type> outputs, llvm::ArrayRef<ProfilingDataSection> profilingSections) {
    const auto enableProfiling = !profilingSections.empty();
    const auto mainFuncNameAttr = mlir::SymbolRefAttr::get(builder.getContext(), mainFuncName);
    auto cnnOp = builder.create<IE::CNNNetworkOp>(builder.getUnknownLoc(), mainFuncNameAttr, enableProfiling);
    cnnOp.getInputsInfo().emplaceBlock();
    cnnOp.getOutputsInfo().emplaceBlock();
    if (enableProfiling) {
        for (auto& section : cnnOp.getProfilingOutputsInfo()) {
            section.emplaceBlock();
        }
    }

    auto inputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.getInputsInfo().front(), builder.getListener());
    for (auto input : enumerate(inputs)) {
        auto inputType = input.value().cast<mlir::ShapedType>();
        auto quantized = inputType.getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedType>();
        auto quantizedPerCh = inputType.getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>();

        if (quantized) {
            inputType = inputType.clone(quantized.getStorageType());
        } else if (quantizedPerCh) {
            inputType = inputType.clone(quantizedPerCh.getStorageType());
        }

        const auto inputName = printToString("input_{0}", input.index());
        const auto nameAttr = builder.getStringAttr(inputName);
        const auto userTypeAttr = mlir::TypeAttr::get(inputType);
        inputsInfoBuilder.create<IE::DataInfoOp>(builder.getUnknownLoc(), nameAttr, userTypeAttr,
                                                 /*profilingSectionsCount=*/0);
    }

    auto outputsInfoBuilder = mlir::OpBuilder::atBlockBegin(&cnnOp.getOutputsInfo().front(), builder.getListener());
    for (auto output : enumerate(outputs)) {
        auto outputType = output.value().cast<mlir::ShapedType>();
        auto quantized = outputType.getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedType>();
        auto quantizedPerCh = outputType.getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>();

        if (quantized) {
            outputType = outputType.clone(quantized.getStorageType());
        } else if (quantizedPerCh) {
            outputType = outputType.clone(quantizedPerCh.getStorageType());
        }

        const auto resultName = printToString("output_{0}", output.index());
        const auto nameAttr = builder.getStringAttr(resultName);
        const auto userTypeAttr = mlir::TypeAttr::get(outputType);
        outputsInfoBuilder.create<IE::DataInfoOp>(builder.getUnknownLoc(), nameAttr, userTypeAttr,
                                                  /*profilingSectionsCount=*/0);
    }

    if (enableProfiling) {
        auto& profilingOutputsInfo = cnnOp.getProfilingOutputsInfo().front();
        auto userInfoBuilder = mlir::OpBuilder::atBlockEnd(&profilingOutputsInfo.front());

        const auto ctx = builder.getContext();

        const auto lastSection = profilingSections[profilingSections.size() - 1];
        const auto totalSize = lastSection.offset + lastSection.size;

        auto newOutputResult =
                mlir::MemRefType::get({static_cast<int64_t>(totalSize / sizeof(uint32_t))}, getUInt32Type(ctx));
        auto newOutputShapedType = newOutputResult.cast<vpux::NDTypeInterface>();
        auto outputUserResult = getTensorType(newOutputShapedType.getShape(), newOutputShapedType.getElementType(),
                                              newOutputShapedType.getDimsOrder(), nullptr);

        auto dataInfo = userInfoBuilder.create<IE::DataInfoOp>(mlir::UnknownLoc::get(ctx),
                                                               mlir::StringAttr::get(ctx, "profilingOutput"),
                                                               mlir::TypeAttr::get(outputUserResult),
                                                               /*profilingSectionsCount=*/1);
        dataInfo.getSections().front().emplaceBlock();

        auto sectionsBuilder =
                mlir::OpBuilder::atBlockBegin(&dataInfo.getSections().front().front(), builder.getListener());
        for (auto section : profilingSections) {
            sectionsBuilder.create<VPUIP::ProfilingSectionOp>(
                    mlir::UnknownLoc::get(ctx), getIntAttr(ctx, section.execType), getIntAttr(ctx, section.offset),
                    getIntAttr(ctx, section.size));
        }
    }
}

mlir::DenseElementsAttr splitWeightsOverC(mlir::DenseElementsAttr wt_vec, ArrayRef<int64_t> wt_shape, mlir::Type dtype,
                                          mlir::MLIRContext* ctx, size_t start_C, size_t end_C) {
    auto qType = dtype.dyn_cast<mlir::quant::UniformQuantizedType>();
    if (!((dtype.isF16() || (qType && qType.getStorageType().isUnsignedInteger(8)))))
        throw std::domain_error{
                printToString("splitWeightsOverC only supports weight data type fp16 or uint8; got {0}", dtype)};

    if (dtype.isF16()) {
        vpux::type::float16 elementType = 0;
        return splitWeightsOverCLoop(wt_vec, wt_shape, dtype, elementType, ctx, start_C, end_C);
    } else {
        uint8_t elementType = 0;
        return splitWeightsOverCLoop(wt_vec, wt_shape, dtype, elementType, ctx, start_C, end_C);
    }
}

template <typename T>
mlir::DenseElementsAttr splitWeightsOverCLoop(mlir::DenseElementsAttr wt_vec, ArrayRef<int64_t> wt_shape,
                                              mlir::Type dtype, T, mlir::MLIRContext* ctx, size_t start_C,
                                              size_t end_C) {
    SmallVector<int64_t> original_shape(wt_shape.begin(), wt_shape.end());

    // Weights from NumericsBench in KCHW
    // For stream-over-C, need to extract weights[startC:endC]
    int64_t K = original_shape[0];
    int64_t C = original_shape[1];
    int64_t H = original_shape[2];
    int64_t W = original_shape[3];
    int64_t new_C = end_C - start_C;

    auto wt_full_itr = wt_vec.getValues<T>();
    std::vector<T> wt_full(wt_full_itr.begin(), wt_full_itr.end());
    const llvm::SmallVector<int64_t> wt_partial_shape({K, new_C, H, W});
    size_t vecSize = static_cast<size_t>(std::accumulate(wt_partial_shape.begin(), wt_partial_shape.end(),
                                                         static_cast<int64_t>(1), std::multiplies<int64_t>()));
    std::vector<T> wt_partial(vecSize);

    for (int64_t k = 0; k < K; k++)
        for (int64_t c = 0; c < new_C; c++)
            for (int64_t h = 0; h < H; h++)
                for (int64_t w = 0; w < W; w++) {
                    // auto old_c = c+start_C;
                    auto old_offset = k * C * H * W + (c + start_C) * H * W + h * W + w;
                    auto new_offset = k * new_C * H * W + c * H * W + h * W + w;
                    wt_partial[new_offset] = wt_full[old_offset];
                }

    auto wtData_ddr_valueType = mlir::RankedTensorType::get(wt_partial_shape, dtype);
    if (auto qtype = dtype.dyn_cast<mlir::quant::QuantizedType>()) {
        dtype = mlir::quant::QuantizedType::castToStorageType(qtype);
        if (qtype.getFlags() & mlir::quant::QuantizationFlags::Signed) {
            wtData_ddr_valueType = mlir::RankedTensorType::get(wt_partial_shape, getSInt8Type(ctx));
        } else {
            wtData_ddr_valueType = mlir::RankedTensorType::get(wt_partial_shape, getUInt8Type(ctx));
        }
    }

    auto wt_data_values = ArrayRef<T>(wt_partial);
    auto wt_data_vals = mlir::DenseElementsAttr::get(wtData_ddr_valueType, wt_data_values);
    return wt_data_vals;
}

mlir::MemRefType getMemRefType(VPURT::BufferSection section, ArrayRef<int64_t> shape, mlir::Type elemType,
                               DimsOrder order) {
    return vpux::getMemRefType(ShapeRef(shape), elemType, order, VPURT::getMemoryKind(section));
}

mlir::MemRefType getMemRefType(VPURT::BufferSection section, size_t sectionIdx, ArrayRef<int64_t> shape,
                               mlir::Type elemType, DimsOrder order) {
    auto symbolAttr =
            IndexedSymbolAttr::get(elemType.getContext(), stringifyEnum(VPURT::getMemoryKind(section)), sectionIdx);
    return vpux::getMemRefType(ShapeRef(shape), elemType, order, symbolAttr);
}

mlir::MemRefType getMemRefType(VPURT::BufferSection section, size_t sectionIdx, ShapeRef shape, mlir::Type elemType,
                               DimsOrder order) {
    auto symbolAttr =
            IndexedSymbolAttr::get(elemType.getContext(), stringifyEnum(VPURT::getMemoryKind(section)), sectionIdx);
    return vpux::getMemRefType(shape, elemType, order, symbolAttr);
}

mlir::MemRefType getMemRefType(VPURT::BufferSection section, ArrayRef<int64_t> shape, mlir::Type elemType,
                               DimsOrder order, StridesRef strides) {
    return vpux::getMemRefType(ShapeRef(shape), elemType, order, VPURT::getMemoryKind(section), strides);
}

mlir::MemRefType getMemRefType(VPURT::BufferSection section, size_t sectionIdx, ArrayRef<int64_t> shape,
                               mlir::Type elemType, DimsOrder order, StridesRef strides) {
    auto symbolAttr =
            IndexedSymbolAttr::get(elemType.getContext(), stringifyEnum(VPURT::getMemoryKind(section)), sectionIdx);
    return vpux::getMemRefType(ShapeRef(shape), elemType, order, symbolAttr, strides);
}

mlir::MemRefType getMemRefType(VPURT::BufferSection section, size_t sectionIdx, ArrayRef<int64_t> shape,
                               mlir::Type elemType, DimsOrder order, StridesRef strides,
                               VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr) {
    auto symbolAttr =
            IndexedSymbolAttr::get(elemType.getContext(), stringifyEnum(VPURT::getMemoryKind(section)), sectionIdx);
    return vpux::getMemRefType(ShapeRef(shape), elemType, order, symbolAttr, strides, swizzlingSchemeAttr);
}

vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, VPURT::BufferSection section,
                                                   ArrayRef<int64_t> shape, mlir::Type elemType, DimsOrder order,
                                                   int64_t locale, size_t offset) {
    const auto type = getMemRefType(section, locale, shape, elemType, order);
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, locale, offset);
}

vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, VPURT::BufferSection section,
                                                   ShapeRef shape, mlir::Type elemType, DimsOrder order, int64_t locale,
                                                   size_t offset) {
    const auto type = getMemRefType(section, locale, shape, elemType, order);
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, locale, offset);
}

vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder builder, VPURT::BufferSection section,
                                                   ArrayRef<int64_t> shape, mlir::Type elemType, DimsOrder order,
                                                   StridesRef strides, int64_t locale, size_t offset) {
    const auto type = getMemRefType(section, locale, shape, elemType, order, strides);
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, locale, offset);
}

vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder builder, VPURT::BufferSection section,
                                                   ArrayRef<int64_t> shape, mlir::Type elemType, DimsOrder order,
                                                   StridesRef strides, int64_t locale, size_t offset,
                                                   VPUIP::SwizzlingSchemeAttr swizzlingSchemeAttr) {
    const auto type = getMemRefType(section, locale, shape, elemType, order, strides, swizzlingSchemeAttr);
    const auto swizzlingKey = swizzlingSchemeAttr.getKey().getInt();
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, locale, offset, swizzlingKey);
}

vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, mlir::Type type,
                                                   VPURT::BufferSection section, size_t offset) {
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, offset);
}

vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, mlir::MemRefType type,
                                                   VPURT::BufferSection section, int64_t locale, size_t offset) {
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, locale, offset);
}

vpux::VPURT::DeclareBufferOp createDeclareTensorOp(mlir::OpBuilder& builder, mlir::Type type,
                                                   VPURT::BufferSection section, ArrayRef<int64_t> locale,
                                                   size_t offset) {
    return builder.create<VPURT::DeclareBufferOp>(builder.getUnknownLoc(), type, section, locale, offset);
}

mlir::OpResult getTensorResult(VPURT::DeclareBufferOp op) {
    return op.getOperation()->getResult(0);
}

mlir::OpResult getConstResult(vpux::Const::DeclareOp op) {
    return op.getOperation()->getResult(0);
}

vpux::VPUIP::DPUTaskOp createDPUTaskOp(mlir::OpBuilder builder, mlir::OpBuilder variantbuilder,
                                       ArrayRef<int64_t> outputShape, ArrayRef<int64_t> inputShape,
                                       const std::vector<int64_t>& paddingVec, VPU::MPEMode mpeMode,
                                       int64_t clusterId) {
    std::vector<int32_t> startVec{0, 0, 0};
    auto start = getIntArrayAttr(builder, startVec);
    const auto outEnd = getIntArrayAttr(
            builder, std::vector<std::int64_t>{outputShape[3] - 1, outputShape[2] - 1, outputShape[1] - 1});
    const auto inEnd = getIntArrayAttr(
            builder, std::vector<std::int64_t>{inputShape[3] - 1, inputShape[2] - 1, inputShape[1] - 1});
    auto pad = VPU::getPaddingAttr(builder.getContext(), paddingVec[PAD_NCETASK_LEFT], paddingVec[PAD_NCETASK_RIGHT],
                                   paddingVec[PAD_NCETASK_TOP], paddingVec[PAD_NCETASK_BOTTOM]);

    auto dpuTask =
            variantbuilder.create<VPUIP::DPUTaskOp>(builder.getUnknownLoc(), start, outEnd, start, inEnd, pad, mpeMode,
                                                    getIntAttr<int64_t>(builder.getContext(), clusterId));

    return dpuTask;
}

vpux::DimsOrder oduPermutationToLayout(const MVCNN::Permutation oduPermutation) {
    switch (oduPermutation) {
    case MVCNN::Permutation::Permutation_ZXY:
        return vpux::DimsOrder::NHWC;
    case MVCNN::Permutation::Permutation_ZYX:
        return vpux::DimsOrder::fromCode(0x1432);  // NWHC
    case MVCNN::Permutation::Permutation_YZX:
        return vpux::DimsOrder::fromCode(0x1423);  // NWCH
    case MVCNN::Permutation::Permutation_YXZ:
        return vpux::DimsOrder::fromCode(0x1243);  // NCWH
    case MVCNN::Permutation::Permutation_XZY:
        return vpux::DimsOrder::NHCW;
    case MVCNN::Permutation::Permutation_XYZ:
        return vpux::DimsOrder::NCHW;
    default:
        return vpux::DimsOrder::NHWC;
    }
}

vpux::Dim getInnermostDim(const vpux::DimsOrder& order) {
    return order.toDim(MemDim(order.numDims() - 1));
}

// Get padding for each cluster based on global padding, segmentation type and current cluster
VPU::PaddingAttr getMulticlusteringPaddings(mlir::MLIRContext* ctx, const int64_t cluster, const int64_t numClusters,
                                            nb::SegmentationType segmentationType, VPU::PaddingAttr globalPadding,
                                            SmallVector<std::int64_t> clustersPerDim) {
    const auto noPad = getIntAttr<int64_t>(ctx, 0);
    switch (segmentationType) {
    case nb::SegmentationType::SOH: {
        if (cluster == 0) {
            return VPU::PaddingAttr::get(ctx, globalPadding.getLeft(), globalPadding.getRight(), globalPadding.getTop(),
                                         noPad);
        }

        if (cluster == numClusters - 1) {
            return VPU::PaddingAttr::get(ctx, globalPadding.getLeft(), globalPadding.getRight(), noPad,
                                         globalPadding.getBottom());
        }

        return VPU::PaddingAttr::get(ctx, globalPadding.getLeft(), globalPadding.getRight(), noPad, noPad);
    }
    case nb::SegmentationType::SOW: {
        if (cluster == 0) {
            return VPU::PaddingAttr::get(ctx, globalPadding.getLeft(), noPad, globalPadding.getTop(),
                                         globalPadding.getBottom());
        }

        if (cluster == numClusters - 1) {
            return VPU::PaddingAttr::get(ctx, noPad, globalPadding.getRight(), globalPadding.getTop(),
                                         globalPadding.getBottom());
        }

        return VPU::PaddingAttr::get(ctx, noPad, noPad, globalPadding.getTop(), globalPadding.getBottom());
    }
    case nb::SegmentationType::SOK:
        return globalPadding;
    case nb::SegmentationType::SOHW: {
        if (clustersPerDim.size() != 2) {
            VPUX_THROW("SOHW needs clustersPerDim size to pe 2");
        }
        const auto heightNumClusters = clustersPerDim[0];
        const auto widthNumClusters = clustersPerDim[1];

        const auto idxH = cluster / widthNumClusters;
        const auto idxW = cluster % widthNumClusters;

        const auto top = (idxH == 0) ? globalPadding.getTop() : noPad;
        const auto bottom = (idxH == heightNumClusters - 1) ? globalPadding.getBottom() : noPad;
        const auto left = (idxW == 0) ? globalPadding.getLeft() : noPad;
        const auto right = (idxW == widthNumClusters - 1) ? globalPadding.getRight() : noPad;

        return VPU::PaddingAttr::get(ctx, left, right, top, bottom);
    }
    case nb::SegmentationType::SOHK: {
        if (clustersPerDim.size() != 2) {
            VPUX_THROW("SOHW needs clustersPerDim size to pe 2");
        }

        const auto heightNumClusters = clustersPerDim[0];
        const auto channelsNumClusters = clustersPerDim[1];

        const auto idxH = cluster / channelsNumClusters;

        const auto top = (idxH == 0) ? globalPadding.getTop() : noPad;
        const auto bottom = (idxH == heightNumClusters - 1) ? globalPadding.getTop() : noPad;

        return VPU::PaddingAttr::get(ctx, globalPadding.getLeft(), globalPadding.getRight(), top, bottom);
    }
    default:
        VPUX_THROW("Segmentation type unsupported {0}", nb::to_string(segmentationType));
    };

    return VPU::PaddingAttr();
}

std::pair<mlir::SmallVector<mlir::Value>, size_t> insertWLMStartSequence(mlir::OpBuilder& builder, bool isWLMEnabled,
                                                                         bool useVirtualBarriers) {
    if (!isWLMEnabled) {
        return {{}, 0};
    }

    auto* ctx = builder.getContext();
    auto inputBuff = VPUIP::createDummyBuffer(builder);
    auto outputBuff = VPUIP::createDummyBuffer(builder);
    auto zeroAttr = vpux::getIntAttr(ctx, 0);

    mlir::Value startBarrier0, startBarrier1;

    auto startBarrier0Loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "wlm_start_barrier0"));
    auto startBarrier1Loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "wlm_start_barrier1"));
    if (useVirtualBarriers) {
        startBarrier0 = builder.create<VPURT::DeclareVirtualBarrierOp>(startBarrier0Loc).getBarrier();
        startBarrier1 = builder.create<VPURT::DeclareVirtualBarrierOp>(startBarrier1Loc).getBarrier();
    } else {
        startBarrier0 = builder.create<VPURT::ConfigureBarrierOp>(startBarrier0Loc, 0).getBarrier();
        startBarrier1 = builder.create<VPURT::ConfigureBarrierOp>(startBarrier1Loc, 1).getBarrier();
    }

    VPURT::wrapIntoTaskOp<VPUIP::SyncDMAOp>(builder, {}, startBarrier0,
                                            mlir::NameLoc::get(mlir::StringAttr::get(ctx, "wlm_start_dma_0")),
                                            inputBuff, outputBuff, /*port*/ zeroAttr,
                                            /*isOutOfOrder*/ nullptr, /*isCritical*/ nullptr, /*dmaHwpId*/ nullptr,
                                            /*dmaProfilingMetaData*/ nullptr);

    VPURT::wrapIntoTaskOp<VPUIP::SyncDMAOp>(builder, startBarrier0, startBarrier1,
                                            mlir::NameLoc::get(mlir::StringAttr::get(ctx, "wlm_start_dma_1")),
                                            inputBuff, outputBuff,
                                            /*port*/ zeroAttr,
                                            /*isOutOfOrder*/ nullptr, /*isCritical*/ nullptr, /*dmaHwpId*/ nullptr,
                                            /*dmaProfilingMetaData*/ nullptr);

    return {{startBarrier1}, 2};
}

}  // namespace hwtest
}  // namespace vpux
