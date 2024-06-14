#include <cstdint>
#include "BitStream.hpp"

namespace vpux::bitc {

#define BYTE_ALIGN(x) static_cast<uint32_t>(((x + 7) / 8) * 8)

const uint32_t BLOCK_SIZE{64u};
enum class CompressionType : uint32_t { eofr, lastblk, uncmprsd, cmprsd };

enum class DecoderAlgorithm : uint32_t {
    NOPROC = 0,
    SIGNSHFTPROC,
    SIGNSHFTADDPROC,
    ADDPROC,
    SIGNSHFTADDBLK,  // NPU40XX only
    LFTSHFTPROC,
    BINEXPPROC,
    BTEXPPROC,
    ALGO_COUNT
};

struct BitcHeader {
    uint64_t compression_type : 2;
    uint64_t algo : 3;
    uint64_t bit_length : 3;
    uint64_t dual_encode : 2;
    uint64_t unused : 54;
};

struct CompressionHeader {
    uint64_t compression_type : 2;
    uint64_t algo : 3;
    uint64_t bit_length : 3;
    uint64_t dual_encode : 2;
    uint64_t dual_encode_length : 10;
    uint64_t unused : 44;
};

struct LastBlockHeader {
    uint64_t compression_type : 2;
    uint64_t block_length : 6;
    uint64_t unused : 56;
};

struct DualEncoder {
    uint32_t length;
    uint64_t bitmap;
};

struct AlgorithmParam {
    uint8_t* p_data;
    uint32_t block_size;
    uint8_t residual[BLOCK_SIZE];
    uint8_t minimum;
    uint8_t maximum;
    uint16_t dual_encoder_enable;
    uint32_t bit_length;
    uint32_t encoding_bits;
    uint32_t dual_bit_length;
    uint32_t dual_encoding_bits;
    uint64_t dual_bitmap;
    uint32_t encoder_index;
    DecoderAlgorithm decoder;
};

enum class EncoderAlgorithm {
    MINPRDCT = 0,
    MINSPRDCT,  // Weight compression only
    MUPRDCT,
    NOPRDCT,
    NOSPRDCT,
    MEDPRDCT,
    PREVBLKPRDCT,  // >= NPU40XX only
    BINCMPCT,
    BTMAP,
    ALGO_COUNT
};

struct BestAlgorithm {
    uint32_t min_encoder_bits{~0u};
    uint32_t min_algo{};

    uint32_t dual_min_encoder_bits{~0u};
    uint32_t dual_min_algo{};
};

}  // namespace vpux::bitc
