// {% copyright %}

#include <custom_cpp_tests.h>
#include <nn_cache.h>
#include <cmath>
#include "layers/param_custom_cpp.h"

__attribute__((aligned(1024)))
#include "sk.vau_dp4m.3720xx.text.xdat"

#include "pss/param_vau_dp4m.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, VauDp4M_pss)) {
    static constexpr std::initializer_list<SingleTest> dp4_test_list{
            {{16, 1, 1}, {4, 1, 1}, orderZYX, FPE("vau_dp4m.3720xx.elf"), {sw_params::Location::NN_CMX}},
            {{32, 1, 1}, {8, 1, 1}, orderZYX, FPE("vau_dp4m.3720xx.elf"), {sw_params::Location::NN_CMX}},
            {{1008, 1, 1}, {252, 1, 1}, orderZYX, FPE("vau_dp4m.3720xx.elf"), {sw_params::Location::NN_CMX}},
    };

    class CustomCppVauDp4MTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppVauDp4MTest(): m_testsLoop(dp4_test_list, "test") {
        }
        virtual ~CustomCppVauDp4MTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppVauDp4MTestpss";
        }

        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, emptyParamData, MAX_LOCAL_PARAMS, 0};

            initElfBuffer();
            initTestCase();

            const SingleTest* test = m_currentTest;

            const Dims& inputDims = test->inputDims;
            const Dims& outputDims = test->outputDims;
            const StorageOrder& storageOrder = m_currentTest->storageOrder;

            const TensorDims dims3In(inputDims.begin()[0], inputDims.begin()[1], inputDims.begin()[2], 1);
            const TensorDims dims3Out(outputDims.begin()[0], outputDims.begin()[1], outputDims.begin()[2], 1);

            m_inTensor1.init(storageOrder, dims3In);
            m_inTensor2.init(storageOrder, dims3In);
            m_outTensor.init(storageOrder, dims3Out);
            m_refTensor.init(storageOrder, dims3Out);

            allocBuffer(m_inTensor1);
            allocBuffer(m_inTensor2);
            allocBuffer(m_outTensor);
            allocBuffer(m_refTensor);

            int32_t ind[subspace::MAX_DIMS] = {0};
            subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);
            m_vauDp4mParams = reinterpret_cast<sw_params::VauDp4mParams*>(paramContainer);
            *m_vauDp4mParams = sw_params::VauDp4mParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::VauDp4mParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_vauDp4mParams);

            m_params.kernel = reinterpret_cast<uint64_t>(sk_vau_dp4m_3720xx_text);
        }

        void initParserRunner() override {
            initMyriadResources();
            initDebugInfo();

            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);
            OpTensor inputBuff1;
            OpTensor inputBuff2;
            OpTensor outputBuff;
            m_inTensor1.exportToBuffer(inputBuff1);
            m_inTensor2.exportToBuffer(inputBuff2);
            m_outTensor.exportToBuffer(outputBuff);

            customCppOp->addInputBuffer(inputBuff1, m_requiredTensorLocation);
            customCppOp->addInputBuffer(inputBuff2, m_requiredTensorLocation);
            customCppOp->addOutputBuffer(outputBuff, m_requiredTensorLocation);
            customCppOp->ops = *getParams();
        }

        void resetOutputData() override {
            resetTensorBuffer(m_outTensor);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 1.0f;
        }

        void formatTestParams(char* str, int maxLength) const override {
            char inSizes_str[100];
            char outSizes_str[100];

            snprintf_append(str, maxLength, "input1: %s, input2: %s, output: %s",
                            m_inTensor1.dimsToStringNCHW(inSizes_str), m_inTensor1.dimsToStringNCHW(inSizes_str),
                            m_outTensor.dimsToStringNCHW(outSizes_str));
        }

        void generateInputData() override {
            rand_seed();

            m_inTensor1.forEach(false, [&](const MemoryDims& indices) {
                int8_t val = INT8_MAX * (float(rand()) / RAND_MAX * 2 - 1);
                m_inTensor1.at(indices) = val;
            });

            m_inTensor2.forEach(false, [&](const MemoryDims& indices) {
                uint8_t val = float(rand()) / RAND_MAX * UINT8_MAX;
                m_inTensor2.at(indices) = val;
            });
        }

        void generateReferenceData() override {
            int8_t* m_inTensor0_buff = (int8_t*)m_inTensor1.buffer();
            uint8_t* m_inTensor1_buff = (uint8_t*)m_inTensor2.buffer();

            int i = 0;

            m_refTensor.forEach(false, [&](const MemoryDims& indices) {
                int8_t a1 = m_inTensor0_buff[i * 4 + 0];
                uint8_t b1 = m_inTensor1_buff[i * 4 + 0];
                int8_t a2 = m_inTensor0_buff[i * 4 + 1];
                uint8_t b2 = m_inTensor1_buff[i * 4 + 1];
                int8_t a3 = m_inTensor0_buff[i * 4 + 2];
                uint8_t b3 = m_inTensor1_buff[i * 4 + 2];
                int8_t a4 = m_inTensor0_buff[i * 4 + 3];
                uint8_t b4 = m_inTensor1_buff[i * 4 + 3];

                i++;

                int32_t val = a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4;
                m_refTensor.at(indices) = val;
            });
        }

        virtual bool checkResult() override {
            m_outTensor.confirmBufferData();

            // save output data
            if (m_save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_inputTensor.buffer()), m_inputTensor.bufferSize(),
                                 "inMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_outTensor.buffer()), m_outTensor.bufferSize(),
                                 "outMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_refTensor.buffer()), m_refTensor.bufferSize(),
                                 "refOutMyriad.bin");
            }

            bool threshold_test_failed = false;

            m_outTensor.forEach(false, [&](const MemoryDims& indices) {
                int32_t value = m_outTensor.at(indices);
                int32_t gt_value = m_refTensor.at(indices);
                float ulp_diff = ulp::absdiff_fp32(value, gt_value);

                bool differ = !bool(ulp_diff <= m_test_threshold);
                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] value: %ld ref_value: %ld ulp: %f\n", ti.height, ti.width,
                           ti.channels, value, gt_value, ulp_diff);
                }
            });

            return !threshold_test_failed;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;
        Tensor<int8_t> m_inTensor1;
        Tensor<uint8_t> m_inTensor2;
        Tensor<int32_t> m_outTensor;
        Tensor<uint32_t> m_refTensor;

        sw_params::VauDp4mParams* m_vauDp4mParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppVauDp4MTest)
}  // namespace )
