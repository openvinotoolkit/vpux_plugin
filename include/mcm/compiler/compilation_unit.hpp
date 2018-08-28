#ifndef MV_COMPILATION_UNIT_HPP_
#define MV_COMPILATION_UNIT_HPP_

#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/logger/logger.hpp"
#include "include/mcm/pass/pass_manager.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include "include/mcm/utils/compositional_model_recorder.hpp"

namespace mv
{

    class CompilationUnit
    {

        static const std::string ma2480DefDescPath_;
        static const std::string compositionalModelRecordingsPath_;

        static Logger& logger_;

        OpModel* model_;
        CompositionalModelRecorder* recordedModel_;
        PassManager passManager_;
        TargetDescriptor targetDescriptor_;
        json::Object compilationDescriptor_;

    public:

        CompilationUnit(mv::Logger::VerboseLevel verboseLevel = mv::Logger::VerboseLevel::VerboseSilent, bool logTime = false);
        ~CompilationUnit();
        
        bool loadTargetDescriptor(const std::string& path);
        bool loadTargetDescriptor(Target target);

        PassManager& passManager();
        json::Object& compilationDescriptor();
        CompositionalModel& model();
        CompositionalModel& recordedModel();

        void loadModelFromJson(const std::string& path);
        bool initialize();
        json::Object runStep();
        json::Object run();
        bool completed() const;

    };

}

#endif // MV_COMPILATION_UNIT_HPP_
