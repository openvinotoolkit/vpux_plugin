#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"

void generateDotFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(GenerateDot)
        .setFunc(generateDotFcn)
        .setGenre({PassGenre::Validation, PassGenre::Serialization})
        .defineArg(json::JSONType::String, "output")
        .defineArg(json::JSONType::String, "scope")
        .defineArg(json::JSONType::String, "content")
        .defineArg(json::JSONType::Bool, "html")
        .setDescription(
            "Generates the DOT representation of computation model"
        );

    }

}

void generateDotFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object&)
{

    using namespace mv;

    if (compDesc["GenerateDot"]["output"].get<std::string>().empty())
        throw ArgumentError(model, "output", "", "Unspecified output name for generate dot pass");

    std::string outputScope = compDesc["GenerateDot"]["scope"].get<std::string>();
    if (outputScope != "OpModel" && outputScope != "ExecOpModel" && outputScope != "ControlModel" &&
        outputScope != "OpControlModel" && outputScope != "ExecOpControlModel" && outputScope != "DataModel")
        throw ArgumentError(model, "scope", outputScope, "Invalid model scope");

    std::string contentLevel = compDesc["GenerateDot"]["content"].get<std::string>();
    if (contentLevel != "full" && outputScope != "name")
        throw ArgumentError(model, "content", contentLevel, "Invalid content scope");

    bool htmlLike = compDesc["GenerateDot"]["html"].get<bool>();

    std::ofstream ostream;
    ostream.open(compDesc["GenerateDot"]["output"].get<std::string>(), std::ios::trunc | std::ios::out);
    if (!ostream.is_open())
        throw ArgumentError(model, "output", compDesc["GenerateDot"]["output"].get<std::string>(), "Unable to open output file");

    ostream << "digraph G {\n\tgraph [splines=spline]\n";

    if (outputScope != "DataModel")
    {

        OpModel opModel(model);

        for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
        {
            if (!(outputScope == "ControlModel" || outputScope == "ExecOpModel" || outputScope == "ExecOpControlModel") || (opIt->isExecutable() || opIt->getOpType() == OpType::Input || opIt->getOpType() == OpType::Output
                || opIt->getOpType() == OpType::Constant))
            {
                std::string nodeDef = "\t\"" + opIt->getName() + "\" [shape=box,";

                if (htmlLike)
                {
                    nodeDef += " label=<<TABLE Bmv::Order=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + opIt->getName() + "</B></FONT></TD></TR>";
                    if (contentLevel == "full")
                    {
                        std::vector<std::string> attrKeys(opIt->attrsKeys());
                        for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                            nodeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">" + *attrIt + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + opIt->get(*attrIt).toString() + "</FONT></TD></TR>";
                    }
                    else
                    {
                        nodeDef += "<TR><TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + opIt->getOpType().toString() + "</FONT></TD></TR>";
                    }
                    nodeDef += "</TABLE>>";
                }
                else
                {
                    nodeDef += " label=\"" + opIt->getName() + "\\n";
                    if (contentLevel == "full")
                    {
                        std::vector<std::string> attrKeys(opIt->attrsKeys());
                        for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                            nodeDef += *attrIt + ": " + opIt->get(*attrIt).toString() + "\\n";
                    }
                    nodeDef += "\"";
                }

                ostream << nodeDef << "];\n";

            }

        }

        if (outputScope == "OpModel" || outputScope == "ExecOpModel" || outputScope == "OpControlModel" || outputScope == "ExecOpControlModel")
        {

            DataModel dataModel(model);

            for (auto opIt = opModel.getInput(); opIt != opModel.opEnd(); ++opIt)
            {
                if (!(outputScope == "ExecOpModel" || outputScope == "ExecOpControlModel") || (opIt->isExecutable() || opIt->getOpType() == OpType::Input || opIt->getOpType() == OpType::Output
                || opIt->getOpType() == OpType::Constant))
                {
                    for (auto dataIt = opIt.leftmostOutput(); dataIt != dataModel.flowEnd(); ++dataIt)
                    {

                        std::string edgeDef = "\t\"" + opIt->getName() + "\" -> \"" + dataIt.sink()->getName() + "\"";
                        if (htmlLike)
                        {
                            edgeDef += " [penwidth=2.0, label=<<TABLE Bmv::Order=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + dataIt->getTensor()->getName() + "</B></FONT></TD></TR>";
                            if (contentLevel == "full")
                            {
                                std::vector<std::string> attrKeys(dataIt->getTensor()->attrsKeys());
                                for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                                    edgeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">" + *attrIt + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + dataIt->getTensor()->get(*attrIt).toString() + "</FONT></TD></TR>";
                            }
                            else
                            {
                                edgeDef += "<TR><TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + dataIt->getTensor()->getShape().toString() + "</FONT></TD></TR>";
                            }
                            edgeDef += "</TABLE>>];";
                        }
                        else
                        {
                            edgeDef += " [label=\"" + dataIt->getTensor()->getName() + "\\n";
                            if (contentLevel == "full")
                            {
                                std::vector<std::string> attrKeys(dataIt->getTensor()->attrsKeys());
                                for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                                    edgeDef += *attrIt + ": " + dataIt->getTensor()->get(*attrIt).toString() + "\\n";
                            }
                            edgeDef += "\"];";
                        }

                        ostream << edgeDef << "\n";

                    }

                }

            }

        }

        if (outputScope == "ControlModel" || outputScope == "OpControlModel" || outputScope == "ExecOpControlModel")
        {

            ControlModel controlModel(model);

            for (auto opIt = controlModel.getFirst(); opIt != controlModel.opEnd(); ++opIt)
            {

                for (auto controlIt = opIt.leftmostOutput(); controlIt != controlModel.flowEnd(); ++controlIt)
                {

                    std::string edgeDef = "\t" + opIt->getName() + " -> " + controlIt.sink()->getName() + " [penwidth=2.0, style=dashed]";
                    ostream << edgeDef << "\n";

                }

            }

        }

    }
    else
    {

        DataModel dataModel(model);

        for (auto tIt = dataModel.tensorBegin(); tIt != dataModel.tensorEnd(); ++tIt)
        {

            std::string nodeDef = "\t\"" + tIt->getName() + "\" [shape=box,";

            if (htmlLike)
            {
                nodeDef += " label=<<TABLE Bmv::Order=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + tIt->getName() + "</B></FONT></TD></TR>";
                if (contentLevel == "full")
                {
                    std::vector<std::string> attrKeys(tIt->attrsKeys());
                    for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                        nodeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">" + *attrIt + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + tIt->get(*attrIt).toString() + "</FONT></TD></TR>";
                }
                else
                {
                    nodeDef += "<TR><TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + tIt->getShape().toString() + "</FONT></TD></TR>";
                }
                nodeDef += "</TABLE>>";
            }
            else
            {
                nodeDef += " label=\"" + tIt->getName() + "\\n";
                if (contentLevel == "full")
                {
                    std::vector<std::string> attrKeys(tIt->attrsKeys());
                    for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                        nodeDef += *attrIt + ": " + tIt->get(*attrIt).toString() + "\\n";
                }
                nodeDef += "\"";
            }

            ostream << nodeDef << "];\n";

        }

        for (auto flowIt = dataModel.flowBegin(); flowIt != dataModel.flowEnd(); ++flowIt)
        {

            if (flowIt.childrenSize() > 0)
            {

                std::string edgeDef = "\t\"" + flowIt->getTensor()->getName() + "\" -> \"" + flowIt.leftmostChild()->getTensor()->getName() + "\"";
                if (htmlLike)
                {
                    edgeDef += " [penwidth=2.0, label=<<TABLE Bmv::Order=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + flowIt.sink()->getName() + "</B></FONT></TD></TR>";
                    if (contentLevel == "full")
                    {
                        std::vector<std::string> attrKeys(flowIt.sink()->attrsKeys());
                        for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                            edgeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">" + *attrIt + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + flowIt.sink()->get(*attrIt).toString() + "</FONT></TD></TR>";
                    }
                    else
                    {
                        edgeDef += "<TR><TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">" + flowIt.sink()->getOpType().toString() + "</FONT></TD></TR>";
                    }

                    edgeDef += "</TABLE>>];";
                }
                else
                {
                    edgeDef += " [label=\"" + flowIt.sink()->getName() + "\\n";
                    if (contentLevel == "full")
                    {
                        std::vector<std::string> attrKeys(flowIt.sink()->attrsKeys());
                        for (auto attrIt = attrKeys.begin(); attrIt != attrKeys.end(); ++attrIt)
                            edgeDef += *attrIt + ": " + flowIt.sink()->get(*attrIt).toString() + "\\n";
                    }
                    edgeDef += "\"];";
                }

                ostream << edgeDef << "\n";

            }

        }

    }

    ostream << "}\n";
    ostream.close();

}