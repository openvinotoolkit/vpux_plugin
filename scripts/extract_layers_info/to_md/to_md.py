#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import os.path
import shutil
import sys
from pathlib import Path
import re

sys.path.append(os.path.abspath("../common"))

from parsing import parsing
from helpers import helpers
from mdutils.mdutils import MdUtils

outDir = "."


def setOutDir(newOutDir):
    outDir = newOutDir


def printUsage(desc=False):
    if desc:
        print("Searches the directory tree (starting from 'SOURCE_DIRECTORY') the xml files with OpenVINO IRs,\n",
              "    should be run in the directory where it lies,\n",
              "    prepares the layers information from IRs into Markdown formatted files with layers lists and pivot table,\n",
              "    places the results in 'OUTPUT_DIRECTORY'.\n")
    print("Usage: \n",
          "    python info.py <SOURCE_DIRECTORY> [<OUTPUT_DIRECTORY>('./' as default)])")


def writeCrossTable(networks, layers):
    mdMain = MdUtils(file_name=outDir + '/LAYERS_NETWORKS', title='Networks to layers correspondence')
    mdMain.new_line()
    networkList = sorted(networks.keys())
    layerSet = set(layers.keys())
    mdRows = [" "]
    for network in networkList:
        netName = networks[network]["netName"]
        template = r'(.{8})' if (len(netName) > 36) else r'(.{6})'
        preparedName = re.sub(template, r'\1 ', netName)
        mdRows.append("[" + preparedName + "](./mdnetworks/" + netName + ".md)")

    nCols = len(mdRows)
    nRows = 1
    for layer in sorted(list(layerSet), key=helpers.sortLayers):
        mdRow = ["[" + layer + "](./mdlayers/" + layer + ".md)"]
        for network in networkList:
            if layer in networks[network]["layerTypes"]:
                mdRow.append("&nbsp; &nbsp; **[ + ](./mdnetworks/" + networks[network]
                             ["netName"] + ".md#" + str(layer).lower() + ")**")
            else:
                mdRow.append(" ")
        nRows += 1
        mdRows.extend(mdRow)
    mdMain.new_table(columns=nCols, rows=nRows, text=mdRows, text_align='left')
    mdMain.create_md_file()


def layerDataForTable(layers, layerType, manyNets=False):
    layerData = layers[layerType]
    listOfLayers = list(set(layerData.keys()) - set(["FOUND_ATTRIBUTES"]))
    listOfAttr = sorted(list(layerData["FOUND_ATTRIBUTES"]), key=helpers.sortKey)
    mdRows = list(listOfAttr)
    mdRows.append("ids")
    nCols = len(mdRows)
    nRows = 0
    for layerKey in listOfLayers:
        for attr in listOfAttr:
            mdRows.append(str(layerData[layerKey]["attribs"][attr] if attr in layerData[layerKey]["attribs"] else " "))
        ids = ""
        delimiter0 = ""
        for net in layerData[layerKey]["ids"]:
            if manyNets:
                ids = ids + delimiter0 + "[" + net + "](../mdnetworks/" + net + \
                    ".md#" + layerType.lower() + ") &nbsp;ids&nbsp;:&nbsp; "
                delimiter0 = "<br>"
            else:
                ids = "ids&nbsp;:&nbsp;&nbsp; "
            delimiter1 = ""
            xmlLineNums = layerData[layerKey]["ids"][net]["lines"]
            for id in layerData[layerKey]["ids"][net]["ids"]:
                ids = ids + delimiter1 + "[" + id + "](../mdnetworks/" + net + \
                    ".1xml#L" + str(xmlLineNums[helpers.get_anchor_id(id)]) + ")"
                delimiter1 = ", "
        mdRows.append(ids)
        nRows = nRows + 1
    return mdRows, nRows, nCols


def writeNetwork(network):
    mdFileName = network["netName"]
    xmlToMdFileName = outDir + "/mdnetworks/" + mdFileName + ".1xml"
    xmlFile = Path(network["path"])
    shutil.copyfile(str(xmlFile), xmlToMdFileName)

    mdNetwork = MdUtils(file_name=outDir + "/mdnetworks/" + mdFileName)
    mdNetwork.new_header(1, "Layers info for " + mdFileName)
    networkIrName = network["irNetName"]
    if networkIrName != None:
        mdNetwork.write("**Network name from IR: " + networkIrName + "**\n")
    mdNetwork.new_header(2, " ")
    mdNetwork.write("[... to 'Networks to layers correspondence'](../LAYERS_NETWORKS.md)\n")

    layerTypes = network["layerTypes"]
    for layerType in sorted(list(layerTypes), key=helpers.sortLayers):
        mdRows, nRows, nCols = layerDataForTable(layerTypes, layerType, False)

        mdNetwork.write('\n<a href id="' + layerType.lower() + '"></a>\n')
        mdNetwork.new_header(3, layerType.lower())
        mdNetwork.write("**of " + mdFileName + " ([... to all " + layerType +
                        " s](../mdlayers/" + layerType + ".md#" + mdFileName.lower() + "))**\n")
        mdNetwork.new_table(columns=nCols, rows=nRows + 1, text=mdRows, text_align='left')

    mdNetwork.create_md_file()


def writeLayer(layers, layerType):
    mdLayer = MdUtils(file_name=outDir + "/mdlayers/" + layerType)
    mdLayer.new_header(1, 'All layers"' + layerType + '"')
    mdLayer.new_header(2, " ")
    mdLayer.write("[... to 'Networks to layers correspondence'](../LAYERS_NETWORKS.md)\n")

    mdRows, nRows, nCols = layerDataForTable(layers, layerType, True)
    mdLayer.new_table(columns=nCols, rows=nRows + 1, text=mdRows, text_align='left')
    mdLayer.create_md_file()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nplease set path to root directory with IRs\n\n")
        printUsage()
        sys.exit(1)

    if re.search(r'^\s*-+[hH]', sys.argv[1]):
        printUsage(True)
        sys.exit(0)

    rootDir = sys.argv[1]
    if len(sys.argv) > 2:
        setOutDir(sys.argv[2])
    print("Start looking for IR-files in directory:", rootDir)

    if not os.path.exists(outDir + "/mdlayers"):
        os.makedirs(outDir + "/mdlayers")
    if not os.path.exists(outDir + "/mdnetworks"):
        os.makedirs(outDir + "/mdnetworks")

    networks = parsing.getNetworks(rootDir)
    allLayers = dict()

    for net in networks:
        print(net, " parsed")
        parsing.parseNetwork(networks, allLayers, net)

    writeCrossTable(networks, allLayers)

    for net in networks:
        writeNetwork(networks[net])
        print(networks[net]["netName"], " written")

    for layer in allLayers:
        writeLayer(allLayers, layer)
        print(layer, " written")
