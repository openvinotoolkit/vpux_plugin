#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import os.path
import sys
from pathlib import Path
import re

sys.path.append(os.path.abspath("../common"))

from parsing import parsing
from helpers import helpers
from writehtml import writehtml

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
    htmlMain = writehtml.startHtml(file_name=outDir + '/LAYERS_NETWORKS' + ".html",
                                   title='Networks to layers correspondence', styles=["body.css", "table1.css"])
    networkList = sorted(networks.keys())
    layerSet = set(layers.keys())
    htmlRows = [" "]
    for network in networkList:
        netName = networks[network]["netName"]
        template = r'(.{8})' if (len(netName) > 36) else r'(.{6})'
        preparedName = re.sub(template, r'\1 ', netName)
        htmlRows.append("<b>" + writehtml.htmlRef(preparedName, "./mdnetworks/" + netName + ".html") + "</b>")

    nCols = len(htmlRows)
    nRows = 1
    for layer in sorted(list(layerSet), key=helpers.sortLayers):
        htmlRow = ["<b>" + writehtml.htmlRef(layer, "./mdlayers/" + layer + ".html") + "</b>"]
        for network in networkList:
            if layer in networks[network]["layerTypes"]:
                htmlRow.append("<b>" + writehtml.htmlRef("+", "./mdnetworks/" +
                               networks[network]["netName"] + ".html#" + layer) + "</b>")
            else:
                htmlRow.append(" ")
        nRows += 1
        htmlRows.extend(htmlRow)
    writehtml.htmlTable(htmlMain, columns=nCols, rows=nRows, text=htmlRows)
    writehtml.writeHtmlFile(htmlMain)


def layerDataForTable(layers, layerType, manyNets=False):
    layerData = layers[layerType]
    listOfLayers = list(set(layerData.keys()) - set(["FOUND_ATTRIBUTES"]))
    listOfAttr = sorted(list(layerData["FOUND_ATTRIBUTES"]), key=helpers.sortKey)
    htmlRows = list(listOfAttr)
    htmlRows.append("ids")
    nCols = len(htmlRows)
    nRows = 1
    for layerKey in listOfLayers:
        for attr in listOfAttr:
            htmlRows.append(str(layerData[layerKey]["attribs"][attr]
                            if attr in layerData[layerKey]["attribs"] else " "))
        ids = ""
        delimiter0 = ""
        for net in layerData[layerKey]["ids"]:
            if manyNets:
                ids = ids + delimiter0 + \
                    writehtml.htmlRef(net, "../mdnetworks/" + net + ".html#" + layerType) + " &nbsp;ids&nbsp;:&nbsp; "
                delimiter0 = "<br>"
            else:
                ids = "ids&nbsp;:&nbsp;&nbsp; "
            delimiter1 = ""
            for id in layerData[layerKey]["ids"][net]["ids"]:
                ids = ids + delimiter1 + \
                    writehtml.htmlRef(id, "../mdnetworks/" + net + "_xml.html#" + helpers.get_anchor_id(id))
                delimiter1 = ", "
        htmlRows.append(ids)
        nRows = nRows + 1
    return htmlRows, nRows, nCols


def writeNetwork(network):
    htmlFileName = network["netName"]
    xmlToHtmlFileName = outDir + "/mdnetworks/" + htmlFileName + "_xml.html"
    xmlFile = Path(network["path"])
    xmlToHtmlFile = Path(xmlToHtmlFileName)

    output_file = open(str(xmlToHtmlFile), 'w')
    print("""<html>
<title>
IR of network
</title>
<body>
<p1>
IR of network
</p1>
    """, file=output_file)
    print(writehtml.htmlRef("... to 'Networks to layers correspondence'",
          "../LAYERS_NETWORKS.html") + "<xmp>", file=output_file)

    with xmlFile.open() as file:
        for line in file:
            line = line.rstrip()
            match = re.search(r'^\s*<layer\s+id\s*=\s*\"\d+', line)
            if match:
                anchor = '<a href id="' + helpers.get_anchor_id(re.search(r'\d+', match.group(0)).group(0)) + '"></a>'
                print("</xmp>" + anchor + "<xmp>", file=output_file)

            print(line, file=output_file)
    print("""</xmp>
    </body>
    </html>
    """, file=output_file)
    output_file.close()

    htmlNetwork = writehtml.startHtml(outDir + "/mdnetworks/" + htmlFileName + ".html",
                                      "Layers info for " + htmlFileName, styles=["../body.css", "../table4.css"])
    networkIrName = network["irNetName"]
    if networkIrName != None:
        writehtml.htmlWrite(htmlNetwork, "<h1>Network name from IR: " + networkIrName + "</h1>\n")
    writehtml.htmlWrite(htmlNetwork, writehtml.htmlRef(
        "... to 'Networks to layers correspondence'", "../LAYERS_NETWORKS.html"))
    layerTypes = network["layerTypes"]
    for layerType in sorted(list(layerTypes), key=helpers.sortLayers):
        htmlRows, nRows, nCols = layerDataForTable(layerTypes, layerType, False)

        writehtml.htmlWrite(htmlNetwork, '<a href id="' + layerType + '"></a>\n')
        writehtml.htmlWrite(htmlNetwork, "<h1>" + layerType + "</h1> of " + htmlFileName)
        writehtml.htmlWrite(htmlNetwork, writehtml.htmlRef(" (... to all " + layerType + " s)",
                            "../mdlayers/" + layerType + ".html#" + htmlFileName))
        writehtml.htmlTable(htmlNetwork, columns=nCols, rows=nRows, text=htmlRows)
    writehtml.writeHtmlFile(htmlNetwork)


def writeLayer(layers, layerType):
    htmlLayer = writehtml.startHtml(outDir + "/mdlayers/" + layerType + ".html", 'All layers"' + layerType + '"',
                                    styles=["../body.css", "../table4.css"])
    writehtml.htmlWrite(htmlLayer, writehtml.htmlRef(
        "... to 'Networks to layers correspondence'", "../LAYERS_NETWORKS.html"))
    htmlRows, nRows, nCols = layerDataForTable(layers, layerType, True)
    writehtml.htmlTable(htmlLayer, columns=nCols, rows=nRows, text=htmlRows)
    writehtml.writeHtmlFile(htmlLayer)


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
        parsing.parseNetwork(networks, allLayers, net)
        print(net, " parsed")

    writeCrossTable(networks, allLayers)

    for net in networks:
        writeNetwork(networks[net])
        print(networks[net]["netName"], " written")

    for layer in allLayers:
        writeLayer(allLayers, layer)
        print(layer, " written")
