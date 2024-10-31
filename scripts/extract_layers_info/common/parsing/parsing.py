#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import os
import os.path
import xml.etree.ElementTree as ET
from helpers import helpers


def getNetworks(rootDir):
    networkDict = dict()
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fileName in fileList:
            if fileName.endswith('.xml'):
                netKey = str(os.path.join(dirName, fileName))
                networkDict[netKey] = dict()
                networkDict[netKey]["path"] = os.path.join(dirName, fileName)
                networkDict[netKey]["relPath"] = os.path.relpath(os.path.join(dirName, fileName), rootDir)
                networkDict[netKey]["fileName"] = fileName
                networkDict[netKey]["dirName"] = dirName
                networkDict[netKey]["netName"] = os.path.splitext(fileName)[0]
                networkDict[netKey]["layerTypes"] = dict()
    return networkDict


def getLayerAttribs(layer):
    dict_one_layer_info = dict()
    dict_one_layer_info['type'] = layer.attrib["type"]
    if layer.find('data') != None:
        for data in layer.findall('data'):
            if "type" in data.attrib.keys():
                data.attrib["type(in data)"] = data.attrib["type"]
                del (data.attrib["type"])
            for key in sorted(data.attrib.keys()):
                if data.attrib.get(key) != None:
                    dict_one_layer_info[key] = data.attrib.get(key)
    if 'output' in dict_one_layer_info:
        del dict_one_layer_info['output']
    layer_input_dimentions = list()
    for layer_in in layer.findall('input'):
        for port in layer_in.findall('port'):
            current_dimention = list()
            for dim in port.findall('dim'):
                current_dimention.append(int(dim.text))
            layer_input_dimentions.append(tuple(current_dimention))
    if len(layer_input_dimentions) > 0:
        dict_one_layer_info["input_dimentions"] = layer_input_dimentions
    layer_output_dimentions = list()
    for layer_out in layer.findall('output'):
        for port in layer_out.findall('port'):
            current_dimention = list()
            for dim in port.findall('dim'):
                current_dimention.append(int(dim.text))
            layer_output_dimentions.append(tuple(current_dimention))
    if len(layer_output_dimentions) > 0:
        dict_one_layer_info["output_dimentions"] = layer_output_dimentions
    dict_one_layer_info["id"] = layer.attrib["id"]
    return dict_one_layer_info


def parseNetwork(networks, allLayers, net):
    xml_root = ET.parse(networks[net]["path"]).getroot()
    if xml_root.tag != 'net':
        del (networks[net])
        return

    networks[net]["irNetName"] = xml_root.attrib.get("name")
    networks[net]["xmlLineNums"] = helpers.getLineNums(networks[net]["path"], r'^\s*<layer\s+id\s*=\s*\"\d+')

    for XMLLayers in xml_root.findall('layers'):
        for layer in XMLLayers.findall('layer'):
            layerAttribs = getLayerAttribs(layer)
            curId = layerAttribs["id"]
            del (layerAttribs["id"])
            layerKey = str(layerAttribs)
            curType = layerAttribs["type"]
            if curType not in networks[net]["layerTypes"]:
                networks[net]["layerTypes"][curType] = dict()
                networks[net]["layerTypes"][curType]["FOUND_ATTRIBUTES"] = set()
            if layerKey not in networks[net]["layerTypes"][curType]:
                networks[net]["layerTypes"][curType][layerKey] = dict()
                networks[net]["layerTypes"][curType][layerKey]["attribs"] = dict(layerAttribs)
                networks[net]["layerTypes"][curType][layerKey]["ids"] = dict()
                networks[net]["layerTypes"][curType][layerKey]["ids"][networks[net]
                                                                      ["netName"]] = {"ids": list(), "lines": networks[net]["xmlLineNums"]}

            if curType not in allLayers:
                allLayers[curType] = dict()
                allLayers[curType]["FOUND_ATTRIBUTES"] = set()
            if layerKey not in allLayers[curType]:
                allLayers[curType][layerKey] = dict()
                allLayers[curType][layerKey]["attribs"] = dict(layerAttribs)
                allLayers[curType][layerKey]["ids"] = dict()
            if networks[net]["netName"] not in allLayers[curType][layerKey]["ids"]:
                allLayers[curType][layerKey]["ids"][networks[net]["netName"]] = {
                    "ids": list(), "lines": networks[net]["xmlLineNums"]}

            networks[net]["layerTypes"][curType][layerKey]["ids"][networks[net]["netName"]]["ids"].append(curId)
            networks[net]["layerTypes"][curType]["FOUND_ATTRIBUTES"] = networks[net]["layerTypes"][curType]["FOUND_ATTRIBUTES"] | set(
                layerAttribs)
            allLayers[curType][layerKey]["ids"][networks[net]["netName"]]["ids"].append(curId)
            allLayers[curType]["FOUND_ATTRIBUTES"] = allLayers[curType]["FOUND_ATTRIBUTES"] | set(layerAttribs)
