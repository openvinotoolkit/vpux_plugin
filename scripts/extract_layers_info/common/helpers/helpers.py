#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import re
from pathlib import Path


def sortLayers(val):
    secondQuality = ["FakeQuantize", "Const"]
    return "zzzzzzzz" + str(secondQuality.index(val)) + val if val in secondQuality else val


def sortKey(val):
    sortedKeys = ["type", "input_dimentions", "output_dimentions", "pool-method", "kernel", "strides", "group",
                  "dilations", "pads_begin", "pads_end"]
    return sortedKeys.index(val) if val in sortedKeys else 1000000


def get_anchor_id(idNum):
    return "id_" + str(idNum)


def getLineNums(fileName, rePattern):
    lineNums = dict()
    file = Path(str(fileName))

    with file.open() as file:
        nLine = 0
        for line in file:
            nLine = nLine + 1
            line = line.rstrip()
            match = re.search(rePattern, line)
            if match:
                lineNums[get_anchor_id(re.search(r'\d+', match.group(0)).group(0))] = nLine

    return lineNums
