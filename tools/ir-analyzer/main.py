#!/usr/bin/env python3
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
import argparse
import xmltodict
import typing
import csv
import logging
import struct
from distutils import util


layers_to_exclude = ["Const"]

# Feel free to supplement and correct this list
# Input numeration start from 0
interesting_layers = {
    "Gather": [2],
    "Pad": [1, 2, 3],
    "Split": [1],
    "ReverseSequence": [1],
    "SpaceToBatch": [1, 2, 3],
    "StridedSlice": [1, 2, 3],
    "Broadcast": [1, 2],
    "ScatterElementsUpdate": [3],
    "ScatterUpdate": [3],
    "Tile": [1],
    "Reshape": [1],
    "Squeeze": [1],
    "Unsqueeze": [1],
    "ConvolutionBackpropData": [2],
    "GroupConvolutionBackpropData": [2],
    "Proposal": [2],
    "ROIAlign": [2],
    "PriorBox": [0, 1],
    "Interpolate": [1, 2, 3],
    "LRN": [1],
    "MVN": [1],
    "NormalizeL2": [1],
    "BatchNormInference": [1, 2, 3, 4],
    "FakeQuantize": [1, 2, 3, 4],
    "ReduceL1": [1],
    "ReduceL2": [1],
    "ReduceLogicalAnd": [1],
    "ReduceLogicalOr": [1],
    "ReduceMax": [1],
    "ReduceMean": [1],
    "ReduceMin": [1],
    "ReduceProd": [1],
    "ReduceSum": [1],
    "CTCGreedyDecoderSeqLen": [1, 2],
    "LSTMSequence": [3],
    "OneHot": [1, 2, 3],
    "ExperimentalDetectronTopKROIs": [1],
    "TopK": [1],
    "NonMaxSuppression": [2, 3, 4]
}


def parse_args():
    parser = argparse.ArgumentParser(description="Tool for extracting layers info from IR")
    parser.add_argument('path_to_model', metavar='PATH_TO_MODEL', type=str,
                        help='Required. Path to an .xml file with a trained model.')
    parser.add_argument('-o', '--path_to_output', metavar='PATH_TO_OUTPUT', type=str, required=False, default="output_data.csv",
                        help='Optional. Path to output file with statistics. (default: output_data.csv)')
    parser.add_argument('-rd', '--remove_duplicates', metavar='TRUE/FALSE', type=util.strtobool, required=False, default=True,
                        help='Optional. Eliminate duplicated combinations from output. (default: True)')
    parser.add_argument('-s', '--sort', metavar='TRUE/FALSE', type=util.strtobool, required=False, default=True,
                        help='Optional. Sort resulting combinations. (default: True)')
    parser.add_argument('-rc', '--read_constants', metavar='TRUE/FALSE', type=util.strtobool, required=False, default=True,
                        help='Optional. Read data from constants. (default: True)')
    return parser.parse_args()


def get_layout_from_ndims(ndims: int) -> typing.List[str]:
    if ndims == 0:
        return []
    elif ndims == 1:
        return ["C"]
    elif ndims == 2:
        return ["N", "C"]
    elif ndims == 3:
        return ["C", "H", "W"]
    elif ndims == 4:
        return ["N", "C", "H", "W"]
    elif ndims == 5:
        return ["N", "C", "D", "H", "W"]
    else:
        logging.warning("Unsupported number of dims! ndims > 5 is unsupported. ndims = {}".format(ndims))
        return ["UNDEFINED"]


def get_inputs(root: dict) -> typing.List[dict]:
    return get_ports(root.get("input", {}))


def get_outputs(root: dict) -> typing.List[dict]:
    return get_ports(root.get("output", {}))


def get_attributes(root: dict) -> dict:
    return root.get("data", [])


def get_ports(root: dict) -> typing.List[dict]:
    ports = root.get("port", [])
    if not isinstance(ports, list):
        ports = [ports]
    return ports


def get_dims(root: dict) -> typing.List[str]:
    dims = root.get("dim", [])
    if isinstance(dims, str):
        dims = [dims]
    return dims


def get_layers(root: dict) -> typing.List[dict]:
    return root.get("layers", {}).get("layer", [{}])


def get_type(root: dict) -> str:
    return root.get("@type")


def get_opset(root: dict) -> str:
    return root.get("@version")


def get_precision(root: dict) -> str:
    return root.get("@precision", "")


def get_edges(root: dict) -> typing.List[dict]:
    return root.get("edges", {}).get("edge", [])


def get_id(root: dict):
    return root.get("@id", "")


def ie_to_unpack_type(ie_prc):
    if ie_prc == 'FP16':
        return 'e'
    if ie_prc == 'FP32':
        return 'f'
    if ie_prc == 'FP64':
        return 'd'
    if ie_prc == 'I8':
        return 'b'
    if ie_prc == 'U8':
        return 'B'
    if ie_prc == 'I16':
        return 'h'
    if ie_prc == 'U16':
        return 'H'
    if ie_prc == 'I32':
        return 'i'
    if ie_prc == 'U32':
        return 'I'
    if ie_prc == 'I64':
        return 'l'
    if ie_prc == 'U64':
        return 'L'
    logging.warning("Unrecognized precision {}. Please append right conversion.".format(ie_prc))


def element_byte_size(ie_prc):
    num_bits = ''.join(i for i in ie_prc if i.isdigit())
    if len(num_bits) == 0:
        return
    num_bits = int(num_bits)
    return num_bits / 8


def get_binary_array(bin_file, offset, bytesize, num_elems, ie_prc):
    unpack_type = ie_to_unpack_type(ie_prc)
    if unpack_type:
        bin_file.seek(offset)
        return struct.unpack("{0}{1}".format(num_elems, unpack_type), bin_file.read(bytesize))
    return ()


def get_const_data(layer_id, port_id, layers, edges, bin_file):
    const_input = list(filter(lambda x: x['@to-layer'] == layer_id and x['@to-port'] == port_id, edges))
    if len(const_input) != 1:
        return
    const_input_id = const_input[0]['@from-layer']
    const_input = list(filter(lambda x: get_id(x) == const_input_id and get_type(x) == 'Const', layers))
    if len(const_input) != 1:
        return
    const_input = const_input[0]
    const_attr = get_attributes(const_input)
    offset = int(const_attr['@offset'])
    bytesize = int(const_attr['@size'])
    output_info = get_outputs(const_input)
    if len(output_info) != 1:
        return
    output_info = output_info[0]
    precision = get_precision(output_info)
    elem_byte_size = element_byte_size(precision)
    if not elem_byte_size:
        return
    return get_binary_array(bin_file, offset, bytesize, int(bytesize / elem_byte_size), precision)


def get_layer_info(root: dict, edges, layers, bin_file, read_const) -> typing.List[typing.List[str]]:
    layer_type = get_type(root)
    if layer_type not in layers_to_exclude:
        output_data = []
        layer_info = []
        layer_info.append(get_type(root))
        layer_info.append(get_opset(root))
        inputs = get_inputs(root)
        outputs = get_outputs(root)
        attributes = get_attributes(root)
        if inputs:
            for i, input_info in enumerate(inputs):
                dims = get_dims(input_info)
                layout = get_layout_from_ndims(len(dims))
                if layer_type in interesting_layers and i in interesting_layers[layer_type] and read_const:
                    const_array = get_const_data(get_id(root), get_id(input_info), layers, edges, bin_file)
                    if const_array:
                        layer_info.append('const in: ' + str(list(map(lambda x: int(x), dims))) + ' (' + ''.join(layout) + " " + get_precision(input_info) + ') ' +
                                          'data: [' + ','.join(list(map(lambda x: str(x), const_array))) + ']')
                        continue
                layer_info.append('in: ' + str(list(map(lambda x: int(x), dims))) +
                                  ' (' + ''.join(layout) + " " + get_precision(input_info) + ')')
        if outputs:
            for output_info in outputs:
                dims = get_dims(output_info)
                layout = get_layout_from_ndims(len(dims))
                layer_info.append('out: ' + str(list(map(lambda x: int(x), dims))) +
                                  ' (' + ''.join(layout) + " " + get_precision(output_info) + ')')
        if attributes:
            for k, v in attributes.items():
                layer_info.append('='.join([k[1:], v]))
        output_data.append(layer_info)
        if "body" in root.keys():
            ti_body = root["body"]
            ti_layers = get_layers(ti_body)
            ti_edges = get_edges(ti_body)
            for layer in ti_layers:
                sublayers_info = get_layer_info(layer, ti_edges, ti_layers, bin_file, read_const)
                if sublayers_info:
                    output_data.extend(sublayers_info)
        return output_data


def get_layers_info(root: dict, is_only_unique: bool, need_sort: bool, read_const: bool, bin_file) -> typing.List[typing.List[str]]:
    net = root["net"]
    layers = get_layers(net)
    edges = get_edges(net)
    output_data = []
    for layer in layers:
        layer_info = get_layer_info(layer, edges, layers, bin_file, read_const)
        if layer_info:
            output_data.extend(layer_info)
    if is_only_unique:
        output_data = list(set(map(lambda x: tuple(x), output_data)))
    if need_sort:
        output_data.sort()
    return output_data


if __name__ == "__main__":
    args = parse_args()

    with open(args.path_to_model) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())

    with open(args.path_to_model[:-3] + "bin", "rb") as bin_file:
        output_data = get_layers_info(data_dict, bool(args.remove_duplicates),
                                      bool(args.sort), bool(args.read_constants), bin_file)

    with open(args.path_to_output, "w") as output_file:
        output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        output_writer.writerows(output_data)
