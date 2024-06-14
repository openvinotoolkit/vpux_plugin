#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

# module with functions for extract information from IR's layers
import xml.etree.ElementTree as ET
import os

# Fuction extract all layer types from IR-file
def get_list_of_all_layer_types_in_IR_file(path_to_file):
    set_of_layer_types_in_CNN = set()

    xml_root = ET.parse(path_to_file).getroot()

    for layers in xml_root.findall('layers'):
        for layer in layers.findall('layer'):
            set_of_layer_types_in_CNN.add(layer.attrib['type'])

    list_of_layer_types = sorted( list(set_of_layer_types_in_CNN) )

    return list_of_layer_types


# Function extract all layer types from list of IR-files found in root directory and subdirs
def find_all_layer_types_in_all_IR_files(list_of_files):
    types_of_all_layers = set()

    for fileName in list_of_files:
        list_of_layers_for_current_CNN = get_list_of_all_layer_types_in_IR_file(fileName)

        # Collect all layer types in one set
        for layer in list_of_layers_for_current_CNN:
            types_of_all_layers.add(layer)

    return sorted(list(types_of_all_layers))


# Function create dictionary networks names vs layers (network names - keys)
def create_dictionary_network_names_vs_layers(list_of_files):
    neural_networks_to_layers_dictionary = dict()

    list_of_all_layer_types = find_all_layer_types_in_all_IR_files(list_of_files)

    for fileName in list_of_files:
        list_of_layers_for_current_CNN = get_list_of_all_layer_types_in_IR_file(fileName)
        # create record in dictionary for current IR
        neural_networks_to_layers_dictionary[ os.path.basename(fileName)[:-4] ] = sorted(list_of_layers_for_current_CNN)

    return neural_networks_to_layers_dictionary


# Function collect set of data attributes for particular layer (except attribute "output")
def collect_data_attributes(layer_type, xml_root):
    set_of_layer_attributes = set()
    dict_one_layer_info = dict()
    layer_input_dimentions = set()
    layer_output_dimentions = set()
    # get info from data attributes
    for layers in xml_root.findall('layers'):
        for layer in layers.findall('layer'):
            if layer.attrib["type"] == layer_type:
                #begin creation of dictionary with info about this particular layer
                dict_one_layer_info.clear()
                dict_one_layer_info['type'] = layer_type
                if layer.find('data') != None:
                    for data in layer.findall('data'):
                        for key in sorted(data.attrib.keys()):
                            if data.attrib.get(key) != None:
                                dict_one_layer_info[key] = data.attrib.get(key)
                if 'output' in dict_one_layer_info:
                    del dict_one_layer_info['output']
                #special processing for input and output dimentions
                for layer_in in layer.findall('input'):
                    layer_input_dimentions.clear()
                    for port in layer_in.findall('port'):
                        current_dimention = list()
                        for dim in port.findall('dim'):
                            current_dimention.append( int(dim.text) )
                        layer_input_dimentions.add( tuple(current_dimention) )
                dict_one_layer_info["input_dimentions"] = layer_input_dimentions
                for layer_out in layer.findall('output'):
                    layer_output_dimentions.clear()
                    for port in layer_out.findall('port'):
                        current_dimention = list()
                        for dim in port.findall('dim'):
                            current_dimention.append( int(dim.text) )
                        layer_output_dimentions.add( tuple(current_dimention) )
                dict_one_layer_info["output_dimentions"] = layer_output_dimentions
                set_of_layer_attributes.add(str(dict_one_layer_info))

    return set_of_layer_attributes
