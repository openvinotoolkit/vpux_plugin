#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import os
import csv
import xml.etree.ElementTree as ET
import layer_functions as layer_func

# Function tries to find all files with mask *uint8_int8*.xml or *uint8-int8*.xml in all subdirs from root directory


def get_list_of_all_IRs(rootDir):
    list_of_found_IR_files = list()
    # go over subdirs, looking for IRs and save info to csv-files
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            if fname.endswith('.xml') and ("uint8_int8" in fname or "uint8-int8" in fname):
                list_of_found_IR_files.append(dirName + "/" + fname)

    print("In directory", rootDir, "were found", len(list_of_found_IR_files), "files")
    num = 1
    for file in list_of_found_IR_files:
        print(num, ':', file)
        num += 1

    return list_of_found_IR_files


# Function writes main csv-file for table (layer names, network names, present of layers in network)
def write_main_csv_file(dict_with_layers_and_networks, outputDirName):
    FILENAME = "main.csv"    # file main.csv is used for creation of main table
    fullPathToOutputFile = os.path.join(outputDirName, FILENAME)

    # Create list of all neural networks
    list_of_neural_networks = sorted(dict_with_layers_and_networks.keys())

    # Create list of all layers in all neural networks
    set_all_layer_types = set()
    for key in dict_with_layers_and_networks.keys():
        for layerType in dict_with_layers_and_networks[key]:
            set_all_layer_types.add(layerType)

    list_of_all_layer_types = sorted(list(set_all_layer_types))

    if not (os.path.exists(outputDirName)):
        os.makedirs(outputDirName)

    with open(fullPathToOutputFile, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["", "", "Layers in Neural Networks"])  # Create title for table
        writer.writerow([""])  # One blank row between title and table

        # create row with 2 additional columes for optional common title "Layers" and names of each layer
        row_with_networks_names = [" ", " "] + list_of_neural_networks
        writer.writerow(row_with_networks_names)   # write to file
        for layer in list_of_all_layer_types:
            current_row = [" ", layer]   # add 1 additional colume with empty string for common title "Layers"
            for network in list_of_neural_networks:
                if layer in dict_with_layers_and_networks[network]:  # Check that layer is in neural network
                    current_row.append("TRUE")
                else:
                    current_row.append("FALSE")
            writer.writerow(current_row)

    print(FILENAME, "has been written to", os.path.abspath(fullPathToOutputFile))


# Function gets info about layers in IR-file and writes it into csv-file
def write_info_about_IR_into_csv_file(path_to_IR_file, outputDirName):
    networkName = os.path.basename(path_to_IR_file)[:-4]
    outputFileName = networkName + ".csv"
    fullPathToOutputFile = os.path.join(outputDirName, outputFileName)

    print('\nStart to collect data for', networkName)
    # Reset file for avoiding of append data to previously created file with the same name
    myfile = open(fullPathToOutputFile, "w")
    simple_string_writer = csv.writer(myfile)  # create this writer for add some simple strings to table
    simple_string_writer.writerow(["Layers info for " + networkName])  # Title (network name) for table
    simple_string_writer.writerow([""])  # Blanck row between title and table
    myfile.close()

    list_of_all_layer_types = layer_func.get_list_of_all_layer_types_in_IR_file(path_to_IR_file)

    xml_root = ET.parse(path_to_IR_file).getroot()  # Get root of xml-file (our current IR-file)

    for layer_type in list_of_all_layer_types:
        set_of_data_attributes = layer_func.collect_data_attributes(layer_type, xml_root)

        list_of_data_for_table = list()
        set_of_keys = set()
        for record in sorted(list(set_of_data_attributes)):
            record = record.replace("set()", "None")
            temp_dict = eval(record)
            list_of_keys = temp_dict.keys()
            for key in list_of_keys:
                set_of_keys.add(key)
            # print(list_of_keys)
            list_of_data_for_table.append(temp_dict)
        if 'type' in set_of_keys:
            set_of_keys.remove('type')
        if 'input_dimentions' in set_of_keys:
            set_of_keys.remove('input_dimentions')
        if 'output_dimentions' in set_of_keys:
            set_of_keys.remove('output_dimentions')
        list_of_keys = ['type', 'input_dimentions', 'output_dimentions'] + sorted(list(set_of_keys))

        with open(fullPathToOutputFile, "a", newline="") as file:
            columns = list_of_keys
            # this writer uses dictionary format for writing data to file
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            # write some rows
            writer.writerows(list_of_data_for_table)

            simple_string_writer = csv.writer(file)  # create this writer for add some simple strings to table
            # write one row
            simple_string_writer.writerow([""])

    print('All data for', networkName, 'have been written to file')
    print(os.path.abspath(fullPathToOutputFile))
