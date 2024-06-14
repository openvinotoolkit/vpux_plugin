#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

# utility for extracting info from IRs
import os
import sys

import layer_functions as layer_func
import file_functions as file_func

# start of script (main)
if len(sys.argv) < 2:
    sys.exit("For using of utility please provide path to root directory with IRs")
elif len(sys.argv) < 3:
    rootDir = sys.argv[1]
    outputDir = "results"
else:
    rootDir = sys.argv[1]
    outputDir = sys.argv[2]

print("Start looking for IR-files in directory:", rootDir)
print("Results will be saved in directory:", os.path.abspath(outputDir) )

list_of_IR_files = file_func.get_list_of_all_IRs(rootDir) # Get list of all IRs for processing
dict_layers_per_IR = layer_func.create_dictionary_network_names_vs_layers(list_of_IR_files)

print("\nWrite data to csv-files.")
file_func.write_main_csv_file(dict_layers_per_IR, outputDir)

for IR_file in list_of_IR_files:
    file_func.write_info_about_IR_into_csv_file(IR_file, outputDir)

print("\nData collection has been finished sucessfully.")
print("Now You could import csv-files to spreadsheet.")
