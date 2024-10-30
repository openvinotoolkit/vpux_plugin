#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import glob
import os
import sys
import xlsxwriter

print('Test Parser V1.0')


def prepareAndSplitLine(line):
    line = line.replace(' ', ',')
    line = line.replace('(', ',')
    line = line.replace(')', ',')
    line = line.replace('[', ',')
    line = line.replace(']', ',')
    line = line.replace('\n', ',')
    splited_line_raw = line.split(',')

    drop_comments = True
    if (line.find('Track,number') != -1):
        drop_comments = False

    splited_line_filtered = []
    for substr in splited_line_raw:
        if ((drop_comments and substr.find('//') != -1) or substr.find('{') != -1):
            break
        else:
            if (substr and substr != '\n'):
                splited_line_filtered.append(substr)
    return splited_line_filtered


def appendTestInfoIntoXlsDoc(worksheet, row, col, source_file, test_type, test_prefix, test_case_or_fixture_name, test_name, jira_ticket, test_status):
    worksheet.write(row, col, row)
    worksheet.write(row, col + 1, source_file)
    worksheet.write(row, col + 2, test_type)
    worksheet.write(row, col + 3, test_prefix)
    worksheet.write(row, col + 4, test_case_or_fixture_name)
    worksheet.write(row, col + 5, test_name)
    worksheet.write(row, col + 6, jira_ticket)
    worksheet.write(row, col + 7, test_status)


def writeWorksheetTitles(worksheet):
    worksheet.write(0, 1, 'SOURCE_FILE')
    worksheet.write(0, 2, 'TEST_TYPE')
    worksheet.write(0, 3, 'TEST_PREFIX')
    worksheet.write(0, 4, 'TEST_CASE/FIXTURE_NAME')
    worksheet.write(0, 5, 'TEST_NAME')
    worksheet.write(0, 6, 'JIRA_TICKET')
    worksheet.write(0, 7, 'TEST_STATUS')


def generateFormat(color):
    return workbook.add_format({'bold': False,
                                'border': 6,
                                'align': 'center',
                                'valign': 'vcenter',
                                'bg_color': color,
                                })


dir_path = sys.argv[1]
result_path = sys.argv[2]

all_files = []

print('List of source files (*.cpp files supported only)')
for root, directories, filenames in os.walk(dir_path):
    for filename in filenames:
        if (filename.find('.cpp') != -1 and filename.find('.cpp.o') == -1):
            print os.path.join(root, filename)
            all_files.append(os.path.join(root, filename))

all_tests_f = []
all_tests_p = []
all_tests_p_instantiate = []

for file_path in all_files:
    print('SCAN:', file_path)
    with open(file_path) as fp:

        line = "initial"
        prev_line_splitted = "previos"
        cnt = 1

        while line:
            line = fp.readline()
            line_splitted = prepareAndSplitLine(line)

            if (len(line_splitted) > 0):
                if (line_splitted[0] == ('TEST_F') or line_splitted[0] == ('TEST_P') or line_splitted[0] == ('INSTANTIATE_TEST_SUITE_P')):
                    related_ticket = ''
                    if (len(prev_line_splitted) > 3 and prev_line_splitted[1] == ('Track') and prev_line_splitted[2] == ('number:')):
                        related_ticket = prev_line_splitted[3]

                    if (len(line_splitted) < 3):
                        next_line = fp.readline()
                        next_line_splitted = prepareAndSplitLine(next_line)
                        for substr in next_line_splitted:
                            line_splitted.append(substr)

                    if (len(line_splitted) > 3):
                        del line_splitted[3: len(line_splitted)]

                    line_splitted.insert(0, file_path)
                    line_splitted.append(related_ticket)
                    if (line_splitted[1] == 'TEST_F'):
                        print(line_splitted)
                        all_tests_f.append(line_splitted)
                    if (line_splitted[1] == 'TEST_P'):
                        print(line_splitted)
                        all_tests_p.append(line_splitted)
                    if (line_splitted[1] == 'INSTANTIATE_TEST_SUITE_P'):
                        print(line_splitted)
                        all_tests_p_instantiate.append(line_splitted)
            prev_line_splitted = line_splitted

print("Found {} TEST_F".format(len(all_tests_f)))
print("Found {} TEST_P".format(len(all_tests_p)))
print("Found {} Instantiate Tests".format(len(all_tests_p_instantiate)))

workbook = xlsxwriter.Workbook(result_path + 'TestParseResult.xlsx')

worksheet_all_tests = workbook.add_worksheet('all_tests')
worksheet_enabled_tests = workbook.add_worksheet('enabled_tests')
worksheet_disabled_tests = workbook.add_worksheet('disabled_tests')
worksheet_nightly_tests = workbook.add_worksheet('nightly_tests')

row_all_tests = 1
row_enabled_tests = 1
row_disabled_tests = 1
row_nightly_tests = 1

col = 0

writeWorksheetTitles(worksheet_all_tests)
writeWorksheetTitles(worksheet_enabled_tests)
writeWorksheetTitles(worksheet_disabled_tests)
writeWorksheetTitles(worksheet_nightly_tests)

for test_info in all_tests_f:

    if (len(test_info) == 5):
        disabled = False
        if (test_info[2].find('DISABLED') != -1 or test_info[3].find('DISABLED') != -1):
            disabled = True
            appendTestInfoIntoXlsDoc(worksheet_disabled_tests, row_disabled_tests, col,
                                     test_info[0], test_info[1], '', test_info[2], test_info[3], test_info[4], 'DISABLE')
            row_disabled_tests += 1
        else:
            appendTestInfoIntoXlsDoc(worksheet_enabled_tests, row_enabled_tests, col,
                                     test_info[0], test_info[1], '', test_info[2], test_info[3], test_info[4], 'ENABLE')
            row_enabled_tests += 1

        if (test_info[2].find('nightly') != -1 or test_info[3].find('nightly') != -1):
            appendTestInfoIntoXlsDoc(worksheet_nightly_tests, row_nightly_tests, col,
                                     test_info[0], test_info[1], '', test_info[2], test_info[3], test_info[4], 'DISABLE' if disabled else 'ENABLE')
            row_nightly_tests += 1

        appendTestInfoIntoXlsDoc(worksheet_all_tests, row_all_tests, col,
                                 test_info[0], test_info[1], '', test_info[2], test_info[3], test_info[4], 'DISABLE' if disabled else 'ENABLE')
    else:
        test_info_concat = ''
        for test_info_part in test_info:
            test_info_concat = test_info_concat + test_info_part
        appendTestInfoIntoXlsDoc(worksheet_all_tests, row_all_tests, col, 'PARSE_ERROR',
                                 test_info_concat, '', '', '', '', 'UNDEF')
    row_all_tests += 1

cnt_instantiated_p_tests = 0
for test_instantiate_info in all_tests_p_instantiate:
    print('<', test_instantiate_info[1], test_instantiate_info[2], test_instantiate_info[3], '>')
    cnt_relatives_tests = 0
    for test_p_info in all_tests_p:
        if (test_instantiate_info[3] == test_p_info[2]):
            disabled = False
            if (test_instantiate_info[2].find('DISABLED') != -1 or test_p_info[2].find('DISABLED') != -1 or test_p_info[3].find('DISABLED') != -1):
                disabled = True
                appendTestInfoIntoXlsDoc(worksheet_disabled_tests, row_disabled_tests, col, test_p_info[0], test_p_info[
                                         1], test_instantiate_info[2], test_p_info[2], test_p_info[3], test_instantiate_info[4] + '/' + test_p_info[4], 'DISABLE')
                row_disabled_tests += 1
            else:
                appendTestInfoIntoXlsDoc(worksheet_enabled_tests, row_enabled_tests, col, test_p_info[0], test_p_info[1], test_instantiate_info[
                                         2], test_p_info[2], test_p_info[3], test_instantiate_info[4] + '/' + test_p_info[4], 'ENABLE')
                row_enabled_tests += 1
                cnt_instantiated_p_tests += 1

            if (test_instantiate_info[2].find('nightly') != -1 or test_p_info[2].find('nightly') != -1 or test_p_info[3].find('nightly') != -1):
                appendTestInfoIntoXlsDoc(worksheet_nightly_tests, row_nightly_tests, col, test_p_info[0], test_p_info[1], test_instantiate_info[
                                         2], test_p_info[2], test_p_info[3], test_instantiate_info[4] + '/' + test_p_info[4], 'DISABLE' if disabled else 'ENABLE')
                row_nightly_tests += 1

            appendTestInfoIntoXlsDoc(worksheet_all_tests, row_all_tests, col, test_p_info[0], test_p_info[1], test_instantiate_info[
                                     2], test_p_info[2], test_p_info[3], test_instantiate_info[4] + '/' + test_p_info[4], 'DISABLE' if disabled else 'ENABLE')
            row_all_tests += 1

            cnt_relatives_tests += 1

    if (cnt_relatives_tests == 0):
        print(test_instantiate_info)
        disabled = False
        if (len(test_instantiate_info) > 4):
            if (test_instantiate_info[2].find('DISABLED') != -1 or test_instantiate_info[3].find('DISABLED') != -1):
                print("FOUND DISABLE")
                disabled = True
                appendTestInfoIntoXlsDoc(worksheet_disabled_tests, row_disabled_tests, col, test_instantiate_info[0], test_instantiate_info[
                                         1], test_instantiate_info[2], test_instantiate_info[3], '', test_instantiate_info[4], 'DISABLE')
                row_disabled_tests += 1
            else:
                appendTestInfoIntoXlsDoc(worksheet_enabled_tests, row_enabled_tests, col, test_instantiate_info[0], test_instantiate_info[
                                         1], test_instantiate_info[2], test_instantiate_info[3], '', test_instantiate_info[4], 'ENABLE')
                row_enabled_tests += 1
                cnt_instantiated_p_tests += 1
            if (test_instantiate_info[2].find('nightly') != -1):
                appendTestInfoIntoXlsDoc(worksheet_nightly_tests, row_nightly_tests, col, test_instantiate_info[0], test_instantiate_info[
                                         1], test_instantiate_info[2], test_instantiate_info[3], '', test_instantiate_info[4], 'DISABLE' if disabled else 'ENABLE')
                row_nightly_tests += 1
            appendTestInfoIntoXlsDoc(worksheet_all_tests, row_all_tests, col, test_instantiate_info[0], test_instantiate_info[
                                     1], test_instantiate_info[2], test_instantiate_info[3], '', test_instantiate_info[4], 'DISABLE' if disabled else 'ENABLE')
        else:
            test_info_concat = ''
            for test_info_part in test_instantiate_info:
                test_info_concat = test_info_concat + test_info_part
            appendTestInfoIntoXlsDoc(worksheet_all_tests, row_all_tests, col, 'PARSE_ERROR',
                                     test_info_concat, '', '', '', '', 'UNDEF')
        row_all_tests += 1


if (len(all_tests_p) != cnt_instantiated_p_tests):
    print("All TEST_P size is:{}. Instantiated test count is {}".format(len(all_tests_p), cnt_instantiated_p_tests))

disabled_fmt = generateFormat('#a8a8a8')
enabled_fmt = generateFormat('#D7E4BC')
error_fmt = generateFormat('#d42a2a')

worksheet_all_tests.conditional_format(
    'E1:E10000', {'type': 'formula', 'criteria': '=FIND("DISABLE",H1,1)', 'format': disabled_fmt})
worksheet_all_tests.conditional_format(
    'F1:F10000', {'type': 'formula', 'criteria': '=FIND("DISABLE",H1,1)', 'format': disabled_fmt})

worksheet_all_tests.conditional_format(
    'E1:E10000', {'type': 'formula', 'criteria': '=FIND("ENABLE",H1,1)', 'format': enabled_fmt})
worksheet_all_tests.conditional_format(
    'F1:F10000', {'type': 'formula', 'criteria': '=FIND("ENABLE",H1,1)', 'format': enabled_fmt})

worksheet_all_tests.conditional_format(
    'E1:E10000', {'type': 'formula', 'criteria': '=FIND("PARSE_ERROR",B1,1)', 'format': error_fmt})
worksheet_all_tests.conditional_format(
    'F1:F10000', {'type': 'formula', 'criteria': '=FIND("PARSE_ERROR",B1,1)', 'format': error_fmt})

workbook.close()
