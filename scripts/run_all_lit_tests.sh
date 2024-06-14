#!/bin/bash
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

if [ $# -gt 0 ]
then
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]
    then
        USAGE="
Script that runs all lit-tests on VPUX30XX and VPUX37XX platforms.

Usage: ./run_all_lit_tests.sh [PATH_TESTS] [PATH_LIT_TOOL]

PATH_TESTS    - optional, the path to the root directory containing all lit-tests to be run; default value '[PATH_SCRIPT]/lit-tests/NPU'
PATH_LIT_TOOL - optional, path to the lit tool lit.py; default value '[PATH_SCRIPT]/lit-tool/lit.py'
"
        echo "$USAGE"
        exit 0
    fi
fi

PATH_SCRIPT="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 || exit ; pwd -P )"
PATH_TESTS="$PATH_SCRIPT/NPU"
PATH_LIT_TOOL="$PATH_SCRIPT/lit-tool/lit.py"

if [ $# -gt 0 ]
then
    PATH_TESTS="$1"
    if [ $# -gt 1 ]
    then
        PATH_LIT_TOOL="$2"
    fi
fi

echo "PATH_TESTS=$PATH_TESTS"
echo "PATH_LIT_TOOL=$PATH_LIT_TOOL"

CMD_VPUX30XX_TESTS="python3 $PATH_LIT_TOOL --param arch=VPUX30XX $PATH_TESTS/NPU"
CMD_VPUX37XX_TESTS="python3 $PATH_LIT_TOOL --param arch=VPUX37XX $PATH_TESTS/NPU"
CMD_VPUX40XX_TESTS="python3 $PATH_LIT_TOOL --param arch=VPUX40XX $PATH_TESTS/NPU"

EXIT_CODE=0

echo ""
echo "Executing tests on VPUX30XX platform: $CMD_VPUX30XX_TESTS"
eval "$CMD_VPUX30XX_TESTS"; EXIT_CODE=$(($EXIT_CODE + $?))
echo ""
echo "Executing tests on VPUX37XX platform: $CMD_VPUX37XX_TESTS"
eval "$CMD_VPUX37XX_TESTS"; EXIT_CODE=$(($EXIT_CODE + $?))
echo ""
echo "Executing tests on VPUX40XX platform: $CMD_VPUX40XX_TESTS"
eval "$CMD_VPUX40XX_TESTS"; EXIT_CODE=$(($EXIT_CODE + $?))
echo ""

if [ $EXIT_CODE -ne 0 ]
then
    echo "FAILURES identified"
    exit 1
else
    echo "All tests PASSED"
fi
