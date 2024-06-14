#!/usr/bin/env python

import argparse
import os
import re
import sys

parser = argparse.ArgumentParser(
    prog="Extract passes list from generated MLIR",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--IE_NPU_IR_PRINTING_ORDER",
    choices=["after", "before", "before_after"],
    type=str.lower,
    help="Extract passes using order",
    required=False,
    default=os.environ.get("IE_NPU_IR_PRINTING_ORDER"),
)
parser.add_argument(
    "-s",
    "--start-from",
    help="Starting from which pass determined by REGEX the tool prints the list of applied passes.",
    default='".*"',
)
parser.add_argument(
    "-o", "--offset", help='Applied to "--start-from"', type=int, default=0
)

args = parser.parse_args()
if not args.IE_NPU_IR_PRINTING_ORDER:
    parser.print_help()
    exit(
        "\n\nERROR:\nNeither commanline argument --IE_NPU_IR_PRINTING_ORDER nor ENV variable IE_NPU_IR_PRINTING_ORDER specified! Please, set it using the either method."
    )
pass_print_order = args.IE_NPU_IR_PRINTING_ORDER.lower()
pass_print_order = pass_print_order.split("_")[0]  # transform before_after into before


start_from_pass_regex = re.compile(args.start_from)
current_pass_offset = None
pass_marker = ("// -----// IR Dump ", len("// -----// IR Dump "))
for line in sys.stdin:
    if not line.startswith(pass_marker[0]):
        continue

    line = line.rstrip()
    if not line.endswith("//----- //"):
        raise Exception(
            f"Misformed pass definition in the row: {line}. Expected row suffix: //----- //"
        )

    pass_attributes_list = line[pass_marker[1] :].split()

    # filter out by pass printing order
    pass_attributes_list[0] = pass_attributes_list[0].lower()
    if pass_attributes_list[0] != pass_print_order:
        continue

    # extract canonized pass name
    pass_attributes_list[2] = pass_attributes_list[2].strip("()")

    # find the start-from pass
    if current_pass_offset is None and start_from_pass_regex.match(
        pass_attributes_list[2]
    ):
        current_pass_offset = 0

    # print pass determined by offset number starting from successful match
    if current_pass_offset is not None:
        if current_pass_offset >= int(args.offset):
            try:
                print("--{}".format(pass_attributes_list[2]), end=" ")
            except IOError:
                # work out Broken pipe due to pipelining composition
                try:
                    sys.stdout.close()
                except IOError:
                    exit(0)
                try:
                    sys.stderr.close()
                except IOError:
                    exit(0)

        current_pass_offset += 1
