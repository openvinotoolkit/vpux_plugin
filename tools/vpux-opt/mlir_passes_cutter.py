#!/usr/bin/env python

import argparse
import json
import os
import re
import sys

from collections import OrderedDict

parser = argparse.ArgumentParser(
    prog="Cut out a textual pipeline by reducing passes placed before some pivot pass",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

def get_if_exist(keys_list, tree):
    if len(keys_list) == 0:
        return tree

    keys_list_copy = keys_list.copy()
    root = keys_list_copy[0]
    if root not in tree.keys():
        tree[root] = OrderedDict()
    keys_list_copy.pop(0)
    return get_if_exist(keys_list_copy, tree[root])

OBJECT_ID_GEN_INCR = 0

def canonize_name(name):
    index = name.find('[')
    return name[:index] if index != -1 else name

def uniquify_name(name):
    global OBJECT_ID_GEN_INCR
    name = name + "[" + str(OBJECT_ID_GEN_INCR) + "]"
    OBJECT_ID_GEN_INCR += 1
    return name

def is_object(name):
    return True if "module" in name or "func.func" in name else False

'''
State Machine conditions
'''
def make_trans(cond, func, new_state):
    def state_change_wrapper(*args, **kwargs):
        ctx = func(*args, **kwargs)
        ctx.state = new_state
        return ctx
    return (cond, state_change_wrapper)

def is_sym(sym):
    return lambda x: (x == sym)

def is_name():
    return lambda x: (x.isalpha() or x == "." or x == "-")

def is_abc():
    return lambda x: (x.isalpha() or x.isdigit() or x == "." or x == "-")


'''
State Machine transitions
'''
def on_create_object(pipeline, context):
    context.nested_objects.append(context.accumulated_string)
    pipeline = context.create_pipeline_node_from_tree_path(pipeline)

    context.accumulated_string = ""
    return context

def on_finalize_object(pipeline, context):
    context = on_finalize_entity(pipeline, context)
    context.nested_objects.pop()
    context.accumulated_string = ""
    return context

def on_accumulate_entity(pipeline, context):
    context.accumulated_string += context.textual_pipeline[context.begin_index]
    return context

def on_finalize_entity(pipeline, context):
    if len(context.accumulated_string) == 0:
        return context

    context.nested_objects.append(context.accumulated_string)
    pipeline = context.create_pipeline_node_from_tree_path(pipeline)
    context.nested_objects.pop()

    context.accumulated_string = ""
    return context

def onCreatePass(pipeline, context):
    context.nested_objects.append(context.accumulated_string)
    pipeline = context.create_pipeline_node_from_tree_path(pipeline)

    context.accumulated_string = ""
    return context

def on_finalize_pass(pipeline, context):
    context = on_finalize_option_value(pipeline, context)
    context.nested_objects.pop()
    context.accumulated_string = ""
    return context

def on_finalize_option_name(pipeline, context):
    node = get_if_exist(context.nested_objects, pipeline)
    node[context.accumulated_string] = ""
    context.option_name=context.accumulated_string

    context.accumulated_string = ""
    return context

def on_finalize_option_value(pipeline, context):
    node = get_if_exist(context.nested_objects, pipeline)
    node[context.option_name] = context.accumulated_string
    context.option_name=""
    context.accumulated_string = ""
    return context

def forbidden(pipeline, context):
    raise Exception(f"Forbidden transitions, ctx: {context.dump()}")

'''
Pipeline printout operations
'''
def print_object(object_name, object_value, sep):
    object_name = canonize_name(object_name)
    if len(object_value) == 0:
        print(object_name, end=sep)
        return

    print(object_name + "(", end='')
    print_pipeline(object_value)
    print(")", end=sep)

def print_options(options):
    if len(options) == 0:
        raise Exception(f"Incorrect printing state, options: {options}")

    options_std_to_print = ""
    for k, v in options.items():
        if len(k) != 0:
            options_std_to_print += f"{k}={v} "
        else:   # aling to MLIR 'canonize'
            options_std_to_print += "  "
    print(options_std_to_print[0:-1], end='')

def print_pass(pass_name, pass_value, sep):
    pass_name = canonize_name(pass_name)
    if len(pass_value) == 0:
        print(pass_name, end=sep)
        return

    print(pass_name + "{", end='')
    print_options(pass_value)
    print("}", end=sep)

def print_pipeline(pipeline):
    if len(pipeline) == 0:
        return

    for k,v in pipeline.items():
        sep = ',' if list(pipeline.keys()).index(k) != len(pipeline) - 1 else ''
        if is_object(k):
            print_object(k, v, sep)
        elif isinstance(v,dict):
            print_pass(k, v, sep)

'''
Pipeline shrink operations
'''
def process_pipeline(pipeline, from_pass_regex, offset_from_pass = 0, current_offset = None):

    list_keys_to_delete = []
    for k,v in pipeline.items():
        processed=False
        if is_object(k):
            processed, current_offset = process_pipeline(v, from_pass_regex, offset_from_pass, current_offset)

        if processed:
            break

        # if regex matches the pass name, start counting
        name = canonize_name(k)
        if current_offset is None and start_from_pass_regex.match(name):
            current_offset = 0

        if current_offset is not None:
            # counts only non-empty objects and passes
            if not is_object(k) or len(v) != 0:
                if current_offset >= int(offset_from_pass):
                    break
                current_offset += 1

        # assign node to remove, as predated to the point of interest (pass + offset)
        list_keys_to_delete.append(k)

    # if we didn't reach the point of interest (pass + value), then remove all children in the node
    # the empty node would recognized on a parent caller context
    for d in list_keys_to_delete:
        pipeline.pop(d)
    return len(pipeline) != 0, current_offset

class Context:
    def __init__(self, textual_pipeline, initial_state):
        self.textual_pipeline = textual_pipeline
        self.state = initial_state
        self.accumulated_string = ""
        self.begin_index = 0
        self.end_index = len(textual_pipeline)

        self.nested_objects = []
        self.created_objects_history = set()
        self.option_name = ""

    def create_pipeline_node_from_tree_path(self, pipeline):
        object_to_create_key = "/".join(self.nested_objects)

        # search an object the history of creations, then uniquify it if exists.
        # MLIR allows duplicate, but python dictionary doesn't.
        # Introducing list of tuples instead of dictionary complicates the logic
        if object_to_create_key in self.created_objects_history:
            self.nested_objects[-1] = uniquify_name(self.nested_objects[-1])

        # create unique node and remember it in the history
        get_if_exist(self.nested_objects, pipeline)
        self.created_objects_history.add(object_to_create_key)
        return pipeline

    def dump(self):
        print(f"state: {self.state}, index: {self.begin_index}, accumulated_string: {self.accumulated_string}, parsed tree path: {'/'.join(nested_objects)}")

args = parser.parse_args()
start_from_pass_regex = re.compile(args.start_from)

raw_pipeline=sys.stdin.read()
parsed_pipeline = OrderedDict()
state_table = { "INIT":     [
                                make_trans(is_sym("("), on_create_object,     "OBJECT"),
                                make_trans(is_sym(")"), forbidden,          None),
                                make_trans(is_sym("{"), forbidden,          None),
                                make_trans(is_sym("}"), forbidden,          None),
                                make_trans(is_sym("="), forbidden,          None),
                                make_trans(is_sym(" "), forbidden,          None),
                                make_trans(is_abc(),    on_accumulate_entity, "OBJECT")
                            ],
                "OBJECT" :  [
                                make_trans(is_sym("("), on_create_object,     "OBJECT"),
                                make_trans(is_sym(")"), on_finalize_object,   "OBJECT"),
                                make_trans(is_sym("{"), onCreatePass,       "PASS"),
                                make_trans(is_sym(","), on_finalize_entity,   "OBJECT"),
                                make_trans(is_sym("}"), forbidden,          None),
                                make_trans(is_abc(),    on_accumulate_entity, "OBJECT")
                            ],
                "PASS" :    [
                                make_trans(is_sym("}"), on_finalize_pass,         "OBJECT"),
                                make_trans(is_sym("="), on_finalize_option_name,   "PASS"),
                                make_trans(is_sym(" "), on_finalize_option_value,  "PASS"),
                                make_trans(is_abc(),    on_accumulate_entity,     "PASS")
                            ]
            }

ctx = Context(raw_pipeline, "INIT")
while ctx.begin_index < ctx.end_index:
    transitions = state_table[ctx.state]
    meet_state = False
    for t in transitions:
        cond, action = t
        if cond(raw_pipeline[ctx.begin_index]):
            ctx = action(parsed_pipeline, ctx)
            ctx.begin_index += 1
            meet_state = True
            break
    if not meet_state:
        raise Exception(f"Unexpected symbol: {raw_pipeline[ctx.begin_index]}.\nContext:\n {ctx.dump()}")

process_pipeline(parsed_pipeline, start_from_pass_regex, args.offset)
print_pipeline(parsed_pipeline)
