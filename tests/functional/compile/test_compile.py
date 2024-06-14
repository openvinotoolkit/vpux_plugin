#!/usr/bin/env python3
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test for IR compilation.
Refer to conftest.py on the test usage.
"""

import os
import subprocess
import tempfile
import logging


def run(args, log=None, verbose=True):
    """ Run command
    """
    if log is None:
        log = logging.getLogger()
    log_out = log.info if verbose else log.debug

    log_out(  # pylint: disable=logging-fstring-interpolation
        f'========== cmd: {" ".join(args)}')

    proc = subprocess.Popen(args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            encoding='utf-8',
                            universal_newlines=True)
    output = []
    for line in iter(proc.stdout.readline, ''):
        log_out(line.strip('\n'))
        output.append(line)
        if line or proc.poll() is None:
            continue
        break
    outs = proc.communicate()[0]

    if outs:
        log_out(outs.strip('\n'))
        output.append(outs)
    log_out('========== Completed. Exit code: %d', proc.returncode)
    return proc.returncode, ''.join(output)


def measured_run(args, **kwargs):
    """ Run command and measure peak memry used
    """
    if os.path.exists('/usr/bin/time'):
        with tempfile.NamedTemporaryFile() as time_file:
            returncode, output = run([
                '/usr/bin/time',
                '--format=%M',
                f'--output={time_file.name}',
                '--quiet'] + args, **kwargs)
            peak_memory = open(time_file.name).read().strip()
        return returncode, output, peak_memory
    returncode, output = run(args, **kwargs)
    return returncode, output, None


def test_compile(request, param_ir):
    """ Test network can be compiled
    """
    out = os.path.splitext(os.path.join(param_ir.output_dir, param_ir.model))[0] + ".blob"
    os.makedirs(os.path.split(out)[0], exist_ok=True)
    config = open("vpu3700.config", "w")
    config.write("NPU_PLATFORM 3700\n")
    config.close()
    returncode, output, peak_memory = measured_run([
        param_ir.compiler_tool,
        '-d=VPU',
        f'-m={os.path.join(param_ir.models_dir, param_ir.model)}',
        f'-o={out}',
        f'-c=vpu3700.config'
    ])
    print(output)
    request.node.peak_memory = peak_memory
    print(f'Peak memory consumption {peak_memory}Kb')
    assert returncode == 0, f'Command exited with non-zero status {returncode}'
