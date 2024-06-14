#!/usr/bin/env python3
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Test for inference of compiled models.
Refer to conftest.py on the test usage.
"""

import os
import subprocess


def test_infer_compiled(param_ir):
    """ Test compiled network can be inferenced
    """
    blob = os.path.splitext(os.path.join(param_ir.models_dir, param_ir.model))[0] + ".blob"
    subprocess.check_call(
        [param_ir.benchmark_app,
         '-d=VPU',
         f'-m={blob}',
         '-niter=1', '-nireq=1'])
