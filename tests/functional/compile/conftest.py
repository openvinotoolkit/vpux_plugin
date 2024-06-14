#!/usr/bin/env python3
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=line-too-long

""" Configuration for models compilation tests.

Models compilation tests run compilation for models and pass if compilation
succeeded without errors.

Pre-requisite to run the tests are model packages.

Test compilation:
python3 -m pytest  --out ./compiled --models=<path to model packages> --compiler=./bin/compile_tool test_compile.py

Test inference of compiled models:
python3 -m pytest  --models=./compiled --benchmark_app=./bin/benchmark_app test_infer_compiled.py
"""

import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest
from py.xml import html  # pylint: disable=no-name-in-module,import-error

# Keem Bay KPI models list
# find <models path> -wholename "*/FP16-INT8/*.xml"
KMB_KPI_MODELS = [
    'vd-vpu-models-kpi-ww14.tar.bz2/yolo-v2-tiny-ava-0001/tf/FP16-INT8/yolo-v2-tiny-ava-0001.xml',
    'vd-vpu-models-kpi-ww14.tar.bz2/yolo-v2-tiny-ava-sparse-30-0001/tf/FP16-INT8/yolo-v2-tiny-ava-sparse-30-0001.xml',
    'vd-vpu-models-kpi-ww14.tar.bz2/yolo-v2-tiny-ava-sparse-60-0001/tf/FP16-INT8/yolo-v2-tiny-ava-sparse-60-0001.xml',
    'vd-vpu-models-kpi-ww14.tar.bz2/resnet-50-pytorch/caffe2/FP16-INT8/resnet-50-pytorch.xml',
    'vd-vpu-models-kpi-ww14.tar.bz2/mobilenet-v2/caffe2/FP16-INT8/mobilenet-v2.xml',
    'vd-vpu-models-kpi-ww14.tar.bz2/googlenet-v1/tf/FP16-INT8/googlenet-v1.xml',
    'vd-vpu-models-kpi-ww14.tar.bz2/googlenet-v3/tf/FP16-INT8/googlenet-v3.xml',
    'vd-vpu-models-kpi-ww14.tar.bz2/squeezenet1.1/caffe2/FP16-INT8/squeezenet1.1.xml',

    {'marks': pytest.mark.xfail, 'path': 'vd-vpu-models-kpi-ww14.tar.bz2/faster-rcnn-resnet101-coco-sparse-60-0001/tf/FP16-INT8/faster-rcnn-resnet101-coco-sparse-60-0001.xml'},
    {'path': 'vd-vpu-models-kpi-ww14.tar.bz2/ssd512/caffe/FP16-INT8/ssd512.xml'},
    {'path': 'vd-vpu-models-kpi-ww14.tar.bz2/icnet-camvid-ava-0001/tf/FP16-INT8/icnet-camvid-ava-0001.xml'},
    {'path': 'vd-vpu-models-kpi-ww14.tar.bz2/yolo-v2-ava-0001/tf/FP16-INT8/yolo-v2-ava-0001.xml'},
    {'path': 'vd-vpu-models-kpi-ww14.tar.bz2/yolo-v2-ava-sparse-35-0001/tf/FP16-INT8/yolo-v2-ava-sparse-35-0001.xml'},
    {'path': 'vd-vpu-models-kpi-ww14.tar.bz2/yolo-v2-ava-sparse-70-0001/tf/FP16-INT8/yolo-v2-ava-sparse-70-0001.xml'},
]


def pytest_addoption(parser):
    """ Define extra options for pytest options
    """
    parser.addoption('--models', default='', help='Models packages')
    parser.addoption('--compiler', default='compile_tool', help='Path to model compilation tool')
    parser.addoption('--benchmark_app', default='benchmark_app', help='Path to benchmark_app tool')
    parser.addoption('--output', help='Output directory for compiled models')


def pytest_generate_tests(metafunc):
    """ Generate tests depending on command line options
    """
    out = metafunc.config.getoption("output")
    if out:
        if os.path.isdir(out):
            shutil.rmtree(out)  # cleanup output folder
    else:
        out = tempfile.mkdtemp(prefix='_blobs-', dir='.')
    params = []
    ids = []

    for model in KMB_KPI_MODELS:
        extra_args = {}
        if isinstance(model, dict):
            path = model['path']
            if 'marks' in model:
                extra_args['marks'] = model['marks']
        else:
            path = model

        params.append(pytest.param(SimpleNamespace(
            model=path,
            compiler_tool=metafunc.config.getoption("compiler"),
            benchmark_app=metafunc.config.getoption("benchmark_app"),
            models_dir=metafunc.config.getoption("models"),
            output_dir=out), **extra_args))
        ids = ids + [path]
    metafunc.parametrize("param_ir", params, ids=ids)


@pytest.mark.optionalhook
def pytest_html_results_table_header(cells):
    """ Add extra columns to HTML report
    """
    cells.insert(2, html.th('Peak memory'))


@pytest.mark.optionalhook
def pytest_html_results_table_row(report, cells):
    """ Add extra columns to HTML report
    """
    cells.insert(2, html.td(f'{getattr(report, "peak_memory", 0)}Kb'))


@pytest.mark.hookwrapper
def pytest_runtest_makereport(item):
    """ Add extra columns to HTML report
    """
    outcome = yield
    report = outcome.get_result()
    report.peak_memory = getattr(item, 'peak_memory', 0)
