import pytest
import numpy as np
import subprocess
import re
import sys
import yaml
import platform
import os.path
import allure

from PaddleCVTestFramwork import RepoInit
from PaddleCVTestFramwork import RepoDataset
from PaddleCVTestFramwork import TestPaddleCVPredict


def get_model_list(filename='paddlecv_list.txt'):
    import sys
    result = []
    with open(filename) as f:
      lines = f.readlines()
      for line in lines:
         result.append(line.strip('\n'))
    return result



def setup_module():
    """
    """
    RepoInit()
    RepoDataset()


@allure.story('paddlecv_gpu_predict')
@pytest.mark.parametrize('model_name', get_model_list())
@pytest.mark.parametrize('run_mode', ['paddle', 'trt_fp32', 'trt_fp16', 'trt_int8'])
def test_paddlecv_gpu_predict(model_name, run_mode):
    allure.dynamic.title(model_name+'_GPU_code_predict_'+run_mode)
    allure.dynamic.description('GPU_code_predict')
    model =  TestPaddleCVPredict(model=model_name)
    model.test_cv_predict(run_mode,'GPU')

@allure.story('paddlecv_cpu_predict')
@pytest.mark.parametrize('model_name', get_model_list())
@pytest.mark.parametrize('run_mode', ['paddle', 'mkldnn', 'mkldnn_bf16'])
def test_paddlecv_cpu_predict(model_name, run_mode):
    allure.dynamic.title(model_name+'_CPU_code_predict_'+run_mode)
    allure.dynamic.description('CPU_code_predict')
    model =  TestPaddleCVPredict(model=model_name)
    model.test_cv_predict(run_mode,'CPU')
