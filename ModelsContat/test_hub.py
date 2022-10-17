import pytest
import numpy as np
import subprocess
import re
import allure
import os

from RocmTestFramework import RepoInit
from RocmTestFramework import RepoRemove

def allure_step(cmd, output):
    with allure.step("运行指令：{}".format(cmd)):
           pass

def custom_instruction(cmd, model):
         repo_result=subprocess.getstatusoutput(cmd)
         exit_code=repo_result[0]
         output=repo_result[1]
         allure_step(cmd, output)
         allure.attach(output, model+'.log', allure.attachment_type.TEXT)
         assert exit_code == 0, " %s  failed!   log information:%s" % (model, output)


def get_case_list(filename='models_list.yaml'):
    import sys
    result = []
    with open(filename) as f:
      lines = f.readlines()
      for line in lines:
         result.append(line.strip('\n'))
    return result


@allure.story('hub_seg')
@pytest.mark.parametrize('case_name', get_case_list('hub_seg_models_list.yaml'))
def test_hub_seg(case_name):
    cmd='python hub_seg_finetune.py --model_name %s' % (case_name)
    custom_instruction(cmd, case_name)

@allure.story('hub_class')
@pytest.mark.parametrize('case_name', get_case_list('hub_class_models_list.yaml'))
def test_hub_class(case_name):
    cmd='python hub_class_finetune.py --model_name %s' % (case_name)
    custom_instruction(cmd, case_name)

@allure.story('hub_nlp')
@pytest.mark.parametrize('case_name', get_case_list('hub_nlp_models_list.yaml'))
def test_hub_nlp(case_name):
    cmd='python hub_nlp_finetune.py --model_name %s' % (case_name)
    custom_instruction(cmd, case_name)
