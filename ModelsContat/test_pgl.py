import pytest
import numpy as np
import subprocess
import re
import allure

from RocmTestFramework import RepoInitTag
from RocmTestFramework import RepoRemove

def allure_step(cmd, output):
    with allure.step("运行指令：{}".format(cmd)):
           pass

def custom_instruction(cmd, model='output'):
         repo_result=subprocess.getstatusoutput(cmd)
         exit_code=repo_result[0]
         output=repo_result[1]
         allure_step(cmd, output)
         allure.attach(output, model+'.log', allure.attachment_type.TEXT)
         assert exit_code == 0, " %s  failed!   log information:%s" % (model, output)

def set_case_list(dir_path=''):
    cmd='cd %s; find . -maxdepth 1 -name "test_*.py" | sort ' % (dir_path)
    print(cmd)
    repo_result=subprocess.getstatusoutput(cmd)
    exit_code=repo_result[0]
    output=repo_result[1]
    result=output
    print(result[0])
    return result

def get_case_list(filename='models_list.yaml'):
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
    RepoInitTag(repo='PGL', tag='2.2.4')


def teardown_module():
    """
    """
#    RepoRemove(repo='PGL')


@allure.story('API')
@pytest.mark.parametrize('case_name', get_case_list('pgl_api.txt'))
def test_pgl_api(case_name):
    cmd='cd PGL/tests; python -m pytest -sv %s' % (case_name)
    custom_instruction(cmd, case_name)

@allure.story('models')
@pytest.mark.parametrize('cmd', get_case_list('pgl_models.txt'))
def test_pgl_models(cmd):
    custom_instruction(cmd)
