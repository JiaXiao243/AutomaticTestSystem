import pytest
import numpy as np
import subprocess
import re

from RocmTestFramework import RepoInit
from RocmTestFramework import RepoRemove


def custom_instruction(cmd, model):
         repo_result=subprocess.getstatusoutput(cmd)
         exit_code=repo_result[0]
         output=repo_result[1]
         allure_step(cmd, output)
         allure.attach(output, model+'.log', allure.attachment_type.TEXT)
         assert exit_code == 0, " %s  failed!   log information:%s" % (model, output)

def get_case_list(dir_path=''):
    cmd='cd %s; `find . -maxdepth 1 -name "test_*.py" | sort `' % (dir_path)
    repo_result=subprocess.getstatusoutput(cmd)
    exit_code=repo_result[0]
    output=repo_result[1]
    result=output
   

    return result


def setup_module():
    """
    """
    RepoInit(repo='PaddleScience')


def teardown_module():
    """
    """
    RepoRemove(repo='PaddleScience')

def setup_function():
    clean_process()


def test_science_api():
    cmd='''PaddleScience/tests/test_api/; 
    model.test_nlp_train(cmd=cmd)'''


@allure.story('API')
@pytest.mark.parametrize('case_name', get_case_list('PaddleScience/tests/test_api'))
def test_science_api(case_name):
    cmd='cd PaddleScience/tests/test_api; python -m pytest -sv %s' % (case_name)
    custom_instruction(cmd)

@allure.story('modles')
@pytest.mark.parametrize('case_name', get_case_list('PaddleScience/tests/test_models'))
def test_science_models(case_name):
    cmd='cd PaddleScience/tests/test_models; python -m pytest -sv %s' % (case_name)
    custom_instruction(cmd)
