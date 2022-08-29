import pytest
import numpy as np
import subprocess
import re

from RocmTestFramework import custom_instruction
from RocmTestFramework import RepoInit
from RocmTestFramework import RepoRemove

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
    RepoRemove(repo='PGL')


@allure.story('API')
@pytest.mark.parametrize('case_name', get_case_list('PGL/tests'))
def test_science_api(case_name):
    cmd='cd PGL/tests; python -m pytest -sv %s' % (case_name)
    custom_instruction(cmd)

