import pytest
import numpy as np
import subprocess
import re
import sys
import yaml
import platform
import os.path
import allure

from ModelsTestFramework import RepoInit
from ModelsTestFramework import RepoDataset
from ModelsTestFramework import TestOcrModelFunction


def get_model_list(filename='ocr_model_list.yaml'):
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
    RepoInit(repo='PaddleOCR')
    RepoDataset()


@allure.story('train')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_ocr_funtion_train(yml_name, use_gpu):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    if use_gpu==True:
       hardware='_GPU'
    else:
       hardware='_CPU'
    allure.dynamic.title(model_name+hardware+'_train')
    allure.dynamic.description('训练')
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_train(use_gpu)

