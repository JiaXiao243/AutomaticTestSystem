import pytest
import numpy as np
import subprocess
import re
import sys
import yaml
import platform
import os.path
import allure

from ModelsTestFramework import RepoInit3D
from ModelsTestFramework import RepoDataset3D
from ModelsTestFramework import Test3DModelFunction


def get_model_list():
    import sys
    result = []
    with open('models_list_3D.yaml') as f:
      lines = f.readlines()
      for line in lines:
         r = re.search('/(.*)/', line)
         result.append(line.strip('\n'))
    return result

def setup_module():
    """
    """
    RepoInit3D(repo='Paddle3D')
    RepoDataset3D()


@allure.story('get_pretrained_model')
@pytest.mark.parametrize('yml_name', get_model_list())
def test_3D_accuracy_get_pretrained_model(yml_name):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    allure.dynamic.title(model_name+'_get_pretrained_model')
    allure.dynamic.description('获取预训练模型')
    model = Test3DModelFunction(model=model_name, yml=yml_name)
    model.test_3D_get_pretrained_model()

@allure.story('eval')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_3D_accuracy_eval(yml_name, use_gpu):
    if sys.platform == 'darwin' or sys.platform == 'win32':
        pytest.skip("mac/windows skip eval")
    if sys.platform == 'darwin' and use_gpu==True:
        pytest.skip("mac skip GPU")

    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    if use_gpu==True:
       hardware='_GPU'
    else:
       hardware='_CPU'
    allure.dynamic.title(model_name+hardware+'_eval')
    allure.dynamic.description('模型评估')
    model = Test3DModelFunction(model=model_name, yml=yml_name)
    model.test_3D_eval(use_gpu)


@allure.story('eval')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_3D_accuracy_eval_bs1(yml_name, use_gpu):
    if sys.platform == 'darwin' or sys.platform == 'win32':
        pytest.skip("mac/windows skip eval")
    if sys.platform == 'darwin' and use_gpu==True:
        pytest.skip("mac skip GPU")

    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    if use_gpu==True:
       hardware='_GPU'
    else:
       hardware='_CPU'
    allure.dynamic.title(model_name+hardware+'_eval_bs1')
    allure.dynamic.description('模型评估')
    model = Test3DModelFunction(model=model_name, yml=yml_name)
    model.test_3D_eval_bs1(use_gpu)


@pytest.mark.parametrize("use_gpu", [True])
def test_3D_accuracy_export_model(yml_name, use_gpu):
    if sys.platform == 'darwin' and use_gpu==True:
        pytest.skip("mac skip GPU")
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    if use_gpu==True:
       hardware='_GPU'
    else:
       hardware='_CPU'
    allure.dynamic.title(model_name+hardware+'_export_model')
    allure.dynamic.description('模型动转静')
    model = Test3DModelFunction(model=model_name, yml=yml_name)
    model.test_3D_export_model(use_gpu)

@allure.story('train')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_3D_funtion_train(yml_name, use_gpu):
    if sys.platform == 'darwin' and use_gpu==True:
        pytest.skip("mac skip GPU")
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    if use_gpu==True:
       hardware='_GPU'
    else:
       hardware='_CPU'
    model = Test3DModelFunction(model=model_name, yml=yml_name)
    model.test_3D_train(use_gpu)

