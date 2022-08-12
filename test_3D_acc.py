import pytest
import numpy as np
import subprocess
import re
import sys
import yaml
import platform
import os.path
import allure
import paddle


from ModelsTestFramework import RepoInit3D
from ModelsTestFramework import RepoDataset3D
from ModelsTestFramework import Test3DModelFunction


def get_model_list():
    import sys
    result = []
    with open('models_list_3D_all.yaml') as f:
      lines = f.readlines()
      for line in lines:
         r = re.search('/(.*)/', line)
         result.append(line.strip('\n'))
    return result

def get_category(yml_name):
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    return category

def get_hardware():
    if (paddle.is_compiled_with_cuda()==True):
       hardware='_GPU'
    else:
       hardware='_CPU'
    return hardware

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
    category=get_category(yml_name)
    if (category=='smoke') or (category=='centpoint'):
        pytest.skip("not suporrted  eval when bs >1")
    if sys.platform == 'darwin':     
        pytest.skip("mac/windows skip eval")

    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    hardware=get_hardware()
    allure.dynamic.title(model_name+hardware+'_eval')
    allure.dynamic.description('模型评估')
    model = Test3DModelFunction(model=model_name, yml=yml_name)
    model.test_3D_eval(use_gpu)


@allure.story('eval')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_3D_accuracy_eval_bs1(yml_name, use_gpu):
    if sys.platform == 'darwin' and use_gpu==True:
        pytest.skip("mac skip GPU")

    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    hardware=get_hardware()
    allure.dynamic.title(model_name+hardware+'_eval_bs1')
    allure.dynamic.description('模型评估')
    model = Test3DModelFunction(model=model_name, yml=yml_name)
    model.test_3D_predict_python(use_gpu, False)

@allure.story('predict')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_3D_accuracy_predict_python_trt(yml_name, use_gpu):
    category=get_category(yml_name)
    if (category=='pointpillars') or (category=='centpoint'):
        pytest.skip("not supoorted for tensorRT predict")
    if sys.platform == 'darwin':
        pytest.skip("mac skip tensorRT predict")
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    hardware='TensorRT'
    allure.dynamic.title(model_name+hardware+'_predict')
    allure.dynamic.description('预测库python预测')
    model = Test3DModelFunction(model=model_name, yml=yml_name)
    model.test_3D_predict_python(use_gpu, True)

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
    allure.dynamic.title(model_name+hardware+'_train')
    allure.dynamic.description('训练')
    model = Test3DModelFunction(model=model_name, yml=yml_name)
    model.test_3D_train(use_gpu)

