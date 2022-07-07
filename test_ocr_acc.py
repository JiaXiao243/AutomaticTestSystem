mport pytest
import numpy as np
import subprocess
import re
import sys
import yaml
import platform
import os.path

from ModelsTestFramework import RepoInit
from ModelsTestFramework import RepoDataset
from ModelsTestFramework import TestOcrModelFunction


def get_model_list():
    import sys
    result = []
    with open('models_list_test.yaml') as f:
      lines = f.readlines()
      for line in lines:
         result.append(line.strip('\n'))
        # print(line)
    return result


def setup_module():
    """
    """
    # RepoInit(repo='PaddleOCR')
    # RepoDataset(cmd='''cd PaddleOCR; ln -s /ssd1/panyan/data/train_data train_data;''') 
    RepoDataset()



@pytest.mark.parametrize('yml_name', get_model_list())
def test_rec_accuracy_get_pretrained_model(yml_name):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    model = TestOcrModelFunction(model=model_name, yml=yml_name)
    model.test_ocr_get_pretrained_model()

# @pytest.mark.skip
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_rec_accuracy_eval(yml_name, use_gpu):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    model = TestOcrModelFunction(model=model_name, yml=yml_name)
    model.test_ocr_eval(use_gpu)

@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True,False])
def test_rec_accuracy_infer(yml_name, use_gpu):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    model = TestOcrModelFunction(model=model_name, yml=yml_name)
    model.test_ocr_rec_infer(use_gpu)

@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True,False])
def test_rec_accuracy_export_model(yml_name, use_gpu):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    model = TestOcrModelFunction(model=model_name, yml=yml_name)
    model.test_ocr_export_model(use_gpu)

@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("enable_mkldnn", [True,False])
def test_rec_accuracy_predict_mkl(yml_name, use_gpu, enable_mkldnn):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    model = TestOcrModelFunction(model=model_name, yml=yml_name)
    model.test_ocr_rec_predict(False, 0, enable_mkldnn)

@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_tensorrt", [True,False])
def test_rec_accuracy_predict_trt(yml_name, use_gpu, use_tensorrt):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    model = TestOcrModelFunction(model=model_name, yml=yml_name)
    model.test_ocr_rec_predict(True, use_tensorrt, 0)

# @pytest.mark.skip
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_rec_funtion_train(yml_name, use_gpu):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    model = TestOcrModelFunction(model=model_name, yml=yml_name)
    model.test_ocr_train(use_gpu)

@pytest.mark.mac
@pytest.mark.parametrize('yml_name', get_model_list())
def test_rec_funtion_mac(yml_name):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    model = TestOcrModelFunction(model=model_name, yml=yml_name)
#    model.test_ocr_train(False)
#    model.test_ocr_get_pretrained_model()
    model.test_ocr_eval(False)
#    model.test_ocr_rec_infer(False)
#    model.test_ocr_export_model(False)
#    model.test_ocr_rec_predict(False, False, False)
