
import pytest
from pytest_assume.plugin import assume
from pytest import approx
import numpy as np
import subprocess
import re
import ast
import logging
import os
import yaml
import os.path
import platform
import allure
import paddle


def exit_check_fucntion(exit_code, output, output_vis, output_json, input_image):
    print(output)
    assert exit_code == 0, "model predict failed!   log information:%s" % (output)
    assert 'Error' not in output, "model predict failed!   log information:%s" % (output)
    logging.info("train model sucessfuly!" )
    allure.attach(output, 'output.log', allure.attachment_type.TEXT)
    allure_attach(output_vis)
    allure_attach(output_json)
    allure_attach(input_image)

def platformAdapter(cmd):
    if (platform.system() == "Windows"):
            cmd=cmd.replace(';','&')
            cmd=cmd.replace('sed','%sed%')
            cmd=cmd.replace('rm -rf','rd /s /q')
            cmd=cmd.replace('export','set')
    if (platform.system() == "Darwin"):
            cmd=cmd.replace('sed -i','sed -i ""')
    return 

def allure_step(cmd):
    with allure.step("运行指令：{}".format(cmd)):
           pass

def allure_attach(filepath):
    
    files=os.listdir("models/paddlecv/output")
    for k in range(len(files)):
    # 提取文件夹内所有文件的后缀
      files[k]=os.path.splitext(files[k])[1]

    if len(files) >0:
        postfix=os.path.splitext(filepath)[-1]   
        if (postfix=='.png') and ('.png' in files):
            with open("models/paddlecv/" + filepath, mode='rb') as f:
                file_content = f.read()
            allure.attach(file_content, filepath, allure.attachment_type.PNG)
        elif (postfix=='.jpeg' or postfix=='.jpg') and ('.jpeg' in files or '.jpg' in files):
            with open("models/paddlecv/" + filepath, mode='rb') as f:
                file_content = f.read()
            allure.attach(file_content, filepath, allure.attachment_type.JPG)
        elif (postfix=='.json' or postfix=='.txt') and ('.json' in files or '.txt' in files):
            with open("models/paddlecv/" + filepath, mode='rb') as f:
                file_content = f.read()
            allure.attach(file_content, filepath, allure.attachment_type.TEXT)
    else:
            pass

class RepoDataset():
      def __init__(self):
         cmd='cd models/paddlecv; wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v1.0.tar && tar -xf drink_dataset_v1.0.tar; wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v2.0.tar && tar -xf drink_dataset_v2.0.tar'
         cmd=platformAdapter(cmd)
         print(cmd)
         repo_result=subprocess.getstatusoutput(cmd)
         exit_code=repo_result[0]
         output=repo_result[1]
         assert exit_code == 0, "configure failed!   log information:%s" % output


class RepoInit():
      def __init__(self):
         print("This is Repo Init!")
         cmd='''git clone -b release/2.3 https://github.com/paddlepaddle/models.git --depth 1; cd models/paddlecv; python -m pip install -r requirements.txt; cd ..;'''
         cmd=platformAdapter(cmd)
         repo_result=subprocess.getstatusoutput(cmd)
         exit_code=repo_result[0]
         output=repo_result[1]
         print(output)
         assert exit_code == 0, "git clone models failed!   log information:%s" % (output)
         logging.info("git clone models sucessfuly!" )



class TestPaddleCVPredict():
      def __init__(self, model=''): 
         self.model=model
         self.model_config=yaml.load(open('paddlecv.yml','rb'), Loader=yaml.Loader)
         self.yml=self.model_config[self.model]['yml']
         self.input=self.model_config[self.model]['input']
         self.output_json=self.model_config[self.model]['output_json']
         self.output_vis=self.model_config[self.model]['output_vis']


      def test_cv_predict(self, run_mode='paddle', device='CPU'):
          cmd='cd models/paddlecv; rm -rf output; python -u tools/predict.py --config=%s --input=%s --run_mode=%s --device=%s' % (self.yml, self.input, run_mode, device)
          cmd=platformAdapter(cmd)
          print(cmd)
          result = subprocess.getstatusoutput(cmd)
          exit_code = result[0]
          output = result[1]
          allure_step(cmd)
          exit_check_fucntion(exit_code, output, self.output_vis, self.output_json, self.input)
