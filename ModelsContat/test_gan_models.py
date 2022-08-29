#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
  * @file 
  * @author jiaxiao01
  * @date 2021/9/3 3:46 PM
  * @brief clas model inference test case
  *
  **************************************************************************/
"""

import pytest
import numpy as np
import subprocess
import re
import os
import allure

from RocmTestFramework import TestGanModel
from RocmTestFramework import RepoInit
from RocmTestFramework import RepoRemove
from RocmTestFramework import RepoDataset
from RocmTestFramework import CustomInstruction
from RocmTestFramework import clean_process
from RocmTestFramework import get_model_list


def setup_module():
    """
    """
    RepoInit(repo='PaddleGAN')

    RepoDataset(cmd='''cd PaddleGAN;
                       rm -rf data;
                       ln -s /ssd2/ce_data/PaddleGAN data 
                       cd ..;
                       yum install epel-release -y;
                       yum update -y;
                       rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro;
                       rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm;
                       yum install ffmpeg ffmpeg-devel -y''') 

def teardown_module():
    """
    """
    RepoRemove(repo='PaddleGAN')


@allure.story('train')
@pytest.mark.parametrize('yml_name', get_model_list('gan_model_list.yaml'))
def test_gan_funtion_train(yml_name):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    hardware='_GPU'
    allure.dynamic.title(model_name+hardware+'_train')
    allure.dynamic.description('训练')
    model = TestGanModel(model=model_name, yaml=yml_name)
    model.test_gan_train()

