#!/bin/bash

# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Mark Harvey, Xilinx Inc

conda activate vitis-ai-pytorch

# folders
export BUILD=./build_1Frame_4096
export LOG=${BUILD}/logs_1Frame_4096
mkdir -p ${LOG}

# run training
# python3 -u ur_train.py -d ${BUILD} 2>&1 | tee ${LOG}/train.log
#python3 -u SDSU_PSG_train.py -d ${BUILD} 2>&1 | tee ${LOG}/train.log


# quantize & export quantized model
python3 -u quantize.py -d ${BUILD} --quant_mode calib 2>&1 | tee ${LOG}/quant_calib.log
python3 -u quantize.py -d ${BUILD} --quant_mode test  2>&1 | tee ${LOG}/quant_test.log


# compile for target boards
source compile.sh kv260 ${BUILD} ${LOG}

# make target folders
# python3 -u target.py --target kv260  -d ${BUILD} 2>&1 | tee ${LOG}/target_kv260.log

