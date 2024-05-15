# Copyright 2019 Xilinx Inc.
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

# train
export CUDA_VISIBLE_DEVICES=0
GPU_NUM=1
CFG=exps/example/custom/yolox_m_tt100k_float.py

# demo
IMG=../data/tt100k/val2017/27179.jpg

# FLOAT DEMO
export W_QUANT=0 # float model
echo "float model demo , you can test the float model  or QAT converted model"
FLOAT_CONVERTED_WEIGHTS=PATH_TO_MODEL

python tools/demo_sign.py image -f ${CFG} -c ${FLOAT_CONVERTED_WEIGHTS} --path ${IMG} --conf 0.25 --nms 0.45 --tsize 640 640 --save_result --device gpu

# QAUNT DEMO
echo "Quant model demo , you can test the QAT model or PTQ model"
export W_QUANT=1 # quantized model
QAT_WEIGHTS=PATH_TO_QAT_MODEL

python tools/demo_sign.py image -f ${CFG} -c ${QAT_WEIGHTS} --path ${IMG} --conf 0.25 --nms 0.45 --tsize 640 640 --save_result --device gpu
