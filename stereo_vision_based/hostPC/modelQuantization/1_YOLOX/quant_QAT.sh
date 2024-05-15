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
# export CUDA_VISIBLE_DEVICES=0
GPU_NUM=4
BATCH_SIZE=32
# CFG=exps/example/custom/yolox_m_tt100k_qat.py
CFG=exps/example/yolox_voc/yolox_voc_s_QAT.py
# pre_train=./YOLOX_outputs/yolox_voc_strial_1/best_ckpt.pth
# python3 tools/train.py -f ${CFG} -d ${GPU_NUM} -b ${BATCH_SIZE} -c ${pre_train}
python3 tools/train.py -f ${CFG} -d ${GPU_NUM} -b ${BATCH_SIZE}