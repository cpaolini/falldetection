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

# export CUDA_VISIBLE_DEVICES=0
GPU_NUM=1

# CFG=exps/example/custom/yolox_m_tt100k_quant.py
CFG=exps/example/yolox_voc/yolox_voc_s_Q.py

# WEIGHTS=./YOLOX_outputs/yolox_voc_s/best_ckpt.pth
WEIGHTS=./YOLOX_outputs/yolox_voc_strial_1/best_ckpt.pth

export W_QUANT=1
BATCH=8
Q_DIR='quantized'

MODE='calib'
python3 tools/quant.py -f ${CFG} -c ${WEIGHTS} -b ${BATCH} -d ${GPU_NUM} --conf 0.001 --quant_mode ${MODE} --quant_dir ${Q_DIR} # --nndct_parse_debug 1

MODE='test'
python3 tools/quant.py -f ${CFG} -c ${WEIGHTS} -b ${BATCH} -d ${GPU_NUM} --conf 0.001 --quant_mode ${MODE} --quant_dir ${Q_DIR} --is_dump $True # --nndct_parse_debug 1

export LOG=./quantized/logs
mkdir -p ${LOG}
source compile.sh kv260 ${LOG}