#!/bin/sh

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

if [ $1 = zcu102 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json
      TARGET=zcu102
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZCU102.."
      echo "-----------------------------------------"
elif [ $1 = kv260 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
      TARGET=kv260
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR KV260."
      echo "-----------------------------------------"
elif [ $1 = zcu104 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
      TARGET=zcu104
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZCU104.."
      echo "-----------------------------------------"
elif [ $1 = vck190 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json
      TARGET=vck190
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR VCK190.."
      echo "-----------------------------------------"
elif [ $1 = u50 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U50/arch.json
      TARGET=u50
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ALVEO U50.."
      echo "-----------------------------------------"
else
      echo  "Target not found. Valid choices are: zcu102, kv260, zcu104, vck190, u50 ..exiting"
      exit 1
fi

# BUILD=$2
LOG=$2

#compile() {
#  vai_c_xir \
#  --xmodel      ./quantized/YOLOX_0_int.xmodel \
#  --arch        $ARCH \
#  --net_name    YOLOX_${TARGET} \
#  --output_dir  ./quantized/compiled_model
#}
#compile 2>&1 | tee ${LOG}/compile_$TARGET.log

#compile() {
#  vai_c_xir \
#  --xmodel      ./converted_qat_results/YOLOX_0_int.xmodel \
#  --arch        $ARCH \
#  --net_name    YOLOX_1024_${TARGET} \
#  --output_dir  ./quantized/compiled_model_1024
#}
#compile 2>&1 | tee ${LOG}/compile_1024_$TARGET.log


# compile() {
#   vai_c_xir \
#   --xmodel      ./quantized/YOLOX_0_int.xmodel \
#   --arch        $ARCH \
#   --net_name    YOLOX_4096_${TARGET} \
#   --output_dir  ./quantized/compiled_model_4096
# }

# compile 2>&1 | tee ${LOG}/compile_4096_$TARGET.log

# compile() {
#   vai_c_xir \
#   --xmodel      ./converted_qat_results/YOLOX_0_int.xmodel \
#   --arch        $ARCH \
#   --net_name    YOLOX_4096_QAT_${TARGET} \
#   --output_dir  ./quantized/compiled_model_4096_QAT
# }

# compile 2>&1 | tee ${LOG}/compile_1600_QAT_$TARGET.log
# compile() {
#   vai_c_xir \
#   --xmodel      ./converted_qat_results/YOLOX_0_int.xmodel \
#   --arch        $ARCH \
#   --net_name    YOLOX_1600_QAT_${TARGET} \
#   --output_dir  ./quantized/compiled_model_1600_QAT
# }

# compile 2>&1 | tee ${LOG}/compile_1600_QAT_$TARGET.log
compile() {
  vai_c_xir \
  --xmodel      ./converted_qat_results/YOLOX_0_int.xmodel \
  --arch        $ARCH \
  --net_name    YOLOX_4096_QAT_${TARGET} \
  --output_dir  ./quantized/compiled_model_4096_QAT_vai3
}

compile 2>&1 | tee ${LOG}/compile_4096_QAT_vai3_$TARGET.log


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"



