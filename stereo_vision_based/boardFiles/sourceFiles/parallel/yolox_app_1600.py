'''
Copyright 2020 Xilinx Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from boxes import postprocessing
from visualize import vis
import functools
import operator
_divider = '-------------------------------'
decode_in_inference = True 
stridess=[8, 16, 32]
confidence_threshold = 0.7
nms_threshold = 0.45
global ratio
global dpu_runner
# model = "/home/root/sourceFiles/YOLOX_kv260.xmodel"
# model = "/home/root/models/YOLOX_4096_kv260.xmodel"
model = "/home/root/models/B1600/YOLOX_1600_QAT_vai3_kv260.xmodel"
g = xir.Graph.deserialize(model)
subgraphs = g.get_root_subgraph().toposort_child_subgraph()
dpu_subgraph1 = subgraphs[1]
dpu_runner = vart.Runner.create_runner(dpu_subgraph1, "run")

def InitQuantModel(model):
    global dpu_runner
    g = xir.Graph.deserialize(model)

    subgraphs = g.get_root_subgraph().toposort_child_subgraph()
        
    dpu_subgraph1 = subgraphs[1]
    
    dpu_runner = vart.Runner.create_runner(dpu_subgraph1, "run")
    
    return dpu_runner

def preprocess_fn(image):
    global ratio, original_image
    '''
    Image pre-processing.
    Opens image , crop image to bbox, resize to 288x288, normalize
    and then scales by input quantization scaling factor
    input arg: path of image file
    return: numpy array
    '''
    original_image = image
    if len(original_image.shape) == 3:
        padded_image = np.ones((640,640,3), dtype=np.uint8)*114
         
    image = original_image.copy()
    ratio = min(640 / original_image.shape[0], 640 / original_image.shape[1])
    try:
        resized_img = cv2.resize(
            image.copy(),
            (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
    except Exception as e:
        print(f"Yolox preprocessing: {e}")
    padded_image[: int(image.shape[0] * ratio), : int(image.shape[1] * ratio)] = resized_img
    image = np.ascontiguousarray(padded_image, dtype=np.float32)
    #backup input image
    # cv2.imwrite("/home/root/sourceFiles/photo_640.jpeg", image)
    return image

def execute_async_method(dpu, tensor_buffers_dict):
    input_tensor_buffers = [tensor_buffers_dict[t.name] for t in dpu.get_input_tensors()]
    output_tensor_buffers = [tensor_buffers_dict[t.name] for t in dpu.get_output_tensors()]
    jid = dpu.execute_async(input_tensor_buffers, output_tensor_buffers)
    return dpu.wait(jid)

def init_dpu_runner(dpu_runner):
    '''
    Setup DPU runner in/out buffers and dictionary
    '''

    io_dict = {}
    inbuffer = []
    outbuffer = []

    # create input buffer, one member for each DPU runner input
    # add inputs to dictionary
    dpu_inputs = dpu_runner.get_input_tensors()
    i=0
    for dpu_input in dpu_inputs:
        #print('DPU runner input:',dpu_input.name,' Shape:',dpu_input.dims)
        inbuffer.append(np.empty(dpu_input.dims, dtype=np.float32, order="C"))
        io_dict[dpu_input.name] = inbuffer[i]
        i += 1

    # create output buffer, one member for each DPU runner output
    # add outputs to dictionary
    dpu_outputs = dpu_runner.get_output_tensors()
    i=0
    for dpu_output in dpu_outputs:
        #print('DPU runner output:',dpu_output.name,' Shape:',dpu_output.dims)
        outbuffer.append(np.empty(dpu_output.dims, dtype=np.float32, order="C"))
        io_dict[dpu_output.name] = outbuffer[i]
        i += 1

    return io_dict, inbuffer, outbuffer

def runInference(dpu_runner, img):    
    # global out_q1, out_q2, out_q3
    '''
    Thread worker function
    '''
    #  Set up encoder DPU runner buffers & I/O mapping dictionary
    global yolox_dict, inbuffer, outbuffer, outputs

    yolox_dict, inbuffer, outbuffer = init_dpu_runner(dpu_runner)
    
    # print(f"INFO: yolox init_dpu_runner DONE")

    '''
    initialise input and execute DPU runner
    '''
    yolox_dict['YOLOX__YOLOX_QuantStub_quant_in__input_1_fix'] = img.reshape(tuple(yolox_dict['YOLOX__YOLOX_QuantStub_quant_in__input_1_fix'].shape[1:]))

    execute_async_method(dpu_runner, yolox_dict)
    
    # print(f"INFO: yolox execute_async_method DONE")
    # write results to global predictions buffer
    out_q1 = yolox_dict['YOLOX__YOLOX_YOLOXHead_head__Cat_cat_list__ModuleList_0__inputs_3_fix'].copy()
    out_q2 = yolox_dict['YOLOX__YOLOX_YOLOXHead_head__Cat_cat_list__ModuleList_1__inputs_5_fix'].copy()
    out_q3 = yolox_dict['YOLOX__YOLOX_YOLOXHead_head__Cat_cat_list__ModuleList_2__inputs_fix'].copy()
    out_q1 = torch.as_tensor(out_q1.transpose(0,3,1,2))
    out_q2 = torch.as_tensor(out_q2.transpose(0,3,1,2))
    out_q3 = torch.as_tensor(out_q3.transpose(0,3,1,2))
    output = [out_q1,out_q2,out_q3]

    output = postprocess(output)
    
    output = postprocessing(output, num_classes = 1, conf_thre=confidence_threshold, nms_thre=nms_threshold, class_agnostic=False)
    # outputs.append(output.copy())#append the output later for individual images
    #print("Done with the DPU runner")
    return output

def postprocess(outputs):
    hw = [x.shape[-2:] for x in outputs]
    # [batch, n_anchors_all, 85]
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
    outputs[..., 4:] = outputs[..., 4:].sigmoid()
    if decode_in_inference:
        return decode_outputs(hw, outputs, dtype=outputs.type())
    else:
        return outputs

def decode_outputs(hw, outputs, dtype):
    grids = []
    strides = []
    hsizes = [640 // stride for stride in stridess]
    wsizes = [640 // stride for stride in stridess]

    for hsize, wsize, stride in zip(hsizes, wsizes, stridess):
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride))

    grids = torch.cat(grids, dim=1).type(dtype)
    strides = torch.cat(strides, dim=1).type(dtype)

    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    return outputs        

def runYolox(image, model, count, saveVideo):
    # global out_q1, out_q2, out_q3, outputs, ratio, original_image, dpu_runner 
    global ratio, dpu_runner 
    # out_q1 = []
    # out_q2 = []
    # out_q3 = []
    # outputs = []

    ''' preprocess images '''    
    img = preprocess_fn(image)

    '''run inference '''
    outputs = runInference(dpu_runner, img)
    # del dpu_runner
    
    ''' post-processing '''
    BBox = 0
    NumofBox = 0
    Bbox = 0
    BBox = outputs[0]
    if BBox != None:
        Bbox = BBox[:,0:4]
        # Bbox /= ratio
        Bbox = Bbox.tolist()
        # print(f"Bbox = {Bbox[0][0]}\t{Bbox[0][1]}\t{Bbox[0][2]}\t{Bbox[0][3]}")
        Bbox = functools.reduce(operator.iconcat, Bbox, [])            
        NumofBox = len(Bbox)/4
        result_image = visual(image, outputs[0], cls_conf = confidence_threshold)
        # cv2.imwrite("/home/root/sourceFiles/yoloxOutput_1600_QAT_"+str(count)+".jpeg", result_image)
        if saveVideo:
            result_image = visual(image, outputs[0], cls_conf = confidence_threshold)
            # cv2.imwrite("/home/root/sourceFiles/output_"+str(count)+".jpeg", result_image)
            # yoloxOutputVideo.write(result_image)
    else:
        if saveVideo:
            result_image = image

    # print(f"NumofBox = {NumofBox}")
    # print(f"Bbox = {Bbox}")
    # print (_divider)
    # print (_divider)
    

    if saveVideo:
        return NumofBox, Bbox, result_image
    else:
        return NumofBox, Bbox

def visual(img, output, cls_conf=0.35):
    global ratio
    if output is None:
        return 0

    bboxes = output[:, 0:4]

    # preprocessing: resize
    # bboxes /= ratio

    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    vis_res = vis(img, bboxes, scores, cls, cls_conf, "person")
    return vis_res

# only used if script is run as 'main' from command line
def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()  
    ap.add_argument('-d', '--image_dir', type=str, default='/home/root/2. YOLOX/images', help='Path to folder of images. Default is images')  
    ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads. Default is 1')
    ap.add_argument('-m', '--model',     type=str, default='/home/root/2. YOLOX/YOLOX_kv260.xmodel', help='Path of xmodel. Default is YOLOX_kv260.xmodel')
    args = ap.parse_args()  

    print ('Command line options:')
    print (' --image_dir : ', args.image_dir)
    print (' --threads   : ', args.threads)
    print (' --model     : ', args.model)

    # runYolox(args.image_dir,args.threads,args.model)

if __name__ == '__main__':
    main()

