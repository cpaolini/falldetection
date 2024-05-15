from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import vitis_ai_library
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
_divider = '-------------------------------'

global dpu_runner, ratio
# model = "/home/root/sourceFiles/FD_kv260.xmodel"
model = "/home/root/models/B4096/FD_1Frame_4096_kv260.xmodel"
g = xir.Graph.deserialize(model)
dpu_runner = vitis_ai_library.GraphRunner.create_graph_runner(g)
# g = xir.Graph.deserialize(model)
# subgraphs = g.get_root_subgraph().toposort_child_subgraph()
# dpu_subgraph1 = subgraphs[1]
# dpu_runner = vart.Runner.create_runner(dpu_subgraph1, "run")

def execute_async_method(tensor_buffers_dict):
    global dpu_runner
    input_tensor_buffers = [tensor_buffers_dict[t.name] for t in dpu_runner.get_input_tensors()]
    output_tensor_buffers = [tensor_buffers_dict[t.name] for t in dpu_runner.get_output_tensors()]
    jid = dpu_runner.execute_async(input_tensor_buffers, output_tensor_buffers)
    return dpu_runner.wait(jid)

def init_dpu_runner():
    global dpu_runner
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

def runInference(joints):
    global fall_detector_dict, inbuffer, outbuffer
    #  Set up encoder DPU runner buffers & I/O mapping dictionary
    fall_detector_dict, inbuffer, outbuffer = init_dpu_runner()
    '''
    initialise input and execute DPU runner
    '''
    fall_detector_dict['FallModel__input_0_fix'] = joints.reshape(tuple(fall_detector_dict['FallModel__input_0_fix'].shape[1:]))

    execute_async_method(fall_detector_dict)
    
    out_q = fall_detector_dict['FallModel__FallModel_ret_fix_'].copy()
    output = torch.as_tensor(out_q)

    #print("Done with the DPU runner")
    return output

def runFallPredictor(pose_joints):
    # global ratio, dpu_runner 

    '''run inference '''
    outputs = runInference(pose_joints)

    # ''' post-processing '''
    prediction = torch.argmax(outputs, dim=1)
    # if prediction.item():
    #     return True
    # else:
    #     return False
    return prediction.item()

if __name__ == '__main__':
    # rand_in = torch.randn([1, 1, 15, 3])
    # rand_in = rand_in.numpy()
    # output = runFallDetector(rand_in)
    # if output:
    #     print("***Fall detected!***")
    # else:
    #     print("ALL OKAY")
    print("ALL OKAY")
