"""
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ctypes import *
from typing import List
import cv2
import numpy as np
import xir
import vart
import os
import math
import threading
import time
import sys
import pdb

from vaitrace_py import vai_tracepoint

"""
Calculate softmax
data: data to be calculated
size: data size
return: softamx result
"""


def CPUCalcSoftmax(data, size, scale):
    sum = 0.0
    result = [0 for i in range(size)]
    for i in range(size):
        result[i] = math.exp(data[i] * scale)
        sum += result[i]
    for i in range(size):
        result[i] /= sum
    return result


def get_script_directory():
    path = os.getcwd()
    return path


"""
Get topk results according to its probability
datain: data result of softmax
filePath: filePath in witch that records the infotmation of kinds
"""


def TopK(datain, size, filePath):

    cnt = [i for i in range(size)]
    pair = zip(datain, cnt)
    pair = sorted(pair, reverse=True)
    softmax_new, cnt_new = zip(*pair)
    fp = open(filePath, "r")
    data1 = fp.readlines()
    fp.close()
    for i in range(5):
        idx = 0
        for line in data1:
            if idx == cnt_new[i]:
                # print(f"Thread: {threading.get_ident()}\t", end=", ")
                print("Top[%d] %d %f %s" % (i, idx, softmax_new[i]*10, (line.strip)("\n")))
            idx = idx + 1


"""
pre-process for resnet50 (caffe)
"""
# _B_MEAN = 104.0
# _G_MEAN = 107.0
# _R_MEAN = 123.0
# MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]
# SCALES = [1.0, 1.0, 1.0]
_B_MEAN = 103.53
_G_MEAN = 116.28
_R_MEAN = 123.675
MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]
SCALES = [0.017429, 0.017507, 0.01712475]


def preprocess_one_image_fn(image_path, fix_scale, width=224, height=224):
    means = MEANS
    scales = SCALES
    image = cv2.imread(image_path)
    image = cv2.resize(image, (width, height))
    B, G, R = cv2.split(image)
    B = (B - means[0]) * scales[0] * fix_scale
    G = (G - means[1]) * scales[1] * fix_scale
    R = (R - means[2]) * scales[2] * fix_scale
    image = cv2.merge([B, G, R])
    image = image.astype(np.int8)
    return image


SCRIPT_DIR = get_script_directory()
# calib_image_dir = SCRIPT_DIR + "/../images/"
calib_image_dir = "./img/"
global threadnum
threadnum = 0
"""
run resnt50 with batch
runner: dpu runner
img: imagelist to be run
cnt: threadnum
"""

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

def execute_async_method(dpu, tensor_buffers_dict):
    input_tensor_buffers = [tensor_buffers_dict[t.name] for t in dpu.get_input_tensors()]
    output_tensor_buffers = [tensor_buffers_dict[t.name] for t in dpu.get_output_tensors()]
    jid = dpu.execute_async(input_tensor_buffers, output_tensor_buffers)
    return dpu.wait(jid)

def runResnet50(runner: "Runner", img, cnt):
    #print(f"\nThread: {threading.get_ident()}\trunner             = {runner}")
    resnet50_dict, inbuffer, outbuffer = init_dpu_runner(runner)
    # print(f"\nThread: {threading.get_ident()}\tresnet50_dict = {list(resnet50_dict)[1]}")
    
    resnet50_dict['ResNet__ResNet_QuantStub_quant_stub__input_1_fix'] = img.reshape(tuple(resnet50_dict['ResNet__ResNet_QuantStub_quant_stub__input_1_fix'].shape[1:]))
    # resnet50_dict[str(list(resnet50_dict)[1])] = img.reshape(tuple(resnet50_dict[str(list(resnet50_dict)[2])].shape[1:]))
    execute_async_method(runner, resnet50_dict)

    output = []
    output = resnet50_dict['ResNet__ResNet_Linear_fc__8088_fix'].copy()

    # print(f"resnet50_dict = {resnet50_dict}\ninbuffer = {inbuffer}\noutbuffer  = {outbuffer }")
    """get tensor"""
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    
    # for inputTensor in inputTensors:
    #     print(f"Thread: {threading.get_ident()}\t", end=", ")
    #     print('Input tensor  :',inputTensor.name, inputTensor.dims)
    # for outputTensor in outputTensors:
    #     print(f"Thread: {threading.get_ident()}\t", end=", ")
    #     print('Output tensor :',outputTensor.name, outputTensor.dims)

    input_ndim = tuple(inputTensors[0].dims)
    pre_output_size = int(outputTensors[0].get_data_size() / input_ndim[0])

    output_ndim = tuple(outputTensors[0].dims)
    output_fixpos = outputTensors[0].get_attr("fix_point")
    output_scale = 1 / (2**output_fixpos)
    
    softmax = CPUCalcSoftmax(output[0], pre_output_size, output_scale)
    TopK(softmax, pre_output_size, "./words.txt")
    #print("-"*100)
    #print(f"Thread: {threading.get_ident()}\tDONE")

"""
 obtain dpu subgrah
"""


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph
            is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def main(argv):
    global threadnum

    listimage = os.listdir(calib_image_dir)
    xmodels = []
    threadAll = []
    graph = []
    subgraphs = []
    threadnum = int(argv[1])
    # threadnum = int(3)
    i = 0
    global runTotall
    runTotall = len(listimage)
    # g = xir.Graph.deserialize(argv[2])
    
    # xmodels.append("/home/root/app/model/ResNet50_4096_QAT_kv260.xmodel")
    # xmodels.append("/home/root/app/model/ResNet50_1600_QAT_kv260.xmodel")
    # xmodels.append("/home/root/app/model/ResNet50_1600_QAT_kv260.xmodel")
    # xmodels.append("/home/root/app/model/ResNet50_1024_QAT_kv260.xmodel")
    # xmodels.append("/home/root/app/model/ResNet50_1024_QAT_kv260.xmodel")
    # xmodels.append("/home/root/app/model/resnet18_pt_1024.xmodel")
    # xmodels.append("/home/root/app/model/ResNet50_1024_QAT_kv260.xmodel")
    for i in range(int(threadnum)):
        xmodels.append(argv[2])
        graph.append(xir.Graph.deserialize(xmodels[i]))
        # print(f"subgraphs = {graph[i].get_root_subgraph().toposort_child_subgraph()}")
        # print(f"graph[{i}] = {graph[i]}")
        # subgraphs.append(get_child_subgraph_dpu(graph[i]))
        sg = graph[i].get_root_subgraph().toposort_child_subgraph()
        #print(f"Thread: {threading.get_ident()}\txmodel = {xmodels[i]}")
        #for j in range(len(sg)):
                #if sg[j].has_attr("device"):
                    #if sg[j].get_attr("device") == "DPU":
                        #print(f"----------------sg[{j}] = {sg[j].get_name()}")
            # print(f"sg[{j}] = {sg[j].get_name()}")
        #print("\n")
        subgraphs.append(sg[1])
        # subgraphs.append(graph[i].get_root_subgraph().toposort_child_subgraph())
        # print(f"subgraphs[{i}] = {subgraphs[i].get_name()}")

    # assert len(subgraphs) == 1  # only one DPU kernel
    all_dpu_runners = []
    
    for i in range(int(threadnum)):
        # pdb.set_trace()
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[i], "run"))

    #for i in range(int(threadnum)):
        #print(f"Thread: {threading.get_ident()}\tall_dpu_runners[{i}] = {all_dpu_runners[i]}")

    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos
    """image list to be run """
    img = []
    for i in range(runTotall):
        path = os.path.join(calib_image_dir, listimage[i])
        img.append(preprocess_one_image_fn(path, input_scale))
    """
      The cnt variable is used to control the number of times a single-thread DPU runs.
      Users can modify the value according to actual needs. It is not recommended to use
      too small number when there are few input images, for example:
      1. If users can only provide very few images, e.g. only 1 image, they should set
         a relatively large number such as 360 to measure the average performance;
      2. If users provide a huge dataset, e.g. 50000 images in the directory, they can
         use the variable to control the test time, and no need to run the whole dataset.
    """
    cnt = 360
    """run with batch """
    time_start = time.time()
    for i in range(int(threadnum)):
        t1 = threading.Thread(target=runResnet50, args=(all_dpu_runners[i], img[i], cnt))
        threadAll.append(t1)
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()

    del all_dpu_runners

    time_end = time.time()
    timetotal = time_end - time_start
    # total_frames = cnt * int(threadnum)
    total_frames = 1
    fps = float(total_frames / timetotal)
    # print("DONE HERE")
    print("FPS=%.2f, total frames = %.2f , time=%.6f seconds" % (fps, total_frames, timetotal))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage : python3 resnet50.py <thread_number> <resnet50_xmodel_file>")
    else:
        main(sys.argv)
    # main(sys.argv)
