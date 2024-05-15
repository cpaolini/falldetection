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
import anchor as anchor
from tqdm import tqdm

_divider = '-------------------------------'
keypointsNumber = 15
cropWidth = 288
cropHeight = 288
Img_mean = 3
Img_std = 2
intrinsics = {'fx': 504.1189880371094, 'fy': 504.042724609375, 'cx': 231.7421875, 'cy': 320.62640380859375}
DATASET = 'MP3DHP'
depthFactor = 50
model = "/home/root/models/B1600/A2J_1600_kv260.xmodel"
imageOutputs = np.ones((cropHeight, cropWidth, 3), dtype="float32")
global dpu_runner

g = xir.Graph.deserialize(model)
subgraphs = g.get_root_subgraph().toposort_child_subgraph() 
dpu_subgraph2 = subgraphs[2]
dpu_runner = vart.Runner.create_runner(dpu_subgraph2, "run")


def pixel2world(x,y,z):
    if DATASET == 'ITOP':
        worldX = (x - 160.0)*z*0.0035
        worldY = (120.0 - y)*z*0.0035
    elif DATASET == 'MP3DHP':        
        worldX = (x - intrinsics['cx']) * z / intrinsics['fx']
        worldY = (y - intrinsics['cy']) * z / intrinsics['fy']
    return worldX,worldY

def world2pixel(x,y,z):
    pixelX = 160.0 + x / (0.0035 * z)
    pixelY = 120.0 - y / (0.0035 * z)
    return pixelX,pixelY

joint_id_to_name = {
    0: 'Head',
    1: 'Neck',
    2: 'RShoulder',
    3: 'LShoulder',
    4: 'RElbow',
    5: 'LElbow',
    6: 'RHand',
    7: 'LHand',
    8: 'Torso',
    9: 'RHip',
    10: 'LHip',
    11: 'RKnee',
    12: 'LKnee',
    13: 'RFoot',
    14: 'LFoot',
}

def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('torso'), keypoints.index('right_hip')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('torso'), keypoints.index('left_hip')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('torso'), keypoints.index('neck')],
        [keypoints.index('neck'), keypoints.index('right_shoulder')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('neck'), keypoints.index('left_shoulder')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('neck'), keypoints.index('head')]
    ]
    return kp_lines

jointColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
               [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
               [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [85, 255, 85]]

def get_keypoints():
    """Get the itop keypoints"""
    keypoints = [
        'head',
        'neck',
        'right_shoulder',
        'left_shoulder',
        'right_elbow',
        'left_elbow',
        'right_wrist',
        'left_wrist',
        'torso',
        'right_hip',
        'left_hip',
        'right_knee',
        'left_knee',
        'right_ankle',
        'left_ankle']
    return keypoints

def draw_humans_visibility(img, humans, limbs, jointColors, visibilities=None):
    visibilities = visibilities or None
    for i, human in enumerate(humans):
        human_vis = np.array(human)
        for k, limb in enumerate(limbs):
            if visibilities is not None and visibilities[i][limb[0]] < 0.5:
                color = [0, 0, 0]
            else:
                color = [0, 0, 255]
            center1 = human_vis[limb[0], :2].astype(int)
            img = cv2.circle(img, tuple(center1), 3, color, thickness=2, lineType=8, shift=0)

            if visibilities is not None and visibilities[i][limb[1]] < 0.5:
                color = [0, 0, 0]
            else:
                color = [0, 0, 255]
            center2 = human_vis[limb[1], :2].astype(int)
            img = cv2.line(img, tuple(center1), tuple(center2), jointColors[k], 2)
            img = cv2.circle(img, tuple(center2), 3, color, thickness=2, lineType=8, shift=0)

    return img

def convertOriginalScale(source, bndbox):
    Test1_ = source.copy()
    Test1_[:, :, 0] = source[:, :, 0]
    Test1_[:, :, 1] = source[:, :, 1]
    Test1 = Test1_  # [x, y, z]

    for i in range(len(Test1_)):
        Test1[i, :, 0] = Test1_[i, :, 0] * (bndbox[2] - bndbox[0]) / cropWidth + bndbox[0]  # x
        Test1[i, :, 1] = Test1_[i, :, 1] * (bndbox[3] - bndbox[1]) / cropHeight + bndbox[1]  # y
        Test1[i, :, 2] = Test1_[i, :, 2] #/ depthFactor
    return Test1


def pattingTest(image, source, bndbox, count):
    # global poseOutputVideo
    Test1_ = source.copy()
    Test1_[:, :, 0] = source[:, :, 0]
    Test1_[:, :, 1] = source[:, :, 1]
    Test1 = Test1_  # [x, y, z]

    for i in range(len(Test1_)):
        Test1[i, :, 0] = Test1_[i, :, 0] * (bndbox[2] - bndbox[0]) / cropWidth + bndbox[0]  # x
        Test1[i, :, 1] = Test1_[i, :, 1] * (bndbox[3] - bndbox[1]) / cropHeight + bndbox[1]  # y
        Test1[i, :, 2] = Test1_[i, :, 2] #/ depthFactor
    
    #print (_divider)
    #print(Test1)
    Test2d = Test1.copy()[:, :, 0:2]

    single_img = image.copy()
    # depth_max = 6
    # single_img[single_img >= depth_max] = depth_max
    # single_img /= depth_max
    # single_img *= 255
    single_img = single_img.astype(np.uint8)
    single_img = cv2.cvtColor(single_img, cv2.COLOR_GRAY2BGR)
    single_img = draw_humans_visibility(single_img,
                                        [Test2d[0]],
                                        kp_connections(get_keypoints()),
                                        jointColors)
    # cv2.imwrite("/home/root/sourceFiles/pose_" + str(count) + ".jpeg", single_img)
    # poseOutputVideo.write(single_img)
    return single_img

def InitQuantModel(model):
    global dpu_runner
    g = xir.Graph.deserialize(model)

    subgraphs = g.get_root_subgraph().toposort_child_subgraph()
        
    dpu_subgraph2 = subgraphs[2]
    
    dpu_runner = vart.Runner.create_runner(dpu_subgraph2, "run")
    
    return dpu_runner

def preprocess_fn(image, bndbox):
    '''
    Image pre-processing.
    Opens image , crop image to bbox, resize to 288x288, normalize
    and then scales by input quantization scaling factor
    input arg: path of image file
    return: numpy array
    '''

    new_Xmin = max(bndbox[0] , 1)
    new_Ymin = max(bndbox[1] , 1)
    new_Xmax = min(bndbox[2] , image.shape[1] - 1)
    new_Ymax = min(bndbox[3] , image.shape[0] - 1)
    # new_Xmin = (bndbox[0])
    # new_Ymin = (bndbox[1])
    # new_Xmax = (bndbox[2])
    # new_Ymax = (bndbox[3])

    imCrop = image.copy()[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)] 
    # Crop = cv2.cvtColor(imCrop, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite("/home/root/sourceFiles/imgResize.jpeg", Crop)
    try:
        imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        print(f"shape of image:{image.shape}\tshape of imCrop:{imCrop.shape}")
        print(f"A2J preprocessing: {e}")
    imgResize = (imgResize - Img_mean) / Img_std
    # imgResize = imgResize * fix_scale 
    
    imageOutputs[:, :, 0] = imgResize.astype(np.float32)
    imageOutputs[:, :, 1] = imgResize.astype(np.float32)
    imageOutputs[:, :, 2] = imgResize.astype(np.float32)  

    return imageOutputs

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
    '''
    Thread worker function
    '''
    post_processing = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None)
    # output = torch.FloatTensor()
    #  Set up encoder DPU runner buffers & I/O mapping dictionary
    global a2j_dict, inbuffer, outbuffer, result
    # result = []
    a2j_dict, inbuffer, outbuffer = init_dpu_runner(dpu_runner)
          
    '''
    initialise input and execute DPU runner
    '''    
    a2j_dict['A2J_model__A2J_model_ResNetBackBone_Backbone__input_1_swim_transpose_0_fix'] = img.reshape(tuple(a2j_dict['A2J_model__A2J_model_ResNetBackBone_Backbone__input_1_swim_transpose_0_fix'].shape[1:]))

    execute_async_method(dpu_runner, a2j_dict)

    # write results to global predictions buffer
    out_q1 = a2j_dict['A2J_model__A2J_model_ClassificationModel_classificationModel__Conv2d_output__11586_fix'].copy()
    out_q2 = a2j_dict['A2J_model__A2J_model_DepthRegressionModel_DepthRegressionModel__Conv2d_output__11920_fix'].copy()
    out_q3 = a2j_dict['A2J_model__A2J_model_RegressionModel_regressionModel__Conv2d_output__11749_fix'].copy()

    classification = (out_q1.reshape(1,5184,15))
    DepthRegressionModel = (out_q2.reshape(1,5184,15))
    regression = (out_q3.reshape(1,5184,15,2))

    pred_keypoints = post_processing((classification, regression, DepthRegressionModel),voting=False)
        # output = torch.cat([output,pred_keypoints.data], 0)  

    # result = (output.data.numpy())
    #print("Done with the DPU runner")
    return pred_keypoints.data.numpy()

# def runPoseEstimation(image, Bbox, dpu_runner, count, saveVideo):
def runPoseEstimation(image, Bbox, model, count, saveVideo):
    # global out_q1, out_q2, out_q3, classification, DepthRegressionModel, regression, dpu_runner 
    global dpu_runner 
    # out_q1 = []
    # out_q2 = []
    # out_q3 = []
    # classification          = []
    # DepthRegressionModel    = []
    # regression              = []

    ''' preprocess images '''
    #print (_divider)
    img = preprocess_fn(image, Bbox)

    '''run inference '''    
    outputs = runInference(dpu_runner, img)
    # del dpu_runner
    outputImage = pattingTest(image, outputs, Bbox, count)
    cv2.imwrite("/home/root/sourceFiles/poseOutput_1600_"+str(count)+".jpeg", outputImage)
   
    ''' post-processing '''
    if saveVideo:
        outputImage = pattingTest(image, outputs, Bbox, count)
        return outputs, outputImage
    else:
        return outputs

# only used if script is run as 'main' from command line
def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()  
    ap.add_argument('-d', '--image_dir', type=str, default='tempImages', help='Path to folder of images. Default is images')  
    ap.add_argument('-t', '--threads',   type=int, default=1,        help='Number of threads. Default is 1')
    ap.add_argument('-m', '--model',     type=str, default='A2J_kv260.xmodel', help='Path of xmodel. Default is A2J_kv260.xmodel')
    args = ap.parse_args()  

    print ('Command line options:')
    print (' --image_dir : ', args.image_dir)
    print (' --threads   : ', args.threads)
    print (' --model     : ', args.model)



    # app(args.image_dir,args.threads,args.model)

if __name__ == '__main__':
    main()

