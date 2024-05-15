# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 18:24:11 2022

@author: nrshr
"""

import cv2
import torch
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import scipy.io as scio
import os
from PIL import Image
from torch.autograd import Variable
import model as model
# import model_depthreg_noncomp as model
import anchor as anchor
# import anchor_depthreg_noncomp  as anchor
from tqdm import tqdm
import random_erasing
import logging
import time
import datetime
import random
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib.patches import Circle, ConnectionPatch
import sys
import json
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
np.set_printoptions(suppress=True)
fx = 288
fy = -288
u0 = 160
v0 = 120

#DataHyperParms
keypointsNumber = 15
cropWidth = 288
cropHeight = 288
batch_size_train = 128
batch_size_val = 1
learning_rate = 0.00035
Weight_Decay = 1e-4
# Weight_Decay = 0.1
nepoch = 100
RegLossFactor = 5#3
spatialFactor = 1#0.5
RandCropShift = 15#5
RandshiftDepth = 5#1
RandRotate = 45#180
RandScale = (0.4, 0.8)#(1.0, 0.5)
# xy_thres = 120
depth_thres = 0.5#0.4
depthFactor = 50

intrinsics = {'fx': 504.1189880371094, 'fy': 504.042724609375, 'cx': 231.7421875, 'cy': 320.62640380859375}
realsense_intrinsics = {'fx': 391.037 , 'fy': 391.037, 'cx': 322.89, 'cy': 244.123}
def pixel2world(x,y,z):
    if DATASET == 'ITOP':
        worldX = (x - 160.0)*z*0.0035
        worldY = (120.0 - y)*z*0.0035
    elif DATASET == 'MP3DHP' or DATASET == 'MP3DHP_BGAUG':        
        worldX = (x - intrinsics['cx']) * z / intrinsics['fx']
        worldY = (y - intrinsics['cy']) * z / intrinsics['fy']
    elif DATASET == 'SDSU_PSG':        
        worldX = (x - realsense_intrinsics['cx']) * z / realsense_intrinsics['fx']
        worldY = (y - realsense_intrinsics['cy']) * z / realsense_intrinsics['fy']
    return worldX,worldY

def world2pixel(x,y,z):
    pixelX = 160.0 + x / (0.0035 * z)
    pixelY = 120.0 - y / (0.0035 * z)
    return pixelX,pixelY
'''
DATASET options available
ITOP
MP3DHP
MP3DHP_BGAUG
UR_Falldetection
SDSU_PSG
'''
DATASET = 'SDSU_PSG'

if DATASET == 'ITOP':
    TrainImgFrames = 17991
    TestImgFrames = 4863
    trainingImageDir = '/mnt/beegfs/home/ramesh/Datasets/ITOP_side/PreProcessed/train/' # mat images
    testingImageDir = '/mnt/beegfs/home/ramesh/Datasets/ITOP_side/PreProcessed/test/' # mat images
    keypointsfileTest = '/mnt/beegfs/home/ramesh/A2J/data/itop_side/itop_side_keypoints3D_test.mat' # in world coordinates
    keypointsfileTrain = '/mnt/beegfs/home/ramesh/A2J/data/itop_side/itop_side_keypoints3D_train.mat' # in world coordinates

    bndbox_test = scio.loadmat('../data/itop_side/itop_side_bndbox_test_npy.mat' )['FRbndbox_test']
    bndbox_train = scio.loadmat('../data/itop_side/itop_side_bndbox_train_npy.mat' )['FRbndbox']
    Img_mean = np.load('../data/itop_side/itop_side_mean.npy')[3] #3.44405131082671
    Img_std = np.load('../data/itop_side/itop_side_std.npy')[3] #0.5403981602753222
    imgWidth = 320
    imgHeight = 240
    model_dir = '/mnt/beegfs/home/ramesh/A2J/model/ITOP_side.pth'
    result_file = 'result_test.txt'
    save_dir = './result/ITOP_NoExpand/'

    #loading GT keypoints
    keypointsWorldtest = scio.loadmat(keypointsfileTest)['keypoints3D'].astype(np.float32) # this is in world coordinates
    keypointsPixeltest = np.ones((len(keypointsWorldtest),15,2),dtype='float32')
    # keypointsPixeltest = world2pixel(keypointsWorldtest.copy()) # this is in pixel coordinates, 320 x 240
    keypointsPixeltest_tuple = world2pixel(keypointsWorldtest[:,:,0],keypointsWorldtest[:,:,1],keypointsWorldtest[:,:,2])
    assert np.shape(keypointsPixeltest_tuple[0])==(len(keypointsWorldtest),15), "data transform error!"
    keypointsPixeltest[:,:,0] = keypointsPixeltest_tuple[0]
    keypointsPixeltest[:,:,1] = keypointsPixeltest_tuple[1]

    keypointsWorldtrain = scio.loadmat(keypointsfileTrain)['keypoints3D'].astype(np.float32)
    keypointsPixeltrain = np.ones((len(keypointsWorldtrain),15,2),dtype='float32')
    # keypointsPixeltrain = world2pixel(keypointsWorldtrain.copy())
    keypointsPixeltrain_tuple = world2pixel(keypointsWorldtrain[:,:,0],keypointsWorldtrain[:,:,1],keypointsWorldtrain[:,:,2])
    assert np.shape(keypointsPixeltrain_tuple[0])==(len(keypointsWorldtrain),15), "data transform error!"
    keypointsPixeltrain[:,:,0] = keypointsPixeltrain_tuple[0]
    keypointsPixeltrain[:,:,1] = keypointsPixeltrain_tuple[1]
elif DATASET == 'MP3DHP':
    trainingImageDir = '/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/dataset/train_val/depth_maps/'
    testingImageDir = '/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/dataset/train_val/depth_maps/'
    Img_mean = 3
    Img_std = 2
    imgWidth = 480
    imgHeight = 512
    save_dir = './result/MP3DHP_A2J/'
    annotations_train = dict()

    with open('/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/labels/train_val/labels_train.json') as f:
        dict = json.load(f)
        annotations_train = dict

    keypointsWorldtrain = []
    keypointsPixeltrain = []
    bndbox_train = []
    for key,value in dict.items():
        if key != 'intrinsics':
            temp = value[0]
            keypointsWorldtrain.append(temp['3d_joints'])
            keypointsPixeltrain.append(temp['2d_joints'])
            bndbox_train.append(temp['bbox'])

    keypointsWorldtrain = np.asarray(keypointsWorldtrain, dtype=np.float32)
    keypointsPixeltrain = np.asarray(keypointsPixeltrain, dtype=np.float32) 
    bndbox_train = np.asarray(bndbox_train, dtype=np.float32)
    TrainImgFrames = len(keypointsPixeltrain)

    with open('/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/labels/train_val/labels_test.json') as f:
        dict = json.load(f)
        annotations_test = dict

    keypointsWorldtest = []
    keypointsPixeltest = []
    bndbox_test = []
    for key,value in dict.items():
        if key != 'intrinsics':
            temp = value[0]
            keypointsWorldtest.append(temp['3d_joints'])
            keypointsPixeltest.append(temp['2d_joints'])
            bndbox_test.append(temp['bbox'])

    keypointsWorldtest = np.asarray(keypointsWorldtest, dtype=np.float32)
    keypointsPixeltest = np.asarray(keypointsPixeltest, dtype=np.float32)
    bndbox_test = np.asarray(bndbox_test, dtype=np.float32)
    TestImgFrames = len(keypointsPixeltest)

    test_image_ids = list(annotations_test.keys())
    train_image_ids = list(annotations_train.keys())
elif DATASET == 'MP3DHP_BGAUG':
    trainingImageDir = '/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/dataset/train_val/depth_maps/'
    testingImageDir = '/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/dataset/train_val/depth_maps/'
    maskDir = '/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/dataset/train_val/seg_maps/'
    bgDir = '/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/dataset/train_val/bg_maps/'
    bg_anno = '/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/labels/train_val/labels_bg.json'
    bg_list = list(json.load(open(bg_anno, 'r')).values())

    Img_mean = 3
    Img_std = 2
    imgWidth = 480
    imgHeight = 512
    save_dir = './result/MP3DHP_BgAugment_A2J/'
    annotations_train = dict()

    with open('/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/labels/train_val/labels_train.json') as f:
        dict = json.load(f)
        annotations_train = dict

    keypointsWorldtrain = []
    keypointsPixeltrain = []
    bndbox_train = []
    for key,value in dict.items():
        if key != 'intrinsics':
            temp = value[0]
            keypointsWorldtrain.append(temp['3d_joints'])
            keypointsPixeltrain.append(temp['2d_joints'])
            bndbox_train.append(temp['bbox'])

    keypointsWorldtrain = np.asarray(keypointsWorldtrain, dtype=np.float32)
    keypointsPixeltrain = np.asarray(keypointsPixeltrain, dtype=np.float32) 
    bndbox_train = np.asarray(bndbox_train, dtype=np.float32)
    TrainImgFrames = len(keypointsPixeltrain)

    with open('/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/labels/train_val/labels_test.json') as f:
        dict = json.load(f)
        annotations_test = dict

    keypointsWorldtest = []
    keypointsPixeltest = []
    bndbox_test = []
    for key,value in dict.items():
        if key != 'intrinsics':
            temp = value[0]
            keypointsWorldtest.append(temp['3d_joints'])
            keypointsPixeltest.append(temp['2d_joints'])
            bndbox_test.append(temp['bbox'])

    keypointsWorldtest = np.asarray(keypointsWorldtest, dtype=np.float32)
    keypointsPixeltest = np.asarray(keypointsPixeltest, dtype=np.float32)
    bndbox_test = np.asarray(bndbox_test, dtype=np.float32)
    TestImgFrames = len(keypointsPixeltest)

    test_image_ids = list(annotations_test.keys())
    train_image_ids = list(annotations_train.keys())
    BgImgFrames = 8680

    trainingImageDir = '/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/dataset/train_val/depth_maps/'
    testingImageDir = '/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/dataset/train_val/depth_maps/'
    Img_mean = 3
    Img_std = 2
    imgWidth = 480
    imgHeight = 512
    save_dir = './result/MP3DHP_A2J/'
    annotations_train = dict()

    with open('/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/labels/train_val/labels_train.json') as f:
        dict = json.load(f)
        annotations_train = dict

    keypointsWorldtrain = []
    keypointsPixeltrain = []
    bndbox_train = []
    for key,value in dict.items():
        if key != 'intrinsics':
            temp = value[0]
            keypointsWorldtrain.append(temp['3d_joints'])
            keypointsPixeltrain.append(temp['2d_joints'])
            bndbox_train.append(temp['bbox'])

    keypointsWorldtrain = np.asarray(keypointsWorldtrain, dtype=np.float32)
    keypointsPixeltrain = np.asarray(keypointsPixeltrain, dtype=np.float32) 
    bndbox_train = np.asarray(bndbox_train, dtype=np.float32)
    TrainImgFrames = len(keypointsPixeltrain)

    with open('/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/labels/train_val/labels_test.json') as f:
        dict = json.load(f)
        annotations_test = dict

    keypointsWorldtest = []
    keypointsPixeltest = []
    bndbox_test = []
    for key,value in dict.items():
        if key != 'intrinsics':
            temp = value[0]
            keypointsWorldtest.append(temp['3d_joints'])
            keypointsPixeltest.append(temp['2d_joints'])
            bndbox_test.append(temp['bbox'])

    keypointsWorldtest = np.asarray(keypointsWorldtest, dtype=np.float32)
    keypointsPixeltest = np.asarray(keypointsPixeltest, dtype=np.float32)
    bndbox_test = np.asarray(bndbox_test, dtype=np.float32)
    TestImgFrames = len(keypointsPixeltest)

    test_image_ids = list(annotations_test.keys())
    train_image_ids = list(annotations_train.keys())
elif DATASET == 'SDSU_PSG':
    trainingImageDir = '/mnt/beegfs/home/ramesh/SDSU_PSG_RGBD/python_scripts/SDSU_PSG_a2j_train_filepath.json'
    testingImageDir = '/mnt/beegfs/home/ramesh/SDSU_PSG_RGBD/python_scripts/SDSU_PSG_a2j_test_filepath.json'
    trainTestImageDirPath = '/mnt/beegfs/home/ramesh/SDSU_PSG_RGBD/python_scripts/SDSU_PSG_a2j_filepath.json'

    Img_mean = 3
    Img_std = 2
    imgWidth = 480
    imgHeight = 640
    save_dir = './result/SDSU_PSG_v2/'
    annotations_train = dict()

    with open(trainTestImageDirPath) as f:
        train_test_dict = json.load(f)   

    # with open(testingImageDir) as f:
    #     test_dict = json.load(f)   

    with open('/mnt/beegfs/home/ramesh/SDSU_PSG_RGBD/python_scripts/SDSU_PSG_a2j_train.json') as f:
        dict = json.load(f)
        annotations_train = dict

    keypointsWorldtrain = []
    keypointsPixeltrain = []
    bndbox_train = []
    for key,value in dict.items():
        if key != 'intrinsics':
            temp = value[0]
            keypointsWorldtrain.append(temp['3d_joints'])
            keypointsPixeltrain.append(temp['2d_joints'])
            bndbox_train.append(temp['bbox'])

    keypointsWorldtrain = np.asarray(keypointsWorldtrain, dtype=np.float32)
    keypointsPixeltrain = np.asarray(keypointsPixeltrain, dtype=np.float32) 
    bndbox_train = np.asarray(bndbox_train, dtype=np.float32)
    TrainImgFrames = len(keypointsPixeltrain)

    with open('/mnt/beegfs/home/ramesh/SDSU_PSG_RGBD/python_scripts/SDSU_PSG_a2j_test.json') as f:
        dict = json.load(f)
        annotations_test = dict

    keypointsWorldtest = []
    keypointsPixeltest = []
    bndbox_test = []
    for key,value in dict.items():
        if key != 'intrinsics':
            temp = value[0]
            keypointsWorldtest.append(temp['3d_joints'])
            keypointsPixeltest.append(temp['2d_joints'])
            bndbox_test.append(temp['bbox'])

    keypointsWorldtest = np.asarray(keypointsWorldtest, dtype=np.float32)
    keypointsPixeltest = np.asarray(keypointsPixeltest, dtype=np.float32)
    bndbox_test = np.asarray(bndbox_test, dtype=np.float32)
    TestImgFrames = len(keypointsPixeltest)

    train_image_ids = list(annotations_train.keys())
    test_image_ids = list(annotations_test.keys())
elif DATASET == 'UR_Falldetection':
    trainingImageDir = '/mnt/beegfs/home/ramesh/Datasets/UR_Fall_Detection_Dataset/validFrames/validDepthTrainMAT/'
    testingImageDir = '/mnt/beegfs/home/ramesh/Datasets/UR_Fall_Detection_Dataset/validFrames/validDepthTestMAT/'

    Img_mean = 3
    Img_std = 2
    imgWidth = 640
    imgHeight = 480
    save_dir = './result/URFall/'
    annotations_train = dict()

    with open('/mnt/beegfs/home/ramesh/Datasets/UR_Fall_Detection_Dataset/UR_fall_valid_train.json') as f:
        dict = json.load(f)
        annotations_train = dict

    keypointsWorldtrain = []
    keypointsPixeltrain = []
    bndbox_train = []
    for key,value in dict.items():
        if key != 'intrinsics':
            temp = value[0]
            keypointsWorldtrain.append(temp['3d_joints'])
            keypointsPixeltrain.append(temp['2d_joints'])
            bndbox_train.append(temp['bbox'])

    keypointsWorldtrain = np.asarray(keypointsWorldtrain, dtype=np.float32)
    keypointsPixeltrain = np.asarray(keypointsPixeltrain, dtype=np.float32) 
    bndbox_train = np.asarray(bndbox_train, dtype=np.float32)
    TrainImgFrames = len(keypointsPixeltrain)

    # for i in range(len(bndbox_train)):
    #     print(f"bndbox[{i}][0] = {bndbox_train[i][0]}")
    #     print(f"bndbox[{i}][1] = {bndbox_train[i][1]}")
    #     print(f"bndbox[{i}][2] = {bndbox_train[i][2]}")
    #     print(f"bndbox[{i}][3] = {bndbox_train[i][3]}")
    #     print("\n")

    # exit()
    with open('/mnt/beegfs/home/ramesh/Datasets/UR_Fall_Detection_Dataset/UR_fall_valid_test.json') as f:
        dict = json.load(f)
        annotations_test = dict

    keypointsWorldtest = []
    keypointsPixeltest = []
    bndbox_test = []
    for key,value in dict.items():
        if key != 'intrinsics':
            temp = value[0]
            keypointsWorldtest.append(temp['3d_joints'])
            keypointsPixeltest.append(temp['2d_joints'])
            bndbox_test.append(temp['bbox'])

    keypointsWorldtest = np.asarray(keypointsWorldtest, dtype=np.float32)
    keypointsPixeltest = np.asarray(keypointsPixeltest, dtype=np.float32)
    bndbox_test = np.asarray(bndbox_test, dtype=np.float32)
    TestImgFrames = len(keypointsPixeltest)


try:
    os.makedirs(save_dir)
except OSError:
    pass

# def pixel2world(x):
    # x[:, :, 0] = (x[:, :, 0] - u0) * x[:, :, 2] * 0.0035
    # x[:, :, 1] = (v0 - x[:, :, 1]) * x[:, :, 2] * 0.0035
    # return x
    
# def world2pixel(x):
#     x[:, :, 0] = u0 + x[:, :, 0] / (x[:, :, 2] * 0.0035)
#     x[:, :, 1] = v0 - x[:, :, 1] / (x[:, :, 2] * 0.0035)
#     return x

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

# use GPU if available   
if (torch.cuda.device_count() > 0):
    print('You have',torch.cuda.device_count(),'CUDA devices available')
    for i in range(torch.cuda.device_count()):
        print(' Device',str(i),': ',torch.cuda.get_device_name(i))
    print('Selecting all available devices')
    device = torch.device('cuda')
    print("device selection DONE")
else:
    print('No CUDA devices available..selecting CPU')
    device = torch.device('cpu')
#
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

def evaluation2D_perJoint(source, target, dist_th_2d):
    assert np.shape(source) == np.shape(target), "source has different shape with target"
    count = 0
    acc_vec = []

    for j in range(keypointsNumber):
        for i in range(len(source)):
            if np.square(source[i, j, 0] - target[i, j, 0]) + np.square(
                    source[i, j, 1] - target[i, j, 1]) < np.square(dist_th_2d):
                count = count + 1

        accuracy = count / (len(source))
        print('joint_', j, joint_id_to_name[j], ', accuracy: ', accuracy)
        acc_vec.append(accuracy)
        accuracy = 0
        count = 0

def evaluation2D(source, target, dist_th_2d):
    assert np.shape(source) == np.shape(target), "source has different shape with target"
    count = 0
    for i in range(len(source)):
        for j in range(keypointsNumber):
            if np.square(source[i, j, 0] - target[i, j, 0]) + np.square(
                    source[i, j, 1] - target[i, j, 1]) < np.square(dist_th_2d):
                count = count + 1
    accuracy = count / (len(source) * keypointsNumber)
    return accuracy

def transform(img, label, matrix):
    '''
    img: [H, W] label, [N,2]
    '''
    img_out = cv2.warpAffine(img,matrix,(cropWidth,cropHeight))
    label_out = np.ones((keypointsNumber, 3))
    label_out[:,:2] = label[:,:2].copy()
    label_out = np.matmul(matrix, label_out.transpose())
    label_out = label_out.transpose()

    return img_out, label_out

def dataPreprocess(index, imgDir, keypointsWorld, keypointsPixel, bndbox, mode, augment=True):
    if DATASET == 'UR_Falldetection':        
        img = scio.loadmat(imgDir + str(index+1) + '.mat')['depth']
        # cv2.imwrite(str(index+1) + '_img.png',img)
    elif DATASET == 'SDSU_PSG':
        if mode == 'TEST':
            img = np.load(train_test_dict[test_image_ids[index]]).astype(float)
        else:
            img = np.load(train_test_dict[train_image_ids[index]]).astype(float)
    elif DATASET == 'MP3DHP':
        if mode == 'TEST':
            img = np.load(os.path.join(imgDir, test_image_ids[index])).astype(float)
        else:
            img = np.load(os.path.join(imgDir, train_image_ids[index])).astype(float)

    elif DATASET == 'MP3DHP_BGAUG':
        if mode == 'TEST':
            depth_img = np.load(os.path.join(imgDir, test_image_ids[index])).astype(float)
            mask = np.load(os.path.join(maskDir, test_image_ids[index])).astype(float)
        else:
            depth_img = np.load(os.path.join(imgDir, train_image_ids[index])).astype(float)
            mask = np.load(os.path.join(maskDir, train_image_ids[index])).astype(float)
        bg_id = index % BgImgFrames
        bg_image_path = os.path.join(bgDir, bg_list[bg_id]['file_name'])
        bg_image = np.load(bg_image_path)
        img = depth_img * mask + bg_image * (np.ones_like(mask) - mask)

    elif DATASET == 'ITOP':
        data4D = scio.loadmat(imgDir + str(index+1) + '.mat')['DepthNormal']
        img = data4D[:,:,3]
    
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32') 
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32') 

    if augment:
        RandomOffset_1 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_2 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_3 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_4 = np.random.randint(-1*RandCropShift,RandCropShift)
        # RandomOffsetDepth = np.random.normal(0, RandshiftDepth, cropHeight*cropWidth).reshape(cropHeight,cropWidth) 
        # RandomOffsetDepth[np.where(RandomOffsetDepth < RandshiftDepth)] = 0
        RandomRotate = np.random.randint(-1*RandRotate,RandRotate)
        RandomScale = np.random.rand()*RandScale[0]+RandScale[1]
        matrix = cv2.getRotationMatrix2D((cropWidth/2,cropHeight/2),RandomRotate,RandomScale)
    else:
        RandomOffset_1, RandomOffset_2, RandomOffset_3, RandomOffset_4 = 0, 0, 0, 0
        RandomRotate = 0
        RandomScale = 1
        # RandomOffsetDepth = 0
        matrix = cv2.getRotationMatrix2D((cropWidth/2,cropHeight/2),RandomRotate,RandomScale)

    new_Xmin = max(bndbox[index][0] + RandomOffset_1, 0)
    new_Ymin = max(bndbox[index][1] + RandomOffset_2, 0)
    new_Xmax = min(bndbox[index][2] + RandomOffset_3, img.shape[1] - 1)
    new_Ymax = min(bndbox[index][3] + RandomOffset_4, img.shape[0] - 1)
    # print(f"new_Xmin = {new_Xmin}")
    # print(f"new_Ymin = {new_Ymin}")
    # print(f"new_Xmax = {new_Xmax}")
    # print(f"new_Ymax = {new_Ymax}")
    # print("-"*50)
    if DATASET == 'UR_Falldetection' or DATASET == 'SDSU_PSG':
        # image[start_row:end_row, start_column:end_column] 
        imCrop = img.copy()[int(new_Ymin):int(new_Ymin+new_Ymax), int(new_Xmin):int(new_Xmin+new_Xmax)]
    else:
        imCrop = img.copy()[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)]
    # cv2.imwrite('/mnt/beegfs/home/ramesh/A2J/src_train/' + str(index+1) + '_imCrop.png',imCrop)
    # print(f"mode = {mode}\timage #{index+1}\timg mean = {np.mean(img)}\timg_shape = {img.shape}\timCrop_shape = {imCrop.shape}\timCrop_mean = {np.mean(imCrop)}")
    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)
    
    imgResize = np.asarray(imgResize,dtype = 'float32')  # H*W*C
   
    imgResize = (imgResize - Img_mean) / Img_std

    ## label
    label_xy = np.ones((keypointsNumber, 2), dtype='float32')
    
    if DATASET == "UR_Falldetection" or DATASET == 'SDSU_PSG':        
        label_xy[:, 0] = (keypointsPixel[index, :, 0] - new_Ymin) * cropHeight / (new_Ymax)  # x
        label_xy[:, 1] = (keypointsPixel[index, :, 1] - new_Xmin) * cropWidth  / (new_Xmax)  # y
    else:
        label_xy[:, 0] = (keypointsPixel[index, :, 0] - new_Xmin) * cropWidth / (new_Xmax - new_Xmin)  # x
        label_xy[:, 1] = (keypointsPixel[index, :, 1] - new_Ymin) * cropHeight / (new_Ymax - new_Ymin)  # y
 
    ## commented temporarily for visulization purpose##
    if augment:
        imgResize, label_xy = transform(imgResize, label_xy, matrix)  ## rotation, scale

    imageOutputs[:,:,0] = imgResize
    
    if DATASET == "UR_Falldetection" or DATASET == 'SDSU_PSG': 
        labelOutputs[:,1] = label_xy[:,1]
        labelOutputs[:,0] = label_xy[:,0]
    else:
        labelOutputs[:,1] = label_xy[:,0]
        labelOutputs[:,0] = label_xy[:,1]
    # labelOutputs[:,2] = (keypointsUVD[index,:,2])#*RandomScale   # Z  
    labelOutputs[:,2] = (keypointsWorld.copy()[index,:,2])*depthFactor   # Z  
    # if index == 0:
        # print(f"new_Xmin = {new_Xmin}")
        # print(f"new_Ymin = {new_Ymin}")
        # print(f"new_Xmax = {new_Xmax}")
        # print(f"new_Ymax = {new_Ymax}")

        # print(f"keypointsUVD[0,:,:] = {keypointsUVD[0,:,:]}") # this is in the 320x240

        # print(f"labelOutputs = {labelOutputs}") # this is in the 288x288
    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)
 
    return data, label

###################### Pytorch dataloader #################
class my_dataloader(torch.utils.data.Dataset):
    def __init__(self, ImgDir, keypointsWorld, keypointsPixel, num, bndbox, mode, augment=True):
    
        self.ImgDir = ImgDir
        self.keypointsWorld = keypointsWorld
        self.keypointsPixel = keypointsPixel
        self.num = num
        self.bndbox = bndbox
        self.mode = mode
        self.augment = augment
        # self.randomErase = random_erasing.RandomErasing(probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0])
        self.randomErase = random_erasing.RandomErasing(probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0], scale=0.2)

    def __getitem__(self, index):
    
        data, label = dataPreprocess(index, self.ImgDir, self.keypointsWorld, self.keypointsPixel, self.bndbox, self.mode, self.augment)
        if self.augment:
            data = self.randomErase(data)
        return data, label

    def __len__(self):
        return self.num
    
train_image_datasets = my_dataloader(trainingImageDir, keypointsWorldtrain, keypointsPixeltrain, TrainImgFrames, bndbox_train, mode = 'TRAIN', augment=False)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size = batch_size_train,
shuffle = True, num_workers = 8, pin_memory=True)

test_image_datasets = my_dataloader(testingImageDir, keypointsWorldtest, keypointsPixeltest, TestImgFrames, bndbox_test, mode = 'TEST', augment=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size_val,
shuffle = False, num_workers = 8, pin_memory=True)

writer = SummaryWriter(os.path.join(save_dir, 'Tensorboard/runs/'))
trainOut = cv2.VideoWriter(save_dir + 'trainOut.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (cropWidth,cropHeight))
# valOut = cv2.VideoWriter(save_dir + 'valOut.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (cropWidth,cropHeight))

def train():
    print('==================Training Loop Start====================================')
    net = model.A2J_model(num_classes = keypointsNumber)
    # net = model.Cls_Reg_Dual_Path_Net(num_classes = keypointsNumber, useNormal=Use_Normal)
    # net = net.cuda()
    net = torch.nn.DataParallel(net).to(device)
    # net.load_state_dict(torch.load('/mnt/beegfs/home/ramesh/A2J/src_train/result/MP3DHP_A2J/net_49_wetD_0.0001_depFact_1_RegFact_5_rndShft_15.pth'), strict=True)
    # net.load_state_dict(torch.load('/mnt/beegfs/home/ramesh/A2J/src_train/result/MP3DHP_BgAugment_A2J/net_34_wetD_0.0001_depFact_1_RegFact_5_rndShft_15.pth'), strict=True)
    net.load_state_dict(torch.load('/mnt/beegfs/home/ramesh/A2J/src_train/result/URFall/net_99_wetD_0.0001_depFact_1_RegFact_5_rndShft_15.pth'), strict=True)
    
    post_precess = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None)
    criterion = anchor.A2J_loss(shape=[cropHeight//16,cropWidth//16],thres = [16.0,32.0],stride=16,\
        spatialFactor=spatialFactor,img_shape=[cropHeight, cropWidth],P_h=None, P_w=None)
    # criterion = anchor.FocalLoss(shape=[cropHeight//16,cropWidth//16],thres = [16.0,32.0],stride=16,\
    #     spatialFactor=spatialFactor,img_shape=[cropHeight, cropWidth],P_h=None, P_w=None)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=Weight_Decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(save_dir, 'train.log'), force=True, level=logging.INFO)
    logging.info('======================================================')
    print('======================================================')
    
    for epoch in range(nepoch):
        net = net.train()
        # scheduler.step()
        train_loss_add = 0.0
        val_loss_add = 0.0
        Cls_loss_add = 0.0
        Cls_loss_val_add = 0.0
        Reg_loss_add = 0.0
        Reg_loss_val_add = 0.0
        timer = time.time()
        valOut = cv2.VideoWriter(save_dir + '/images/validationOut/epoch_' + str(epoch)+ '_valOut.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (cropWidth,cropHeight))

        # Training loop
        for i, (img, label) in enumerate(train_dataloaders):
            
            torch.cuda.synchronize() 

            # img, label = img.cuda(), label.cuda()
            img, label = img.to(device), label.to(device)
            heads  = net(img)     
            optimizer.zero_grad()  

#           visualize input images to the network with GT points -- block taken form MP3DHP
            depth_max = 5#6
            depth_mean = 3#Img_mean
            depth_std = 2#Img_std
            img, label = img.cpu(), label.cpu()
            img_ = img.numpy().copy()
            img_ = img_[0, 0, :, :]
            label = label.numpy()
            label_new = np.copy(label)
            label_new[:, :, 0] = label[:, :, 1]
            label_new[:, :, 1] = label[:, :, 0]

            human_2d = [label_new[0]]

            single_img = np.copy(img_)
            single_img *= depth_std
            single_img += depth_mean
            single_img[single_img >= depth_max] = depth_max
            single_img /= depth_max
            single_img *= 255
            single_img = single_img.astype(np.uint8)
            single_img = cv2.cvtColor(single_img, cv2.COLOR_GRAY2BGR)
            single_img = draw_humans_visibility(single_img,
                                                human_2d,
                                                kp_connections(get_keypoints()),
                                                jointColors)
            # plt.imshow(single_img)
            # plt.show()
            single_img = single_img.astype(np.uint8)
            # cv2.imwrite(save_dir + '/images/trainOut/' + str(i) + '_' + train_image_ids[i] + '.png', single_img)
            # trainOut.write(single_img)
#
            # label = torch.from_numpy(label).cuda()
            label = torch.from_numpy(label).to(device)
            Cls_loss, Reg_loss = criterion(heads, label)
    
            loss = 1*Cls_loss + Reg_loss*RegLossFactor
            loss.backward()
            optimizer.step()
    
            torch.cuda.synchronize()
            
            train_loss_add = train_loss_add + (loss.item())*len(img)
            Cls_loss_add = Cls_loss_add + (Cls_loss.item())*len(img)
            Reg_loss_add = Reg_loss_add + (Reg_loss.item())*len(img)
            # printing loss info
            if i%10 == 0:
                print('epoch: ',epoch, ' step: ', i, 'Cls_loss ',Cls_loss.item(), 'Reg_loss ',Reg_loss.item(), ' total loss ',loss.item())
    
        scheduler.step()
    
        # time taken
        torch.cuda.synchronize()
        timer = time.time() - timer
        timer = timer / TrainImgFrames
        print('==> time to learn 1 sample = %f (ms)' %(timer*1000))
    
        train_loss_add = train_loss_add / TrainImgFrames
        Cls_loss_add = Cls_loss_add / TrainImgFrames
        Reg_loss_add = Reg_loss_add / TrainImgFrames
        print('mean train_loss_add of 1 sample: %f, #train_indexes = %d' %(train_loss_add, TrainImgFrames))
        print('mean Cls_loss_add of 1 sample: %f, #train_indexes = %d' %(Cls_loss_add, TrainImgFrames))
        print('mean Reg_loss_add of 1 sample: %f, #train_indexes = %d' %(Reg_loss_add, TrainImgFrames))
    
        Error_test = 0
        Error_train = 0
        Error_test_wrist = 0
        Accuracy_test = 0

        if (epoch % 1 == 0):  
            net = net.eval()
            output = torch.FloatTensor()
            labels = torch.FloatTensor()
            outputTrain = torch.FloatTensor()
                
            for i, (img, label) in tqdm(enumerate(test_dataloaders)):
                with torch.no_grad():
                    # img, label = img.cuda(), label.cuda()   
                    img, label = img.to(device), label.to(device)     
                    heads = net(img)  
                    pred_keypoints = post_precess(heads, voting=False)
                    output = torch.cat([output,pred_keypoints.data.cpu()], 0)
                    labels = torch.cat([labels,label.data.cpu()], 0)
                    
                    Cls_loss_val, Reg_loss_val = criterion(heads, label)
                    loss_val = 1 * Cls_loss_val + Reg_loss_val * RegLossFactor
                    val_loss_add = val_loss_add + (loss_val.item()) * len(img)
                    Cls_loss_val_add = Cls_loss_val_add + (Cls_loss_val.item()) * len(img)
                    Reg_loss_val_add = Reg_loss_val_add + (Reg_loss_val.item()) * len(img)

                    #Visualize outputs in Validation - block taken from MP3DHP
                    depth_max = 5#6
                    depth_mean = 3#Img_mean
                    depth_std = 2#Img_std
                    img, label = img.cpu(), label.cpu()
                    img_ = img.numpy().copy()
                    img_ = img_[0, 0, :, :]
                    pred_keypoints = pred_keypoints.cpu().numpy()
                    pred = pred_keypoints[0]
                    pred_new = pred.copy()
                    pred_new[:, 0] = pred[:, 1]
                    pred_new[:, 1] = pred[:, 0]
                    pred_new = [pred_new]

                    single_img = np.copy(img_)
                    single_img *= depth_std
                    single_img += depth_mean
                    single_img[single_img >= depth_max] = depth_max
                    single_img /= depth_max
                    single_img *= 255
                    single_img = single_img.astype(np.uint8)
                    single_img = cv2.cvtColor(single_img, cv2.COLOR_GRAY2BGR)
                    single_img = draw_humans_visibility(single_img,
                                                        pred_new,
                                                        kp_connections(get_keypoints()),
                                                        jointColors)
                    # plt.imshow(single_img)
                    single_img = single_img.astype(np.uint8)
                    # cv2.imwrite(save_dir + '/images/validationOut/epoch_' + str(epoch) + '_output_' + str(i) + '.png', single_img)
                    valOut.write(single_img)
#    
            val_loss_add     = val_loss_add / TestImgFrames
            Cls_loss_val_add = Cls_loss_val_add / TestImgFrames
            Reg_loss_val_add = Reg_loss_val_add / TestImgFrames
            print('mean val_loss_add of 1 sample: %f, #test_indexes = %d' % (val_loss_add, TestImgFrames))
            print('mean Cls_loss_val_add of 1 sample: %f, #test_indexes = %d' % (Cls_loss_val_add, TestImgFrames))
            print('mean Reg_loss_val_add of 1 sample: %f, #test_indexes = %d' % (Reg_loss_val_add, TestImgFrames))

            result = output.cpu().data.numpy() # result is in 288x288
            targetLabels = labels.cpu().data.numpy() # this label is in 288x288

            if DATASET == "UR_Falldetection" or DATASET == 'SDSU_PSG':
                w_org = imgWidth
                h_org = imgHeight
                dist_th_2d = 0.02 * np.sqrt(w_org ** 2 + h_org ** 2)

                Test1_ = result.copy()
                Test1_[:, :, 0] = result[:, :, 0]
                Test1_[:, :, 1] = result[:, :, 1]
                Test1 = Test1_  # [x, y, z]

                # if DATASET == "UR_Falldetection":        
                # xmin = 0
                # ymin = 1
                # xmax = 2
                # ymax = 3
                #     label_xy[:, 0] = (keypointsPixel[index, :, 0] - new_Ymin) * cropHeight / (new_Ymax)  # x
                #     label_xy[:, 1] = (keypointsPixel[index, :, 1] - new_Xmin) * cropWidth  / (new_Xmax)  # y
                for i in range(len(Test1_)):
                    Test1[i, :, 0] = Test1_[i, :, 0] * (bndbox_test[i, 3]) / cropHeight + bndbox_test[i, 1]  # x
                    Test1[i, :, 1] = Test1_[i, :, 1] * (bndbox_test[i, 2]) / cropWidth  + bndbox_test[i, 0]  # y
                    Test1[i, :, 2] = Test1_[i, :, 2] / depthFactor
                    # Test1[i, :, 0] = Test1_[i, :, 0] * (bndbox_test[i, 2] - bndbox_test[i, 0]) / cropWidth + bndbox_test[i, 0]  # x
                    # Test1[i, :, 1] = Test1_[i, :, 1] * (bndbox_test[i, 3] - bndbox_test[i, 1]) / cropHeight + bndbox_test[i, 1]  # y
                    # Test1[i, :, 2] = Test1_[i, :, 2] 

                Test2d = Test1.copy()[:, :, 0:2]
                accuracy_2d = evaluation2D(Test2d, keypointsPixeltest, dist_th_2d)
                print('Accuracy:', accuracy_2d)
                evaluation2D_perJoint(Test2d, keypointsPixeltest, dist_th_2d)
            else:
                Accuracy_test = evaluation10CMRule(result,keypointsWorldtest,bndbox_test)
                print('Accuracy:', Accuracy_test)
                evaluation10CMRule_perJoint(result,keypointsWorldtest,bndbox_test)
                Error_test = errorCompute(result,keypointsWorldtest,bndbox_test)
                print('epoch: ', epoch, 'Test error:', Error_test)

            saveNamePrefix = '%s/net_%d_wetD_' % (save_dir, epoch) + str(Weight_Decay) + '_depFact_' + str(spatialFactor) + '_RegFact_' + str(RegLossFactor) + '_rndShft_' + str(RandCropShift)
            torch.save(net.state_dict(), saveNamePrefix + '.pth')
    
        # tensorboard and log 
        # writer.add_scalar('Loss/Train', train_loss_add, epoch)
        # writer.add_scalar('Loss/Test', val_loss_add, epoch)
        writer.add_scalar('Accuracy/Test', accuracy_2d if (DATASET=="UR_Falldetection" or DATASET == 'SDSU_PSG') else Accuracy_test, epoch)
        # writer.add_scalar('Error/Test', Error_test, epoch)
        writer.add_scalars('Loss', {'Training Loss':train_loss_add, 'Validation Loss':val_loss_add}, epoch)
        logging.info('Epoch#%d: total loss = %.4f, Cls_loss = %.4f, Reg_loss = %.4f, Err_test = %.4f, lr = %.6f, Training_loss = %.4f, Validation_loss = %.4f, Accuracy_test = %.4f'
        %(epoch, train_loss_add, Cls_loss_add, Reg_loss_add, Error_test, scheduler.get_last_lr()[0], train_loss_add, val_loss_add,  accuracy_2d if (DATASET=="UR_Falldetection" or DATASET == 'SDSU_PSG') else Accuracy_test))

    writer.close()


def evaluation10CMRule(source, target, bndbox):
    assert np.shape(source) == np.shape(target), "source has different shape with target"
    Test1_ = np.zeros(source.shape)
    Test1_[:, :, 0] = source[:,:,1]
    Test1_[:, :, 1] = source[:,:,0]
    Test1_[:, :, 2] = source[:,:,2]
    Test1 = Test1_  # [x, y, z]

    for i in range(len(source)):
        Test1[i,:,0] = Test1_[i,:,0]*(bndbox[i,2]-bndbox[i,0])/cropWidth + bndbox[i,0]  # x
        Test1[i,:,1] = Test1_[i,:,1]*(bndbox[i,3]-bndbox[i,1])/cropHeight + bndbox[i,1]  # y
        Test1[i,:,2] = Test1_[i,:,2]/depthFactor

    outputs = np.ones((len(Test1),keypointsNumber,3))
    TestWorld_tuple = pixel2world(Test1[:,:,0],Test1[:,:,1],Test1[:,:,2])

    outputs[:,:,0] = TestWorld_tuple[0]
    outputs[:,:,1] = TestWorld_tuple[1]
    outputs[:,:,2] = Test1[:,:,2]
    count = 0
    accuracy = 0
    for i in range(len(source)):
        for j in range(keypointsNumber):
            answer = np.square(outputs[i, j, 0] - target[i, j, 0]) + np.square(outputs[i, j, 1] - target[i, j, 1]) + np.square(outputs[i, j, 2] - target[i, j, 2])
            if answer < np.square(0.1):  # 10cm # 0.1
                count = count + 1

    accuracy = count / (len(source) * keypointsNumber)
    return accuracy


def evaluation10CMRule_perJoint(source, target, bndbox):
    assert np.shape(source) == np.shape(target), "source has different shape with target"
    Test1_ = np.zeros(source.shape)
    Test1_[:, :, 0] = source[:,:,1]
    Test1_[:, :, 1] = source[:,:,0]
    Test1_[:, :, 2] = source[:,:,2]
    Test1 = Test1_  # [x, y, z]

    for i in range(len(source)):
        Test1[i,:,0] = Test1_[i,:,0]*(bndbox[i,2]-bndbox[i,0])/cropWidth + bndbox[i,0]  # x
        Test1[i,:,1] = Test1_[i,:,1]*(bndbox[i,3]-bndbox[i,1])/cropHeight + bndbox[i,1]  # y
        Test1[i,:,2] = Test1_[i,:,2]/depthFactor

    outputs = np.ones((len(Test1),keypointsNumber,3))
    TestWorld_tuple = pixel2world(Test1[:,:,0],Test1[:,:,1],Test1[:,:,2])

    outputs[:,:,0] = TestWorld_tuple[0]
    outputs[:,:,1] = TestWorld_tuple[1]
    outputs[:,:,2] = Test1[:,:,2]

    count = 0
    accuracy = 0
    for j in range(keypointsNumber):
        for i in range(len(source)):
            answer = np.square(outputs[i, j, 0] - target[i, j, 0]) + np.square(outputs[i, j, 1] - target[i, j, 1]) + np.square(outputs[i, j, 2] - target[i, j, 2])
            if answer < np.square(0.1):  # 10cm # 0.1
                count = count + 1   

        accuracy = count / (len(source))
        print('joint_', j, joint_id_to_name[j], ', accuracy: ', accuracy)
        accuracy = 0
        count = 0


def errorCompute(source, target, bndbox):
    assert np.shape(source) == np.shape(target), "source has different shape with target"
    Test1_ = np.zeros(source.shape)
    Test1_[:, :, 0] = source[:,:,1]
    Test1_[:, :, 1] = source[:,:,0]
    Test1_[:, :, 2] = source[:,:,2]
    Test1 = Test1_  # [x, y, z]

    for i in range(len(source)):
        Test1[i,:,0] = Test1_[i,:,0]*(bndbox[i,2]-bndbox[i,0])/cropWidth + bndbox[i,0]  # x
        Test1[i,:,1] = Test1_[i,:,1]*(bndbox[i,3]-bndbox[i,1])/cropHeight + bndbox[i,1]  # y
        Test1[i,:,2] = Test1_[i,:,2]/depthFactor

    outputs = np.ones((len(Test1),keypointsNumber,3))
    TestWorld_tuple = pixel2world(Test1[:,:,0],Test1[:,:,1],Test1[:,:,2])

    outputs[:,:,0] = TestWorld_tuple[0]
    outputs[:,:,1] = TestWorld_tuple[1]
    outputs[:,:,2] = Test1[:,:,2]
    errors = np.sqrt(np.sum((target - outputs) ** 2, axis=2))
    
    return np.mean(errors)

    
if __name__ == '__main__':
    train()
    # test()
