import copy
import logging
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import torch.utils.data
import torchvision
import torch.optim.lr_scheduler as lr_scheduler
import logging
import time
import datetime
import random
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, '..')
# from A2J_experiments import model, anchor, resnet, random_erasing
import model as model
import anchor as anchor
import torch, torch.nn as nn

# from numba import jit, cuda
from timeit import default_timer as timer   

trainingImageDir = '/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/dataset/test_bgaug/depth_maps/'

annotations_train = dict()

with open('/mnt/beegfs/home/ramesh/Datasets/MP-3DHP/ardata/labels/test_bgaug/labels.json') as f:
    dict = json.load(f)
    annotations_train = dict

train_image_ids = list(annotations_train.keys())

# @jit(target_backend='cuda')
def readFiles():
    for index in tqdm(range(29331)):
        try:
            depth_img = np.load(os.path.join(trainingImageDir, train_image_ids[index])).astype(float)

        except Exception as e:
            print("----------------------------------------------------------------------------------------------------")
            print(f"Caught an exception reading the file: {train_image_ids[index]}")
            print (str(e))
            print("----------------------------------------------------------------------------------------------------")
            pass

    print(f"All the files are loaded without issues")

if __name__=="__main__":
    start = timer()
    readFiles()
    print("with CPU:", timer()-start)