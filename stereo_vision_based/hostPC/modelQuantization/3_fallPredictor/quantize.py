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


'''
Simple PyTorch MNIST example - quantization
'''

'''
Author: Mark Harvey, Xilinx inc
'''

import os
import sys
import argparse
import json
import random
import torch
from torch.utils.data import Dataset
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

from model import FallModel

DIVIDER = '-----------------------------------------'


class loadDataset(Dataset):
    def __init__(self, json_file_path, window_size):
        self.data_list, self.label_list = self.read_data_from_json(json_file_path, window_size)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        label = self.label_list[index]

        return data, label

    def read_data_from_json(self, json_file_path, window_size):
        with open(json_file_path, 'r') as f:
            grouped_data_dict = json.load(f)

        data_list = []
        label_list = []

        for group_id, data in grouped_data_dict.items():
            keypoints = torch.Tensor(data['3d_joints'])
            label = data['isFall']
            num_frames = keypoints.shape[0]
            # if keypoints.shape == (200, 15, 3):
            if keypoints.shape == (15, 3):
                keypoints.resize_(1,15,3)
                data_list.append(keypoints)
                label_list.append(label)

        return data_list, label_list

def test(model, device, test_loader):
  model.eval() # Set the model to evaluation mode
  accuracy_val = 0
  with torch.no_grad():
    for idx, (keypoint, label) in enumerate(test_loader): 
      keypoint, label = keypoint.to(device), label.to(device)
      # Forward pass
      output = model(keypoint)
      # Update metrics
      pred = torch.argmax(output, dim=1)
      accuracy_val += torch.sum(pred == label).item()

    acc = 100. * accuracy_val / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(accuracy_val, len(test_loader.dataset), acc))
  return

def quantize(build_dir,quant_mode,batchsize):

  dset_dir = build_dir + '/dataset'
  # float_model = build_dir + '/float_model'
  float_model = build_dir + "/float_model/61_fall_detection_model.pth"
  quant_model = build_dir + '/quant_model_1Frame_4096'


  # use GPU if available   
  if (torch.cuda.device_count() > 0):
    print('You have',torch.cuda.device_count(),'CUDA devices available')
    for i in range(torch.cuda.device_count()):
      print(' Device',str(i),': ',torch.cuda.get_device_name(i))
    print('Selecting device 0..')
    device = torch.device('cuda')
  else:
    print('No CUDA devices available..selecting CPU')
    device = torch.device('cpu')

  # device = torch.device('cpu')
  print(f"device = {device}")
  # load trained model
  model = FallModel().to(device)
  # model.load_state_dict(torch.load(os.path.join(float_model,'4_fall_detection_model.pth')))
  model.load_state_dict(torch.load(float_model))

  # force to merge BN with CONV for better quantization accuracy
  optimize = 1

  # override batchsize if in test mode
  if (quant_mode=='test'):
    batchsize = 1
  
  # rand_in = torch.randn([batchsize, 200, 15, 3])
  rand_in = torch.randn([batchsize, 1, 15, 3])
  quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model) 
  quantized_model = quantizer.quant_model


  # data loader
  # Create the testing dataset
  # test_dataset = loadDataset('./grouped_data_test.json', window_size = 5)
  # test_dataset = loadDataset('./validation_data.json', window_size = 5)
  test_dataset = loadDataset('./validation_data_1fram.json', window_size = 5)

  # Define your training dataloader
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

  # evaluate 
  test(quantized_model, device, test_dataloader)


  # export config
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if quant_mode == 'test':
    quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
  
  return



def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d',  '--build_dir',  type=str, default='build',    help='Path to build folder. Default is build')
  ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
  ap.add_argument('-b',  '--batchsize',  type=int, default=1,        help='Testing batchsize - must be an integer. Default is 100')
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('PyTorch version : ',torch.__version__)
  print(sys.version)
  print(DIVIDER)
  print(' Command line options:')
  print ('--build_dir    : ',args.build_dir)
  print ('--quant_mode   : ',args.quant_mode)
  print ('--batchsize    : ',args.batchsize)
  print(DIVIDER)

  quantize(args.build_dir,args.quant_mode,args.batchsize)

  return



if __name__ == '__main__':
    run_main()

