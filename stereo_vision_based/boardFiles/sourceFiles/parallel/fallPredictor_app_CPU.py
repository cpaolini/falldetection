import os
import time
import json
import torch
from model import FallModel
from vaitrace_py import vai_tracepoint


float_model = "/home/root/sourceFiles/fallDetectorCPU/float_model/61_fall_detection_model.pth"
device = torch.device('cpu')

model = FallModel().to(device)
model.load_state_dict(torch.load(float_model, map_location=torch.device('cpu')))
model.eval()

@vai_tracepoint
def runFallPredictor(keypoint):
    keypoint = keypoint.to(device)
    with torch.no_grad():
        output = model(keypoint)
        pred = torch.argmax(output, dim=1)
        # print(f"pred = {pred.item()}")
    return pred.item()