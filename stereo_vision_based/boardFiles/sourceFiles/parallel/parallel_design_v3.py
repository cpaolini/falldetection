import os
import cv2 
import numpy as np
import time
import multiprocessing
import fcntl
import signal
import logging
import yolox_app_1600 as yolox
import a2j_app_1600 as a2j
import fallPredictor_app_CPU as fallPredictor
import csv

from ctypes import *
import cv2
import numpy as np
import xir
import vart
import threading
import queue

import torch
from boxes import postprocessing
import functools
import operator


import anchor as anchor
from vaitrace_py import vai_tracepoint
from datetime import datetime

'''
A2J related
'''
keypointsNumber = 15
cropWidth = 288
cropHeight = 288
Img_mean = 3
Img_std = 2
depthFactor = 50
imageOutputs = np.ones((cropHeight, cropWidth, 3), dtype="float32")
a2jInferenceComplete = 0
'''
YOLOX related
'''
decode_in_inference = True 
stridess=[8, 16, 32]
confidence_threshold = 0.7
nms_threshold = 0.45
yoloxInferenceComplete = 0

'''
FALL MODEL related
'''
pose_joints_list = []  # List to store poseJoints
predictionOutput = 0
prediction_list  = []  # List to store Fall Predictions
fields = ['LOG_timestamp', 'FrameCount', 'originalFrameTimestamp', 'yolox_original_frame_number', 'YoloxCount', 'yolox_timestamp', 'yolox_lag_timestamp', 'a2j_original_frame_number', 'A2JCount', 'a2j_timestamp', 'a2j_lag_timestamp','Prediction', 'fall_prediction_timestamp', 'fall_prediction_lag_timestamp']
filename = "/home/root/sourceFiles/prediction_log.csv"
csvfile = open(filename, 'w')
csvwriter = csv.writer(csvfile)
csvwriter.writerow(fields)
'''
Named PIPE 
'''
fifo_path = "/home/root/sourceFiles/myfifo"  
F_SETPIPE_SZ = 1031  # Linux 2.6.35+
F_GETPIPE_SZ = 1032  # Linux 2.6.35+

'''
Quantixed models
'''
YOLOX_MODEL = "/home/root/models/B1600/YOLOX_1600_QAT_vai3_kv260.xmodel"
A2J_MODEL = "/home/root/models/B1600/A2J_1600_kv260.xmodel"
xmodels = [YOLOX_MODEL, A2J_MODEL]

'''
system variables
'''
A2J_OTF = True
YOLOX_OTF = True
MSG = ["Loading xmodels and deserializing....", 
       "Creating VART runners for each DPU subgraphs....",
       "Create a thread for yolox",
       "Create a thread for Pose Estimation",
       "Start YOLOX Thread (yoloxThread)",
       "Start Pose Estimation Thread (a2jThread)",
       "Wait for Semaphore from PRODUCER",
       "Extract the color and depth frame from the decorated data packet",
       "Master Thread will run the YOLOX model with rgb image and yolox VART runner",
       "If the inference result of the YOLOX model returns Number of Box > 0 then enqueue the depth frame",
       "A2J thread will be waiting for depth frame, and immediately infer the pose using the depth frame and a2j VART runner",
       "After getting the pose information, run the FALL Detection Model"]


'''
Frame Details
'''
color_frame = np.zeros((480, 640, 3), dtype=np.uint8)
depth_frame = np.zeros((480, 640), dtype=np.uint16)
frame_width = 640
frame_height = 480
num_channels = 3
color_frame_size = frame_width * frame_height * num_channels
depth_frame_size = frame_width * frame_height * 2 
depth_frame_fifo = queue.Queue(maxsize=10)
color_depth_frame_fifo = queue.Queue(maxsize=10)

'''
Data Logging and inference video
'''
frame_count         = 0
count_a2j           = 0
count_yolox         = 0
a2j_original_frame_number   = 0
yolox_original_frame_number = 0
originalFrameTimestamp     = 0
a2j_timestamp = 0
frame_timestamp = 0
a2j_lag_timestamp = 0
yolox_timestamp = 0
yolox_lag_timestamp = 0
fall_prediction_timestamp = 0
fall_prediction_lag_timestamp = 0
previousCount_a2j   = 0; minCounter         = 0
previousCount_yolox = 0; minCounter_yolox   = 0
logging.basicConfig(filename='count_log_1600.log', level=logging.INFO, format='%(asctime)s - %(message)s')
saveVideo = True
yolox_poseVideo = cv2.VideoWriter('/home/root/sourceFiles/output/a2j/yolox_pose_1600.avi',cv2.VideoWriter_fourcc(*'MJPG'), 15, (640,480))
yoloxOutputVideo = cv2.VideoWriter('/home/root/sourceFiles/output/yolox/yolox.avi',cv2.VideoWriter_fourcc(*'MJPG'), 15, (640,480))
time_interval_100ms = 0.1  # 250 milliseconds
count_100ms_interval = 0
count_10s_interval = 0

'''
Thread Details
'''
threadnum = int(2)

'''
shared variables between threads
'''
numberOfBoxes = 0; boundaryBox = None

'''
graph and DPU runners
'''
graph           = []
subgraphs       = None
dpu_subgraphs   = []


print(f"{MSG[0]}")
for i in range(threadnum):
    graph.append(xir.Graph.deserialize(xmodels[i]))
    subgraphs = graph[i].get_root_subgraph().toposort_child_subgraph()
    for j in range(len(subgraphs)):
                if subgraphs[j].has_attr("device"):
                    if subgraphs[j].get_attr("device") == "DPU":
                        dpu_subgraphs.append(subgraphs[j])

print(f"{MSG[1]}")
yolox_dpu_runner = (vart.Runner.create_runner(dpu_subgraphs[0], "run"))
a2j_dpu_runner = (vart.Runner.create_runner(dpu_subgraphs[1], "run"))

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

# ''' Set up encoder DPU runner buffers & I/O mapping dictionary '''
# yolox_dict, yolox_inbuffer, yolox_outbuffer = init_dpu_runner(yolox_dpu_runner)
# a2j_dict, a2j_inbuffer, a2j_outbuffer = init_dpu_runner(a2j_dpu_runner)


def execute_async_method(dpu, tensor_buffers_dict):
    input_tensor_buffers = [tensor_buffers_dict[t.name] for t in dpu.get_input_tensors()]
    output_tensor_buffers = [tensor_buffers_dict[t.name] for t in dpu.get_output_tensors()]
    jid = dpu.execute_async(input_tensor_buffers, output_tensor_buffers)
    return dpu.wait(jid)

# def runYoloxInference(dpu_runner, yolox_dict, img):  
def runYoloxInference(dpu_runner, img):    
    '''    Thread worker function    '''
    
    ''' Set up encoder DPU runner buffers & I/O mapping dictionary '''
    yolox_dict, yolox_inbuffer, yolox_outbuffer = init_dpu_runner(dpu_runner)
    '''    initialize input and execute DPU runner    '''
    yolox_dict['YOLOX__YOLOX_QuantStub_quant_in__input_1_fix'] = img.reshape(tuple(yolox_dict['YOLOX__YOLOX_QuantStub_quant_in__input_1_fix'].shape[1:]))

    execute_async_method(dpu_runner, yolox_dict)
    
    # print(f"INFO: yolox execute_async_method DONE")
    out_q1 = yolox_dict['YOLOX__YOLOX_YOLOXHead_head__Cat_cat_list__ModuleList_0__inputs_3_fix'].copy()
    out_q2 = yolox_dict['YOLOX__YOLOX_YOLOXHead_head__Cat_cat_list__ModuleList_1__inputs_5_fix'].copy()
    out_q3 = yolox_dict['YOLOX__YOLOX_YOLOXHead_head__Cat_cat_list__ModuleList_2__inputs_fix'].copy()
    out_q1 = torch.as_tensor(out_q1.transpose(0,3,1,2))
    out_q2 = torch.as_tensor(out_q2.transpose(0,3,1,2))
    out_q3 = torch.as_tensor(out_q3.transpose(0,3,1,2))
    output = [out_q1,out_q2,out_q3]

    output = yolox.postprocess(output)
    
    output = postprocessing(output, num_classes = 1, conf_thre=confidence_threshold, nms_thre=nms_threshold, class_agnostic=False)
    #print("Done with the DPU runner")
    return output

def runYolox(image, dpu_runner, saveVideo):
    global ratio , count_a2j, count_yolox, frame_count, yoloxInferenceComplete

    ''' preprocess images '''    
    img = yolox.preprocess_fn(image)
    
    '''run inference '''
    outputs = runYoloxInference(dpu_runner, img)
    
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
        if saveVideo:
            result_image = yolox.visual(image, outputs[0], cls_conf = confidence_threshold)
            # cv2.imwrite("/home/root/sourceFiles/yoloxOutput_"+str(frame_count)+".jpeg", result_image)
            # yoloxOutputVideo.write(result_image)
    else:
        if saveVideo:
            result_image = image
    yoloxInferenceComplete = 1
    if saveVideo:
        return NumofBox, Bbox, result_image
    else:
        return NumofBox, Bbox

def runA2JInference(dpu_runner, img):
    '''    Thread worker function    '''
    ''' Set up encoder DPU runner buffers & I/O mapping dictionary '''
    a2j_dict, a2j_inbuffer, a2j_outbuffer = init_dpu_runner(dpu_runner)
    post_processing = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None)
          
    '''    initialize input and execute DPU runner    '''    
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

    #print("Done with the DPU runner")
    return pred_keypoints.data.numpy()

def runPoseEstimation(image, Bbox, dpu_runner, saveVideo):
    global count_a2j, count_yolox, frame_count, a2jInferenceComplete
    ''' preprocess images '''
    img = a2j.preprocess_fn(image, Bbox)

    '''run inference '''    
    outputs = runA2JInference(dpu_runner, img)
    # print("I'm here in A2J")
    a2jInferenceComplete = 1
    ''' post-processing '''
    if saveVideo:
        outputImage = a2j.pattingTest(image, outputs, Bbox, frame_count)
        # cv2.imwrite("/home/root/sourceFiles/poseOutput_1600_"+str(frame_count)+".jpeg", outputImage)
        outputs = a2j.convertOriginalScale(outputs, Bbox)
        return outputs, outputImage
    else:
        outputs = a2j.convertOriginalScale(outputs, Bbox)
        return outputs

def a2j_thread():   
    global numberOfBoxes, count_a2j, stop_a2j_event, count_yolox, frame_count, predictionOutput, csvwriter, a2jInferenceComplete, start_time,\
          a2j_original_frame_number,a2j_timestamp, a2j_lag_timestamp, fall_prediction_timestamp, fall_prediction_lag_timestamp
    
    while not stop_a2j_event.is_set() or not depth_frame_fifo.empty():
        try:
            if not depth_frame_fifo.empty():
                # print(f"Remaining items in depth_frame_fifo: {depth_frame_fifo.qsize()}")
                depth_frame, Bbox, original_frame_number, originalFrameTime = depth_frame_fifo.get()
                if saveVideo != True:
                    poseJoints = runPoseEstimation(depth_frame, Bbox, a2j_dpu_runner, saveVideo)
                else:
                    poseJoints, keypointJoints_image = runPoseEstimation(depth_frame, Bbox, a2j_dpu_runner, saveVideo)
                    yolox_poseVideo.write(keypointJoints_image)
                while(a2jInferenceComplete != 1):
                    pass
                a2jInferenceComplete = 0
                count_a2j = count_a2j + 1
                a2j_original_frame_number = original_frame_number
                # temp = float((datetime.now() - start_time).total_seconds() * 1000)
                temp = float((time.time() - start_time) * 1000)
                a2j_lag_timestamp = round(temp - originalFrameTime, 0)
                a2j_timestamp = round(temp, 0)


                '''Fall Prediction'''
                poseJoints = torch.from_numpy(poseJoints)
                poseJoints.resize_(1,1,15,3)
                predictionOutput = fallPredictor.runFallPredictor(poseJoints)
                prediction_list.append(predictionOutput)
                
                # temp = float((datetime.now() - start_time).total_seconds() * 1000)
                temp = float((time.time() - start_time) * 1000)
                fall_prediction_lag_timestamp = round(temp - originalFrameTime, 0)
                fall_prediction_timestamp = round(temp, 0)
                # csvwriter.writerow([timestamp, frame_count, count_yolox, count_a2j, predictionOutput])
            # else:
                # print(f"Waiting for depth frame...")
        
        except Exception as e:
            print(f"Error in A2J thread: {e}")

def yolox_thread():
    global count_a2j, stop_yolox_event, count_yolox, frame_count, predictionOutput, csvwriter, a2j_thread_h, yoloxInferenceComplete, A2J_OTF, \
        yolox_original_frame_number,originalFrameTimestamp, a2j_timestamp, yolox_timestamp, yolox_lag_timestamp
    
    while not stop_yolox_event.is_set() or not color_depth_frame_fifo.empty():
        try:
            if not color_depth_frame_fifo.empty():
                # print(f"Remaining items in depth_frame_fifo: {depth_frame_fifo.qsize()}")
                color_frame, depth_frame, originalFrameCount, originalFrameTimestamp = color_depth_frame_fifo.get()
                if saveVideo != True:
                        numberOfBoxes, boundaryBox = runYolox(color_frame, yolox_dpu_runner, saveVideo)
                else:
                    numberOfBoxes, boundaryBox, result_image = runYolox(color_frame, yolox_dpu_runner, saveVideo)
                    yoloxOutputVideo.write(result_image)

                while yoloxInferenceComplete != 1:
                    pass
                yoloxInferenceComplete = 0
                
                temp = float((time.time() - start_time) * 1000)
                yolox_lag_timestamp = round(temp - originalFrameTimestamp, 0)
                yolox_timestamp = round(temp, 0)

                if numberOfBoxes > 0:
                    count_yolox = count_yolox + 1
                    # yolox_timestamp = round(float((datetime.now() - originalFrameTimestamp).total_seconds() * 1000), 2)
                    yolox_original_frame_number = originalFrameCount
                    # temp = float((datetime.now() - start_time).total_seconds() * 1000)
                    # temp = float((time.time() - start_time) * 1000)
                    # yolox_lag_timestamp = round(temp - originalFrameTimestamp, 0)
                    # yolox_timestamp = round(temp, 0)

                    frame_tuple = (depth_frame, boundaryBox, yolox_original_frame_number, originalFrameTimestamp)
                    depth_frame_fifo.put(frame_tuple)# Put depth frame and its boundaryBox into FIFO
                    if A2J_OTF == True:
                        print(f"{MSG[5]}")
                        a2j_thread_h.start()
                        A2J_OTF = False
        
        except Exception as e:
            print(f"Error in YOLOX thread: {e}")

def signal_handler(signum, frame):
    global previousCount_a2j, minCounter, count_a2j, count_yolox, previousCount_yolox, frame_count, count_100ms_interval, count_10s_interval,\
        frame_timestamp,originalFrameTimestamp, yolox_timestamp, yolox_lag_timestamp, a2j_timestamp, a2j_lag_timestamp, fall_prediction_timestamp, fall_prediction_lag_timestamp

    if signum == signal.SIGALRM:
        # timestamp_100ms = round(float((datetime.now() - start_time).total_seconds() * 1000), 0)
        timestamp_100ms = round(float((time.time() - start_time) * 1000), 0)
        csvwriter.writerow([timestamp_100ms, frame_count, frame_timestamp, yolox_original_frame_number, count_yolox, yolox_timestamp, yolox_lag_timestamp, a2j_original_frame_number, count_a2j, a2j_timestamp, a2j_lag_timestamp, predictionOutput, fall_prediction_timestamp, fall_prediction_lag_timestamp])
        count_100ms_interval += 1

        # Check if 10-second interval is reached
        if count_100ms_interval % int(10 / time_interval_100ms) == 0:
            
            logging.info(f"frame count = {count_a2j - previousCount_a2j} per min \t{(count_a2j - previousCount_a2j)/60} fps")
            print(f"{minCounter}: frame count = {frame_count}\tYOLOX frame rate : {(count_yolox - previousCount_yolox)/10} fps\tA2J frame rate : {(count_a2j - previousCount_a2j)/10} fps", flush=True)
            previousCount_a2j = count_a2j
            previousCount_yolox = count_yolox
            minCounter = minCounter + 1

def main():
    global frame_count, count_a2j, count_yolox, saveVideo, pose_joints_list, numberOfBoxes, boundaryBox, A2J_OTF, YOLOX_OTF, \
    stop_a2j_event, predictionOutput, csvwriter, yoloxInferenceComplete, a2jInferenceComplete, stop_yolox_event, a2j_thread_h, start_time, frame_timestamp
    # start_time = datetime.now()
    start_time = time.time()
    print(f"start_time = {start_time}")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, 0.1, 0.1)
    # signal.setitimer(signal.ITIMER_VIRTUAL, time_interval_250ms, 0)
    print(f"{MSG[2]}")
    yolox_thread_h = threading.Thread(target=yolox_thread)
    stop_yolox_event = threading.Event()
    print(f"{MSG[3]}")
    a2j_thread_h = threading.Thread(target=a2j_thread)
    stop_a2j_event = threading.Event()
    # a2j_thread_h.start()
    try:
        with open(fifo_path, "rb") as fifo:
            semaphore = multiprocessing.Semaphore(1)
            while True:
                semaphore.acquire()
                frame_buffer = fifo.read(color_frame_size + depth_frame_size + 2)
                if frame_buffer and len(frame_buffer) == color_frame_size + depth_frame_size + 2 and frame_buffer[0] == 0xFF and frame_buffer[-1] == 0xEE:
                    frame_count = frame_count + 1
                    # frame_timestamp = round(float((datetime.now() - start_time).total_seconds() * 1000), 2)
                    frame_timestamp = round(float((time.time() - start_time) * 1000), 0)
                    color_frame_data = frame_buffer[1:color_frame_size+1]
                    depth_frame_data = frame_buffer[color_frame_size+1:-1]
                    color_frame = np.frombuffer(color_frame_data, dtype=np.uint8).reshape(480, 640, 3)
                    depth_frame = np.frombuffer(depth_frame_data, dtype=np.uint16).reshape(480, 640)
                    color_depth_frame_tuple = (color_frame, depth_frame, frame_count, frame_timestamp)
                    color_depth_frame_fifo.put(color_depth_frame_tuple)                    
                    if YOLOX_OTF == True:
                        print(f"{MSG[4]}")
                        yolox_thread_h.start()
                        YOLOX_OTF = False
                    # csvwriter.writerow([frame_count, count_yolox, count_a2j, predictionOutput])
                else:
                    print(f"Error: frame_buffer[0] = {frame_buffer[0]}\tframe_buffer[-1] = {frame_buffer[-1]}\tPipe size = {(fcntl.fcntl(fifo, F_GETPIPE_SZ))} Bytes")
                    print(f"frame_count         = {frame_count}")
                    print(f"count_yolox         = {count_yolox}")
                    print(f"count_a2j           = {count_a2j}")
                    print(f"prediction_count    = {len(prediction_list)}")
                semaphore.release()
                # csvwriter.writerow([frame_count, count_yolox, count_a2j, predictionOutput])
    except FileNotFoundError:
        print(f"FIFO '{fifo_path}' not found. Make sure the C++ PRODUCER program is running.")
    except Exception as e:
        print(f"An error occurred in main thread: {e}")
        # result.release()
        yoloxOutputVideo.release()
        # poseOutputVideo.release()
        yolox_poseVideo.release()
        cv2.destroyAllWindows()
        stop_yolox_event.set()
        stop_a2j_event.set()
        yolox_thread_h.join()
        a2j_thread_h.join()
        # csvwriter.writerow([frame_count, count_yolox, count_a2j, predictionOutput])
        csvfile.close()
        # del yolox_dpu_runner
        # del a2j_dpu_runner
        # print(f"Thread joined")
    finally:
        #if OTF == False:
        # a2j_thread_h.join()
        print(f"frame_count = {frame_count},\tcount_yolox = {count_yolox},\tcount_a2j = {count_a2j}\tprediction_count = {len(prediction_list)}")
        
        print(f"end_time = {time.time()}")
        print("DONE")

if __name__ == "__main__":
    main()
