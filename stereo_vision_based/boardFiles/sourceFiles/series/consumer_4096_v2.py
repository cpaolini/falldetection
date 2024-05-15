import os
import cv2 
import numpy as np
import time
import multiprocessing
import fcntl
import signal
import logging
import yolox_app_4096 as yolox
import a2j_app_4096 as a2j
import fallDetector_app_4096_vai3_1frame as fallPredictor
import csv

logging.basicConfig(filename='count_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')

YOLOX_MODEL = "/home/root/models/B4096/YOLOX_4096_QAT_kv260.xmodel"
A2J_MODEL = "/home/root/models/B4096/A2J_4096_kv260.xmodel"
FALL_MODEL = "/home/root/models/B4096/FD_1Frame_4096_kv260.xmodel"
NumofBox = 0
Bbox = 0
F_SETPIPE_SZ = 1031  # Linux 2.6.35+
F_GETPIPE_SZ = 1032  # Linux 2.6.35+

fifo_path = "/home/root/sourceFiles/myfifo"  
# result = cv2.VideoWriter('color_frame.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15, (640,480))
# yoloxOutputVideo = cv2.VideoWriter('/home/root/sourceFiles/output/yolox/yolox.avi',cv2.VideoWriter_fourcc(*'MJPG'), 1, (640,480))
# poseOutputVideo = cv2.VideoWriter('/home/root/sourceFiles/output/a2j/pose.avi',cv2.VideoWriter_fourcc(*'MJPG'), 1, (640,480))
yolox_poseVideo = cv2.VideoWriter('/home/root/sourceFiles/output/a2j/yolox_pose.avi',cv2.VideoWriter_fourcc(*'MJPG'), 1, (1280,480))
fields = ['LOG_timestamp', 'FrameCount', 'originalFrameTimestamp', 'yolox_original_frame_number', 'YoloxCount', 'yolox_timestamp', 'yolox_lag_timestamp', 'a2j_original_frame_number', 'A2JCount', 'a2j_timestamp', 'a2j_lag_timestamp','Prediction', 'fall_prediction_timestamp', 'fall_prediction_lag_timestamp']
filename = "/home/root/sourceFiles/series/prediction_log_serial.csv"
csvfile = open(filename, 'w')
csvwriter = csv.writer(csvfile)
csvwriter.writerow(fields)

prediction_list  = []  # List to store Fall Predictions
predictionOutput = 0

color_frame = np.zeros((480, 640, 3), dtype=np.uint8)
depth_frame = np.zeros((480, 640), dtype=np.uint16)
frame_width = 640
frame_height = 480
num_channels = 3
color_frame_size = frame_width * frame_height * num_channels
depth_frame_size = frame_width * frame_height * 2 

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
previousCount = 0
minCounter = 0
saveVideo = False
time_interval_100ms = 0.1  # 100 milliseconds
count_100ms_interval = 0

def signal_handler(signum, frame):
    global saveVideo, predictionOutput, \
        frame_count, count_yolox, yolox_original_frame_number, count_a2j, a2j_original_frame_number, \
        start_time, frame_timestamp, yolox_timestamp, yolox_lag_timestamp, a2j_timestamp, a2j_lag_timestamp, fall_prediction_timestamp, fall_prediction_lag_timestamp,\
        previousCount, minCounter, count_100ms_interval, previousCount_yolox, previousCount_a2j
    if signum == signal.SIGALRM:
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
            # logging.info(f"frame count = {frame_count - previousCount} per min \t{(frame_count - previousCount)/60} fps")
            # # print(f"{minCounter}: frame count = {count - previousCount} per min \t{(count - previousCount)/60} fps", flush=True)
            # print(f"{minCounter}: frame rate : {(frame_count - previousCount)/10} fps", flush=True)
            # previousCount = frame_count
            # minCounter = minCounter + 1
    
def main():
    global saveVideo, predictionOutput, \
        frame_count, count_yolox, yolox_original_frame_number, count_a2j, a2j_original_frame_number, \
        start_time, frame_timestamp, yolox_timestamp, yolox_lag_timestamp, a2j_timestamp, a2j_lag_timestamp, fall_prediction_timestamp, fall_prediction_lag_timestamp
    # signal.setitimer(signal.ITIMER_REAL, 1, 10)
    start_time = time.time()
    print(f"start = {start_time}")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, 0.1, 0.1)
    try:
        with open(fifo_path, "rb") as fifo:
            semaphore = multiprocessing.Semaphore(1)
            while True:
                semaphore.acquire()
                frame_buffer = fifo.read(color_frame_size + depth_frame_size + 2)
                if frame_buffer and len(frame_buffer) == color_frame_size + depth_frame_size + 2 and frame_buffer[0] == 0xFF and frame_buffer[-1] == 0xEE:
                    frame_count = frame_count + 1
                    frame_timestamp = round(float((time.time() - start_time) * 1000), 0)
                    color_frame_data = frame_buffer[1:color_frame_size+1]
                    depth_frame_data = frame_buffer[color_frame_size+1:-1]
                    color_frame = np.frombuffer(color_frame_data, dtype=np.uint8).reshape(480, 640, 3)
                    depth_frame = np.frombuffer(depth_frame_data, dtype=np.uint16).reshape(480, 640)
                    if saveVideo != True:
                        NumofBox, Bbox = yolox.runYolox(color_frame, YOLOX_MODEL, frame_count, saveVideo)
                        # print(f"NumofBox = {NumofBox}")
                    else:
                        NumofBox, Bbox, result_image = yolox.runYolox(color_frame, YOLOX_MODEL, frame_count, saveVideo)
                        # yoloxOutputVideo.write(result_image)
                    if NumofBox > 0:
                        count_yolox = count_yolox + 1
                        yolox_original_frame_number = frame_count
                        temp = float((time.time() - start_time) * 1000)
                        yolox_lag_timestamp = round(temp - frame_timestamp, 0)
                        yolox_timestamp = round(temp, 0)
                        NumofBox = 0
                        if saveVideo != True:
                            poseJoints = a2j.runPoseEstimation(depth_frame, Bbox, A2J_MODEL, frame_count, saveVideo)
                        else:
                            poseJoints, keypointJoints_image = a2j.runPoseEstimation(depth_frame, Bbox, A2J_MODEL, frame_count, saveVideo)
                            # poseOutputVideo.write(keypointJoints_image)                        
                            yolox_pose = np.hstack((result_image, keypointJoints_image))
                            yolox_poseVideo.write(yolox_pose)

                        count_a2j = count_a2j + 1
                        a2j_original_frame_number = frame_count
                        temp = float((time.time() - start_time) * 1000)
                        a2j_lag_timestamp = round(temp - frame_timestamp, 0)
                        a2j_timestamp = round(temp, 0)
                        predictionOutput = fallPredictor.runFallPredictor(poseJoints)
                        prediction_list.append(predictionOutput)

                        temp = float((time.time() - start_time) * 1000)
                        fall_prediction_lag_timestamp = round(temp - frame_timestamp, 0)
                        fall_prediction_timestamp = round(temp, 0)
                        # if predictionOutput:
                        #     print("***Fall detected!***")
                        # else:
                        #     print("ALL OKAY")
                else:
                    print(f"Error: frame_buffer[0] = {frame_buffer[0]}\tframe_buffer[-1] = {frame_buffer[-1]}\tPipe size = {(fcntl.fcntl(fifo, F_GETPIPE_SZ))} Bytes")
                    print(f"frame_count         = {frame_count}")
                    print(f"count_yolox         = {count_yolox}")
                    print(f"count_a2j           = {count_a2j}")
                    print(f"prediction_count    = {len(prediction_list)}")
                semaphore.release()
    except FileNotFoundError:
        print(f"FIFO '{fifo_path}' not found. Make sure the C++ PRODUCER program is running.")
    except Exception as e:
        print(f"An error occurred: {e}")
        # result.release()
        # yoloxOutputVideo.release()
        # poseOutputVideo.release()
        yolox_poseVideo.release()
        cv2.destroyAllWindows()
        csvfile.close()
    finally:
        print(f"frame_count = {frame_count},\tcount_yolox = {count_yolox},\tcount_a2j = {count_a2j}\tprediction_count = {len(prediction_list)}")
        print(f"end_time = {time.time()}")
        print("DONE")

if __name__ == "__main__":
    main()
