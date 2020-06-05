"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-framerate", type=int, default=1,
                       help="frame rate - 1 by default")
    parser.add_argument("-streammode", type=str, default="video",
                        help="video or image - video by default")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def process_frame(frame, box, width, height):
    # get the coordinates of the rectangle for CV2
    x_ul = int(box[3] * width) # upper-left x
    y_ul = int(box[4] * height) # upper-left y
    x_br = int(box[5] * width) # bottom-right x
    y_br = int(box[6] * height) # bottom-right y
                    
    return cv2.rectangle(frame, (x_ul, y_ul), (x_br, y_br), (0,0,255), 1)

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, CPU_EXTENSION)
    net_input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    if cap.get(cv2.CAP_PROP_FPS):
        frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    elif cap.get(cv2.CAP_PROP_FPS):
        frames_per_second = args.framerate
    else:
        frames_per_second = 1
        

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    #out = cv2.VideoWriter('output_video.mp4', 0x00000021, 30, (width,height))
    
    # Define total_count variable
    total_count = 0
    frame_count = 0
    previous_frame_person_count = 0
    frame_rate = args.framerate
    #person_frames = 0
    #inference_time = 0
    
    # implement detection count history to mitigate 
    # fluctuations in the number of detections
    # nr_history is odd for correctness of median calculation
    nr_history = 35
    history = numpy.full(nr_history, 0)
    old_median = 0
    
    ssd_width = 300
    ssd_height = 300
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        # increase the frame count
        frame_count += 1
        
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (ssd_width, ssd_height))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        ### TODO: Start asynchronous inference for specified request ###
        #infer_network.async_inference(p_frame)
        #time_start = time.time()
        infer_network.exec_net(p_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            #time_end = time.time()
            #inference_time += time_end - time_start
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
            # this counts how many people we have detected in the current frame
            current_frame_person_count = 0
            
            for box in result[0,0,:,:]:
                # the highest class and probability of detected box
                box_class = int(box[1])
                box_score = box[2]
                
                if box_score > args.prob_threshold and box_class == 1:
                    # the box is for a person
                    #person_frames += 1
                    current_frame_person_count += 1
                    frame = process_frame(frame, box, width, height)
                
                # save info about number of detected people in this frame
                history[frame_count % nr_history] = current_frame_person_count
            
            
            # calculate the new median of the last nr_history results
            new_median = numpy.median(history)
            
            if new_median > old_median:
                # new persons have appeared
                total_count += int(new_median - old_median)
                start_time = time.time()
                start_frame = frame_count
            elif old_median > new_median:
                # someone left - send duration
                end_time = time.time()
                end_frame = frame_count
                duration = int(end_time - start_time)
                duration_f = int( (end_frame - start_frame) / frames_per_second )
                client.publish("person/duration", payload=json.dumps({"duration":duration}))
            
            # update the old_median for the next frame
            old_median = new_median
                
            #if current_frame_person_count > previous_frame_person_count:
            #    total_count += current_frame_person_count - previous_frame_person_count
                
            #previous_frame_person_count = current_frame_person_count
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            client.publish("person", payload=json.dumps({"count":current_frame_person_count, "total":total_count}))
            ### Topic "person/duration": key of "duration" ###
            # this is sent once we detect that someone has left

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        #out.write(frame)
        
        ### TODO: Write an output image if `single_image_mode` ###
        if args.streammode == "image":
            cv2.imwrite("output_image.png", frame)
    
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
    #out.release()
    
    ### TODO: Disconnect from MQTT
    client.disconnect()
    #print(person_frames)
    #print(inference_time)

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
