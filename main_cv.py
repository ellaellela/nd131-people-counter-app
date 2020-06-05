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
"""
this file has been adjusted using an example given at:
https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
"""

import os
import sys
import time
import socket
import json
import cv2

import numpy as np
import tensorflow as tf

import logging as log

from argparse import ArgumentParser

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an saved_model.pb file with a trained model.")
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



def process_frame(frame, box, width, height):
    # get the coordinates of the rectangle for CV2
    x_ul = int(box[3] * width) # upper-left x
    y_ul = int(box[4] * height) # upper-left y
    x_br = int(box[5] * width) # bottom-right x
    y_br = int(box[6] * height) # bottom-right y
                    
    return cv2.rectangle(frame, (x_ul, y_ul), (x_br, y_br), (0,0,255), 1)

def infer_on_stream(args, client=None):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    model_dir = args.model

    # Read the graph.
    with tf.gfile.FastGFile(model_dir + 'frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())


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
    out = cv2.VideoWriter('output_video_cv2.mp4', 0x00000021, 30, (width,height))
    
    # Define total_count variable
    total_count = 0
    frame_count = 0
    previous_frame_person_count = 0
    frame_rate = args.framerate
    #person_frames = 0
    #inference_time = 0
    
    ssd_width = 300
    ssd_height = 300
    
    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    
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
            img = frame #cv2.imread(frame)
            rows = img.shape[0]
            cols = img.shape[1]
            inp = cv2.resize(img, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
            
            #time_start = time.time()
            # Run the model
            output = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                           feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
            #time_end = time.time()
            #inference_time += time_end - time_start
            
            # Visualize detected bounding boxes.
            num_detections = int(output[0][0])
            for i in range(num_detections):
                classId = int(output[3][0][i])
                score = float(output[1][0][i])
                bbox = [float(v) for v in output[2][0][i]]
                if score > prob_threshold and classId == 1:
                    #person_frames += 1
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

            ### TODO: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()

            #out.write(img)
            #cv2.imshow(img)
            #cv2.imwrite("output_image.png", img)

            ### TODO: Write an output image if `single_image_mode` ###
            #if args.streammode == "image":
            #    cv2.imwrite("output_image.png", frame)
    
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
    out.release()
    #print(person_frames)
    #print(inference_time)

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    infer_on_stream(args)


if __name__ == '__main__':
    main()
