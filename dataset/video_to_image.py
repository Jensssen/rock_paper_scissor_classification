# -*- coding: utf-8 -*-
"""
 
File:
    video_to_image.py
 
Authors: soe
Date:
    23.10.20
 
"""

import argparse
import cv2
import os

from datetime import datetime
import time


IMAGE_HEIGHT = 192
IMAGE_WIDTH = 144

def arg_pars():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True,  help="Path to video that you want to extract.")
    parser.add_argument("--image_counter", default=5, help="How many Video Frames to skip until next image is saved.")
    args = parser.parse_args()
    return args

def main(args):
    cap = cv2.VideoCapture(args.video_path)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    framecount = 0
    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if framecount%args.image_counter == 0:
                # resize image
                dt_object = datetime.fromtimestamp(time.time())
                resized = cv2.resize(frame, (IMAGE_HEIGHT,IMAGE_WIDTH), interpolation=cv2.INTER_AREA)
                print(os.path.join(os.path.dirname(args.video_path), f"{dt_object}.png"))
                cv2.imwrite(os.path.join(os.path.dirname(args.video_path), f"{dt_object}.png"), resized)


        else:
            break
        framecount += 1


if __name__ == '__main__':
    args = arg_pars()
    main(args)
