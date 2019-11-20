#!/usr/bin/env python3
from __future__ import print_function

import roslib
roslib.load_manifest('beginner_tutorials')
import sys
# import tensorflow
import rospy
import cv2
import numpy as np
import message_filters
from std_msgs.msg import Float32
from sensor_msgs.msg import CompressedImage
import csv
import time

from utils import preprocess
# from tensorflow.keras import load_model
from model import build_model, get_seg

from DepthProcessing import get_sign




pub_steer = rospy.Publisher('team1/set_angle', Float32, queue_size=10)
pub_speed = rospy.Publisher('team1/set_speed', Float32, queue_size=10)

# pub_steer = rospy.Publisher('Team1_steerAngle', Float32, queue_size=10)
# pub_speed = rospy.Publisher('Team1_speed', Float32, queue_size=10)

rospy.init_node('control', anonymous=True)
msg_speed = Float32()
msg_speed.data = 50
msg_steer = Float32()
msg_steer.data = 0


depth_np = 0 
depth_hist = 0

from tf_bisenet.BiSeNet_Loader import BiseNet_Loader
from SegProcessing import get_steer, binary_sign

model = BiseNet_Loader()

msg_speed = Float32()
msg_speed.data = 20
msg_steer = Float32()
msg_steer.data = 0


smooth = cv2.imread("/home/sonduong/catkin_ws/src/beginner_tutorials/scripts/hi.png", 0)

# sign_rect = []
def bias_callback(depth_data):
    # global angle_bias
    # global depth_np
    # global depth_hist
    # global sign_rect
    depth_arr = np.fromstring(depth_data.data, np.uint8)
    depth_np = cv2.imdecode(depth_arr, cv2.IMREAD_GRAYSCALE)
    sign_rect = get_sign(depth_np)
    # depth_np = cv2.absdiff(depth_np, smooth)
    # depth_hist = get_collums_hist(depth_np[120: 220, :])
    # angle_bias = (get_weights(depth_np[120: 220, :])) 

    # cv2.waitKey(1)



start_time = time.time()

def drive_callback(rgb_data):
    global start_time
    global model

    if(time.time()-start_time > 0.01):

        np_arr = np.fromstring(rgb_data.data, np.uint8)
        # print(np_arr.shape)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        # cv2.imshow('cv_img', image_np)
        pr_mask = model.predict(image_np)
        # cv2.imshow('pr_mask', pr_mask)
        speed, angle = get_steer(image_np, pr_mask)
        msg_steer.data = float(angle)
        msg_speed.data = float(speed)
        pub_steer.publish(msg_steer)
        pub_speed.publish(msg_speed)
        # cv2.waitKey(1)
        # print(time.time() - start_time)
        start_time = time.time()

class sync_listener:
    def __init__(self):
        self.depth_sub = message_filters.Subscriber('team1/camera/depth/compressed', CompressedImage)
        self.image_sub = message_filters.Subscriber('team1/camera/rgb/compressed', CompressedImage)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], queue_size = 4,slop = 1) # Changed code
        self.ts.registerCallback(self.callback)

    def callback(self, image, depth):
        global start_time
        global model

        if(time.time()-start_time > 0.04):
            print("start processing ... ")


            # print("start processing rgb data... ")
            np_arr = np.fromstring(image.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            # print("start showing image ... ")
            # cv2.imshow('view', image_np)

            pr_mask = model.predict(image_np)
            speed, angle = get_steer(image_np, pr_mask)
            msg_steer.data = float(angle) 
            msg_speed.data = float(speed)
            pub_steer.publish(msg_steer)
            pub_speed.publish(msg_speed)
            # cv2.waitKey(1)
            # print(time.time() - start_time)
            start_time = time.time()


def listener():

    # rospy.Subscriber('team1/camera/depth/compressed', CompressedImage, bias_callback)
    rospy.Subscriber('team1/camera/rgb/compressed', CompressedImage, drive_callback,  buff_size=2**24)
    # rospy.Subscriber('Team1_image/compressed', CompressedImage, drive_callback,  buff_size=2**24)
    # ls = sync_listener()
    # depth_sub = message_filters.Subscriber('team1/camera/depth/compressed', CompressedImage)
    # image_sub = message_filters.Subscriber('team1/camera/rgb/compressed', CompressedImage)

    # ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
    # ts.registerCallback(callback)

    # spin() simply keeps python from exiting until this node is stopped
 
    rospy.spin()

if __name__ == '__main__':
    listener()
