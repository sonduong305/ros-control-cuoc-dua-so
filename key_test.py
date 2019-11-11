#!/usr/bin/env python3
from pynput import keyboard
import time
import rospy
from std_msgs.msg import Float32

class MyListener(keyboard.Listener):
    def __init__(self):
        super(MyListener, self).__init__(self.on_press, self.on_release)
        self.go = None
        self.right = None
        self.left = None
    def on_press(self, key):
        if key == keyboard.Key.up:
            self.go = True
        if key == keyboard.Key.right:
            self.right = True
        if key == keyboard.Key.left:
            self.left = True
    def on_release(self, key):
        if key == keyboard.Key.up:
            self.go = False
        if key == keyboard.Key.right:
            self.right = False
        if key == keyboard.Key.left:
            self.left = False

msg_speed = Float32()
msg_speed.data = 50
msg_steer = Float32()
pub_steer = rospy.Publisher('team1/set_angle', Float32, queue_size=10)
pub_speed = rospy.Publisher('team1/set_speed', Float32, queue_size=10)
rospy.init_node('control', anonymous=True)
right_accelerate = 0
left_accelerate = 0
amount = 5
rate = 0.1

listener = MyListener()
listener.start()

started = False

while True:
    time.sleep(rate)
    if listener.go == True:
        msg_speed.data = 50
        pub_speed.publish(msg_speed)
    elif listener.go == False:
        msg_speed.data = 25
        pub_speed.publish(msg_speed)
    if listener.right == True:
        left_accelerate = 0
        msg_steer.data = amount + right_accelerate
        right_accelerate = right_accelerate + rate * 13
        pub_steer.publish(msg_steer)
    elif listener.left == True:
        right_accelerate = 0
        msg_steer.data = -(amount + left_accelerate)
        left_accelerate = left_accelerate + rate * 13
        pub_steer.publish(msg_steer)
    if listener.right == False and listener.left == False:
        msg_steer.data = 0
        pub_steer.publish(msg_steer)
        right_accelerate = 0
        left_accelerate = 0
	
	
        
