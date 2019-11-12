import cv2 
import numpy as np 
from keras.models import model_from_json
import tensorflow as tf 

# from scipy import signal
from LineIterator import createLineIterator, get_line_cor

graph = tf.get_default_graph()
json_file = open('/home/sonduong/WS/cuoc-dua-so/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("/home/sonduong/WS/cuoc-dua-so/model.h5")
with graph.as_default():
    model.predict(np.ones((1,32,32,3)))
print("Loaded model from disk")



def dynamic_speed(angle):
    return 70 - (abs(angle) * 5)
def get_bird_view(img):
    IMAGE_H = 160
    IMAGE_W = 320

    src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[135, IMAGE_H], [185, IMAGE_H], [0 - 20, 0], [IMAGE_W + 20, 0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    # Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

    # img = cv2.imread('./test_img.jpg') # Read the test img
    img = img[80:(80+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
    return warped_img
    # return cv2.resize(warped_img, (IMAGE_W * 2, IMAGE_H * 2))

def get_road_mask(img):
    road_color = (128, 64, 128)
    result = cv2.inRange(img, road_color, road_color)
    return result
def get_sign_mask(img):
    sign_color = (0, 220, 220)
    thresh = cv2.inRange(img, sign_color, sign_color)
    sign = -1
    # cv2.imshow("sign", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours) > 0):
        # print(cv2.contourArea(contours[0]))
        if cv2.contourArea(contours[0]) > 70:
            sign = cv2.boundingRect(contours[0])
    return sign
i = 0
def binary_sign(gray_sign):
    global graph
    global i
    with graph.as_default():        
        # sign = cv2.cvtColor(gray_sign, cv2.COLOR_RGB2GRAY)
        sign = np.array([np.reshape(gray_sign, (32,32,3))])
        result = model.predict(sign, batch_size = 1) + 1
    cv2.imwrite("/home/sonduong/catkin_ws/src/beginner_tutorials/scripts/sign_data_temp/" + str(i) + str(result) + ".png", gray_sign)
    i += 1
    return result
def sign_classify(img_rgb, mask):
    global graph
    sign_rect = get_sign_mask(mask)
    result = -1
    
    if(sign_rect != -1):
        with graph.as_default():
            sign = img_rgb[sign_rect[1]: sign_rect[1] + sign_rect[3], sign_rect[0]: sign_rect[0] + sign_rect[2]]
            sign = cv2.resize(sign, (32, 32))
            # sign = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("sign", sign)
            sign = np.array([sign])
            result = model.predict(sign, batch_size = 1) + 1
    return result
left_cors = get_line_cor((159, 159), (0, 130)).T
right_cors = get_line_cor((159, 160), (0, 190)).T
sub_left_cors = get_line_cor((120, 159), (100, 0)).T
sub_right_cors = get_line_cor((120, 160), (100, 319)).T
turn_thresh = 0
angle_bias = 0.5
turn_dis_graph = []
turn_dis_cur = 0
def turn_status():
    global turn_dis_graph
    global turn_dis_cur
    mean = np.array(turn_dis_graph).mean()
    if (mean - 10) < turn_dis_cur < (mean + 10):
        return 0
    elif turn_dis_cur >= mean + 10:
        return 1
    elif turn_dis_cur <= mean -10:
        return 2
def get_distance_to_obstacles(vector, line):
    if not len(vector[0]) == 0:
        return vector[0][0]
    else:
        return line.shape[1]

def start_turning(bird_view_img, direction):
    # global angle_bias
    # global left_cors
    # global right_cors
    dir = (direction + 1) * 159
    left_cors = get_line_cor((139, 159), (90, dir)).T
    right_cors = get_line_cor((159, 160), (120, dir)).T
    # turn_cors = get_line_cor((90, 159), (70, 0)).T
    # angle_bias = 
    left =  np.where(bird_view_img[left_cors[0], left_cors[1]] == 0)
    right =  np.where(bird_view_img[right_cors[0], right_cors[1]] == 0)
    # turn = np.where(bird_view_img[turn_cors[0], turn_cors[1]] == 0)


    left = get_distance_to_obstacles(left, left_cors)
    right = get_distance_to_obstacles(right, right_cors)
    if left == 0:
        left = left_cors.shape[1]
    turn = ((left / (left + right)) - angle_bias) * 2
    # print(left, right)
    return turn * 60

def get_confident_vectors(bird_view_img):
    angle = 15
    # discrete_line = get_line_cor((159, 159), (0, 140), bird_view_img)
    # print(discrete_line.T)
    left =  np.where(bird_view_img[left_cors[0], left_cors[1]] == 0)
    right =  np.where(bird_view_img[right_cors[0], right_cors[1]] == 0)
    sub_left =  np.where(bird_view_img[sub_left_cors[0], sub_left_cors[1]] == 0)
    sub_right =  np.where(bird_view_img[sub_right_cors[0], sub_right_cors[1]] == 0)


    left = get_distance_to_obstacles(left, left_cors)
    right = get_distance_to_obstacles(right, right_cors)
    sub_left = get_distance_to_obstacles(sub_left, sub_left_cors)
    sub_right = get_distance_to_obstacles(sub_right, sub_right_cors)


    # l, r, t = start_turning(bird_view_img)
    left += 3 * sub_left
    right += 3 * sub_right
    turn = ((left / (left + right)) - angle_bias) * 2


    return - turn

turning_frame = 0
turn_dir = 0
def get_steer(img_rgb, mask):
    # global turn_thresh
    global turning_frame
    global turn_dir
    frame = cv2.resize(mask, ( 320, 240)).astype(np.uint8)
    road_mask = get_road_mask(frame)
    bird_view = get_bird_view(road_mask)
    cv2.imshow("mask", mask)
    cv2.imshow("bird_view", bird_view)
    angle = (get_confident_vectors(bird_view)) * 20
    speed = dynamic_speed(angle)



    sign = sign_classify(cv2.resize(img_rgb, ( 320, 240)).astype(np.uint8), frame)
    if sign != -1:
        print("Co bien bao! " + str(sign))
        sign = sign[0][0]
        sign = int(round(sign))
        turning_frame = 35
        speed = 30
        turn_dir = sign - 1
        if turn_dir == 0:
            turn_dir = -1 

    if turning_frame > 0:
        #turning ...
        turning_frame -= 1
        angle_bias = (start_turning(bird_view , turn_dir))
        print(angle_bias)
        if angle_bias < 0:
            angle_bias = 0
        angle = (angle_bias * turn_dir) + angle
        speed = 30
    else:
        turn_dir = 0
    
    
    return  speed, angle

# import os
# path = "/home/sonduong/Documents/data_segment/GT/"
# fnames = os.listdir(path)
# fnames.sort()
# i = 0 
# road_color = [128, 64, 128]

# while(True):
#     name  = fnames[i]
#     frame = cv2.imread(path + name, 1)
#     road_mask = get_road_mask(frame)
#     sign_classify(frame, frame)
#     print(get_steer(frame, frame))
#     bird_view = get_bird_view(road_mask)
#     # print(get_confident_vectors(bird_view))
#     cv2.imshow("Frame", frame)
#     cv2.imshow("Mask", bird_view)
#     key = cv2.waitKey(0)
#     # i+=1
#     # print(key)
#     if key == 27:
#         break
#     if (key == 81) or (key == 82):
#         # print("back")
#         i = i - 1
#         continue
#     elif (key == 83) or (key == 84):
#         i += 1
#         # print("next")
#         continue