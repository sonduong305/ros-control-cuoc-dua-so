import cv2 
import numpy as np 
from keras.models import model_from_json
import tensorflow as tf 

# from scipy import signal
from LineIterator import createLineIterator, get_line_cor

json_file = open('/home/sonduong/WS/cuoc-dua-so/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("/home/sonduong/WS/cuoc-dua-so/model.h5")
print("Loaded model from disk")

graph = tf.get_default_graph()
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

def sign_classify(img_rgb, mask):
    global graph
    sign_rect = get_sign_mask(mask)
    result = -1
    if(sign_rect != -1):
        with graph.as_default():
            sign = img_rgb[sign_rect[1]: sign_rect[1] + sign_rect[3], sign_rect[0]: sign_rect[0] + sign_rect[2]]
            sign = cv2.resize(sign, (32, 32))
            
            sign = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)
            sign = np.array([np.reshape(sign, (32,32,1))])
            result = model.predict(sign, batch_size = 1) + 1
    return result
left_cors = get_line_cor((159, 159), (0, 130)).T
right_cors = get_line_cor((159, 160), (0, 190)).T
turn_cors = get_line_cor((150, 159), (0, 0)).T
turn_thresh = 0
angle_bias = 0.5
def start_turning(bird_view_img):
    # global angle_bias
    # global left_cors
    # global right_cors
    left_cors = get_line_cor((120, 159), (100, 0)).T
    right_cors = get_line_cor((120, 160), (100, 319)).T
    turn_cors = get_line_cor((150, 159), (0, 0)).T
    # angle_bias = 
    left =  np.where(bird_view_img[left_cors[0], left_cors[1]] == 0)
    right =  np.where(bird_view_img[right_cors[0], right_cors[1]] == 0)
    turn = np.where(bird_view_img[turn_cors[0], turn_cors[1]] == 0)
    if not len(left[0]) == 0:
        left = left[0][0]
    else:
        left = left_cors.shape[1]
    if not len(right[0]) == 0:
        right = right[0][0]
    else:
        right = right_cors.shape[1]
    if not len(turn[0]) == 0:
        turn = turn[0][0]
    else:
        turn = turn_cors.shape[1]
    # print(left, right)
    return left, right, turn

def get_confident_vectors(bird_view_img):
    angle = 15
    # discrete_line = get_line_cor((159, 159), (0, 140), bird_view_img)
    # print(discrete_line.T)
    left =  np.where(bird_view_img[left_cors[0], left_cors[1]] == 0)
    right =  np.where(bird_view_img[right_cors[0], right_cors[1]] == 0)
    # print(left[0][0])
    if not len(left[0]) == 0:
        left = left[0][0]
    else:
        left = left_cors.shape[1]
    if not len(right[0]) == 0:
        right = right[0][0]
    else:
        right = right_cors.shape[1]
    # print(left, right)
    l, r, t = start_turning(bird_view_img)
    left += 2.6 * l
    right += 2.6 * r
    turn = ((left / (left + right)) - angle_bias) * 2
    print(t)
    # print("")
    return - turn
    # line = cv2.line

def get_steer(img_rgb, mask):
    global turn_thresh
    frame = cv2.resize(mask, ( 320, 240)).astype(np.uint8)
    road_mask = get_road_mask(frame)
    bird_view = get_bird_view(road_mask)
    cv2.imshow("sign", bird_view)
    sign = sign_classify(cv2.resize(img_rgb, ( 320, 240)).astype(np.uint8), frame)
    if sign != -1:
        print("Co bien bao! " + str(sign))
        # turn =  np.where(bird_view[turn_cors[0], turn_cors[1]] == 0)

    #     if not len(turn[0]) == 0:;
    #         turn = turn[0][0]
    #     else:
    #         turn = turn_cors.shape[1]

    #     if turn_thresh == 0:
    #         turn_thresh = turn

    #     else:
    #         angle_bias = (turn / turn_thresh) - 1
    # print(angle_bias)
    
    return (get_confident_vectors(bird_view)) * 20


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