import cv2
from utils import preprocess
from model import build_model, get_seg
from tf_bisenet.BiSeNet_Loader import BiseNet_Loader
from SegProcessing import get_road_mask, get_bird_view, get_confident_vectors, get_steer
import time
import numpy as np

# model = get_seg()

# model.load_weights("/home/sonduong/catkin_ws/src/beginner_tutorials/scripts/models/bisenet/model_bisenet.h5")

model = BiseNet_Loader()

import os
path = "/home/sonduong/catkin_ws/src/lane_detect/src/img3/"

fnames = os.listdir(path)
fnames.sort()
i = 0 
# print(fnames)

def bird_view(img):
    IMAGE_H = 160
    IMAGE_W = 320

    src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[135, IMAGE_H], [185, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    # Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

    # img = cv2.imread('./test_img.jpg') # Read the test img
    img = img[100:(100+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping

    return warped_img * 2

def edge(img):
    result = cv2.Canny(img, 50, 70)
    return result
while(True):
    name  = fnames[i]
    frame = cv2.imread(path + name)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    start_time = time.time()
    pr_mask = model.predict(frame)
    print(get_steer(pr_mask))

    image_np = cv2.resize(frame, (320, 320))
    image = np.asarray(image_np)       # from PIL image to numpy array
    image = 2 * ((image - image.min()) / (image.max() - image.min()) ) - 1
    # image = preprocess(image) # apply the preprocessing
    image = np.array([image])       # the model expects 4D array

    # print(image.shape)

    
    cv2.imshow('cv_img', image_np)
    # cv2.imshow('road mask', (pr_mask[0,: ,:, 1]))
    # cv2.imshow('car mask', (pr_mask[0,: ,:, 0]) )
    # cv2.imshow('sign mask', (pr_mask[0,: ,:, 2]))

    # print(pr_mask.shape)
    # angle = float(model.predict(image, batch_size=1))
    # print(angle)

    print("--- %s seconds ---" % (time.time() - start_time))

    key = cv2.waitKey(0)
    # i+=1
    # print(key)
    if key == 27:
        break
    if (key == 81) or (key == 82):
        print("back")
        i = i - 1
        continue
    elif (key == 83) or (key == 84):
        i += 1
        print("next")
        continue
