import cv2 
import numpy as np 
from scipy import signal

left_weights = [np.arange(0, 160, 1)]
left_weights = np.repeat(left_weights, 240, axis= 0)
# right_weights = 
# right_weights = [np.arange(160, 0, -1)]
# right_weights = np.repeat(left_weights, 240, axis= 0)
right_weights = np.flip(left_weights, axis= 1)

def getGradientMagnitude(im):
    "Get magnitude of gradient for given image"
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(im, ddepth, 1, 0)
    dy = cv2.Sobel(im, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    return mag

def get_hist(hist, bins):

    imhist = np.zeros((240,320))
    for _hist, _bin in zip(hist[1:], bins[1:]):
        cv2.line(imhist, (int(_bin), 0), (int(_bin), int(_hist)), 255, thickness=1)
    imhist = cv2.rotate(imhist, cv2.ROTATE_180)
    imhist = cv2.flip( imhist, 1 )
    return imhist

def smooth_hist(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_collums_hist(img):
    # img = img[50:, :]
    hist = np.std(img, axis= 0) * 5
    bins = np.arange(0, 320, 1)
    hist = smooth_hist(hist, 5)
    # print(hist.shape)
    # print(bins.shape)
    return get_hist(hist, bins)
def median(hist):
    sum = hist.sum() / 255
    # print(sum) 
    temp = 0
    for col in range(20, 300):
        temp += hist[:, col].sum() / 255
        if temp > sum / 2:
            break
    return col

def get_peaks(img):
    a = np.std(img, axis= 0) * 3
    # a = smooth_hist(a, 3)
    bins = np.arange(0, 320, 1)
    cv2.imshow("hist 2", get_hist(a, bins))
    peaks = np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True]
    result = []
    for i, peak in enumerate(peaks):
        if peak:
            if a[i] > a.mean() * 2:
                result.append(i)
    result = np.array(result)
    # print(result.shape)
    
    
    left_peak = result[result < 160].mean()
    if np.isnan(left_peak):
        left_peak = 0
    right_peak = result[result > 160].mean()
    if np.isnan(right_peak):
        right_peak = 320

    return (left_peak + right_peak )/ 2

def get_weights(img):
    a = np.std(img, axis= 0) * 3
    # a = smooth_hist(a, 3)
    bins = np.arange(0, 320, 1)
    hist = get_hist(a, bins)
    left = np.multiply(left_weights, hist[:, :160] // 255)
    right = np.multiply(right_weights, hist[:, 160:] // 255)
    # right /= 255
    # cv2.imshow("hist 2", hist)
    # print(right.sum() / left.sum())
    return ((right.sum() / left.sum()) - 1 ) * - 15


import os
path = "/home/sonduong/catkin_ws/src/lane_detect/src/img6/"
smooth = cv2.imread("/home/sonduong/catkin_ws/src/beginner_tutorials/scripts/hi.png", 0)
fnames = os.listdir(path)
fnames.sort()
i = 0 
# np.median()
while(True):
    name  = fnames[i]
    frame = cv2.imread(path + name, 0)
    frame_r = cv2.absdiff(frame, smooth)
    hist = get_collums_hist(frame[130: 220, :])
    hist_r = get_collums_hist(frame_r[130 : 220, :])
    lap = getGradientMagnitude(frame)
    lap *= 2
    lap = np.where(lap < 255, lap, 255)
    cv2.imshow('depth', frame_r)
    cv2.imshow("hist", lap)
    cv2.imshow("hist_r", hist_r)
    print(get_weights(frame_r[120: 220, :]))


    
    key = cv2.waitKey(0)
    # i+=1
    # print(key)
    if key == 27:
        break
    if (key == 81) or (key == 82):
        # print("back")
        i = i - 1
        continue
    elif (key == 83) or (key == 84):
        i += 1
        # print("next")
        continue