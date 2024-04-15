import cv2 as cv
import numpy as np
import process_key_frame
import feature_process
import camera_motion_ransaclike
import constant_defination as const
import math
import time
import motion_compensation as mocom
img1 = cv.imread("images/f1.png")
img2 = cv.imread("images/rot45.png")
frame1 = process_key_frame.KeyFrame(img1.copy(), 0)
frame1.find_key_points()
frame2 = process_key_frame.KeyFrame(img2.copy(), 0)
frame2.find_key_points()
start_time = time.time()
matched_features = feature_process.match_features(frame1, frame2, True)
top_good_matches = feature_process.find_top_good_matches(frame1, frame2, matched_features, const.TOP_MATCHES, True)
if len(top_good_matches) > 0:
    del_x, del_y, del_theta= camera_motion_ransaclike.estimate_camera_motion(frame1, frame2, top_good_matches)
    mocom.motion_compensat(frame1, frame2, del_x, del_y, del_theta)
    print("Del X: ", del_x, " Del Y: ", del_y, " Del Theta: ", del_theta)
 
    # print(math.degrees(del_theta))
    # x=abs(math.degrees(del_theta))-45
    # print(x)
#cv.imshow("Original", img1)
#cv.imshow("Transformed", img2)
cv.waitKey(0)