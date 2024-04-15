
import constant_defination as const
import math
import numpy as np
import random as rn
import cv2 as cv

# Estimate the camera motion from matched features
def estimate_camera_motion(frame_1, frame_2, top_matches):
   
    N_Matches = len(top_matches)                            # N observations
    ksamples = select_krandom_samples(N_Matches)
    prev_kp = frame_1.get_key_points()
    print(type(prev_kp))
    cur_kp = frame_2.get_key_points()
    prev_pts = []
    cur_pts = []
    for i in range(N_Matches):                           # Select N observation from set of keypoints
        prev_pts.append(prev_kp[top_matches[i].queryIdx].pt)
        cur_pts.append(cur_kp[top_matches[i].trainIdx].pt)
    # print("prev_pts ",prev_pts[2])
    # print("prev_pts ",prev_pts)
    # print("cur_pts ",cur_pts)
    also_inliers_idx = []
    #maybe_inliers_prev = []
    #maybe_inliers_cur = []
    #remain_prev_pts = []
    #remain_cur_pts = []
    also_inliers = 0
    final_del_x, final_del_y, final_del_theta = 0.0, 0.0, 0.0
    min_sum_sqr_dist = 99999999
    for i in range(len(ksamples)):
        maybe_inliers_idx = [ksamples[i][0], ksamples[i][1]]
        maybe_inliers_prev = [prev_pts[maybe_inliers_idx[0]], prev_pts[maybe_inliers_idx[1]]]
        maybe_inliers_cur = [cur_pts[maybe_inliers_idx[0]], cur_pts[maybe_inliers_idx[1]]]
        if((maybe_inliers_prev[0] == maybe_inliers_cur[0]) and (maybe_inliers_prev[1] == maybe_inliers_cur[1])):
            continue
        remain_prev_pts = prev_pts.copy()           # [x for x in prev_pts if prev_pts.index(x) not in idx]
        remain_cur_pts = cur_pts.copy()             # [x for x in cur_pts if cur_pts.index(x) not in idx]
        del remain_prev_pts[maybe_inliers_idx[0]]                 # Remove the first point
        del remain_cur_pts[maybe_inliers_idx[0]]
        if(maybe_inliers_idx[1] > maybe_inliers_idx[0]):
            del remain_prev_pts[maybe_inliers_idx[1] - 1]             # Remove the second point, the index is reduced by 1 by the previous del
            del remain_cur_pts[maybe_inliers_idx[1] - 1]
        elif(maybe_inliers_idx[1] < maybe_inliers_idx[0]):
            del remain_prev_pts[maybe_inliers_idx[1]]
            del remain_cur_pts[maybe_inliers_idx[1]]
        else:
            print("Same index is selected::")
        prev_dist = math.dist(maybe_inliers_prev[0], maybe_inliers_prev[1])
        cur_dist = math.dist(maybe_inliers_cur[0], maybe_inliers_cur[1])
        #prev_dist = math.sqrt(((prev_pts_pair[0][0] - prev_pts_pair[0][1]) + (prev_pts_pair[1][0] - prev_pts_pair[1][1]))**2)
        #cur_dist = math.sqrt(((cur_pts_pair[0][0] - cur_pts_pair[0][1]) + (cur_pts_pair[1][0] - cur_pts_pair[1][1]))**2)
        #if prev_dist == cur_dist:
        #    continue
        # print(prev_dist)
        # print(cur_dist)
        # print(math.fabs(prev_dist - cur_dist))
        if (math.fabs(prev_dist - cur_dist)) < const.DIST_POINT_PAIR_THRES:
            del_x, del_y, del_theta = calculate_ego_motion(maybe_inliers_prev, maybe_inliers_cur)
            transformed_pts = calculate_transformed_points(remain_cur_pts, del_x, del_y, del_theta)
            num_inliers, inliers_idx = calculate_inliers(remain_prev_pts, transformed_pts)
           
            if num_inliers > also_inliers:
                also_inliers = num_inliers
                also_inliers_idx = inliers_idx
                #final_del_x, final_del_y, final_del_theta = del_x, del_y, del_theta
        if also_inliers > const.NUM_INLIERS:
            inliers_prev  = [remain_prev_pts[i] for i in also_inliers_idx]
            inliers_cur = [remain_cur_pts[i] for i in also_inliers_idx]
            inliers_prev.append(maybe_inliers_prev[0])
            inliers_prev.append(maybe_inliers_prev[1])
            inliers_cur.append(maybe_inliers_cur[0])
            inliers_cur.append(maybe_inliers_cur[1])
            #final_del_x, final_del_y, final_del_theta = calculate_ego_motion(inliers_prev, inliers_cur)
            del_x, del_y, del_theta = calculate_ego_motion(inliers_prev, inliers_cur)
            transformed_pts = calculate_transformed_points(cur_pts, final_del_x, final_del_y, final_del_theta)
            sum_sqr_dist =  calculate_sum_squared_distance(prev_pts, transformed_pts)
            if sum_sqr_dist < min_sum_sqr_dist:
                min_sum_sqr_dist = sum_sqr_dist
                print("Also Inliers: ", also_inliers, " Sum Squared Dist: ", min_sum_sqr_dist)
                final_del_x, final_del_y, final_del_theta = del_x, del_y, del_theta
        else:
            final_del_x, final_del_y, final_del_theta = 0.0, 0.0, 0.0
    return final_del_x, final_del_y, final_del_theta
    

# Calculate ego-motion from points correspondances
def calculate_ego_motion(prev_pts, cur_pts):
    s_xt_1, s_yt_1, s_xt, s_yt = 0.0, 0.0, 0.0, 0.0
    s_xt_1xt, s_xt_1yt, s_xt_yt_1, s_yt_1yt = 0.0, 0.0, 0.0, 0.0
    np = len(cur_pts)
    for i in range(np):
        s_xt_1 += prev_pts[i][0]
        s_yt_1 += prev_pts[i][1]
        s_xt += cur_pts[i][0]
        s_yt += cur_pts[i][1]
        s_xt_1xt += prev_pts[i][0]*cur_pts[i][0]
        s_xt_1yt += prev_pts[i][0]*cur_pts[i][1]
        s_xt_yt_1 += cur_pts[i][0]*prev_pts[i][1]
        s_yt_1yt += prev_pts[i][1]*cur_pts[i][1]
    numerator = np*s_xt_1yt - np*s_xt_yt_1 - s_xt_1*s_yt + s_yt_1*s_xt
    denominator = np*s_xt_1xt + np*s_yt_1yt - s_xt_1*s_xt - s_yt_1*s_yt
    if denominator == 0:
        denominator = 0.000000001
    del_theta = math.atan(numerator/denominator)
    del_x = (s_xt_1 - s_xt*math.cos(del_theta) - s_yt*math.sin(del_theta))/np
    del_y = (s_yt_1 + s_xt*math.sin(del_theta) - s_yt*math.cos(del_theta))/np
    return del_x, del_y, del_theta,


# Calculate the transformed points positions for the current frame
def calculate_transformed_points(rem_cur_pts, delx, dely, deltheta):
    transformed_pts = []
    rot_mat = np.asmatrix(((math.cos(deltheta), math.sin(deltheta)), (-math.sin(deltheta), math.cos(deltheta))))
    for i in range(len(rem_cur_pts)):
        xt_1 = rem_cur_pts[i][0]*rot_mat[0,0] + rem_cur_pts[i][1]*rot_mat[0,1] + delx
        yt_1 = rem_cur_pts[i][0]*rot_mat[1,0] + rem_cur_pts[i][1]*rot_mat[1,1] + dely
        transformed_pts.append((xt_1, yt_1))
    return transformed_pts

outliers=0
# Calculate the Number of inliers for the transformed Points
def calculate_inliers(prev_pts, trans_pts):
    num_inliers = 0
    inliers_idx = []
    for i in range(len(prev_pts)):
        square_dist = (prev_pts[i][0] - trans_pts[i][0])**2 + (prev_pts[i][1] - trans_pts[i][1])**2
        if square_dist < const.T_THRES:
            num_inliers += 1
            inliers_idx.append(i) 

    return num_inliers, inliers_idx

# Select k random samples from the N observations
def select_krandom_samples(N_Matches):
    krandom_samples = []
    sample_list = [x for x in range(N_Matches)]
    for i in range(const.K_SAMPLE):
        s = rn.sample(sample_list, 2)
        krandom_samples.append(s)
        sample_list.remove(s[0])
        sample_list.remove(s[1])
    print("ksample ",krandom_samples)
    return krandom_samples

# Calculate the sum squared distance
def calculate_sum_squared_distance(prev_pts, trans_pts):
    sum_sqr_dist = 0.0
    for i in range(len(prev_pts)):
        sum_sqr_dist += (prev_pts[i][0] - trans_pts[i][0])**2 + (prev_pts[i][1] - trans_pts[i][1])**2
    return sum_sqr_dist
