import constant_defination as const
import math
import numpy as np
import random as rn
import time

# Estimate the camera motion from matched features
def estimate_camera_motion(frame_1, frame_2, top_matches):
    N_Matches = len(top_matches)                            # N observations
    ksamples = select_krandom_samples(N_Matches)
    prev_kp = frame_1.get_key_points()
    cur_kp = frame_2.get_key_points()
    prev_pts = []
    cur_pts = []
    for i in range(N_Matches):                           # Select N observation from set of keypoints
        prev_pts.append(prev_kp[top_matches[i].queryIdx].pt)
        cur_pts.append(cur_kp[top_matches[i].trainIdx].pt)
    also_inliers_idx = []
    prev_pts_np = np.asarray(prev_pts)
    cur_pts_np = np.asarray(cur_pts)

    also_inliers = 0
    final_del_x, final_del_y, final_del_theta = 0.0, 0.0, 0.0
    min_sum_sqr_dist = 99999999
    for i in range(const.K_SAMPLE):
        maybe_inliers_idx = [ksamples[i][0], ksamples[i][1]]
        maybe_inliers_prev = np.take(prev_pts_np, maybe_inliers_idx, axis = 0)
        maybe_inliers_cur = np.take(cur_pts_np, maybe_inliers_idx, axis = 0)
        if((maybe_inliers_prev[0] == maybe_inliers_cur[0]).all() and (maybe_inliers_prev[1] == maybe_inliers_cur[1]).all()):
            continue
        remain_prev_pts = np.copy(prev_pts_np)           # [x for x in prev_pts if prev_pts.index(x) not in idx]
        remain_cur_pts = np.copy(cur_pts_np)             # [x for x in cur_pts if cur_pts.index(x) not in idx]
        remain_prev_pts = np.delete(remain_prev_pts, maybe_inliers_idx, axis = 0)                 # Remove the first point
        remain_cur_pts = np.delete(remain_cur_pts, maybe_inliers_idx, axis = 0)

        prev_dist = math.dist(maybe_inliers_prev[0], maybe_inliers_prev[1])
        cur_dist = math.dist(maybe_inliers_cur[0], maybe_inliers_cur[1])
        #prev_dist = math.sqrt(((prev_pts_pair[0][0] - prev_pts_pair[0][1]) + (prev_pts_pair[1][0] - prev_pts_pair[1][1]))**2)
        #cur_dist = math.sqrt(((cur_pts_pair[0][0] - cur_pts_pair[0][1]) + (cur_pts_pair[1][0] - cur_pts_pair[1][1]))**2)
        #if prev_dist == cur_dist:
        #    continue
        if (math.fabs(prev_dist - cur_dist)) < const.DIST_POINT_PAIR_THRES:
            del_x, del_y, del_theta = calculate_ego_motion(maybe_inliers_prev, maybe_inliers_cur)
            transformed_pts = calculate_transformed_points(remain_cur_pts, del_x, del_y, del_theta)
            num_inliers, inliers_idx = calculate_inliers(remain_prev_pts, transformed_pts)
            if num_inliers > also_inliers:
                also_inliers = num_inliers
                also_inliers_idx = inliers_idx
                #final_del_x, final_del_y, final_del_theta = del_x, del_y, del_theta
        if also_inliers > const.NUM_INLIERS:
            inliers_prev  = np.take(remain_prev_pts, also_inliers_idx, axis =0)
            inliers_cur = np.take(remain_cur_pts, also_inliers_idx, axis = 0)
            inliers_prev = np.append(inliers_prev, maybe_inliers_prev, axis = 0)
            inliers_cur = np.append(inliers_cur, maybe_inliers_cur, axis = 0)
            #final_del_x, final_del_y, final_del_theta = calculate_ego_motion(inliers_prev, inliers_cur)
            del_x, del_y, del_theta = calculate_ego_motion(inliers_prev, inliers_cur)
            transformed_pts = calculate_transformed_points(cur_pts_np, final_del_x, final_del_y, final_del_theta)
            sum_sqr_dist =  calculate_sum_squared_distance(prev_pts_np, transformed_pts)
            if sum_sqr_dist < min_sum_sqr_dist:
                min_sum_sqr_dist = sum_sqr_dist
                #print("Also Inliers: ", also_inliers, " Sum Squared Dist: ", min_sum_sqr_dist)
                final_del_x, final_del_y, final_del_theta = del_x, del_y, del_theta
        else:
            final_del_x, final_del_y, final_del_theta = 0.0, 0.0, 0.0
    return final_del_x, final_del_y, final_del_theta

# Calculate ego-motion from points correspondances
def calculate_ego_motion(prev_pts, cur_pts):
    n, _ = cur_pts.shape
    s_xt_1, s_yt_1 = np.sum(prev_pts, axis = 0)
    s_xt, s_yt = np.sum(cur_pts, axis = 0)
    s_xt_1xt, s_yt_1yt = np.sum(prev_pts*cur_pts, axis = 0)
    s_xt_1yt = np.sum(prev_pts[:, 0]*cur_pts[:,1])
    s_xt_yt_1 = np.sum(cur_pts[:, 0]*prev_pts[:,1])
    numerator = n*s_xt_1yt - n*s_xt_yt_1 - s_xt_1*s_yt + s_yt_1*s_xt
    denominator = n*s_xt_1xt + n*s_yt_1yt - s_xt_1*s_xt - s_yt_1*s_yt
    if denominator == 0:
        denominator = 0.000000001
    del_theta = math.atan(numerator/denominator)
    del_x = (s_xt_1 - s_xt*math.cos(del_theta) - s_yt*math.sin(del_theta))/n
    del_y = (s_yt_1 + s_xt*math.sin(del_theta) - s_yt*math.cos(del_theta))/n
    return del_x, del_y, del_theta


# Calculate the transformed points positions for the current frame
def calculate_transformed_points(rem_cur_pts, delx, dely, deltheta):
    trans_mat = np.asmatrix(((math.cos(deltheta), math.sin(deltheta), delx),
                             (-math.sin(deltheta), math.cos(deltheta), dely), (0, 0, 1.0)))
    ptsxy = np.asarray(rem_cur_pts)
    ptsx, ptsy = ptsxy.T
    lin_trans_pts = np.asarray([ptsx, ptsy, np.ones_like(ptsx)])
    transformed_pts = np.asarray(trans_mat.dot(lin_trans_pts)[:-1].T)
    return transformed_pts


# Calculate the Number of inliers for the transformed Points
def calculate_inliers(prev_pts, trans_pts):
    square_dist = np.sum((prev_pts - trans_pts)**2, axis = 1)
    inliers_idx = np.asarray(np.where(square_dist < const.T_THRES)).ravel()
    num_inliers = inliers_idx.size
    return num_inliers, inliers_idx


# Select k random samples from the N observations
def select_krandom_samples(N_Matches):
    sample_array = np.asarray(range(N_Matches))
    krandom_samples = np.random.choice(sample_array, (const.K_SAMPLE, 2), replace= False)
    return krandom_samples

# Calculate the sum squared distance
def calculate_sum_squared_distance(prev_pts, trans_pts):
    sum_sqr_dist = np.sum((prev_pts - trans_pts)**2)
    return sum_sqr_dist