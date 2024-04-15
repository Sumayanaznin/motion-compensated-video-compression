import math
import numpy as np
import  cv2 as cv

def motion_compensat(prev_frame, cur_frame, del_x, del_y, del_theta):

    cur_image = cur_frame.get_image().copy()
    prev_image = prev_frame.get_image().copy()

    diff_img = cv.subtract(prev_image, cur_image)
    Conv_hsv_Gray = cv.cvtColor(diff_img, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(Conv_hsv_Gray, 50, 255, cv.THRESH_BINARY)
    diff_img[mask != 0] = [0, 0, 255]
    cv.imshow("Difference Image Before Compensation", diff_img)

    # create 3x3 transformation matrix from camera motion parameters
    trans_mat = np.asmatrix(((math.cos(del_theta), math.sin(del_theta), del_x), (-math.sin(del_theta), math.cos(del_theta), del_y), (0, 0, 1.0)))
    # create indices of the destination image and linearize them
    h, w = prev_image.shape[:2]
    indy, indx = np.indices((h, w), dtype=np.float32)
    lin_trans_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])

    # warp the coordinates of cur_image to those of prev_image
    map_ind = trans_mat.dot(lin_trans_ind)
    map_x, map_y = map_ind[:-1]
    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)
    max_map_x = round(map_x[0,:].max())
    min_map_x = round(map_x[0,:].min())
    max_map_y = round(map_y[:,0].max())
    min_map_y = round(map_y[:,0].min()) 

    #dstmap1, dstmap2 = cv.convertMaps(map_x, map_y, cv.CV_16SC2, cv.CV_16SC1)

    # remap from previous image to the transformed current image
    transformed_image = cv.remap(prev_image, map_x, map_y, cv.INTER_LINEAR)
    
    transformed_image = cv.addWeighted(cur_image, 0.5, transformed_image, 0.5, 0)  # blended two consecutive frame for temporal cross dissolve and better compression
    transformed_image = copy_make_border(cur_image, transformed_image, max_map_x+15, min_map_x-20, max_map_y+15, min_map_y-20) #increased border pixel by 5 for max and decrease by 10 for min and compression optimized
    return transformed_image
    # find the difference between transformed current and true current image
    diff_img = cv.subtract(cur_image, transformed_image)
    Conv_hsv_Gray = cv.cvtColor(diff_img, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(Conv_hsv_Gray, 50, 255, cv.THRESH_BINARY)
    diff_img[mask != 0] = [0, 0, 255]
    cv.imshow("Difference Image After Compensation", diff_img)
    return diff_img
    
    #cv.waitKey()


# Copy the border pixel from current image to the transfomed image

def copy_make_border(cur_img, trans_img, max_x, min_x, max_y, min_y):
    h, w = trans_img.shape[:2]
    j = 0
    while min_x < 0:
        for i in range(h):
            trans_img[i, j] = cur_img[i, j]
        j += 1
        min_x += 1
    j = w - 1
    while max_x > w-1:
        for i in range(h):
            trans_img[i, j] = cur_img[i, j]
        j -= 1
        max_x -= 1
    i = 0
    while min_y < 0:
        for j in range(w):
            trans_img[i, j] = cur_img[i, j]
        i += 1
        min_y += 1
    i = h - 1
    while max_y > h - 1:
        for j in range(w):
            trans_img[i, j] = cur_img[i, j]
        i -= 1
        max_y -= 1
    return  trans_img
