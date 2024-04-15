import cv2 as cv

# Uncomment the followings for ORB Feature and FLANN Based Matcher; FLANN parameters for ORB features
# features_detection_engine = cv.ORB_create()
# FLANN_INDEX_LSH = 6
# index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)


# Uncomment the followings for SIFT feature and FLANN Based Matcher; FLANN parameters for SIFT features
# features_detection_engine = cv.SIFT_create()
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# 
# features_detection_engine = cv.BRISK_create()
# FLANN_INDEX_LSH = 6
# index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)

# features_detection_engine =cv.KAZE_create()
# # FLANN_INDEX_LSH = 5
# # index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# # Uncomment the followings for AKAZE feature and FLANN Based Matcher; FLANN parameters for AKAZE features
features_detection_engine = cv.AKAZE_create()
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)

search_params = dict(checks=50)  # or pass empty dictionary

feature_matcher_engine = cv.FlannBasedMatcher(index_params, search_params)


# Detect and describe the features using feature detection engine set above
def discover_features(image, draw_key_points=True, key_points_color=(255, 0, 0)):
    key_points, key_points_desc = features_detection_engine.detectAndCompute(image, None)
    
    if draw_key_points:
        image_copy = image.copy()
        cv.drawKeypoints(image_copy, key_points, image_copy, key_points_color)
        cv.putText(image_copy, str(len(key_points)) + " FEATURES", (10, 30), cv.FONT_HERSHEY_PLAIN, 2, key_points_color)
        cv.imshow("Frame Features", image_copy)

    return key_points, key_points_desc

# Find the matched features between two consequtive frames of the Video using the matching engine set above
def match_features(frame_1, frame_2, draw_matches=False):
    assert(frame_1 is not None and frame_2 is not None)
    matches = feature_matcher_engine.knnMatch(frame_1.get_key_points_descriptors(), frame_2.get_key_points_descriptors(), k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    good_matches = []
    for i in range(len(matches)):
        if len(matches[i]) == 2:        #two cluster found
            m, n = matches[i][0], matches[i][1]
            if m.distance < 0.7 * n.distance:       #perform ratio test between two clusters
                matchesMask[i] = [1, 0]
                good_matches.append(m)            #add good matches
        elif len(matches[i]) == 1:        #one cluster found; append the matches without comparison
            matchesMask[i] = [1, 0]
            m = matches[i][0]
            good_matches.append(m)
        else:
            print("No matches found")

    if draw_matches:
        img1 = frame_1.get_image().copy()
        img2 = frame_2.get_image().copy()
        kp1  = frame_1.get_key_points()
        kp2  = frame_2.get_key_points()

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)
        img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        cv.imshow('Matched Features', img3)
    
    return good_matches

# Find the top N good matches of features by sorting matched features of list and taking N (default 100) good features
def find_top_good_matches(frame_1, frame_2, matches, N = 100, draw_matches = False):
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) <= N:
        topngood_matches = matches
    else:
        topngood_matches = matches[:N]

    if draw_matches:
        img1 = frame_1.get_image().copy()
        img2 = frame_2.get_image().copy()
        kp1  = frame_1.get_key_points()
        kp2  = frame_2.get_key_points()

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           flags=0)
        #img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, topngood_matches, None, **draw_params)
        img3 = cv.drawMatches(img1, kp1, img2, kp2, topngood_matches, None, **draw_params)#flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow('Top Sorted Matched Features', img3)

    return topngood_matches