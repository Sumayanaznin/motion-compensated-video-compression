import cv2 as cv
import feature_match
import time
import matplotlib as plt

img1 = cv.imread('images/f1.png', cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread('images/tx_50ty_70.png', cv.IMREAD_GRAYSCALE)  # trainImage
start_time = time.time()
#ORB based detector+descriptor and BF matcher
# img3 = feature_match.orbdetect_bfmatch(img1,img2)
# img3 = feature_match.orbdetect_bfmatchwithratiotest(img1, img2)
# img3 = feature_match.orbdetect_flannmatch(img1, img2)
# cv.imshow("ORB Matched Feature", img3)


#SIFT based detector and descriptor and BF matcher
# img3 = feature_match.siftdetect_bfmatch(img1,img2)
# img3 = feature_match.siftdetect_bfmatchwithratiotest(img1, img2)
img3 = feature_match.siftdetect_flannmatch(img1, img2)
cv.imshow("SIFT Matched Feature", img3)
# plt.imshow(img3), plt.show()
end_time = time.time()
execution_time = end_time - start_time
print(execution_time)
cv.waitKey(0)
# closing all open windows
cv.destroyAllWindows()