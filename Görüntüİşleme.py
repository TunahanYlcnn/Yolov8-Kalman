import cv2
import numpy as np

def enhance_drone(img, boost=2.5):
    blurred = cv2.GaussianBlur(img, (3, 3), sigmaX=0)
    sharp = cv2.addWeighted(img, boost, blurred, 1.0 - boost, 0)
    sharp = cv2.normalize(sharp, None, 0, 255, cv2.NORM_MINMAX)
    return sharp

def sift_match(orig, enhanced, ratio_thresh=0.75):
    gray1 = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher()
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append([m])
    match_img = cv2.drawMatchesKnn(
        orig, kp1, enhanced, kp2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return match_img

img = cv2.imread('C:/Users/tunahan/Desktop/modelTest/res3.png')
enh = enhance_drone(img, boost=2.5)

matches = sift_match(img, enh, ratio_thresh=0.75)

# Pencerelerde göster
cv2.imshow('Orijinal', img)
cv2.imshow('İyileştirilmiş', enh)
cv2.imshow('SIFT Eşleşmeleri', matches)
cv2.waitKey(0)
cv2.destroyAllWindows()



#blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
#sharpened = cv2.addWeighted(img, 2.5, blurred, -1.5, 0)