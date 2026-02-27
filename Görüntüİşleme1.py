import cv2
import numpy as np

def enhance_drone(img, boost=2.5):
    """
    boost: yüksek değer → daha agresif keskinleştirme
    """
    blurred = cv2.GaussianBlur(img, (3, 3), sigmaX=0)
    sharp = cv2.addWeighted(img, boost, blurred, 1.0 - boost, 0)
    return cv2.normalize(sharp, None, 0, 255, cv2.NORM_MINMAX)

def sift_match(orig, enhanced, ratio_thresh=0.75):
    """
    Orig ve enhanced img üzerinde SIFT ile eşleştirme yapar.
    Hiç descriptor bulunamazsa veya tip uyumsuzluğu olsa bile
    bir görsel (çift sütunlu) döner.
    """
    gray1 = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Eğer descriptor yoksa doğrudan iki resmi yanyana koyup dön
    if des1 is None or des2 is None:
        # Yanyana bileştirme
        blank = np.hstack([orig, enhanced])
        cv2.putText(blank, 
                    "No descriptors found!", 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,255), 2)
        return blank

    # Tip uyumu
    if des1.dtype != des2.dtype:
        des2 = des2.astype(des1.dtype)

    # BFMatcher + KNN
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in knn:
        if m.distance < ratio_thresh * n.distance:
            good.append([m])

    # Eğer hiç iyi eşleşme yoksa, yine de uyarı yazılı bir görsel dön
    if len(good) == 0:
        blank = np.hstack([orig, enhanced])
        cv2.putText(blank, 
                    "No good matches found!", 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,255,255), 2)
        return blank

    # Eşleşmeleri çiz
    match_img = cv2.drawMatchesKnn(
        orig, kp1, enhanced, kp2, good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return match_img

if __name__ == '__main__':
    img = cv2.imread('C:/Users/tunahan/Desktop/modelTest/res1.png')
    enh = enhance_drone(img, boost=2.5)
    matches = sift_match(img, enh, ratio_thresh=0.75)

    # Artık matches her zaman bir görsel
    cv2.imshow('Orijinal', img)
    cv2.imshow('Iyilestirilmis', enh)
    cv2.imshow('SIFT Eslesmeleri veya Uyarı', matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
