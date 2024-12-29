# 导入CV模块
import cv2 as cv

def face_detect_demo():
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray, 1.01, 5, 0, (100,100), (300,300)) # （图像，缩放倍数，检测几次并确定，方框最大多大最小多小）
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=2)
    cv.imshow('result', img)
# 读取图片
img = cv.imread('data/jm/1.jpg')
# 检测函数
face_detect_demo()
# 等待
while True:
    if ord('1') == cv.waitKey(0):
        break
# 释放内存
cv.destroyAllWindows()
