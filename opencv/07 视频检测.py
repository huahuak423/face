# 多个头像检测，修改图像就好
# 导入CV模块
import cv2 as cv

def face_detect_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier('D:/opencv/opencv//sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray) # （图像，缩放倍数，检测几次并确定，方框最大多大最小多小）
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=2)
    cv.imshow('result', img)

# 读取摄像头
cap = cv.VideoCapture(0)

# 等待
while True:
    flag, frame = cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord('1') == cv.waitKey(1000//24):
        break
# 释放内存
cv.destroyAllWindows()
# 释放摄像头
cap.release()
