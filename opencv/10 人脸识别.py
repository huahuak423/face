from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
# coding=utf-8
import urllib
import urllib.request
import hashlib

#加载训练数据集文件
recogizer = cv2.face.LBPHFaceRecognizer_create() # 创建一个人脸识别器
recogizer.read('trainer/trainer.yml')            # 从train/trainer.yml文件中提取训练好的模型

names = []                                       # 创建一个names集合
warningtime = 0                                  # 初始化warningtime=0

# 准备识别的图片
def face_detect_demo(img):
# 转换为灰度并命名为gray
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 加载人脸检测模型
    face_detector=cv2.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
# 利用模型检测灰度图像中的所有人脸
    face=face_detector.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(100,100),(300,300))
    for x,y,w,h in face:
    # 绘制矩形
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    # 绘制圆形
        cv2.circle(img,center=(x+w//2,y+h//2),radius=w//2,color=(0,255,0),thickness=1)
    # 对每个人脸区域进行识别，返回img的标签ids（ids反应的是被检测的图形与最相近的人脸数据集的标签）和置信度（检测人脸与最相近人脸数据集的差距）并打印
        ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
        print('标签id:', ids, '置信评分：', confidence)
    # 如果置信度大于65，检测成功但是识别失败，显示“未知”
        if confidence > 80: # 利用置信度来判断
            #cv2.putText(img, 'unknown', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            text = "未知"
    # 如果置信度小于65，检测成功并识别成功
        else:
        # ids是标签，如果识别的（ids-1）小于names的长度，就可以打印出来；比如识别的ids是1，那么就会打印names的第0个元素
            if ids - 1 < len(names):
                text = names[ids - 1]
        # 如果识别的（ids-1）大于names长度，那么ids没有对应的names打印，那么就无法打印，也则显示为“未知”
            else:
                text = "未知"  # 或者其他默认值
            """
            if ids - 1 < len(names):  # 添加边界检查
                cv2.putText(img, str(names[ids - 1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            else:
                cv2.putText(img, 'unknown', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            """
        # 使用Pillow绘制中文文本
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype("C:\Windows\Fonts\STSONG.TTF", 20)  # 微软雅黑字体，20号字体大小
        draw.text((x + 10, y - 30), text, font=font, fill=(0, 255, 0))
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imshow('result', img)
    # print('bug:',ids)

names = ['刘华剑','林梓雄']

"""
def name():
    path = 'data/jm/'
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
       name = str(os.path.split(imagePath)[1].split('.',2)[0])
       names.append(name)
    print(names)
"""

cap = cv2.VideoCapture(0)

# name()
print(names)

while True:
    flag,frame = cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord('1') == cv2.waitKey(1000//24):
        break
cv2.destroyAllWindows()
cap.release()
print(names)
