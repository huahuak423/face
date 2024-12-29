import os
import cv2
from PIL import Image
import numpy as np

def getImageAndLabels(path):
    # 储存人脸数据
    facesSamples = []
    # 储存姓名数据
    ids = []
    # 储存图片信息（得到一个可以遍历每一张图片的列表，遍历的都是图片的具体路径）
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    # 加载分类项
    face_detector = cv2.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    # 遍历列表中的图片、
    for imagePath in imagePaths:
        # 打开图片，灰度化PIL有九种不同模式:1,L,P,RGB,JGBA,CMYK,YCbCr,I,F
        PIL_img = Image.open(imagePath).convert('L')
        # 将图片转化为数组，以黑白深浅
        img_numpy = np.array(PIL_img, 'uint8')
        # 获取图片人脸特征
        faces = face_detector.detectMultiScale(img_numpy)
        # 获取每张图片的id和姓名（获取文件名，并不包含扩展名）
        id = int(os.path.split(imagePath)[1].split('.')[0])
        # 预防无面容图片
        for x, y, w, h in faces:
            # 将括号内的数据存入ids列表，facesSamples列表中
            ids.append(id)
            facesSamples.append(img_numpy[y:y+h, x:x+w])
        # 打印脸部特征和id
    print('id:', id)
    print('fs:', facesSamples)
    return facesSamples, ids

if __name__ == '__main__':
    # 图片路径
    path= 'data/train1/'
    # 获取图像数组和id标签数组和姓名
    faces,ids = getImageAndLabels(path)
    # 加载识别器
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # 训练
    recognizer.train(faces, np.array(ids))
    # 保存文件
    recognizer.write('trainer/trainer.yml')

