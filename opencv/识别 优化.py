from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import time

# 设置字体
font_path = "C:\\Windows\\Fonts\\STSONG.TTF"  # 宋体字体
title_font = ImageFont.truetype(font_path, 30)  # 标题字体
info_font = ImageFont.truetype(font_path, 20)  # 信息字体
button_font = ImageFont.truetype(font_path, 18)  # 按钮字体

# 加载训练数据集文件
recogizer = cv2.face.LBPHFaceRecognizer_create()  # 创建一个人脸识别器
recogizer.read('trainer/trainer.yml')  # 从train/trainer.yml文件中提取训练好的模型

names = ['刘华剑', '林梓雄']  # 姓名列表

# 初始化摄像头
cap = cv2.VideoCapture(0)

def draw_ui(frame, recognition_status, recognized_name):
    """
    绘制 UI 界面
    :param frame: 视频帧
    :param recognition_status: 识别状态（如 "识别中..."）
    :param recognized_name: 识别到的姓名
    """
    # 将 OpenCV 图像转换为 PIL 图像
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 获取图像尺寸
    height, width, _ = frame.shape

    # 绘制顶部标题栏
    title = "人脸识别系统"
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    draw.text((10, 10), title, font=title_font, fill=(255, 255, 255))  # 白色标题
    draw.text((width - 250, 10), current_time, font=info_font, fill=(255, 255, 255))  # 白色时间

    # 绘制底部状态栏
    draw.rectangle([(0, height - 60), (width, height)], fill=(50, 50, 50))  # 灰色背景
    draw.text((20, height - 50), recognition_status, font=info_font, fill=(255, 255, 255))  # 白色状态信息
    draw.text((width - 150, height - 50), "按 'q' 退出", font=button_font, fill=(255, 255, 255))  # 退出提示

    # 绘制识别结果
    if recognized_name:
        draw.text((width // 2 - 50, height // 2 - 30), recognized_name, font=title_font, fill=(0, 255, 0))  # 绿色姓名

    # 将 PIL 图像转换回 OpenCV 图像
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def face_detect_demo(frame):
    """
    人脸检测与识别
    :param frame: 视频帧
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    faces = face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))

    recognized_name = None
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
        ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
        if confidence < 80 and ids - 1 < len(names):
            recognized_name = names[ids - 1]

    return frame, recognized_name

# 主循环
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 人脸检测与识别
    frame, recognized_name = face_detect_demo(frame)

    # 绘制 UI
    recognition_status = "识别中..."
    frame = draw_ui(frame, recognition_status, recognized_name)

    # 显示结果
    cv2.imshow("Face Recognition System", frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()