import cv2
import face_recognition
import pickle

def recognize_from_camera(model_file='face_model.pkl'):
    """
    使用摄像头实时进行面部识别
    加载训练好的模型，并与摄像头图像进行匹配
    """
    # 加载已训练的模型
    with open(model_file, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)

    # 启动摄像头（0为默认摄像头）
    video_capture = cv2.VideoCapture(0)

    while True:
        # 捕获一帧图像
        ret, frame = video_capture.read()
        
        # 转换为RGB格式
        rgb_frame = frame[:, :, ::-1]

        # 查找面部位置和面部编码
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # 遍历每个面部，进行匹配
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 比较当前帧的面部编码与已知面部编码
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # 如果有匹配的面孔，显示名字
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # 在图像上绘制框和名字
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 显示视频流
        cv2.imshow('Face Recognition', frame)

        # 按'q'退出摄像头识别
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源并关闭窗口
    video_capture.release()
    cv2.destroyAllWindows()

# 启动实时识别
recognize_from_camera('face_model.pkl')
