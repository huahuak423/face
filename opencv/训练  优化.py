import face_recognition
import os
import pickle

def train_and_save_model(image_folder, model_file='face_model.pkl'):
    known_face_encodings = []
    known_face_names = []

    # 遍历图片文件夹，处理每张图片
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            
            # 加载图像
            image = face_recognition.load_image_file(image_path)
            
            # 获取图片中的面部位置和编码
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            for encoding in face_encodings:
                known_face_encodings.append(encoding)
                
                # 提取名字或编号作为类别
                # 当前使用的是文件名的数字部分，例如 "1.photo1.jpg" 提取为 "1"
                name = filename.split('.')[0]  # 分隔并提取 "1" 或 "2"
                known_face_names.append(name)

    # 将编码保存为pickle文件
    with open(model_file, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    print(f"训练完成！已保存模型到 {model_file}")
# 使用图片文件夹进行训练，保存模型
train_and_save_model('data/train1')
