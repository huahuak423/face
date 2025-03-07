# 导入模型
import cv2
# 摄像头
cap = cv2.VideoCapture(0)

flag = 1
num = 1

while(cap.isOpened):#检测是否在开启状态
    ret_flag, Vshow = cap.read() # 得到每帧图像
    cv2.imshow("Capture_Test", Vshow)
    k = cv2.waitKey(1) & 0xFF # 按键判断
    if k == ord('s'): # 保存
        cv2.imwrite('D:/opencv/opencv/data/jm'+str(num)+".name"+".jpg", Vshow)
        print("success to save"+str(num)+".jpg")
        print("-------------------------------")
        num += 1
    elif k == ord(' '): # 退出
        break

# 释放摄像头
cap.release()
# 释放内存
cv2.destroyAllWindows()
