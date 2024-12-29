# 导入CV模块
import cv2 as cv
# 读取图片
img = cv.imread('face1.jpg')
# 修改尺寸
resize_img = cv.resize(img, dsize=(855, 640))
# 显示修改后的
cv.imshow('resize_img', resize_img)

# 坐标
x, y, w, h = 100,100,100,100
# 绘制矩形
cv.rectangle(resize_img,(x,y,x+w,y+h),color=(0,0,255),thickness=1)
# 绘制圆形
cv.circle(resize_img,center=(x+w,y+h),radius=100,color=(255,0,0),thickness=1)
# 显示
cv.imshow('resize_img',resize_img)
# 等待
while True:
    if ord('1') == cv.waitKey(0):
        break
# 释放内存
cv.destroyAllWindows()
