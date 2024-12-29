# 导入CV模块
import cv2 as cv
# 读取图片
img = cv.imread('face1.jpg')
# 修改尺寸
resize_img = cv.resize(img, dsize=(855, 640))
# 显示原图
cv.imshow('img', img)
# 显示修改后的
cv.imshow('resize_img', resize_img)
# 打印原尺寸大小
print('未修改',img.shape)
# 打印修改后大小
print('已修改',resize_img.shape)
# 等待
while True:
    if ord('1') == cv.waitKey(0):
        break
# 释放内存
cv.destroyAllWindows()
