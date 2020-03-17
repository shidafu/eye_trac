import numpy as np
import cv2
from cv2 import face

# 加载算法模型
detector = cv2.CascadeClassifier('etc/lbpcascades/lbpcascade_frontalface.xml')
# predictor = face.createFacemarkAAM() # Not support!!!

# 读取图像文件
img = cv2.imread("data\girl.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 识别人脸区域
rects = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1)
if len(rects) > 0:
    (x, y, w, h) = rects[0]
    # 识别人脸特征点
    # predictor()
    # landmarks = np.array([[p.x, p.y] for p in predictor(img, rect).parts()])
    # 标记人脸区域和特征点
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255))
    # for idx, point in enumerate(landmarks):
    #     pos = (point[0], point[1])
    #     cv2.circle(img, pos, 1, color=(0, 255, 0))
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(idx + 1), pos, font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
cv2.imshow('landmarks', img)
# 释放资源
cv2.waitKey()
cv2.destroyAllWindows()