import numpy as np
import cv2
from cv2 import face

# 加载算法模型
detector = cv2.CascadeClassifier('etc/lbpcascades/lbpcascade_frontalface.xml')
predictor = face.createFacemarkAAM() # Not support!!!

# 相机资源
cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 采集循环
while cap.isOpened():
    ret, img = cap.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 识别人脸区域
        rects = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=1)
        if len(rects) > 0:
            (x, y, w, h) = rects[0]
            # 标记人脸区域和特征点
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        # 显示图像
        cv2.imshow('Camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# 释放资源
cap.release()
cv2.destroyAllWindows()