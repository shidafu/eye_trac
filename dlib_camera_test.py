import numpy as np
import cv2
import dlib

# 加载算法模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('etc/shape_predictor_5_face_landmarks.dat')

# 相机资源
cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 采集循环
while cap.isOpened():
    ret, img = cap.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 识别人脸区域
        rects = detector(gray, 1)
        if len(rects) > 0:
            rect = rects[0]
            # 识别人脸特征点
            landmarks = np.array([[p.x, p.y] for p in predictor(img, rect).parts()])
            # 标记人脸区域和特征点
            img = cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 255, 255))
            for idx, point in enumerate(landmarks):
                pos = (point[0], point[1])
                cv2.circle(img, pos, 1, color=(0, 255, 0))
        # 显示图像
        cv2.imshow('Camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# 释放资源
cap.release()
cv2.destroyAllWindows()
