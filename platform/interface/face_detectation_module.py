
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont







from ultralytics import YOLO
from cv2 import getTickCount, getTickFrequency
from scipy.spatial import distance as dist
from imutils import face_utils
from matplotlib import pyplot as plt
import numpy as np
import imutils
import cv2
import dlib

class FaceDetectionWidget(QWidget):
    def __init__(self, parent=None):
        super(FaceDetectionWidget, self).__init__(parent)
        self.initUI()

        # 加载 YOLOv8 模型
        self.model = YOLO("./model/best.pt")

        # 标记人脸关键点
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.detection_enabled = False  # 标志变量，控制检测状态

        self.face = True
        self.isDrawing = True
        self.isShowing = True
        self.isCndo = False
        self.isWatch = False
        self.t = 0
        self.COUNTER = 0

        if self.isDrawing:
            plt.ion()
            plt.figure(1)
            self.t_list = []
            self.result_list1 = []
            self.result_list2 = []
            self.result_list3 = []
            self.ax1 = plt.subplot(2, 2, 1)
            self.ax2 = plt.subplot(2, 2, 2)
            self.ax3 = plt.subplot(2, 1, 2)

        self.X_D = 0
        self.Y_D = 0
        self.Rx, self.Ry, self.Rw, self.Rh = 0, 0, 0, 0
        self.Lx, self.Ly, self.Lw, self.Lh = 0, 0, 0, 0

    def initUI(self):
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        self.toggle_button = QPushButton("开启检测", self)
        self.toggle_button.setFont(QFont('Times', 20, QFont.Black))
        self.toggle_button.setFixedSize(200, 80)
        self.toggle_button.clicked.connect(self.toggle_detection)

        # 使用 QHBoxLayout 将按钮居中
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.toggle_button)
        button_layout.addStretch()

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def toggle_detection(self):
        self.detection_enabled = not self.detection_enabled
        if self.detection_enabled:
            self.toggle_button.setText("关闭检测")
        else:
            self.toggle_button.setText("开启检测")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            if self.detection_enabled:
                frame = self.detect_faces(frame)
            image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(image))

    def detect_faces(self, frame):
        loop_start = cv2.getTickCount()
        success, frame = self.cap.read()  # 读取摄像头的一帧图像

        faces = self.detector(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = self.detector(gray, 0)
        if success:
            results = self.model.predict(source=frame)  # 对当前帧进行目标检测并显示结果

        frame = results[0].plot()

        # 中间放自己的显示程序
        loop_time = cv2.getTickCount() - loop_start
        total_time = loop_time / (cv2.getTickFrequency())
        FPS = int(1 / total_time)

        # 在图像左上角添加FPS文本
        fps_text = f"FPS: {FPS:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (0, 0, 255)
        text_position = (10, 30)

        if len(faces) > 1:
            self.face = False
        else:
            self.face = True

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        EYE_AR_THRESH = 0.2
        EYE_AR_CONSEC_FRAMES = 10

        ALARM_ON = False

        if self.face:
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                global leftWide,rightWide,leftHigh,rightHigh
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                leftWide, leftHigh = self.eye_wide_ratio(leftEye)
                rightWide, rightHigh = self.eye_wide_ratio(rightEye)

                ear = (leftEAR + rightEAR) / 2.0

                cv2.putText(frame, fps_text, text_position, font, font_scale, text_color, font_thickness)

                if ear < EYE_AR_THRESH:
                    self.COUNTER += 1
                    if self.COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if not ALARM_ON:
                            ALARM_ON = True

                            frame_height, frame_width = frame.shape[:2]

                            cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (0, 0, 255), -1)

                            text = "Don't close your eyes"
                            font = cv2.FONT_HERSHEY_COMPLEX
                            font_scale = 1
                            font_thickness = 2

                            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                            text_width, text_height = text_size

                            x_position = (frame_width - text_width) // 2
                            y_position = (frame_height + text_height) // 2

                            cv2.putText(frame, text, (x_position, y_position - text_height), font, font_scale,
                                        (255, 255, 255), font_thickness)

                for (name, (j, k)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                    (x, y, w, h) = cv2.boundingRect(np.array([shape[j:k]]))
                    roi = gray[y - 10:y + h + 10, x - 10:x + w + 10]

                    if j == 36:
                        right_eye = roi
                        self.Rx, self.Ry, self.Rw, self.Rh = x - 10, y - 10, w + 10, h + 10
                    elif j == 42:
                        left_eye = roi
                        self.Lx, self.Ly, self.Lw, self.Lh = x - 10, y - 10, w + 10, h + 10

                new_right_eye, rex, rey = self.finding_right_eye_pupil_centre(right_eye, ear)

                if self.isShowing:
                    if rex != 0 and rey != 0:
                        cv2.circle(frame, (self.Rx + rex, self.Ry + rey), 3, (255, 255, 0), 2)
                        cv2.circle(frame, (self.Rx + rightWide + 10, self.Ry + rightHigh + 10), 3, (0, 0, 255), 2)
                        cv2.imshow("right_eye", imutils.resize(new_right_eye, width=250, inter=cv2.INTER_CUBIC))

                new_left_eye, lex, ley = self.finding_left_eye_pupil_centre(left_eye, ear)
                if self.isShowing:
                    if lex != 0 and ley != 0:
                        cv2.circle(frame, (self.Lx + lex, self.Ly + ley), 3, (255, 255, 0), 2)
                        cv2.circle(frame, (self.Lx + leftWide + 10, self.Ly + leftHigh + 10), 3, (0, 0, 255), 2)
                        cv2.imshow("left_eye", imutils.resize(new_left_eye, width=250, inter=cv2.INTER_CUBIC))

                self.t += 1
                if self.isDrawing:
                    if self.t % 10 == 0:
                        self.t_list.append(self.t)

                        plt.sca(self.ax1)
                        self.result_list1.append(leftEAR)
                        plt.plot(self.t_list, self.result_list1, c='r', ls='-', marker='o', mec='b', mfc='w')

                        plt.sca(self.ax2)
                        self.result_list2.append(rightEAR)
                        plt.plot(self.t_list, self.result_list2, c='r', ls='-', marker='o', mec='b', mfc='w')

                        plt.sca(self.ax3)
                        self.result_list3.append(ear)
                        plt.plot(self.t_list, self.result_list3, c='r', ls='-', marker='o', mec='b', mfc='w')

                        plt.pause(0.01)


                c = cv2.waitKey(10)
                if c == 27:
                    plt.ioff()
                    break

        else:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)

            text = "Just One Face"
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 1
            font_thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = int((frame.shape[1] - text_size[0]) / 2)
            text_y = int((frame.shape[0] + text_size[1]) / 2)
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def eye_wide_ratio(self, eye):
        C = dist.euclidean(eye[0], eye[3])
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        return int(C // 2.0), int((A + B) // 4.0)

    def finding_right_eye_pupil_centre(self, right_eye, ear):
        equ = cv2.equalizeHist(right_eye)
        hist, bin_edges = np.histogram(equ.flatten(), 256, [0, 256])
        pupil_pixels = equ.size * 0.08

        threshold_fake = 0

        for i in range(0, 256):
            threshold_fake = threshold_fake + hist[i]
            if threshold_fake > pupil_pixels:
                break

        threshold1 = i

        new_img = np.zeros((equ.shape[0], equ.shape[1]))
        for i in range(5, equ.shape[0] - 5):
            for j in range(5, equ.shape[1] - 5):
                if equ[i, j] < threshold1:
                    new_img[i, j] = 255
                else:
                    new_img[i, j] = 0

        kernel = np.ones((2, 2), np.uint8)
        new_img_er1 = cv2.erode(new_img, kernel, iterations=1)

        new_img_er1 = np.uint8(new_img_er1)

        if self.isShowing:
            cv2.imshow("right_eye_img_er1", imutils.resize(new_img_er1, width=250, inter=cv2.INTER_CUBIC))

        contours, hierarchy = cv2.findContours(new_img_er1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return right_eye

        right_eye2 = right_eye
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        if ear < 0.15:
            cX, cY = 0, 0
        cv2.circle(right_eye2, (cX, cY), 2, (255, 255, 255), 2)
        cv2.arrowedLine(right_eye2, (cX, cY), ((2 * cX - rightWide - 10), (2 * cY - rightHigh - 10)),
                        (255, 255, 255), 1, 1, 0, 0.2)
        return right_eye2, cX, cY

    def finding_left_eye_pupil_centre(self, left_eye, ear):
        equ = cv2.equalizeHist(left_eye)
        hist, bin_edges = np.histogram(equ.flatten(), 256, [0, 256])
        pupil_pixels = equ.size * 0.08
        
        threshold_fake = 0

        for i in range(0, 256):
            threshold_fake = threshold_fake + hist[i]
            if threshold_fake > pupil_pixels:
                break

        threshold1 = i

        new_img = np.zeros((equ.shape[0], equ.shape[1]))
        for i in range(0, equ.shape[0]):
            for j in range(0, equ.shape[1]):
                if equ[i, j] < threshold1:
                    new_img[i, j] = 255
                else:
                    new_img[i, j] = 0

        kernel = np.ones((2, 2), np.uint8)
        new_img_er1 = cv2.erode(new_img, kernel, iterations=1)

        new_img_er1 = np.uint8(new_img_er1)

        if self.isShowing:
            cv2.imshow("left_eye_img_er1", imutils.resize(new_img_er1, width=250, inter=cv2.INTER_CUBIC))

        contours, hierarchy = cv2.findContours(new_img_er1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return left_eye

        left_eye2 = left_eye
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        if ear < 0.15:
            cX, cY = 0, 0

        cv2.circle(left_eye2, (cX, cY), 2, (255, 255, 255), 2)
        cv2.arrowedLine(left_eye2, (cX, cY), ((2 * cX - leftWide - 10), (2 * cY - leftHigh - 10)),
                        (255, 255, 255), 1, 1, 0, 0.2)

        return left_eye2, cX, cY

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()