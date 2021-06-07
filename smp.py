from __future__ import print_function
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import *
import sys
import cv2 as cv
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import pyautogui as p
import numpy as np
import time as t
import playsound
import math
import argparse
import imutils
import time
import dlib

global ALARM_ON
global ear


def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


class Window (QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Media Player")
        self.setGeometry(350, 100, 700, 500)
        self.setWindowIcon(QIcon('Player.png'))
        self.mainMenu = QMenuBar()
        self.setStyleSheet("""
               QMenuBar {
                   background-color: rgb(49,49,49);
                   color: rgb(255,255,255);
                   border: 1px solid #000;
               }

               QMenuBar::item {
                   background-color: rgb(49,49,49);
                   color: rgb(255,255,255);
               }

               QMenuBar::item::selected {
                   background-color: rgb(30,30,30);
               }

               QMenu {
                   background-color: rgb(49,49,49);
                   color: rgb(255,255,255);
                   border: 1px solid #000;           
               }

               QMenu::item::selected {
                   background-color: rgb(30,30,30);
               }
           """)
        self.fileMenu = self.mainMenu.addMenu('&File')
        self.fileMenu.addAction(QAction("Open Video", self))
        self.fileMenu.addAction(QAction("Exit", self))
        self.fileMenu.addAction(QAction("Help ", self))
        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)
        self.init_ui()
        self.show()

    def init_ui(self):

        self.mediaplayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        videowidget = QVideoWidget()

        faceDetetction = QPushButton("Face Detection")
        faceDetetction.setStyleSheet(
            "color: white; font-size: 16px; background-color: #2b5b84;" "border-radius: 10px; padding: 10px; text-align: center; ")
        faceDetetction.clicked.connect(self.FaceDetection)

        gestureDetetction = QPushButton("Gesture Detection")
        gestureDetetction.setStyleSheet(
            "color: white; font-size: 16px; background-color: #2b5b84;" "border-radius: 10px; padding: 10px; text-align: center; ")
        gestureDetetction.clicked.connect(self.GestureDetection)

        openBtn = QPushButton('Open Video')
        openBtn.clicked.connect(self.open_file)
        openBtn.setStyleSheet(
            "QPushButton::pressed""{""background-color : white;""}")
        openBtn.setStyleSheet(
            "color: white; font-size: 12px; background-color: #2b5b84; border-radius: 10px;"" padding: 10px; text-align: center;")

        self.label2 = QLabel()
        self.label2.setStyleSheet(
            "color:#2b5b84 ; font-size: 12px; border-radius: 10px; padding:"" 10px; text-align: center;")
        self.label2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.label2.setText(
            "To Exit Face Detection or Gesture Detection Press ESC")

        self.playBtn = QPushButton()
        self.playBtn.setShortcut("space")
        self.playBtn.setIcon(QIcon("blueplay.jpg"))
        self.playBtn.clicked.connect(self.play_video)
        self.playBtn.setStyleSheet(
            "color: black; font-size: 12px; background-color: #FF8C00;"" border-radius: 10px; padding: 10px; text-align: center;")
        self.playBtn.setStyleSheet(
            "QPushButton::pressed" "{" "background-color : green;""}")

        self.stopBtn = QPushButton()
        self.stopBtn.setIcon(QIcon("bluestop.jpg"))
        self.stopBtn.setStyleSheet(
            "QPushButton::pressed""{" "background-color : red;""}")
        self.stopBtn.pressed.connect(self.stop_video)

        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.sliderMoved.connect(self.set_position)

        self.label1 = QLabel()
        self.label1.setText("")
        self.label1.setPixmap(QPixmap("speaker-volume"))

        self.volumeSlider = QSlider()
        self.volumeSlider.setMaximum(100)
        self.volumeSlider.setProperty("value", 100)
        self.volumeSlider.setOrientation(Qt.Horizontal)
        self.volumeSlider.setObjectName("volumeSlider")
        self.volumeSlider.valueChanged.connect(self.mediaplayer.setVolume)

        spacer = QSpacerItem(
            20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacerItem1 = QSpacerItem(
            40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        hboxlayout = QHBoxLayout()
        hboxlayout.setContentsMargins(0, 0, 0, 0)

        hboxlayout.addWidget(openBtn)
        hboxlayout.addWidget(self.playBtn)
        hboxlayout.addWidget(self.stopBtn)
        hboxlayout.addItem(spacer)
        hboxlayout.addWidget(self.label1)
        hboxlayout.addWidget(self.volumeSlider)

        vboxlayout = QVBoxLayout()
        vboxlayout.addWidget(videowidget)
        vboxlayout.addWidget(self.slider)
        vboxlayout.addLayout(hboxlayout)
        vboxlayout.addWidget(self.label2)
        vboxlayout.addWidget(faceDetetction)
        vboxlayout.addWidget(gestureDetetction)

        self.setLayout((vboxlayout))

        self.mediaplayer.setVideoOutput(videowidget)

        self.mediaplayer.stateChanged.connect(self.mediastate_changed)
        self.mediaplayer.positionChanged.connect(self.position_changed)
        self.mediaplayer.durationChanged.connect(self.duration_changed)

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open file", "", "mp3 Audio (*.mp3);mp4 Video (*.mp4);"
                                                                         "Movie files (*.mov);All files (*.*)")

        if filename != '':
            self.mediaplayer.setMedia(
                QMediaContent(QUrl.fromLocalFile(filename)))
            self.playBtn.setEnabled(True)

        if not filename.endswith('.mp3') | filename.endswith('.mp4') | filename.endswith('.mov') | filename.endswith('.mkv')\
                | filename.endswith('.MP3') | filename.endswith('.MP4') | filename.endswith('.MOV') | filename.endswith('.MKV')\
                | filename.endswith('.wav') | filename.endswith('.WAV'):

            msg1 = QMessageBox()
            msg1.setWindowTitle("File Error !")
            msg1.setText("Invalid File Type")
            msg1.setIcon(QMessageBox.Warning)
            msg1.setWindowIcon(QIcon('file error.png'))
            msg1.setStandardButtons(QMessageBox.Retry | QMessageBox.Abort)
            msg1.setStyleSheet(
                'QMessageBox {background-color: #2b5b84; color: white;}\n QMessageBox {color: white;}\n ''QPushButton{color: white; font-size: 16px; background-color: #1d1d1d; '  'border-radius: 10px; padding: 10px; text-align: center;}\n QPushButton:hover{color: #2b5b84;}')
            msg1.buttonClicked.connect(self.popup1)
            y = msg1.exec_()

    def popup1(self, i):

        if i.text() == 'Retry':
            self.open_file()
        if i.text() == 'Abort':
            cv.destroyAllWindows()

    def stop_video(self):
        self.mediaplayer.stop()
        self.playBtn.setIcon(QIcon('blueplay.jpg'))

    def play_video(self):
        if self.mediaplayer.state() == QMediaPlayer.PlayingState:
            self.mediaplayer.pause()
            self.playBtn.setIcon(QIcon('blueplay.jpg'))
        else:
            self.mediaplayer.play()
            self.playBtn.setIcon(QIcon('bluepause.jpg'))

    def mediastate_changed(self, state):
        if self.mediaplayer.state() == QMediaPlayer.PlayingState:
            self.playBtn.setIcon(QIcon('blueplay.jpg'))
        else:
            self.playBtn.setIcon(QIcon('bluepause.jpg'))

    def position_changed(self, position):
        self.slider.setValue(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)

    def set_position(self, position):
        self.mediaplayer.setPosition(position)

    def handle_errors(self):
        self.playBtn.setEnabled(False)
        self.label.setText("Error: " + self.mediaplayer.errorString())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_D:
            self.slider.setValue(self.slider.value() + 10)
        elif event.key() == Qt.Key_A:
            self.slider.setValue(self.slider.value() - 10)
        elif event.key() == Qt.Key_W:
            self.volumeSlider.setValue(self.volumeSlider.value() + 5)
        elif event.key() == Qt.Key_S:
            self.volumeSlider.setValue(self.volumeSlider.value() - 5)
        else:
            QWidget.keyPressEvent(self, event)

    def FaceDetection(self):

        EYE_AR_THRESH = 0.2

        EYE_AR_CONSEC_FRAMES = 30

        COUNTER = 0
        ALARM_ON = False

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(
            'shape_predictor_68_face_landmarks.dat')

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        cap = cv.VideoCapture(0)

        face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")

        while True:

            ret, frame = cap.read()

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame_gray = cv.equalizeHist(frame_gray)

            rects = detector(frame_gray, 0)

            faces = face_cascade.detectMultiScale(frame_gray, minSize=(85, 85))

            how_many_faces = len(faces)

            for (x, y, w, h) in faces:

                center = (x + w // 2, y + h // 2)

                frame = cv.ellipse(
                    frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
                faceROI = frame_gray[y:y + h, x:x + w]

            for rect in rects:

                shape = predictor(frame_gray, rect)

                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv.convexHull(leftEye)
                rightEyeHull = cv.convexHull(rightEye)

                cv.drawContours(frame, [leftEyeHull], -1, (75, 50, 130), 1)
                cv.drawContours(frame, [rightEyeHull], -1, (75, 50, 130), 1)

                if ear < EYE_AR_THRESH:
                    COUNTER += 1

                    if COUNTER >= EYE_AR_CONSEC_FRAMES:

                        if not ALARM_ON:
                            ALARM_ON = True

                        cv.putText(frame, "Sleepy Eyes Detected", (10, 30),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                else:
                    COUNTER = 0
                    ALARM_ON = False

                cv.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv.namedWindow('CAMERA FEED ', cv.WINDOW_NORMAL)
            cv.resizeWindow('CAMERA FEED ', 700, 500)
            cv.imshow('CAMERA FEED ', frame)

            if how_many_faces == 0:
                self.mediaplayer.pause()
                self.mediaplayer.stateChanged
                self.playBtn.setIcon(QIcon('blueplay.jpg'))
                self.mediaplayer.positionChanged.connect(self.position_changed)
                self.mediaplayer.durationChanged.connect(self.duration_changed)

            elif ALARM_ON:
                self.mediaplayer.pause()
                self.mediaplayer.stateChanged
                self.playBtn.setIcon(QIcon('blueplay.jpg'))
                self.mediaplayer.positionChanged.connect(self.position_changed)
                self.mediaplayer.durationChanged.connect(self.duration_changed)

                msg = QMessageBox()
                msg.setWindowTitle("Drowsiness ")
                msg.setText("You're getting sleepy would you like to ")
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowIcon(QIcon('sleepy1.png'))
                msg.setStandardButtons(QMessageBox.Retry | QMessageBox.Abort)
                msg.setStyleSheet('QMessageBox {background-color: #2b5b84; color: white;}\n'
                                  ' QMessageBox {color: white;}\n ''QPushButton{color: white; font-size: 16px;'
                                  'background-color: #1d1d1d;''border-radius: 10px; padding: 10px;'
                                  ' text-align: center;}\n QPushButton:hover{color: #2b5b84;}')
                msg.buttonClicked.connect(self.popup)
                x = msg.exec_()

            else:
                self.mediaplayer.play()
                self.mediaplayer.stateChanged
                self.playBtn.setIcon(QIcon('bluepause.jpg'))

            if cv.waitKey(10) == 27:
                cap.release()
                cv.destroyWindow('CAMERA FEED ')
                break

    def GestureDetection(self):

        cap = cv.VideoCapture(0)

        def nothing(x):
            pass

        cv.namedWindow("Color Adjustments", cv.WINDOW_NORMAL)
        cv.resizeWindow("Color Adjustments", (300, 300))
        cv.createTrackbar("Thresh", "Color Adjustments", 0, 255, nothing)

        cv.createTrackbar("Lower_H", "Color Adjustments", 0, 255, nothing)
        cv.createTrackbar("Lower_S", "Color Adjustments", 0, 255, nothing)
        cv.createTrackbar("Lower_V", "Color Adjustments", 0, 255, nothing)
        cv.createTrackbar("Upper_H", "Color Adjustments", 255, 255, nothing)
        cv.createTrackbar("Upper_S", "Color Adjustments", 255, 255, nothing)
        cv.createTrackbar("Upper_V", "Color Adjustments", 255, 255, nothing)

        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv.flip(frame, 2)
            frame = cv.resize(frame, (600, 500))

            cv.rectangle(frame, (0, 1), (300, 500), (255, 0, 0), 0)
            crop_image = frame[1:500, 0:300]

            hsv = cv.cvtColor(crop_image, cv.COLOR_BGR2HSV)

            l_h = cv.getTrackbarPos("Lower_H", "Color Adjustments")
            l_s = cv.getTrackbarPos("Lower_S", "Color Adjustments")
            l_v = cv.getTrackbarPos("Lower_V", "Color Adjustments")

            u_h = cv.getTrackbarPos("Upper_H", "Color Adjustments")
            u_s = cv.getTrackbarPos("Upper_S", "Color Adjustments")
            u_v = cv.getTrackbarPos("Upper_V", "Color Adjustments")

            lower_bound = np.array([l_h, l_s, l_v])
            upper_bound = np.array([u_h, u_s, u_v])

            mask = cv.inRange(hsv, lower_bound, upper_bound)

            filtr = cv.bitwise_and(crop_image, crop_image, mask=mask)

            mask1 = cv.bitwise_not(mask)
            # getting track bar value
            m_g = cv.getTrackbarPos("Thresh", "Color Adjustments")
            ret, thresh = cv.threshold(mask1, m_g, 255, cv.THRESH_BINARY)
            dilata = cv.dilate(thresh, (3, 3), iterations=6)

            cnts, hier = cv.findContours(
                thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            try:

                cm = max(cnts, key=lambda x: cv.contourArea(x))

                epsilon = 0.0005 * cv.arcLength(cm, True)
                data = cv.approxPolyDP(cm, epsilon, True)

                hull = cv.convexHull(cm)

                cv.drawContours(crop_image, [cm], -1, (50, 50, 150), 2)
                cv.drawContours(crop_image, [hull], -1, (0, 255, 0), 2)

                hull = cv.convexHull(cm, returnPoints=False)
                defects = cv.convexityDefects(cm, hull)
                count_defects = 0

                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]

                    start = tuple(cm[s][0])
                    end = tuple(cm[e][0])
                    far = tuple(cm[f][0])

                    a = math.sqrt((end[0] - start[0]) **
                                  2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) **
                                  2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 +
                                  (end[1] - far[1]) ** 2)
                    angle = (math.acos((b ** 2 + c ** 2 - a ** 2) /
                             (2 * b * c)) * 180) / 3.14

                    if angle <= 50:
                        count_defects += 1
                        cv.circle(crop_image, far, 5, [255, 255, 255], -1)

                print("count==", count_defects)

                if count_defects == 0:

                    cv.putText(frame, " ", (50, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif count_defects == 1:

                    p.press("space")
                    cv.putText(frame, "Play/Pause", (50, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif count_defects == 2:
                    p.press("w")

                    cv.putText(frame, "Volume UP", (5, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif count_defects == 3:
                    p.press("s")

                    cv.putText(frame, "Volume Down", (50, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                elif count_defects == 4:
                    p.press("d")

                    cv.putText(frame, "Forward", (50, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                else:
                    pass

            except:
                pass

            cv.imshow("Thresh", thresh)

            cv.imshow("filter==", filtr)
            cv.imshow("Result", frame)

            key = cv.waitKey(25) & 0xFF
            if key == 27:
                break
        cap.release()
        cv.destroyAllWindows()

    def popup(self, i):

        if i.text() == "Retry":
            ear = 0.31
            ALARM_ON = False
            self.mediaplayer.play()
            self.mediaplayer.stateChanged
            self.playBtn.setIcon(QIcon('bluepause.jpg'))

        if i.text() == "Abort":
            sys.exit(app.exec_())


app = QApplication(sys.argv)
app.setStyle("Fusion")
window = Window()
sys.exit(app.exec_())
