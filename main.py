from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import QObject, QThread,pyqtSlot,Qt,pyqtSignal
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import sys
import time

# CV线程 不断读取摄像头的图片信息
class CV_Thread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.run_flag:
            ret, image = cap.read()
            if ret:
                self.change_pixmap_signal.emit(image)
            time.sleep(0.15)
        cap.release()
    
    def stop(self):
        self.run_flag = False
        self.wait()

class FaceRecUI(QWidget):
    def __init__(self):
        super().__init__()
        self.display_width = 640
        self.display_height = 480
        
        self.initUI()
    
    def closeEvent(self, event):
        self.cv_thread.stop()
        event.accept()

    def initUI(self):
        # init Window 
        self.resize(self.display_width,800)
        self.move(300,300)
        self.setWindowTitle("Face Recognition")
        # 初始化AI模型
        self.init_models()
        self.setup_ui()
        self.show()

    def setup_ui(self):
        # Display widget impl by pixmap
        self.cv_display = QLabel(self)
        self.cv_display.resize(self.display_width, self.display_height)

        self.label = QLabel("WebCam")
        self.label.resize(100,50)

        self.record_btn = QPushButton("录入",self)
        self.record_btn.clicked.connect(self.record)

        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.cv_display)
        vbox.addWidget(self.record_btn)
        self.setLayout(vbox)

        # CV Thread Start
        self.cv_thread = CV_Thread()
        self.cv_thread.change_pixmap_signal.connect(self.update_image)
        self.cv_thread.start()

    def init_models(self):
        self.yolov8_model = YOLO("yolov8n.pt")
        pass

    # 录入人脸数据
    def record(self):
        
        pass

    # 人脸检测
    def detect(self, image):
        result = self.yolov8_model.predict(image)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence > 0.8]
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        labels = [
            result.names[class_id]+str(confidence)
            for class_id,confidence in zip(detections.class_id,detections.confidence)
        ]
        annotated_image = bounding_box_annotator.annotate(
            scene=image, detections=detections)
        return label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # 接收到图片信号,更新图片
    @pyqtSlot(np.ndarray)
    def update_image(self, image):
        # qt_img = self.convert_cv_qt(image)
        # self.cv_display.setPixmap(qt_img)
        frame = self.detect(image)
        qt_img = self.convert_cv_qt(frame)
        self.cv_display.setPixmap(qt_img)

    # CV格式的图片 -> Qt pixmap
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = FaceRecUI()
    sys.exit(app.exec_())