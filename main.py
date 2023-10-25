from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QMessageBox, QLineEdit
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import QObject, QThread,pyqtSlot,Qt,pyqtSignal,QReadWriteLock
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import supervision as sv
import numpy as np
import torch
import cv2
import sys

pixmap_signal_lock = QReadWriteLock()
detections_lock = QReadWriteLock()
detections = None

yolov8_model = YOLO("best.pt")
embeding_model = InceptionResnetV1(pretrained='vggface2').eval()

def img2vec(image: np.ndarray):
    t = torch.from_numpy(image)
    return embeding_model(t.unsqueeze(0))

# CV线程 不断读取摄像头的图片信息
class CV_Thread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray,name="cv_image")
    
    def __init__(self):
        super().__init__()
        self.run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.run_flag:
            ret, image = cap.read()
            if ret and pixmap_signal_lock.tryLockForWrite():
                self.change_pixmap_signal.emit(image)
                pixmap_signal_lock.unlock()
            # time.sleep(0.2)
        cap.release()
    
    def stop(self):
        self.run_flag = False
        self.wait()

class FaceRecUI(QWidget):
    def __init__(self):
        super().__init__()
        self.display_width = 640
        self.display_height = 480
        self.record_thread = None
        self.initUI()
    
    def closeEvent(self, event):
        self.cv_thread.stop()
        if self.record_thread != None:
            self.record_thread.stop()
        detections_lock.unlock()
        pixmap_signal_lock.unlock()
        event.accept()

    def initUI(self):
        # init Window 
        self.resize(self.display_width,800)
        self.move(300,300)
        self.setWindowTitle("Face Recognition")

        # Display widget impl by pixmap
        self.cv_display = QLabel(self)
        self.cv_display.resize(self.display_width, self.display_height)

        self.label = QLabel("WebCam")
        self.label.resize(100,50)
        # self.name_edit = QLineEdit("Hello",self)
        # self.name_edit.setText("mmmy")
        self.name_edit = QLineEdit(self)
        self.record_btn = QPushButton("录入",self)
        self.record_btn.clicked.connect(self.record)

        vbox = QVBoxLayout()
        vbox.addWidget(self.name_edit)
        vbox.addWidget(self.label)
        vbox.addWidget(self.cv_display)
        vbox.addWidget(self.record_btn)
        self.setLayout(vbox)

        # CV Thread Start
        self.cv_thread = CV_Thread()
        self.cv_thread.change_pixmap_signal.connect(self.update_image)
        self.cv_thread.start()

        self.show()

    # 录入人脸数据
    def record(self):
        record_name = self.name_edit.text()
        if record_name == "":
            warning_box = QMessageBox()
            warning_box.setIcon(QMessageBox.Icon.Warning)
            warning_box.setText("请输入姓名")
            warning_box.exec()
            return
        if self.record_thread != None:
            warning_box = QMessageBox()
            warning_box.setIcon(QMessageBox.Icon.Warning)
            warning_box.setText("上一个录入未完成")
            warning_box.exec()
            return
        print("Start Record")
        if pixmap_signal_lock.tryLockForRead(1000):
            self.record_thread = RecordThread(image=self.image, name=record_name)
            self.record_thread.record_flag_signal.connect(self.handler_record_result)
            self.record_thread.start()
            pixmap_signal_lock.unlock()
        
    

    # 人脸检测
    def detect(self):
        global detections
        result = yolov8_model.predict(self.image, verbose=False)[0]
        if detections_lock.tryLockForWrite():
            detections = sv.Detections.from_ultralytics(result)
            
            if len(detections) == 0:
                detections_lock.unlock()
                return self.image
            bounding_box_annotator = sv.BoundingBoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            labels = [
                result.names[class_id]+str(confidence)
                for class_id,confidence in zip(detections.class_id,detections.confidence)
            ]
            
            annotated_image = bounding_box_annotator.annotate(
                scene=self.image, detections=detections)
            res_frame = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            detections_lock.unlock()
            return res_frame
        else:
            # detections_lock.unlock()
            return self.image

    # 接收到图片信号,更新图片
    @pyqtSlot(np.ndarray, name="cv_image")
    def update_image(self, image):
        if pixmap_signal_lock.tryLockForRead():
            self.image = image
            frame = self.detect()
            qt_img = self.convert_cv_qt(frame)
            self.cv_display.setPixmap(qt_img)
            pixmap_signal_lock.unlock()
    
    @pyqtSlot(bool,name="record_result")
    def handler_record_result(self, flag):
        self.record_thread.stop()
        detections_lock.unlock()
        self.record_thread = None
        if flag:
            warning_box = QMessageBox()
            warning_box.setIcon(QMessageBox.Icon.Information)
            warning_box.setText("录入成功")
            warning_box.exec()
            return
        else:
            warning_box = QMessageBox()
            warning_box.setIcon(QMessageBox.Icon.Warning)
            warning_box.setText("录入失败")
            warning_box.exec()
            return

    # CV格式的图片 -> Qt pixmap
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class RecordThread(QThread):
    record_flag_signal = pyqtSignal(bool,name="record_result")
    def __init__(self, image, name):
        super().__init__()
        self.flag = True
        self.name = name
        self.image = image
        
    def run(self):
        global detections
        if detections_lock.tryLockForRead(1000):
            cur_image = self.image
            cur_detections = detections
            detections_lock.unlock()
            # 通过读写锁不断读取detections值,知道detections不为空
            while cur_detections is None or len(cur_detections) == 0:
                if detections_lock.tryLockForRead(1000):
                    cur_detections = detections
                    detections_lock.unlock()
            # 裁剪图片
            cropped_image = sv.crop_image(cur_image, cur_detections.xyxy[0])
            
            
        else:
            self.flag = False
        print("Record Emit")
        self.record_flag_signal.emit(self.flag)
    
    def stop(self):
        self.wait()
    
    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = FaceRecUI()
    sys.exit(app.exec_())