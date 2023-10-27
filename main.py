from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QMessageBox, QLineEdit
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import  QThread,pyqtSlot,Qt,pyqtSignal,QReadWriteLock
from ultralytics import YOLO
from towhee import ops, pipe
from pymilvus import (
    connections,
    Collection,
)

import supervision as sv
import numpy as np
import cv2
import sys

pixmap_signal_lock = QReadWriteLock()
image = None
detections_lock = QReadWriteLock()
detections = None

yolov8_model = YOLO("best.pt")
embeding_pipline = (
    pipe.input("img")
    .map("img", "vec", ops.image_embedding.timm(model_name = 'vit_base_patch8_224'))
    .output("vec")
)

# 连接向量数据库
connections.connect("default", host="124.223.65.182", port="19530")
faces_collection = Collection("faces")
faces_collection.load()
search_params = {
    "metric_type": "L2", 
    "offset": 0, 
    "ignore_growing": False, 
}

def img2vec(image: np.ndarray):
    return embeding_pipline(image)

# CV线程 不断读取摄像头的图片信息
class CV_Thread(QThread):
    change_pixmap_signal = pyqtSignal(bool,name="cv_image")
    
    def __init__(self):
        super().__init__()
        self.run_flag = True

    def run(self):
        global image
        cap = cv2.VideoCapture(0)
        while self.run_flag:
            ret, frame = cap.read()
            if ret and pixmap_signal_lock.tryLockForWrite():
                image = frame
                self.change_pixmap_signal.emit(True)
                pixmap_signal_lock.unlock()
            # time.sleep(0.2)
        cap.release()
    
    def stop(self):
        self.run_flag = False
        self.wait()

class FaceRecUI(QWidget):
    def __init__(self):
        super().__init__()
        pixmap_signal_lock.unlock()
        detections_lock.unlock()
        self.display_width = 640
        self.display_height = 480
        self.record_thread = None
        self.signin_thread = None
        self.initUI()
    
    def closeEvent(self, event):
        self.cv_thread.stop()
        if self.record_thread != None:
            self.record_thread.stop()
        if self.signin_thread != None:
            self.signin_thread.stop()
        detections_lock.unlock()
        pixmap_signal_lock.unlock()
        faces_collection.release()
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
        
        self.name_edit = QLineEdit(self)

        self.record_btn = QPushButton("录入", self)
        self.record_btn.clicked.connect(self.record)

        self.signin_btn = QPushButton("签到", self)
        self.signin_btn.clicked.connect(self.signin)

        vbox = QVBoxLayout()
        vbox.addWidget(self.name_edit)
        vbox.addWidget(self.label)
        vbox.addWidget(self.cv_display)
        vbox.addWidget(self.record_btn)
        vbox.addWidget(self.signin_btn)
        self.setLayout(vbox)

        # CV Thread Start
        self.cv_thread = CV_Thread()
        self.cv_thread.change_pixmap_signal.connect(self.update_image)
        self.cv_thread.start()

        self.show()

    # 录入人脸数据
    def record(self):
        record_name = self.name_edit.text()
        self.signin_btn.setDisabled(True)
        self.record_btn.setDisabled(True)
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
            self.record_thread = RecordThread(name=record_name)
            self.record_thread.record_flag_signal.connect(self.handler_record_result)
            self.record_thread.start()
            pixmap_signal_lock.unlock()
        
    def signin(self):
        self.signin_btn.setDisabled(True)
        self.record_btn.setDisabled(True)
        if self.signin_thread != None:
            warning_box = QMessageBox()
            warning_box.setIcon(QMessageBox.Icon.Warning)
            warning_box.setText("上一个签到未完成")
            warning_box.exec()
            return
        print("Start Signin")
        self.signin_thread = SignInThread()
        self.signin_thread.signin_flag_signal.connect(self.handler_signin_result)
        self.signin_thread.start()
        pass

    # 人脸检测
    def detect(self):
        global detections
        result = yolov8_model.predict(self.image, verbose=False)[0]
        if detections_lock.tryLockForWrite():
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > 0.8]
            detections = detections[detections.area > 1500.0]
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
    @pyqtSlot(bool, name="cv_image")
    def update_image(self, flag):
        global image
        if pixmap_signal_lock.tryLockForRead():
            self.image = image
            frame = self.detect()
            qt_img = self.convert_cv_qt(frame)
            self.cv_display.setPixmap(qt_img)
            pixmap_signal_lock.unlock()
    
    @pyqtSlot(bool, name="record_result")
    def handler_record_result(self, flag):
        self.record_thread.stop()
        detections_lock.unlock()
        pixmap_signal_lock.unlock()
        self.record_btn.setEnabled(True)
        self.signin_btn.setEnabled(True)
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

    @pyqtSlot(str, name="signin_result")
    def handler_signin_result(self, name):
        self.signin_thread.stop()
        detections_lock.unlock()
        pixmap_signal_lock.unlock()
        self.record_btn.setEnabled(True)
        self.signin_btn.setEnabled(True)
        self.signin_thread = None
        if name is None or name == "":
            warning_box = QMessageBox()
            warning_box.setIcon(QMessageBox.Icon.Warning)
            warning_box.setText("签到失败")
            warning_box.exec()
            return
        else:
            warning_box = QMessageBox()
            warning_box.setIcon(QMessageBox.Icon.Information)
            warning_box.setText(name+"签到成功")
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
    def __init__(self, name):
        super().__init__()
        self.flag = True
        self.name = name
        
    def run(self):
        global detections
        global image
        if detections_lock.tryLockForRead(1000) and pixmap_signal_lock.tryLockForRead(1000):
            cur_image = image
            cur_detections = detections
            detections_lock.unlock()
            pixmap_signal_lock.unlock()
            # 通过读写锁不断读取detections值,知道detections不为空
            while cur_detections is None or len(cur_detections) == 0:
                if detections_lock.tryLockForRead(1000) and pixmap_signal_lock.tryLockForRead(1000):
                    cur_detections = detections
                    cur_image = image
                    detections_lock.unlock()
                    pixmap_signal_lock.unlock()
                    QThread.msleep(300)
                

            # 裁剪图片
            cropped_image = sv.crop_image(cur_image, cur_detections.xyxy[0])
            print(cropped_image.shape)
            vec = img2vec(cropped_image)
            vec = vec.get()
            vec = vec[0]
            data = {
                "data":vec,
                "name":self.name
            }
            res = faces_collection.insert(data)
            faces_collection.flush()
            if res.insert_count == 1:
                self.flag = True
            else:
                self.flag = False
        else:
            self.flag = False
        print("Record Emit")
        self.record_flag_signal.emit(self.flag)
    
    def stop(self):
        self.wait()
    
    
class SignInThread(QThread):
    signin_flag_signal = pyqtSignal(str, name="sigin_result")
    def __init__(self) -> None:
        super().__init__()

    def run(self):
        global detections
        global image
        name = ""
        if detections_lock.tryLockForRead(1000) and pixmap_signal_lock.tryLockForRead(1000):
            cur_image = image
            cur_detections = detections
            detections_lock.unlock()
            pixmap_signal_lock.unlock()
            # 通过读写锁不断读取detections值,知道detections不为空
            while cur_detections is None or len(cur_detections) == 0:
                if detections_lock.tryLockForRead(1000) and pixmap_signal_lock.tryLockForRead(1000):
                    cur_detections = detections
                    cur_image = image
                    detections_lock.unlock()
                    pixmap_signal_lock.unlock()
                    QThread.msleep(300)
            cropped_image = sv.crop_image(cur_image, cur_detections.xyxy[0])
            print(cropped_image.shape)
            vec = img2vec(cropped_image)
            vec = vec.get()
            vec = vec[0]
            res = faces_collection.search(data=[vec], anns_field="data", param=search_params, limit=1, output_fields=['name'])
            if len(res[0].ids) == 0:
                name = ""
            else:
                name = res[0][0].entity.get("name")
        else:
            name = ""
        print("Name Emit")
        self.signin_flag_signal.emit(name)
    
    def stop(self):
        self.wait()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = FaceRecUI()
    sys.exit(app.exec_())