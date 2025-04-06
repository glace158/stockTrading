from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from qt.ui_main import Ui_MainWindow
from qt.info_dial import HollowDial
import threading
import subprocess

import psutil
import GPUtil
import cpuinfo
import torch
import time

class MainPage(QWidget):
    def __init__(self, widgets : Ui_MainWindow):
        super().__init__()
        self.widgets = widgets

        self.process = None
        self.model_path = ""
        self.watch_log_file()
        
        self.widgets.File_Button_4.clicked.connect(self.pth_file_load)
        self.widgets.clearLogPushButton.clicked.connect(self.clear_log)
        # 학습 버튼
        self.widgets.learingPushButton.clicked.connect(self.learningPPO)
        self.widgets.testPushButton.clicked.connect(self.testStart)

        # 다이얼 부분 바꾸기 (CPU, GPU 정보)
        self.widgets.verticalLayout_27.removeWidget(self.widgets.cpu_dial)
        self.widgets.cpu_dial.deleteLater()
        self.widgets.cpu_dial = HollowDial(self.widgets.cpu_frame)
        self.widgets.verticalLayout_27.addWidget(self.widgets.cpu_dial)

        self.widgets.verticalLayout_28.removeWidget(self.widgets.gpu_dial)
        self.widgets.gpu_dial.deleteLater()
        self.widgets.gpu_dial = HollowDial(self.widgets.gpu_frame)
        self.widgets.verticalLayout_28.addWidget(self.widgets.gpu_dial)

        self.info_thread = threading.Thread(target=self.computer_usage_info, daemon=True)
        self.info_thread.start()

        self.ppo_test_thread = None
        self.ppo_train_thread = None
    
    # 모델 테스트
    def testStart(self):
        self.widgets.ConsolePlainTextEdit.setPlainText("")
        
        self.widgets.testPushButton.setEnabled(False)

        self.widgets.learingPushButton.clicked.disconnect(self.learningPPO)
        self.widgets.learingPushButton.clicked.connect(self.stoplearningPPO)
        self.widgets.learingPushButton.setIcon(QIcon('qt/images/icons/cil-media-stop.png')) 
        self.widgets.learingPushButton.setText("Testing Stop")
        self.widgets.testPushButton.setText("Wait Model Testing ..")

        self.ppo_test_thread = PPOThread('test', self.model_path)
        self.ppo_test_thread.finished_signal.connect(self.update_label)
        self.ppo_test_thread.start()
        print("테스트 시작")

    # 학습 시작하기
    def learningPPO(self):
        self.widgets.ConsolePlainTextEdit.setPlainText("")
        
        self.widgets.testPushButton.setEnabled(False)

        self.widgets.learingPushButton.clicked.disconnect(self.learningPPO)
        self.widgets.learingPushButton.clicked.connect(self.stoplearningPPO)
        self.widgets.learingPushButton.setIcon(QIcon('qt/images/icons/cil-media-stop.png')) 
        self.widgets.learingPushButton.setText("Learning Stop")
        self.widgets.testPushButton.setText("Wait Model Learning ..")
        
        self.ppo_train_thread = PPOThread('train')
        self.ppo_train_thread.finished_signal.connect(self.update_label)
        self.ppo_train_thread.start()
        print("학습 시작")

    # 학습 종료하기 버튼
    def stoplearningPPO(self):
        # 작업 중단 요청
        if self.ppo_train_thread != None:   
            self.ppo_train_thread.stop()  
        if self.ppo_test_thread != None:
            self.ppo_test_thread.stop()
        print("중단 중..")

    # 학습 종료 시 실행
    def update_label(self):
        self.widgets.learingPushButton.setEnabled(True)
        self.widgets.testPushButton.setEnabled(True)
        
        self.widgets.learingPushButton.clicked.disconnect(self.stoplearningPPO)
        self.widgets.learingPushButton.clicked.connect(self.learningPPO)
        self.widgets.learingPushButton.setIcon(QIcon('qt/images/icons/cil-media-play.png')) 
        self.widgets.learingPushButton.setText("Learning Start")
        self.widgets.testPushButton.setText("Test Model Start")
        print("중단 완료")

    
    # log 파일 읽기
    def watch_log_file(self):
        self.log_path = "PPO_console" + '/' + "PPO_console_log.txt"
        self.file_watcher = QFileSystemWatcher([self.log_path])
        self.file_watcher.fileChanged.connect(self.update_text_edit)

    # log 파일 변경시 출력
    def update_text_edit(self):
        try:
            with open(self.log_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.widgets.ConsolePlainTextEdit.setPlainText(content)
        except Exception as e:
            self.widgets.ConsolePlainTextEdit.setPlainText(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    

    # 컴퓨터 정보 출력
    def computer_usage_info(self):
        self.widgets.cpu_name_label.setText(cpuinfo.get_cpu_info()['brand_raw'])
        self.widgets.gpu_name_label.setText(GPUtil.getGPUs()[0].name)
        self.widgets.cuda_label.setText('Available CUDA : ' + str(torch.cuda.is_available()))
        while True:
            self.widgets.cpu_dial.setValue( psutil.cpu_percent(interval=1) )
            self.widgets.gpu_dial.setValue( GPUtil.getGPUs()[0].load * 100)
            time.sleep(1)
    
    # 모델 파일 읽기
    def pth_file_load(self):
        pth_file_paths, _ = QFileDialog.getOpenFileNames(self, "모델 파일 선택", "", "pth Files (*.pth)")
        
        self.model_path = pth_file_paths[0]
        self.widgets.filepath_lineEdit_2.setText(self.model_path)
        print(self.model_path)
    
    def clear_log(self):
        self.widgets.ConsolePlainTextEdit.setPlainText("")
        
    def closeEvent(self,event):  # QCloseEvent 
        
        self.stoplearningPPO()

        event.accept()

class PPOThread(QThread):
    finished_signal = Signal()
    
    def __init__(self, run_mode="train", model_path='', parent = None):
        super().__init__(parent)
        self.run_mode = run_mode
        self.model_path = model_path
    
    def run(self):
        if self.run_mode == 'train':
            self.learningPPO()
        elif self.run_mode == 'test':
            self.testStart()

    # 모델 테스트
    def testStart(self):
        self.process = subprocess.Popen(
        ['python', 'main.py', 'test', self.model_path],  # 실행할 Python 스크립트
        stdout=subprocess.PIPE,    # 표준 출력을 파이프로 전달
        stderr=subprocess.PIPE,    # 표준 오류를 파이프로 전달
        text=True                  # 출력 결과를 텍스트로 받기
        )

        print(self.process.stdout.read())
        print(self.process.stderr.read())
        self.finished_signal.emit() 

    # 학습 시작하기
    def learningPPO(self):
        self.process = subprocess.Popen(
        ['python', 'main.py', 'train'],  # 실행할 Python 스크립트
        stdout=subprocess.PIPE,    # 표준 출력을 파이프로 전달
        stderr=subprocess.PIPE,    # 표준 오류를 파이프로 전달
        text=True                  # 출력 결과를 텍스트로 받기
        )
        print(self.process.stdout.read())
        print(self.process.stderr.read())
        self.finished_signal.emit() 
    
    def stop(self):
        self.process.kill() 