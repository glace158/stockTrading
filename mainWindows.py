#uic -g python main.ui > ui_main.py
#pyside6-rcc resources.qrc -o resources_rc.py

import sys
import os
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from qt.ui_main import Ui_MainWindow

from fileManager import Config
from types import SimpleNamespace

import subprocess
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas as pd

from qt.info_dial import HollowDial

import psutil
import GPUtil
import cpuinfo
import torch
import time
import threading

widgets = None
class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

        self.setWindowTitle("Rich Dog")

        widgets.stackedWidget.setCurrentIndex(0) # 메인 화면 띄우기

        # 메뉴 버튼 
        widgets.btn_home.clicked.connect(self.menu_btns)
        widgets.btn_parameter.clicked.connect(self.menu_btns)
        widgets.btn_graph.clicked.connect(self.menu_btns)
        widgets.btn_setting.clicked.connect(self.menu_btns)
        
        ##################################메인 페이지##########################################
        self.process = None
        self.model_path = ""
        self.watch_log_file()
        
        widgets.File_Button_4.clicked.connect(self.pth_file_load)
        widgets.clearLogPushButton.clicked.connect(self.clear_log)
        # 학습 버튼
        widgets.learingPushButton.clicked.connect(self.learningPPO)
        widgets.testPushButton.clicked.connect(self.testStart)

        # 다이얼 부분 바꾸기 (CPU, GPU 정보)
        widgets.verticalLayout_27.removeWidget(widgets.cpu_dial)
        widgets.cpu_dial.deleteLater()
        widgets.cpu_dial = HollowDial(widgets.cpu_frame)
        widgets.verticalLayout_27.addWidget(widgets.cpu_dial)

        widgets.verticalLayout_28.removeWidget(widgets.gpu_dial)
        widgets.gpu_dial.deleteLater()
        widgets.gpu_dial = HollowDial(widgets.gpu_frame)
        widgets.verticalLayout_28.addWidget(widgets.gpu_dial)

        self.info_thread = threading.Thread(target=self.computer_usage_info, daemon=True)
        self.info_thread.start()

        #################################파라미터 페이지##########################################
        # 파일 버튼 
        widgets.File_Button_3.clicked.connect(self.read_file)
        widgets.SavePerarametersButton.clicked.connect(self.save_file)
        widgets.SaveAsPerarametersButton.clicked.connect(self.save_as)

        # 파일 화면 초기화
        widgets.tableWidget_2.setColumnCount(2)
        self.path = str(os.path.dirname(__file__)) + "/config/" + "Hyperparameters.yaml"
        widgets.filepath_lineEdit.setText(self.path)
        self.load_Hyperparameters_file(self.path)

        #################################그래프 페이지##############################################
        self.tree_widgets = [widgets.treeWidget]  # 생성된 트리 위젯 목록
        self.current_tree_widget = widgets.treeWidget # 현재 선택된 트리 위젯
        widgets.treeWidget.mouseDoubleClickEvent = lambda event: self.set_current_tree_widget(widgets.treeWidget)
        
        # 그래픽 관련 버튼
        widgets.addCSVPushButton.clicked.connect(self.add_csv_files_to_current_tree)
        widgets.addGraphPushButton.clicked.connect(self.add_new_tree_widget)
        widgets.removePushButton.clicked.connect(self.remove_graph)
        widgets.removeCSVPushButton.clicked.connect(self.remove_csv)

        widgets.imageSizeUpPushButton.clicked.connect(self.increase_size)
        widgets.imageSizeDownPushButton.clicked.connect(self.decrease_size)
        
        widgets.imageSizeUpPushButton.setEnabled(False)
        widgets.imageSizeDownPushButton.setEnabled(False)

        #################################설정 페이지##############################################
        # 파일 버튼 
        widgets.File_Button_5.clicked.connect(self.read_devlp_file)
        widgets.SavePerarametersButton_2.clicked.connect(self.save_devlp_file)

        # 파일 화면 초기화
        self.devlp_path = str(os.path.dirname(__file__)) + "/API/" + "kis_devlp.yaml"
        widgets.filepath_lineEdit_3.setText(self.devlp_path)
        self.load_devlp_file(self.devlp_path)

    def menu_btns(self):
        # GET BUTTON CLICKED
        btn = self.sender()
        btnName = btn.objectName()

        # 홈 화면
        if btnName == "btn_home":
            widgets.stackedWidget.setCurrentWidget(widgets.home)

        # 파라미터 설정
        if btnName == "btn_parameter":
            widgets.stackedWidget.setCurrentWidget(widgets.parameters_page)

        # 그래프 보기
        if btnName == "btn_graph":
            widgets.stackedWidget.setCurrentWidget(widgets.graph_page)

        # 설정 하기
        if btnName == "btn_setting":
            widgets.stackedWidget.setCurrentWidget(widgets.kis_devlp_page)

    #################################메인 페이지 메서드##########################################
    # 모델 테스트
    def testStart(self):
        widgets.ConsolePlainTextEdit.setPlainText("")
        
        widgets.testPushButton.setEnabled(False)

        widgets.learingPushButton.clicked.disconnect(self.learningPPO)
        widgets.learingPushButton.clicked.connect(self.stoplearningPPO)
        widgets.learingPushButton.setIcon(QIcon('qt/images/icons/cil-media-stop.png')) 
        widgets.learingPushButton.setText("Testing Stop")
        widgets.testPushButton.setText("Wait Model Testing ..")

        self.ppo_test_thread = PPOThread('test', self.model_path)
        self.ppo_test_thread.finished_signal.connect(self.update_label)
        self.ppo_test_thread.start()
        print("테스트 시작")

    # 학습 시작하기
    def learningPPO(self):
        widgets.ConsolePlainTextEdit.setPlainText("")
        
        widgets.testPushButton.setEnabled(False)

        widgets.learingPushButton.clicked.disconnect(self.learningPPO)
        widgets.learingPushButton.clicked.connect(self.stoplearningPPO)
        widgets.learingPushButton.setIcon(QIcon('qt/images/icons/cil-media-stop.png')) 
        widgets.learingPushButton.setText("Learning Stop")
        widgets.testPushButton.setText("Wait Model Learning ..")
        
        self.ppo_train_thread = PPOThread('train')
        self.ppo_train_thread.finished_signal.connect(self.update_label)
        self.ppo_train_thread.start()
        print("학습 시작")

    # 학습 종료하기 버튼
    def stoplearningPPO(self):
        self.ppo_train_thread.stop()  # 작업 중단 요청
        print("중단 중..")

    # 학습 종료 시 실행
    def update_label(self):
        widgets.learingPushButton.setEnabled(True)
        widgets.testPushButton.setEnabled(True)
        
        widgets.learingPushButton.clicked.disconnect(self.stoplearningPPO)
        widgets.learingPushButton.clicked.connect(self.learningPPO)
        widgets.learingPushButton.setIcon(QIcon('qt/images/icons/cil-media-play.png')) 
        widgets.learingPushButton.setText("Learning Start")
        widgets.testPushButton.setText("Test Model Start")
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
                widgets.ConsolePlainTextEdit.setPlainText(content)
        except Exception as e:
            widgets.ConsolePlainTextEdit.setPlainText(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    

    # 컴퓨터 정보 출력
    def computer_usage_info(self):
        widgets.cpu_name_label.setText(cpuinfo.get_cpu_info()['brand_raw'])
        widgets.gpu_name_label.setText(GPUtil.getGPUs()[0].name)
        widgets.cuda_label.setText('Available CUDA : ' + str(torch.cuda.is_available()))
        while True:
            widgets.cpu_dial.setValue( psutil.cpu_percent(interval=1) )
            widgets.gpu_dial.setValue( GPUtil.getGPUs()[0].load * 100)
            time.sleep(1)
    
    # 모델 파일 읽기
    def pth_file_load(self):
        pth_file_paths, _ = QFileDialog.getOpenFileNames(self, "모델 파일 선택", "", "pth Files (*.pth)")
        
        self.model_path = pth_file_paths[0]
        widgets.filepath_lineEdit_2.setText(self.model_path)
        print(self.model_path)
    
    def clear_log(self):
        widgets.ConsolePlainTextEdit.setPlainText("")   

    #################################파라미터 페이지 메서드##########################################
    # 파일 읽어오기 버튼
    def read_file(self):
        fname= QFileDialog.getOpenFileName(self, "yaml 파일 선택", "", "yaml Files (*.yaml)")

        if fname[0]:
            widgets.filepath_lineEdit.setText(fname[0])
            self.load_Hyperparameters_file(fname[0])
    
    # 파일 저장하기 버튼
    def save_file(self):
        self.read_table_data()
        Config.save_config( self.config, self.path )

    # 파일 읽어오기
    def load_Hyperparameters_file(self, path):
        self.config = Config.load_config(path)
        widgets.tableWidget_2.setRowCount(len(vars(self.config)))
        widgets.tableWidget_2.setHorizontalHeaderLabels(['Hyperparameter', 'Value'])
        
        for row, (key, value) in enumerate(vars(self.config).items()):
            text_edit = QTextEdit()
            text_edit.setPlainText(f"{value}") 
            widgets.tableWidget_2.setItem(row + 1, 0, QTableWidgetItem(key))
            widgets.tableWidget_2.setCellWidget(row + 1, 1, text_edit)

    # 파일 저장하기
    def read_table_data(self):
        config_dict = {}
        for row in range(widgets.tableWidget_2.rowCount()):
            # Key 데이터 읽기 (QTableWidgetItem)
            key_item = widgets.tableWidget_2.item(row + 1, 0)
            key = key_item.text() if key_item else "No Key"

            # QTextEdit 데이터 읽기 (Cell Widget)
            text_edit = widgets.tableWidget_2.cellWidget(row + 1, 1)
            value = text_edit.toPlainText() if text_edit else "No Value"

            config_dict[key] = value
            #print(f"Row {row + 1}: Key = {key}, Value = {value}")

        self.config = SimpleNamespace(**config_dict)

    def save_as(self):
        # 파일 저장 대화상자 열기
        options = QFileDialog.Options()
        self.path, _ = QFileDialog.getSaveFileName(self, "다른 이름으로 저장", "", "yaml Files (*.yaml)", options=options)
        
        self.read_table_data()
        Config.save_config( self.config, self.path )
        widgets.filepath_lineEdit.setText(self.path)
        
    ########################################그래프 페이지 메서드#################################################
    # 현재 클릭한 트리 위젯
    def set_current_tree_widget(self, tree_widget):
        self.current_tree_widget = tree_widget

        for tw in self.tree_widgets:
            tw.setStyleSheet(""" """)
        
        tree_widget.setStyleSheet("""
                                  QTreeWidget{
                                    border: 2px solid rgb(189, 147, 249);
                                  }
                                  """)

        print("트리 위젯이 선택되었습니다.")

    # csv 파일 읽기
    def add_csv_files_to_current_tree(self):
        if self.current_tree_widget:
            csv_file_paths, _ = QFileDialog.getOpenFileNames(self, "CSV 파일 선택", "", "CSV Files (*.csv)")
            for csv_file_path in csv_file_paths:
                if csv_file_path:
                    self.load_csv_to_tree(self.current_tree_widget, csv_file_path)
        else:
            print("먼저 트리를 선택하세요.")

    # csv 트리 구성
    def load_csv_to_tree(self, tree_widget, file_path):
        try:
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                
                # 루트 항목 추가 (파일명, 체크박스 없음)
                root_item = QTreeWidgetItem([file_path])
                tree_widget.addTopLevelItem(root_item)

                # 첫 번째 행은 헤더로 설정
                headers = next(reader, None)
                if headers:
                    for header in headers:
                        child_item = QTreeWidgetItem([header])
                        child_item.setFlags(child_item.flags() | Qt.ItemIsUserCheckable)  # 체크박스 플래그 추가
                        child_item.setCheckState(0, Qt.Unchecked)  # 초기 체크 상태
                        root_item.addChild(child_item)

                # 트리 위젯의 stateChanged 처리
                tree_widget.itemChanged.connect(self.handle_item_change)

        except Exception as e:
            print(f"CSV 파일 읽기 오류: {e}")

    # csv 트리 요소 체크 이벤트
    def handle_item_change(self, item, column):
        if item.flags() & Qt.ItemIsUserCheckable:  # 체크박스 항목인지 확인
            state = item.checkState(column)
            if state == Qt.Checked:
                self.make_graph()
                self.load_graph_image()
                print(f"'{item.text(0)}'가 선택되었습니다.")
            elif state == Qt.Unchecked:
                self.make_graph()
                self.load_graph_image()
                print(f"'{item.text(0)}'가 선택 해제되었습니다.")
    
    # 새로운 그래프 (트리) 추가하기
    def add_new_tree_widget(self):
        # 새로운 QTreeWidget 추가
        new_tree_widget = QTreeWidget(widgets.graphDataFrame)
        new_tree_widget.setHeaderLabels(["데이터 항목"])
        self.tree_widgets.append(new_tree_widget)

        widgets.verticalLayout_5.addWidget(new_tree_widget)
        # 새 트리 위젯 클릭 이벤트 연결
        new_tree_widget.mouseDoubleClickEvent = lambda event: self.set_current_tree_widget(new_tree_widget)
        print("새로운 트리가 추가되었습니다.")
    
    # csv 삭제
    def remove_csv(self):
        if not self.current_tree_widget:
            return
        
        selected_items = self.current_tree_widget.selectedItems()
        
        if selected_items:
            for item in selected_items:
                index = self.current_tree_widget.indexOfTopLevelItem(item)
                self.current_tree_widget.takeTopLevelItem(index)
        
        self.make_graph()
        self.load_graph_image()

    # 그래프 삭제
    def remove_graph(self):
        if not self.current_tree_widget:
            return
        
        self.tree_widgets.remove(self.current_tree_widget)
        self.current_tree_widget.deleteLater() 
        self.current_tree_widget = None
        
        if self.tree_widgets:
            self.make_graph()
            self.load_graph_image()
        else:
            widgets.graph_image.clear()
            widgets.graph_image.setText("No Images")
            widgets.imageSizeUpPushButton.setEnabled(False)
            widgets.imageSizeDownPushButton.setEnabled(False)

    # 그래프 그리기
    def make_graph(self):
        fig_count = len(self.tree_widgets)
        plt.figure(figsize=(25, 10 * fig_count))
        
        
        for i in range(fig_count):
            rootItem = self.tree_widgets[i].invisibleRootItem()

            items = self.get_items_recursively(rootItem)
            datas = pd.DataFrame()
            for item in items:
                item_name = item.text(0)
                if item_name.endswith('.csv'):
                    path = item_name
                    datas = pd.read_csv(path)
                    continue
                
                state = item.checkState(0)
                if "stck_bsop_date" in datas.columns and state == Qt.Checked: 
                    datas["stck_bsop_date"] = pd.to_datetime(datas["stck_bsop_date"],format="%Y%m%d")
                    
                    ax = plt.subplot(fig_count,1, i + 1)
                    
                    ax.plot(datas["stck_bsop_date"] ,datas[item_name])
                    plt.xticks(rotation=45)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(dates.MonthLocator())
                elif "timestep" in datas.columns and state == Qt.Checked:
                    ax = plt.subplot(fig_count,1, i + 1)
                    ax.plot(datas["timestep"] ,datas[item_name])

        plt.savefig("./Data_graph/graph.png")

    # 저장한 그래프 이미지 불러오기
    def load_graph_image(self):

        self.pixmap = QPixmap("./Data_graph/graph.png")
        widgets.graph_image.adjustSize()
        widgets.graph_image.setPixmap(self.pixmap)
        
        self.image_width = self.pixmap.width()
        self.image_height = self.pixmap.height()
        
        widgets.imageSizeUpPushButton.setEnabled(True)
        widgets.imageSizeDownPushButton.setEnabled(True)

    # 트리 위젯 자식 순회
    def get_items_recursively(self, item):
        items = [item]  # 현재 항목을 리스트에 추가
        for i in range(item.childCount()):  # 자식 아이템 순회
            items.extend(self.get_items_recursively(item.child(i)))  # 자식 아이템들에 대해 재귀 호출
        return items
    
    def update_image(self):
        # 이미지를 현재 크기로 조정하여 QLabel에 표시
        resized_pixmap = self.pixmap.scaled(self.image_width, self.image_height)
        widgets.graph_image.setPixmap(resized_pixmap)

    def increase_size(self):
        # 이미지 크기를 증가
        self.image_width += 20
        self.image_height += 20
        self.update_image()

    def decrease_size(self):
        # 이미지 크기를 감소
        if self.image_width > 20 and self.image_height > 20:  # 최소 크기 제한
            self.image_width -= 20
            self.image_height -= 20
            self.update_image()

    ########################################설정 페이지#################################################
    # 파일 읽어오기 버튼
    def read_devlp_file(self):
        fname= QFileDialog.getOpenFileName(self, "yaml 파일 선택", "", "yaml Files (*.yaml)")

        if fname[0]:
            widgets.filepath_lineEdit_3.setText(fname[0])
            self.devlp_path = fname[0]
            self.load_devlp_file(fname[0])

    # 파일 저장하기 버튼
    def save_devlp_file(self):
        self.read_table_data()
        Config.save_config( self.devlp_config, self.devlp_path )

    # 파일 읽어오기
    def load_devlp_file(self, path):
            self.devlp_config = Config.load_config(path)
            widgets.paper_app_lineEdit.setText(self.devlp_config.paper_app)
            widgets.paper_sec_lineEdit.setText(self.devlp_config.paper_sec)

            widgets.my_app_lineEdit.setText(self.devlp_config.my_app)
            widgets.my_sec_lineEdit.setText(self.devlp_config.my_sec)
            
            widgets.my_acct_stock_lineEdit.setText(self.devlp_config.my_acct_stock)
            widgets.my_paper_stock_lineEdit.setText(self.devlp_config.my_paper_stock)
            
            widgets.my_prod_lineEdit.setText(self.devlp_config.my_prod)
    ###############################################################################################
    def mousePressEvent(self, event):
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

        # PRINT MOUSE EVENTS
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')

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

if __name__ == "__main__":
    app = QApplication()
    window = MainWindow()
    app.setWindowIcon(QIcon("icon.ico"))
    window.show()
    sys.exit(app.exec())