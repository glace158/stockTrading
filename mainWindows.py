#uic -g python main.ui > ui_main.py
#pyside6-rcc resources.qrc -o resources_rc.py

import sys
import os
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from qt.ui_main import Ui_MainWindow

from fileManager import Config, File
from types import SimpleNamespace

import subprocess
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas as pd

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

        self.process = None

        widgets.stackedWidget.setCurrentIndex(0)

        # 메뉴 버튼 
        widgets.btn_home.clicked.connect(self.menu_btns)
        widgets.btn_parameter.clicked.connect(self.menu_btns)
        widgets.btn_graph.clicked.connect(self.menu_btns)
        
        # 파일 버튼
        widgets.File_Button_3.clicked.connect(self.File_btns)
        widgets.SavePerarametersButton.clicked.connect(self.File_btns)

        # 파일 화면 초기화
        widgets.tableWidget_2.setColumnCount(2)
        self.path = str(os.path.dirname(__file__)) + "/config/" + "Hyperparameters.yaml"
        widgets.filepath_lineEdit.setText(self.path)
        self.load_Hyperparameters_file(self.path)
        
        # 학습 버튼
        widgets.pushButton_4.clicked.connect(self.learningPPO)


        self.tree_widgets = [widgets.treeWidget]  # 생성된 트리 위젯 목록
        self.current_tree_widget = widgets.treeWidget # 현재 선택된 트리 위젯
        widgets.treeWidget.mouseDoubleClickEvent = lambda event: self.set_current_tree_widget(widgets.treeWidget)
        
        # 그래픽 관련 버튼
        widgets.addCSVPushButton.clicked.connect(self.add_csv_files_to_current_tree)
        widgets.addGraphPushButton.clicked.connect(self.add_new_tree_widget)
        widgets.removePushButton.clicked.connect(self.remove_graph)
        widgets.removeCSVPushButton.clicked.connect(self.remove_csv)

    
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

    # 파일 버튼 클릭 시
    def File_btns(self):
        # GET BUTTON CLICKED
        btn = self.sender()
        btnName = btn.objectName()

        # 파일 읽어오기 버튼
        if btnName == "File_Button_3":
            fname= QFileDialog.getOpenFileName(self)

            if fname[0]:
                widgets.filepath_lineEdit.setText(fname[0])
                self.load_Hyperparameters_file(fname[0])

        # 파일 저장하기 버튼
        if btnName == "SavePerarametersButton" and self.config:
            self.read_table_data()
            Config.save_config( self.config, widgets.filepath_lineEdit.text() )

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

    # 학습 시작하기
    def learningPPO(self):
        widgets.pushButton_4.clicked.disconnect(self.learningPPO)
        widgets.pushButton_4.clicked.connect(self.stoplearningPPO)
        widgets.pushButton_4.setIcon(QIcon('qt/images/icons/cil-media-stop.png')) 
        widgets.pushButton_4.setText("Learning Stop")
        
        self.process = subprocess.Popen(
        ['python', 'main.py', 'train'],  # 실행할 Python 스크립트
        stdout=subprocess.PIPE,    # 표준 출력을 파이프로 전달
        stderr=subprocess.PIPE,    # 표준 오류를 파이프로 전달
        text=True                  # 출력 결과를 텍스트로 받기
        )   

    # 학습 종료하기
    def stoplearningPPO(self):
        widgets.pushButton_4.clicked.disconnect(self.stoplearningPPO)
        widgets.pushButton_4.clicked.connect(self.learningPPO)
        widgets.pushButton_4.setIcon(QIcon('qt/images/icons/cil-media-play.png')) 
        widgets.pushButton_4.setText("Learning Start")
        
        self.process.kill()   

    ###################################################################################################
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
        selected_items = self.current_tree_widget.selectedItems()
        
        if selected_items:
            for item in selected_items:
                index = self.current_tree_widget.indexOfTopLevelItem(item)
                self.current_tree_widget.takeTopLevelItem(index)

    # 그래프 삭제
    def remove_graph(self):
        self.tree_widgets.remove(self.current_tree_widget)
        self.current_tree_widget.deleteLater() 
        self.current_tree_widget = None

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

        plt.savefig("./Data_graph/graph.png")

    def load_graph_image(self):
        pixmap = QPixmap("./Data_graph/graph.png")
        widgets.graph_image.adjustSize()
        widgets.graph_image.setPixmap(pixmap)

    def get_items_recursively(self, item):
        items = [item]  # 현재 항목을 리스트에 추가
        for i in range(item.childCount()):  # 자식 아이템 순회
            items.extend(self.get_items_recursively(item.child(i)))  # 자식 아이템들에 대해 재귀 호출
        return items

    def mousePressEvent(self, event):
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

        # PRINT MOUSE EVENTS
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')

    def closeEvent(self,event):  # QCloseEvent 
        if self.process:
            self.process.kill()

        event.accept()
        
if __name__ == "__main__":
    app = QApplication()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())