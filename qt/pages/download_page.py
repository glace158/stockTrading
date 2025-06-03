from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from stock.stock_adaptor import DailyStockAdaptor
from stock.stock_data import Stock
from qt.ui_main import Ui_MainWindow
from common.fileManager import Config
import os
import numpy as np
import pandas as pd

class DownloadPage(QWidget):
    def __init__(self, widgets : Ui_MainWindow):
        super().__init__()
        self.widgets = widgets
        
        self.stock_config_path = "config/StockConfig.yaml"
        self.set_stock_code_list()

        self.widgets.addStockCodePushButton.clicked.connect(lambda: self.add_stock_code(self.widgets.lineEdit_2, self.widgets.stockCodeListWidget_3))
        self.widgets.removeStockCodePushButton.clicked.connect(lambda: self.remove_stock_code(self.widgets.stockCodeListWidget_3))

        self.widgets.addStockCodePushButton_2.clicked.connect(lambda: self.add_stock_code(self.widgets.lineEdit_3, self.widgets.stockCodeListWidget_4))
        self.widgets.removeStockCodePushButton_2.clicked.connect(lambda: self.remove_stock_code(self.widgets.stockCodeListWidget_4))

        self.widgets.DownloadPushButton.clicked.connect(lambda: self.download_stock_datas())
    
        self.info_list = ["moving_average_line", "rsi", "bollinger_band", "kospi", "kosdaq", "nasdaq", "spx"] 
        self.set_extra_info_list()

    def set_extra_info_list(self):
        for text in self.info_list:
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # 체크 가능하도록 설정
            item.setCheckState(Qt.Checked)  # 초기 체크 상태 설정
            self.widgets.stockCodeListWidget_5.addItem(item)
        
        self.widgets.stockCodeListWidget_5.itemChanged.connect(self.data_list_item_change)

    def data_list_item_change(self, item : QListWidgetItem):
        if item.flags() & Qt.ItemIsUserCheckable: # 체크박스 항목인지 확인
            if item.checkState() == Qt.Checked:
                self.info_list.append(item.text())
            else: 
                if item.text() in self.info_list: 
                    self.info_list.remove(item.text())

        print(self.info_list)

    def set_stock_code_list(self):
        self.stock_config = Config.load_config(self.stock_config_path)
        
        stock_list = list(set(self.stock_config.stock_codes)) 
        for code in stock_list:
            # 확장자를 제외한 파일 이름 추가
            self.widgets.stockCodeListWidget_3.addItem(str(code).strip())

        stock_list = list(set(self.stock_config.test_stock_codes))
        for code in stock_list:
            # 확장자를 제외한 파일 이름 추가
            self.widgets.stockCodeListWidget_4.addItem(str(code).strip())

        stock_list

    def add_stock_code(self, lineEdit : QLineEdit, listWidget : QListWidget):
        code = lineEdit.text().strip()
        if code:
            for i in range(listWidget.count()):
                    if listWidget.item(i).text() == code:
                        print("이미 존재하는 항목입니다.")
                        return  # 중복 항목이 있으면 추가하지 않음

            listWidget.addItem(code)
        
    def remove_stock_code(self, listWidget : QListWidget):
        code_item = listWidget.currentRow()
        
        if code_item != -1:
            listWidget.takeItem(code_item)

    def download_stock_datas(self):
        # 다운로드 버튼 설정
        self.widgets.DownloadPushButton.setEnabled(False)
        self.widgets.DownloadPushButton.setText("Wait Downloading ..")

        # 날짜, 개수 설정
        max_dt = self.widgets.LastDateEdit.date().toString("yyyyMMdd")
        count = self.widgets.countSpinBox.value()
        
        # 데이터 다운로드
        train_stock_code_list = []
        for i in range(self.widgets.stockCodeListWidget_3.count()): # 학습데이터 주식 코드 가져오기
            train_stock_code_list.append(self.widgets.stockCodeListWidget_3.item(i).text())
        
        test_stock_code_list = []
        for i in range(self.widgets.stockCodeListWidget_4.count()): # 학습데이터 주식 코드 가져오기
            test_stock_code_list.append(self.widgets.stockCodeListWidget_4.item(i).text())

        # 코드 리스트 저장
        self.stock_config.stock_codes = train_stock_code_list
        self.stock_config.test_stock_codes = test_stock_code_list
        Config.save_config(self.stock_config, self.stock_config_path)
        
        if os.path.exists("API/datas/") and self.widgets.isDataFileRemoveCheckBox.isChecked():# 기존에 있던 파일 지우기
            for file in os.scandir("API/datas/"):
                print("Remove File: ",file)
                os.remove(file)

        if os.path.exists("API/test_datas/") and self.widgets.isDataFileRemoveCheckBox.isChecked():# 기존에 있던 파일 지우기
            for file in os.scandir("API/test_datas/"):
                print("Remove File: ",file)
                os.remove(file)

        self.download_thread = DownloadStockData(train_stock_code_list, test_stock_code_list, max_dt, count, self.info_list)
        self.download_thread.progress_updated.connect(self.update_progress)  # 진행도 신호 연결
        self.download_thread.start()  # QThread 실행
        
    def update_progress(self, value, code):
        self.widgets.downloadProgressBar.setValue(value)  # QProgressBar 업데이트

        # 다운된 데이터
        if self.widgets.stockCodeListWidget_3.findItems(code, Qt.MatchExactly): # 학습 코드 리스트에 코드 포함여부 확인
            item = self.widgets.stockCodeListWidget_3.takeItem(0)
            self.widgets.stockCodeListWidget.addItem(item)
        elif self.widgets.stockCodeListWidget_4.findItems(code, Qt.MatchExactly): # 테스트 코드 리스트에 코드 포함여부 확인
            item = self.widgets.stockCodeListWidget_4.takeItem(0)
            self.widgets.stockCodeListWidget_2.addItem(item)

        if value == 100:
            self.widgets.DownloadPushButton.setEnabled(True)  # 작업 완료 후 버튼 활성화
            self.widgets.DownloadPushButton.setText("Download")
            self.widgets.downloadProgressBar.setValue(0)
    
class DownloadStockData(QThread):
    progress_updated = Signal(int, str) 

    def __init__(self, train_stock_codes, test_stock_codes, max_dt, count, info_list = [], parent = None):
        super().__init__(parent)
        self.train_stock_codes = train_stock_codes
        self.test_stock_codes = test_stock_codes
        self.max_dt = max_dt
        self.count = count
        self.info_list = info_list

        self.codes_len = len(self.train_stock_codes) + len(self.test_stock_codes)

    def run(self):
        for i in range(len(self.train_stock_codes)) :
            stock = Stock()
            stock.get_all_datas(itm_no=self.train_stock_codes[i],inqr_strt_dt=self.max_dt,count=self.count, is_remove_date=False, info_list=self.info_list)
            stock.save_datas("API/datas/" + self.train_stock_codes[i] + ".csv")

            self.split_file("API/datas/", self.train_stock_codes[i])  
            self.progress_updated.emit(((i + 1) / self.codes_len) * 100, self.train_stock_codes[i])
        
        for i in range(len(self.test_stock_codes)) :
            stock = Stock()
            stock.get_all_datas(itm_no=self.test_stock_codes[i],inqr_strt_dt=self.max_dt,count=self.count, is_remove_date=False, info_list=self.info_list)
            stock.save_datas("API/test_datas/" + self.test_stock_codes[i] + ".csv")

            self.split_file("API/test_datas/", self.test_stock_codes[i])
            self.progress_updated.emit(((len(self.train_stock_codes) + i + 1) / self.codes_len) * 100, self.test_stock_codes[i])
    
    # 액분 또는 감자 감지 
    def split_file(self, data_folder, code):
        file_path = os.path.join(data_folder, code + ".csv")

        # CSV 파일 읽기
        df = pd.read_csv(file_path)

        # 액분 또는 감자 감지 
        for i in range(0, len(df["stck_clpr"].values) - 1):
            price = df["stck_clpr"].values[i]
            next_price = df["stck_clpr"].values[i + 1]

            diff = next_price - price
            rate = (diff / price) * 100

            if np.abs(rate) > 50:
                print(f"{code} : index {i}")
                if 'Unnamed: 0' in df.columns:
                    df.drop(['Unnamed: 0'], axis = 1, inplace = True)
                
                df_1 = df.iloc[0:i+1]
                df_2 = df.iloc[i+60:]
                
                df_2.reset_index(drop=True, inplace=True) 

                df_1.to_csv(data_folder + code + "_1.csv", header=True, index=True)
                df_2.to_csv(data_folder + code + "_2.csv", header=True, index=True)

                os.remove(file_path)
                break

