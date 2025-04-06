from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from stock.stock_adaptor import DailyStockAdaptor
from qt.ui_main import Ui_MainWindow
from common.fileManager import Config
import os

class DownloadPage(QWidget):
    def __init__(self, widgets : Ui_MainWindow):
        super().__init__()
        self.widgets = widgets
        
        self.stock_config_path = "config/StockConfig.yaml"
        self.set_stock_code_list()

        self.widgets.addStockCodePushButton.clicked.connect(self.add_stock_code)
        self.widgets.removeStockCodePushButton.clicked.connect(self.remove_stock_code)
        self.widgets.DownloadPushButton.clicked.connect(self.download_stock_datas)
    
    def set_stock_code_list(self):
        stock_config = Config.load_config(self.stock_config_path)
        
        for code in stock_config.stock_codes:
            # 확장자를 제외한 파일 이름 추가
            self.widgets.stockCodeListWidget.addItem(code.strip())

    def add_stock_code(self):
        code = self.widgets.lineEdit_2.text().strip()
        if code:
            for i in range(self.widgets.stockCodeListWidget.count()):
                    if self.widgets.stockCodeListWidget.item(i).text() == code:
                        print("이미 존재하는 항목입니다.")
                        return  # 중복 항목이 있으면 추가하지 않음

            self.widgets.stockCodeListWidget.addItem(code)

    def remove_stock_code(self):
        code_item = self.widgets.stockCodeListWidget.currentRow()
        
        if code_item != -1:
            self.widgets.stockCodeListWidget.takeItem(code_item)

    def download_stock_datas(self):
        self.widgets.DownloadPushButton.setEnabled(False)
        self.widgets.DownloadPushButton.setText("Wait Downloading ..")

        max_dt = self.widgets.LastDateEdit.date().toString("yyyyMMdd")
        count = self.widgets.countSpinBox.value()
        
        stock_code_list = []
        for i in range(self.widgets.stockCodeListWidget.count()):
            stock_code_list.append(self.widgets.stockCodeListWidget.item(i).text())
        
        if os.path.exists("API/datas/"):
            for file in os.scandir("API/datas/"):
                print("Remove File: ",file)
                os.remove(file)

        self.download_thread = DownloadStockData(stock_code_list,max_dt,count)
        self.download_thread.progress_updated.connect(self.update_progress)  # 진행도 신호 연결
        self.download_thread.start()  # QThread 실행
        
        
    def update_progress(self, value):
        self.widgets.downloadProgressBar.setValue(value)  # QProgressBar 업데이트
        self.widgets.stockCodeListWidget.takeItem(0)
        if value == 100:
            self.widgets.DownloadPushButton.setEnabled(True)  # 작업 완료 후 버튼 활성화
            self.widgets.DownloadPushButton.setText("Download")
            self.widgets.downloadProgressBar.setValue(0)
    
class DownloadStockData(QThread):
    progress_updated = Signal(int) 

    def __init__(self, stock_codes, max_dt, count, parent = None):
        super().__init__(parent)
        self.stock_codes = stock_codes
        self.max_dt = max_dt
        self.count = count

    def run(self):
        for i in range(len(self.stock_codes)) :
            daily_stock = DailyStockAdaptor()
            daily_stock.set_init_datas(itm_no=self.stock_codes[i],inqr_strt_dt=self.max_dt,count=self.count, is_remove_date=False)
            daily_stock.save_datas("API/datas/ " + self.stock_codes[i] + ".csv")

            self.progress_updated.emit(((i + 1) / len(self.stock_codes)) * 100)
