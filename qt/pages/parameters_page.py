from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

import os
from common.fileManager import Config
import pandas as pd
from types import SimpleNamespace
from qt.ui_main import Ui_MainWindow

class ParameterPage(QWidget):
    def __init__(self, widgets : Ui_MainWindow):
        super().__init__()
        self.widgets = widgets 
        
        self.widgets.File_Button_3.clicked.connect(self.read_file)
        self.widgets.SavePerarametersButton.clicked.connect(self.save_file)
        self.widgets.SaveAsPerarametersButton.clicked.connect(self.save_as)

        # 파일 화면 초기화
        self.widgets.tableWidget_2.setColumnCount(5)
        self.path = "config" + '/' + "Hyperparameters.yaml"
        self.widgets.filepath_lineEdit.setText(self.path)
        self.load_Hyperparameters_file(self.path)
        
        # 데이터 지정
        self.stock_config_path = "config/StockConfig.yaml"
        self.choose_datas()
        
#################################파라미터 페이지 메서드##########################################
    # 파일 읽어오기 버튼
    def read_file(self):
        fname = QFileDialog.getOpenFileName(self, "yaml 파일 선택", "", "yaml Files (*.yaml)")

        if fname[0]:
            self.widgets.filepath_lineEdit.setText(fname[0])
            self.load_Hyperparameters_file(fname[0])
    
    # 파일 저장하기 버튼
    def save_file(self):
        self.read_table_data()
        Config.save_config( self.config, self.path )

    # 파일 읽어오기
    def load_Hyperparameters_file(self, path):
        self.config = Config.load_config(path)
        
        self.widgets.tableWidget_2.setRowCount(len(vars(self.config)) + 1)
        self.widgets.tableWidget_2.setHorizontalHeaderLabels(['Hyperparameter', 'Value', 'Max', 'Min','Note'])
        
        for row, (key, item) in enumerate(vars(self.config).items()):
            value_text_edit = QTextEdit()
            value_text_edit.setPlainText(f"{item.value}") 
            self.widgets.tableWidget_2.setItem(row + 1, 0, QTableWidgetItem(key))
            self.widgets.tableWidget_2.setCellWidget(row + 1, 1, value_text_edit)
                        
            value_text_edit = QTextEdit()
            value_text_edit.setPlainText(f"{item.min}")
            self.widgets.tableWidget_2.setCellWidget(row + 1, 2, value_text_edit)

            value_text_edit = QTextEdit()
            value_text_edit.setPlainText(f"{item.max}")
            self.widgets.tableWidget_2.setCellWidget(row + 1, 3, value_text_edit)

            self.widgets.tableWidget_2.setItem(row + 1, 4, QTableWidgetItem(item.note))


    # 파일 저장하기
    def read_table_data(self):
        print("read")
        config_dict = {}
        for row in range(self.widgets.tableWidget_2.rowCount()):
            # QTextEdit 데이터 읽기 (Cell Widget)
            text_edit = self.widgets.tableWidget_2.cellWidget(row, 1)
            value = text_edit.toPlainText() if text_edit else "No Value"

            if value == "No Value":
                continue

            # QTextEdit 데이터 읽기 (Cell Widget)
            text_edit = self.widgets.tableWidget_2.cellWidget(row, 2)
            min_value = text_edit.toPlainText() if text_edit else "None"

            # QTextEdit 데이터 읽기 (Cell Widget)
            text_edit = self.widgets.tableWidget_2.cellWidget(row, 3)
            max_value = text_edit.toPlainText() if text_edit else "None"

            # Key 데이터 읽기 (QTableWidgetItem)
            key_item = self.widgets.tableWidget_2.item(row, 0)
            key = key_item.text() if key_item else "No Key"

            # note 데이터 읽기 (QTableWidgetItem)
            note_item = self.widgets.tableWidget_2.item(row, 4)
            note = note_item.text() if note_item else "No Note"

            config_dict[key] = {"value" : value, "min" : min_value, "max" : max_value, "note" : note}

        self.config = SimpleNamespace(**config_dict)

    def save_as(self):
        # 파일 저장 대화상자 열기
        options = QFileDialog.Options()
        self.path, _ = QFileDialog.getSaveFileName(self, "다른 이름으로 저장", "", "yaml Files (*.yaml)", options=options)
        
        self.read_table_data()
        Config.save_config( self.config, self.path )
        self.widgets.filepath_lineEdit.setText(self.path)

    # 학습 데이터 지정하기
    def choose_datas(self):
        stock_config = Config.load_config(self.stock_config_path)
        data_dir = "API/datas"
        file_list = next(os.walk(data_dir))[2]
        if len(file_list) > 0:
            df = pd.read_csv(data_dir + '/' + file_list[0])
            df.drop(['Unnamed: 0'], axis = 1, inplace = True)
            
            for column in df.columns:
                if column == "stck_bsop_date":
                    continue

                check_state = Qt.Checked if column in stock_config.stock_columns else Qt.Unchecked

                item = QListWidgetItem(f"{column}")
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # 체크 가능하도록 설정
                item.setCheckState(check_state)  # 초기 체크 상태 설정
                self.widgets.dataListWidget.addItem(item)
            
            self.widgets.dataListWidget.itemChanged.connect(self.data_list_item_change)


    # 학습 데이터 체크박스 이벤트
    def data_list_item_change(self, item):
        if item.flags() & Qt.ItemIsUserCheckable: # 체크박스 항목인지 확인

            stock_config = Config.load_config(self.stock_config_path)
            temp_list = list(stock_config.stock_columns)

            if item.checkState() == Qt.Checked:
                print(f"{item.text()}이(가) 체크되었습니다.")
                temp_list.insert(self.widgets.dataListWidget.row(item), item.text())
                stock_config.stock_columns = temp_list
            else:
                print(f"{item.text()}이(가) 체크 해제되었습니다.")
                temp_list.remove(str(item.text()))
                stock_config.stock_columns = temp_list

            Config.save_config(stock_config,self.stock_config_path)