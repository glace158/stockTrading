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
        
        self.widgets.File_Button_3.clicked.connect(self.read_config_file)
        self.widgets.SavePerarametersButton.clicked.connect(self.save_file)
        self.widgets.File_Button_8.clicked.connect(self.read_stock_config_file)

        # 파일 화면 초기화
        self.widgets.tableWidget_2.setColumnCount(5)
        self.path = "./config" + '/' + "Hyperparameters.yaml"
        self.widgets.filepath_lineEdit.setText(self.path)
        self.config = Config.load_config(self.path)
        self.config = self.load_Hyperparameters_file(self.config, self.widgets.tableWidget_2)
        
        self.stock_config_path = "./config/StockConfig.yaml"
        self.widgets.filepath_lineEdit_6.setText(self.stock_config_path)
        self.stock_config = Config.load_config(self.stock_config_path)
        self.stock_config.parameters = self.load_Hyperparameters_file(self.stock_config.parameters, self.widgets.tableWidget_3)

        # 데이터 지정
        self.choose_datas(self.widgets.dataListWidget, self.stock_config.stock_columns)
        self.choose_datas(self.widgets.dataListWidget_2, self.stock_config.visualization_columns)
        
#################################파라미터 페이지 메서드##########################################
    # 파일 읽어오기 버튼
    def read_config_file(self):
        fname = QFileDialog.getOpenFileName(self, "yaml 파일 선택", "", "yaml Files (*.yaml)")

        if fname[0]:
            self.widgets.filepath_lineEdit.setText(fname[0])
            self.config = Config.load_config(fname[0])
            self.config = self.load_Hyperparameters_file(self.config, self.widgets.tableWidget_2)
    
    def read_stock_config_file(self):
        fname = QFileDialog.getOpenFileName(self, "yaml 파일 선택", "", "yaml Files (*.yaml)")

        if fname[0]:
            self.widgets.filepath_lineEdit.setText(fname[0])
            self.stock_config = Config.load_config(fname[0])
            self.stock_config.parameters = self.load_Hyperparameters_file(self.stock_config.parameters, self.widgets.tableWidget_3)
    
    # 파일 저장하기 버튼
    def save_file(self):
        self.config = self.read_table_data(self.widgets.tableWidget_2)
        Config.save_config( self.config, self.path )

        self.stock_config.parameters = self.read_table_data(self.widgets.tableWidget_3)
        Config.save_config( self.stock_config, self.stock_config_path )


    # 파일 읽어오기
    def load_Hyperparameters_file(self, config, table_widget : QTableWidget):
        
        table_widget.setRowCount(len(vars(config)) + 1)
        table_widget.setHorizontalHeaderLabels(['Hyperparameter', 'Value', 'Max', 'Min','Note'])
        
        for row, (key, item) in enumerate(vars(config).items()):
            
            value_text_edit = QTextEdit()
            value_text_edit.setPlainText(f"{item.value}") 
            table_widget.setItem(row + 1, 0, QTableWidgetItem(key))
            table_widget.setCellWidget(row + 1, 1, value_text_edit)
            
            if str(item.min) != "None": 
                value_text_edit = QTextEdit()
                value_text_edit.setPlainText(f"{item.min}")
                table_widget.setCellWidget(row + 1, 2, value_text_edit)

            if str(item.max) != "None": 
                value_text_edit = QTextEdit()
                value_text_edit.setPlainText(f"{item.max}")
                table_widget.setCellWidget(row + 1, 3, value_text_edit)

            table_widget.setItem(row + 1, 4, QTableWidgetItem(item.note))
        
        return config


    # 파일 저장하기
    def read_table_data(self, table_widget : QTableWidget):
        print("read")
        config_dict = {}
        for row in range(table_widget.rowCount()):
            # QTextEdit 데이터 읽기 (Cell Widget)
            text_edit = table_widget.cellWidget(row, 1)
            value = text_edit.toPlainText() if text_edit else "No Value"

            if value == "No Value":
                continue

            # QTextEdit 데이터 읽기 (Cell Widget)
            text_edit = table_widget.cellWidget(row, 2)
            min_value = text_edit.toPlainText() if text_edit else "None"

            # QTextEdit 데이터 읽기 (Cell Widget)
            text_edit = table_widget.cellWidget(row, 3)
            max_value = text_edit.toPlainText() if text_edit else "None"

            # Key 데이터 읽기 (QTableWidgetItem)
            key_item = table_widget.item(row, 0)
            key = key_item.text() if key_item else "No Key"

            # note 데이터 읽기 (QTableWidgetItem)
            note_item = table_widget.item(row, 4)
            note = note_item.text() if note_item else "No Note"

            config_dict[key] = {"value" : value, "min" : min_value, "max" : max_value, "note" : note}

        return SimpleNamespace(**config_dict)
    
    # 학습 데이터 지정하기
    def choose_datas(self, list_widget : QListWidget, columns):
        data_dir = "API/datas"
        file_list = next(os.walk(data_dir))[2]
        
        if len(file_list) > 0:
            self.df = pd.read_csv(data_dir + '/' + file_list[0])
            self.df.drop(['Unnamed: 0'], axis = 1, inplace = True)
            
            for column in self.df.columns:
                if column == "stck_bsop_date":# 날짜 칼럼은 제외
                    continue

                check_state = Qt.Checked if column in columns else Qt.Unchecked

                item = QListWidgetItem(f"{column}")
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # 체크 가능하도록 설정
                item.setCheckState(check_state)  # 초기 체크 상태 설정
                list_widget.addItem(item)
            
            list_widget.itemChanged.connect(self.data_list_item_change)
        
    # 학습 데이터 체크박스 이벤트
    def data_list_item_change(self, item : QListWidgetItem):
        if item.flags() & Qt.ItemIsUserCheckable: # 체크박스 항목인지 확인

            if item.listWidget() == self.widgets.dataListWidget:
                columns = self.stock_config.stock_columns
                
            elif item.listWidget() == self.widgets.dataListWidget_2:
                columns = self.stock_config.visualization_columns
                
            else:
                raise ValueError(f"{item.listWidget()} is not found")
            
            temp_list = list(columns)
            
            if item.checkState() == Qt.Checked:
                print(f"{item.text()}이(가) 체크되었습니다.")
                i = 0
                for column in self.df.columns:
                    
                    try:
                        i = temp_list.index(column) + 1
                    except:
                        pass
                    
                    if column == item.text():
                        temp_list.insert(i, item.text())
                        break
                
            else:
                print(f"{item.text()}이(가) 체크 해제되었습니다.")
                temp_list.remove(str(item.text()))

            if item.listWidget() == self.widgets.dataListWidget:
                self.stock_config.stock_columns = temp_list
            elif item.listWidget() == self.widgets.dataListWidget_2:
                self.stock_config.visualization_columns = temp_list
                
            Config.save_config(self.stock_config, self.stock_config_path)