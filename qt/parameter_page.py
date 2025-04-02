from PySide6.QtWidgets import QWidget, QFileDialog, QTableWidgetItem, QTextEdit
from fileManager import Config
from types import SimpleNamespace

import os

from qt.ui_main import Ui_MainWindow

class ParameterPage(QWidget):
    def __init__(self):
        super(ParameterPage, self).__init__()
        
        # 파라미터 페이지 UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.File_Button_3.clicked.connect(self.read_file)
        self.ui.SavePerarametersButton.clicked.connect(self.save_file)
        self.ui.SaveAsPerarametersButton.clicked.connect(self.save_as)
        
        self.path = str(os.path.dirname(__file__)) + "/config/" + "Hyperparameters.yaml"
        self.ui.filepath_lineEdit.setText(self.path)
        self.load_Hyperparameters_file(self.path)

    def read_file(self):
        fname = QFileDialog.getOpenFileName(self, "yaml 파일 선택", "", "yaml Files (*.yaml)")
        if fname[0]:
            self.ui.filepath_lineEdit.setText(fname[0])
            self.load_Hyperparameters_file(fname[0])

    def save_file(self):
        self.read_table_data()
        Config.save_config(self.config, self.path)

    def save_as(self):
        options = QFileDialog.Options()
        self.path, _ = QFileDialog.getSaveFileName(self, "다른 이름으로 저장", "", "yaml Files (*.yaml)", options=options)
        self.read_table_data()
        Config.save_config(self.config, self.path)
        self.ui.filepath_lineEdit.setText(self.path)
        
    def load_Hyperparameters_file(self, path):
        self.config = Config.load_config(path)
        self.ui.tableWidget_2.setRowCount(len(vars(self.config)))
        self.ui.tableWidget_2.setHorizontalHeaderLabels(['Hyperparameter', 'Value'])

        for row, (key, value) in enumerate(vars(self.config).items()):
            text_edit = QTextEdit()
            text_edit.setPlainText(f"{value}")
            self.ui.tableWidget_2.setItem(row + 1, 0, QTableWidgetItem(key))
            self.ui.tableWidget_2.setCellWidget(row + 1, 1, text_edit)

    def read_table_data(self):
        config_dict = {}
        for row in range(self.ui.tableWidget_2.rowCount()):
            key_item = self.ui.tableWidget_2.item(row + 1, 0)
            key = key_item.text() if key_item else "No Key"
            text_edit = self.ui.tableWidget_2.cellWidget(row + 1, 1)
            value = text_edit.toPlainText() if text_edit else "No Value"
            config_dict[key] = value

        self.config = SimpleNamespace(**config_dict)
