from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from qt.ui_main import Ui_MainWindow
from common.fileManager import Config

class SettingPage(QWidget):
    def __init__(self, widgets : Ui_MainWindow):
        super().__init__()
        self.widgets = widgets
        
        # 파일 버튼 
        self.widgets.File_Button_5.clicked.connect(self.read_devlp_file)
        self.widgets.SavePerarametersButton_2.clicked.connect(self.save_devlp_file)

        # 파일 화면 초기화
        self.devlp_path = "API/" + "kis_devlp.yaml"
        self.widgets.filepath_lineEdit_3.setText(self.devlp_path)
        self.load_devlp_file(self.devlp_path)

    # 파일 읽어오기 버튼
    def read_devlp_file(self):
        fname= QFileDialog.getOpenFileName(self, "yaml 파일 선택", "", "yaml Files (*.yaml)")

        if fname[0]:
            self.widgets.filepath_lineEdit_3.setText(fname[0])
            self.devlp_path = fname[0]
            self.load_devlp_file(fname[0])

    # 파일 저장하기 버튼
    def save_devlp_file(self):
        Config.save_config( self.devlp_config, self.devlp_path )

    # 파일 읽어오기
    def load_devlp_file(self, path):
            self.devlp_config = Config.load_config(path)
            self.widgets.paper_app_lineEdit.setText(self.devlp_config.paper_app)
            self.widgets.paper_sec_lineEdit.setText(self.devlp_config.paper_sec)

            self.widgets.my_app_lineEdit.setText(self.devlp_config.my_app)
            self.widgets.my_sec_lineEdit.setText(self.devlp_config.my_sec)
            
            self.widgets.my_acct_stock_lineEdit.setText(self.devlp_config.my_acct_stock)
            self.widgets.my_paper_stock_lineEdit.setText(self.devlp_config.my_paper_stock)
            
            self.widgets.my_prod_lineEdit.setText(self.devlp_config.my_prod)