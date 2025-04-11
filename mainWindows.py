#uic -g python main.ui > ui_main.py
#pyside6-rcc resources.qrc -o resources_rc.py

import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from qt.ui_main import Ui_MainWindow


from qt.pages.parameters_page import ParameterPage
from qt.pages.graph_page import GraphPage
from qt.pages.setting_page import SettingPage
from qt.pages.download_page import DownloadPage
from qt.pages.main_page import MainPage

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        #global widgets
        self.widgets = self.ui

        self.setWindowTitle("Rich Dog")

        self.widgets.stackedWidget.setCurrentIndex(0) # 메인 화면 띄우기

        # 메뉴 버튼 
        self.widgets.btn_home.clicked.connect(self.menu_btns)
        self.widgets.btn_parameter.clicked.connect(self.menu_btns)
        self.widgets.btn_graph.clicked.connect(self.menu_btns)
        self.widgets.btn_setting.clicked.connect(self.menu_btns)
        self.widgets.btn_download.clicked.connect(self.menu_btns)

        self.widgets.version.setText("v1.1.2")
        
    def menu_btns(self):
        # GET BUTTON CLICKED
        btn = self.sender()
        btnName = btn.objectName()

        # 홈 화면
        if btnName == "btn_home":
            self.widgets.stackedWidget.setCurrentWidget(self.widgets.home)

        # 파라미터 설정
        if btnName == "btn_parameter":
            self.widgets.stackedWidget.setCurrentWidget(self.widgets.parameters_page)

        # 그래프 보기
        if btnName == "btn_graph":
            self.widgets.stackedWidget.setCurrentWidget(self.widgets.graph_page)

        # 설정 하기
        if btnName == "btn_setting":
            self.widgets.stackedWidget.setCurrentWidget(self.widgets.kis_devlp_page)
            
        # 데이터 다운로드 하기
        if btnName == "btn_download":
            self.widgets.stackedWidget.setCurrentWidget(self.widgets.download_page)

    def mousePressEvent(self, event):
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

        # PRINT MOUSE EVENTS
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')


if __name__ == "__main__":
    app = QApplication()
    window = MainWindow()
    main_page = MainPage(window.widgets)
    parameter_page = ParameterPage(window.widgets)
    graph_page = GraphPage(window.widgets)
    setting_page = SettingPage(window.widgets)
    download_page = DownloadPage(window.widgets)
    app.setWindowIcon(QIcon("icon.ico"))
    window.show()
    sys.exit(app.exec())