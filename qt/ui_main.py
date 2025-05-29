# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainiiBToj.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QAbstractScrollArea, QApplication, QCheckBox,
    QComboBox, QCommandLinkButton, QDateEdit, QDial,
    QFrame, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QPlainTextEdit, QProgressBar, QPushButton,
    QRadioButton, QScrollArea, QScrollBar, QSizePolicy,
    QSlider, QSpinBox, QStackedWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget)
import qt.resources_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1200, 800)
        MainWindow.setMinimumSize(QSize(1200, 800))
        self.styleSheet = QWidget(MainWindow)
        self.styleSheet.setObjectName(u"styleSheet")
        font = QFont()
        font.setFamilies([u"Segoe UI"])
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        self.styleSheet.setFont(font)
        self.styleSheet.setStyleSheet(u"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"\n"
"SET APP STYLESHEET - FULL STYLES HERE\n"
"DARK THEME - DRACULA COLOR BASED\n"
"\n"
"///////////////////////////////////////////////////////////////////////////////////////////////// */\n"
"\n"
"QWidget{\n"
"	color: rgb(221, 221, 221);\n"
"	font: 10pt \"Segoe UI\";\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Tooltip */\n"
"QToolTip {\n"
"	color: #ffffff;\n"
"	background-color: rgba(33, 37, 43, 180);\n"
"	border: 1px solid rgb(44, 49, 58);\n"
"	background-image: none;\n"
"	background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 2px solid rgb(255, 121, 198);\n"
"	text-align: left;\n"
"	padding-left: 8px;\n"
"	margin: 0px;\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Bg App */\n"
"#bgApp {	\n"
"	background"
                        "-color: rgb(40, 44, 52);\n"
"	border: 1px solid rgb(44, 49, 58);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Left Menu */\n"
"#leftMenuBg {	\n"
"	background-color: rgb(33, 37, 43);\n"
"}\n"
"#topLogo {\n"
"	background-color: rgb(33, 37, 43);\n"
"	background-image: url(:/images/images/images/PyDracula.png);\n"
"	background-position: centered;\n"
"	background-repeat: no-repeat;\n"
"}\n"
"#titleLeftApp { font: 63 12pt \"Segoe UI Semibold\"; }\n"
"#titleLeftDescription { font: 8pt \"Segoe UI\"; color: rgb(189, 147, 249); }\n"
"\n"
"/* MENUS */\n"
"#topMenu .QPushButton {	\n"
"	background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 22px solid transparent;\n"
"	background-color: transparent;\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"}\n"
"#topMenu .QPushButton:hover {\n"
"	background-color: rgb(40, 44, 52);\n"
"}\n"
"#topMenu .QPushButton:pressed {	\n"
"	background-color: rgb(18"
                        "9, 147, 249);\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"#bottomMenu .QPushButton {	\n"
"	background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 20px solid transparent;\n"
"	background-color:transparent;\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"}\n"
"#bottomMenu .QPushButton:hover {\n"
"	background-color: rgb(40, 44, 52);\n"
"}\n"
"#bottomMenu .QPushButton:pressed {	\n"
"	background-color: rgb(189, 147, 249);\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"#leftMenuFrame{\n"
"	border-top: 3px solid rgb(44, 49, 58);\n"
"}\n"
"\n"
"/* Toggle Button */\n"
"#toggleButton {\n"
"	background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 20px solid transparent;\n"
"	background-color: rgb(37, 41, 48);\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"	color: rgb(113, 126, 149);\n"
"}\n"
"#toggleButton:hover {\n"
"	background-color: rgb(40, 44, 52);\n"
"}\n"
"#toggleButton:pressed {\n"
"	background-color: rgb("
                        "189, 147, 249);\n"
"}\n"
"\n"
"/* Title Menu */\n"
"#titleRightInfo { padding-left: 10px; }\n"
"\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Extra Tab */\n"
"#extraLeftBox {	\n"
"	background-color: rgb(44, 49, 58);\n"
"}\n"
"#extraTopBg{	\n"
"	background-color: rgb(189, 147, 249)\n"
"}\n"
"\n"
"/* Icon */\n"
"#extraIcon {\n"
"	background-position: center;\n"
"	background-repeat: no-repeat;\n"
"	background-image: url(:/icons/images/icons/icon_settings.png);\n"
"}\n"
"\n"
"/* Label */\n"
"#extraLabel { color: rgb(255, 255, 255); }\n"
"\n"
"/* Btn Close */\n"
"#extraCloseColumnBtn { background-color: rgba(255, 255, 255, 0); border: none;  border-radius: 5px; }\n"
"#extraCloseColumnBtn:hover { background-color: rgb(196, 161, 249); border-style: solid; border-radius: 4px; }\n"
"#extraCloseColumnBtn:pressed { background-color: rgb(180, 141, 238); border-style: solid; border-radius: 4px; }\n"
"\n"
"/* Extra Content */\n"
"#extraContent{\n"
"	border"
                        "-top: 3px solid rgb(40, 44, 52);\n"
"}\n"
"\n"
"/* Extra Top Menus */\n"
"#extraTopMenu .QPushButton {\n"
"background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 22px solid transparent;\n"
"	background-color:transparent;\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"}\n"
"#extraTopMenu .QPushButton:hover {\n"
"	background-color: rgb(40, 44, 52);\n"
"}\n"
"#extraTopMenu .QPushButton:pressed {	\n"
"	background-color: rgb(189, 147, 249);\n"
"	color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Content App */\n"
"#contentTopBg{	\n"
"	background-color: rgb(33, 37, 43);\n"
"}\n"
"#contentBottom{\n"
"	border-top: 3px solid rgb(44, 49, 58);\n"
"}\n"
"\n"
"/* Top Buttons */\n"
"#rightButtons .QPushButton { background-color: rgba(255, 255, 255, 0); border: none;  border-radius: 5px; }\n"
"#rightButtons .QPushButton:hover { background-color: rgb(44, 49, 57); border-sty"
                        "le: solid; border-radius: 4px; }\n"
"#rightButtons .QPushButton:pressed { background-color: rgb(23, 26, 30); border-style: solid; border-radius: 4px; }\n"
"\n"
"/* Theme Settings */\n"
"#extraRightBox { background-color: rgb(44, 49, 58); }\n"
"#themeSettingsTopDetail { background-color: rgb(189, 147, 249); }\n"
"\n"
"/* Bottom Bar */\n"
"#bottomBar { background-color: rgb(44, 49, 58); }\n"
"#bottomBar QLabel { font-size: 11px; color: rgb(113, 126, 149); padding-left: 10px; padding-right: 10px; padding-bottom: 2px; }\n"
"\n"
"/* CONTENT SETTINGS */\n"
"/* MENUS */\n"
"#contentSettings .QPushButton {	\n"
"	background-position: left center;\n"
"    background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 22px solid transparent;\n"
"	background-color:transparent;\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"}\n"
"#contentSettings .QPushButton:hover {\n"
"	background-color: rgb(40, 44, 52);\n"
"}\n"
"#contentSettings .QPushButton:pressed {	\n"
"	background-color: rgb(189, 147, 249);\n"
"	color: rgb"
                        "(255, 255, 255);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"QTableWidget */\n"
"QTableWidget {	\n"
"	background-color: transparent;\n"
"	padding: 10px;\n"
"	border-radius: 5px;\n"
"	gridline-color: rgb(44, 49, 58);\n"
"	border-bottom: 1px solid rgb(44, 49, 60);\n"
"}\n"
"QTableWidget::item{\n"
"	border-color: rgb(44, 49, 60);\n"
"	padding-left: 5px;\n"
"	padding-right: 5px;\n"
"	gridline-color: rgb(44, 49, 60);\n"
"}\n"
"QTableWidget::item:selected{\n"
"	background-color: rgb(189, 147, 249);\n"
"}\n"
"QHeaderView::section{\n"
"	background-color: rgb(33, 37, 43);\n"
"	max-width: 30px;\n"
"	border: 1px solid rgb(44, 49, 58);\n"
"	border-style: none;\n"
"    border-bottom: 1px solid rgb(44, 49, 60);\n"
"    border-right: 1px solid rgb(44, 49, 60);\n"
"}\n"
"QTableWidget::horizontalHeader {	\n"
"	background-color: rgb(33, 37, 43);\n"
"}\n"
"QHeaderView::section:horizontal\n"
"{\n"
"    border: 1px solid rgb(33, 37, 43);\n"
"	background-co"
                        "lor: rgb(33, 37, 43);\n"
"	padding: 3px;\n"
"	border-top-left-radius: 7px;\n"
"    border-top-right-radius: 7px;\n"
"}\n"
"QHeaderView::section:vertical\n"
"{\n"
"    border: 1px solid rgb(44, 49, 60);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"QTreeWidget*/\n"
"\n"
"QTreeWidget{\n"
"	background-color: rgb(44, 49, 60);\n"
"	border: 2px solid rgb(33, 37, 43);\n"
"	border-radius: 10px;\n"
"}\n"
"\n"
"QTreeWidget::item {\n"
"	background-color: rgb(44, 49, 60);\n"
"	border-radius: 1px;\n"
"}\n"
"\n"
"QTreeWidget::item:selected {\n"
"	border-radius: 1px;\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"LineEdit */\n"
"QLineEdit {\n"
"	background-color: rgb(33, 37, 43);\n"
"	border-radius: 5px;\n"
"	border: 2px solid rgb(33, 37, 43);\n"
"	padding-left: 10px;\n"
"	selection-color: rgb(255, 255, 255);\n"
"	selection-background-color: rgb(255, 121, 198);\n"
"}\n"
"QLineEdit"
                        ":hover {\n"
"	border: 2px solid rgb(64, 71, 88);\n"
"}\n"
"QLineEdit:focus {\n"
"	border: 2px solid rgb(91, 101, 124);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"PlainTextEdit */\n"
"QPlainTextEdit {\n"
"	background-color: rgb(27, 29, 35);\n"
"	border-radius: 5px;\n"
"	padding: 10px;\n"
"	selection-color: rgb(255, 255, 255);\n"
"	selection-background-color: rgb(255, 121, 198);\n"
"}\n"
"QPlainTextEdit  QScrollBar:vertical {\n"
"    width: 8px;\n"
" }\n"
"QPlainTextEdit  QScrollBar:horizontal {\n"
"    height: 8px;\n"
" }\n"
"QPlainTextEdit:hover {\n"
"	border: 2px solid rgb(64, 71, 88);\n"
"}\n"
"QPlainTextEdit:focus {\n"
"	border: 2px solid rgb(91, 101, 124);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"ScrollBars */\n"
"QScrollBar:horizontal {\n"
"    border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    height: 8px;\n"
"    margin: 0px 21px 0 21px;\n"
""
                        "	border-radius: 0px;\n"
"}\n"
"QScrollBar::handle:horizontal {\n"
"    background: rgb(189, 147, 249);\n"
"    min-width: 25px;\n"
"	border-radius: 4px\n"
"}\n"
"QScrollBar::add-line:horizontal {\n"
"    border: none;\n"
"    background: rgb(55, 63, 77);\n"
"    width: 20px;\n"
"	border-top-right-radius: 4px;\n"
"    border-bottom-right-radius: 4px;\n"
"    subcontrol-position: right;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"QScrollBar::sub-line:horizontal {\n"
"    border: none;\n"
"    background: rgb(55, 63, 77);\n"
"    width: 20px;\n"
"	border-top-left-radius: 4px;\n"
"    border-bottom-left-radius: 4px;\n"
"    subcontrol-position: left;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"QScrollBar::up-arrow:horizontal, QScrollBar::down-arrow:horizontal\n"
"{\n"
"     background: none;\n"
"}\n"
"QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal\n"
"{\n"
"     background: none;\n"
"}\n"
" QScrollBar:vertical {\n"
"	border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    width: 8px;\n"
"   "
                        " margin: 21px 0 21px 0;\n"
"	border-radius: 0px;\n"
" }\n"
" QScrollBar::handle:vertical {	\n"
"	background: rgb(189, 147, 249);\n"
"    min-height: 25px;\n"
"	border-radius: 4px\n"
" }\n"
" QScrollBar::add-line:vertical {\n"
"     border: none;\n"
"    background: rgb(55, 63, 77);\n"
"     height: 20px;\n"
"	border-bottom-left-radius: 4px;\n"
"    border-bottom-right-radius: 4px;\n"
"     subcontrol-position: bottom;\n"
"     subcontrol-origin: margin;\n"
" }\n"
" QScrollBar::sub-line:vertical {\n"
"	border: none;\n"
"    background: rgb(55, 63, 77);\n"
"     height: 20px;\n"
"	border-top-left-radius: 4px;\n"
"    border-top-right-radius: 4px;\n"
"     subcontrol-position: top;\n"
"     subcontrol-origin: margin;\n"
" }\n"
" QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {\n"
"     background: none;\n"
" }\n"
"\n"
" QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {\n"
"     background: none;\n"
" }\n"
"\n"
"/* ///////////////////////////////////////////////////////////////////////"
                        "//////////////////////////\n"
"CheckBox */\n"
"QCheckBox::indicator {\n"
"    border: 3px solid rgb(52, 59, 72);\n"
"	width: 15px;\n"
"	height: 15px;\n"
"	border-radius: 10px;\n"
"    background: rgb(44, 49, 60);\n"
"}\n"
"QCheckBox::indicator:hover {\n"
"    border: 3px solid rgb(58, 66, 81);\n"
"}\n"
"QCheckBox::indicator:checked {\n"
"    background: 3px solid rgb(52, 59, 72);\n"
"	border: 3px solid rgb(52, 59, 72);	\n"
"	background-image: url(:/icons/images/icons/cil-check-alt.png);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"RadioButton */\n"
"QRadioButton::indicator {\n"
"    border: 3px solid rgb(52, 59, 72);\n"
"	width: 15px;\n"
"	height: 15px;\n"
"	border-radius: 10px;\n"
"    background: rgb(44, 49, 60);\n"
"}\n"
"QRadioButton::indicator:hover {\n"
"    border: 3px solid rgb(58, 66, 81);\n"
"}\n"
"QRadioButton::indicator:checked {\n"
"    background: 3px solid rgb(94, 106, 130);\n"
"	border: 3px solid rgb(52, 59, 72);	\n"
"}\n"
""
                        "\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"ComboBox */\n"
"QComboBox{\n"
"	background-color: rgb(27, 29, 35);\n"
"	border-radius: 5px;\n"
"	border: 2px solid rgb(33, 37, 43);\n"
"	padding: 5px;\n"
"	padding-left: 10px;\n"
"}\n"
"QComboBox:hover{\n"
"	border: 2px solid rgb(64, 71, 88);\n"
"}\n"
"QComboBox::drop-down {\n"
"	subcontrol-origin: padding;\n"
"	subcontrol-position: top right;\n"
"	width: 25px; \n"
"	border-left-width: 3px;\n"
"	border-left-color: rgba(39, 44, 54, 150);\n"
"	border-left-style: solid;\n"
"	border-top-right-radius: 3px;\n"
"	border-bottom-right-radius: 3px;	\n"
"	background-image: url(:/icons/images/icons/cil-arrow-bottom.png);\n"
"	background-position: center;\n"
"	background-repeat: no-reperat;\n"
" }\n"
"QComboBox QAbstractItemView {\n"
"	color: rgb(255, 121, 198);	\n"
"	background-color: rgb(33, 37, 43);\n"
"	padding: 10px;\n"
"	selection-background-color: rgb(39, 44, 54);\n"
"}\n"
"\n"
"/* //////////////////////////"
                        "///////////////////////////////////////////////////////////////////////\n"
"Sliders */\n"
"QSlider::groove:horizontal {\n"
"    border-radius: 5px;\n"
"    height: 10px;\n"
"	margin: 0px;\n"
"	background-color: rgb(52, 59, 72);\n"
"}\n"
"QSlider::groove:horizontal:hover {\n"
"	background-color: rgb(55, 62, 76);\n"
"}\n"
"QSlider::handle:horizontal {\n"
"    background-color: rgb(189, 147, 249);\n"
"    border: none;\n"
"    height: 10px;\n"
"    width: 10px;\n"
"    margin: 0px;\n"
"	border-radius: 5px;\n"
"}\n"
"QSlider::handle:horizontal:hover {\n"
"    background-color: rgb(195, 155, 255);\n"
"}\n"
"QSlider::handle:horizontal:pressed {\n"
"    background-color: rgb(255, 121, 198);\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"    border-radius: 5px;\n"
"    width: 10px;\n"
"    margin: 0px;\n"
"	background-color: rgb(52, 59, 72);\n"
"}\n"
"QSlider::groove:vertical:hover {\n"
"	background-color: rgb(55, 62, 76);\n"
"}\n"
"QSlider::handle:vertical {\n"
"    background-color: rgb(189, 147, 249);\n"
"	border: n"
                        "one;\n"
"    height: 10px;\n"
"    width: 10px;\n"
"    margin: 0px;\n"
"	border-radius: 5px;\n"
"}\n"
"QSlider::handle:vertical:hover {\n"
"    background-color: rgb(195, 155, 255);\n"
"}\n"
"QSlider::handle:vertical:pressed {\n"
"    background-color: rgb(255, 121, 198);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"CommandLinkButton */\n"
"QCommandLinkButton {	\n"
"	color: rgb(255, 121, 198);\n"
"	border-radius: 5px;\n"
"	padding: 5px;\n"
"	color: rgb(255, 170, 255);\n"
"}\n"
"QCommandLinkButton:hover {	\n"
"	color: rgb(255, 170, 255);\n"
"	background-color: rgb(44, 49, 60);\n"
"}\n"
"QCommandLinkButton:pressed {	\n"
"	color: rgb(189, 147, 249);\n"
"	background-color: rgb(52, 58, 71);\n"
"}\n"
"\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"Button */\n"
"#pagesContainer QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;	\n"
"	background-color: r"
                        "gb(52, 59, 72);\n"
"}\n"
"#pagesContainer QPushButton:hover {\n"
"	background-color: rgb(57, 65, 80);\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"#pagesContainer QPushButton:pressed {	\n"
"	background-color: rgb(35, 40, 49);\n"
"	border: 2px solid rgb(43, 50, 61);\n"
"}\n"
"\n"
"")
        self.verticalLayout_21 = QVBoxLayout(self.styleSheet)
        self.verticalLayout_21.setSpacing(0)
        self.verticalLayout_21.setObjectName(u"verticalLayout_21")
        self.verticalLayout_21.setContentsMargins(0, 0, 0, 0)
        self.bgApp = QFrame(self.styleSheet)
        self.bgApp.setObjectName(u"bgApp")
        self.bgApp.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.bgApp.setStyleSheet(u"")
        self.bgApp.setFrameShape(QFrame.Shape.NoFrame)
        self.bgApp.setFrameShadow(QFrame.Shadow.Raised)
        self.appLayout = QHBoxLayout(self.bgApp)
        self.appLayout.setSpacing(0)
        self.appLayout.setObjectName(u"appLayout")
        self.appLayout.setContentsMargins(0, 0, 0, 0)
        self.leftMenuBg = QFrame(self.bgApp)
        self.leftMenuBg.setObjectName(u"leftMenuBg")
        self.leftMenuBg.setMinimumSize(QSize(60, 0))
        self.leftMenuBg.setMaximumSize(QSize(60, 16777215))
        self.leftMenuBg.setFrameShape(QFrame.Shape.NoFrame)
        self.leftMenuBg.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.leftMenuBg)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.topLogoInfo = QFrame(self.leftMenuBg)
        self.topLogoInfo.setObjectName(u"topLogoInfo")
        self.topLogoInfo.setMinimumSize(QSize(0, 50))
        self.topLogoInfo.setMaximumSize(QSize(16777215, 50))
        self.topLogoInfo.setFrameShape(QFrame.Shape.NoFrame)
        self.topLogoInfo.setFrameShadow(QFrame.Shadow.Raised)
        self.topLogo = QFrame(self.topLogoInfo)
        self.topLogo.setObjectName(u"topLogo")
        self.topLogo.setGeometry(QRect(10, 5, 42, 42))
        self.topLogo.setMinimumSize(QSize(42, 42))
        self.topLogo.setMaximumSize(QSize(42, 42))
        self.topLogo.setStyleSheet(u"background-image: url(:/images/images/images/DogeIconSmall.png)")
        self.topLogo.setFrameShape(QFrame.Shape.NoFrame)
        self.topLogo.setFrameShadow(QFrame.Shadow.Raised)
        self.titleLeftApp = QLabel(self.topLogoInfo)
        self.titleLeftApp.setObjectName(u"titleLeftApp")
        self.titleLeftApp.setGeometry(QRect(70, 8, 160, 20))
        font1 = QFont()
        font1.setFamilies([u"Segoe UI Semibold"])
        font1.setPointSize(12)
        #font1.setWeight(QFont.)
        font1.setItalic(False)
        self.titleLeftApp.setFont(font1)
        self.titleLeftApp.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)
        self.titleLeftDescription = QLabel(self.topLogoInfo)
        self.titleLeftDescription.setObjectName(u"titleLeftDescription")
        self.titleLeftDescription.setGeometry(QRect(70, 27, 160, 16))
        self.titleLeftDescription.setMaximumSize(QSize(16777215, 16))
        font2 = QFont()
        font2.setFamilies([u"Segoe UI"])
        font2.setPointSize(8)
        font2.setBold(False)
        font2.setItalic(False)
        self.titleLeftDescription.setFont(font2)
        self.titleLeftDescription.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.verticalLayout_3.addWidget(self.topLogoInfo)

        self.leftMenuFrame = QFrame(self.leftMenuBg)
        self.leftMenuFrame.setObjectName(u"leftMenuFrame")
        self.leftMenuFrame.setFrameShape(QFrame.Shape.NoFrame)
        self.leftMenuFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalMenuLayout = QVBoxLayout(self.leftMenuFrame)
        self.verticalMenuLayout.setSpacing(0)
        self.verticalMenuLayout.setObjectName(u"verticalMenuLayout")
        self.verticalMenuLayout.setContentsMargins(0, 0, 0, 0)
        self.topMenu = QFrame(self.leftMenuFrame)
        self.topMenu.setObjectName(u"topMenu")
        self.topMenu.setFrameShape(QFrame.Shape.NoFrame)
        self.topMenu.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_8 = QVBoxLayout(self.topMenu)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.btn_home = QPushButton(self.topMenu)
        self.btn_home.setObjectName(u"btn_home")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_home.sizePolicy().hasHeightForWidth())
        self.btn_home.setSizePolicy(sizePolicy)
        self.btn_home.setMinimumSize(QSize(0, 45))
        self.btn_home.setFont(font)
        self.btn_home.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_home.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.btn_home.setStyleSheet(u"background-image: url(:/icons/images/icons/cil-home.png)")

        self.verticalLayout_8.addWidget(self.btn_home)

        self.btn_parameter = QPushButton(self.topMenu)
        self.btn_parameter.setObjectName(u"btn_parameter")
        sizePolicy.setHeightForWidth(self.btn_parameter.sizePolicy().hasHeightForWidth())
        self.btn_parameter.setSizePolicy(sizePolicy)
        self.btn_parameter.setMinimumSize(QSize(0, 45))
        self.btn_parameter.setFont(font)
        self.btn_parameter.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_parameter.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.btn_parameter.setStyleSheet(u"background-image: url(:/icons/images/icons/cil-equalizer.png)")

        self.verticalLayout_8.addWidget(self.btn_parameter)

        self.btn_graph = QPushButton(self.topMenu)
        self.btn_graph.setObjectName(u"btn_graph")
        sizePolicy.setHeightForWidth(self.btn_graph.sizePolicy().hasHeightForWidth())
        self.btn_graph.setSizePolicy(sizePolicy)
        self.btn_graph.setMinimumSize(QSize(0, 45))
        self.btn_graph.setFont(font)
        self.btn_graph.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_graph.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.btn_graph.setStyleSheet(u"background-image: url(:/icons/images/icons/cil-chart-line.png)")

        self.verticalLayout_8.addWidget(self.btn_graph)

        self.btn_setting = QPushButton(self.topMenu)
        self.btn_setting.setObjectName(u"btn_setting")
        sizePolicy.setHeightForWidth(self.btn_setting.sizePolicy().hasHeightForWidth())
        self.btn_setting.setSizePolicy(sizePolicy)
        self.btn_setting.setMinimumSize(QSize(0, 45))
        self.btn_setting.setFont(font)
        self.btn_setting.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_setting.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.btn_setting.setStyleSheet(u"background-image: url(:/icons/images/icons/cil-settings.png)")

        self.verticalLayout_8.addWidget(self.btn_setting)

        self.btn_download = QPushButton(self.topMenu)
        self.btn_download.setObjectName(u"btn_download")
        sizePolicy.setHeightForWidth(self.btn_download.sizePolicy().hasHeightForWidth())
        self.btn_download.setSizePolicy(sizePolicy)
        self.btn_download.setMinimumSize(QSize(0, 45))
        self.btn_download.setFont(font)
        self.btn_download.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_download.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.btn_download.setStyleSheet(u"background-image: url(:/icons/images/icons/cil-data-transfer-down.png)")

        self.verticalLayout_8.addWidget(self.btn_download)


        self.verticalMenuLayout.addWidget(self.topMenu, 0, Qt.AlignmentFlag.AlignTop)

        self.bottomMenu = QFrame(self.leftMenuFrame)
        self.bottomMenu.setObjectName(u"bottomMenu")
        self.bottomMenu.setFrameShape(QFrame.Shape.NoFrame)
        self.bottomMenu.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_9 = QVBoxLayout(self.bottomMenu)
        self.verticalLayout_9.setSpacing(0)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)

        self.verticalMenuLayout.addWidget(self.bottomMenu, 0, Qt.AlignmentFlag.AlignBottom)


        self.verticalLayout_3.addWidget(self.leftMenuFrame)


        self.appLayout.addWidget(self.leftMenuBg)

        self.contentBox = QFrame(self.bgApp)
        self.contentBox.setObjectName(u"contentBox")
        self.contentBox.setFrameShape(QFrame.Shape.NoFrame)
        self.contentBox.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.contentBox)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.contentTopBg = QFrame(self.contentBox)
        self.contentTopBg.setObjectName(u"contentTopBg")
        self.contentTopBg.setMinimumSize(QSize(0, 50))
        self.contentTopBg.setMaximumSize(QSize(16777215, 50))
        self.contentTopBg.setFrameShape(QFrame.Shape.NoFrame)
        self.contentTopBg.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.contentTopBg)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 10, 0)
        self.leftBox = QFrame(self.contentTopBg)
        self.leftBox.setObjectName(u"leftBox")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.leftBox.sizePolicy().hasHeightForWidth())
        self.leftBox.setSizePolicy(sizePolicy1)
        self.leftBox.setFrameShape(QFrame.Shape.NoFrame)
        self.leftBox.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.leftBox)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.titleRightInfo = QLabel(self.leftBox)
        self.titleRightInfo.setObjectName(u"titleRightInfo")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.titleRightInfo.sizePolicy().hasHeightForWidth())
        self.titleRightInfo.setSizePolicy(sizePolicy2)
        self.titleRightInfo.setMaximumSize(QSize(16777215, 45))
        self.titleRightInfo.setFont(font)
        self.titleRightInfo.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_3.addWidget(self.titleRightInfo)


        self.horizontalLayout.addWidget(self.leftBox)


        self.verticalLayout_2.addWidget(self.contentTopBg)

        self.contentBottom = QFrame(self.contentBox)
        self.contentBottom.setObjectName(u"contentBottom")
        self.contentBottom.setFrameShape(QFrame.Shape.NoFrame)
        self.contentBottom.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.contentBottom)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.content = QFrame(self.contentBottom)
        self.content.setObjectName(u"content")
        self.content.setFrameShape(QFrame.Shape.NoFrame)
        self.content.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.content)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.pagesContainer = QFrame(self.content)
        self.pagesContainer.setObjectName(u"pagesContainer")
        self.pagesContainer.setStyleSheet(u"")
        self.pagesContainer.setFrameShape(QFrame.Shape.NoFrame)
        self.pagesContainer.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_15 = QVBoxLayout(self.pagesContainer)
        self.verticalLayout_15.setSpacing(0)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.verticalLayout_15.setContentsMargins(10, 10, 10, 10)
        self.stackedWidget = QStackedWidget(self.pagesContainer)
        self.stackedWidget.setObjectName(u"stackedWidget")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.stackedWidget.sizePolicy().hasHeightForWidth())
        self.stackedWidget.setSizePolicy(sizePolicy3)
        self.stackedWidget.setStyleSheet(u"background: transparent;")
        self.home = QWidget()
        self.home.setObjectName(u"home")
        self.home.setStyleSheet(u"")
        self.horizontalLayout_8 = QHBoxLayout(self.home)
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.home_loge = QFrame(self.home)
        self.home_loge.setObjectName(u"home_loge")
        sizePolicy3.setHeightForWidth(self.home_loge.sizePolicy().hasHeightForWidth())
        self.home_loge.setSizePolicy(sizePolicy3)
        self.home_loge.setStyleSheet(u"")
        self.home_loge.setFrameShape(QFrame.Shape.StyledPanel)
        self.home_loge.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_29 = QVBoxLayout(self.home_loge)
        self.verticalLayout_29.setSpacing(0)
        self.verticalLayout_29.setObjectName(u"verticalLayout_29")
        self.verticalLayout_29.setContentsMargins(0, 0, 0, 0)
        self.logo_frame = QFrame(self.home_loge)
        self.logo_frame.setObjectName(u"logo_frame")
        self.logo_frame.setStyleSheet(u"")
        self.logo_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.logo_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_31 = QVBoxLayout(self.logo_frame)
        self.verticalLayout_31.setObjectName(u"verticalLayout_31")
        self.logo_label = QLabel(self.logo_frame)
        self.logo_label.setObjectName(u"logo_label")
        self.logo_label.setMaximumSize(QSize(16777215, 16777215))
        self.logo_label.setPixmap(QPixmap(u":/images/images/images/RichDoge.png"))
        self.logo_label.setScaledContents(False)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setWordWrap(False)

        self.verticalLayout_31.addWidget(self.logo_label)


        self.verticalLayout_29.addWidget(self.logo_frame)

        self.console_frame = QFrame(self.home_loge)
        self.console_frame.setObjectName(u"console_frame")
        sizePolicy2.setHeightForWidth(self.console_frame.sizePolicy().hasHeightForWidth())
        self.console_frame.setSizePolicy(sizePolicy2)
        self.console_frame.setMinimumSize(QSize(0, 0))
        self.console_frame.setMaximumSize(QSize(16777215, 16777215))
        self.console_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.console_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_30 = QVBoxLayout(self.console_frame)
        self.verticalLayout_30.setSpacing(10)
        self.verticalLayout_30.setObjectName(u"verticalLayout_30")
        self.verticalLayout_30.setContentsMargins(10, 10, 10, 20)
        self.label = QLabel(self.console_frame)
        self.label.setObjectName(u"label")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy4)

        self.verticalLayout_30.addWidget(self.label)

        self.ConsolePlainTextEdit = QPlainTextEdit(self.console_frame)
        self.ConsolePlainTextEdit.setObjectName(u"ConsolePlainTextEdit")
        self.ConsolePlainTextEdit.setEnabled(True)
        sizePolicy3.setHeightForWidth(self.ConsolePlainTextEdit.sizePolicy().hasHeightForWidth())
        self.ConsolePlainTextEdit.setSizePolicy(sizePolicy3)
        self.ConsolePlainTextEdit.setMinimumSize(QSize(0, 0))
        self.ConsolePlainTextEdit.setStyleSheet(u"background-color: rgb(33, 37, 43);\n"
"border: 2px solid rgb(52, 59, 72);")
        self.ConsolePlainTextEdit.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByKeyboard|Qt.TextInteractionFlag.TextSelectableByMouse)

        self.verticalLayout_30.addWidget(self.ConsolePlainTextEdit)

        self.clearLogPushButton = QPushButton(self.console_frame)
        self.clearLogPushButton.setObjectName(u"clearLogPushButton")
        self.clearLogPushButton.setMinimumSize(QSize(0, 40))
        self.clearLogPushButton.setStyleSheet(u"background-color: rgb(52, 59, 72);")

        self.verticalLayout_30.addWidget(self.clearLogPushButton)


        self.verticalLayout_29.addWidget(self.console_frame)


        self.horizontalLayout_8.addWidget(self.home_loge)

        self.home_frame = QFrame(self.home)
        self.home_frame.setObjectName(u"home_frame")
        self.home_frame.setMinimumSize(QSize(0, 0))
        self.home_frame.setMaximumSize(QSize(16777215, 16777215))
        self.home_frame.setStyleSheet(u"")
        self.home_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.home_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_13 = QVBoxLayout(self.home_frame)
        self.verticalLayout_13.setSpacing(0)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.verticalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.dial_frame = QFrame(self.home_frame)
        self.dial_frame.setObjectName(u"dial_frame")
        self.dial_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.dial_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_16 = QHBoxLayout(self.dial_frame)
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.cpu_frame = QFrame(self.dial_frame)
        self.cpu_frame.setObjectName(u"cpu_frame")
        self.cpu_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.cpu_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_27 = QVBoxLayout(self.cpu_frame)
        self.verticalLayout_27.setObjectName(u"verticalLayout_27")
        self.verticalLayout_27.setContentsMargins(0, 0, 0, 0)
        self.cpu_label = QLabel(self.cpu_frame)
        self.cpu_label.setObjectName(u"cpu_label")
        sizePolicy4.setHeightForWidth(self.cpu_label.sizePolicy().hasHeightForWidth())
        self.cpu_label.setSizePolicy(sizePolicy4)
        font3 = QFont()
        font3.setFamilies([u"Segoe UI"])
        font3.setBold(False)
        font3.setItalic(False)
        self.cpu_label.setFont(font3)
        self.cpu_label.setStyleSheet(u"font-size: 20px;")

        self.verticalLayout_27.addWidget(self.cpu_label)

        self.cpu_dial = QDial(self.cpu_frame)
        self.cpu_dial.setObjectName(u"cpu_dial")

        self.verticalLayout_27.addWidget(self.cpu_dial)

        self.cpu_name_label = QLabel(self.cpu_frame)
        self.cpu_name_label.setObjectName(u"cpu_name_label")
        sizePolicy4.setHeightForWidth(self.cpu_name_label.sizePolicy().hasHeightForWidth())
        self.cpu_name_label.setSizePolicy(sizePolicy4)

        self.verticalLayout_27.addWidget(self.cpu_name_label)


        self.horizontalLayout_16.addWidget(self.cpu_frame)

        self.gpu_frame = QFrame(self.dial_frame)
        self.gpu_frame.setObjectName(u"gpu_frame")
        self.gpu_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.gpu_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_28 = QVBoxLayout(self.gpu_frame)
        self.verticalLayout_28.setObjectName(u"verticalLayout_28")
        self.verticalLayout_28.setContentsMargins(0, 0, 0, 0)
        self.gpu_label = QLabel(self.gpu_frame)
        self.gpu_label.setObjectName(u"gpu_label")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.gpu_label.sizePolicy().hasHeightForWidth())
        self.gpu_label.setSizePolicy(sizePolicy5)
        self.gpu_label.setStyleSheet(u"font-size: 20px;")

        self.verticalLayout_28.addWidget(self.gpu_label)

        self.gpu_dial = QDial(self.gpu_frame)
        self.gpu_dial.setObjectName(u"gpu_dial")

        self.verticalLayout_28.addWidget(self.gpu_dial)

        self.gpu_name_label = QLabel(self.gpu_frame)
        self.gpu_name_label.setObjectName(u"gpu_name_label")
        sizePolicy4.setHeightForWidth(self.gpu_name_label.sizePolicy().hasHeightForWidth())
        self.gpu_name_label.setSizePolicy(sizePolicy4)

        self.verticalLayout_28.addWidget(self.gpu_name_label)


        self.horizontalLayout_16.addWidget(self.gpu_frame)


        self.verticalLayout_13.addWidget(self.dial_frame)

        self.setting_frame = QFrame(self.home_frame)
        self.setting_frame.setObjectName(u"setting_frame")
        self.setting_frame.setMaximumSize(QSize(16777215, 300))
        self.setting_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.setting_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_26 = QVBoxLayout(self.setting_frame)
        self.verticalLayout_26.setSpacing(0)
        self.verticalLayout_26.setObjectName(u"verticalLayout_26")
        self.verticalLayout_26.setContentsMargins(0, 0, 0, 0)
        self.cuda_label = QLabel(self.setting_frame)
        self.cuda_label.setObjectName(u"cuda_label")
        sizePolicy4.setHeightForWidth(self.cuda_label.sizePolicy().hasHeightForWidth())
        self.cuda_label.setSizePolicy(sizePolicy4)

        self.verticalLayout_26.addWidget(self.cuda_label)

        self.frame_2 = QFrame(self.setting_frame)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setStyleSheet(u"")
        self.frame_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_14 = QVBoxLayout(self.frame_2)
        self.verticalLayout_14.setSpacing(0)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.verticalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.frame_div_content_3 = QFrame(self.frame_2)
        self.frame_div_content_3.setObjectName(u"frame_div_content_3")
        sizePolicy3.setHeightForWidth(self.frame_div_content_3.sizePolicy().hasHeightForWidth())
        self.frame_div_content_3.setSizePolicy(sizePolicy3)
        self.frame_div_content_3.setMinimumSize(QSize(0, 0))
        self.frame_div_content_3.setMaximumSize(QSize(16777215, 100))
        self.frame_div_content_3.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_div_content_3.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_24 = QVBoxLayout(self.frame_div_content_3)
        self.verticalLayout_24.setSpacing(0)
        self.verticalLayout_24.setObjectName(u"verticalLayout_24")
        self.verticalLayout_24.setContentsMargins(0, 0, 0, 0)
        self.frame_title_wid_3 = QFrame(self.frame_div_content_3)
        self.frame_title_wid_3.setObjectName(u"frame_title_wid_3")
        self.frame_title_wid_3.setMaximumSize(QSize(16777215, 35))
        self.frame_title_wid_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_title_wid_3.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_25 = QVBoxLayout(self.frame_title_wid_3)
        self.verticalLayout_25.setObjectName(u"verticalLayout_25")
        self.verticalLayout_25.setContentsMargins(12, -1, -1, 0)
        self.labelBoxBlenderInstalation_3 = QLabel(self.frame_title_wid_3)
        self.labelBoxBlenderInstalation_3.setObjectName(u"labelBoxBlenderInstalation_3")
        self.labelBoxBlenderInstalation_3.setFont(font)
        self.labelBoxBlenderInstalation_3.setStyleSheet(u"")

        self.verticalLayout_25.addWidget(self.labelBoxBlenderInstalation_3)


        self.verticalLayout_24.addWidget(self.frame_title_wid_3)

        self.frame_content_wid_3 = QFrame(self.frame_div_content_3)
        self.frame_content_wid_3.setObjectName(u"frame_content_wid_3")
        self.frame_content_wid_3.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_content_wid_3.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_18 = QHBoxLayout(self.frame_content_wid_3)
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.gridLayout_4.setContentsMargins(-1, -1, -1, 0)
        self.File_Button_4 = QPushButton(self.frame_content_wid_3)
        self.File_Button_4.setObjectName(u"File_Button_4")
        sizePolicy5.setHeightForWidth(self.File_Button_4.sizePolicy().hasHeightForWidth())
        self.File_Button_4.setSizePolicy(sizePolicy5)
        self.File_Button_4.setMinimumSize(QSize(100, 30))
        self.File_Button_4.setFont(font)
        self.File_Button_4.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.File_Button_4.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        icon = QIcon()
        icon.addFile(u":/icons/images/icons/cil-folder-open.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.File_Button_4.setIcon(icon)

        self.gridLayout_4.addWidget(self.File_Button_4, 0, 1, 1, 1)

        self.filepath_lineEdit_2 = QLineEdit(self.frame_content_wid_3)
        self.filepath_lineEdit_2.setObjectName(u"filepath_lineEdit_2")
        sizePolicy5.setHeightForWidth(self.filepath_lineEdit_2.sizePolicy().hasHeightForWidth())
        self.filepath_lineEdit_2.setSizePolicy(sizePolicy5)
        self.filepath_lineEdit_2.setMinimumSize(QSize(0, 30))
        self.filepath_lineEdit_2.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.gridLayout_4.addWidget(self.filepath_lineEdit_2, 0, 0, 1, 1)


        self.horizontalLayout_18.addLayout(self.gridLayout_4)


        self.verticalLayout_24.addWidget(self.frame_content_wid_3)


        self.verticalLayout_14.addWidget(self.frame_div_content_3)

        self.testPushButton = QPushButton(self.frame_2)
        self.testPushButton.setObjectName(u"testPushButton")
        self.testPushButton.setMinimumSize(QSize(0, 40))
        self.testPushButton.setFont(font)
        self.testPushButton.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        icon1 = QIcon()
        icon1.addFile(u":/icons/images/icons/cil-gamepad.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.testPushButton.setIcon(icon1)

        self.verticalLayout_14.addWidget(self.testPushButton)


        self.verticalLayout_26.addWidget(self.frame_2)

        self.frame_6 = QFrame(self.setting_frame)
        self.frame_6.setObjectName(u"frame_6")
        sizePolicy3.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy3)
        self.frame_6.setMaximumSize(QSize(16777215, 80))
        self.frame_6.setStyleSheet(u"")
        self.frame_6.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_20 = QVBoxLayout(self.frame_6)
        self.verticalLayout_20.setSpacing(0)
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.verticalLayout_20.setContentsMargins(0, 0, 0, 0)
        self.learingPushButton = QPushButton(self.frame_6)
        self.learingPushButton.setObjectName(u"learingPushButton")
        self.learingPushButton.setMinimumSize(QSize(0, 40))
        self.learingPushButton.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        icon2 = QIcon()
        icon2.addFile(u":/icons/images/icons/cil-media-play.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.learingPushButton.setIcon(icon2)

        self.verticalLayout_20.addWidget(self.learingPushButton)


        self.verticalLayout_26.addWidget(self.frame_6)


        self.verticalLayout_13.addWidget(self.setting_frame)


        self.horizontalLayout_8.addWidget(self.home_frame)

        self.stackedWidget.addWidget(self.home)
        self.parameters_page = QWidget()
        self.parameters_page.setObjectName(u"parameters_page")
        sizePolicy3.setHeightForWidth(self.parameters_page.sizePolicy().hasHeightForWidth())
        self.parameters_page.setSizePolicy(sizePolicy3)
        self.verticalLayout_11 = QVBoxLayout(self.parameters_page)
        self.verticalLayout_11.setSpacing(0)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.row_5 = QFrame(self.parameters_page)
        self.row_5.setObjectName(u"row_5")
        sizePolicy3.setHeightForWidth(self.row_5.sizePolicy().hasHeightForWidth())
        self.row_5.setSizePolicy(sizePolicy3)
        self.row_5.setMinimumSize(QSize(0, 0))
        self.row_5.setFrameShape(QFrame.Shape.StyledPanel)
        self.row_5.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_33 = QVBoxLayout(self.row_5)
        self.verticalLayout_33.setSpacing(10)
        self.verticalLayout_33.setObjectName(u"verticalLayout_33")
        self.verticalLayout_33.setContentsMargins(0, 0, 0, 0)
        self.frame_div_content_2 = QFrame(self.row_5)
        self.frame_div_content_2.setObjectName(u"frame_div_content_2")
        sizePolicy3.setHeightForWidth(self.frame_div_content_2.sizePolicy().hasHeightForWidth())
        self.frame_div_content_2.setSizePolicy(sizePolicy3)
        self.frame_div_content_2.setMinimumSize(QSize(0, 0))
        self.frame_div_content_2.setMaximumSize(QSize(16777215, 110))
        self.frame_div_content_2.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_div_content_2.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_22 = QVBoxLayout(self.frame_div_content_2)
        self.verticalLayout_22.setSpacing(0)
        self.verticalLayout_22.setObjectName(u"verticalLayout_22")
        self.verticalLayout_22.setContentsMargins(0, 0, 0, 0)
        self.frame_content_wid_2 = QFrame(self.frame_div_content_2)
        self.frame_content_wid_2.setObjectName(u"frame_content_wid_2")
        sizePolicy4.setHeightForWidth(self.frame_content_wid_2.sizePolicy().hasHeightForWidth())
        self.frame_content_wid_2.setSizePolicy(sizePolicy4)
        self.frame_content_wid_2.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_content_wid_2.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_10 = QHBoxLayout(self.frame_content_wid_2)
        self.horizontalLayout_10.setSpacing(0)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setHorizontalSpacing(0)
        self.gridLayout_3.setVerticalSpacing(10)
        self.gridLayout_3.setContentsMargins(-1, -1, -1, 0)
        self.File_Button_3 = QPushButton(self.frame_content_wid_2)
        self.File_Button_3.setObjectName(u"File_Button_3")
        self.File_Button_3.setMinimumSize(QSize(150, 30))
        self.File_Button_3.setFont(font)
        self.File_Button_3.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.File_Button_3.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        self.File_Button_3.setIcon(icon)

        self.gridLayout_3.addWidget(self.File_Button_3, 1, 1, 1, 1)

        self.filepath_lineEdit = QLineEdit(self.frame_content_wid_2)
        self.filepath_lineEdit.setObjectName(u"filepath_lineEdit")
        self.filepath_lineEdit.setMinimumSize(QSize(0, 30))
        self.filepath_lineEdit.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.gridLayout_3.addWidget(self.filepath_lineEdit, 1, 0, 1, 1)

        self.labelBoxBlenderInstalation_2 = QLabel(self.frame_content_wid_2)
        self.labelBoxBlenderInstalation_2.setObjectName(u"labelBoxBlenderInstalation_2")
        sizePolicy3.setHeightForWidth(self.labelBoxBlenderInstalation_2.sizePolicy().hasHeightForWidth())
        self.labelBoxBlenderInstalation_2.setSizePolicy(sizePolicy3)
        self.labelBoxBlenderInstalation_2.setFont(font)
        self.labelBoxBlenderInstalation_2.setStyleSheet(u"")

        self.gridLayout_3.addWidget(self.labelBoxBlenderInstalation_2, 0, 0, 1, 1)


        self.horizontalLayout_10.addLayout(self.gridLayout_3)


        self.verticalLayout_22.addWidget(self.frame_content_wid_2)


        self.verticalLayout_33.addWidget(self.frame_div_content_2)

        self.tableWidget_2 = QTableWidget(self.row_5)
        if (self.tableWidget_2.columnCount() < 5):
            self.tableWidget_2.setColumnCount(5)
        __qtablewidgetitem = QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        if (self.tableWidget_2.rowCount() < 30):
            self.tableWidget_2.setRowCount(30)
        font4 = QFont()
        font4.setFamilies([u"Segoe UI"])
        __qtablewidgetitem3 = QTableWidgetItem()
        __qtablewidgetitem3.setFont(font4);
        self.tableWidget_2.setVerticalHeaderItem(0, __qtablewidgetitem3)
        __qtablewidgetitem4 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(1, __qtablewidgetitem4)
        __qtablewidgetitem5 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(2, __qtablewidgetitem5)
        __qtablewidgetitem6 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(3, __qtablewidgetitem6)
        __qtablewidgetitem7 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(4, __qtablewidgetitem7)
        __qtablewidgetitem8 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(5, __qtablewidgetitem8)
        __qtablewidgetitem9 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(6, __qtablewidgetitem9)
        __qtablewidgetitem10 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(7, __qtablewidgetitem10)
        __qtablewidgetitem11 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(8, __qtablewidgetitem11)
        __qtablewidgetitem12 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(9, __qtablewidgetitem12)
        __qtablewidgetitem13 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(10, __qtablewidgetitem13)
        __qtablewidgetitem14 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(11, __qtablewidgetitem14)
        __qtablewidgetitem15 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(12, __qtablewidgetitem15)
        __qtablewidgetitem16 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(13, __qtablewidgetitem16)
        __qtablewidgetitem17 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(14, __qtablewidgetitem17)
        __qtablewidgetitem18 = QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(15, __qtablewidgetitem18)
        __qtablewidgetitem19 = QTableWidgetItem()
        self.tableWidget_2.setItem(0, 0, __qtablewidgetitem19)
        __qtablewidgetitem20 = QTableWidgetItem()
        self.tableWidget_2.setItem(0, 1, __qtablewidgetitem20)
        __qtablewidgetitem21 = QTableWidgetItem()
        self.tableWidget_2.setItem(0, 2, __qtablewidgetitem21)
        __qtablewidgetitem22 = QTableWidgetItem()
        self.tableWidget_2.setItem(0, 3, __qtablewidgetitem22)
        __qtablewidgetitem23 = QTableWidgetItem()
        self.tableWidget_2.setItem(0, 4, __qtablewidgetitem23)
        self.tableWidget_2.setObjectName(u"tableWidget_2")
        sizePolicy3.setHeightForWidth(self.tableWidget_2.sizePolicy().hasHeightForWidth())
        self.tableWidget_2.setSizePolicy(sizePolicy3)
        self.tableWidget_2.setMinimumSize(QSize(0, 0))
        self.tableWidget_2.setMaximumSize(QSize(16777215, 300))
        palette = QPalette()
        brush = QBrush(QColor(221, 221, 221, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.WindowText, brush)
        brush1 = QBrush(QColor(33, 37, 43, 255))
        brush1.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Button, brush1)
        palette.setBrush(QPalette.Active, QPalette.Text, brush)
        palette.setBrush(QPalette.Active, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Active, QPalette.Base, brush1)
        palette.setBrush(QPalette.Active, QPalette.Window, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Active, QPalette.PlaceholderText, brush)
#endif
        palette.setBrush(QPalette.Inactive, QPalette.WindowText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Text, brush)
        palette.setBrush(QPalette.Inactive, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Window, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush)
#endif
        palette.setBrush(QPalette.Disabled, QPalette.WindowText, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Text, brush)
        palette.setBrush(QPalette.Disabled, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Window, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush)
#endif
        self.tableWidget_2.setPalette(palette)
        self.tableWidget_2.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.tableWidget_2.setFrameShape(QFrame.Shape.NoFrame)
        self.tableWidget_2.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.tableWidget_2.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.tableWidget_2.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tableWidget_2.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tableWidget_2.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tableWidget_2.setShowGrid(True)
        self.tableWidget_2.setGridStyle(Qt.PenStyle.SolidLine)
        self.tableWidget_2.setSortingEnabled(False)
        self.tableWidget_2.setRowCount(30)
        self.tableWidget_2.setColumnCount(5)
        self.tableWidget_2.horizontalHeader().setVisible(False)
        self.tableWidget_2.horizontalHeader().setCascadingSectionResizes(True)
        self.tableWidget_2.horizontalHeader().setDefaultSectionSize(200)
        self.tableWidget_2.horizontalHeader().setStretchLastSection(True)
        self.tableWidget_2.verticalHeader().setVisible(False)
        self.tableWidget_2.verticalHeader().setCascadingSectionResizes(False)
        self.tableWidget_2.verticalHeader().setHighlightSections(False)
        self.tableWidget_2.verticalHeader().setStretchLastSection(True)

        self.verticalLayout_33.addWidget(self.tableWidget_2)


        self.verticalLayout_11.addWidget(self.row_5)

        self.frame_18 = QFrame(self.parameters_page)
        self.frame_18.setObjectName(u"frame_18")
        sizePolicy3.setHeightForWidth(self.frame_18.sizePolicy().hasHeightForWidth())
        self.frame_18.setSizePolicy(sizePolicy3)
        self.frame_18.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_18.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_49 = QVBoxLayout(self.frame_18)
        self.verticalLayout_49.setSpacing(10)
        self.verticalLayout_49.setObjectName(u"verticalLayout_49")
        self.verticalLayout_49.setContentsMargins(0, 0, 0, 0)
        self.frame_div_content_6 = QFrame(self.frame_18)
        self.frame_div_content_6.setObjectName(u"frame_div_content_6")
        sizePolicy4.setHeightForWidth(self.frame_div_content_6.sizePolicy().hasHeightForWidth())
        self.frame_div_content_6.setSizePolicy(sizePolicy4)
        self.frame_div_content_6.setMinimumSize(QSize(0, 0))
        self.frame_div_content_6.setMaximumSize(QSize(16777215, 110))
        self.frame_div_content_6.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_div_content_6.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_46 = QVBoxLayout(self.frame_div_content_6)
        self.verticalLayout_46.setSpacing(0)
        self.verticalLayout_46.setObjectName(u"verticalLayout_46")
        self.verticalLayout_46.setContentsMargins(0, 0, 0, 0)
        self.frame_content_wid_7 = QFrame(self.frame_div_content_6)
        self.frame_content_wid_7.setObjectName(u"frame_content_wid_7")
        sizePolicy4.setHeightForWidth(self.frame_content_wid_7.sizePolicy().hasHeightForWidth())
        self.frame_content_wid_7.setSizePolicy(sizePolicy4)
        self.frame_content_wid_7.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_content_wid_7.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_33 = QHBoxLayout(self.frame_content_wid_7)
        self.horizontalLayout_33.setObjectName(u"horizontalLayout_33")
        self.horizontalLayout_33.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_8 = QGridLayout()
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.gridLayout_8.setHorizontalSpacing(0)
        self.gridLayout_8.setVerticalSpacing(10)
        self.gridLayout_8.setContentsMargins(-1, -1, -1, 0)
        self.filepath_lineEdit_6 = QLineEdit(self.frame_content_wid_7)
        self.filepath_lineEdit_6.setObjectName(u"filepath_lineEdit_6")
        self.filepath_lineEdit_6.setMinimumSize(QSize(0, 30))
        self.filepath_lineEdit_6.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.gridLayout_8.addWidget(self.filepath_lineEdit_6, 3, 0, 1, 1)

        self.File_Button_8 = QPushButton(self.frame_content_wid_7)
        self.File_Button_8.setObjectName(u"File_Button_8")
        self.File_Button_8.setMinimumSize(QSize(150, 30))
        self.File_Button_8.setFont(font)
        self.File_Button_8.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.File_Button_8.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        self.File_Button_8.setIcon(icon)

        self.gridLayout_8.addWidget(self.File_Button_8, 3, 1, 1, 1)

        self.labelBoxBlenderInstalation_7 = QLabel(self.frame_content_wid_7)
        self.labelBoxBlenderInstalation_7.setObjectName(u"labelBoxBlenderInstalation_7")
        sizePolicy3.setHeightForWidth(self.labelBoxBlenderInstalation_7.sizePolicy().hasHeightForWidth())
        self.labelBoxBlenderInstalation_7.setSizePolicy(sizePolicy3)
        self.labelBoxBlenderInstalation_7.setFont(font)
        self.labelBoxBlenderInstalation_7.setStyleSheet(u"")

        self.gridLayout_8.addWidget(self.labelBoxBlenderInstalation_7, 2, 0, 1, 1)


        self.horizontalLayout_33.addLayout(self.gridLayout_8)


        self.verticalLayout_46.addWidget(self.frame_content_wid_7)


        self.verticalLayout_49.addWidget(self.frame_div_content_6)

        self.tableWidget_3 = QTableWidget(self.frame_18)
        if (self.tableWidget_3.columnCount() < 5):
            self.tableWidget_3.setColumnCount(5)
        __qtablewidgetitem24 = QTableWidgetItem()
        self.tableWidget_3.setHorizontalHeaderItem(0, __qtablewidgetitem24)
        __qtablewidgetitem25 = QTableWidgetItem()
        self.tableWidget_3.setHorizontalHeaderItem(1, __qtablewidgetitem25)
        __qtablewidgetitem26 = QTableWidgetItem()
        self.tableWidget_3.setHorizontalHeaderItem(2, __qtablewidgetitem26)
        if (self.tableWidget_3.rowCount() < 30):
            self.tableWidget_3.setRowCount(30)
        __qtablewidgetitem27 = QTableWidgetItem()
        __qtablewidgetitem27.setFont(font4);
        self.tableWidget_3.setVerticalHeaderItem(0, __qtablewidgetitem27)
        __qtablewidgetitem28 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(1, __qtablewidgetitem28)
        __qtablewidgetitem29 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(2, __qtablewidgetitem29)
        __qtablewidgetitem30 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(3, __qtablewidgetitem30)
        __qtablewidgetitem31 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(4, __qtablewidgetitem31)
        __qtablewidgetitem32 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(5, __qtablewidgetitem32)
        __qtablewidgetitem33 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(6, __qtablewidgetitem33)
        __qtablewidgetitem34 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(7, __qtablewidgetitem34)
        __qtablewidgetitem35 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(8, __qtablewidgetitem35)
        __qtablewidgetitem36 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(9, __qtablewidgetitem36)
        __qtablewidgetitem37 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(10, __qtablewidgetitem37)
        __qtablewidgetitem38 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(11, __qtablewidgetitem38)
        __qtablewidgetitem39 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(12, __qtablewidgetitem39)
        __qtablewidgetitem40 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(13, __qtablewidgetitem40)
        __qtablewidgetitem41 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(14, __qtablewidgetitem41)
        __qtablewidgetitem42 = QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(15, __qtablewidgetitem42)
        __qtablewidgetitem43 = QTableWidgetItem()
        self.tableWidget_3.setItem(0, 0, __qtablewidgetitem43)
        __qtablewidgetitem44 = QTableWidgetItem()
        self.tableWidget_3.setItem(0, 1, __qtablewidgetitem44)
        __qtablewidgetitem45 = QTableWidgetItem()
        self.tableWidget_3.setItem(0, 2, __qtablewidgetitem45)
        __qtablewidgetitem46 = QTableWidgetItem()
        self.tableWidget_3.setItem(0, 3, __qtablewidgetitem46)
        __qtablewidgetitem47 = QTableWidgetItem()
        self.tableWidget_3.setItem(0, 4, __qtablewidgetitem47)
        self.tableWidget_3.setObjectName(u"tableWidget_3")
        sizePolicy3.setHeightForWidth(self.tableWidget_3.sizePolicy().hasHeightForWidth())
        self.tableWidget_3.setSizePolicy(sizePolicy3)
        self.tableWidget_3.setMinimumSize(QSize(0, 0))
        self.tableWidget_3.setMaximumSize(QSize(16777215, 300))
        palette1 = QPalette()
        palette1.setBrush(QPalette.Active, QPalette.WindowText, brush)
        palette1.setBrush(QPalette.Active, QPalette.Button, brush1)
        palette1.setBrush(QPalette.Active, QPalette.Text, brush)
        palette1.setBrush(QPalette.Active, QPalette.ButtonText, brush)
        palette1.setBrush(QPalette.Active, QPalette.Base, brush1)
        palette1.setBrush(QPalette.Active, QPalette.Window, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.Active, QPalette.PlaceholderText, brush)
#endif
        palette1.setBrush(QPalette.Inactive, QPalette.WindowText, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette1.setBrush(QPalette.Inactive, QPalette.Text, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.ButtonText, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        palette1.setBrush(QPalette.Inactive, QPalette.Window, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush)
#endif
        palette1.setBrush(QPalette.Disabled, QPalette.WindowText, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette1.setBrush(QPalette.Disabled, QPalette.Text, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.ButtonText, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette1.setBrush(QPalette.Disabled, QPalette.Window, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush)
#endif
        self.tableWidget_3.setPalette(palette1)
        self.tableWidget_3.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.tableWidget_3.setFrameShape(QFrame.Shape.NoFrame)
        self.tableWidget_3.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.tableWidget_3.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.tableWidget_3.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tableWidget_3.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tableWidget_3.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tableWidget_3.setShowGrid(True)
        self.tableWidget_3.setGridStyle(Qt.PenStyle.SolidLine)
        self.tableWidget_3.setSortingEnabled(False)
        self.tableWidget_3.setRowCount(30)
        self.tableWidget_3.setColumnCount(5)
        self.tableWidget_3.horizontalHeader().setVisible(False)
        self.tableWidget_3.horizontalHeader().setCascadingSectionResizes(True)
        self.tableWidget_3.horizontalHeader().setDefaultSectionSize(200)
        self.tableWidget_3.horizontalHeader().setStretchLastSection(True)
        self.tableWidget_3.verticalHeader().setVisible(False)
        self.tableWidget_3.verticalHeader().setCascadingSectionResizes(False)
        self.tableWidget_3.verticalHeader().setHighlightSections(False)
        self.tableWidget_3.verticalHeader().setStretchLastSection(True)

        self.verticalLayout_49.addWidget(self.tableWidget_3)


        self.verticalLayout_11.addWidget(self.frame_18)

        self.row_7 = QFrame(self.parameters_page)
        self.row_7.setObjectName(u"row_7")
        sizePolicy3.setHeightForWidth(self.row_7.sizePolicy().hasHeightForWidth())
        self.row_7.setSizePolicy(sizePolicy3)
        self.row_7.setMinimumSize(QSize(0, 250))
        self.row_7.setFrameShape(QFrame.Shape.StyledPanel)
        self.row_7.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_34 = QHBoxLayout(self.row_7)
        self.horizontalLayout_34.setObjectName(u"horizontalLayout_34")
        self.frame_19 = QFrame(self.row_7)
        self.frame_19.setObjectName(u"frame_19")
        self.frame_19.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_19.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_23 = QVBoxLayout(self.frame_19)
        self.verticalLayout_23.setObjectName(u"verticalLayout_23")
        self.label_17 = QLabel(self.frame_19)
        self.label_17.setObjectName(u"label_17")

        self.verticalLayout_23.addWidget(self.label_17)

        self.dataListWidget = QListWidget(self.frame_19)
        self.dataListWidget.setObjectName(u"dataListWidget")
        self.dataListWidget.setMinimumSize(QSize(0, 250))
        self.dataListWidget.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.verticalLayout_23.addWidget(self.dataListWidget)


        self.horizontalLayout_34.addWidget(self.frame_19)

        self.frame_20 = QFrame(self.row_7)
        self.frame_20.setObjectName(u"frame_20")
        self.frame_20.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_20.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_40 = QVBoxLayout(self.frame_20)
        self.verticalLayout_40.setObjectName(u"verticalLayout_40")
        self.label_19 = QLabel(self.frame_20)
        self.label_19.setObjectName(u"label_19")

        self.verticalLayout_40.addWidget(self.label_19)

        self.dataListWidget_2 = QListWidget(self.frame_20)
        self.dataListWidget_2.setObjectName(u"dataListWidget_2")
        self.dataListWidget_2.setMinimumSize(QSize(0, 250))
        self.dataListWidget_2.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.verticalLayout_40.addWidget(self.dataListWidget_2)


        self.horizontalLayout_34.addWidget(self.frame_20)


        self.verticalLayout_11.addWidget(self.row_7)

        self.frame = QFrame(self.parameters_page)
        self.frame.setObjectName(u"frame")
        sizePolicy3.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy3)
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.frame)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.SavePerarametersButton = QPushButton(self.frame)
        self.SavePerarametersButton.setObjectName(u"SavePerarametersButton")
        self.SavePerarametersButton.setMinimumSize(QSize(150, 30))
        self.SavePerarametersButton.setFont(font)
        self.SavePerarametersButton.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.SavePerarametersButton.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        icon3 = QIcon()
        icon3.addFile(u":/icons/images/icons/cil-save.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.SavePerarametersButton.setIcon(icon3)

        self.horizontalLayout_6.addWidget(self.SavePerarametersButton)


        self.verticalLayout_11.addWidget(self.frame)

        self.stackedWidget.addWidget(self.parameters_page)
        self.download_page = QWidget()
        self.download_page.setObjectName(u"download_page")
        self.verticalLayout_12 = QVBoxLayout(self.download_page)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.stocklistFrame = QFrame(self.download_page)
        self.stocklistFrame.setObjectName(u"stocklistFrame")
        self.stocklistFrame.setFrameShape(QFrame.Shape.StyledPanel)
        self.stocklistFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_39 = QVBoxLayout(self.stocklistFrame)
        self.verticalLayout_39.setObjectName(u"verticalLayout_39")
        self.label_18 = QLabel(self.stocklistFrame)
        self.label_18.setObjectName(u"label_18")

        self.verticalLayout_39.addWidget(self.label_18)

        self.stockCodeListWidget = QListWidget(self.stocklistFrame)
        self.stockCodeListWidget.setObjectName(u"stockCodeListWidget")
        self.stockCodeListWidget.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.verticalLayout_39.addWidget(self.stockCodeListWidget)

        self.frame_15 = QFrame(self.stocklistFrame)
        self.frame_15.setObjectName(u"frame_15")
        self.frame_15.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_15.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_26 = QHBoxLayout(self.frame_15)
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.label_14 = QLabel(self.frame_15)
        self.label_14.setObjectName(u"label_14")

        self.horizontalLayout_26.addWidget(self.label_14)

        self.lineEdit_2 = QLineEdit(self.frame_15)
        self.lineEdit_2.setObjectName(u"lineEdit_2")
        self.lineEdit_2.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.horizontalLayout_26.addWidget(self.lineEdit_2)

        self.addStockCodePushButton = QPushButton(self.frame_15)
        self.addStockCodePushButton.setObjectName(u"addStockCodePushButton")
        self.addStockCodePushButton.setMinimumSize(QSize(100, 0))
        self.addStockCodePushButton.setStyleSheet(u"background-color: rgb(52, 59, 72);")

        self.horizontalLayout_26.addWidget(self.addStockCodePushButton)

        self.removeStockCodePushButton = QPushButton(self.frame_15)
        self.removeStockCodePushButton.setObjectName(u"removeStockCodePushButton")
        self.removeStockCodePushButton.setMinimumSize(QSize(100, 0))
        self.removeStockCodePushButton.setStyleSheet(u"background-color: rgb(52, 59, 72);")

        self.horizontalLayout_26.addWidget(self.removeStockCodePushButton)


        self.verticalLayout_39.addWidget(self.frame_15)


        self.verticalLayout_12.addWidget(self.stocklistFrame)

        self.frame_17 = QFrame(self.download_page)
        self.frame_17.setObjectName(u"frame_17")
        self.frame_17.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_17.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_31 = QHBoxLayout(self.frame_17)
        self.horizontalLayout_31.setObjectName(u"horizontalLayout_31")
        self.dateEditFrame = QFrame(self.frame_17)
        self.dateEditFrame.setObjectName(u"dateEditFrame")
        self.dateEditFrame.setFrameShape(QFrame.Shape.StyledPanel)
        self.dateEditFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_27 = QHBoxLayout(self.dateEditFrame)
        self.horizontalLayout_27.setObjectName(u"horizontalLayout_27")
        self.label_16 = QLabel(self.dateEditFrame)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_27.addWidget(self.label_16)

        self.LastDateEdit = QDateEdit(self.dateEditFrame)
        self.LastDateEdit.setObjectName(u"LastDateEdit")
        self.LastDateEdit.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.LastDateEdit.setDateTime(QDateTime(QDate(2025, 1, 28), QTime(9, 0, 0)))

        self.horizontalLayout_27.addWidget(self.LastDateEdit)


        self.horizontalLayout_31.addWidget(self.dateEditFrame)

        self.countFrame = QFrame(self.frame_17)
        self.countFrame.setObjectName(u"countFrame")
        self.countFrame.setFrameShape(QFrame.Shape.StyledPanel)
        self.countFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_28 = QHBoxLayout(self.countFrame)
        self.horizontalLayout_28.setObjectName(u"horizontalLayout_28")
        self.label_15 = QLabel(self.countFrame)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_28.addWidget(self.label_15)

        self.countSpinBox = QSpinBox(self.countFrame)
        self.countSpinBox.setObjectName(u"countSpinBox")
        self.countSpinBox.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.countSpinBox.setMaximum(9999)
        self.countSpinBox.setValue(1500)

        self.horizontalLayout_28.addWidget(self.countSpinBox)


        self.horizontalLayout_31.addWidget(self.countFrame)

        self.frame_14 = QFrame(self.frame_17)
        self.frame_14.setObjectName(u"frame_14")
        self.frame_14.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_14.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_30 = QHBoxLayout(self.frame_14)
        self.horizontalLayout_30.setObjectName(u"horizontalLayout_30")
        self.isDataFileRemoveCheckBox = QCheckBox(self.frame_14)
        self.isDataFileRemoveCheckBox.setObjectName(u"isDataFileRemoveCheckBox")
        self.isDataFileRemoveCheckBox.setAutoFillBackground(False)
        self.isDataFileRemoveCheckBox.setStyleSheet(u"")
        self.isDataFileRemoveCheckBox.setChecked(True)
        self.isDataFileRemoveCheckBox.setTristate(False)

        self.horizontalLayout_30.addWidget(self.isDataFileRemoveCheckBox)


        self.horizontalLayout_31.addWidget(self.frame_14)

        self.DownloadPushButton = QPushButton(self.frame_17)
        self.DownloadPushButton.setObjectName(u"DownloadPushButton")
        self.DownloadPushButton.setStyleSheet(u"background-color: rgb(52, 59, 72);")

        self.horizontalLayout_31.addWidget(self.DownloadPushButton)


        self.verticalLayout_12.addWidget(self.frame_17)

        self.downloadProgressBar = QProgressBar(self.download_page)
        self.downloadProgressBar.setObjectName(u"downloadProgressBar")
        self.downloadProgressBar.setStyleSheet(u"QProgressBar{    \n"
"	background-color: rgb(98, 114, 164);\n"
"    color:rgb(200,200,200);\n"
"    border-style: none;\n"
"    border-bottom-right-radius: 5px;\n"
"    border-bottom-left-radius: 5px;\n"
"    border-top-right-radius: 5px;\n"
"    border-top-left-radius: 5px;\n"
"    text-align: center;\n"
"}\n"
"\n"
"QProgressBar::chunk{\n"
"    border-bottom-right-radius: 5px;\n"
"    border-bottom-left-radius: 5px;\n"
"    border-top-right-radius: 5px;\n"
"    border-top-left-radius: 5px;\n"
"    background-color: qlineargradient(spread:pad, x1:0, y1:0.511364, x2:1, y2:0.523, stop:0 rgba(254, 121, 199, 255), stop:1 rgba(170, 85, 255, 255));\n"
"}")
        self.downloadProgressBar.setValue(0)

        self.verticalLayout_12.addWidget(self.downloadProgressBar)

        self.stackedWidget.addWidget(self.download_page)
        self.kis_devlp_page = QWidget()
        self.kis_devlp_page.setObjectName(u"kis_devlp_page")
        self.verticalLayout_37 = QVBoxLayout(self.kis_devlp_page)
        self.verticalLayout_37.setObjectName(u"verticalLayout_37")
        self.verticalLayout_37.setContentsMargins(9, -1, -1, -1)
        self.row_6 = QFrame(self.kis_devlp_page)
        self.row_6.setObjectName(u"row_6")
        sizePolicy3.setHeightForWidth(self.row_6.sizePolicy().hasHeightForWidth())
        self.row_6.setSizePolicy(sizePolicy3)
        self.row_6.setMaximumSize(QSize(16777215, 100))
        self.row_6.setFrameShape(QFrame.Shape.StyledPanel)
        self.row_6.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_34 = QVBoxLayout(self.row_6)
        self.verticalLayout_34.setSpacing(0)
        self.verticalLayout_34.setObjectName(u"verticalLayout_34")
        self.verticalLayout_34.setContentsMargins(0, 0, 0, 0)
        self.frame_div_content_4 = QFrame(self.row_6)
        self.frame_div_content_4.setObjectName(u"frame_div_content_4")
        self.frame_div_content_4.setMinimumSize(QSize(0, 110))
        self.frame_div_content_4.setMaximumSize(QSize(16777215, 110))
        self.frame_div_content_4.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_div_content_4.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_35 = QVBoxLayout(self.frame_div_content_4)
        self.verticalLayout_35.setSpacing(0)
        self.verticalLayout_35.setObjectName(u"verticalLayout_35")
        self.verticalLayout_35.setContentsMargins(0, 0, 0, 0)
        self.frame_title_wid_4 = QFrame(self.frame_div_content_4)
        self.frame_title_wid_4.setObjectName(u"frame_title_wid_4")
        self.frame_title_wid_4.setMaximumSize(QSize(16777215, 35))
        self.frame_title_wid_4.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_title_wid_4.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_36 = QVBoxLayout(self.frame_title_wid_4)
        self.verticalLayout_36.setObjectName(u"verticalLayout_36")
        self.labelBoxBlenderInstalation_4 = QLabel(self.frame_title_wid_4)
        self.labelBoxBlenderInstalation_4.setObjectName(u"labelBoxBlenderInstalation_4")
        self.labelBoxBlenderInstalation_4.setFont(font)
        self.labelBoxBlenderInstalation_4.setStyleSheet(u"")

        self.verticalLayout_36.addWidget(self.labelBoxBlenderInstalation_4)


        self.verticalLayout_35.addWidget(self.frame_title_wid_4)

        self.frame_content_wid_4 = QFrame(self.frame_div_content_4)
        self.frame_content_wid_4.setObjectName(u"frame_content_wid_4")
        self.frame_content_wid_4.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_content_wid_4.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_14 = QHBoxLayout(self.frame_content_wid_4)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.gridLayout_5 = QGridLayout()
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setContentsMargins(-1, -1, -1, 0)
        self.filepath_lineEdit_3 = QLineEdit(self.frame_content_wid_4)
        self.filepath_lineEdit_3.setObjectName(u"filepath_lineEdit_3")
        self.filepath_lineEdit_3.setMinimumSize(QSize(0, 30))
        self.filepath_lineEdit_3.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.gridLayout_5.addWidget(self.filepath_lineEdit_3, 0, 0, 1, 1)

        self.File_Button_5 = QPushButton(self.frame_content_wid_4)
        self.File_Button_5.setObjectName(u"File_Button_5")
        self.File_Button_5.setMinimumSize(QSize(150, 30))
        self.File_Button_5.setFont(font)
        self.File_Button_5.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.File_Button_5.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        self.File_Button_5.setIcon(icon)

        self.gridLayout_5.addWidget(self.File_Button_5, 0, 1, 1, 1)

        self.labelVersion_5 = QLabel(self.frame_content_wid_4)
        self.labelVersion_5.setObjectName(u"labelVersion_5")
        self.labelVersion_5.setStyleSheet(u"color: rgb(113, 126, 149);")
        self.labelVersion_5.setLineWidth(1)
        self.labelVersion_5.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_5.addWidget(self.labelVersion_5, 1, 0, 1, 2)


        self.horizontalLayout_14.addLayout(self.gridLayout_5)


        self.verticalLayout_35.addWidget(self.frame_content_wid_4)


        self.verticalLayout_34.addWidget(self.frame_div_content_4)


        self.verticalLayout_37.addWidget(self.row_6)

        self.frame_5 = QFrame(self.kis_devlp_page)
        self.frame_5.setObjectName(u"frame_5")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.frame_5.sizePolicy().hasHeightForWidth())
        self.frame_5.setSizePolicy(sizePolicy6)
        self.frame_5.setStyleSheet(u"QFrame{\n"
"	border-radius: 10px;\n"
"}")
        self.frame_5.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_38 = QVBoxLayout(self.frame_5)
        self.verticalLayout_38.setObjectName(u"verticalLayout_38")
        self.frame_4 = QFrame(self.frame_5)
        self.frame_4.setObjectName(u"frame_4")
        sizePolicy3.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy3)
        self.frame_4.setStyleSheet(u"")
        self.frame_4.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_17 = QHBoxLayout(self.frame_4)
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")

        self.verticalLayout_38.addWidget(self.frame_4)

        self.label_9 = QLabel(self.frame_5)
        self.label_9.setObjectName(u"label_9")
        sizePolicy4.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy4)

        self.verticalLayout_38.addWidget(self.label_9)

        self.frame_7 = QFrame(self.frame_5)
        self.frame_7.setObjectName(u"frame_7")
        self.frame_7.setStyleSheet(u"")
        self.frame_7.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_7.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_19 = QHBoxLayout(self.frame_7)
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.label_2 = QLabel(self.frame_7)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_19.addWidget(self.label_2)

        self.paper_app_lineEdit = QLineEdit(self.frame_7)
        self.paper_app_lineEdit.setObjectName(u"paper_app_lineEdit")
        self.paper_app_lineEdit.setMinimumSize(QSize(0, 30))
        self.paper_app_lineEdit.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.horizontalLayout_19.addWidget(self.paper_app_lineEdit)


        self.verticalLayout_38.addWidget(self.frame_7)

        self.frame_8 = QFrame(self.frame_5)
        self.frame_8.setObjectName(u"frame_8")
        self.frame_8.setStyleSheet(u"")
        self.frame_8.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_8.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_20 = QHBoxLayout(self.frame_8)
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.label_3 = QLabel(self.frame_8)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_20.addWidget(self.label_3)

        self.paper_sec_lineEdit = QLineEdit(self.frame_8)
        self.paper_sec_lineEdit.setObjectName(u"paper_sec_lineEdit")
        self.paper_sec_lineEdit.setMinimumSize(QSize(0, 30))
        self.paper_sec_lineEdit.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.horizontalLayout_20.addWidget(self.paper_sec_lineEdit)


        self.verticalLayout_38.addWidget(self.frame_8)

        self.label_10 = QLabel(self.frame_5)
        self.label_10.setObjectName(u"label_10")
        sizePolicy4.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy4)

        self.verticalLayout_38.addWidget(self.label_10)

        self.frame_9 = QFrame(self.frame_5)
        self.frame_9.setObjectName(u"frame_9")
        self.frame_9.setStyleSheet(u"")
        self.frame_9.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_9.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_21 = QHBoxLayout(self.frame_9)
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.label_4 = QLabel(self.frame_9)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_21.addWidget(self.label_4)

        self.my_app_lineEdit = QLineEdit(self.frame_9)
        self.my_app_lineEdit.setObjectName(u"my_app_lineEdit")
        self.my_app_lineEdit.setMinimumSize(QSize(0, 30))
        self.my_app_lineEdit.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.horizontalLayout_21.addWidget(self.my_app_lineEdit)


        self.verticalLayout_38.addWidget(self.frame_9)

        self.frame_10 = QFrame(self.frame_5)
        self.frame_10.setObjectName(u"frame_10")
        self.frame_10.setStyleSheet(u"")
        self.frame_10.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_10.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_22 = QHBoxLayout(self.frame_10)
        self.horizontalLayout_22.setObjectName(u"horizontalLayout_22")
        self.label_5 = QLabel(self.frame_10)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_22.addWidget(self.label_5)

        self.my_sec_lineEdit = QLineEdit(self.frame_10)
        self.my_sec_lineEdit.setObjectName(u"my_sec_lineEdit")
        self.my_sec_lineEdit.setMinimumSize(QSize(0, 30))
        self.my_sec_lineEdit.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.horizontalLayout_22.addWidget(self.my_sec_lineEdit)


        self.verticalLayout_38.addWidget(self.frame_10)

        self.label_11 = QLabel(self.frame_5)
        self.label_11.setObjectName(u"label_11")
        sizePolicy4.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy4)

        self.verticalLayout_38.addWidget(self.label_11)

        self.frame_11 = QFrame(self.frame_5)
        self.frame_11.setObjectName(u"frame_11")
        self.frame_11.setStyleSheet(u"")
        self.frame_11.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_11.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_23 = QHBoxLayout(self.frame_11)
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.label_6 = QLabel(self.frame_11)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_23.addWidget(self.label_6)

        self.my_acct_stock_lineEdit = QLineEdit(self.frame_11)
        self.my_acct_stock_lineEdit.setObjectName(u"my_acct_stock_lineEdit")
        self.my_acct_stock_lineEdit.setMinimumSize(QSize(0, 30))
        self.my_acct_stock_lineEdit.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.horizontalLayout_23.addWidget(self.my_acct_stock_lineEdit)


        self.verticalLayout_38.addWidget(self.frame_11)

        self.label_12 = QLabel(self.frame_5)
        self.label_12.setObjectName(u"label_12")
        sizePolicy4.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy4)

        self.verticalLayout_38.addWidget(self.label_12)

        self.frame_12 = QFrame(self.frame_5)
        self.frame_12.setObjectName(u"frame_12")
        self.frame_12.setStyleSheet(u"")
        self.frame_12.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_12.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_24 = QHBoxLayout(self.frame_12)
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.label_7 = QLabel(self.frame_12)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_24.addWidget(self.label_7)

        self.my_paper_stock_lineEdit = QLineEdit(self.frame_12)
        self.my_paper_stock_lineEdit.setObjectName(u"my_paper_stock_lineEdit")
        self.my_paper_stock_lineEdit.setMinimumSize(QSize(0, 30))
        self.my_paper_stock_lineEdit.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.horizontalLayout_24.addWidget(self.my_paper_stock_lineEdit)


        self.verticalLayout_38.addWidget(self.frame_12)

        self.label_13 = QLabel(self.frame_5)
        self.label_13.setObjectName(u"label_13")
        sizePolicy4.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy4)

        self.verticalLayout_38.addWidget(self.label_13)

        self.frame_13 = QFrame(self.frame_5)
        self.frame_13.setObjectName(u"frame_13")
        self.frame_13.setStyleSheet(u"")
        self.frame_13.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_13.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_25 = QHBoxLayout(self.frame_13)
        self.horizontalLayout_25.setObjectName(u"horizontalLayout_25")
        self.label_8 = QLabel(self.frame_13)
        self.label_8.setObjectName(u"label_8")

        self.horizontalLayout_25.addWidget(self.label_8)

        self.my_prod_lineEdit = QLineEdit(self.frame_13)
        self.my_prod_lineEdit.setObjectName(u"my_prod_lineEdit")
        self.my_prod_lineEdit.setMinimumSize(QSize(0, 30))
        self.my_prod_lineEdit.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.horizontalLayout_25.addWidget(self.my_prod_lineEdit)


        self.verticalLayout_38.addWidget(self.frame_13)

        self.SavePerarametersButton_2 = QPushButton(self.frame_5)
        self.SavePerarametersButton_2.setObjectName(u"SavePerarametersButton_2")
        self.SavePerarametersButton_2.setMinimumSize(QSize(150, 30))
        self.SavePerarametersButton_2.setFont(font)
        self.SavePerarametersButton_2.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.SavePerarametersButton_2.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        self.SavePerarametersButton_2.setIcon(icon3)

        self.verticalLayout_38.addWidget(self.SavePerarametersButton_2)


        self.verticalLayout_37.addWidget(self.frame_5)

        self.stackedWidget.addWidget(self.kis_devlp_page)
        self.graph_page = QWidget()
        self.graph_page.setObjectName(u"graph_page")
        self.verticalLayout_7 = QVBoxLayout(self.graph_page)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.graph_frame = QFrame(self.graph_page)
        self.graph_frame.setObjectName(u"graph_frame")
        self.graph_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.graph_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.graph_frame)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.graph_bg = QFrame(self.graph_frame)
        self.graph_bg.setObjectName(u"graph_bg")
        self.graph_bg.setFrameShape(QFrame.Shape.StyledPanel)
        self.graph_bg.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_32 = QVBoxLayout(self.graph_bg)
        self.verticalLayout_32.setSpacing(0)
        self.verticalLayout_32.setObjectName(u"verticalLayout_32")
        self.verticalLayout_32.setContentsMargins(0, 0, 0, 0)
        self.frame_3 = QFrame(self.graph_bg)
        self.frame_3.setObjectName(u"frame_3")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy7)
        self.frame_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_7 = QHBoxLayout(self.frame_3)
        self.horizontalLayout_7.setSpacing(5)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.imageSizeUpPushButton = QPushButton(self.frame_3)
        self.imageSizeUpPushButton.setObjectName(u"imageSizeUpPushButton")
        self.imageSizeUpPushButton.setEnabled(True)
        self.imageSizeUpPushButton.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        icon4 = QIcon()
        icon4.addFile(u":/icons/images/icons/cil-plus.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.imageSizeUpPushButton.setIcon(icon4)

        self.horizontalLayout_7.addWidget(self.imageSizeUpPushButton)

        self.imageSizeDownPushButton = QPushButton(self.frame_3)
        self.imageSizeDownPushButton.setObjectName(u"imageSizeDownPushButton")
        self.imageSizeDownPushButton.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        icon5 = QIcon()
        icon5.addFile(u":/icons/images/icons/cil-minus.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.imageSizeDownPushButton.setIcon(icon5)

        self.horizontalLayout_7.addWidget(self.imageSizeDownPushButton)


        self.verticalLayout_32.addWidget(self.frame_3)

        self.scrollArea_3 = QScrollArea(self.graph_bg)
        self.scrollArea_3.setObjectName(u"scrollArea_3")
        self.scrollArea_3.setWidgetResizable(True)
        self.scrollAreaWidgetContents_3 = QWidget()
        self.scrollAreaWidgetContents_3.setObjectName(u"scrollAreaWidgetContents_3")
        self.scrollAreaWidgetContents_3.setGeometry(QRect(0, 0, 862, 675))
        self.horizontalLayout_15 = QHBoxLayout(self.scrollAreaWidgetContents_3)
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.frame_16 = QFrame(self.scrollAreaWidgetContents_3)
        self.frame_16.setObjectName(u"frame_16")
        self.frame_16.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_16.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_41 = QVBoxLayout(self.frame_16)
        self.verticalLayout_41.setObjectName(u"verticalLayout_41")
        self.verticalLayout_42 = QVBoxLayout()
        self.verticalLayout_42.setObjectName(u"verticalLayout_42")
        self.graph_image = QLabel(self.frame_16)
        self.graph_image.setObjectName(u"graph_image")
        sizePolicy8 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy8.setHorizontalStretch(0)
        sizePolicy8.setVerticalStretch(0)
        sizePolicy8.setHeightForWidth(self.graph_image.sizePolicy().hasHeightForWidth())
        self.graph_image.setSizePolicy(sizePolicy8)
        self.graph_image.setStyleSheet(u"")
        self.graph_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_42.addWidget(self.graph_image)


        self.verticalLayout_41.addLayout(self.verticalLayout_42)


        self.horizontalLayout_15.addWidget(self.frame_16)

        self.scrollArea_3.setWidget(self.scrollAreaWidgetContents_3)

        self.verticalLayout_32.addWidget(self.scrollArea_3)


        self.horizontalLayout_2.addWidget(self.graph_bg)

        self.data_list_bg = QFrame(self.graph_frame)
        self.data_list_bg.setObjectName(u"data_list_bg")
        self.data_list_bg.setMinimumSize(QSize(250, 0))
        self.data_list_bg.setMaximumSize(QSize(250, 16777215))
        self.data_list_bg.setStyleSheet(u"\n"
"#data_list_bg{\n"
"	background-color: rgb(33, 37, 43);\n"
" 	border-radius: 10px;\n"
"}")
        self.data_list_bg.setFrameShape(QFrame.Shape.StyledPanel)
        self.data_list_bg.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.data_list_bg)
        self.verticalLayout_4.setSpacing(10)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(10, 10, 10, 10)
        self.scrollArea_2 = QScrollArea(self.data_list_bg)
        self.scrollArea_2.setObjectName(u"scrollArea_2")
        self.scrollArea_2.setToolTipDuration(0)
        self.scrollArea_2.setStyleSheet(u" QScrollBar:vertical {\n"
"    background: rgb(52, 59, 72);\n"
" }\n"
" QScrollBar:horizontal {\n"
"    background: rgb(52, 59, 72);\n"
" }\n"
"\n"
"QScrollArea{\n"
"	background-color: rgb(52, 59, 72);\n"
"	 border-radius: 10px;\n"
"}")
        self.scrollArea_2.setFrameShape(QFrame.Shape.NoFrame)
        self.scrollArea_2.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scrollArea_2.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollAreaWidgetContents_2 = QWidget()
        self.scrollAreaWidgetContents_2.setObjectName(u"scrollAreaWidgetContents_2")
        self.scrollAreaWidgetContents_2.setGeometry(QRect(0, 0, 222, 46))
        sizePolicy5.setHeightForWidth(self.scrollAreaWidgetContents_2.sizePolicy().hasHeightForWidth())
        self.scrollAreaWidgetContents_2.setSizePolicy(sizePolicy5)
        self.scrollAreaWidgetContents_2.setStyleSheet(u" QScrollBar:vertical {\n"
"	border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    width: 14px;\n"
"    margin: 21px 0 21px 0;\n"
"	border-radius: 0px;\n"
" }")
        self.horizontalLayout_13 = QHBoxLayout(self.scrollAreaWidgetContents_2)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.graphDataFrame = QFrame(self.scrollAreaWidgetContents_2)
        self.graphDataFrame.setObjectName(u"graphDataFrame")
        sizePolicy3.setHeightForWidth(self.graphDataFrame.sizePolicy().hasHeightForWidth())
        self.graphDataFrame.setSizePolicy(sizePolicy3)
        self.graphDataFrame.setMinimumSize(QSize(0, 0))
        self.graphDataFrame.setFrameShape(QFrame.Shape.StyledPanel)
        self.graphDataFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_5 = QVBoxLayout(self.graphDataFrame)
        self.verticalLayout_5.setSpacing(10)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(10, 10, 10, 10)

        self.horizontalLayout_13.addWidget(self.graphDataFrame)

        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)

        self.verticalLayout_4.addWidget(self.scrollArea_2)

        self.data_buttons_frame = QFrame(self.data_list_bg)
        self.data_buttons_frame.setObjectName(u"data_buttons_frame")
        self.data_buttons_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.data_buttons_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_10 = QVBoxLayout(self.data_buttons_frame)
        self.verticalLayout_10.setSpacing(10)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.addGraphPushButton = QPushButton(self.data_buttons_frame)
        self.addGraphPushButton.setObjectName(u"addGraphPushButton")
        self.addGraphPushButton.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        self.addGraphPushButton.setIcon(icon4)

        self.verticalLayout_10.addWidget(self.addGraphPushButton)

        self.removePushButton = QPushButton(self.data_buttons_frame)
        self.removePushButton.setObjectName(u"removePushButton")
        self.removePushButton.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        self.removePushButton.setIcon(icon5)

        self.verticalLayout_10.addWidget(self.removePushButton)


        self.verticalLayout_4.addWidget(self.data_buttons_frame)


        self.horizontalLayout_2.addWidget(self.data_list_bg)


        self.verticalLayout_7.addWidget(self.graph_frame)

        self.stackedWidget.addWidget(self.graph_page)
        self.widgets = QWidget()
        self.widgets.setObjectName(u"widgets")
        self.widgets.setStyleSheet(u"b")
        self.verticalLayout = QVBoxLayout(self.widgets)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.row_1 = QFrame(self.widgets)
        self.row_1.setObjectName(u"row_1")
        self.row_1.setFrameShape(QFrame.Shape.StyledPanel)
        self.row_1.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_16 = QVBoxLayout(self.row_1)
        self.verticalLayout_16.setSpacing(0)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.verticalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.frame_div_content_1 = QFrame(self.row_1)
        self.frame_div_content_1.setObjectName(u"frame_div_content_1")
        self.frame_div_content_1.setMinimumSize(QSize(0, 110))
        self.frame_div_content_1.setMaximumSize(QSize(16777215, 110))
        self.frame_div_content_1.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_div_content_1.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_17 = QVBoxLayout(self.frame_div_content_1)
        self.verticalLayout_17.setSpacing(0)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.verticalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.frame_title_wid_1 = QFrame(self.frame_div_content_1)
        self.frame_title_wid_1.setObjectName(u"frame_title_wid_1")
        self.frame_title_wid_1.setMaximumSize(QSize(16777215, 35))
        self.frame_title_wid_1.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_title_wid_1.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_18 = QVBoxLayout(self.frame_title_wid_1)
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        self.labelBoxBlenderInstalation = QLabel(self.frame_title_wid_1)
        self.labelBoxBlenderInstalation.setObjectName(u"labelBoxBlenderInstalation")
        self.labelBoxBlenderInstalation.setFont(font)
        self.labelBoxBlenderInstalation.setStyleSheet(u"")

        self.verticalLayout_18.addWidget(self.labelBoxBlenderInstalation)


        self.verticalLayout_17.addWidget(self.frame_title_wid_1)

        self.frame_content_wid_1 = QFrame(self.frame_div_content_1)
        self.frame_content_wid_1.setObjectName(u"frame_content_wid_1")
        self.frame_content_wid_1.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_content_wid_1.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_9 = QHBoxLayout(self.frame_content_wid_1)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(-1, -1, -1, 0)
        self.lineEdit = QLineEdit(self.frame_content_wid_1)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setMinimumSize(QSize(0, 30))
        self.lineEdit.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.gridLayout.addWidget(self.lineEdit, 0, 0, 1, 1)

        self.pushButton = QPushButton(self.frame_content_wid_1)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setMinimumSize(QSize(150, 30))
        self.pushButton.setFont(font)
        self.pushButton.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.pushButton.setStyleSheet(u"background-color: rgb(52, 59, 72);")
        self.pushButton.setIcon(icon)

        self.gridLayout.addWidget(self.pushButton, 0, 1, 1, 1)

        self.labelVersion_3 = QLabel(self.frame_content_wid_1)
        self.labelVersion_3.setObjectName(u"labelVersion_3")
        self.labelVersion_3.setStyleSheet(u"color: rgb(113, 126, 149);")
        self.labelVersion_3.setLineWidth(1)
        self.labelVersion_3.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout.addWidget(self.labelVersion_3, 1, 0, 1, 2)


        self.horizontalLayout_9.addLayout(self.gridLayout)


        self.verticalLayout_17.addWidget(self.frame_content_wid_1)


        self.verticalLayout_16.addWidget(self.frame_div_content_1)


        self.verticalLayout.addWidget(self.row_1)

        self.row_2 = QFrame(self.widgets)
        self.row_2.setObjectName(u"row_2")
        self.row_2.setMinimumSize(QSize(0, 150))
        self.row_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.row_2.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_19 = QVBoxLayout(self.row_2)
        self.verticalLayout_19.setObjectName(u"verticalLayout_19")
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.checkBox = QCheckBox(self.row_2)
        self.checkBox.setObjectName(u"checkBox")
        self.checkBox.setAutoFillBackground(False)
        self.checkBox.setStyleSheet(u"")

        self.gridLayout_2.addWidget(self.checkBox, 0, 0, 1, 1)

        self.radioButton = QRadioButton(self.row_2)
        self.radioButton.setObjectName(u"radioButton")
        self.radioButton.setStyleSheet(u"")

        self.gridLayout_2.addWidget(self.radioButton, 0, 1, 1, 1)

        self.verticalSlider = QSlider(self.row_2)
        self.verticalSlider.setObjectName(u"verticalSlider")
        self.verticalSlider.setStyleSheet(u"")
        self.verticalSlider.setOrientation(Qt.Orientation.Vertical)

        self.gridLayout_2.addWidget(self.verticalSlider, 0, 2, 3, 1)

        self.verticalScrollBar = QScrollBar(self.row_2)
        self.verticalScrollBar.setObjectName(u"verticalScrollBar")
        self.verticalScrollBar.setStyleSheet(u" QScrollBar:vertical { background: rgb(52, 59, 72); }\n"
" QScrollBar:horizontal { background: rgb(52, 59, 72); }")
        self.verticalScrollBar.setOrientation(Qt.Orientation.Vertical)

        self.gridLayout_2.addWidget(self.verticalScrollBar, 0, 4, 3, 1)

        self.scrollArea = QScrollArea(self.row_2)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setStyleSheet(u" QScrollBar:vertical {\n"
"    background: rgb(52, 59, 72);\n"
" }\n"
" QScrollBar:horizontal {\n"
"    background: rgb(52, 59, 72);\n"
" }")
        self.scrollArea.setFrameShape(QFrame.Shape.NoFrame)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 320, 224))
        self.scrollAreaWidgetContents.setStyleSheet(u" QScrollBar:vertical {\n"
"	border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    width: 14px;\n"
"    margin: 21px 0 21px 0;\n"
"	border-radius: 0px;\n"
" }")
        self.horizontalLayout_11 = QHBoxLayout(self.scrollAreaWidgetContents)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.plainTextEdit = QPlainTextEdit(self.scrollAreaWidgetContents)
        self.plainTextEdit.setObjectName(u"plainTextEdit")
        self.plainTextEdit.setMinimumSize(QSize(200, 200))
        self.plainTextEdit.setStyleSheet(u"background-color: rgb(33, 37, 43);")

        self.horizontalLayout_11.addWidget(self.plainTextEdit)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.gridLayout_2.addWidget(self.scrollArea, 0, 5, 3, 1)

        self.comboBox = QComboBox(self.row_2)
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setFont(font)
        self.comboBox.setAutoFillBackground(False)
        self.comboBox.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.comboBox.setIconSize(QSize(16, 16))
        self.comboBox.setFrame(True)

        self.gridLayout_2.addWidget(self.comboBox, 1, 0, 1, 2)

        self.horizontalScrollBar = QScrollBar(self.row_2)
        self.horizontalScrollBar.setObjectName(u"horizontalScrollBar")
        sizePolicy.setHeightForWidth(self.horizontalScrollBar.sizePolicy().hasHeightForWidth())
        self.horizontalScrollBar.setSizePolicy(sizePolicy)
        self.horizontalScrollBar.setStyleSheet(u" QScrollBar:vertical { background: rgb(52, 59, 72); }\n"
" QScrollBar:horizontal { background: rgb(52, 59, 72); }")
        self.horizontalScrollBar.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_2.addWidget(self.horizontalScrollBar, 1, 3, 1, 1)

        self.commandLinkButton = QCommandLinkButton(self.row_2)
        self.commandLinkButton.setObjectName(u"commandLinkButton")
        self.commandLinkButton.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.commandLinkButton.setStyleSheet(u"")
        icon6 = QIcon()
        icon6.addFile(u":/icons/images/icons/cil-link.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.commandLinkButton.setIcon(icon6)

        self.gridLayout_2.addWidget(self.commandLinkButton, 1, 6, 1, 1)

        self.horizontalSlider = QSlider(self.row_2)
        self.horizontalSlider.setObjectName(u"horizontalSlider")
        self.horizontalSlider.setStyleSheet(u"")
        self.horizontalSlider.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_2.addWidget(self.horizontalSlider, 2, 0, 1, 2)


        self.verticalLayout_19.addLayout(self.gridLayout_2)


        self.verticalLayout.addWidget(self.row_2)

        self.row_3 = QFrame(self.widgets)
        self.row_3.setObjectName(u"row_3")
        self.row_3.setMinimumSize(QSize(0, 150))
        self.row_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.row_3.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_12 = QHBoxLayout(self.row_3)
        self.horizontalLayout_12.setSpacing(0)
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.tableWidget = QTableWidget(self.row_3)
        if (self.tableWidget.columnCount() < 4):
            self.tableWidget.setColumnCount(4)
        __qtablewidgetitem48 = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, __qtablewidgetitem48)
        __qtablewidgetitem49 = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, __qtablewidgetitem49)
        __qtablewidgetitem50 = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, __qtablewidgetitem50)
        __qtablewidgetitem51 = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, __qtablewidgetitem51)
        if (self.tableWidget.rowCount() < 16):
            self.tableWidget.setRowCount(16)
        __qtablewidgetitem52 = QTableWidgetItem()
        __qtablewidgetitem52.setFont(font4);
        self.tableWidget.setVerticalHeaderItem(0, __qtablewidgetitem52)
        __qtablewidgetitem53 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, __qtablewidgetitem53)
        __qtablewidgetitem54 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(2, __qtablewidgetitem54)
        __qtablewidgetitem55 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(3, __qtablewidgetitem55)
        __qtablewidgetitem56 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(4, __qtablewidgetitem56)
        __qtablewidgetitem57 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(5, __qtablewidgetitem57)
        __qtablewidgetitem58 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(6, __qtablewidgetitem58)
        __qtablewidgetitem59 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(7, __qtablewidgetitem59)
        __qtablewidgetitem60 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(8, __qtablewidgetitem60)
        __qtablewidgetitem61 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(9, __qtablewidgetitem61)
        __qtablewidgetitem62 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(10, __qtablewidgetitem62)
        __qtablewidgetitem63 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(11, __qtablewidgetitem63)
        __qtablewidgetitem64 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(12, __qtablewidgetitem64)
        __qtablewidgetitem65 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(13, __qtablewidgetitem65)
        __qtablewidgetitem66 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(14, __qtablewidgetitem66)
        __qtablewidgetitem67 = QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(15, __qtablewidgetitem67)
        __qtablewidgetitem68 = QTableWidgetItem()
        self.tableWidget.setItem(0, 0, __qtablewidgetitem68)
        __qtablewidgetitem69 = QTableWidgetItem()
        self.tableWidget.setItem(0, 1, __qtablewidgetitem69)
        __qtablewidgetitem70 = QTableWidgetItem()
        self.tableWidget.setItem(0, 2, __qtablewidgetitem70)
        __qtablewidgetitem71 = QTableWidgetItem()
        self.tableWidget.setItem(0, 3, __qtablewidgetitem71)
        self.tableWidget.setObjectName(u"tableWidget")
        sizePolicy9 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy9.setHorizontalStretch(0)
        sizePolicy9.setVerticalStretch(0)
        sizePolicy9.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy9)
        palette2 = QPalette()
        palette2.setBrush(QPalette.Active, QPalette.WindowText, brush)
        palette2.setBrush(QPalette.Active, QPalette.Button, brush1)
        palette2.setBrush(QPalette.Active, QPalette.Text, brush)
        palette2.setBrush(QPalette.Active, QPalette.ButtonText, brush)
        palette2.setBrush(QPalette.Active, QPalette.Base, brush1)
        palette2.setBrush(QPalette.Active, QPalette.Window, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette2.setBrush(QPalette.Active, QPalette.PlaceholderText, brush)
#endif
        palette2.setBrush(QPalette.Inactive, QPalette.WindowText, brush)
        palette2.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette2.setBrush(QPalette.Inactive, QPalette.Text, brush)
        palette2.setBrush(QPalette.Inactive, QPalette.ButtonText, brush)
        palette2.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        palette2.setBrush(QPalette.Inactive, QPalette.Window, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette2.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush)
#endif
        palette2.setBrush(QPalette.Disabled, QPalette.WindowText, brush)
        palette2.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette2.setBrush(QPalette.Disabled, QPalette.Text, brush)
        palette2.setBrush(QPalette.Disabled, QPalette.ButtonText, brush)
        palette2.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette2.setBrush(QPalette.Disabled, QPalette.Window, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette2.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush)
#endif
        self.tableWidget.setPalette(palette2)
        self.tableWidget.setStyleSheet(u"background-color: rgb(33, 37, 43);")
        self.tableWidget.setFrameShape(QFrame.Shape.NoFrame)
        self.tableWidget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.tableWidget.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.tableWidget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tableWidget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tableWidget.setShowGrid(True)
        self.tableWidget.setGridStyle(Qt.PenStyle.SolidLine)
        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(True)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(200)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setCascadingSectionResizes(False)
        self.tableWidget.verticalHeader().setHighlightSections(False)
        self.tableWidget.verticalHeader().setStretchLastSection(True)

        self.horizontalLayout_12.addWidget(self.tableWidget)


        self.verticalLayout.addWidget(self.row_3)

        self.stackedWidget.addWidget(self.widgets)

        self.verticalLayout_15.addWidget(self.stackedWidget)


        self.horizontalLayout_4.addWidget(self.pagesContainer)


        self.verticalLayout_6.addWidget(self.content)

        self.bottomBar = QFrame(self.contentBottom)
        self.bottomBar.setObjectName(u"bottomBar")
        self.bottomBar.setMinimumSize(QSize(0, 22))
        self.bottomBar.setMaximumSize(QSize(16777215, 22))
        self.bottomBar.setFrameShape(QFrame.Shape.NoFrame)
        self.bottomBar.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.bottomBar)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.creditsLabel = QLabel(self.bottomBar)
        self.creditsLabel.setObjectName(u"creditsLabel")
        self.creditsLabel.setMaximumSize(QSize(16777215, 16))
        self.creditsLabel.setFont(font3)
        self.creditsLabel.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_5.addWidget(self.creditsLabel)

        self.version = QLabel(self.bottomBar)
        self.version.setObjectName(u"version")
        self.version.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_5.addWidget(self.version)

        self.frame_size_grip = QFrame(self.bottomBar)
        self.frame_size_grip.setObjectName(u"frame_size_grip")
        self.frame_size_grip.setMinimumSize(QSize(20, 0))
        self.frame_size_grip.setMaximumSize(QSize(20, 16777215))
        self.frame_size_grip.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_size_grip.setFrameShadow(QFrame.Shadow.Raised)

        self.horizontalLayout_5.addWidget(self.frame_size_grip)


        self.verticalLayout_6.addWidget(self.bottomBar)


        self.verticalLayout_2.addWidget(self.contentBottom)


        self.appLayout.addWidget(self.contentBox)


        self.verticalLayout_21.addWidget(self.bgApp)

        MainWindow.setCentralWidget(self.styleSheet)

        self.retranslateUi(MainWindow)

        self.stackedWidget.setCurrentIndex(4)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.titleLeftApp.setText(QCoreApplication.translate("MainWindow", u"PyDracula", None))
        self.titleLeftDescription.setText(QCoreApplication.translate("MainWindow", u"Modern GUI / Flat Style", None))
        self.btn_home.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.btn_parameter.setText(QCoreApplication.translate("MainWindow", u"Widgets", None))
        self.btn_graph.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.btn_setting.setText(QCoreApplication.translate("MainWindow", u"Widgets", None))
        self.btn_download.setText(QCoreApplication.translate("MainWindow", u"Widgets", None))
        self.titleRightInfo.setText(QCoreApplication.translate("MainWindow", u"RichDog - AI Stock Trading Program", None))
        self.logo_label.setText("")
        self.label.setText(QCoreApplication.translate("MainWindow", u" Log", None))
        self.clearLogPushButton.setText(QCoreApplication.translate("MainWindow", u"Clear Log", None))
        self.cpu_label.setText(QCoreApplication.translate("MainWindow", u"CPU", None))
        self.cpu_name_label.setText("")
        self.gpu_label.setText(QCoreApplication.translate("MainWindow", u"GPU", None))
        self.gpu_name_label.setText("")
        self.cuda_label.setText(QCoreApplication.translate("MainWindow", u"Available CUDA : ", None))
        self.labelBoxBlenderInstalation_3.setText(QCoreApplication.translate("MainWindow", u"Model FIle", None))
        self.File_Button_4.setText(QCoreApplication.translate("MainWindow", u"Open", None))
        self.filepath_lineEdit_2.setText("")
        self.filepath_lineEdit_2.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Type here", None))
        self.testPushButton.setText(QCoreApplication.translate("MainWindow", u"Test Model Start", None))
        self.learingPushButton.setText(QCoreApplication.translate("MainWindow", u"Learing Start", None))
        self.File_Button_3.setText(QCoreApplication.translate("MainWindow", u"Open", None))
        self.filepath_lineEdit.setText("")
        self.filepath_lineEdit.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Type here", None))
        self.labelBoxBlenderInstalation_2.setText(QCoreApplication.translate("MainWindow", u"Hyperparameters FIle", None))
        ___qtablewidgetitem = self.tableWidget_2.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"0", None));
        ___qtablewidgetitem1 = self.tableWidget_2.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"1", None));
        ___qtablewidgetitem2 = self.tableWidget_2.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("MainWindow", u"2", None));
        ___qtablewidgetitem3 = self.tableWidget_2.verticalHeaderItem(0)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("MainWindow", u"1", None));
        ___qtablewidgetitem4 = self.tableWidget_2.verticalHeaderItem(1)
        ___qtablewidgetitem4.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem5 = self.tableWidget_2.verticalHeaderItem(2)
        ___qtablewidgetitem5.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem6 = self.tableWidget_2.verticalHeaderItem(3)
        ___qtablewidgetitem6.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem7 = self.tableWidget_2.verticalHeaderItem(4)
        ___qtablewidgetitem7.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem8 = self.tableWidget_2.verticalHeaderItem(5)
        ___qtablewidgetitem8.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem9 = self.tableWidget_2.verticalHeaderItem(6)
        ___qtablewidgetitem9.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem10 = self.tableWidget_2.verticalHeaderItem(7)
        ___qtablewidgetitem10.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem11 = self.tableWidget_2.verticalHeaderItem(8)
        ___qtablewidgetitem11.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem12 = self.tableWidget_2.verticalHeaderItem(9)
        ___qtablewidgetitem12.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem13 = self.tableWidget_2.verticalHeaderItem(10)
        ___qtablewidgetitem13.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem14 = self.tableWidget_2.verticalHeaderItem(11)
        ___qtablewidgetitem14.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem15 = self.tableWidget_2.verticalHeaderItem(12)
        ___qtablewidgetitem15.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem16 = self.tableWidget_2.verticalHeaderItem(13)
        ___qtablewidgetitem16.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem17 = self.tableWidget_2.verticalHeaderItem(14)
        ___qtablewidgetitem17.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem18 = self.tableWidget_2.verticalHeaderItem(15)
        ___qtablewidgetitem18.setText(QCoreApplication.translate("MainWindow", u"New Row", None));

        __sortingEnabled = self.tableWidget_2.isSortingEnabled()
        self.tableWidget_2.setSortingEnabled(False)
        ___qtablewidgetitem19 = self.tableWidget_2.item(0, 0)
        ___qtablewidgetitem19.setText(QCoreApplication.translate("MainWindow", u"Hyperparameter", None));
        ___qtablewidgetitem20 = self.tableWidget_2.item(0, 1)
        ___qtablewidgetitem20.setText(QCoreApplication.translate("MainWindow", u"Value", None));
        ___qtablewidgetitem21 = self.tableWidget_2.item(0, 2)
        ___qtablewidgetitem21.setText(QCoreApplication.translate("MainWindow", u"Min", None));
        ___qtablewidgetitem22 = self.tableWidget_2.item(0, 3)
        ___qtablewidgetitem22.setText(QCoreApplication.translate("MainWindow", u"Max", None));
        ___qtablewidgetitem23 = self.tableWidget_2.item(0, 4)
        ___qtablewidgetitem23.setText(QCoreApplication.translate("MainWindow", u"Note", None));
        self.tableWidget_2.setSortingEnabled(__sortingEnabled)

        self.filepath_lineEdit_6.setText("")
        self.filepath_lineEdit_6.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Type here", None))
        self.File_Button_8.setText(QCoreApplication.translate("MainWindow", u"Open", None))
        self.labelBoxBlenderInstalation_7.setText(QCoreApplication.translate("MainWindow", u"Stock Config FIle", None))
        ___qtablewidgetitem24 = self.tableWidget_3.horizontalHeaderItem(0)
        ___qtablewidgetitem24.setText(QCoreApplication.translate("MainWindow", u"0", None));
        ___qtablewidgetitem25 = self.tableWidget_3.horizontalHeaderItem(1)
        ___qtablewidgetitem25.setText(QCoreApplication.translate("MainWindow", u"1", None));
        ___qtablewidgetitem26 = self.tableWidget_3.horizontalHeaderItem(2)
        ___qtablewidgetitem26.setText(QCoreApplication.translate("MainWindow", u"2", None));
        ___qtablewidgetitem27 = self.tableWidget_3.verticalHeaderItem(0)
        ___qtablewidgetitem27.setText(QCoreApplication.translate("MainWindow", u"1", None));
        ___qtablewidgetitem28 = self.tableWidget_3.verticalHeaderItem(1)
        ___qtablewidgetitem28.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem29 = self.tableWidget_3.verticalHeaderItem(2)
        ___qtablewidgetitem29.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem30 = self.tableWidget_3.verticalHeaderItem(3)
        ___qtablewidgetitem30.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem31 = self.tableWidget_3.verticalHeaderItem(4)
        ___qtablewidgetitem31.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem32 = self.tableWidget_3.verticalHeaderItem(5)
        ___qtablewidgetitem32.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem33 = self.tableWidget_3.verticalHeaderItem(6)
        ___qtablewidgetitem33.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem34 = self.tableWidget_3.verticalHeaderItem(7)
        ___qtablewidgetitem34.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem35 = self.tableWidget_3.verticalHeaderItem(8)
        ___qtablewidgetitem35.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem36 = self.tableWidget_3.verticalHeaderItem(9)
        ___qtablewidgetitem36.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem37 = self.tableWidget_3.verticalHeaderItem(10)
        ___qtablewidgetitem37.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem38 = self.tableWidget_3.verticalHeaderItem(11)
        ___qtablewidgetitem38.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem39 = self.tableWidget_3.verticalHeaderItem(12)
        ___qtablewidgetitem39.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem40 = self.tableWidget_3.verticalHeaderItem(13)
        ___qtablewidgetitem40.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem41 = self.tableWidget_3.verticalHeaderItem(14)
        ___qtablewidgetitem41.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem42 = self.tableWidget_3.verticalHeaderItem(15)
        ___qtablewidgetitem42.setText(QCoreApplication.translate("MainWindow", u"New Row", None));

        __sortingEnabled1 = self.tableWidget_3.isSortingEnabled()
        self.tableWidget_3.setSortingEnabled(False)
        ___qtablewidgetitem43 = self.tableWidget_3.item(0, 0)
        ___qtablewidgetitem43.setText(QCoreApplication.translate("MainWindow", u"Hyperparameter", None));
        ___qtablewidgetitem44 = self.tableWidget_3.item(0, 1)
        ___qtablewidgetitem44.setText(QCoreApplication.translate("MainWindow", u"Value", None));
        ___qtablewidgetitem45 = self.tableWidget_3.item(0, 2)
        ___qtablewidgetitem45.setText(QCoreApplication.translate("MainWindow", u"Min", None));
        ___qtablewidgetitem46 = self.tableWidget_3.item(0, 3)
        ___qtablewidgetitem46.setText(QCoreApplication.translate("MainWindow", u"Max", None));
        ___qtablewidgetitem47 = self.tableWidget_3.item(0, 4)
        ___qtablewidgetitem47.setText(QCoreApplication.translate("MainWindow", u"Note", None));
        self.tableWidget_3.setSortingEnabled(__sortingEnabled1)

        self.label_17.setText(QCoreApplication.translate("MainWindow", u"Stock Columns", None))
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"Visualization Columns", None))
        self.SavePerarametersButton.setText(QCoreApplication.translate("MainWindow", u"Save Hyperparameters", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"Stock Code List", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Stock Code", None))
        self.addStockCodePushButton.setText(QCoreApplication.translate("MainWindow", u"Add", None))
        self.removeStockCodePushButton.setText(QCoreApplication.translate("MainWindow", u"Remove", None))
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"Last Date", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"count", None))
        self.isDataFileRemoveCheckBox.setText(QCoreApplication.translate("MainWindow", u"Is data File Remove", None))
        self.DownloadPushButton.setText(QCoreApplication.translate("MainWindow", u"Download", None))
        self.labelBoxBlenderInstalation_4.setText(QCoreApplication.translate("MainWindow", u"kis_devlp file", None))
        self.filepath_lineEdit_3.setText("")
        self.filepath_lineEdit_3.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Type here", None))
        self.File_Button_5.setText(QCoreApplication.translate("MainWindow", u"Open", None))
        self.labelVersion_5.setText("")
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"  \ubaa8\uc758 \ud22c\uc790 ( AppKey, Appsecret )", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"paper_app", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"paper_sec", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"  \uc2e4\uc804 \ud22c\uc790 ( AppKey, Appsecret )", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"my_app", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"my_sec", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"  \uc2e4\uc804 \ud22c\uc790 \uacc4\uc88c\ubc88\ud638 \uc55e 8 \uc790\ub9ac", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"my_acct_stock", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"  \ubaa8\uc758 \ud22c\uc790 \uacc4\uc88c\ubc88\ud638 \uc55e 8 \uc790\ub9ac", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"my_paper_stock", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"  \uacc4\uc88c\ubc88\ud638 \ub4a4 2\uc790\ub9ac ( \uc77c\ubc18 \uacc4\uc88c : 01 )", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"my_prod", None))
        self.my_prod_lineEdit.setText(QCoreApplication.translate("MainWindow", u"01", None))
        self.SavePerarametersButton_2.setText(QCoreApplication.translate("MainWindow", u"Save Devlp File", None))
        self.imageSizeUpPushButton.setText("")
        self.imageSizeDownPushButton.setText("")
        self.graph_image.setText(QCoreApplication.translate("MainWindow", u"No Images", None))
        self.addGraphPushButton.setText(QCoreApplication.translate("MainWindow", u"Add Graph", None))
        self.removePushButton.setText(QCoreApplication.translate("MainWindow", u"Remove Graph", None))
        self.labelBoxBlenderInstalation.setText(QCoreApplication.translate("MainWindow", u"FILE BOX", None))
        self.lineEdit.setText("")
        self.lineEdit.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Type here", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Open", None))
        self.labelVersion_3.setText(QCoreApplication.translate("MainWindow", u"Label description", None))
        self.checkBox.setText(QCoreApplication.translate("MainWindow", u"CheckBox", None))
        self.radioButton.setText(QCoreApplication.translate("MainWindow", u"RadioButton", None))
        self.comboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"Test 1", None))
        self.comboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"Test 2", None))
        self.comboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"Test 3", None))

        self.commandLinkButton.setText(QCoreApplication.translate("MainWindow", u"Link Button", None))
        self.commandLinkButton.setDescription(QCoreApplication.translate("MainWindow", u"Link description", None))
        ___qtablewidgetitem48 = self.tableWidget.horizontalHeaderItem(0)
        ___qtablewidgetitem48.setText(QCoreApplication.translate("MainWindow", u"0", None));
        ___qtablewidgetitem49 = self.tableWidget.horizontalHeaderItem(1)
        ___qtablewidgetitem49.setText(QCoreApplication.translate("MainWindow", u"1", None));
        ___qtablewidgetitem50 = self.tableWidget.horizontalHeaderItem(2)
        ___qtablewidgetitem50.setText(QCoreApplication.translate("MainWindow", u"2", None));
        ___qtablewidgetitem51 = self.tableWidget.horizontalHeaderItem(3)
        ___qtablewidgetitem51.setText(QCoreApplication.translate("MainWindow", u"3", None));
        ___qtablewidgetitem52 = self.tableWidget.verticalHeaderItem(0)
        ___qtablewidgetitem52.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem53 = self.tableWidget.verticalHeaderItem(1)
        ___qtablewidgetitem53.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem54 = self.tableWidget.verticalHeaderItem(2)
        ___qtablewidgetitem54.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem55 = self.tableWidget.verticalHeaderItem(3)
        ___qtablewidgetitem55.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem56 = self.tableWidget.verticalHeaderItem(4)
        ___qtablewidgetitem56.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem57 = self.tableWidget.verticalHeaderItem(5)
        ___qtablewidgetitem57.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem58 = self.tableWidget.verticalHeaderItem(6)
        ___qtablewidgetitem58.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem59 = self.tableWidget.verticalHeaderItem(7)
        ___qtablewidgetitem59.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem60 = self.tableWidget.verticalHeaderItem(8)
        ___qtablewidgetitem60.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem61 = self.tableWidget.verticalHeaderItem(9)
        ___qtablewidgetitem61.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem62 = self.tableWidget.verticalHeaderItem(10)
        ___qtablewidgetitem62.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem63 = self.tableWidget.verticalHeaderItem(11)
        ___qtablewidgetitem63.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem64 = self.tableWidget.verticalHeaderItem(12)
        ___qtablewidgetitem64.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem65 = self.tableWidget.verticalHeaderItem(13)
        ___qtablewidgetitem65.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem66 = self.tableWidget.verticalHeaderItem(14)
        ___qtablewidgetitem66.setText(QCoreApplication.translate("MainWindow", u"New Row", None));
        ___qtablewidgetitem67 = self.tableWidget.verticalHeaderItem(15)
        ___qtablewidgetitem67.setText(QCoreApplication.translate("MainWindow", u"New Row", None));

        __sortingEnabled2 = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)
        ___qtablewidgetitem68 = self.tableWidget.item(0, 0)
        ___qtablewidgetitem68.setText(QCoreApplication.translate("MainWindow", u"Test", None));
        ___qtablewidgetitem69 = self.tableWidget.item(0, 1)
        ___qtablewidgetitem69.setText(QCoreApplication.translate("MainWindow", u"Text", None));
        ___qtablewidgetitem70 = self.tableWidget.item(0, 2)
        ___qtablewidgetitem70.setText(QCoreApplication.translate("MainWindow", u"Cell", None));
        ___qtablewidgetitem71 = self.tableWidget.item(0, 3)
        ___qtablewidgetitem71.setText(QCoreApplication.translate("MainWindow", u"Line", None));
        self.tableWidget.setSortingEnabled(__sortingEnabled2)

        self.creditsLabel.setText(QCoreApplication.translate("MainWindow", u"By: Glace", None))
        self.version.setText(QCoreApplication.translate("MainWindow", u"v2.0.0", None))
    # retranslateUi

