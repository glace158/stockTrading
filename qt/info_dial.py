from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

class HollowDial(QDial):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: transparent;")  # 배경을 투명으로 설정
        self.setRange(0, 100)  # 0부터 100까지 값 설정
        self.setValue(0)  # 기본 값 설정
        self.setEnabled(False)
        #self.setSizePolicy(Qt.Preferred, Qt.Preferred)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 다이얼 크기와 중심 계산
        rect = self.rect()
        center = rect.center()
        radius = min(rect.width(), rect.height()) / 2 * 0.8

        # 배경 원을 비워둔 원 그래프 모양
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(240, 240, 240, 0))  # 배경색
        painter.drawEllipse(center, radius, radius)  # 속이 빈 원

        # 회전할 수 있는 바깥쪽 원 그리기
        painter.setPen(QPen(QColor(33, 37, 43), 15))  # 원 둘레 색상 설정
        painter.setBrush(Qt.transparent)  # 채우지 않음
        painter.drawEllipse(center, radius, radius)

        # 다이얼의 회전 값에 따라 섹션 그리기
        angle = self.value() / (self.maximum() - self.minimum()) * 360
        arcRect = QRect(center.x() - radius, center.y() - radius, 2 * radius, 2 * radius)  # 아크를 그릴 사각형의 크기 줄이기
        painter.setPen(QPen(QColor(189, 147, 249), 15))  # 다이얼 진행 부분 색상 설정
        painter.setBrush(Qt.transparent)
        painter.drawArc(arcRect, 90 * 16, -angle * 16)  # 회전 값에 따라 아크 그리기

        # 현재 값 텍스트로 표시
        painter.setPen(QPen(QColor(255, 255, 255)))  # 텍스트 색상
        painter.setFont(QFont("Arial", 40))  # 폰트 설정
        text = str(self.value()) + '%' # 현재 다이얼 값
        painter.drawText(rect, Qt.AlignCenter, text)  # 텍스트를 다이얼의 중앙에 표시

        painter.end()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        dial = HollowDial(self)
        dial.setGeometry(100, 100, 200, 200)

        self.setCentralWidget(dial)
        self.setWindowTitle("Hollow Dial with Value")
        self.setGeometry(200, 200, 400, 400)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
