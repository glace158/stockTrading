from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PySide6.QtGui import QPixmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("이미지 크기 조절")

        # 기본 이미지 크기 설정
        self.image_width = 300
        self.image_height = 200

        # QLabel 설정
        self.label = QLabel(self)
        self.pixmap = QPixmap("qt/images/images/DogeIcon.png")  # 이미지 파일 경로
        self.update_image()

        # 버튼 설정
        self.plus_button = QPushButton("+", self)
        self.plus_button.clicked.connect(self.increase_size)

        self.minus_button = QPushButton("-", self)
        self.minus_button.clicked.connect(self.decrease_size)

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.plus_button)
        layout.addWidget(self.minus_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def update_image(self):
        # 이미지를 현재 크기로 조정하여 QLabel에 표시
        resized_pixmap = self.pixmap.scaled(self.image_width, self.image_height)
        self.label.setPixmap(resized_pixmap)

    def increase_size(self):
        # 이미지 크기를 증가
        self.image_width += 20
        self.image_height += 20
        self.update_image()

    def decrease_size(self):
        # 이미지 크기를 감소
        if self.image_width > 20 and self.image_height > 20:  # 최소 크기 제한
            self.image_width -= 20
            self.image_height -= 20
            self.update_image()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()