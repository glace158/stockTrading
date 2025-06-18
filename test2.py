from PySide6.QtCore import Qt
from PySide6.QtGui import QMovie, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout

class GifOnPngViewer(QWidget):
    def __init__(self):
        super().__init__()

        # 레이아웃 설정
        layout = QVBoxLayout()

        # QLabel 생성 (PNG 이미지를 표시할 레이블)
        label_png = QLabel(self)
        pixmap = QPixmap("your_image.png")  # PNG 파일 경로 지정
        label_png.setPixmap(pixmap)
        label_png.setAlignment(Qt.AlignCenter)

        # GIF 표시 레이블 생성
        label_gif = QLabel(self)
        movie = QMovie("your_gif_file.gif")  # GIF 파일 경로 지정
        label_gif.setMovie(movie)
        label_gif.setAlignment(Qt.AlignCenter)
        
        # GIF 애니메이션 시작
        movie.start()

        # 레이아웃에 PNG와 GIF 레이블 추가
        layout.addWidget(label_png)
        layout.addWidget(label_gif)

        # 기본 윈도우 설정
        self.setLayout(layout)
        self.setWindowTitle("PNG 위에 GIF 표시")
        self.resize(400, 400)

if __name__ == "__main__":
    app = QApplication([])

    # 윈도우 실행
    window = GifOnPngViewer()
    window.show()

    app.exec()
