

# 사용할 이미지 파일 경로 (실제 파일 경로로 변경하세요)
# 'background.png'와 'animation.gif' 파일이 스크립트와 같은 폴더에 있다고 가정합니다.
# 'animation.gif'는 배경이 투명해야 합니다.
import sys
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtGui import QPainter, QPixmap, QMovie
from PySide6.QtCore import Qt, QRect
# 이미지 경로는 전역 변수로 두어도 안전합니다 (문자열일 뿐이므로).
PNG_PATH = "./qt/images/images/RichDoge.png"
GIF_PATH = "./qt/images/images/MoneyRaining.gif"


class OverlayWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QPainter로 겹치기 예제")
        self.setGeometry(100, 100, 800, 600)

        self.background_pixmap = QPixmap(PNG_PATH)
        self.movie = QMovie(GIF_PATH)

        # GIF의 프레임이 바뀔 때마다 위젯을 다시 그리도록 신호 연결
        self.movie.frameChanged.connect(self.update)
        self.movie.start()

    def paintEvent(self, event):
        """위젯을 그려야 할 때마다 호출되는 함수"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. 배경 PNG 그리기 (위젯 크기에 꽉 차게)
        painter.drawPixmap(self.rect(), self.background_pixmap)

        # 2. 현재 GIF 프레임 가져와서 그리기
        current_frame = self.movie.currentPixmap()
        
        # GIF를 중앙에 위치시키기 위한 좌표 계산
        frame_rect = current_frame.rect()
        x = (self.width() - frame_rect.width()) / 2
        y = (self.height() - frame_rect.height()) / 2
        
        # 계산된 위치에 현재 GIF 프레임 그리기
        painter.drawPixmap(x, y, current_frame)

if __name__ == "__main__":
    # 실행 전, 이미지 파일 준비
    # ... (위 예제와 동일한 파일 체크 로직) ...
    app = QApplication(sys.argv)
    widget = OverlayWidget()
    widget.show()
    sys.exit(app.exec())