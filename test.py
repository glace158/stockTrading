import sys
import os
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout, # 버튼 추가를 위해
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QPushButton # 버튼 추가
)
from PySide6.QtGui import QPixmap, QPalette
from PySide6.QtCore import Qt, QSize, Slot # Slot 데코레이터 추가

class ImageDisplayWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PNG Image Viewer in frame_16")
        self.setGeometry(100, 100, 800, 650) # 버튼 공간을 위해 높이 약간 늘림

        # 중앙 위젯 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 메인 레이아웃 (중앙 위젯용)
        main_layout = QVBoxLayout(central_widget)

        # --- 버튼 추가 ---
        self.clear_button = QPushButton("이미지 지우기")
        self.clear_button.clicked.connect(self.clear_displayed_images) # 슬롯 연결
        
        self.reload_button = QPushButton("이미지 다시 불러오기") # 다시 불러오기 버튼 추가
        self.reload_button.clicked.connect(self.reload_images)

        # 버튼들을 담을 수평 레이아웃
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.reload_button)
        button_layout.addWidget(self.clear_button)
        main_layout.addLayout(button_layout) # 메인 레이아웃에 버튼 레이아웃 추가
        # -----------------

        # QScrollArea 생성
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.scroll_area)

        # frame_16 (QFrame) 생성
        self.frame_16 = QFrame()
        self.frame_16.setObjectName("frame_16")
        self.frame_16.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_16.setFrameShadow(QFrame.Shadow.Raised)
        self.scroll_area.setWidget(self.frame_16)

        # frame_16 내부의 이미지들을 담을 레이아웃
        self.frame_16_layout = QVBoxLayout(self.frame_16)
        self.frame_16.setLayout(self.frame_16_layout)

        self.load_images_to_frame()

    def _clear_q_layout(self, layout):
        """Helper method to clear all items from a QLayout."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                if item is None:
                    continue

                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    child_layout = item.layout()
                    if child_layout is not None:
                        self._clear_q_layout(child_layout) # Recursive call for nested layouts

    @Slot()
    def clear_displayed_images(self):
        """Clears all images from frame_16_layout."""
        print("이미지 지우기 버튼 클릭됨")
        self._clear_q_layout(self.frame_16_layout)
        # 선택 사항: 비워진 후 메시지 표시
        # info_label = QLabel("이미지가 모두 제거되었습니다.")
        # info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.frame_16_layout.addWidget(info_label)
        # self.frame_16_layout.addStretch(1) # 메시지가 위로 붙도록
        print("frame_16_layout 비워짐")

    @Slot()
    def reload_images(self):
        """Clears and reloads images into frame_16_layout."""
        print("이미지 다시 불러오기 버튼 클릭됨")
        self.load_images_to_frame() # 내부에서 clear 후 로드

    def load_images_to_frame(self):
        # 이미지를 로드하기 전에 기존 내용을 비웁니다.
        self._clear_q_layout(self.frame_16_layout)
        print("기존 이미지 레이아웃 비우고 새로 로드 시작")

        image_dir = Path("./Data_graph/")

        if not image_dir.is_dir():
            error_label = QLabel(f"오류: '{image_dir}' 디렉토리를 찾을 수 없습니다.")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.frame_16_layout.addWidget(error_label)
            self.frame_16_layout.addStretch(1)
            return

        png_files = sorted(list(image_dir.glob("*.png")))

        if not png_files:
            no_files_label = QLabel(f"'{image_dir}' 디렉토리에 PNG 파일이 없습니다.")
            no_files_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.frame_16_layout.addWidget(no_files_label)
            self.frame_16_layout.addStretch(1)
            return

        for image_path in png_files:
            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                print(f"경고: 이미지 로드 실패 - {image_path}")
                error_image_label = QLabel(f"이미지 로드 실패:\n{image_path.name}")
                error_image_label.setStyleSheet("color: red;")
                self.frame_16_layout.addWidget(error_image_label)
                continue

            image_label = QLabel()
            scaled_pixmap = pixmap.scaledToWidth(300, Qt.TransformationMode.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setToolTip(str(image_path))
            self.frame_16_layout.addWidget(image_label)

        if isinstance(self.frame_16_layout, QVBoxLayout):
            self.frame_16_layout.addStretch(1)
        print(f"{len(png_files)}개 이미지 로드 완료")


if __name__ == "__main__":
    test_dir = Path("./Data_graph/")
    if not test_dir.exists():
        test_dir.mkdir(parents=True, exist_ok=True)
        print(f"'{test_dir}' 디렉토리를 생성했습니다. 이 디렉토리에 PNG 파일을 넣어주세요.")
        try:
            from PIL import Image, ImageDraw
            # 기존 이미지 파일이 있다면 덮어쓰지 않도록 확인
            if not list(test_dir.glob("sample_image_*.png")):
                for i in range(5):
                    img = Image.new('RGB', (400, 300), color = (73, 109, 137))
                    d = ImageDraw.Draw(img)
                    d.text((10,10), f"Sample Image {i+1}", fill=(255,255,0))
                    img.save(test_dir / f"sample_image_{i+1}.png")
                print(f"'{test_dir}'에 5개의 샘플 PNG 이미지를 생성했습니다.")
            else:
                print(f"'{test_dir}'에 이미 샘플 이미지가 존재합니다. 새로 생성하지 않습니다.")
        except ImportError:
            print("Pillow 라이브러리가 없어 샘플 이미지를 생성할 수 없습니다. 수동으로 PNG 파일을 넣어주세요.")
        except Exception as e:
            print(f"샘플 이미지 생성 중 오류: {e}")

    app = QApplication(sys.argv)
    window = ImageDisplayWindow()
    window.show()
    sys.exit(app.exec())