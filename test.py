from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget


class WorkerThread(QThread):
    finished_signal = Signal(str)  # 작업 상태를 전달할 시그널

    def __init__(self):
        super().__init__()
        self._is_running = True  # 작업 실행 상태 플래그

    def run(self):
        try:
            for i in range(5):  # 작업 수행
                if not self._is_running:  # 강제 종료 플래그 확인
                    self.finished_signal.emit("작업 강제 종료")
                    return
                print(f"작업 진행 중: {i + 1}")
                self.sleep(1)
            self.finished_signal.emit("작업 정상 종료")  # 작업 완료 시 신호
        except Exception as e:
            self.finished_signal.emit(f"에러: {str(e)}")  # 예외 발생 시 처리

    def stop(self):
        self._is_running = False  # 강제 종료 플래그 설정


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QThread 종료 유형 예제")
        
        self.label = QLabel("준비 완료")
        self.start_button = QPushButton("작업 시작")
        self.stop_button = QPushButton("작업 중단")
        self.stop_button.setEnabled(False)  # 초기 상태에서 비활성화

        self.start_button.clicked.connect(self.start_thread)
        self.stop_button.clicked.connect(self.stop_thread)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.thread = WorkerThread()
        self.thread.finished_signal.connect(self.update_label)  # 작업 완료 신호 연결

    def start_thread(self):
        self.label.setText("작업 진행 중...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.thread = WorkerThread()  # 새로운 스레드 생성 (재사용 가능)
        self.thread.finished_signal.connect(self.update_label)  # 신호 연결
        self.thread.start()

    def stop_thread(self):
        self.label.setText("작업 중단 요청...")
        self.thread.stop()  # 강제 종료 요청

    def update_label(self, message):
        self.label.setText(message)  # 종료 상태에 따라 메시지 업데이트
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()