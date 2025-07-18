from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from plot_graph import save_log_graph, save_action_graph
from qt.ui_main import Ui_MainWindow
import csv
import os

class GraphPage(QWidget):
    def __init__(self, widgets : Ui_MainWindow):
        super().__init__()
        self.widgets = widgets 

        self.tree_widgets = []  # 생성된 트리 위젯 목록
        self.current_tree_widget = None # 현재 선택된 트리 위젯

        # 그래픽 관련 버튼
        self.widgets.addGraphPushButton.clicked.connect(self.add_new_tree_widget)
        self.widgets.removePushButton.clicked.connect(self.remove_graph)

        self.widgets.imageSizeUpPushButton.clicked.connect(self.increase_size)
        self.widgets.imageSizeDownPushButton.clicked.connect(self.decrease_size)
        
        self.widgets.imageSizeUpPushButton.setEnabled(False)
        self.widgets.imageSizeDownPushButton.setEnabled(False)

        self.pixmap_list = []
        self.graph_image_list = []
        self.image_width_list = []
        self.image_height_list = []

        self.csv_file_path_list = []

    # 현재 클릭한 트리 위젯
    def set_current_tree_widget(self, tree_widget):
        self.current_tree_widget = tree_widget

        for tw in self.tree_widgets:
            tw.setStyleSheet(""" """)
        
        tree_widget.setStyleSheet("""
                                  QTreeWidget{
                                    border: 2px solid rgb(189, 147, 249);
                                  }
                                  """)

        print("트리 위젯이 선택되었습니다.")


    # csv 트리 구성
    def load_csv_to_tree(self, tree_widget, file_path):
        try:
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                
                # 루트 항목 추가 (파일명, 체크박스 없음)
                root_item = QTreeWidgetItem([file_path])
                tree_widget.addTopLevelItem(root_item)

                # 첫 번째 행은 헤더로 설정
                headers = next(reader, None)
                if headers:
                    for header in headers:
                        child_item = QTreeWidgetItem([header])
                        child_item.setFlags(child_item.flags() | Qt.ItemIsUserCheckable)  # 체크박스 플래그 추가
                        child_item.setCheckState(0, Qt.Unchecked)  # 초기 체크 상태
                        root_item.addChild(child_item)

                # 트리 위젯의 stateChanged 처리
                tree_widget.itemChanged.connect(self.handle_item_change)

        except Exception as e:
            print(f"CSV 파일 읽기 오류: {e}")

    # csv 트리 요소 체크 이벤트
    def handle_item_change(self, item, column):
        if item.flags() & Qt.ItemIsUserCheckable:  # 체크박스 항목인지 확인
            state = item.checkState(column)
            if state == Qt.Checked:
                print(f"'{item.text(0)}'가 선택되었습니다.")
            elif state == Qt.Unchecked:
                print(f"'{item.text(0)}'가 선택 해제되었습니다.")
    
    # 새로운 그래프 (트리) 추가하기
    def add_new_tree_widget(self):
        # csv 파일 읽기
        csv_file_paths, _ = QFileDialog.getOpenFileNames(self, "CSV 파일 선택", "", "CSV Files (*.csv)")

        if csv_file_paths:
            # 새로운 QTreeWidget 추가
            new_tree_widget = QTreeWidget(self.widgets.graphDataFrame)
            new_tree_widget.setHeaderLabels(["데이터 항목"])
            self.tree_widgets.append(new_tree_widget)
            self.set_current_tree_widget(new_tree_widget)

            self.widgets.verticalLayout_5.addWidget(new_tree_widget)
            # 새 트리 위젯 클릭 이벤트 연결
            new_tree_widget.mouseDoubleClickEvent = lambda event: self.set_current_tree_widget(new_tree_widget)
            print("새로운 트리가 추가되었습니다.")
            self.current_tree_widget = self.tree_widgets[-1]

            self.load_csv_to_tree(self.current_tree_widget, csv_file_paths[0])
            self.csv_file_path_list.append(csv_file_paths[0])
            self.make_graph()

    # 그래프 삭제
    def remove_graph(self):
        if not self.current_tree_widget:
            return
        
        rootItem = self.current_tree_widget.invisibleRootItem()
        items = self.get_items_recursively(rootItem)

        for item in items:
            item_name = item.text(0)
            if item_name.endswith('.csv'):
                path = item_name
                break

        self.tree_widgets.remove(self.current_tree_widget)
        self.csv_file_path_list.remove(path)
        self.current_tree_widget.deleteLater() 
        self.current_tree_widget = None
        
        if self.tree_widgets:
            self.make_graph()
        else:
            self.pixmap_list = []
            self.graph_image_list = []
            self.image_width_list = []
            self.image_height_list = []

            self._layout_clear(self.widgets.verticalLayout_42)

            no_image = QLabel()
            no_image.setText("No Images")
            self.widgets.verticalLayout_42.addWidget(no_image)

            self.widgets.imageSizeUpPushButton.setEnabled(False)
            self.widgets.imageSizeDownPushButton.setEnabled(False)


    # 그래프 그리고 표시하기
    def make_graph(self):
        # 그래프 설정 초기화
        self.pixmap_list = []
        self.graph_image_list = []
        self.image_width_list = []
        self.image_height_list = []
    
        self._layout_clear(self.widgets.verticalLayout_42)
        self.remove_graph_images()

        for i, csv_file_path in enumerate(self.csv_file_path_list):
            if "action" in csv_file_path:
                save_path = save_action_graph(csv_file_path, fig_num=i)
            elif "log" in csv_file_path:
                save_path = save_log_graph(csv_file_path, fig_num=i)
        
            self.load_graph_image(save_path)

    # 기존에 있던 이미지 지우기
    def remove_graph_images(self):
        action_fig_directory = "./Data_graph/Richdog/"
        state_fig_directory = "./PPO_figs/Richdog/"

        for fig_directory in [action_fig_directory, state_fig_directory]:
            for file in os.scandir(fig_directory): # 기존에 있던 이미지 지우기
                print("Remove File: ",file)
                os.remove(file)
        
    # 저장한 그래프 이미지 불러오기
    def load_graph_image(self, file_path):
        
        pixmap = QPixmap(file_path)
        graph_image = QLabel()
        graph_image.adjustSize()
        graph_image.setPixmap(pixmap)
    
        self.image_width_list.append(pixmap.width())
        self.image_height_list.append(pixmap.height())

        self.pixmap_list.append(pixmap)
        self.graph_image_list.append(graph_image)

        self.widgets.verticalLayout_42.addWidget(graph_image)

        self.widgets.imageSizeUpPushButton.setEnabled(True)
        self.widgets.imageSizeDownPushButton.setEnabled(True)

    # 트리 위젯 자식 순회
    def get_items_recursively(self, item):
        items = [item]  # 현재 항목을 리스트에 추가
        for i in range(item.childCount()):  # 자식 아이템 순회
            items.extend(self.get_items_recursively(item.child(i)))  # 자식 아이템들에 대해 재귀 호출
        return items
    

    # 이미지 크기 조정
    def update_image(self):
        for i, pixmap in enumerate(self.pixmap_list):
            resized_pixmap = pixmap.scaled(self.image_width_list[i], self.image_height_list[i])
            self.graph_image_list[i].setPixmap(resized_pixmap)

    # 이미지 크기 키우기
    def increase_size(self):
        for i in range(len(self.pixmap_list)):
            # 이미지 크기를 증가
            self.image_width_list[i] += 20
            self.image_height_list[i] += 20

        self.update_image()

    # 이미지 크기 감소
    def decrease_size(self):
        for i in range(len(self.pixmap_list)):
            if self.image_width_list[i] > 20 and self.image_height_list[i] > 20:  # 최소 크기 제한
                self.image_width_list[i] -= 20
                self.image_height_list[i] -= 20

        self.update_image()

    # 레이아웃 비우기
    def _layout_clear(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                if item is None:
                    continue

                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()