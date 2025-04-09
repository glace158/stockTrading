from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from qt.ui_main import Ui_MainWindow
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import numpy as np

class GraphPage(QWidget):
    def __init__(self, widgets : Ui_MainWindow):
        super().__init__()
        self.widgets = widgets 

        self.tree_widgets = [self.widgets.treeWidget]  # 생성된 트리 위젯 목록
        self.current_tree_widget = self.widgets.treeWidget # 현재 선택된 트리 위젯
        self.widgets.treeWidget.mouseDoubleClickEvent = lambda event: self.set_current_tree_widget(self.widgets.treeWidget)
        
        # 그래픽 관련 버튼
        self.widgets.addCSVPushButton.clicked.connect(self.add_csv_files_to_current_tree)
        self.widgets.addGraphPushButton.clicked.connect(self.add_new_tree_widget)
        self.widgets.removePushButton.clicked.connect(self.remove_graph)
        self.widgets.removeCSVPushButton.clicked.connect(self.remove_csv)

        self.widgets.imageSizeUpPushButton.clicked.connect(self.increase_size)
        self.widgets.imageSizeDownPushButton.clicked.connect(self.decrease_size)
        
        self.widgets.imageSizeUpPushButton.setEnabled(False)
        self.widgets.imageSizeDownPushButton.setEnabled(False)

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

    # csv 파일 읽기
    def add_csv_files_to_current_tree(self):
        if self.current_tree_widget:
            csv_file_paths, _ = QFileDialog.getOpenFileNames(self, "CSV 파일 선택", "", "CSV Files (*.csv)")
            for csv_file_path in csv_file_paths:
                if csv_file_path:
                    self.load_csv_to_tree(self.current_tree_widget, csv_file_path)
        else:
            print("먼저 트리를 선택하세요.")

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
                self.make_graph()
                self.load_graph_image()
                print(f"'{item.text(0)}'가 선택되었습니다.")
            elif state == Qt.Unchecked:
                self.make_graph()
                self.load_graph_image()
                print(f"'{item.text(0)}'가 선택 해제되었습니다.")
    
    # 새로운 그래프 (트리) 추가하기
    def add_new_tree_widget(self):
        # 새로운 QTreeWidget 추가
        new_tree_widget = QTreeWidget(self.widgets.graphDataFrame)
        new_tree_widget.setHeaderLabels(["데이터 항목"])
        self.tree_widgets.append(new_tree_widget)

        self.widgets.verticalLayout_5.addWidget(new_tree_widget)
        # 새 트리 위젯 클릭 이벤트 연결
        new_tree_widget.mouseDoubleClickEvent = lambda event: self.set_current_tree_widget(new_tree_widget)
        print("새로운 트리가 추가되었습니다.")
    
    # csv 삭제
    def remove_csv(self):
        if not self.current_tree_widget:
            return
        
        selected_items = self.current_tree_widget.selectedItems()
        
        if selected_items:
            for item in selected_items:
                index = self.current_tree_widget.indexOfTopLevelItem(item)
                self.current_tree_widget.takeTopLevelItem(index)
        
        self.make_graph()
        self.load_graph_image()

    # 그래프 삭제
    def remove_graph(self):
        if not self.current_tree_widget:
            return
        
        self.tree_widgets.remove(self.current_tree_widget)
        self.current_tree_widget.deleteLater() 
        self.current_tree_widget = None
        
        if self.tree_widgets:
            self.make_graph()
            self.load_graph_image()
        else:
            self.widgets.graph_image.clear()
            self.widgets.graph_image.setText("No Images")
            self.widgets.imageSizeUpPushButton.setEnabled(False)
            self.widgets.imageSizeDownPushButton.setEnabled(False)

    # 그래프 그리기
    def make_graph(self):
        color_list = ['red', 'olive', 'blue', 'hotpink','orange', 'green', 'brown', 'yellow', 'lime', 'cyan', 'navy', 'violet', 'purple', 'magenta', 'pink', 'gray']
        
        fig_count = len(self.tree_widgets)
        fig = plt.figure(figsize=(25, 10 * fig_count))
        
        ax = fig.add_subplot()
        ax.grid()
        for i in range(fig_count):
            rootItem = self.tree_widgets[i].invisibleRootItem()
            ax = plt.subplot(fig_count,1, i + 1)

            items = self.get_items_recursively(rootItem)

            datas = pd.DataFrame()

            color_index = 0    
            gap = 0
            for item in items:
                item_name = item.text(0)

                if item_name.endswith('.csv'):
                    path = item_name
                    datas = pd.read_csv(path)
                    ax = ax.twinx()
                    
                    continue
                
                state = item.checkState(0)
                if state == Qt.Checked:
                    if "stck_bsop_date" in datas.columns: 
                        datas["stck_bsop_date"] = pd.to_datetime(datas["stck_bsop_date"],format="%Y%m%d")
                        
                        ax.plot(datas["stck_bsop_date"], datas[item_name], color=color_list[color_index], label=item_name)
                        
                        ax.set_xticklabels(datas["stck_bsop_date"], rotation=45)
                        ax.xaxis.set_major_locator(dates.MonthLocator())
                    elif "timestep" in datas.columns:
                        if item_name == 'action':
                            for i in range(len(datas[item_name])):
                                if datas["order_qty"][i] == 0:
                                    color = 'gray' 
                                elif datas[item_name][i] < 0:
                                    color = 'blue'
                                elif datas[item_name][i] > 0:
                                    color = 'red'
                                else:
                                    color = 'gray'

                                plt.vlines(i, -1.0, 1.0, color=color, linestyle='solid', linewidth=3)
                        else:       
                            ax.plot(datas["timestep"], datas[item_name], color=color_list[color_index], label=item_name)
                        
                        ax.set_xticks(datas["timestep"])

                        
                    else:
                        ax.plot(datas[item_name], color=color_list[color_index], label=item_name)
                        
                        ax.set_xticks(ticks=np.arange(len(datas.values)))

                    ax.spines.right.set_position(("axes",gap + 1))
                    ax.set_ylabel(item_name, color=color_list[color_index])
                    color_index = 0 if len(color_list) - 1 <= color_index else color_index + 1
                    gap += 0.03
        
        plt.savefig("./Data_graph/graph.png")

    # 저장한 그래프 이미지 불러오기
    def load_graph_image(self):

        self.pixmap = QPixmap("./Data_graph/graph.png")
        self.widgets.graph_image.adjustSize()
        self.widgets.graph_image.setPixmap(self.pixmap)
        
        self.image_width = self.pixmap.width()
        self.image_height = self.pixmap.height()
        
        self.widgets.imageSizeUpPushButton.setEnabled(True)
        self.widgets.imageSizeDownPushButton.setEnabled(True)

    # 트리 위젯 자식 순회
    def get_items_recursively(self, item):
        items = [item]  # 현재 항목을 리스트에 추가
        for i in range(item.childCount()):  # 자식 아이템 순회
            items.extend(self.get_items_recursively(item.child(i)))  # 자식 아이템들에 대해 재귀 호출
        return items
    
    def update_image(self):
        # 이미지를 현재 크기로 조정하여 QLabel에 표시
        resized_pixmap = self.pixmap.scaled(self.image_width, self.image_height)
        self.widgets.graph_image.setPixmap(resized_pixmap)

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