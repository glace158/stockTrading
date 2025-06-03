from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from plot_graph import save_graph
from qt.ui_main import Ui_MainWindow
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import numpy as np
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
                self.make_action_graph()
                self.load_graph_image()
                print(f"'{item.text(0)}'가 선택되었습니다.")
            elif state == Qt.Unchecked:
                self.make_action_graph()
                self.load_graph_image()
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

            for csv_file_path in csv_file_paths:
                if csv_file_path:
                    self.load_csv_to_tree(self.current_tree_widget, csv_file_path)

                if "action" in csv_file_path:
                    self.make_action_graph()
                    self.load_graph_image()
                elif "log" in csv_file_path:
                    self.make_log_graph(csv_file_path)
    
    # csv 삭제
    def remove_csv(self):
        if not self.current_tree_widget:
            return
        
        selected_items = self.current_tree_widget.selectedItems()
        
        if selected_items:
            for item in selected_items:
                index = self.current_tree_widget.indexOfTopLevelItem(item)
                self.current_tree_widget.takeTopLevelItem(index)
        
        self.make_action_graph()
        self.load_graph_image()

    # 그래프 삭제
    def remove_graph(self):
        if not self.current_tree_widget:
            return
        
        self.tree_widgets.remove(self.current_tree_widget)
        self.current_tree_widget.deleteLater() 
        self.current_tree_widget = None
        
        if self.tree_widgets:
            self.make_action_graph()
            self.load_graph_image()
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

    def make_log_graph(self, log_path):
        fig_directory = "PPO_figs" + '/' + "Richdog" + '/'

        for file in os.scandir(fig_directory): # 기존에 있던 이미지 지우기
            print("Remove File: ",file)
            os.remove(file)

        save_graph(log_path)

    def make_action_graph(self):
        rootItem = self.current_tree_widget.invisibleRootItem()
        items = self.get_items_recursively(rootItem)
        
        data = pd.DataFrame() 

        for item in items:
            item_name = item.text(0)
            if item_name.endswith('.csv'):
                path = item_name
                data = pd.read_csv(path)


        # 그래프 그`리기
        fig, ax1 = plt.subplots(figsize=(30, 10))

        # Reward 막대 그래프 (아래쪽 Y축 공유)
        ax1.spines['right'].set_position(('outward', 60))  # 추가 Y축을 오른쪽으로 이동

        reward_colors = ['red' if r < 0 else 'blue' for r in data['reward']]  # 음수는 빨간색, 양수는 파란색
        ax1.bar(data['timestep'], data['reward'], width=0.4, alpha=0.8, label='Reward', color=reward_colors)

        # y축 0에 선 추가
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, label='y = 0 Line')

        ax1.set_ylabel('Reward and Daily Rate')
        ax1.legend(loc='lower center')
        
        # Current Amount 선 그래프 (오른쪽 Y축)
        ax2 = ax1.twinx()  # 오른쪽 Y축 추가
        ax2.plot(data['timestep'], data['total_amt'], label='Total Amount', color='Orange')
        ax2.set_ylabel('Total Amount')
        ax2.legend(loc='upper right')
        ax2.axhline(y=data['total_amt'][0], color='Orange', linestyle='--', linewidth=1, label='init Amount Line')
        
        # Price 선 그래프 (왼쪽 Y축)
        ax3 = ax1.twinx()  # 새로운 Y축 추가
        ax3.plot(data['timestep'], data['price'], label='Price', color='black')
        
        for index, row in data.iterrows():
            if row['order_qty'] == 0:  # order_qty가 0인 경우 회색 점
                ax3.scatter(row['timestep'], row['price'], color='gray', label='Order Qty = 0' if index == 0 else "")
                ax3.annotate(f"{int(row['order_qty'])}", (row['timestep'], row['price']), textcoords="offset points", xytext=(-10, -10), color='gray')
            elif row['action'] > 0:  # action이 음수인 경우 빨간 점
                ax3.scatter(row['timestep'], row['price'], color='red', label='Action > 0' if index == 0 else "")
                ax3.annotate(f"{int(row['order_qty'])}", (row['timestep'], row['price']), textcoords="offset points", xytext=(-10, -10), color='red')
            elif row['action'] < 0:  # action이 양수인 경우 파란 점
                ax3.scatter(row['timestep'], row['price'], color='blue', label='Action < 0' if index == 0 else "")
                ax3.annotate(f"{int(row['order_qty'])}", (row['timestep'], row['price']), textcoords="offset points", xytext=(-10, 10), color='blue')
        
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Price')
        ax3.set_xticks(data['timestep'])  # x축 간격을 timestep에 맞춤
        ax3.legend(loc='upper left')
        ax3.grid(True)

        # 그래프 제목
        plt.title('Price, Current Amount, Reward, and Daily Rate vs Timestep')
        
        plt.savefig("./Data_graph/graph.png")
        
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
        path = "./Data_graph/"
        file_list = os.listdir(path)
        file_list = [file for file in file_list if file.endswith(".png")]
        
        self.pixmap_list = []
        self.graph_image_list = []
        self.image_width_list = []
        self.image_height_list = []
        
        self._layout_clear(self.widgets.verticalLayout_42)

        for file in file_list:
            pixmap = QPixmap(path + file)
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
    
    def update_image(self):
        # 이미지를 현재 크기로 조정하여
        for i, pixmap in enumerate(self.pixmap_list):
            resized_pixmap = pixmap.scaled(self.image_width_list[i], self.image_height_list[i])
            self.graph_image_list[i].setPixmap(resized_pixmap)

    def increase_size(self):

        for i in range(len(self.pixmap_list)):
            # 이미지 크기를 증가
            self.image_width_list[i] += 20
            self.image_height_list[i] += 20

        self.update_image()

    def decrease_size(self):
        # 이미지 크기를 감소
        for i in range(len(self.pixmap_list)):
            if self.image_width_list[i] > 20 and self.image_height_list[i] > 20:  # 최소 크기 제한
                self.image_width_list[i] -= 20
                self.image_height_list[i] -= 20

        self.update_image()

    def _layout_clear(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                if item is None:
                    continue

                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()