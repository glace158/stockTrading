from PySide6.QtWidgets import QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.treeWidget = QTreeWidget(self)
        self.treeWidget.setHeaderLabels(["Column 1", "Column 2"])

        # 첫 번째 항목 생성
        rootItem = QTreeWidgetItem(self.treeWidget, ["Root Item", "Root Value"])
        childItem1 = QTreeWidgetItem(rootItem, ["Child 1", "Value 1"])
        childItem2 = QTreeWidgetItem(rootItem, ["Child 2", "Value 2"])
        grandChildItem1 = QTreeWidgetItem(childItem1, ["Grandchild 1", "Value 3"])

        # 두 번째 항목 생성
        secondRootItem = QTreeWidgetItem(self.treeWidget, ["Second Root", "Value 4"])
        
        # UI 설정
        self.treeWidget.expandAll()  # 모든 항목을 확장
        self.setCentralWidget(self.treeWidget)

        # 모든 아이템 가져오기
        self.get_all_items()

    def get_all_items(self):
        # 루트 항목 가져오기 (invisibleRootItem은 실제 루트 아이템)
        rootItem = self.treeWidget.invisibleRootItem()
        # 모든 아이템을 순회하여 출력
        items = self.get_items_recursively(rootItem)
        print(items)
        for item in items:
            print(f"Item: {item.text(0)}, Value: {item.text(1)}")

    def get_items_recursively(self, item):
        items = [item]  # 현재 항목을 리스트에 추가
        for i in range(item.childCount()):  # 자식 아이템 순회
            items.extend(self.get_items_recursively(item.child(i)))  # 자식 아이템들에 대해 재귀 호출
        return items


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
