from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QListView
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt

class CheckBoxListView(QWidget):
    def __init__(self):
        super().__init__()

        self.list_view = QListView()
        self.model = QStandardItemModel(self.list_view)

        # 체크박스가 포함된 항목 추가
        for text in ["Option A", "Option B", "Option C"]:
            item = QStandardItem(text)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setData(Qt.Unchecked, Qt.CheckStateRole)
            self.model.appendRow(item)

        self.list_view.setModel(self.model)

        layout = QVBoxLayout()
        layout.addWidget(self.list_view)
        self.setLayout(layout)

app = QApplication([])
window = CheckBoxListView()
window.show()
app.exec()