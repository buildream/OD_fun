import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton
from Ui_test import Ui_MainWindow

class Example(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()
    def do_button(self):
        self.textBrowser.append('Hello ANU!!!')  

app = QApplication([])
ex=Example()
sys.exit(app.exec_())
