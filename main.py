import sys
import os
from PyQt5 import QtWidgets, QtCore, QtGui, uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication,QMessageBox ,QWidget, QVBoxLayout, QLineEdit, QPushButton, QFileDialog , QLabel, QTextEdit
from gui import Ui_MainWindow
import numpy as np
import time

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(r'footage\Logo BK.png')) 
        self.browseButton.clicked.connect(self.get_file_path)
        
    def get_file_path(self):
        test_path = os.path.join(os.getcwd(),"img_test")
        name = QFileDialog.getOpenFileName(self, 'Open file',test_path, "Image files (*.jpg *.png)")
        self.imagePath = name[0]
        print(self.imagePath)
        pixmap = QPixmap(self.imagePath)
        self.inputImage.setPixmap(QPixmap(pixmap))   
    
    def process(self):
        pass
    
    def load_model(self):
        pass
            
            
def main():        
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    
    main()