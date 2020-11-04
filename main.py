import sys
import os
from PyQt5 import QtWidgets, QtCore, QtGui, uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication,QMessageBox ,QWidget, QVBoxLayout, QLineEdit, QPushButton, QFileDialog , QLabel, QTextEdit
from gui import Ui_MainWindow
import numpy as np
import time
import logging
import torch
import matplotlib.pyplot as plt
from eval_utils.PSNR_eval import PSNR
from eval_utils.SSIM_eval import SSIM
from eval_utils.BCH_eval import BCH
from model import AUNET
import PIL.Image as Image

# print(os.getcwd())

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(r'footage\Logo BK.png')) 
        self.init()
        self.browseButton.clicked.connect(self.get_file_path)
        self.processButton.clicked.connect(self.process) 
             
        
    def get_file_path(self):
        test_path = os.path.join(os.getcwd(),"img_test")
        name = QFileDialog.getOpenFileName(self, 'Open file',test_path, "Image files (*.jpg *.png)")
        self.imagePath = name[0]
        logging.info("{} Path to test Image: {}".format(time.ctime(), self.imagePath))
        pixmap = QPixmap(self.imagePath)
        self.inputImage.setPixmap(QPixmap(pixmap))  
        self.imgDark = plt.imread(self.imagePath) 
        self.imgDark = self.imgDark.transpose((2, 0, 1))
        self.imgDark = torch.from_numpy(self.imgDark).type(torch.FloatTensor)
        self.imgDark = torch.unsqueeze(self.imgDark, dim=0)
        self.imgDark = torch.cat((self.zero_channel.to(self.device), self.imgDark), dim=1)
        logging.info("{} Finish loading image: {}".format(time.ctime(), self.imagePath))
    
    def process(self):
        start_time = time.time()
        self.imgOut = self.net(self.imgDark)
        self.imgOut = torch.clamp(self.imgOut, 0.0, 1.0)
        logging.info("{} Processing time: {}".format(time.ctime(), time.time()-start_time))
        self.evaluate(self.imgDark, self.imgOut)
        self.save_image_imshow(self.imgOut)
        
    def evaluate(self, dark_img, out_img):
        pass
    
    def save_image_imshow(self, image):
        output = self.imgOut.cpu().detach()[0,1:,:,:].numpy()*255.0
        output = output.astype(np.uint8)
        output = np.transpose(output, (1,2,0))
        print(output.shape)
        output_ = Image.fromarray(output, "RGB")
        output_.save(self.result_dir + '%s.jpg'%("output"))
        logging.info("{} Image saved ".format(time.ctime()))        
        image_show = QtGui.QImage(output.tobytes(), 
                                  output.shape[1],
                                  output.shape[0], 
                                  output.shape[1] * 3,
                                  QtGui.QImage.Format_RGB888)        
        pix = QtGui.QPixmap(image_show)
        self.outputImage.setPixmap(pix)
   
    def load_model(self):
        logging.info("{} Loading model to device {}".format(time.ctime(), self.device))
        start_time = time.time()
        self.net = AUNET(n_classes=4)
        self.net = self.net.to(self.device)
        self.net.eval()
        logging.info("{} Loading checkpoint '{}'".format(time.ctime(), self.checkpoint_path))
        checkpoint = torch.load(self.checkpoint_path, map_location = self.device)
        self.net.load_state_dict(checkpoint['state_dict'])
        logging.info("{} Loaded checkpoint '{}' (epoch {})".format(time.ctime(), self.checkpoint_path, checkpoint['epoch']))
        logging.info("{} Finish loading model in {}".format(time.ctime(), time.time()-start_time))
    
    def init(self):
        self.result_dir = "result/"
        self.checkpoint_path = "checkpoint/checkpoint_best.pth.tar"
        self.zero_channel = torch.zeros((1,1,400,600))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model() 
                  
def main():        
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    logging.basicConfig(filename='log/runtime.log', level=logging.INFO)
    logging.info("\n")
    main()