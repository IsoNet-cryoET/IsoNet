#!usr/bin/env python3
# -*- coding: utf-8 -*-

#from mwr.bin.mwr_main import *
import os, sys
from PyQt5.QtWidgets import *


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # self.resize(300, 300)
        self.status = self.statusBar()
        self.status.showMessage("Telda Wang")
        self.setWindowTitle("WMR")
        self.pwd = ""

        # open file1
        self.line_f1 = QLineEdit()
        self.line_f1.textEdited.connect(self.updateUi)
        self.button_f1 = QPushButton("Get File")
        self.button_f1.clicked.connect(self.get_path_f1)

        # open file2
        self.line_f2 = QLineEdit()
        self.line_f2.textEdited.connect(self.updateUi)
        self.button_f2 = QPushButton("Data File")
        self.button_f2.clicked.connect(self.get_path_f2)


        # paras
        weight_label = QLabel("weight")
        self.line_weight = QLineEdit()
        weight_label.setBuddy(self.line_weight)
        self.line_weight.textEdited.connect(self.updateUi)
        self.line_weight.setText("None")

        data_label = QLabel("data")
        self.line_data = QLineEdit()
        data_label.setBuddy(self.line_data)
        self.line_data.textEdited.connect(self.updateUi)
        self.line_data.setText("None")

        ip_label = QLabel("IP")
        self.line_ip = QLineEdit()
        ip_label.setBuddy(self.line_ip)
        self.line_ip.textEdited.connect(self.updateUi)
        self.line_ip.setText("24")

        DIM_label = QLabel("DIM")
        self.line_DIM = QLineEdit()
        ip_label.setBuddy(self.line_DIM)
        self.line_DIM.textEdited.connect(self.updateUi)
        self.line_DIM.setText("2D")

        nGPU_label = QLabel("nGPU")
        self.line_nGPU = QLineEdit()
        ip_label.setBuddy(self.line_nGPU)
        self.line_nGPU.textEdited.connect(self.updateUi)
        self.line_nGPU.setText("2")

        epochs_label = QLabel("epochs")
        self.line_epochs = QLineEdit()
        epochs_label.setBuddy(self.line_epochs)
        self.line_epochs.textEdited.connect(self.updateUi)
        self.line_epochs.setText(str(40))

        batch_size_label = QLabel("batch_size")
        self.line_batch_size = QLineEdit()
        batch_size_label.setBuddy(self.line_batch_size)
        self.line_batch_size.textEdited.connect(self.updateUi)
        self.line_batch_size.setText(str(32))

        steps_per_epoch_label = QLabel("steps_per_epoch")
        self.line_steps_per_epoch = QLineEdit()
        steps_per_epoch_label.setBuddy(self.line_steps_per_epoch)
        self.line_steps_per_epoch.textEdited.connect(self.updateUi)
        self.line_steps_per_epoch.setText(str(28))

        # run train
        self.button_run = QPushButton("RUN")
        self.button_run.clicked.connect(self.run)

        # close exe
        self.button_close = QPushButton("Close")
        self.button_close.clicked.connect(self.onbuttonclick_close)

        # layout grid
        layout = QGridLayout()
        layout.addWidget(self.line_f1, 0, 0)
        layout.addWidget(self.button_f1, 0, 1)
        layout.addWidget(self.line_f2, 1, 0)
        layout.addWidget(self.button_f2, 1, 1)

        layout.addWidget(weight_label, 3, 0)
        layout.addWidget(self.line_weight, 3, 1)

        layout.addWidget(data_label, 4, 0)
        layout.addWidget(self.line_data, 4, 1)

        layout.addWidget(ip_label, 5, 0)
        layout.addWidget(self.line_ip, 5, 1)

        layout.addWidget(DIM_label, 6, 0)
        layout.addWidget(self.line_DIM, 6, 1)

        layout.addWidget(nGPU_label, 7, 0)
        layout.addWidget(self.line_nGPU, 7, 1)

        layout.addWidget(epochs_label, 8, 0)
        layout.addWidget(self.line_epochs, 8, 1)

        layout.addWidget(batch_size_label, 9, 0)
        layout.addWidget(self.line_batch_size, 9, 1)

        layout.addWidget(steps_per_epoch_label, 10, 0)
        layout.addWidget(self.line_steps_per_epoch, 10, 1)

        layout.addWidget(self.button_run, 11, 3)
        layout.addWidget(self.button_close, 12, 3)

        main_frame = QWidget()
        main_frame.setLayout(layout)
        layout.setSizeConstraint(QLayout.SetFixedSize)
        self.setCentralWidget(main_frame)
        self.updateUi

    def onbuttonclick_close(self):
        print("closing")
        qApp = QApplication.instance()
        qApp.quit()

    def get_path_f1(self):
        # 设置文件扩展名过滤,注意用双分号间隔
        filename1, type = QFileDialog.getOpenFileName(self, "OPEN F1",
                                                      "./",
                                                      "All Files (*);;Text Files (*.txt)")
        self.line_f1.setText(filename1)
        self.pwd = os.path.dirname(self.line_f1.text())
        self.line_f2.setText(self.pwd + os.path.sep + "data.data")
        self.line_data.setText(self.pwd + os.path.sep + "data.data")
        self.line_weight.setText(self.pwd + os.path.sep + "weight.weight")

    def get_path_f2(self):
        # 设置文件扩展名过滤,注意用双分号间隔
        filename2, type = QFileDialog.getOpenFileName(self, "OPEN F2",
                                                      "./",
                                                      "All Files (*);;Text Files (*.txt)")
        self.line_f2.setText(filename2)

    def updateUi(self):
        pass
        # enable = True #bool(self.line_f1.text())
        # self.line_f1.setEnabled(enable)
        # self.line_f2.setEnabled(enable)
        # self.line_ip.setEnabled(enable)
        # self.line_DIM.setE

    def run(self):
        settings = Setting()
        if self.line_ip.text() == "24":
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # specify which GPU(s) to be used

        if self.line_DIM.text() == '2D':
            average_train(self.line_f1.text(), os.path.splitext(self.line_f1.text())[-1], self.line_data.text(), self.line_weight.text(), int(self.line_nGPU.text()), settings)
            # prepare_and_train2D(args.file,args.type,args.data,args.weight,args.gpus,settings)
        elif self.line_DIM.text() == '3D':
            prepare_and_train3D(self.line_f1.text(), os.path.splitext(self.line_f1.text())[-1], self.line_weight.text(), int(self.line_nGPU.text()), settings, args.direc)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    sys.exit(app.exec_())
