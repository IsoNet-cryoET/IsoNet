#!/user/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
from PyQt5.QtWidgets import *


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.resize(600, 500)
        self.status = self.statusBar()
        self.status.showMessage("Telda Wang")
        self.setWindowTitle("WMR")

        # open file1
        self.line_f1 = QLineEdit()
        self.line_f1.textEdited.connect(self.updateUi)
        self.button_f1 = QPushButton("F1")
        self.button_f1.clicked.connect(self.get_path_f1)

        # open file2
        self.line_f2 = QLineEdit()
        self.line_f2.textEdited.connect(self.updateUi)
        self.button_f2 = QPushButton("F2")
        self.button_f2.clicked.connect(self.get_path_f2)

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
        layout.addWidget(self.button_run, 2, 1)
        layout.addWidget(self.button_close)
        main_frame = QWidget()
        main_frame.setLayout(layout)
        layout.setSizeConstraint(QLayout.SetFixedSize)
        self.setCentralWidget(main_frame)


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

    def get_path_f2(self):
        # 设置文件扩展名过滤,注意用双分号间隔
        filename2, type = QFileDialog.getOpenFileName(self, "OPEN F2",
                                                          "./",
                                                          "All Files (*);;Text Files (*.txt)")
        self.line_f2.setText(filename2)

    def run(self):
        self.f1 = open(self.line_f1.text(),'r')
        self.f2 = open(self.line_f2.text(),'r')
        path = os.path.dirname(self.line_f1.text())
        print(path)
        self.output = open((path + os.path.sep + 'output.txt'), 'a')
        self.output.write("111\n")
        self.output.write("111")
        print("done!")
        self.f1.close()
        self.f2.close()
        self.output.close()

    def updateUi(self):
        enable = bool(self.line_f1.text())
        self.line_f1.setEnabled(enable)
        self.line_f2.setEnabled(enable)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    app.exec_()
