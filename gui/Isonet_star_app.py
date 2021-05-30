'''
    File name: Isonet_star_app.py
    Author: Hui Wang (EICN)
    Date created: 4/21/2021
    Date last modified: 5/8/2021
    Python Version: 3.6.5
'''
from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import QTableWidgetItem,QMessageBox
from PyQt5.QtCore import QObject, pyqtSlot,QProcess
from IsoNet.gui.isonet_gui import Ui_MainWindow ##need to change in the package
import sys
import os
from IsoNet.gui.model_star import Model, setTableWidget #need to change in the package
import time
from threading import Thread
from IsoNet.util.metadata import MetaData,Label,Item
#from subprocess import call,Popen

class MainWindowUIClass( Ui_MainWindow ):
    def __init__( self ):
        '''Initialize the super class
        '''
        super().__init__()
        self.model = Model()
        self.p = None
        if os.path.isfile(self.model.pid_file):
            os.remove(self.model.pid_file)
        
    def setupUi( self, MW ):
        ''' Setup the UI of the super class, and add here code
        that relates to the way we want our UI to operate.
        '''
        super().setupUi( MW )

        setTableWidget(self.tableWidget, self.model.md)

        self.tableWidget.cellPressed[int, int].connect(self.browseSlotTable)

        #self.tableWidget = TableWidget(df)
        #self.tableWidget.setCurrentCell(0,0)
        #self.tableWidget.currentCellChanged[int,int,int,int].connect(self.updateMDItem) 
        self.tableWidget.cellChanged[int,int].connect(self.updateMDItem) 
        #self.tableWidget.setCellWidget(0,0,self.pushButton_insert)
        #self.horizontalLayout.addWidget(self.tableWidget)

        ########################
        # connect function to buttons
        ########################
        self.pushButton_insert.clicked.connect(self.copyRow)
        self.pushButton_delete.clicked.connect(self.removeRow)
        #self.pushButton_update.clicked.connect(self.updateMD)
        self.pushButton_open_star.clicked.connect(self.open_star)
        self.pushButton_3dmod.clicked.connect(self.view_3dmod)

        self.button_deconov_dir.clicked.connect(lambda: self.browseFolderSlot("deconv_dir"))
        self.button_mask_dir.clicked.connect(lambda: self.browseFolderSlot("mask_dir"))
        self.button_subtomo_dir.clicked.connect(lambda: self.browseFolderSlot("subtomo_dir"))
        self.button_result_dir_refine.clicked.connect(lambda: self.browseFolderSlot("result_dir_refine"))
        self.button_result_dir_predict.clicked.connect(lambda: self.browseFolderSlot("result_dir_predict"))
        
        self.button_subtomo_star_refine.clicked.connect(lambda: self.browseSlot("subtomo_star_refine"))
        self.button_pretrain_model_refine.clicked.connect(lambda: self.browseSlot("pretrain_model_refine"))
        self.button_tomo_star_predict.clicked.connect(lambda: self.browseSlot("tomo_star_predict"))
        self.button_pretrain_model_predict.clicked.connect(lambda: self.browseSlot("pretrain_model_predict"))
        self.button_continue_iter.clicked.connect(lambda: self.browseSlot("continue_from"))
        
        self.pushButton_deconv.clicked.connect(self.deconvolve)
        self.pushButton_generate_mask.clicked.connect(self.make_mask)
        self.pushButton_extract.clicked.connect(self.extract_subtomo)
        self.pushButton_refine.clicked.connect(self.refine)
        self.pushButton_predict.clicked.connect(self.predict)
        self.pushButton_predict_3dmod.clicked.connect(self.view_predict_3dmod)

        self.actionGithub.triggered.connect(self.openGithub)
        #########################
        #set icon location
        #########################
        isonet_path = os.popen("which isonet.py").read()
        tmp = isonet_path.split("bin/isonet.py")
        root_path = tmp[0]

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(root_path+"gui/icons/icon_folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button_deconov_dir.setIcon(icon)
        self.button_mask_dir.setIcon(icon)
        self.button_subtomo_star_refine.setIcon(icon)
        self.button_subtomo_dir.setIcon(icon)
        self.button_pretrain_model_refine.setIcon(icon)
        self.button_result_dir_refine.setIcon(icon)
        self.button_tomo_star_predict.setIcon(icon)
        self.button_pretrain_model_predict.setIcon(icon)
        self.button_result_dir_predict.setIcon(icon)
        self.button_continue_iter.setIcon(icon)
        
        ###Set up log file monitor###
        import datetime
        now = datetime.datetime.now()
        #self.textBrowser_log.setText(now.strftime("%Y-%m-%d %H:%M:%S"))
        if not self.model.isValid(self.model.log_file):
            os.system("echo {} > {}".format(now.strftime("%Y-%m-%d %H:%M:%S"), self.model.log_file))
            
        self.textBrowser_log.setText(self.model.getLogContent(self.model.log_file))
        self.textBrowser_log.moveCursor(QtGui.QTextCursor.End)
        self.log_watcher = QtCore.QFileSystemWatcher([self.model.log_file])
        self.log_watcher.fileChanged.connect(self.update_log)
    
    def update_log(self, filename):

        self.textBrowser_log.setText(self.model.getLogContent(filename))
        self.textBrowser_log.moveCursor(QtGui.QTextCursor.End)
        self.log_watcher.addPath(self.model.log_file)
        
    def start_process(self, cmd, btn):
  
        if self.mw.p is None:  # No process running.
            #self.message("Executing process")

            self.mw.p = QProcess()  # Keep a reference to the QProcess (e.g. on self) while it's running.
            if btn.text() == "Refine":
                btn.setText("Stop")
                btn.setStyleSheet('QPushButton {color: red;}')
            else:
                btn.setEnabled(False)
            self.mw.p.finished.connect(lambda: self.process_finished(btn))  # Clean up once complete.
            self.mw.p.start(cmd)
            
        elif btn.text() =="Stop":
            if self.mw.p:
                self.mw.p.kill()
            #elif self.model.refine_pid:
            #    pass
            else:
                btn.setText("Refine")
        else:
            msg = QMessageBox()
            msg.setWindowTitle("Warning!")
            msg.setText("Already runing another job, please wait until it finished!")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()
            #print("already runing another job, please wait until it finished!")

    def process_finished(self, btn):
        
        if btn.text() == "Stop":
            btn.setText("Refine")
            btn.setStyleSheet('QPushButton {color: black;}')
        else:
            btn.setEnabled(True)
        self.model.read_star()
        setTableWidget(self.tableWidget, self.model.md)   
        
        #print("Process finished.")
        self.mw.p = None
        #self.p = None
        
    def removeRow(self):
        #print(self.tableWidget.selectionModel().selectedIndexes()[0].row())
        #print(self.tableWidget.selectionModel().selectedIndexes()[0].column())

        indices = self.tableWidget.selectionModel().selectedRows() 
        if indices:
            for index in sorted(indices,reverse=True):
                self.tableWidget.removeRow(index.row()) 
        self.updateMD()

    def copyRow(self):
        rowCount = self.tableWidget.rowCount()
        columnCount = self.tableWidget.columnCount()
        if rowCount <=0 :
            self.tableWidget.insertRow(self.tableWidget.rowCount())
            for j in range(columnCount):
                #self.model.md._setItemValue(it,Label(self.model.header[j+1]),self.tableWidget.item(i, j).text())
                #print(self.default_value(self.model.header[j+1]))
                self.tableWidget.setItem(0, j, QTableWidgetItem(self.default_value(self.model.header[j+1])))
                #print(self.tableWidget.item(0, j).text())
        else:
            indices = self.tableWidget.selectionModel().selectedRows() 

            if indices:
                for index in sorted(indices):
                    self.tableWidget.insertRow(self.tableWidget.rowCount())
                    rowCount = self.tableWidget.rowCount()
                    for j in range(columnCount):
                        #self.tableWidget.cellChanged[rowCount-1, j].connect(self.updateMD)  
                        self.tableWidget.setItem(rowCount-1, j, QTableWidgetItem(self.tableWidget.item(index.row(), j).text()))
            else:
                self.tableWidget.insertRow(self.tableWidget.rowCount())
                rowCount = self.tableWidget.rowCount()
                for j in range(columnCount):
                    if not self.tableWidget.item(rowCount-2, j) is None:
                        self.tableWidget.setItem(rowCount-1, j, QTableWidgetItem(self.tableWidget.item(rowCount-2, j).text()))
        self.updateMD()
    
    def default_value(self, label):
        
        switcher = {
            "rlnMicrographName": "None",
            "rlnPixelSize": "1",
            "rlnDefocus": "0",
            "rlnNumberSubtomo":"100",
            "rlnSnrFalloff":"1",
            "rlnDeconvStrength": "1",
            "rlnDeconvTomoName":"None",
            "rlnMaskDensityPercentage": "50",
            "rlnMaskStdPercentage": "50",
            "rlnMaskName": "None"
        }
        #print(label,switcher.get(label, "None"))
        return switcher.get(label, "None")
        
    def updateMD ( self ):
        star_file = self.model.tomogram_star
        rowCount = self.tableWidget.rowCount()
        columnCount = self.tableWidget.columnCount()
        data = self.model.md._data
        self.model.md = MetaData()
        self.model.md.addLabels('rlnIndex')
        for j in range(columnCount):
            self.model.md.addLabels(self.model.header[j+1])
            #self.model.md.addLabels(self.tableWidget.horizontalHeaderItem(j).text())

        for i in range(rowCount):
            #TODO check the folder contains only tomograms.
            it = Item()
            self.model.md.addItem(it)
            self.model.md._setItemValue(it,Label('rlnIndex'),str(i+1))
            for j in range(columnCount):
                try:
                    #print("update:",Label(self.model.header[j+1]),self.tableWidget.item(i, j).text())
                    self.model.md._setItemValue(it,Label(self.model.header[j+1]),self.tableWidget.item(i, j).text())

                    #self.model.md._setItemValue(it,Label(self.tableWidget.horizontalHeaderItem(j).text()),self.tableWidget.item(i, j).text())
                except:
                    previous_value = getattr(data[i],self.model.header[j+1])
                    self.model.md._setItemValue(it,Label(self.model.header[j+1]),previous_value)
                    self.tableWidget.setItem(i, j, QTableWidgetItem(str(previous_value)))
                    #print("error in seeting values for {}! set it to previous value automatically.".format(self.tableWidget.horizontalHeaderItem(j).text()))
        self.model.md.write(star_file)

    def updateMDItem ( self, i, j ):
        try:
            current_value = self.tableWidget.item(i, j).text()
            #self.model.md._setItemValue(self.mnodel.md._data[i],Label(self.model.header[j+1]),current_value)
            #for row,it in enumerate(self.model.md):
            #    print(i,j)
            #    if row == i:
            #        self.model.md._setItemValue(it,Label(self.tableWidget.horizontalHeaderItem(j).text()),self.tableWidget.item(i, j).text())
            self.updateMD()
        except:
            pass
     
    def view_3dmod(self):
        slected_items = self.tableWidget.selectedItems()
        if len(slected_items) > 0:
            cmd = "3dmod"
            for item in slected_items:
                i = item.row()
                j = item.column()
                item_text = self.tableWidget.item(i, j).text()
                if item_text[-4:] == '.mrc' or item_text[-4:] == '.rec':
                    cmd = "{} {}".format(cmd,item_text)
            #print(cmd)
            if cmd != "3dmod":
                os.system(cmd)
            else:
                print("selected items are not mrc or rec file(s)")
            
        
        '''
        i = self.tableWidget.currentRow()
        j = self.tableWidget.currentColumn() 
        item = self.tableWidget.item(i, j).text()
        if item[-4:] in ['.mrc','.rec']:
            cmd = "3dmod {}".format(item)
            os.system(cmd)
        else:
       '''
		
    # slot
    
    def switch_btn(self, btn):
        switcher = {
            "mask_dir": self.lineEdit_mask_dir,
            "deconv_dir": self.lineEdit_deconv_dir,
            "subtomo_dir": self.lineEdit_subtomo_dir,
            "result_dir_refine": self.lineEdit_result_dir_refine,
            "result_dir_predict": self.lineEdit_result_dir_predict,
            "subtomo_star_refine":self.lineEdit_subtomo_star_refine,
            "pretrain_model_refine":self.lineEdit_pretrain_model_refine,
            "tomo_star_predict": self.lineEdit_tomo_star_predict,
            "pretrain_model_predict":self.lineEdit_pretrain_model_predict,
            "continue_from": self.lineEdit_continue_iter
        }
        return switcher.get(btn, "Invaid btn name")
        
        
    def browseSlot( self , btn ):
        ''' Called when the user presses the Browse button
        '''
        lineEdit = self.switch_btn(btn)
        
        pwd = os.getcwd().replace("\\","/")
        
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        
        flt = "All Files (*)"
        if btn == "continue_from":
            flt = "json file (*.json);;All Files (*)"
        if btn == "subtomo_star_refine" or btn == "tomo_star_predict":
            flt = "star file (*.star);;All Files (*)"
        if btn == "pretrain_model_refine" or btn == "pretrain_model_predict":
            flt = "model file (*.h5);;All Files (*)"
            
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "Choose File",
                        "",
                        flt,
                        options=options)
        if fileName:
            #self.model.setFileName( fileName )
            #######
            #cmd = "echo choose file: {} >> log.txt ".format(fileName)
            #os.system(cmd)
            #self.logWindow.append("choose file: {}".format(fileName) )
            simple_name = self.model.sim_path(pwd,fileName)
            lineEdit.setText( simple_name )
            #self.logWindow.moveCursor(QtGui.QTextCursor.End) 
            #######
            #self.refreshAll()
        #self.debugPrint( "Browse button pressed" )

    def browseFolderSlot( self , btn):
        ''' 
            Called when the user presses the Browse folder button
            TODO: add file name filter
        '''
        lineEdit = self.switch_btn(btn)
        try:
            pwd = os.getcwd().replace("\\","/")
            dir_path=QtWidgets.QFileDialog.getExistingDirectory(None,"Choose Directory",pwd)
            #self.model.setFolderName( dir_path )
            #cmd = "echo choose folder: {} >> log.txt ".format(dir_path)
            #os.system(cmd)
            #self.logWindow.append("choose folder: {}".format(dir_path) )
            #pwd = os.getcwd().replace("\\","/")

            simple_path = self.model.sim_path(pwd,dir_path)

            lineEdit.setText( simple_path )
            #self.logWindow.moveCursor(QtGui.QTextCursor.End) 
            #self.refreshAll()
        except:
            ##TODO: record to log.
            pass

    def browseSlotTable( self , i, j):
        ''' Called when the user presses the Browse folder button
        '''
        if j == 0:
            #lineEdit = self.switch_btn(btn)
            try:
                options = QtWidgets.QFileDialog.Options()
                options |= QtWidgets.QFileDialog.DontUseNativeDialog
                fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                                None,
                                "Choose File",
                                "",
                                "mrc or rec file (*.mrc *.rec) ;; All Files (*)",
                                options=options)
                if not fileName:
                    fileName = self.tableWidget.item(i, j).text()
                pwd = os.getcwd().replace("\\","/")

                simple_path = self.model.sim_path(pwd,fileName)

                self.tableWidget.setItem(i, j, QTableWidgetItem(simple_path))
            except:
                ##TODO: record to log.
                pass
        else:
            pass

    def deconvolve( self ):
          
        #self.pushButton_deconv.setEnabled(False)
        #self.pushButton_deconv.setDisabled(True)
        #self.pushButton_deconv.update()
        #self.pushButton_deconv.repaint()

        tomogram_star = self.model.tomogram_star
        cmd = "isonet.py deconv {} ".format(tomogram_star)
        
        if self.lineEdit_deconv_dir.text():
            cmd = "{} --deconv_folder {}".format(cmd, self.lineEdit_deconv_dir.text())
        if self.lineEdit_tomo_index_deconv.text():
            cmd = "{} --tomo_idx {}".format(cmd, self.lineEdit_tomo_index_deconv.text())
        if self.lineEdit_ncpu.text():
            cmd = "{} --ncpu {}".format(cmd, self.lineEdit_ncpu.text())
        if self.lineEdit_highpassnyquist.text():
            cmd = "{} --highpassnyquist {}".format(cmd, self.lineEdit_highpassnyquist.text())
        if self.lineEdit_tile.text():
            cmd = "{} --tile {}".format(cmd, self.lineEdit_tile.text())
        if self.lineEdit_overlap.text():
            cmd = "{} --overlap {}".format(cmd, self.lineEdit_overlap.text())

        if self.checkBox_only_print_command_prepare.isChecked():
            print(cmd)
        else:
            #print("deconv_else")
            #cmd = "{} &>> {}".format(cmd,self.model.log_file)
            #print(cmd)

            self.start_process(cmd,self.pushButton_deconv)
            #process_deconv = Popen(cmd,shell=True,stdin=None,stdout=None, stderr=None)
            #process_deconv.communicate()
            #os.system("echo {} >> {}".format(process_deconv.pid,self.model.pid_file))
            #while True:
            #    poll = process_deconv.poll()
            #   print(process_deconv.pid, None)
            #    if poll != None:
            #        break
            #print("deconv_after")
            
            #os.system(cmd)
        #self.pushButton_deconv.setEnabled(True)
        #os.remove(self.model.pid_file)
        #self.model.read_star()
        #setTableWidget(self.tableWidget, self.model.md)   


    def make_mask( self ):
        #print("#####making mask############")

        tomogram_star = self.model.tomogram_star

        cmd = "isonet.py make_mask {} ".format(tomogram_star)

        if self.lineEdit_mask_dir.text():
            cmd = "{} --mask_folder {}".format(cmd, self.lineEdit_mask_dir.text())
        if self.lineEdit_patch_size.text():
            cmd = "{} --patch_size {}".format(cmd, self.lineEdit_patch_size.text())
        if not self.checkBox_use_deconv_mask.isChecked():
            cmd = "{} --use_deconv_tomo {}".format(cmd, False)
        if self.lineEdit_tomo_index_mask.text():
            cmd = "{} --tomo_idx {}".format(cmd, self.lineEdit_tomo_index_mask.text())
        if self.lineEdit_z_crop.text():
            cmd = "{} --z_crop {}".format(cmd, self.lineEdit_z_crop.text())

        if self.checkBox_only_print_command_prepare.isChecked():
            print(cmd)
        else:
            self.start_process(cmd,self.pushButton_generate_mask)
            #os.system(cmd)
                          
        #self.model.read_star()
        #setTableWidget(self.tableWidget, self.model.md)

    def extract_subtomo( self ):
        tomogram_star = self.model.tomogram_star

        cmd = "isonet.py extract {} ".format(tomogram_star)
        
        if self.lineEdit_subtomo_dir.text():
            cmd = "{} --subtomo_folder {}".format(cmd, self.lineEdit_subtomo_dir.text())
        if self.lineEdit_subtomo_star_extract.text():
            cmd = "{} --subtomo_star {}".format(cmd, self.lineEdit_subtomo_star_extract.text())
        if self.lineEdit_cube_size_extract.text():
            cmd = "{} --cube_size {}".format(cmd, self.lineEdit_cube_size_extract.text())
        if not self.checkBox_use_deconv_extract.isChecked():
            cmd = "{} --use_deconv_tomo {}".format(cmd, False)
        if self.lineEdit_tomo_index_extract.text():
            cmd = "{} --tomo_idx {}".format(cmd, self.lineEdit_tomo_index_extract.text())
        
        if self.checkBox_only_print_command_prepare.isChecked():
            print(cmd)
        else:
            self.start_process(cmd,self.pushButton_extract)
            #self.create_subprocess(cmd)
            #os.system(cmd)

    def refine( self ):

        subtomo_star = self.lineEdit_subtomo_star_refine.text() if self.lineEdit_subtomo_star_refine.text() else "subtomo.star"

        cmd = "isonet.py refine {} ".format(subtomo_star)

        if self.lineEdit_gpuID_refine.text():
            cmd = "{} --gpuID {}".format(cmd, self.lineEdit_gpuID_refine.text())
        if self.lineEdit_pretrain_model_refine.text():
            cmd = "{} --pretrained_model {}".format(cmd, self.lineEdit_pretrain_model_refine.text())
        if self.lineEdit_continue_iter.text():
            cmd = "{} --continue_from {}".format(cmd, self.lineEdit_continue_iter.text())
            #cmd = "{} --continue_iter {}".format(cmd, self.lineEdit_continue_iter.text())
        if self.lineEdit_result_dir_refine.text():
            cmd = "{} --result_dir {}".format(cmd, self.lineEdit_result_dir_refine.text())
        if self.lineEdit_preprocessing_ncpus.text():
            cmd = "{} --preprocessing_ncpus {}".format(cmd, self.lineEdit_preprocessing_ncpus.text())
            
        if self.lineEdit_iteration.text():
            cmd = "{} --iterations {}".format(cmd, self.lineEdit_iteration.text())
        if self.lineEdit_batch_size.text():
            cmd = "{} --batch_size {}".format(cmd, self.lineEdit_batch_size.text())
        if self.lineEdit_epoch.text():
            cmd = "{} --epochs {}".format(cmd, self.lineEdit_epoch.text()) 
        if self.lineEdit_steps_per_epoch.text():
            cmd = "{} --steps_per_epoch {}".format(cmd, self.lineEdit_steps_per_epoch.text())
            
        if self.lineEdit_noise_level.text():
            cmd = "{} --noise_level {}".format(cmd, self.lineEdit_noise_level.text())
        if self.lineEdit_noise_start_iter.text():
            cmd = "{} --noise_start_iter {}".format(cmd, self.lineEdit_noise_start_iter.text())
        if self.lineEdit_noise_mode.text():
            cmd = "{} --noise_mode {}".format(cmd, self.lineEdit_noise_mode.text())
        #if self.lineEdit_noise_pause.text():
        #    cmd = "{} --noise_pause {}".format(cmd, self.lineEdit_noise_pause.text())
        
        if self.lineEdit_drop_out.text():
            cmd = "{} --drop_out {}".format(cmd, self.lineEdit_drop_out.text())
        if self.lineEdit_network_depth.text():
            cmd = "{} --unet_depth {}".format(cmd, self.lineEdit_network_depth.text())
        if self.lineEdit_convs_per_depth.text():
            cmd = "{} --convs_per_depth {}".format(cmd, self.lineEdit_convs_per_depth.text())
        if self.lineEdit_kernel.text():
            cmd = "{} --kernel {}".format(cmd, self.lineEdit_kernel.text())
        if self.lineEdit_filter_base.text():
            cmd = "{} --filter_base {}".format(cmd, self.lineEdit_filter_base.text())
        if self.lineEdit_pool.text():
            cmd = "{} --pool {}".format(cmd, self.lineEdit_pool.text()) 
            
        if not self.checkBox_batch_normalization.isChecked():
            cmd = "{} --batch_normalization {}".format(cmd, False)    
        if not self.checkBox_normalization_percentile.isChecked():
            cmd = "{} --normalize_percentile {}".format(cmd, False)
            
            
        if self.checkBox_only_print_command_refine.isChecked():
            print(cmd)
        else:
            self.start_process(cmd,self.pushButton_refine)
            # os.system(cmd)
            #self.create_subprocess(cmd)
            #Popen(cmd,shell=True,stdin=None,stdout=None, stderr=None)
            
    def predict( self ):
        tomo_star = self.lineEdit_tomo_star_predict.text() if self.lineEdit_tomo_star_predict.text() else "tomograms.star"
        gpuID = self.lineEdit_gpuID_predict.text() if self.lineEdit_gpuID_predict.text() else '0,1,2,3'
        cmd = "isonet.py predict {} --gpuID {}".format(tomo_star,gpuID)
        
        if self.lineEdit_pretrain_model_predict.text() and self.model.isValid(self.lineEdit_pretrain_model_predict.text()):
            cmd = "{} {}".format(cmd, self.lineEdit_pretrain_model_predict.text())
        else:
            print("no trained model detected")
            return
        # if self.lineEdit_gpuID_predict.text():
        #     cmd = "{} --gpuID {}".format(cmd, self.lineEdit_gpuID_predict.text())
        
        if self.lineEdit_tomo_index_predict.text():
            cmd = "{} --tomo_idx {}".format(cmd, self.lineEdit_tomo_index_predict.text())

        if self.lineEdit_result_dir_predict.text():
            cmd = "{} --output_dir {}".format(cmd, self.lineEdit_result_dir_predict.text())
        
        if self.lineEdit_cube_size_predict.text():
            cmd = "{} --cube_size {}".format(cmd, self.lineEdit_cube_size_predict.text())

        if self.lineEdit_crop_size_predict.text():
            cmd = "{} --crop_size {}".format(cmd, self.lineEdit_crop_size_predict.text())
            
                    
        if self.checkBox_only_print_command_predict.isChecked():
            print(cmd)
        else:
            self.start_process(cmd,self.pushButton_predict)
            #self.create_subprocess(cmd)
            #os.system(cmd)
            
    def view_predict_3dmod(self):
        try:
            result_dir_predict = self.lineEdit_result_dir_predict.text()
            list_file = os.listdir(result_dir_predict)
            for f in list_file:
                if f[-4:] == ".mrc" or f[-4:] == ".rec":                   
                    cmd = "3dmod {}/{}".format(result_dir_predict,f)
                    os.system(cmd)
        except:
            pass
    
    def open_star( self ):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "Choose File",
                        "",
                        "Star file (*.star)",
                        options=options)
        if fileName:
            try:
                tomo_file = self.model.sim_path(self.model.pwd, fileName)
                self.model.read_star_gui(tomo_file)
                setTableWidget(self.tableWidget, self.model.md)
            except:
                print("warning")
                pass
    def openGithub(self):
        import webbrowser
        webbrowser.open(self.model.github_addr)
        
    def create_subprocess (self, cmd):
        process_ping = Popen(cmd,shell=True,stdin=None,stdout=None, stderr=None)
        pid = process_ping.pid
        os.system("echo {} >> {}".format(pid,self.model.pid_file))
        
     
class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.p = None

    def closeEvent(self, event):
        if self.p:
        
            result = QtWidgets.QMessageBox.question(self,
                          "Confirm Exit...",
                          "Do you want to continue the existing job in the background?",
                          QtWidgets.QMessageBox.Yes| QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
            event.ignore()
            if result == QtWidgets.QMessageBox.Yes:
                event.accept()
            if result == QtWidgets.QMessageBox.No:
                self.p.kill()
                event.accept()
                #kill the old process
        else:
            result = QtWidgets.QMessageBox.question(self,
                          "Confirm Exit...",
                          "Do you want to exit? ",
                          QtWidgets.QMessageBox.Yes| QtWidgets.QMessageBox.No )
            event.ignore()
            if result == QtWidgets.QMessageBox.Yes:
                event.accept()
            if result == QtWidgets.QMessageBox.No:
                pass
                #kill the old process

     
def main():
    """
    This is the MAIN ENTRY POINT of our application.  The code at the end
    of the mainwindow.py script will not be executed, since this script is now
    our main program.   We have simply copied the code from mainwindow.py here
    since it was automatically generated by '''pyuic5'''.
    """
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MyWindow()
    #MainWindow = QtWidgets.QMainWindow()
    ui = MainWindowUIClass()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

main()
