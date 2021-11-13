

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSlot
from IsoNet.gui.mainwindow_v3 import Ui_MainWindow
import sys
import os
from IsoNet.gui.model import Model
import time
from threading import Thread

class MainWindowUIClass( Ui_MainWindow ):
    def __init__( self ):
        '''Initialize the super class
        '''
        super().__init__()
        self.model = Model()
        
    def setupUi( self, MW ):
        ''' Setup the UI of the super class, and add here code
        that relates to the way we want our UI to operate.
        '''
        super().setupUi( MW )
        self.logWindow.setFontPointSize(12)
        self.model.processing = False
        if self.model.isValid("log.txt"):
            #qcolor = QtGui.QColor("red")
            #self.logWindow.setTextColor(qcolor)
            self.logWindow.setText( self.model.getFileContents("log.txt") )
        # close the lower part of the splitter to hide the 
        # debug window under normal operations
        #self.splitter.setSizes([300, 0])
        isonet_path = os.popen("which isonet.py").read()
        tmp = isonet_path.split("bin/isonet.py")
        root_path = tmp[0]

        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(root_path+"gui/icons/icon_folder.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button_tomo_dir.setIcon(icon)
        self.button_mask_dir.setIcon(icon)
        self.button_pretrain_model.setIcon(icon)
        self.button_output.setIcon(icon)
        self.button_refined_model.setIcon(icon)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(root_path+"gui/icons/icon_advanced.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button_advance_mask.setIcon(icon1)
        self.button_advance_refine.setIcon(icon1)
        self.button_advance_predict.setIcon(icon1)
    def refreshAll( self ):
        '''
        Updates the widgets whenever an interaction happens.
        Typically some interaction takes place, the UI responds,
        and informs the model of the change.  Then this method
        is called, pulling from the model information that is
        updated in the GUI.
        '''
        self.lineEdit_mask_dir.setText( self.model.getFolderName() )
        print(QtGui.QTextCursor.END)
        
        self.logWindow.setText( self.model.getFileContents("log.txt") )

        #self.lineEdit.setText( self.model.getFileName() )
        #self.textEdit.setText( self.model.getFileContents() )
         
    # slot
    def browseSlot( self , btn ):
        ''' Called when the user presses the Browse button
        '''
        lineEdit = self.switch_btn(btn)

        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "Choose File",
                        "",
                        "All Files (*)",
                        options=options)
        if fileName:
            #self.model.setFileName( fileName )
            #######
            #cmd = "echo choose file: {} >> log.txt ".format(fileName)
            #os.system(cmd)
            #self.logWindow.append("choose file: {}".format(fileName) )
            lineEdit.setText( fileName )
            #self.logWindow.moveCursor(QtGui.QTextCursor.End) 
            #######
            #self.refreshAll()
        #self.debugPrint( "Browse button pressed" )

    def browseFolderSlot( self , btn):
        ''' Called when the user presses the Browse folder button
        '''
        lineEdit = self.switch_btn(btn)
        try:
            dir_path=QtWidgets.QFileDialog.getExistingDirectory(None,"Choose Directory",self.model.getPwd())
            #self.model.setFolderName( dir_path )
            #cmd = "echo choose folder: {} >> log.txt ".format(dir_path)
            #os.system(cmd)
            #self.logWindow.append("choose folder: {}".format(dir_path) )
            lineEdit.setText( dir_path )
            #self.logWindow.moveCursor(QtGui.QTextCursor.End) 
            #self.refreshAll()
        except:
            ##TODO: record to log.
            pass


    def advancedMenu( self , btn ):
        ''' Called when the user presses the Browse button
        '''
        widget = self.switch_btn(btn)
        if widget.isVisible():
            widget.setVisible(False)
        else:
            widget.setVisible(True)

    def switch_btn(self, btn):
        switcher = {
            "mask_dir": self.lineEdit_mask_dir,
            "tomo_dir": self.lineEdit_tomo_dir,
            "pretrain_model":self.lineEdit_pretrain_model,
            "output": self.lineEdit_output,
            "refined_model":self.lineEdit_refined_model,
            "advance_mask":self.widget_mask,
            "advance_refine":self.widget_refine,
            "advance_predict":self.widget_predict
        }
        return switcher.get(btn, "Invaid btn name")




    def printCmds( self ):
        
        pwd = os.getcwd().replace("\\","/")

        cmd2run = []
        if self.checkBox_mask.isChecked():

            tomo_dir = self.lineEdit_tomo_dir.text()
            mask_dir = self.lineEdit_mask_dir.text()
            percentile = self.lineEdit_percentile.text() if self.lineEdit_percentile.text() else 100
            threshold = self.lineEdit_threshold.text() if self.lineEdit_threshold.text() else 0.5

            error_msg = self.model.paraChecksMask(tomo_dir, mask_dir, percentile, threshold)
            if error_msg != "":
                error_msg = "##########    Error!    ##########\n" + error_msg
                #cmd = "echo \"{}\" >> log.txt".format(error_msg)
                #print(cmd)
                #os.system(cmd)
                self.logWindow.append(error_msg)
                self.logWindow.moveCursor(QtGui.QTextCursor.End)
                return None
            else:
                tomo_dir = self.model.sim_path(pwd, tomo_dir)
                mask_dir = self.model.sim_path(pwd, mask_dir)
                description = "run following command(s) to make mask:"
                #cmd = "echo {} >> log.txt".format( description )
                #os.system(cmd)
                self.logWindow.append(description)
                line = "isonet.py make_mask {} {}  --percentile {}  --threshold {} ".format( tomo_dir,mask_dir,percentile,threshold )
                #cmd = "echo {} >> log.txt".format( line )
                #os.system(cmd)
                
                line += "\n"
                self.logWindow.append(line)
                cmd2run.append(line)
                self.logWindow.moveCursor(QtGui.QTextCursor.End)


        if self.checkBox_deconvolve.isChecked():
            tomo_dir = self.lineEdit_tomo_dir.text()
            angpix = self.lineEdit_angpix.text() 
            defocus = self.lineEdit_defocus.text()

            error_msg,fileList = self.model.paraChecksDeconvolve(tomo_dir, angpix, defocus)

            if error_msg != "":
                error_msg = "##########    Error!    ##########\n" + error_msg
                #cmd = "echo \"{}\" >> log.txt".format(error_msg)
                #print(cmd)
                #os.system(cmd)
                self.logWindow.append(error_msg)
                self.logWindow.moveCursor(QtGui.QTextCursor.End)
                return None
            elif len(fileList) > 0:
                tomo_dir = self.model.sim_path(pwd, tomo_dir)

                tomo_dir_deconv = tomo_dir + "_deconv"
                description = "run following command(s) to deconvolve:"
                #cmd = "echo {} >> log.txt".format( description )
                #os.system(cmd)
                self.logWindow.append(description)
                if not self.model.isValidPath(tomo_dir_deconv):
                    os.mkdir(tomo_dir_deconv)
                for file in fileList:
                    basename = os.path.basename(file) 
                    
                    line = "python deconvolve.py {} {} {} {} ".format( tomo_dir+"/"+file, tomo_dir_deconv+"/"+basename, angpix, defocus)
                    #cmd = "echo {} >> log.txt".format( line )
                    #os.system(cmd)
                    
                    #line += "\n"
                    self.logWindow.append(line)
                    cmd2run.append(line)
                    self.logWindow.moveCursor(QtGui.QTextCursor.End)      
                line = "\n"
                self.logWindow.append(line)
        if self.checkBox_train.isChecked():
            tomo_dir = self.lineEdit_tomo_dir.text()
            mask_dir = self.lineEdit_mask_dir.text()
            iteration = self.lineEdit_iteration.text() if self.lineEdit_iteration.text() else '30'
            epochs = self.lineEdit_epochs.text() if self.lineEdit_epochs.text() else '8'
            steps_per_epoch = self.lineEdit_steps_per_epoch.text() if self.lineEdit_steps_per_epoch.text() else '200'
            ncube = self.lineEdit_ncube.text() if self.lineEdit_ncube.text() else '300'
            noise_level = self.lineEdit_noise_level.text() if self.lineEdit_noise_level.text() else '0.1'
            noise_start_iter = self.lineEdit_noise_start_iteration.text() if self.lineEdit_noise_start_iteration.text() else '15'
            noise_pause = self.lineEdit_noise_pause.text() if self.lineEdit_noise_pause.text() else '3'
            batch_size = self.lineEdit_batch_size.text() if self.lineEdit_batch_size.text() else '8'
            gpuID = self.lineEdit_gpu.text() if self.lineEdit_gpu.text() else '0,1,2,3'

            pretrain_model = self.lineEdit_pretrain_model.text()

            error_msg = self.model.paraChecksRefine( tomo_dir, mask_dir, pretrain_model,
                iteration, epochs, steps_per_epoch, ncube, 
                noise_level,noise_start_iter, noise_pause, batch_size, gpuID)

            if error_msg != "":
                error_msg = "##########    Error!    ##########\n" + error_msg
                #cmd = "echo \"{}\" >> log.txt".format(error_msg)
                #print(cmd)
                #os.system(cmd)
                self.logWindow.append(error_msg)
                self.logWindow.moveCursor(QtGui.QTextCursor.End)
                return None
            #with pretrain model
            else :
                tomo_dir = self.model.sim_path(pwd, tomo_dir)
                mask_dir = self.model.sim_path(pwd, mask_dir)
                if pretrain_model:
                    pretrain_model = self.model.sim_path(pwd, pretrain_model)
                
                    line = "isonet.py refine --input_dir {} --mask_dir {} --pretrain_model {} --iterations {} --steps_per_epoch {} --ncube {} --noise_level {} --noise_start_iter {} --noise_pause {} --epochs {} --batch_size {} --gpuID {}".format( 
                        tomo_dir, mask_dir, pretrain_model, iteration,steps_per_epoch,ncube,noise_level,noise_start_iter,noise_pause,epochs,batch_size,gpuID)
                    
                else:
                    line = "isonet.py refine --input_dir {} --mask_dir {}  --iterations {} --steps_per_epoch {} --ncube {} --noise_level {} --noise_start_iter {} --noise_pause {} --epochs {} --batch_size {} --gpuID {}".format( 
                        tomo_dir, mask_dir, iteration,steps_per_epoch,ncube,noise_level,noise_start_iter,noise_pause,epochs,batch_size,gpuID)
                
                description = "run following command(s) to refine:"
                #cmd = "echo {} >> log.txt".format( description )
                #os.system(cmd)
                self.logWindow.append(description)
                #cmd = "echo {} >> log.txt".format( line )
                #os.system(cmd)
                
                #line += "\n"
                self.logWindow.append(line)
                cmd2run.append(line)
                self.logWindow.moveCursor(QtGui.QTextCursor.End) 


        if self.checkBox_predict.isChecked():
            tomo_dir = self.lineEdit_tomo_dir.text()
            output_dir = self.lineEdit_output.text()
            refined_model = self.lineEdit_refined_model.text()
            gpuID = self.lineEdit_gpu.text() if self.lineEdit_gpu.text() else '0,1,2,3'

            error_msg,fileList = self.model.paraChecksPredict(tomo_dir, output_dir, refined_model, gpuID)

            if error_msg != "":
                error_msg = "##########    Error!    ##########\n" + error_msg
                #cmd = "echo \"{}\" >> log.txt".format(error_msg)
                #print(cmd)
                #os.system(cmd)
                self.logWindow.append(error_msg)
                self.logWindow.moveCursor(QtGui.QTextCursor.End)
                return None
            elif len(fileList) > 0:
                tomo_dir = self.model.sim_path(pwd, tomo_dir)
                output_dir = self.model.sim_path(pwd, output_dir)
                refined_model = self.model.sim_path(pwd, refined_model)
                description = "run following command(s) to predict:"
                #cmd = "echo {} >> log.txt".format( description )
                #os.system(cmd)
                self.logWindow.append(description)
                for file in fileList:
                    basename = file[:-4]
                    output_file = basename + "_pred.mrc"

                    line = "isonet.py predict {} {} {} --gpuID {} ".format( tomo_dir+"/"+file, output_dir+"/"+output_file, refined_model, gpuID)
                    #cmd = "echo {} >> log.txt".format( line )
                    #os.system(cmd)
                    
                    line += "\n"
                    self.logWindow.append(line)
                    cmd2run.append(line)
                    self.logWindow.moveCursor(QtGui.QTextCursor.End)  

        return cmd2run

    #TODO: add a function to update the log window
    def runProgram(self):
        cmd2run = self.printCmds()
        #print(cmd2run)
        #self.model.processing = True
        #t = Thread(target=self.update_log)
        #t.start()
        #self.update_log()
        #self.model.processing = False
        #t.terminate() 
        #import subprocess

        #text = os.popen("ls -l").read()
        #command = ['ls', '-l']
        #p = subprocess.Popen(command, stdout=subprocess.PIPE)
        #text = p.stdout.read()
        #retcode = p.wait()
        #print(str(text))
        #file_log = open('log.txt', 'a')
        #file_log.write(str(text))
        #file_log.close()
        #self.update_log()


        for line in cmd2run:
            cmd = "{} ".format( line)
            print(cmd)
            os.system(cmd)

        
    def showResults( self ):
        output_dir = self.lineEdit_output.text()
        
        error_message = ""
        if output_dir:
            if not self.model.isValidPath(output_dir):
                error_message += "output directory does not exist! \n"
            else:
                fileList = self.model.is_file_exist(output_dir, '.mrc')
                if len(fileList) == 0:
                   error_message += "no mrc file exists in output directory \n" 
        else:
            error_message += "output directory is not provided! \n"

        if error_message == "":
            cmd = "3dmod {}/*mrc".format(output_dir)
            os.system(cmd)
        else:
            self.logWindow.append(error_message)
            self.logWindow.moveCursor(QtGui.QTextCursor.End) 
    '''
    def update_log(self):
        f = open("log.txt", 'r')
        where = f.tell()
        line = f.readline()
        while line:
            self.logWindow.append(line)
            self.logWindow.moveCursor(QtGui.QTextCursor.End) 
            where+=1
        while self.model.processing:
            where = f.tell()
            line = f.readline()
            if not line:
                time.sleep(1)
                f.seek(where)
            else:
                self.logWindow.append(line)
                self.logWindow.moveCursor(QtGui.QTextCursor.End) 
    '''
def main():
    """
    This is the MAIN ENTRY POINT of our application.  The code at the end
    of the mainwindow.py script will not be executed, since this script is now
    our main program.   We have simply copied the code from mainwindow.py here
    since it was automatically generated by '''pyuic5'''.

    """
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MainWindowUIClass()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

main()