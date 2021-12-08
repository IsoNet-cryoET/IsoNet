# model.py
# D. Thiebaut
# This is the model part of the Model-View-Controller
# The class holds the name of a text file and its contents.
# Both the name and the contents can be modified in the GUI
# and updated through methods of this model.
# 
import os
from IsoNet.util.metadata import MetaData,Label,Item
from PyQt5.QtWidgets import QHeaderView,QTableWidgetItem


def setTableWidget(tw, md):
        nRows = len(md)

        labels = md.getLabels()
        nColumns = len(labels)

        tw.setColumnCount(nColumns- 1 ) 
        tw.setRowCount(nRows)

        label_2 = [label[3:] for label in labels]
        for i,lab in enumerate(label_2):
            if lab == 'Defocus' or lab == 'PixelSize':
                label_2[i] =  lab+" (A)"

        tw.setHorizontalHeaderLabels(label_2[1:])
        tw.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        tw.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        # data insertion
        for i, it in enumerate(md):
            for j in range(tw.columnCount()):
                tw.setItem(i, j, QTableWidgetItem(str(getattr(it,labels[j+1]))))


class Model:
    def __init__(self):
        '''
        Initializes the two members the class holds:
        the file name and its contents.
        '''
        self.tomogram_star = "tomograms.star"
        self.github_addr = "https://github.com/Heng-Z/IsoNet"
        self.pid_file = "pid.txt"
        #self.commands2run = []
        self.read_star()
        self.pwd = os.getcwd().replace("\\","/")
        self.log_file = "log.txt"
        self.btn_pressed_text = None


    def read_star(self):
        if not self.isValid(self.tomogram_star):
            
            self.md = MetaData()
            #self.md.addLabels('rlnIndex','rlnMicrographName','rlnPixelSize','rlnDefocus','rlnNumberSubtomo')
            self.md.addLabels('rlnIndex','rlnMicrographName','rlnPixelSize','rlnDefocus','rlnNumberSubtomo','rlnSnrFalloff','rlnDeconvStrength','rlnDeconvTomoName','rlnMaskBoundary','rlnMaskDensityPercentage','rlnMaskStdPercentage','rlnMaskName')
            self.md.write(self.tomogram_star)

        else:
            self.md = MetaData()
            self.md.read(self.tomogram_star)

        self.header = self.md.getLabels()

    def read_star_gui(self,star_file):

        if self.isValid(star_file):
            md_cad = MetaData()
            md_cad.read(star_file)
            if "rlnMicrographName" not in md_cad.getLabels():
                return 1
            else:
                self.tomogram_star = star_file
                self.md = MetaData()
                self.md.read(self.tomogram_star)
                self.header = self.md.getLabels()
            return 0

    def isValid(self, fileName):
        '''
        returns True if the file exists and can be
        opened.  Returns False otherwise.
        '''
        try:
            file = open(fileName, 'r')
            file.close()
            return True
        except:
            return False

    def isValidPath(self, path):
        '''
        returns True if the file exists and can be
        opened.  Returns False otherwise.
        '''
        try:
            isDir = os.path.isdir(path)
            return isDir
        except:
            return False

    def is_number(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    def is_file_exist(self, path, suffix):
        fileList = []
        for fname in os.listdir(path):
            if fname.endswith(suffix):
                # do stuff on the file
                fileList.append(fname)
        return fileList


    def sim_path(self, pwd, path):
        if pwd in path:
            return "." + path[len(pwd):]
        else:
            return path
            
    def getLogContent( self, fileName ):
        '''
        sets the member fileName to the value of the argument
        if the file exists.  Otherwise resets both the filename
        and file contents members.
        '''
        if self.isValid( fileName ):
            self.fileName = fileName
            content = open( fileName, 'r' ).read()
            return content
        else:
            return None
