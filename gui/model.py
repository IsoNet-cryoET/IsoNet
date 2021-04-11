# model.py
# D. Thiebaut
# This is the model part of the Model-View-Controller
# The class holds the name of a text file and its contents.
# Both the name and the contents can be modified in the GUI
# and updated through methods of this model.
# 
import os


class Model:
    def __init__(self):
        '''
        Initializes the two members the class holds:
        the file name and its contents.
        '''
        self.fileName = None
        self.folderName = None
        self.pwd = None
        self.currentLines = None
        self.fileContent = ""
        self.commands2run = []


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
    def setPwd(self):
        '''
        Returns the name of the file name member.
        '''
        self.pwd = os.getcwd()

    def setFileName(self, fileName):
        '''
        sets the member fileName to the value of the argument
        if the file exists.  Otherwise resets both the filename
        and file contents members.
        '''
        if self.isValid(fileName):
            self.fileName = fileName
            self.fileContents = open(fileName, 'r').read()
        else:
            self.fileContents = ""
            self.fileName = ""

    def setCurrentLines(self, currentLines):
        '''
        sets the member fileName to the value of the argument
        if the file exists.  Otherwise resets both the filename
        and file contents members.
        '''
        self.currentLines = currentLines

    def setFolderName(self, folderName):
        '''
        sets the member fileName to the value of the argument
        if the file exists.  Otherwise resets both the filename
        and file contents members.
        '''
        if self.isValidPath(folderName):
            self.folderName = folderName
        else:
            self.folderName = ""

    def getPwd(self):
        '''
        Returns the name of the file name member.
        '''
        return self.pwd

    def getFileName(self):
        '''
        Returns the name of the file name member.
        '''
        return self.fileName

    def getFolderName(self):
        '''
        Returns the name of the file name member.
        '''
        return self.folderName

    def getCurrentLines(self):
        '''
        Returns the name of the file name member.
        '''
        return self.currentLines

    def getFileContents(self):
        '''
        Returns the contents of the file if it exists, otherwise
        returns an empty string.
        '''
        return self.fileContents

    def getFileContents(self, fileName):
        '''
        Returns the contents of the file if it exists, otherwise
        returns an empty string.
        '''
        if self.isValid(fileName):
            return open(fileName, 'r').read()
        else:
            return ""

    def writeDoc(self, text):
        '''
        Writes the string that is passed as argument to a
        a text file with name equal to the name of the file
        that was read, plus the suffix ".bak"
        '''
        if self.isValid(self.fileName):
            fileName = self.fileName + ".bak"
            file = open(fileName, 'w')
            file.write(text)
            file.close()


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

    def paraChecksMask(self, tomo_dir, mask_dir, percentile, threshold):
        error_message = ""
        if tomo_dir:
            if not self.isValidPath(tomo_dir):
                error_message += "tomo directory does not exist! \n"
            else:
                fileList = self.is_file_exist(tomo_dir, '.mrc')
                if len(fileList) == 0:
                   error_message += "no mrc file exists in tomo directory \n" 
        else:
            error_message += "tomo directory is not provided! \n"

        if mask_dir:
            if not self.isValidPath(mask_dir):
                error_message += "mask directory does not exist! \n"
        else:
            error_message += "mask directory is not provided! \n"

        if (not self.is_number(percentile)) or (float(percentile) < 0 or float(percentile) > 100):
            error_message += "percentile should be a number between 0 and 100! \n"

        if (not self.is_number(threshold)) or (float(threshold) < 0 or float(threshold) > 1):
            error_message += "threshold should be a number between 0 and 1! \n"

        return error_message

    def paraChecksDeconvolve(self, tomo_dir, angpix, defocus):
        fileList = []
        error_message = ""
        if tomo_dir:
            if not self.isValidPath(tomo_dir):
                error_message += "tomo directory does not exist! \n"
            else:
                fileList = self.is_file_exist(tomo_dir, '.mrc')
                if len(fileList) == 0:
                   error_message += "no mrc file exists in tomo directory \n" 
        else:
            error_message += "tomo directory is not provided! \n"


        if (not self.is_number(angpix)) or (float(angpix) < 0 ):
            error_message += "pixel size should be a number larger than 0\n"

        if not self.is_number(defocus):
            error_message += "defoucs should be a number\n"
        return [error_message,fileList]

    def paraChecksRefine(self, tomo_dir, mask_dir, pretrain_model,
        iteration, epochs, steps_per_epoch, ncube, 
        noise_level,noise_start_iter, noise_pause, batch_size, gpuID):

        error_message = ""
        if tomo_dir:
            if not self.isValidPath(tomo_dir):
                error_message += "tomo directory does not exist! \n"
            else:
                fileList = self.is_file_exist(tomo_dir, '.mrc')
                if len(fileList) == 0:
                   error_message += "no mrc file exists in tomo directory \n" 
        else:
            error_message += "tomo directory is not provided! \n"

        if mask_dir:
            if not self.isValidPath(mask_dir):
                error_message += "mask directory does not exist! \n"
        else:
            error_message += "mask directory is not provided! \n"

        if pretrain_model and (not self.isValid(pretrain_model)):
            error_message += "pretrain_model does not exist! \n"

        if (not iteration.isdigit()) or (float(iteration) < 0 ):
            error_message += "iteration should be a integer larger than 0\n"

        if (not epochs.isdigit()) or (float(epochs) < 0 ):
            error_message += "epochs should be a integer larger than 0\n"

        if (not steps_per_epoch.isdigit()) or (float(steps_per_epoch) < 0 ):
            error_message += "steps_per_epoch should be a integer larger than 0\n"

        if (not ncube.isdigit()) or (float(ncube) < 0 ):
            error_message += "subtomograms number should be a integer larger than 0\n"

        if (not self.is_number(noise_level)) or (float(noise_level) < 0 ):
            error_message += "noise_level should be a number larger than 0\n"

        if (not noise_start_iter.isdigit()) or (float(noise_start_iter) < 0 ):
            error_message += "noise_start_iter should be a integer larger than 0\n"

        if (not noise_pause.isdigit()) or (float(noise_pause) < 0 ):
            error_message += "noise_pause should be a integer larger than 0\n"

        if gpuID:
            ngpu = len(gpuID.split(","))

        if (not batch_size.isdigit()) or (float(batch_size) < 0 ):
            error_message += "batch_size should be a integer larger than 0\n"
        else:
            if gpuID and ngpu > 0:
                if int(batch_size) % ngpu !=0:
                    error_message += "batch_size should be a integer equals to ngpu*n \n"

        return error_message

    def paraChecksPredict(self, tomo_dir, output_dir, refined_model, gpuID):
        fileList = []
        error_message = ""
        if tomo_dir:
            if not self.isValidPath(tomo_dir):
                error_message += "tomo directory does not exist! \n"
            else:
                fileList = self.is_file_exist(tomo_dir, '.mrc')
                if len(fileList) == 0:
                   error_message += "no mrc file exists in tomo directory \n" 
        else:
            error_message += "tomo directory is not provided! \n"

        if output_dir:
            if not self.isValidPath(output_dir):
                error_message += "output directory does not exist! \n"
        else:
            error_message += "output directory is not provided! \n"

        if refined_model:
            if not self.isValid(refined_model):
                error_message += "refined model does not exist! \n"
        else:
            error_message += "refined model is not provided! \n"
        return [error_message,fileList]

    def sim_path(self, pwd, path):

        if pwd in path:
            return "." + path[len(pwd):]
        else:
            return path