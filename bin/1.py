import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

MAC = True
try:
    from PyQt4.QtGui import qt_mac_set_native_menubar
except ImportError:
    MAC = False


class FindAndReplaceDlg(QDialog):

    def __init__(self, parent=None):
        super(FindAndReplaceDlg, self).__init__(parent)

        findLabel = QLabel("Find &what:")
        self.findLineEdit = QLineEdit()
        findLabel.setBuddy(self.findLineEdit)
        replaceLabel = QLabel("Replace w&ith:")
        self.replaceLineEdit = QLineEdit()
        replaceLabel.setBuddy(self.replaceLineEdit)
        self.caseCheckBox = QCheckBox("&Case sensitive")
        self.wholeCheckBox = QCheckBox("Wh&ole words")
        self.wholeCheckBox.setChecked(True)
        moreFrame = QFrame()
        moreFrame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.backwardsCheckBox = QCheckBox("Search &Backwards")
        self.regexCheckBox = QCheckBox("Regular E&xpression")
        self.ignoreNotesCheckBox = QCheckBox("Ignore foot&notes "
                                             "and endnotes")
        line = QFrame()
        line.setFrameStyle(QFrame.VLine | QFrame.Sunken)
        self.findButton = QPushButton("&Find")
        self.replaceButton = QPushButton("&Replace")
        closeButton = QPushButton("Close")
        moreButton = QPushButton("&More")
        moreButton.setCheckable(True)
        if not MAC:
            self.findButton.setFocusPolicy(Qt.NoFocus)
            self.replaceButton.setFocusPolicy(Qt.NoFocus)
            closeButton.setFocusPolicy(Qt.NoFocus)
            moreButton.setFocusPolicy(Qt.NoFocus)

        gridLayout = QGridLayout()
        gridLayout.addWidget(findLabel, 0, 0)
        gridLayout.addWidget(self.findLineEdit, 0, 1)
        gridLayout.addWidget(replaceLabel, 1, 0)
        gridLayout.addWidget(self.replaceLineEdit, 1, 1)
        frameLayout = QVBoxLayout()
        frameLayout.addWidget(self.backwardsCheckBox)
        frameLayout.addWidget(self.regexCheckBox)
        frameLayout.addWidget(self.ignoreNotesCheckBox)
        moreFrame.setLayout(frameLayout)
        leftLayout = QVBoxLayout()
        leftLayout.addLayout(gridLayout)
        leftLayout.addWidget(self.caseCheckBox)
        leftLayout.addWidget(self.wholeCheckBox)
        leftLayout.addWidget(moreFrame)
        buttonLayout = QVBoxLayout()
        buttonLayout.addWidget(self.findButton)
        buttonLayout.addWidget(self.replaceButton)
        buttonLayout.addWidget(closeButton)
        buttonLayout.addWidget(moreButton)
        buttonLayout.addStretch()
        mainLayout = QHBoxLayout()
        mainLayout.addLayout(leftLayout)
        mainLayout.addWidget(line)
        mainLayout.addLayout(buttonLayout)
        self.setLayout(mainLayout)

        moreFrame.hide()
        mainLayout.setSizeConstraint(QLayout.SetFixedSize)

        moreButton.toggled.connect(moreFrame.setVisible)
        self.findLineEdit.textEdited.connect(self.updateUi)
        self.findButton.clicked.connect(self.findClicked)
        self.replaceButton.clicked.connect(self.replaceClicked)

        self.updateUi()
        self.setWindowTitle("Find and Replace")

    find = pyqtSignal(str, tuple)
    replace = pyqtSignal(str, str, tuple)

    def findClicked(self):
        self.find.emit(self.findLineEdit.text(),
                       (self.caseCheckBox.isChecked(),
                        self.wholeCheckBox.isChecked(),
                        self.backwardsCheckBox.isChecked(),
                        self.regexCheckBox.isChecked(),
                        self.ignoreNotesCheckBox.isChecked()))

    def replaceClicked(self):
        self.replace.emit(self.findLineEdit.text(),
                          self.replaceLineEdit.text(),
                          (self.caseCheckBox.isChecked(),
                           self.wholeCheckBox.isChecked(),
                           self.backwardsCheckBox.isChecked(),
                           self.regexCheckBox.isChecked(),
                           self.ignoreNotesCheckBox.isChecked()))

    def updateUi(self):
        enable = bool(self.findLineEdit.text())
        self.findButton.setEnabled(enable)
        self.replaceButton.setEnabled(enable)


if __name__ == "__main__":
    def find(what, *args):
        print("Find {} {}".format(what, [x for x in args]))


    def replace(old, new, *args):
        print("Replace {} with {} {}".format(
            old, new, [x for x in args]))


    app = QApplication(sys.argv)
    form = FindAndReplaceDlg()
    form.find.connect(find)
    form.replace.connect(replace)
    form.show()
    app.exec_()
