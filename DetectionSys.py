# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'classificationSys.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import SysBackground


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(977, 606)
        MainWindow.setStyleSheet("border-image: url(:/bkg/bkg2.jpg);")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(780, 380, 131, 71))

        font = QtGui.QFont()
        font.setFamily("方正姚体")
        font.setPointSize(18)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)

        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("font: 18pt \"方正姚体\";\n"
"border-image: url(:/bkg/bkg5.jpg);\n"
"\n"
"color: rgb(53, 75, 118);\n"
"\n"
"")
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(780, 490, 131, 71))

        font = QtGui.QFont()
        font.setFamily("方正姚体")
        font.setPointSize(18)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)

        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("font: 18pt \"方正姚体\";\n"
"border-image: url(:/bkg/bkg5.jpg);\n"
"color: rgb(53, 75, 118);")
        self.pushButton_2.setObjectName("pushButton_2")

        MainWindow.setCentralWidget(self.centralwidget)
        self.actionOpen_File = QtWidgets.QAction(MainWindow)
        self.actionOpen_File.setObjectName("actionOpen_File")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "高光谱目标检测系统"))
        self.pushButton.setText(_translate("MainWindow", "目标检测"))
        self.pushButton_2.setText(_translate("MainWindow", "退出系统"))
        self.actionOpen_File.setText(_translate("MainWindow", "Open File"))
