# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'EvaluateDialog.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import SysBackground


class Ui_Dialog2(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(660, 660)
        Dialog.setStyleSheet("background-image: url(:/bkg/bkg5.jpg);")
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(30, 30, 601, 601))
        self.textBrowser.setStyleSheet("background-image: url(:/bkg/white.jpg);\n"
"font: 11pt \"Agency FB\";")
        self.textBrowser.setObjectName("textBrowser")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "评估结果"))
