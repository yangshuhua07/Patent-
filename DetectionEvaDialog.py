# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DetectionEvaDialog.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import SysBackground


class Ui_Dialog3(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(800, 640)
        Dialog.setStyleSheet("background-image: url(:/bkg/bkg5.jpg);")

        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(80, 60, 641, 481))
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(360, 574, 91, 31))
        self.label_2.setStyleSheet("font: 14pt \"Agency FB\";")
        self.label_2.setObjectName("label_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "评估结果"))
        self.label.setText(_translate("Dialog", "ROC display"))
        self.label_2.setText(_translate("Dialog", "ROC曲线"))
