# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'EvaPicDisplay.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import SysBackground


class Ui_Form2(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(800, 640)
        Form.setStyleSheet("background-image: url(:/bkg/bkg5.jpg);")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(80, 60, 641, 481))
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(360, 580, 91, 31))
        self.label_2.setStyleSheet("font: 14pt \"Agency FB\";")
        self.label_2.setObjectName("label_2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "评估结果"))
        self.label.setText(_translate("Form", "ROC display"))
        self.label_2.setText(_translate("Form", "ROC曲线"))
