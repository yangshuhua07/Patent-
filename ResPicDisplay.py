# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ResPicDisplay.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import SysBackground

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(590, 340)
        Form.setStyleSheet("background-image: url(:/bkg/bkg5.jpg);")

        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(0, 0, 290, 290))
        self.label.setText("")
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(300, 0, 290, 290))
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(145, 300, 21, 31))
        self.label_3.setStyleSheet("font: 14pt \"Agency FB\";")
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(445, 300, 71, 31))
        self.label_4.setStyleSheet("font: 14pt \"Agency FB\";")
        self.label_4.setObjectName("label_4")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "目标检测结果"))
        self.label.setText(_translate("Form", "GT Display"))
        self.label_2.setText(_translate("Form", "Proposed Display"))
        self.label_3.setText(_translate("Form", "GT"))
        self.label_4.setText(_translate("Form", "Propesed"))

