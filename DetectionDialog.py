# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ClassificationDialog.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import SysBackground


class Ui_Dialog1(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1015, 454)
        Dialog.setStyleSheet("background-image: url(:/bkg/bkg5.jpg);")

        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(40, 20, 141, 41))
        self.label.setStyleSheet("color: rgb(56, 56, 170);\n"
"font: 12pt \"黑体\";")
        self.label.setObjectName("label")

        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setGeometry(QtCore.QRect(270, 20, 151, 31))
        self.comboBox.setStyleSheet("font: 12pt \"Arial Narrow\";")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")

        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(50, 290, 120, 40))
        self.pushButton.setStyleSheet("background-image: url(:/bkg/bkg3.jpg);\n"
"color: rgb(56, 56, 170);\n"
"font: 12pt \"微软雅黑\";")
        self.pushButton.setObjectName("pushButton")

        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(40, 100, 181, 31))
        self.label_2.setStyleSheet("color: rgb(56, 56, 170);\n"
"font: 12pt \"黑体\";")
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(40, 170, 261, 51))
        self.label_3.setStyleSheet("color: rgb(56, 56, 170);\n"
"font: 12pt \"黑体\";")
        self.label_3.setObjectName("label_3")

        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(270, 100, 151, 31))
        self.lineEdit.setStyleSheet("background-image: url(:/bkg/white.jpg);\n"
"font: 12pt \"Arial Narrow\";")
        self.lineEdit.setObjectName("lineEdit")

        self.lineEdit_2 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_2.setGeometry(QtCore.QRect(270, 180, 151, 31))
        self.lineEdit_2.setStyleSheet("background-image: url(:/bkg/white.jpg);\n"
"font: 12pt \"Arial Narrow\";")
        self.lineEdit_2.setObjectName("lineEdit_2")

        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(520, 20, 471, 411))
        self.textBrowser.setStyleSheet("background-image: url(:/bkg/white.jpg);\n"
"font: 63 11pt \"Bahnschrift SemiBold\";")
        self.textBrowser.setObjectName("textBrowser")

        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(170, 380, 120, 40))
        self.pushButton_2.setStyleSheet("background-image: url(:/bkg/bkg3.jpg);\n"
"color: rgb(56, 56, 170);\n"
"font: 12pt \"微软雅黑\";")
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(290, 290, 120, 40))
        self.pushButton_3.setStyleSheet("background-image: url(:/bkg/bkg3.jpg);\n"
"color: rgb(56, 56, 170);\n"
"font: 12pt \"微软雅黑\";")
        self.pushButton_3.setObjectName("pushButton_3")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "目标检测"))
        self.label.setText(_translate("Dialog", "请选择数据集："))
        self.comboBox.setItemText(0, _translate("Dialog", "Indian Pines"))
        self.comboBox.setItemText(1, _translate("Dialog", "Salinas"))
        self.comboBox.setItemText(2, _translate("Dialog", "Pavia University"))
        self.pushButton.setText(_translate("Dialog", "开始训练"))
        self.label_2.setText(_translate("Dialog", "请设置学习率："))
        self.label_3.setText(_translate("Dialog", "请设置训练样本占比量："))
        self.lineEdit.setPlaceholderText(_translate("Dialog", "0.00013"))
        self.lineEdit_2.setPlaceholderText(_translate("Dialog", "0.15"))
        self.pushButton_2.setText(_translate("Dialog", "评估结果"))
        self.pushButton_3.setText(_translate("Dialog", "查看结果"))
