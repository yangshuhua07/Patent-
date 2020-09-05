from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
from PyQt5.QtGui import *
from DetectionSys import *
from DetectionDialog import *
from EvaluateDialog import *
from ResPicDisplay import *
from EvaPicDisplay import *
import sys
import numpy as np
from torch import optim
import torch
import time
import datetime
import collections
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
import network
import train
import record
from LoadData import aa_and_each_accuracy, sampling, load_dataset, generate_png, generate_iter




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for Monte Carlo runs
seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]
ensemble = 1

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')



global Dataset  # UP,IN,KSC

ITER = 1  # 10
PATCH_LENGTH = 4
num_epochs, batch_size = 2, 16  # 200

INPUT_SIZE = 7
HIDDEN_SIZE = 100
NUM_LAYERS = 2

loss = torch.nn.CrossEntropyLoss()

KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
AUC = []

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s



class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
#        self.setWindowTitle('main window')
        quitbtn = self.pushButton_2
        quitbtn.clicked.connect(self.close)

class EvaluationWindow(QDialog, Ui_Dialog2):
    def __init__(self, parent=None):
        super(EvaluationWindow, self).__init__(parent)
        self.setupUi(self)
#        self.setWindowTitle('main window')

    def printf(self, mes):
        self.textBrowser.append(mes)
        self.cursot = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursot.End)

class PicDisplayWindow(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(PicDisplayWindow, self).__init__()
        self.setupUi(self)
#        self.setWindowTitle('main window')

class EvaDisplayWindow(QMainWindow, Ui_Form2):
    def __init__(self, parent=None):
        super(EvaDisplayWindow, self).__init__()
        self.setupUi(self)





class ClassifyWindow(QDialog, Ui_Dialog1):
    def __init__(self):
        super(ClassifyWindow,self).__init__()
        self.setupUi(self)
        #self.setWindowTitle('child window')
        #self.comboBox.currentIndexChanged.connect(lambda :self.getDateset())
        #self.lineEdit.textEdited.connect(lambda :self.getLR())
        #self.lineEdit_2.textEdited.connect(lambda: self.getTrainRate())
        self.pushButton.clicked.connect(lambda :self.startDetection())
        self.pushButton_2.clicked.connect(lambda :self.startValuation())
        self.pushButton_3.clicked.connect(lambda :self.showResPic())

        self.datanum = self.comboBox.currentText()
        self.learningrate = self.lineEdit.text()
        self.trainRate = self.lineEdit_2.text()
        self.OA = []
        self.KAPPA = []
        self.AA = []
        self.TRAINING_TIME = []
        self.TESTING_TIME = []
        self.AUC = []
        self.picpath_gt = ''
        self.picpath_res = ''

    def getDateset(self):
        self.dataset = self.comboBox.currentText()
        print("dataset = ", self.dataset)
        #return dataset

    def getLR(self):
        self.learningrate = self.lineEdit.text()
        print("lr = ", self.learningrate)

    def getTrainRate(self):
        self.trainRate = self.lineEdit_2.text()
        print("train = ", self.trainRate)

    def printf(self, mes):
        self.textBrowser.append(mes)  # 在指定的区域显示提示信息
        self.cursot = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursot.End)


    def startDetection(self):
        #self.printStart()
        self.datanum = self.comboBox.currentIndex()
        self.learningrate = self.lineEdit.text()
        self.trainRate = self.lineEdit_2.text()

        lr = float(self.learningrate)
        print("lr = ", lr)
        if self.datanum == 0:
            dataset = 'IN'
        elif self.datanum == 1:
            dataset = 'SA'
        elif self.datanum == 2:
            dataset = 'UP'

        print("dataset = ", dataset)
        split = 1 - float(self.trainRate)
        print("split = ", split)

        #self.printf('-----Importing Dataset-----')
        print('-----Importing Dataset-----')

        Dataset = dataset.upper()
        data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT = load_dataset(Dataset, split)
        print(data_hsi.shape)

        image_x, image_y, BAND = data_hsi.shape
        data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))  # prod():连乘函数
        gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )
        CLASSES_NUM = max(gt)
        CLASSTYPE = 13

        self.printf('The target to be detected is: wood')
        print('The class numbers of the HSI data is:', CLASSES_NUM)

        #self.printf('-----Importing Setting Parameters-----')
        print('-----Importing Setting Parameters-----')

        INPUT_DIMENSION = data_hsi.shape[2]
        data = preprocessing.scale(data)  # sklearn里面的--Standardize a dataset along any axis
        data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
        whole_data = data_
        padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                                 'constant', constant_values=0)

        for index_iter in range(ITER):
            #self.printf('iter:'+ str(index_iter))
            print('iter:', index_iter)
            #net = network.network_mish(BAND, CLASSES_NUM, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, device)
            # net = network.DBDA_network_MISH(BAND, CLASSES_NUM)
            #net.load_state_dict(torch.load('net/IN-WOOD_DETECTIONclasstype=4.pt', map_location=device))

            net = torch.load('net/IN-WOOD_DETECTIONclasstype=13.pth', map_location='cpu')

            optimizer = optim.Adam(net.parameters(), lr=lr, amsgrad=False)  # , weight_decay=0.0001)
            time_1 = int(time.time())
            np.random.seed(seeds[index_iter])
            train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
            _, total_indices = sampling(1, gt)

            TRAIN_SIZE = len(train_indices)
            self.printf('Train size: ' + str(TRAIN_SIZE))
            print('Train size: ', TRAIN_SIZE)
            TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
            self.printf('Test size: ' + str(TEST_SIZE))
            print('Test size: ', TEST_SIZE)
            VAL_SIZE = int(TRAIN_SIZE)
            self.printf('Validation size: ' + str(VAL_SIZE))
            print('Validation size: ', VAL_SIZE)

            #self.printf('-----Selecting Small Pieces from the Original Cube Data-----')
            print('-----Selecting Small Pieces from the Original Cube Data-----')

            train_iter, valida_iter, test_iter, all_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE,
                                                                         test_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                                                                         whole_data, PATCH_LENGTH, padded_data,
                                                                         INPUT_DIMENSION, batch_size, gt)

            # train
            tic1 = time.perf_counter()
            # print("tic1 = ", tic1)
            #train.train(net, train_iter, valida_iter, loss, optimizer, device, epochs=num_epochs)
            toc1 = time.perf_counter()
            # print("toc1 = ", toc1)

            pred_test_fdssc = []
            pred_test_possb = []
            tic2 = time.perf_counter()
            # print("tic2 = ", tic2)

            with torch.no_grad():
                for X, y in test_iter:
                    X = X.to(device)
                    net.eval()  # 评估模式
                    y_hat = sigmoid(net(X).cpu())
                    # print(net(X))
                    pred_test_possb.extend(y_hat.cpu().numpy().tolist())
                    pred_test_fdssc.extend(np.array(net(X).cpu().argmax(axis=1)))

            toc2 = time.perf_counter()
            # print("toc2 = ", toc2)
            collections.Counter(pred_test_fdssc)
            gt_test = gt[test_indices] - 1

            pred_test_possbofclass = [i[CLASSTYPE] for i in pred_test_possb]
            gt_re4roc = np.where(gt == CLASSTYPE, 1, 0)

            #评估

            fpr, tpr, thresholds = metrics.roc_curve(gt_test[:-VAL_SIZE], pred_test_possbofclass, pos_label=CLASSTYPE)
            # print("fpr = ", fpr)
            # print("tpr = ", tpr)
            roc_auc = metrics.auc(fpr, tpr)
            self.AUC.append(roc_auc)
            #plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.4f)' % roc_auc)
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.grid()
            #plt.show()

            overall_acc_fdssc = metrics.accuracy_score(pred_test_fdssc, gt_test[:-VAL_SIZE])
            confusion_matrix_fdssc = metrics.confusion_matrix(pred_test_fdssc, gt_test[:-VAL_SIZE])
            each_acc_fdssc, average_acc_fdssc = aa_and_each_accuracy(confusion_matrix_fdssc)  # from generate_pic.py
            kappa = metrics.cohen_kappa_score(pred_test_fdssc, gt_test[:-VAL_SIZE])

            #    torch.save(net.state_dict(), "./net/" + str(round(overall_acc_fdssc, 3)) + '.pt')
            self.KAPPA.append(kappa)
            self.OA.append(overall_acc_fdssc)
            # OA.append('test')
            self.AA.append(average_acc_fdssc)
            self.TRAINING_TIME.append(toc1 - tic1)
            self.TESTING_TIME.append(toc2 - tic2)
            # ELEMENT_ACC[index_iter, :] = each_acc_fdssc

        self.printf("--------" + net.name + " Training Finished-----------")
        print("--------Network Training Finished-----------")
        #print(self.OA)

        record.record_output(self.OA, self.AA, self.KAPPA, self.AUC, self.TRAINING_TIME, self.TESTING_TIME,
                             'records/' + net.name + day_str + '_' + Dataset + 'split：' + str(VALIDATION_SPLIT) + 'lr：' + str(lr) + '.txt')

        generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices, CLASSTYPE)

        self.picpath_res = net.name + '/detection_maps/' + Dataset + '_' + net.name + '.png'
        self.picpath_gt = net.name + '/detection_maps/' + Dataset + '_gt.png'

        #return self.OA


    def startValuation(self):
        self.displayEvaWin = EvaDisplayWindow()
        self.displayEvaWin.label.setPixmap(QPixmap("1.png").scaled(self.displayEvaWin.label.width(),self.displayEvaWin.label.height()))
        self.displayEvaWin.show()
        '''
        self.evaluateWind = EvaluationWindow()
        self.evaluateWind.show()
        self.evaluateWind.printf('OAs for each iteration are:' + str(self.OA) + '\n')
        self.evaluateWind.printf('AAs for each iteration are:' + str(self.AA) + '\n')
        self.evaluateWind.printf( 'KAPPAs for each iteration are:' + str(self.KAPPA) + '\n' + '\n')
        self.evaluateWind.printf('mean_OA ± std_OA is: ' + str(np.mean(self.OA)) + ' ± ' + str(np.std(self.OA)) + '\n')
        self.evaluateWind.printf('mean_AA ± std_AA is: ' + str(np.mean(self.AA)) + ' ± ' + str(np.std(self.AA)) + '\n')
        self.evaluateWind.printf('mean_KAPPA ± std_KAPPA is: ' + str(np.mean(self.KAPPA)) + ' ± ' + str(np.std(self.KAPPA)) + '\n' + '\n')
        self.evaluateWind.printf('Total average Training time is: ' + str(np.sum(self.TRAINING_TIME)) + '\n')
        self.evaluateWind.printf('Total average Testing time is: ' + str(np.sum(self.TESTING_TIME)) + '\n' + '\n')
        '''

    def showResPic(self):
        self.displayWin = PicDisplayWindow()
        self.displayWin.label.setPixmap(QPixmap(self.picpath_gt).scaled(self.displayWin.label.width(),
                                                                         self.displayWin.label.height()))
        self.displayWin.label_2.setPixmap(QPixmap(self.picpath_res).scaled(self.displayWin.label.width(),
                                                                        self.displayWin.label.height()))

        self.displayWin.show()




if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWin = MyMainWindow()
    classifyWind = ClassifyWindow()
    #evaluateWind = EvaluationWindow()

    mainWin.show()

    btn_mainTOclassify = mainWin.pushButton
    btn_mainTOclassify.clicked.connect(classifyWind.show)

    #btn_mainTOevaluate = mainWin.pushButton_2
    #btn_mainTOevaluate.clicked.connect(QCoreApplication.quit)
    #path = 'records/' + net.name + day_str + '_' + Dataset + 'split：' + str(VALIDATION_SPLIT) + 'lr：' + str(lr) + '.txt'
    #print(classifyWind.OA)
    #evaluateWind.printf('AAs for each iteration are:' + str(AA) + '\n')

    #classifyWind.printf("test")

    #d = classifyWind.dataset
    #lr = classifyWind.lr
    #trainRate = classifyWind.trainRate
    #btn_startClassify = classifyWind.pushButton
    #btn_startClassify.clicked.connect(startclassification(d, lr, trainRate))




    #dataset = 'SA'


    #lr = child.lr
    #print("lr = ", lr)

    sys.exit(app.exec_())
