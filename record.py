# -*- coding: utf-8 -*-
import numpy as np

def record_roc(tpr, fpr, auc, path):
    f = open(path, 'a')

    sentence0 = 'TPRs for the network:' + str(tpr) + '\n' + '\n'
    f.write(sentence0)

    sentence1 = 'FPRs for the network:' + str(fpr) + '\n' + '\n'
    f.write(sentence1)

    sentence2 = 'AUC for the network:' + str(auc) + '\n' + '\n'
    f.write(sentence2)


#def record_output(oa_ae, aa_ae, kappa_ae, element_acc_ae, training_time_ae, testing_time_ae, path):
def record_output(oa_ae, aa_ae, kappa_ae, auc_ae, training_time_ae, testing_time_ae, path):
    f = open(path, 'a')

    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)

    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n'
    f.write(sentence2)
    sentence3 = 'mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) + '\n'
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + ' ± ' + str(np.std(aa_ae)) + '\n'
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) + ' ± ' + str(np.std(kappa_ae)) + '\n' + '\n'
    f.write(sentence5)

    sentence6 = 'Total average Training time is: ' + str(np.sum(training_time_ae)) + '\n'
    f.write(sentence6)
    sentence7 = 'Total average Testing time is: ' + str(np.sum(testing_time_ae)) + '\n' + '\n'
    f.write(sentence7)

    sentence8 = 'AUCs for each iteration are:' + str(auc_ae) + '\n'
    f.write(sentence8)


    '''
    element_mean = np.mean(element_acc_ae, axis=0)
    element_std = np.std(element_acc_ae, axis=0)
    sentence8 = "Mean of all elements in confusion matrix: " + str(element_mean) + '\n'
    f.write(sentence8)
    sentence9 = "Standard deviation of all elements in confusion matrix: " + str(element_std) + '\n'
    f.write(sentence9)
    '''
    f.close()





