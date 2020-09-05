import torch
from torch import nn
from AttentionModule import PAM_Module, CAM_Module, Spectral_Module
import math
import tensorflow as tf
import torch.nn.functional as F
from torch.autograd import Variable
from ActivationFunc import mish


class Residual_2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, batch_normal = False, stride=1):
        super(Residual_2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                    kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding,stride=stride)
        if batch_normal:
            self.bn = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.bn = nn.ReLU()
    def forward(self, X):
        Y = F.relu(self.conv1(self.bn(X)))
        Y = self.conv2(Y)
        return F.relu(Y + X)


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU()
        )
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding,stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)



class Separable_Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, kernel_size=1, stride=1):
        #super(Separable_Convolution, self).__init__()
        self.depth_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        self.point_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out






class network_mish(nn.Module):
    def __init__(self, band, classes, input_size, hidden_size, num_layers, device):

        super(network_mish, self).__init__()
        self.name = 'network_mish'

        #spectral part
        #self.device = device
        self.band = band
        self.num_class = classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers


        #self.spe_weights = Variable(torch.randn(hidden_size * 2, 60), requires_grad=True).to(device)
        #self.spe_biases = Variable(torch.randn(60), requires_grad=True).to(device)
        self.spe_weights = Variable(torch.randn(hidden_size * 2, 60), requires_grad=True)
        self.spe_biases = Variable(torch.randn(60), requires_grad=True)

        self.rnn = nn.RNN(band, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(60, classes)
        self.device = device

        #对比实验
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))




        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, band), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
                                    nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
                                    #gelu_new()
                                    #swish()
            mish()
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))


        self.conv25 = nn.Sequential(
                                nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                                kernel_size=(3, 3, 2), stride=(1, 1, 1)),
                                nn.Sigmoid()
        )

        self.batch_norm_spectral = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new(),
                                    #swish(),
            mish(),
                                    nn.Dropout(p=0.5)
        )
        self.batch_norm_spatial = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.001, momentum=0.1, affine=True), # 动量默认值为0.1
                                    #gelu_new(),
                                    #swish(),
            mish(),
                                    nn.Dropout(p=0.5)
        )


        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                #nn.Dropout(p=0.5),
                                nn.Linear(120, classes) # ,
                                # nn.Softmax()
        )
        #self.full_connection = nn.Sequential(
        #    # nn.Dropout(p=0.5),
        #    nn.Linear(classes*2, classes)  # ,
        #    # nn.Softmax()
        #)

        self.attention_spectral = Spectral_Module(60, hidden_size * 2, device)
        self.attention_spatial = PAM_Module(60)

        #for detection
        self.sigmoid = nn.Sigmoid()

        #fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, X):



        # spectral
        '''
        # num_classes = 6
        num_hidden = 512
        ATTENTION_SIZE = 32
        x11 = torch.nn.GRUCell(X, num_hidden)
        x12 = torch.nn.GRUCell(X, num_hidden)

        #x11 = tf.nn.rnn_cell.GRUCell(num_hidden)  # num_hidden     = 512
        #x12 = tf.nn.rnn_cell.GRUCell(num_hidden)
        x1, _ = tf.nn.bidirectional_dynamic_rnn(x11, x12, X, dtype=tf.float32)
        '''

        #x110 = self.conv11(X)
        #print('x110', x110.shape)

        x11 = X.float()
        #print('x11', x11.shape)


        #x11 = x11.to(self.device)

        x12 = x11.squeeze(1)
        #print('x12', x12.shape)
        batchsize = x12.size(0)
        x12 = x12.reshape(batchsize, -1, self.band)
        #print('x12-', x12.shape)
        #x12 = x11.to(self.device)
        #h0 = torch.zeros(self.num_layers * 2, x11.size(0), self.hidden_size).to(self.device)


        #h0 = torch.zeros(self.num_layers * 2, x12.size(0), self.hidden_size).to(self.device)
        h0 = torch.zeros(self.num_layers * 2, x12.size(0), self.hidden_size)

        #print('h0', h0.shape)
        # 反向传播
        x13, _ = self.rnn(x12, h0)
        #print("x13", x13.shape)
        # print('h',h)
        # print('c',c)
        # 最后一步全连接输出
        #x14 = self.fc(x13[:, -1, :])
        #x14 = self.fc(x13[:, :, :])
        #print("x14", x14.shape)
        #hsize = x13.size(2)
        x15 = self.attention_spectral(x13)
        #print("x15", x15.shape)
        #spe_weights = Variable(torch.randn(self.hidden_size * 2, self.num_class), requires_grad=True)
        #spe_biases = Variable(torch.randn(self.num_class), requires_grad=True)
        #spe_weights = Variable(torch.randn(self.hidden_size * 2, 60), requires_grad=True).to(self.device)
        #spe_biases = Variable(torch.randn(60), requires_grad=True).to(self.device)

        x1 = torch.matmul(x15, self.spe_weights) + self.spe_biases
        #print("x1", x1.shape)       #x1 torch.Size([16, 14])
        #x1 = tf.nn.xw_plus_b(x13, spe_weights, spe_biases)



        # spatial
        #print('x', X.shape)
        x21 = self.conv21(X)
        #print('x21', x21.shape)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        #print('x25', x25.shape)


        # 空间注意力机制
        x2 = self.attention_spatial(x25)
        x2 = torch.mul(x2, x25)
        #print("x2", x2.shape)       #x2 torch.Size([16, 60, 9, 9, 1])

        # model1
        #x1 = self.batch_norm_spectral(x1)
        #x1 = self.global_pooling(x1)
        #x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = self.batch_norm_spatial(x2)
        x2 = self.global_pooling(x2)
        #print("x2", x2.shape)        #x2 torch.Size([16, 60, 1, 1, 1])
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)
        #print("x2", x2.shape)       #x2 torch.Size([16, 60])

        x1_ = self.fc(x1)
        x2_ = self.fc(x2)

        #x_pre = torch.cat((x1_, x2_), dim=1)
        x_pre = torch.cat((x1, x2), dim=1)
        #print('x_pre', x_pre.shape)

        output = self.full_connection(x_pre)
        #output = self.full_connection(x_pre)
        # output = self.fc(x_pre)
        #for detection
        output_ = self.sigmoid(output)

        #return x1_
        return output

