import numpy as np
import torch
import tensorflow as tf
import math
from torch.nn import Module, Sequential, Conv2d, Conv3d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding, Tanh
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # self.query_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.key_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.value_conv = Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        # m_batchsize, channle, height, width, C = x.size()
        x = x.squeeze(-1)
        # m_batchsize, C, height, width, channle = x.size()

        # proj_query = self.query_conv(x).view(m_batchsize, -1, width*height*channle).permute(0, 2, 1)
        # proj_key = self.key_conv(x).view(m_batchsize, -1, width*height*channle)
        # energy = torch.bmm(proj_query, proj_key)
        # attention = self.softmax(energy)
        # proj_value = self.value_conv(x).view(m_batchsize, -1, width*height*channle)
        #
        # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out = out.view(m_batchsize, C, height, width, channle)
        # print('out', out.shape)
        # print('x', x.shape)

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  #view()相当于numpy中resize（）的功能
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = (self.gamma*out + x).unsqueeze(-1)
        return out



class Spectral_Module(Module):
    def __init__(self, attention_size, hidden_size, device):
        super(Spectral_Module, self).__init__()
        self.attention_size = attention_size
        self.w_omega = Variable(torch.randn(hidden_size, attention_size), requires_grad=True).to(device)
        self.b_omega = Variable(torch.randn(attention_size), requires_grad=True).to(device)
        self.u_omega = Variable(torch.randn(attention_size), requires_grad=True).to(device)

        self.softmax = Softmax(dim=-1)
        self.tanh = Tanh()
        self.device = device



    def forward(self, x):
        #print("in attention x:", x.shape)
        #print("x.shape[2]:", x.shape[2])
        #x_temp = torch.cat((x,x),2)
        hidden_size = x.shape[2]
        #hidden_size = x_temp.shape[2]

        # gpu上需取消注释
        #w_omega = Variable(torch.randn(hidden_size, self.attention_size), requires_grad=True).to(self.device)
        #b_omega = Variable(torch.randn(self.attention_size), requires_grad=True).to(self.device)
        #u_omega = Variable(torch.randn(self.attention_size), requires_grad=True).to(self.device)
        w_omega = Variable(torch.randn(hidden_size, self.attention_size), requires_grad=True)
        b_omega = Variable(torch.randn(self.attention_size), requires_grad=True)
        u_omega = Variable(torch.randn(self.attention_size), requires_grad=True)

        temp = torch.tensordot(x, w_omega, dims=([-1], [0])) + b_omega
        #temp = torch.tensordot(x, w_omega, dims=([-1], [0])) + b_omega
        #temp = torch.tensordot(x_temp, w_omega, dims=([-1], [0])) + b_omega
        v = self.tanh(temp)
        vu = torch.tensordot(v, u_omega, dims=([-1], [0]))
        #vu = torch.tensordot(v, u_omega, dims=([-1], [0]))
        alphas = self.softmax(vu)

        inputs_ = x * torch.unsqueeze(alphas, -1)
        #inputs_ = x_temp * torch.unsqueeze(alphas, -1)
        out = inputs_.sum(1)


        return out


        
