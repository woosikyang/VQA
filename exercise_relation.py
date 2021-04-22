'''
GCN
'''

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.init import kaiming_uniform_, normal_
class GCN(Module) :
    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GCN,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias :
            self.bias = Parameter(torch.FloatTensor(out_features))
        else :
            self.register_parameter('bias',None)

        if init == 'uniform' :
            self.reset_parameters_uniform()
        elif init == 'xavier' :
            self.reset_parameters_xavier()
        elif init == 'kaiming' :
            self.reset_parameters_kaiming()
        else :
            raise NotImplementedError

    def reset_parameters_uniform(self) :
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)



    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)


    def forward(self,input, adj):
        support = torch.mm(input,self.weight)
        output = torch.spmm(adj,support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + '(' +str(self.in_features) + '->' + str(self.out_features) + ')'



class GAT(Module) :
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GAT,self).__init__()
        self.dropout=dropout
        self.alpha = alpha
        self.concat = concat
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(in_features, out_features).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                                requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(out_features, 1).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                               requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(out_features, 1).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                               requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self,input,adj) :
        h = torch.mm(input,self.W)
        N = h.size()[0]
        f_1 = torch.matmul(h, self.a1)
        f_2 = torch.matmul(h,self.a2)

        # e = self.leakyrelu(f_1 + f_2.transpose(0,1))
        print(f_1.shape)
        print(f_2.shape)
        print(f_2.transpose(0, 1).shape)
        print((f_1 + f_2.transpose(0, 1)).shape)
        e = self.leakyrelu(torch.cat((f_1, f_2)))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)  # normalized attention weights
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class RelationNetworks(Module) :
    def __init__(self, n_vocab, conv_hidden=24, embed_hidden=32, lstm_hidden=128, mlp_hidden=256, classes = 29):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,conv_hidden, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_hidden),
            nn.Relu(),
            nn.Conv2d(conv_hidden, conv_hidden, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_hidden),
            nn.Relu(),
            nn.Conv2d(conv_hidden, conv_hidden, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_hidden),
            nn.Relu(),
            nn.Conv2d(conv_hidden, conv_hidden, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_hidden),
            nn.Relu(),
        )
        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, lstm_hidden, batch_first=True)
        self.n_concat = conv_hidden *2 + lstm_hidden + 2*2
        self.g = nn.Sequential(

        )

        coords = torch.linspace(-4, 4, 8)
        x = coords.unsqueeze(0).repeat(8, 1)
        y = coords.unsqueeze(1).repeat(1, 8)
        coords = torch.stack([x, y]).unsqueeze(0)
