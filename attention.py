import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from language_model import QuestionEmbedding2, WordEmbedding
from fc import FCNet


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits





class NewAttention2(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention2, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits


class Implicit(nn.Module):
    """Implicit graph for image objects
    """
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(Implicit, self).__init__()
        # using last rnn hidden states to self attention for question embedding

        self.u_proj = FCNet([num_hid, q_dim + v_dim])
        self.v_proj = FCNet([num_hid, q_dim + v_dim])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v,b,q):
        '''

        :param v: [batch, k, vdim]
        :param b: [batch, k, 6]
        scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height, scaled_width, scaled_height
        :param q: [batch, 14]
        :return:
        '''


#len(train_dset[0])
# features, spatials, question, target , objects_oh,qt_oh

#train_dset[0][2].shape

import math
from torch.autograd import Variable
class PositionalEncoding(nn.Module) :
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        #Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0,max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0.0, d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x+ Variable(self.pe[:,:x.size(1)], requires_grad=False)
        return self.dropout(x)

import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(4, 0)
y = pe.forward(Variable(torch.zeros(1, 5, 4)))


