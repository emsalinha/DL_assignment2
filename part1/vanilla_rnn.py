################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable


################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()

        self.batch_size = batch_size
        self.device = device
        self.seq_length = seq_length #10
        self.num_hidden = num_hidden
        # input_dim 1
        # num_hidden 128
        # num_classes 10
        # batch_size 128

        # parameters
        self.W_hx = nn.Parameter(Variable(torch.randn(self.num_hidden, input_dim), requires_grad = True)) #1 x 128
        self.W_hh = nn.Parameter(Variable(torch.randn(self.num_hidden, self.num_hidden), requires_grad = True)) #128 x 128
        self.bias_h = nn.Parameter(Variable(torch.zeros(self.num_hidden), requires_grad = True)) #128
        self.W_ph = nn.Parameter(Variable(torch.randn(self.num_hidden, num_classes), requires_grad = True)) #128x10
        self.bias_p = nn.Parameter(Variable(torch.zeros(num_classes), requires_grad = True)) #10

        self.h_zero = torch.zeros(num_hidden)

        self.activation = nn.Tanh()

    def forward(self, x):
        self.hs = [self.h_zero]
        self.ys = []

        for i in range(0, self.seq_length):
            x_i = torch.reshape(x[:, i], shape=(1, self.num_hidden))
            xh = torch.matmul(self.W_hx, x_i)
            # print(xh.size())
            hh = torch.matmul(self.hs[-1], self.W_hh) + self.bias_h
            # print(hh.size())
            h = torch.tanh(xh + hh)
            # print(h.size())
            self.hs.append(h)
            p = torch.matmul(h,self.W_ph) + self.bias_p
            # print(p.size())
            y = self.sigmoid(p)
            self.ys.append(y)

        return self.ys[-1]

    def sigmoid(self,x):
        # print(x.size())
        b = x.max(dim=1)[0]
        # print(b.size())
        x = torch.transpose(x, -1, 0)
        # print(x.size())
        y = torch.exp(x - b)
        # print(y.size())
        y_sum = y.sum(dim=0)
        # print(y_sum.size())
        s = y / y.sum(dim=0)
        s = torch.transpose(s, -1, 0)
        print(s.size())
        # 128 * 10
        return s
