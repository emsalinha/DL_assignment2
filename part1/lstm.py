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

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...

        self.batch_size = batch_size
        self.device = device
        self.seq_length = seq_length #10
        self.num_hidden = num_hidden
        # input_dim 1
        # num_hidden 128
        # num_classes 10
        # batch_size 128

        self.W_gx = nn.Parameter(torch.randn(self.num_hidden, input_dim, device=self.device), requires_grad = True)
        self.W_gh = nn.Parameter(torch.randn(self.num_hidden, self.num_hidden, device=self.device), requires_grad = True)
        self.bias_g = nn.Parameter(torch.zeros(self.num_hidden, device=self.device), requires_grad = True)

        self.W_ix = nn.Parameter(torch.randn(self.num_hidden, input_dim, device=self.device), requires_grad = True)
        self.W_ih = nn.Parameter(torch.randn(self.num_hidden, self.num_hidden, device=self.device), requires_grad = True)
        self.bias_i = nn.Parameter(torch.zeros(self.num_hidden, device=self.device), requires_grad = True)

        self.W_fx = nn.Parameter(torch.randn(self.num_hidden, input_dim, device=self.device), requires_grad = True)
        self.W_fh = nn.Parameter(torch.randn(self.num_hidden, self.num_hidden, device=self.device), requires_grad = True)
        self.bias_f = nn.Parameter(torch.zeros(self.num_hidden, device=self.device), requires_grad = True)

        self.W_ox = nn.Parameter(torch.randn(self.num_hidden, input_dim, device=self.device), requires_grad = True)
        self.W_oh = nn.Parameter(torch.randn(self.num_hidden, self.num_hidden, device=self.device), requires_grad = True)
        self.bias_o = nn.Parameter(torch.zeros(self.num_hidden, device=self.device), requires_grad = True)

        self.W_ph = nn.Parameter(torch.randn(self.num_hidden, num_classes, device=self.device), requires_grad = True)
        self.bias_p = nn.Parameter(torch.zeros(num_classes, device=self.device), requires_grad = True)

        self.h_zero = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        self.c_zero = torch.zeros(self.batch_size, self.num_hidden, device=self.device)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()


    def forward(self, x):
        # Implementation here ...
        self.hs = [self.h_zero]
        self.cs = [self.c_zero]
        self.ys = []
        x = x.to(self.device)

        for i in range(0, self.seq_length):

            x_i = torch.reshape(x[:, i], shape=(1, self.num_hidden))
            g = self.tanh(torch.matmul(self.W_gx, x_i) + torch.matmul(self.W_gh, self.hs[-1]) + self.bias_g)
            i = self.sig(torch.matmul(self.W_ix, x_i) + torch.matmul(self.W_ih, self.hs[-1]) + self.bias_i)
            f = self.sig(torch.matmul(self.W_fx, x_i) + torch.matmul(self.W_fh, self.hs[-1]) + self.bias_f)
            o = self.sig(torch.matmul(self.W_ox, x_i) + torch.matmul(self.W_oh, self.hs[-1]) + self.bias_o)

            c = g * i + self.cs[-1] * f
            h = self.tanh(c) * o
            self.hs.append(h)

        p = torch.matmul(h, self.W_ph) + self.bias_p
        y = self.softmax(p)
        self.ys.append(y)

        return self.ys[-1]

    def softmax(self,x):
        b = x.max(dim=1)[0]
        x = torch.transpose(x, -1, 0)
        y = torch.exp(x - b)
        y_sum = y.sum(dim=0)
        s = y / y_sum
        s = torch.transpose(s, -1, 0)
        # 128 * 10
        return s
