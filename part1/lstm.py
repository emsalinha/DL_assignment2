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

        self.h = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        self.c = torch.zeros(self.batch_size, self.num_hidden, device=self.device)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()


    def forward(self, x):
        # Implementation here ...
        x = x.to(self.device)

        for i in range(0, self.seq_length):

            x_i = x[:, i].view(-1,1)
            g = self.tanh(torch.matmul(x_i, self.W_gx.t()) + torch.matmul(self.h, self.W_gh.t()) + self.bias_g)
            i = self.sig(torch.matmul(x_i, self.W_ix.t())+ torch.matmul(self.h, self.W_ih.t()) + self.bias_i)
            f = self.sig(torch.matmul(x_i, self.W_fx.t()) + torch.matmul(self.h, self.W_fh.t()) + self.bias_f)
            o = self.sig(torch.matmul(x_i, self.W_ox.t()) + torch.matmul(self.h, self.W_oh.t()) + self.bias_o)

            self.c = g * i + self.c * f
            self.h = self.tanh(self.c) * o

        p = torch.matmul(self.h, self.W_ph) + self.bias_p

        return p

