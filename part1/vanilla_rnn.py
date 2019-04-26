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
from collections import deque
import numpy as np

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()

        self.batch_size = batch_size
        self.device = device
        self.seq_length = seq_length #10
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        # input_dim 1
        # num_hidden 128
        # num_classes 10
        # batch_size 128

        # parameters
        # self.W_hx = nn.Parameter(Variable(torch.randn(self.num_hidden, input_dim), requires_grad = True)) #1 x 128
        # self.W_hh = nn.Parameter(Variable(torch.randn(self.num_hidden, self.num_hidden), requires_grad = True)) #128 x 128
        # self.bias_h = nn.Parameter(Variable(torch.zeros(self.num_hidden), requires_grad = True)) #128
        # self.W_ph = nn.Parameter(Variable(torch.randn(self.num_hidden, num_classes), requires_grad = True)) #128x10
        # self.bias_p = nn.Parameter(Variable(torch.zeros(num_classes), requires_grad = True)) #10
        #
        # self.h_zero = torch.zeros(num_hidden)

        self.W_hx = nn.Parameter(torch.randn(self.num_hidden, input_dim, device=self.device), requires_grad = True)#1 x 128
        self.W_hh = nn.Parameter(torch.randn(self.num_hidden, self.num_hidden, device=self.device), requires_grad = True) #128 x 128
        self.bias_h = nn.Parameter(torch.zeros(self.num_hidden, device=self.device), requires_grad = True) #128
        self.W_ph = nn.Parameter(torch.randn(self.num_hidden, num_classes, device=self.device), requires_grad = True) #128x10
        self.bias_p = nn.Parameter(torch.zeros(num_classes, device=self.device), requires_grad = True) #10

        self.activation = nn.Tanh()

    def forward(self, x):

        x = x.to(self.device)

        self.h = torch.zeros(self.num_hidden, self.batch_size, device=self.device)

        for i in range(0, self.seq_length):
            #print(i)
            x_i = x[:, i].view(-1,1)
            #x_i = self.make_one_hot(x_i).t()

            xh = torch.mm(x_i, self.W_hx.t())
            hh = torch.mm(self.W_hh, self.h).t() + self.bias_h
            self.h = self.activation(xh + hh)
            self.h = self.h.t()

        p = torch.mm(self.h.t(), self.W_ph) + self.bias_p

        return p

