# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        # Initialization here...

        self.batch_size = batch_size
        self.device = device
        self.seq_length = seq_length #30
        self.num_hidden = lstm_num_hidden
        self.num_layers = lstm_num_layers
        self.vocabulary_size = vocabulary_size

        self.LSTM = nn.LSTM(input_size=1, hidden_size=self.num_hidden,
                            num_layers=self.num_layers, batch_first=False)

        self.classifier = nn.Linear(self.num_hidden, self.vocabulary_size)


    def forward(self, x, batch = True):
        # Implementation here...
        # input and output size  (batch, seq, feature)
        if batch:
            self.h = torch.zeros(self.num_layers, self.batch_size, self.num_hidden).to(device=self.device)
            self.c = torch.zeros(self.num_layers, self.batch_size, self.num_hidden).to(device=self.device)
        else:
            self.h = torch.zeros(self.num_layers, 1, self.num_hidden).to(device=self.device)
            self.c = torch.zeros(self.num_layers, 1, self.num_hidden).to(device=self.device)

        output.to(device=self.device), (self.h.to(device=self.device), self.c.to(device=self.device)) = self.LSTM(x, (self.h, self.c))

        pred = self.classifier(output)
        pred = pred.view(-1, self.vocabulary_size)
        #pred = pred[-1, :, :]
        return pred

