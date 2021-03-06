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

import argparse
import time
from datetime import datetime
import csv
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM
from part1.plot_Q13 import open_results

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def get_accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    """
    i_pred = predictions.max(dim=-1)[1]
    # i_target = targets.max(dim=-1)[1]

    correct = (i_pred == targets).to(dtype = torch.float64)
    accuracy = correct.mean()

    return accuracy

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use

    if config.model_type == 'RNN':
        model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes,
                           config.batch_size, device)
        retain_graph = None

    elif config.model_type == 'LSTM':
        model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes,
                     config.batch_size, device)
        retain_graph = True

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    accuracies = []
    losses = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        t1 = time.time()

        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        predictions = model(batch_inputs)

        loss = criterion(predictions, batch_targets)
        accuracy = get_accuracy(predictions, batch_targets)

        losses.append(loss.item())
        accuracies.append(accuracy.item())

        optimizer.zero_grad()


        loss.backward()

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################


        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

        if loss < 0.1:
            break


    return accuracies, losses

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    accuracies = []
    losses = []
    model = 'RNN'
    T = [9]

    results, headers, accuracies, n_epochs = open_results(model)
    #results = {}

    for t in T:
        # Parse training configuration
        parser = argparse.ArgumentParser()

        # Model params
        parser.add_argument('--model_type', type=str, default=model, help="Model type, should be 'RNN' or 'LSTM'")
        parser.add_argument('--input_length', type=int, default=t, help='Length of an input sequence')
        parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
        parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
        parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
        parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
        parser.add_argument('--max_norm', type=float, default=10.0)
        parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

        config = parser.parse_args()

        # Train the model

        accuracies, losses = train(config)

        results['loss {}'.format(t)] = losses
        results['accuracy {}'.format(t)] = accuracies

    with open('results_{}.csv'.format(model), 'w') as f:
        for key in results.keys():
            f.write("%s,%s\n" % (key, results[key]))
