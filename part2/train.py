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

import time
from datetime import datetime
import argparse
import random

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

################################################################################
def get_accuracy(predictions, batch_targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    """
    i_pred = predictions.max(dim=-1)[1]

    correct = (i_pred == batch_targets).float()
    accuracy = correct.mean()

    return accuracy

def generate_sentence(dataset, model, config):

    char_idx = random.randint(0, dataset.vocab_size-1)
    char_idxs = [char_idx]
    char_tensor = torch.tensor(char_idx).view(1, 1, 1).float()

    softmax = torch.nn.Softmax()

    for i in range(0, config.seq_length):

        prediction = model(char_tensor, batch = False).to(device=config.device)

        if config.greedy:
            prediction = softmax(prediction)
            i_preds = prediction.max(dim=-1)[1]
            i_pred = i_preds[-1].item()
        else:
            prediction = softmax(prediction * config.temp)
            i_preds = torch.multinomial(prediction,1)
            i_pred = i_preds[-1].item()

        char_idxs.append(i_pred)

        # sentence = dataset.convert_to_string(char_idxs)
        # print(sentence)

        char_tensor = torch.tensor(char_idxs).view(len(char_idxs), 1, 1).float().to(device=config.device)

    sentence = dataset.convert_to_string(char_idxs)
    return sentence

def train(config):
    
    # Initialize the device which to run the model on
    device = torch.device(config.device)


    # Initialize the dataset and data loader
    dataset = TextDataset(config.txt_file, config.seq_length, config.train_steps, config.batch_size)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size,
             config.lstm_num_hidden, config.lstm_num_layers, config.device).to(device=config.device)

    if config.load:
        model.load_state_dict(torch.load('model_{}_{}.pt'.format(str(config.greedy), config.temp)))

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    if config.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optim == 'RMS':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config.learning_rate_step,
                                   gamma=config.learning_rate_decay)

    if config.save:
        file = open(config.output_dir + 'sentences_{}_{}.txt'.format(str(config.greedy), config.temp), 'w')

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()
        optimizer.zero_grad()

        batch_inputs = torch.unsqueeze(torch.stack(batch_inputs), 2).float()
        batch_targets = torch.stack(batch_targets)
        batch_targets = batch_targets.view(-1)

        batch_inputs, batch_targets = batch_inputs.to(device=config.device), batch_targets.to(device=config.device)

        predictions = model(batch_inputs)

        loss = criterion(predictions, batch_targets)
        accuracy = get_accuracy(predictions, batch_targets)


        loss.backward()
        optimizer.step()
        scheduler.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:

            report = "[{}] Train Step {}/{}, Batch Size = {}, Examples/Sec = {}, " \
                     "Accuracy = {}, Loss = {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss)


        if step % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            sentence = generate_sentence(dataset, model, config)
            print(report)
            print(sentence)
            if config.save:
                file.write(report)
                file.write(sentence)

            torch.save(model, config.output_dir + 'model_{}_{}'.format(str(config.greedy), config.temp))


        if step == config.train_steps or loss < config.conv_criterion:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    if config.save:
        file.close()

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default=(os.getenv("HOME")+'/DL_assignment2/part2/assets/book_EN_grimms_fairy_tails.txt'), help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--conv_criterion', type=float, default=0.2, help='Converge criterion')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--temp', type=float, default=1, help="temperature")
    parser.add_argument('--greedy', type=bool, default=False, help="greedy vs random sampling")
    parser.add_argument('--save', type=bool, default=True, help="save sentences")
    parser.add_argument('--load', type=bool, default=False, help="load pretrained model")

    parser.add_argument('--optim', type=str, default='Adam', help="RMS vs Adam")
    parser.add_argument('--input_dir', type=str, default=None, help="")
    parser.add_argument('--output_dir', type=str, default=(os.getenv("HOME")+'/'), help="")

    config = parser.parse_args()

    # Train the model
    train(config)
