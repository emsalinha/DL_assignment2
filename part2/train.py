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

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from part2.dataset import TextDataset
from part2.model import TextGenerationModel

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

def generate_sentence(dataset, model, seq_length):

    char_idx = random.randint(0, dataset.vocab_size-1)
    char_idxs = [char_idx]
    char_tensor = torch.tensor(char_idx).view(1, 1, 1).float()

    for i in range(0, seq_length):
        prediction = model(char_tensor, batch = False)

        i_preds = prediction.max(dim=-1)[1]
        i_pred = i_preds[-1].item()
        char_idxs.append(i_pred)

        # sentence = dataset.convert_to_string(char_idxs)
        # print(sentence)

        char_tensor = torch.tensor(char_idxs).view(len(char_idxs), 1, 1).float()

    sentence = dataset.convert_to_string(char_idxs)
    print(sentence)

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader
    dataset = TextDataset(config.txt_file, config.seq_length, config.train_steps, config.batch_size)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size,
                 config.lstm_num_hidden, config.lstm_num_layers, config.device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()
        optimizer.zero_grad()

        batch_inputs = torch.unsqueeze(torch.stack(batch_inputs), 2).float()
        batch_targets = torch.stack(batch_targets)
        batch_targets = batch_targets.view(-1)
        #batch_targets = batch_targets[-1, :]

        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        predictions = model(batch_inputs)

        loss = criterion(predictions, batch_targets)
        accuracy = get_accuracy(predictions, batch_targets)


        loss.backward()
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:

            print("[{}] Train Step {}/{}, Batch Size = {}, Examples/Sec = {}, "
                  "Accuracy = {}, Loss = {}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            generate_sentence(dataset, model, config.seq_length)
            #pass

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='assets/book_EN_grimms_fairy_tails.txt', help="Path to a .txt file to train on")
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
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)
