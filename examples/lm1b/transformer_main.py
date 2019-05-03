import argparse
import time
import math
import os
import os.path
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from apex.fp16_utils import *
from apex import amp, optimizers

#from stream_gbw import Vocabulary, StreamGBWDataset
from gbw import GBWDataset
from fast_gbw import FastGBWDataset
import util
import argument
import transformer as m
from adam_base import Adam
from rmsprop import RMSprop

parser = argparse.ArgumentParser(description='PyTorch LSTM Language Model')
parser.add_argument('--data', type=str, default='../data/gbw',
                    help='location of the data corpus')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='initial epoch')
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=32,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--save', type=str,  default='gbw_model.pt',
                    help='path to save the final model')
#argument.add_recurrent_args(parser)
argument.add_transformer_args(parser)
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

# Torch
word_freq = torch.load(os.path.join(args.data, 'word_freq.pt')).numpy()
mapto = torch.from_numpy(util.reverse(np.argsort(-word_freq))).long()
print("load word frequency mapping - complete")

ntokens = len(word_freq)
nsampled = 16384

train_corpus = FastGBWDataset(args.data, 'train_data.pt', 'train_data.sid', mapto, seq_length=args.bptt, batch_size=args.batch_size)
print("load train data - complete")

test_corpus = GBWDataset(args.data, 'test_data.pt', mapto)
print("load test data - complete")

# Streaming
'''
vocabulary = Vocabulary.from_file(os.path.join(args.data, "1b_word_vocab.txt"))

ntokens = len(vocabulary)
nsampled = 16384

train_corpus = StreamGBWDataset(vocabulary, os.path.join(args.data, "training-monolingual.tokenized.shuffled/*"))
test_corpus = StreamGBWDataset(vocabulary, os.path.join(args.data, "heldout-monolingual.tokenized.shuffled/*"), deterministic=True)
print("load dataset - complete")
'''

###############################################################################
# Build the model
###############################################################################
embed = nn.Embedding(ntokens, args.emsize)
net = m.TransformerModel(m.DecoderPreprocessor(args, embed), m.TransformerDecoder(args, embed), nsampled)
util.initialize(embed.weight)
net.cuda()

print("Sampled Softmax:", nsampled, "Batch Size:", args.batch_size, "Initial LR:", args.lr)
#optimizer = Adam(net.parameters(), args.lr, betas=(0.0, 0.999))
optimizer = RMSprop(net.parameters(), args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_corpus.batch_num*args.epochs, eta_min=1e-8)
net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

###############################################################################
# Training code
###############################################################################

def get_batch(item, device_id=0):
    data, target, wrd_cnt, batch_num = item
    return Variable(data.cuda(device_id)), Variable(target.view(-1).cuda(device_id)), wrd_cnt, batch_num

def evaluate(data_source, data_gen):
    # Turn on evaluation mode which disables dropout.
    net.eval()

    total_loss = 0
    total_word_count = 0

    for item in data_gen:
        data, targets, word_cnt, batch_num = get_batch(item)
        batch_size = data.size(0) * data.size(1)

        # Sampled Softmax
        logits, new_targets = net(data, targets)
        logits_flat = logits.view(-1, ntokens)
        total_loss += F.cross_entropy(logits_flat, targets, reduction='sum').item()
        total_word_count += batch_size
    return total_loss / total_word_count

def train(train_step):
    train_loader = train_corpus.batch_generator()

    start_time = time.time()
    for batch, item in enumerate(train_loader):
        net.train()
        data, targets, word_cnt, batch_len = get_batch(item)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()

        # Network
        logits, new_targets = net(data, targets)
        loss = F.cross_entropy(logits.view(-1, nsampled+1), new_targets)

        # AMP
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)

        optimizer.step()
        scheduler.step(train_step)
        train_step += 1

        interval = 125
        if batch % interval == 0:
            elapsed = time.time() - start_time
            print('Epoch: {:3d} | {:5d}/{:5d} batches | lr {:.6f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'
                  .format(epoch, batch, batch_len, scheduler.get_lr()[0], elapsed * 1000 / interval, loss.item(), math.exp(loss.item())))
            start_time = time.time()
            sys.stdout.flush()
    return train_step

# Load the saved model.
if os.path.isfile(args.save):
    print("Loading Saved Model")
    with open(args.save, 'rb') as f:
        net.load_state_dict(torch.load(f))
else:
    print("Random Initialization - No Saved Model")

# At any point you can hit Ctrl + C to break out of training early.
train_step = (args.start_epoch * train_corpus.batch_num)
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_step = train(train_step)
        with open(args.save, 'wb') as f:
             torch.save(net.state_dict(), f)

        test_loader = test_corpus.batch_generator(seq_length=args.bptt, batch_size=1, shuffle=False)
        val_loss = evaluate(test_corpus, test_loader)
        print('-' * 89)
        print('Test: {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'
               .format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        print('-' * 89)
        sys.stdout.flush()
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    sys.stdout.flush()

# Run on test data.
test_loader = test_corpus.batch_generator(seq_length=args.bptt, batch_size=1, shuffle=False)
test_loss = evaluate(test_corpus, test_loader)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)
