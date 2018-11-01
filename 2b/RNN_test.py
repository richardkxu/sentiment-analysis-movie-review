import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys
import io
import argparse

from RNN_model import RNN_model


# parse input
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='which gpu(cuda visible device) to use')
args = parser.parse_args()

if not args.gpu:
    print("Using all available GPUs, data parallelism")
else:
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    print("Using gpu: {}".format(args.gpu))

######## load test set ########
glove_embeddings = np.load('../preprocessed_data/glove_embeddings.npy')
vocab_size = 100000

x_test = []
with io.open('../preprocessed_data/imdb_test_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

# add 1 to vocab_size. Remember we actually added 1 to all
# of the token ids so we could use id 0 for the unknown token
vocab_size += 1
model = torch.load('rnn_seq250.model')
model.cuda()


batch_size = 200
no_of_epochs = 10
L_Y_test = len(y_test)

model.eval()

test_accu = []

# test
for epoch in range(no_of_epochs):
    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0
    epoch_counter = 0

    sequence_length = (epoch+1)*50

    time1 = time.time()

    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):

        x_input2 = [x_test[j] for j in I_permutation[i:i + batch_size]]
        x_input = np.zeros((batch_size, sequence_length), dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if (sl < sequence_length):
                x_input[j, 0:sl] = x
            else:
                start_index = np.random.randint(sl - sequence_length + 1)
                x_input[j, :] = x[start_index:(start_index + sequence_length)]
        y_input = y_test[I_permutation[i:i + batch_size]]

        data = Variable(torch.LongTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        with torch.no_grad():
            loss, pred = model(data, target, train=False)

        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    test_accu.append(epoch_acc)

    time2 = time.time()
    sec = time2-time1

    min, sec = divmod(sec, 60)
    hr, min = divmod(min, 60)
    print('Seq Length: {} | Test Acc: {:.3f}% | Test Loss: {:.3f} | Time: {:.2f} hr {:.2f} min {:.2f} sec'.format(sequence_length, epoch_acc*100.0, epoch_loss, hr, min, sec))


