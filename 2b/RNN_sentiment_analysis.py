import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import time
import os
import io
import argparse

from RNN_model import RNN_model


# parse input
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='which gpu(cuda visible device) to use')
parser.add_argument('--seq_len', type=int, help='number of words per document to cut')
args = parser.parse_args()

if not args.gpu:
    print("Using all available GPUs, data parallelism")
else:
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    print("Using gpu: {}".format(args.gpu))

print("Using sequence length of: {}".format(args.seq_len))

######## load training set ########
glove_embeddings = np.load('../preprocessed_data/glove_embeddings.npy')
vocab_size = 100000

x_train = []
with io.open('../preprocessed_data/imdb_train_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)
    line[line>vocab_size] = 0
    x_train.append(line)

x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1


vocab_size += 1
model = RNN_model(500)
model.cuda()


opt = 'adam'
LR = 0.001
if(opt=='adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

seq_length = args.seq_len
batch_size = 200
no_of_epochs = 100
L_Y_train = len(y_train)

model.train()

train_loss = []
train_accu = []
test_accu = []


######## training loop ########
for epoch in range(no_of_epochs):

    # training
    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0
    epoch_counter = 0

    time1 = time.time()

    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):

        x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
        sequence_length = 100
        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        x_input = glove_embeddings[x_input]
        y_input = y_train[I_permutation[i:i+batch_size]]

        data = Variable(torch.FloatTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(data,target,train=True)
        loss.backward()

        optimizer.step()   # update weights

        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    time2 = time.time()
    sec = time2-time1
    min, sec = divmod(sec, 60)
    hr, min = divmod(min, 60)
    print('Epoch: {} | Train Acc: {:.3f}% | Train Loss: {:.3f} | Time: {:.2f} hr {:.2f} min {:.2f} sec'.format(epoch, epoch_acc*100.0, epoch_loss, hr, min, sec))

torch.save(model, "rnn_seq" + str(args.seq_len) + ".model")


