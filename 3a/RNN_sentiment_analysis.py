import time
import os
import io
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from RNN_model import RNN_language_model


# parse input
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='which gpu(cuda visible device) to use')
args = parser.parse_args()

if not args.gpu:
    print("Using all available GPUs, data parallelism")
else:
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    print("Using gpu: {}".format(args.gpu))


######## load training set ########
vocab_size = 8000

x_train = []
with io.open('../preprocessed_data/imdb_train.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
# each line within our imdb_train.txt file is a single review made up of token ids
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)
    line[line>vocab_size] = 0
    x_train.append(line)

# grab the first 25,000 sequences (train set)
# first 12500 are labeled 1 for positive; next 12500 are 0 for negative
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1

######## load test set ########
vocab_size = 8000
x_test = []
with io.open('../preprocessed_data/imdb_test.txt','r',encoding='utf-8') as f:
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
# no_of_hidden_units equal to 500
model = RNN_language_model(vocab_size, 500)
model.cuda()

opt = 'adam'
LR = 0.001
if opt=='adam':
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif opt=='sgd':
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

batch_size = 200
no_of_epochs = 20
L_Y_train = len(y_train)
L_Y_test = len(y_test)

model.train()

train_loss = []
train_accu = []
test_accu = []


######## training loop ########
for epoch in range(75):

    if epoch == 50 :
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR/10.0

    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0
    epoch_counter = 0

    time1 = time.time()

    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):

        x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
        sequence_length = 50
        x_input = np.zeros((batch_size, sequence_length), dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if sl<sequence_length:
                x_input[j, 0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        x_input = Variable(torch.LongTensor(x_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(x_input)
        loss.backward()

        # Recurrent neural networks can sometimes experience extremely large
        # gradients for a single batch which can cause them to be difficult
        # to train without the gradient clipping
        norm = nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        optimizer.step()

        values, prediction = torch.max(pred, 1)
        prediction = prediction.cpu().data.numpy()
        accuracy = float(np.sum(prediction==x_input.cpu().data.numpy()[:,1:])) / sequence_length
        epoch_acc += accuracy
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

    # test
    if (epoch+1)%1 == 0:
        model.eval()

        epoch_acc = 0.0
        epoch_loss = 0.0
        epoch_counter = 0

        time1 = time.time()

        I_permutation = np.random.permutation(L_Y_test)

        for i in range(0, L_Y_test, batch_size):
            sequence_length = 100
            x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
            x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
            for j in range(batch_size):
                x = np.asarray(x_input2[j])
                sl = x.shape[0]
                if sl<sequence_length:
                    x_input[j,0:sl] = x
                else:
                    start_index = np.random.randint(sl-sequence_length+1)
                    x_input[j,:] = x[start_index:(start_index+sequence_length)]
            x_input = Variable(torch.LongTensor(x_input)).cuda()

            with torch.no_grad():
                pred = model(x_input, train=False)

            values, prediction = torch.max(pred, 1)
            prediction = prediction.cpu().data.numpy()
            accuracy = float(np.sum(prediction==x_input.cpu().data.numpy()[:,1:]))/sequence_length
            epoch_acc += accuracy
            epoch_counter += batch_size

            if (i+batch_size) % 1000 == 0 and epoch==0:
               print(i+batch_size, accuracy/batch_size)

        epoch_acc /= epoch_counter
        epoch_loss /= (epoch_counter/batch_size)
        test_accu.append(epoch_acc)

        time2 = time.time()
        sec = time2-time1
        min, sec = divmod(sec, 60)
        hr, min = divmod(min, 60)

        print('            Test Acc: {:.3f}% | Test Loss: {:.3f} | Time: {:.2f} hr {:.2f} min {:.2f} sec'.format(epoch_acc*100.0, epoch_loss, hr, min, sec))

    torch.cuda.empty_cache()

    if ((epoch+1)%2)==0:
        torch.save(model,'temp.model')
        torch.save(optimizer,'temp.state')
        data = [train_loss,train_accu,test_accu]
        data = np.asarray(data)
        np.save('data.npy',data)

torch.save(model, 'language.model')

