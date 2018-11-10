import torch
import torch.nn as nn
from torch.autograd import Variable


class StatefulLSTM(nn.Module):
    def __init__(self,in_size,out_size):
        super(StatefulLSTM,self).__init__()

        # nn.LSTMCell() contains all of the actual LSTM weights as well as all of the operations
        self.lstm = nn.LSTMCell(in_size,out_size)
        self.out_size = out_size

        self.h = None
        self.c = None

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self,x):

        batch_size = x.data.size()[0]
        if self.h is None:
            state_size = [batch_size, self.out_size]
            self.c = Variable(torch.zeros(state_size)).cuda()
            self.h = Variable(torch.zeros(state_size)).cuda()
        self.h, self.c = self.lstm(x,(self.h,self.c))

        return self.h


# use same dropout mask compared with the traditional dropout
# It has been shown to be more effective to use the same dropout mask
# for an entire sequence as opposed to a different dropout mask each time
class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout,self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        if not train:
            return x
        if(self.m is None):
            self.m = x.data.new(x.size()).bernoulli_(1 - dropout)
        mask = Variable(self.m, requires_grad=False) / (1 - dropout)

        return mask * x


class RNN_model(nn.Module):
    def __init__(self, no_of_hidden_units):
        super(RNN_model, self).__init__()

        self.lstm1 = StatefulLSTM(300, no_of_hidden_units)
        self.bn_lstm1 = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = LockedDropout()

        self.fc_output = nn.Linear(no_of_hidden_units, 1)

        self.loss = nn.BCEWithLogitsLoss()

    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()

    def forward(self, x, t, train=True):
        no_of_timesteps = x.shape[1]

        self.reset_state()

        outputs = []
        for i in range(no_of_timesteps):
            h = self.lstm1(x[:, i, :])
            h = self.bn_lstm1(h)
            h = self.dropout1(h, dropout=0.5, train=train)

            outputs.append(h)

        outputs = torch.stack(outputs)  # (time_steps,batch_size,features)
        outputs = outputs.permute(1, 2, 0)  # (batch_size,features,time_steps)

        pool = nn.MaxPool1d(x.shape[1])
        h = pool(outputs)
        h = h.view(h.size(0), -1)
        h = self.fc_output(h)

        return self.loss(h[:, 0], t), h[:, 0]

