import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BOW_model(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        """
        :param vocab_size: # of unique words in all sequences
        :param no_of_hidden_units: embedding dim
        """
        super(BOW_model, self).__init__()
        # input to embedding is token id,
        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)

        self.fc_hidden = nn.Linear(no_of_hidden_units,no_of_hidden_units)
        self.bn_hidden = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout = torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, t):

        bow_embedding = []
        for i in range(len(x)):
            lookup_tensor = Variable(torch.LongTensor(x[i])).cuda()
            # x[i] is a list of token ids that represent a document
            # x[i]s are of different length
            # output is an embedding vector that represents the document
            embed = self.embedding(lookup_tensor)
            # bag of words
            embed = embed.mean(dim=0)
            bow_embedding.append(embed)
        # bow_embedding is: batch_size by embedding_size
        bow_embedding = torch.stack(bow_embedding)

        h = self.dropout(F.relu(self.bn_hidden(self.fc_hidden(bow_embedding))))
        h = self.fc_output(h)

        # Note that two values are returned, the loss and the logit score.
        # The logit score can be converted to an actual probability by passing
        # it through the sigmoid function or it can be viewed as any score
        # greater than 0 is considered positive sentiment.
        return self.loss(h[:,0],t), h[:,0]

