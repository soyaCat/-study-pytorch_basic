'''
파일 전체적으로 토치 텐서를 사용해보려는 노력을 해보자
'''

import torch
import torch.nn as nn
import numpy as np

import unidecode
import string
import random
import time, math

#load data
all_characters = string.printable
file = unidecode.unidecode(open('./RNNdata/shakespeare.txt').read())
file_len = len(file)
#
print("target characters: ", all_characters)
print("Count of target characters: ", len(all_characters))
print("train_data_length: ", file_len)

#set parameters
totalEpochs = 2000
chunk_len = 200
hidden_size = 100
batch_size = 1
num_layers = 1
embedding_size = 70
lr = 0.002

class ConvertDataType():
    def __init__(self, device):
        self.device = device
    def string2IndexTensor(self, string):
        tensor = torch.zeros(len(string)).long().to(self.device)
        for index in range(len(string)):
            tensor[index] = all_characters.index(string[index])
        return tensor

class RNNModel(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers = 1):
        super(RNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size

        self.encoder = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        out = self.encoder(input.view(1,-1))
        out, hidden = self.rnn(out, hidden)
        out = self.decoder(out.view(batch_size, -1))
        return out, hidden

    def init_hidden(self, device):
        hidden = torch.zeros(self.num_layers, batch_size, hidden_size).to(device)
        return hidden


class RNN():
    def __init__(self, device, ConvertDataType):
        self.device = device
        self.ConvertDataType = ConvertDataType
        self.file_len = file_len
        self.chunk_len = chunk_len

        self.model = RNNModel(input_size = len(all_characters),
                                embedding_size= embedding_size,
                                hidden_size = hidden_size,
                                output_size = len(all_characters),
                                num_layers=1).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= lr)
        self.loss_func = nn.CrossEntropyLoss()


    def train_model(self, file):
        sliced_file = self.ConvertDataType.string2IndexTensor(self.random_chunk(file))
        train_data = sliced_file[:-1]
        answer_data = sliced_file[1:]
        hidden = self.model.init_hidden(self.device)

        loss = torch.tensor([0]).type(torch.FloatTensor).to(self.device)
        self.optimizer.zero_grad()
        for index in range(self.chunk_len-1):
            x = train_data[index]
            y = answer_data[index].unsqueeze(0).type(torch.LongTensor).to(self.device)
            predict, hidden = self.model(x,hidden)
            loss += self.loss_func(predict, y)

        loss.backward()
        self.optimizer.step()

        return loss


    def model_test(self):
        with torch.no_grad():
            start_str = "b"
            input_data = self.ConvertDataType.string2IndexTensor(start_str)
            hidden = self.model.init_hidden(self.device)
            x = input_data
            print(start_str, end = "")
            for i in range(200):
                output, hidden = self.model(x, hidden)
                output_dist = output.data.view(-1).div(0.8).exp()
                index = torch.multinomial(output_dist, 1)[0]
                predicted_char = all_characters[index]

                print(predicted_char, end = "")
                x = self.ConvertDataType.string2IndexTensor(predicted_char)

    def random_chunk(self, file):
        '''
        Randomly cuts the file by chunk length.
        '''
        start_index = random.randint(0, self.file_len - self.chunk_len)
        end_index = start_index + chunk_len + 1
        return file[start_index:end_index]

    def random_training_set(self, file):
        chunk = self.random_chunk(file)
        train_data = self.ConvertDataType.string2IndexTensor(chunk[:-1])
        target_data = self.ConvertDataType.string2IndexTensor(chunk[1:])
        return train_data, target_data

    





if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ConvertDataType = ConvertDataType(device)
    rnn = RNN(device, ConvertDataType)

    for epoch in range(totalEpochs):
        loss = rnn.train_model(file)

        if epoch % 100 == 0:
            print(loss/chunk_len)
            rnn.model_test()
            print("\n \n")
