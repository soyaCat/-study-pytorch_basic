'''
순환 신경망은 다른 신경망들의 유니폼과는 다르게 형변환의 최소화를 위해서 모델에서 바로 파이토치 텐서에서 넘파이 에러이로 변환하는 것이 아니라
평소에는 파이토치 텐서형태로 변수들을 보관하다가 필요한 순간 직전에 넘파이 어레이로 변환해준다.
'''

import torch
import torch.nn as nn
import numpy as np

hidden_size = 35
lr = 0.01
epochs = 1000

string = "hello pytorch, how long can rnn cell remember?"
char_list = "abcdefghijklmnopqrtuvwxyz ?!.,:;01"
char_list = [i for i in char_list]

class ConvertDataType():
    def __init__(self, char_list):
        self.char_list = char_list
    
    def string_to_onehot_array(self, string):
        start = np.zeros(shape = len(self.char_list), dtype = int)
        end = np.zeros(shape = len(self.char_list), dtype = int)
        start[-2] = 1
        end[-1] = 1
        for letter in string:
            idx = self.char_list.index(letter)
            onehot = np.zeros(shape = len(self.char_list), dtype = int)
            onehot[idx] = 1
            start = np.vstack([start, onehot])
        onehot_array = np.vstack([start, end])
        return onehot_array
    
    def onehot_to_letter(self, onehot):
        onehot = onehot.detach().cpu().numpy()
        return self.char_list[onehot.argmax()]

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input2hidden = nn.Linear(input_size, hidden_size)
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, input, hidden):
        hidden = self.activation(self.input2hidden(input) + self.hidden2hidden(hidden))
        output = self.output(hidden)
        return output, hidden
    
    def init_hidden(self, device):
        return torch.zeros(1, self.hidden_size).to(device)

class RNN():
    def __init__(self, ConvertDataType):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Model = RNNModel(len(char_list), hidden_size, len(char_list)).to(self.device)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Model.parameters(), lr = lr)

        self.ConvertDataType = ConvertDataType

    def train_model(self, onehot_array):
        self.Model.zero_grad()
        hidden = self.Model.init_hidden(self.device)
        total_loss = 0

        for j in range(onehot_array.size()[0]-1):
            input = onehot_array[j:j+1, :]
            target = onehot_array[j+1]

            output, hidden = self.Model.forward(input, hidden)
            loss = self.loss_func(output.view(-1), target.view(-1))
            total_loss += loss
            input = output

        total_loss.backward()
        self.optimizer.step()

        return total_loss

    def get_sentence_from_letter(self, start, predict_len):
        with torch.no_grad():
            hidden = self.Model.init_hidden(self.device)
            input = start
            output_string = ""
            for i in range(predict_len):
                output, hidden = self.Model.forward(input, hidden)
                output_string += self.ConvertDataType.onehot_to_letter(output.data)
                input = output
            
        return output_string


        





ConvertDataType = ConvertDataType(char_list)
rnn = RNN(ConvertDataType)
onehot_array = ConvertDataType.string_to_onehot_array(string)
onehot_array = torch.FloatTensor(onehot_array).to(rnn.device)

for epoch in range(epochs):
    total_loss  = rnn.train_model(onehot_array)
    
    if epoch % 10 == 0:
        print(total_loss)

start = torch.zeros(1, len(char_list)).to(rnn.device)
start[:, -2] = 1

result_string = rnn.get_sentence_from_letter(start, len(string))
print(result_string)




 
    