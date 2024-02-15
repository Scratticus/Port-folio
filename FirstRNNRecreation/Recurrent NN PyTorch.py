import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import string

from torch.autograd import Variable
from torchvision import datasets, transforms

import os

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
os.chdir(current_dir)

def char2tensor(input_string):
    char_tensor = torch.zeros(len(input_string))
    char_tensor = char_tensor.long()
    for i in range(len(input_string)):
        char_tensor[i] = all_chars.index(input_string[i])
    return char_tensor

def make_batches_new(initial_tensor, final_tensor, batch_size):
    #check tensor lengths are equal
    if initial_tensor.shape[0] != final_tensor.shape[0]:
        raise ValueError(f'Tensor lengths are not identical, check how tensors are created/set.')
    for i in range(0, initial_tensor.shape[0], batch_size):
        yield (initial_tensor[i:i+batch_size], final_tensor[i:i+batch_size])

class MyRNN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.input_layer = nn.Embedding(in_size, hidden_size)
        self.hidden_layer = nn.GRU(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, out_size)
        
        
    def forward(self, input_data, hidden_data):
        input_data = self.input_layer(input_data.view(1,-1))
        output_data, hidden_data = self.hidden_layer(input_data.view(1,1,-1), hidden_data)
        output_data = self.output_layer(output_data.view(1,-1))
        return output_data, hidden_data
    
    def initialize_hidden_params(self):
        # This is a convenience function given to set the hidden state to all zeros in the beginning.
        # Doesn't affect the actual hidden layer weights
        return Variable(torch.zeros(1, 1, self.hidden_size))
    
def trainer_rnn(model, input_data, out_expected, batch_size):
    hidden_params = model.initialize_hidden_params()
    model.zero_grad()
    loss_function = nn.CrossEntropyLoss()
    loss = 0.0
    for char in range(batch_size):
        out_rnn, hidden_params = model(input_data[char], hidden_params)
        loss = loss + loss_function(out_rnn, out_expected[char].view(1))
    loss.backward()
    optimizer_rnn.step()
    return loss.data.item()/batch_size

def generate_samples(model, start_char, sample_length):
    with torch.no_grad():
        hidden_params = model.initialize_hidden_params()
        start_tens = char2tensor(start_char)
        gen_text = start_char

        for _ in range(sample_length):
            out_tens, hidden_params = model(start_tens, hidden_params)
            out_prob = torch.softmax(out_tens, dim=1)
            next_tens = torch.multinomial(out_prob.view(-1), 1)[0]
            next_char = all_chars[next_tens]
            gen_text += next_char
            start_tens = next_tens

        return gen_text

with open('Docker Remarks.txt', 'r') as file:
    all_text = file.read()

print(len(all_text))

all_chars = string.printable
n_chars = len(all_chars)

initial = all_text[:-1]
final = all_text[1:]
initial_tensor = char2tensor(initial)
final_tensor = char2tensor(final)

text_size = n_chars
init_tens = initial_tensor[:text_size]
fin_tens = final_tensor[:text_size]
first_char = 'I'
batch_size = 10

first_rnn = MyRNN(in_size=n_chars, hidden_size=n_chars, out_size=n_chars)
learning_rate = 0.005
optimizer_rnn = torch.optim.Adam(first_rnn.parameters(), lr = learning_rate)


n_epochs = 40
for epoch in range(n_epochs):
    batches = make_batches_new(init_tens, fin_tens, batch_size)
    epoch_loss = 0
    for init, final in batches:
        loss = trainer_rnn(first_rnn, init, final, batch_size)    
        epoch_loss += loss
    
    epoch_loss = epoch_loss/init_tens.shape[0]
    print(f'Epoch: {epoch}\nEpoch loss:{epoch_loss}')
    print(generate_samples(first_rnn, start_char=first_char, sample_length=100))