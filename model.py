import torch
import torch.nn as nn

class LSTM_cell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTM_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2hf = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2hf =  nn.Linear(hidden_size, hidden_size, bias=bias)

        self.x2hi = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2hi =  nn.Linear(hidden_size, hidden_size, bias=bias)

        self.x2ho = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2ho =  nn.Linear(hidden_size, hidden_size, bias=bias)

        self.x2hc = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2hc =  nn.Linear(hidden_size, hidden_size, bias=bias)
        
    def forward(self, x, hidden):
        cx, hx = hidden
        x = x.view(-1, x.size(1))
        ft = torch.sigmoid(self.x2hf(x)+self.h2hf(hx))
        it = torch.sigmoid(self.x2hi(x)+self.h2hi(hx))
        ot = torch.sigmoid(self.x2ho(x)+self.h2ho(hx))
        ct_tilt = torch.tanh(self.x2hc(x)+self.h2hc(hx))

        ct = ft*cx + ct_tilt*it
        ht = torch.tanh(ct)*ot

        return ct, ht
    
class LSTM(nn.Module):
    def __init__(self, input, hidden, layer, output,bias=True):
        super(LSTM, self).__init__()
        self.input = input
        self.hidden = hidden
        self.layer = layer
        self.output = output
        self.bias = bias
        self.c0 = []
        self.h0 = []
        self.cell_layer = []
        self.cell_layer.append(LSTM_cell(input, hidden, bias))
        for _ in range(1, self.layer):
            self.cell_layer.append(LSTM_cell(self.hidden, self.hidden, self.bias))

    def _init_hidden(self):
        c = torch.zeros(-1, self.hidden)
        h = torch.zeros(-1, self.hidden)
        return c, h

    def forward(self, x, ):
        for layer in self.cell_layer:
            c, h = layer(x, (c, h))
        x = h
        p = torch.softmax(h, dim=1)
        return p