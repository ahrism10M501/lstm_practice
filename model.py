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
        
    def forward(self, x, c, h):
        x = x.view(-1, x.size(1))
        ft = torch.sigmoid(self.x2hf(x)+self.h2hf(h))
        it = torch.sigmoid(self.x2hi(x)+self.h2hi(h))
        ot = torch.sigmoid(self.x2ho(x)+self.h2ho(h))
        ct_tilt = torch.tanh(self.x2hc(x)+self.h2hc(h))

        ct = torch.mul(ft, c) + torch.mul(ct_tilt, it)
        ht = torch.mul(torch.tanh(ct), ot)

        return ct, ht
    
class LSTM(nn.Module):
    def __init__(self, seq_len, batch, input, hidden, layer, output,bias=True):
        super(LSTM, self).__init__()
        self.seq_len = seq_len
        self.batch = batch
        self.input = input
        self.hidden = hidden
        self.layer = layer
        self.output = output
        self.bias = bias

        self.cell_layer = nn.ModuleList()
        self.cell_layer.append(LSTM_cell(input, hidden, bias))
        for _ in range(1, self.layer):
            self.cell_layer.append(LSTM_cell(self.hidden, self.hidden, self.bias))
        self.fc = nn.Linear(hidden, output)

    def _init_hidden(self):
        c = torch.zeros(self.batch, self.hidden)
        h = torch.zeros(self.batch, self.hidden)
        return c, h

    def forward(self, x, c, h):
        cx, hx = c, h
        # TODO: seq에 따라 시계열 데이터 처리하기
        # 지금은 one-to-one -> 생성은 또 어케하는걸까
        for layer in self.cell_layer:
            cx, hx = layer(x, cx, hx)

        x = self.fc(hx)
        x = torch.softmax(x, dim=1)

        self.cx = cx
        self.hx = hx

        return x, cx, hx