import torch
import torch.nn as nn

class LSTM_cell(nn.Module):
    def __init__(self, input_size_size, hidden_size, bias=True):

        super(LSTM_cell, self).__init__()
        _size = input_size_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2hf = nn.Linear(input_size_size, hidden_size, bias=bias)
        self.h2hf =  nn.Linear(hidden_size, hidden_size, bias=bias)

        self.x2hi = nn.Linear(input_size_size, hidden_size, bias=bias)
        self.h2hi =  nn.Linear(hidden_size, hidden_size, bias=bias)

        self.x2ho = nn.Linear(input_size_size, hidden_size, bias=bias)
        self.h2ho =  nn.Linear(hidden_size, hidden_size, bias=bias)

        self.x2hc = nn.Linear(input_size_size, hidden_size, bias=bias)
        self.h2hc =  nn.Linear(hidden_size, hidden_size, bias=bias)
        
    def forward(self, x, c, h):
        ft = torch.sigmoid(self.x2hf(x)+self.h2hf(h))
        it = torch.sigmoid(self.x2hi(x)+self.h2hi(h))
        ot = torch.sigmoid(self.x2ho(x)+self.h2ho(h))
        ct_tilt = torch.tanh(self.x2hc(x)+self.h2hc(h))

        ct = torch.mul(ft, c) + torch.mul(ct_tilt, it)
        ht = torch.mul(torch.tanh(ct), ot)

        return ct, ht

class LSTM(nn.Module):
    def __init__(self, input_size: int,
                 hidden_size: int,
                 layer: int,
                 output: int,
                 bias: bool=True):
        
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden = hidden_size
        self.layer = layer
        self.bias = bias

        # 원하는 수만큼 위로 레이어 쌓기
        self.cell_layer = nn.ModuleList()
        self.cell_layer.append(LSTM_cell(input_size, hidden_size, bias))
        for _ in range(1, self.layer):
            self.cell_layer.append(LSTM_cell(self.hidden, self.hidden, self.bias))
        self.fc = nn.Linear(hidden_size, output)

    def forward(self, x):
        # x: [batch, seq_len, input_size_dim] = (4, 10, 64)
        # 4배치, 단어 10개짜리 문장, 64개 단어에 대해

        # 초기 값
        c0 = torch.zeros(self.layer, x.size(0), self.hidden, device=x.device)
        h0 = torch.zeros(self.layer, x.size(0), self.hidden, device=x.device)

        # 각 layer의 c, h 값을 각자 관리
        c_list = [c0[i] for i in range(self.layer)]
        h_list = [h0[i] for i in range(self.layer)]

        # seq_len 만큼 LSTM_cell 반복( t를 위해 t-10까지 갔다온다)

        outs = []
        for t in range(x.size(1)):
            inp = x[:, t, :]
            for i, cell in enumerate(self.cell_layer):
                c_list[i], h_list[i] = cell(inp, c_list[i], h_list[i])
                inp = h_list[i]
            
            # 각 단어에 적합한 단어 예측(시계열 순서대로)
            out = self.fc(inp)
            outs.append(out)

        # [batch, seq_len, output_dim]
        outs = torch.stack(outs, dim=1)

        return outs
