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
        ft = torch.sigmoid(self.x2hf(x)+self.h2hf(h))
        it = torch.sigmoid(self.x2hi(x)+self.h2hi(h))
        ot = torch.sigmoid(self.x2ho(x)+self.h2ho(h))
        ct_tilt = torch.tanh(self.x2hc(x)+self.h2hc(h))

        ct = torch.mul(ft, c) + torch.mul(ct_tilt, it)
        ht = torch.mul(torch.tanh(ct), ot)

        return ct, ht

class LSTM(nn.Module):
    def __init__(self, vocab_size: int,
                 embedding_dim: int,
                 hidden_size: int,
                 layer: int,
                 output: int,
                 bias: bool=True):
        
        super(LSTM, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.input_size = embedding_dim
        self.hidden = hidden_size
        self.layer = layer
        self.bias = bias

        # 원하는 수만큼 위로 레이어 쌓기
        self.cell_layer = nn.ModuleList()
        self.cell_layer.append(LSTM_cell(embedding_dim, hidden_size, bias))
        
        for _ in range(1, self.layer):
            self.cell_layer.append(LSTM_cell(self.hidden, self.hidden, self.bias))
        self.fc = nn.Linear(hidden_size, output)

    def forward(self, x):
        # x: [batch, seq_len] -> embedding: [batch, seq_len, embedding_dim]
        x = self.encoder(x)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 각 레이어의 초기 상태 직접 생성 - 명확하고 효율적
        c_states = [torch.zeros(batch_size, self.hidden, device=x.device) 
                    for _ in range(self.layer)]
        h_states = [torch.zeros(batch_size, self.hidden, device=x.device) 
                    for _ in range(self.layer)]
        
        # 각 시간 스텝별 출력 저장
        outputs = []
        
        # 시퀀스의 각 시간 스텝 처리
        for t in range(seq_len):
            # 현재 시간 스텝의 입력: [batch, embedding_dim]
            current_input = x[:, t, :]
            
            # 각 LSTM 레이어 통과
            for layer_idx, cell in enumerate(self.cell_layer):
                c_states[layer_idx], h_states[layer_idx] = cell(
                    current_input, c_states[layer_idx], h_states[layer_idx]
                )
                # 다음 레이어의 입력은 현재 레이어의 hidden state
                current_input = h_states[layer_idx]
            
            # 최종 레이어의 출력을 fully connected layer에 통과
            output = self.fc(current_input)
            outputs.append(output)
        
        # 리스트를 텐서로 변환: [batch, seq_len, output_dim]
        outputs = torch.stack(outputs, dim=1)
        
        return outputs
