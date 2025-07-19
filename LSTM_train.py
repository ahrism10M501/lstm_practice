import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import json
import logging
import argparse
from tqdm import tqdm

from data_collector import DataCollector
from LSTM import LSTM

#TODO: SET PROJECT PATH

ROOT = 'c:/Users/ahris/Desktop/LSTM'
DATA_PATH = f'{ROOT}/data'
HYPER_PARAM_SAVE_PATH = f'{ROOT}'
#TODO: LOAD HYPER PARAM

hyper_param = {
        'train':{
                'epoch':1000, 'lr':0.01,
                'seq_len':10, 'hidden_size':64,
                'out_dim':64, 'bias':True
                },

        'weight':{
                'best':0, 'last':0
                }
    }

#TODO : Data Load -> [batch, seq_len, input_dim]

## TODO : 1. DataCollector -> [tokenized data]

# Token화된 문장을 하나씩 뱉어줌. __getitem__ 오버라이딩
collector = DataCollector(DATA_PATH, 'sentence')
# default train.
# valid 데이터를 부르고 싶다면, (obj).setMode('valid') 후 사용하면 됨. 단, 다시 train으로 바꿔야함.
collector.tokenizerTrain()

## TODO : 2. integer encoding

tokens = []
for i in range(len(collector)):
    tokens += collector[i]

from collections import Counter

counter = Counter(tokens)
vocab = {token:idx+2 for idx, (token, _) in enumerate(counter.most_common())} # 가장 많이 나오는 순서 빈도수 알려주기
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1


## TODO : 3. Data Loader


#TODO: MODEL_LOAD

model = LSTM()

#TODO: DEFINE OPTIM, LOSS

optimizer = optim.Adam(model.parameters(), hyper_param['train']['lr'])
loss = nn.CrossEntropyLoss()

#TODO: DEFINE VALID LOOP AND SAVE WEIGHT LOGIC

def validLoop():
    pass

#TODO: DEFINE TRAIN LOOP

def trainLoop():
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LSTM training")
    parser.add_argument('-p', '--path', type=str, help='Data root path')
    parser.add_argument('-hp', type=str, help="hyper param file path")

    with open(HYPER_PARAM_SAVE_PATH) as hpf:
        json.dump(hyper_param, hpf)

    trainLoop()