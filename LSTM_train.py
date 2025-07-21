import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import json
import logging
import argparse
from tqdm import tqdm

from data_collector import DataCollector, KoreanDataset
from LSTM import LSTM

ROOT = 'c:/Users/ahris/Desktop/LSTM'
DATA_PATH = f'{ROOT}/dadad'

HYPER_PARAM_PATH = f'{ROOT}'
HYPER_PARAM_NAME = 'hyper_param.json'
HYPER_PARAM_SAVE_PATH = f'{ROOT}/{HYPER_PARAM_NAME}'

WORD_SCORE_PATH = f'{ROOT}'
WORD_SCORE_NAME = 'word_score.pkl'
WORD_SCORE_SAVE_PATH = f'{ROOT}/{WORD_SCORE_NAME}'

VOCAB_PATH = f'{ROOT}'
VOCAB_NAME = 'vocab.json'
VOCAB_SAVE_PATH = f'{ROOT}/{VOCAB_NAME}'


try:
    with open(os.path.join(HYPER_PARAM_PATH, HYPER_PARAM_NAME), 'rb') as jsonfile:
        hyper_param = json.load(jsonfile)
except:
    if not os.path.exists(HYPER_PARAM_PATH):
        os.makedirs(HYPER_PARAM_PATH, exist_ok=True)
    print("Can't find the hyper_param.json. Use dafault settings")
    hyper_param = {
                    'epoch':1000, 'lr':0.01,
                    'batch':4, 'embedding_size': 4,
                    'hidden_size':16, 'layer':1,
                    'bias':True,

            'weight':{
                    'best':1000000, 'last':1000000
                    }
        }


if torch.cuda.is_available():
    device = 'cuda' 
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
else: 
    device = 'cpu'

print('CUDA Device? ',device)

# Token화된 문장을 하나씩 뱉어줌. __getitem__ 오버라이딩
train_data = DataCollector(DATA_PATH, 'sentence', 'train')
valid_data = DataCollector(DATA_PATH, 'sentence', 'valid')

train_data.dataSearch()
valid_data.dataSearch()

# 클래스로 공유하므로 한 번만 실행하기
train_data.tokenizerTrain(save_path=WORD_SCORE_SAVE_PATH)
train_data.makeVocab(save_path=VOCAB_SAVE_PATH)

train_dataset = KoreanDataset(train_data)
valid_dataset = KoreanDataset(valid_data)

if DataCollector.vocab is None:
    raise ValueError("Tokenizer와 vocab을 먼저 초기화하세요.")
vocab:dict[str, int] = DataCollector.vocab
vocab_dim:int = len(vocab)

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch] 

    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    labels_padded = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    return inputs_padded, labels_padded

train_loader = DataLoader(train_dataset, batch_size=hyper_param['batch'], shuffle=True, collate_fn=collate_fn, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=hyper_param['batch'], shuffle=False, collate_fn=collate_fn, num_workers=4)

model = LSTM(
                vocab_size=vocab_dim, 
                embedding_dim=hyper_param['embedding_size'],
                hidden_size=hyper_param['hidden_size'],
                layer=hyper_param['layer'],
                output=vocab_dim,
                bias=hyper_param['bias']
            )
model.to(device)

optimizer = optim.Adam(model.parameters(), hyper_param['lr'])
criterion = nn.CrossEntropyLoss(ignore_index=0)

#TODO: DEFINE VALID LOOP AND SAVE WEIGHT LOGIC

def validation():
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred.view(-1, pred.size(-1)), y.view(-1))
            valid_loss += loss.item()

    return valid_loss

#TODO: DEFINE TRAIN LOOP

def trainLoop():
    model.train()
    epochs = hyper_param['epoch']
    
    for epoch in tqdm(range(epochs), desc="Model Train"):
        total_loss = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred.view(-1, pred.size(-1)), y.view(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_loss = validation()
        
        if hyper_param['weight']['best'] > val_loss:
            hyper_param['weight']['best'] = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val_loss: {val_loss:.4f}")
            with open(HYPER_PARAM_SAVE_PATH, 'w') as jfh:
                json.dump(hyper_param, jfh)
                print("hyper_parameter is updated")

if __name__ == "__main__":
    trainLoop()