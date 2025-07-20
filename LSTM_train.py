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
DATA_PATH = f'{ROOT}/data'

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
                    'batch':16, 'embedding_size': 128,
                    'hidden_size':256, 'layer':3,
                    'out_dim':64, 'bias':True,

            'weight':{
                    'best':0, 'last':0
                    }
        }

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Token화된 문장을 하나씩 뱉어줌. __getitem__ 오버라이딩
train_data = DataCollector(DATA_PATH, 'sentence')
valid_data = DataCollector(DATA_PATH, 'sentence', 'valid')
print('\nCollector\n', train_data.__len__())
print(valid_data.__len__())
# 클래스로 공유하므로 한 번만 실행하기
train_data.tokenizerTrain(path=WORD_SCORE_SAVE_PATH)
train_data.makeVocab(path=VOCAB_SAVE_PATH)

train_dataset = KoreanDataset(train_data)
valid_dataset = KoreanDataset(valid_data)

dataset_example1:torch.Tensor = train_dataset[0][0]
dataset_example2:torch.Tensor = train_dataset[0][1]
print('\nDataset')
print('train data shpae\n{}\n{}'.format(train_dataset.__len__(), dataset_example1.shape, dataset_example2.shape))
print(valid_dataset.__len__())

if DataCollector.vocab is None:
    raise ValueError("Tokenizer와 vocab을 먼저 초기화하세요.")
vocab:dict[str, int] = DataCollector.vocab
vocab_dim:int = len(vocab)
print('\nvocab\n',vocab_dim)

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch] 
    print(f'\nCollate in {len(inputs)}, label {len(labels)}')

    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    labels_padded = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    return inputs_padded, labels_padded

print('loader')
train_loader = DataLoader(train_dataset, batch_size=hyper_param['batch'], shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=hyper_param['batch'], shuffle=False, collate_fn=collate_fn)

model = LSTM(
                vocab_size=vocab_dim, 
                embedding_dim=hyper_param['embedding_size'],
                hidden_size=hyper_param['hidden_size'],
                layer=hyper_param['layer'],
                output=hyper_param['out_dim'],
                bias=hyper_param['bias']
            )

optimizer = optim.Adam(model.parameters(), hyper_param['lr'])
criterion = nn.CrossEntropyLoss()

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
    model.to(device)
    epochs = hyper_param['epoch']
    
    for epoch in tqdm(range(epochs)):
        total_loss = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            print(x.shape, y.shape)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred.view(-1, pred.size(-1)), y.view(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_loss = validation()
        
        if hyper_param['weight']['best'] < val_loss:
            hyper_param['weight']['best'] = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val_loss: {val_loss:.4f}")
            with open(HYPER_PARAM_SAVE_PATH, 'w') as jfh:
                json.dump(hyper_param, jfh)
                print("hyper_parameter is updated")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LSTM training")
    parser.add_argument('-p', '--path', type=str, help='Data root path')
    parser.add_argument('-hp', type=str, help="hyper param file path")
    
    print(f'train_loop')
    trainLoop()