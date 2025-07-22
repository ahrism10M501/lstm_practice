import os
import json
import pickle
import logging
from tqdm import tqdm

from soynlp.tokenizer import MaxScoreTokenizer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import amp
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device {}'.format(device), end="\n")

import re_datacolllector
from re_datacolllector import DataCollector, KoreanDataset
from LSTM import LSTM

### 프로젝트 파일 관리 ###
ROOT = '.'
DATA_PATH = f'{ROOT}/dadad'
HYPER_PARAM_PATH = f'{ROOT}/hyper_param.json'
CORPUS = None
WORD_SCORE = None
VOCAB = None

### HYPER PARAM ###
if not os.path.exists(HYPER_PARAM_PATH):
    hyper_param = {
        'epoch':100,
        'lr':1e-4,
        'batch':4,
        'num_workers':0,
        'embedding_size':4,
        'hidden_size':8,
        'layer':1,
        'bias':True,
        'print_interval':1,
        'weight':{
                'best':-1.0,
                'last':0
                }
    }
else:
    with open(HYPER_PARAM_PATH, 'r', encoding='utf-8') as param_json:
        hyper_param = json.load(param_json)

####### DATA PREPROCESSING #######
### DATA LOAD ###
train_data = DataCollector(DATA_PATH, 'train', True)
valid_data = DataCollector(DATA_PATH, 'valid', True)

### TOKNIZER ###
if os.path.exists(os.path.join(ROOT, "word_score.pkl")):
    with open(os.path.join(ROOT, "word_score.pkl"), 'rb') as f:
        score = pickle.load(f)
else:
    score = re_datacolllector.makeWordScore(train_data, save_path=ROOT)

tokenizer = MaxScoreTokenizer(scores=score)

if os.path.exists(os.path.join(ROOT, "vocab.json")):
    with open(os.path.join(ROOT, "vocab.json"), 'r', encoding='utf-8') as f:
        vocab = json.load(f)
else:
    vocab = re_datacolllector.makeVocab(train_data, tokenizer, save_path=ROOT)


### DATA LOADER ###
train_dataset = KoreanDataset(train_data, tokenizer, vocab, device)
valid_dataset = KoreanDataset(valid_data, tokenizer, vocab, device)

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch] 

    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    labels_padded = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    return inputs_padded, labels_padded

train_loader = DataLoader(train_dataset, batch_size=hyper_param['batch'], shuffle=True, collate_fn=collate_fn, num_workers=hyper_param['num_workers'])
valid_loader = DataLoader(valid_dataset, batch_size=hyper_param['batch'], shuffle=False, collate_fn=collate_fn, num_workers=hyper_param['num_workers'])

### MODEL LOAD ###
model = LSTM(
                vocab_size=len(vocab), 
                embedding_dim=hyper_param['embedding_size'],
                hidden_size=hyper_param['hidden_size'],
                layer=hyper_param['layer'],
                output=len(vocab),
                bias=hyper_param['bias']
            )
model.to(device)

### TRAIN ###
optimizer = optim.Adam(model.parameters(), hyper_param['lr'])
criterion = nn.CrossEntropyLoss(ignore_index=0)

# GEMINI 2.5 flash prompt: evaluate를 완성하라
def evaluate(model, valid_loader, criterion, device, pad_idx=0): # pad_idx를 인자로 추가했습니다.
    model.eval() # 모델을 평가 모드로 설정
    total_loss = 0 # 전체 손실을 누적할 변수
    total_correct_predictions = 0 # 정확한 예측 수를 누적할 변수
    total_tokens_evaluated = 0 # 패딩이 아닌 실제 토큰 수를 누적할 변수

    with torch.no_grad(): # 역전파를 위한 그래디언트 계산을 비활성화 (메모리 절약 및 속도 향상)
        for x, y in tqdm(valid_loader, desc="evaluate... "): # 검증 데이터로더에서 배치(x, y)를 가져옴
            x = x.to(device) # 입력 텐서를 지정된 디바이스로 이동
            y = y.to(device) # 정답 텐서를 지정된 디바이스로 이동 (y는 실제 다음 토큰을 포함)

            pred = model(x) # 모델에 입력(x)을 전달하여 예측(pred)을 얻음
            # pred의 형태: (batch_size, sequence_length, vocab_size)

            # 손실 계산
            # pred.view(-1, pred.size(-1))는 (배치_크기 * 시퀀스_길이, 어휘_사전_크기)로 reshape
            # y.view(-1)는 (배치_크기 * 시퀀스_길이,)로 reshape
            loss = criterion(pred.view(-1, pred.size(-1)), y.view(-1))
            total_loss += loss.item() # 현재 배치의 손실 값을 Python 숫자로 변환하여 누적

            # 정확도 계산
            # 각 토큰에 대해 가장 높은 확률을 가진 예측 토큰의 인덱스를 가져옴
            predicted_tokens = torch.argmax(pred, dim=-1) # 형태: (batch_size, sequence_length)

            # 정확도 계산에서 패딩 토큰을 무시하기 위한 마스크 생성
            # pad_idx가 0이라고 가정하고, y가 0이 아닌 경우에만 True
            non_pad_mask = (y != pad_idx)

            # 예측과 실제 레이블을 비교하고, 패딩이 아닌 토큰에 대해서만 정확성 계산
            correct_predictions = (predicted_tokens == y) & non_pad_mask

            # 현재 배치의 정확한 예측 수와 패딩이 아닌 실제 토큰 수를 누적
            total_correct_predictions += correct_predictions.sum().item()
            total_tokens_evaluated += non_pad_mask.sum().item()
            
    # 평균 손실 계산
    avg_loss = total_loss / len(valid_loader)
    
    # 전체 정확도 계산
    if total_tokens_evaluated > 0: # 평가할 실제 토큰이 있는 경우에만 정확도 계산
        accuracy = total_correct_predictions / total_tokens_evaluated
    else:
        accuracy = 0.0 # 평가할 토큰이 없으면 정확도는 0

    return avg_loss, accuracy # 평균 손실과 정확도를 반환

def train():
    epochs = hyper_param['epoch']
    scaler = amp.GradScaler(device)
    for epoch in tqdm(range(epochs), desc="Model Train... "):
        model.train()
        epoch_loss = 0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}: batch... "):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            with amp.autocast(device):
                pred = model(x)
                loss = criterion(pred.view(-1, pred.size(-1)), y.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        val_loss, val_acc = evaluate(model=model, valid_loader=valid_loader, criterion=criterion, device=device)

        if (epoch+1) % hyper_param['print_interval'] == 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Val_loss: {val_loss:.4f}, Val_acc: {val_acc:.4f}")
            hyper_param['weight']['last'] = val_acc
            with open(HYPER_PARAM_PATH, 'w', encoding='utf-8') as param_file:
                json.dump(hyper_param, param_file)

        if hyper_param['weight']['best'] < val_acc:
            hyper_param['weight']['best'] = val_acc
            torch.save(model.state_dict(), f'{ROOT}/best_model.pt')
            with open(HYPER_PARAM_PATH, 'w', encoding='utf-8') as param_file:
                json.dump(hyper_param, param_file)

        
if __name__ == "__main__":
    train()

# Epoch 1, Loss: 231928.1931, Val_loss: 8.9853, Val_acc: 0.0619
# Epoch 2, Loss: 211142.0237, Val_loss: 8.9709, Val_acc: 0.0666
# Epoch 3, Loss: 204052.7258, Val_loss: 9.0789, Val_acc: 0.0698
# Epoch 4, Loss: 200195.8319, Val_loss: 9.1823, Val_acc: 0.0708
# Epoch 5, Loss: 197818.5858, Val_loss: 9.3530, Val_acc: 0.0722
# Epoch 6, Loss: 196256.6193, Val_loss: 9.4595, Val_acc: 0.0719