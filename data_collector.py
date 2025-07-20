import glob
import json
import re
from tqdm import tqdm
import pickle
from typing import Optional
from collections import Counter

from soynlp.utils import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import MaxScoreTokenizer

import torch
from torch.utils.data import Dataset

# 데이터를 받아서 일정하게 처리한 후 내보냄.

class DataCollector():
    """
    path: data가 담겨 있는 최상위 파일. 아래 train/labels/** 혹은 valid/labels/**로 저장되어야하며, json 파일을 glob으로 탐색함
    mode: train, valid
    tokenizerTrain() : 실행 전 미리 실행할 것
    openUp() : [json_path]를 주면 안의 내용 return
    """
    tokenizer: Optional[MaxScoreTokenizer] = None
    vocab:Optional[dict] = None

    def __init__(self, path, key:str, mode='train'):
        self.root_path = path

        if mode in ['train', 'valid']:
            self.mode = mode.strip().lower()
        else:
            raise ValueError(f"invalid input {mode}: Choose either train or valid")
        
        self.key:str = key

        self.train_data = self.openUp(glob.glob(f'{path}/train/labels/**/*.json', recursive=True))
        self.valid_data = self.openUp(glob.glob(f'{path}/valid/labels/**/*.json', recursive=True))

    def openUp(self, datases):
        out = []
        for datas in tqdm(datases, desc="Data Collecting "):
            with open(datas, encoding='utf-8') as json_data:
                out += json.load(json_data)
        return out

    def _clean_text(self, text: str) -> str:
        """
        정규 표현식을 사용하여 텍스트를 전처리합니다.
        - URL 제거
        - 이메일 주소 제거
        - 특수 문자 제거 (한글, 영어, 숫자, 기본 구두점(.,?!), 공백 제외)
        - 반복되는 공백을 하나의 공백으로 치환
        """
        text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'[^가-힣a-zA-Z0-9.,?! ]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenizerTrain(self, path=None, save_path=None):
        """
        path : 이미 word score table이 있을 경우 파일 주세요
        save_path : \'저장 위치(경로)/이름.pkl\' . 이 란이 비어있으면 저장하지 않음.
        """
        data = self.train_data + self.valid_data
        sentences = [data['sentence'] for data in data]

        if path:
            with open(path, "rb") as score:
                word_score_table = pickle.load(score)

        else:
            with open('corpus.txt', 'w', encoding='utf-8') as file:
                for sent in sentences:
                    sent = self._clean_text(sent)
                    file.write(sent + '\n\n')

            sent = DoublespaceLineCorpus('corpus.txt', iter_sent=True)
            Word_Extractor = WordExtractor()
            print("Tokenizer", end=" ")
            Word_Extractor.train(sent)
            word_score_table:dict = Word_Extractor.extract()
    
            if save_path:
                with open(save_path, "wb") as file:
                    pickle.dump(word_score_table, file)

        scores = {word:score.cohesion_forward for word, score in word_score_table.items()}

        DataCollector.tokenizer = MaxScoreTokenizer(scores=scores)

    def makeVocab(self, path=None, save_path=None):
        tokens = []
        current_mode = self.mode

        if path:
            with open(path, 'r') as vocabulary:
                vocab = json.load(vocabulary)
                DataCollector.vocab = vocab
                return vocab
            
        for mode in ['train', 'valid']:
            self.setMode(mode)
            for i in tqdm(range(len(self)), desc="Making Vocab..."):
                tokens += self[i]

        self.setMode(current_mode) 

        counter = Counter(tokens)
        vocab = {token: idx + 2 for idx, (token, _) in enumerate(counter.most_common())}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1

        DataCollector.vocab = vocab

        if save_path:
            with open(save_path, "w") as score:
                json.dump(vocab, score, ensure_ascii=False, indent=2)

        return vocab

    def head(self, num=5):
        if self.mode=='train':
            return self.train_data[:num]
        elif self.mode=='valid':
            return self.valid_data[:num]
        else:
            return 0
    
    def tail(self, num=5):
        if self.mode=='train':
            return self.train_data[-num:]
        elif self.mode=='valid':
            return self.valid_data[-num:]
        else:
            return 0

    def setMode(self, mode: str):
        mode = mode.strip().lower()
        if mode in ['train', 'valid']:
            self.mode = mode
        else:
            raise ValueError(f"Invalid mode: '{mode}'. Choose either 'train' or 'valid'.")

    def __len__(self) -> int:
        if self.mode=='train':
            return len(self.train_data)
        elif self.mode=='valid':
            return len(self.valid_data)
        else:
            return 0
        
    def __getitem__(self, idx) -> list: # [seq_len, data]

        if self.mode=='train':
            data:dict = self.train_data[idx]
        elif self.mode=='valid':
            data:dict = self.valid_data[idx]
        else:
            raise Exception("getitem error")

        key:str = self.key.strip()
        out = data[key]

        cleaned_text = self._clean_text(out)

        if DataCollector.tokenizer is not None:
           token = DataCollector.tokenizer.tokenize(cleaned_text) # [token dim]
        else:
            raise ValueError("Tokenizer is not initailized. Initialize tokenizerTrain() first.")
        
        return token
    
class KoreanDataset(Dataset):
    def __init__(self, collector:DataCollector):
        super(KoreanDataset, self).__init__()
        self.collector = collector
        self.vocab = DataCollector.vocab

    def encoder(self, tokens, vocab):
        return [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    def __len__(self):
        return len(self.collector)
    
    def __getitem__(self, idx):
        tokens = self.collector[idx] # ['나', '는', '학생', '이다']
        encoded = self.encoder(tokens, self.vocab) # [1, 2, 4, 3]

        input_seq = encoded[:-1]
        label_seq = encoded[1:]

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(label_seq, dtype=torch.long)