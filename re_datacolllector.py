import os
import re
import json
import pickle
import glob

from tqdm import tqdm
from typing import Optional
from collections import Counter

from soynlp.utils import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import MaxScoreTokenizer

import torch
from torch.utils.data import Dataset

class DataCollector():
    """
    DataCollector는 수많은 json 데이터를 서치해서 하나씩 열어보내준다.
    이떄, 모든 josn을 한 번에 열지 않고 하나씩 열어서 보내줌. 이 때문에 연산이 조금 많지만, 램 사용량은 줄일 수 있음.

    path: data의 ROOT. ROOT/[train, valid, tset]/[labels, texts]/*.json
    phase: [train, valid, test]
    annot: [True, False:default] -> ROOT/phase/[sents(False), labels(True)]

    """
    def __init__(self, path, phase, annot):
        super(DataCollector, self).__init__()
        self.path = path

        if phase in ['train', 'valid', 'test']:
            self.phase = phase
        else:
            raise ValueError("{} is invalid phase".format(phase))
        
        if annot == False:
            in_folder_data = glob.glob(f'{path}/{phase}/texts/**/*.json', recursive=True)
        elif annot == True:
            in_folder_data = glob.glob(f'{path}/{phase}/labels/**/*.json', recursive=True)
        else:
            raise ValueError("{} is invalid. annot must be bool type".format(annot))
        
        self.data_path = {}
        self.total_data_length = 0
        self.file_cache = {}

        self.all_data = []
        for path in in_folder_data:
            with open(path, 'r', encoding='utf-8') as f:
                self.all_data.extend(json.load(f))

        # for json_file_path in in_folder_data:
        #     try:
        #         with open(json_file_path, 'r', encoding='utf-8') as datas:
        #             data = json.load(datas)
        #         self.data_path[json_file_path] = len(data) # [file_path: len(data)] -> RAM에 올라가는 데이터양을 줄이기 위해 인덱스로 남겨두기. 실제로는 파일하나씩 열어서 할거임.
        #         self.total_data_length += len(data)
        #         del data
                
        #     except FileNotFoundError:
        #         print(f"경고: 파일을 찾을 수 없습니다: {json_file_path}")
        #     except json.JSONDecodeError:
        #         print(f"경고: 파일에서 JSON을 디코드할 수 없습니다: {json_file_path}")
        #     except Exception as e:
        #         print(f"파일 {json_file_path} 처리 중 예기치 않은 오류 발생: {e}")

        # if self.total_data_length == 0:
        #     print(f"어떤 데이터도 찾을 수 없습니다.")

    def __len__(self) -> int:
        return len(self.all_data)

    # FIXME : 매 인덱싱마다 파일을 여닫아서 성능하락. 다만, 메모리 효율 좋음. 메모리 효율은 가져가되, 성능은 높이는 방법 필요
    def __getitem__(self, idx) -> dict:
        # if not (0 <= idx < self.total_data_length):
        #     raise IndexError(f"인덱스 {idx}는 전체 데이터 크기 {self.total_data_length} 범위 밖 입니다.")
        
        # idx_buff = idx
        # for key, length in self.data_path.items():
        #     if idx_buff < length:
        #         if key not in self.file_cache:
        #             with open(key, 'r', encoding='utf-8') as data_file:
        #                 self.file_cache[key] = json.load(data_file)
        #         return self.file_cache[key][idx_buff]
        #     idx_buff -= length

        # raise RuntimeError("DataCollector.__getitem__() indexing Error Occurred")
        return self.all_data[idx]
            
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

def claenText(sent:str):
    """
        정규 표현식을 사용하여 텍스트를 전처리합니다.
        - URL 제거
        - 이메일 주소 제거
        - 특수 문자 제거 (한글, 영어, 숫자, 기본 구두점(.,?!), 공백 제외)
        - 반복되는 공백을 하나의 공백으로 치환
    """
    sent = re.sub(r'http[s]?://\S+|www\.\S+', '', sent)
    sent = re.sub(r'\S*@\S*\s?', '', sent)
    sent = re.sub(r'[^가-힣a-zA-Z0-9.,?! ]', '', sent)
    sent = re.sub(r'\s+', ' ', sent).strip()
    return sent

def getToken(tokenizer, sent):
    return tokenizer.tokenize(claenText(sent))

def makeWordScore(collector:DataCollector, corpus:Optional[str]=None, save_path:Optional[str]=None) -> dict:
    """
        collector : Data path가 입력되어있는 DataCollector 객체
        corpus: corpus.txt의 위치. 존재하지 않으면 corpus.txt 만듦.
        word_score: word_score의 위치
        return: 훈련된 MaxScoreTokenizer를 반환함.
    """
    if not len(collector):
        raise ValueError("Collector didn't have any datas")
    
    if not corpus:
        with open('corpus.txt', 'w', encoding='utf-8') as file:
            for data in tqdm(collector, desc="Making corpus.txt... "):
                sent = claenText(data['sentence'])
                file.write(sent + '\n\n')
    
    sents = DoublespaceLineCorpus('corpus.txt', iter_sent=True)
    Word_extractor = WordExtractor()
    print("WordScore ", end="")
    Word_extractor.train(sents)
    word_score_table:dict = Word_extractor.extract()
    
    scores = {word:score.cohesion_forward for word, score in word_score_table.items()}

    if save_path:
        with open(os.path.join(save_path, "word_score.pkl"), 'wb') as save_:
            pickle.dump(scores, save_)
            
    return scores

def makeVocab(collector:DataCollector, tokenizer, save_path:Optional[str]=None) -> dict:
    tokens = []
    for data in tqdm(collector, desc="Making vocab.json"):
        tokens += getToken(tokenizer, data['sentence'])
    
    counter = Counter(tokens)

    vocab = {token: idx + 2 for idx, (token, _) in enumerate(counter.most_common())}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1

    if save_path:
        with open(os.path.join(save_path, "vocab.json"), "w", encoding='utf-8') as score:
            json.dump(vocab, score, ensure_ascii=False, indent=2)
            
    print("vocab.json Done")

    return vocab

class KoreanDataset(Dataset):
    def __init__(self,
                 collector:DataCollector,
                 tokenizer,
                 vocab:dict[str, int],
                 device='cpu'):
        
        super(KoreanDataset, self).__init__()

        self.collector = collector
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.device = device

    def encoder(self, tokens):
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
    
    def __len__(self):
        return len(self.collector)
    
    def __getitem__(self, idx):
        data = self.collector[idx]

        # FIXME: 매번 tokenize하는 것으로 인해 병목현상 발생
        sent = claenText(data.get('sentence', ''))
        tokens = getToken(self.tokenizer, sent)
        encoded = self.encoder(tokens)

        if len(encoded) <2:
            print(f"경고: 인덱스 {idx}의 문장이 너무 짧아 시퀀스를 만들 수 없습니다. PAD 토큰으로 채웁니다.")
            input_seq = [self.vocab['<PAD>']]
            label_seq = [self.vocab['<PAD>']]
        else:
            input_seq = encoded[:-1]
            label_seq = encoded[1:]

        x = torch.tensor(input_seq, dtype=torch.long)
        y = torch.tensor(label_seq, dtype=torch.long)
        x = x.to(self.device)
        y = y.to(self.device)

        return x, y