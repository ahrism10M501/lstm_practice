
import os
import json
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import re_datacolllector
from re_datacolllector import DataCollector
from LSTM import LSTM

### 프로젝트 파일 관리 ###
PATH = ''
DATA_PATH = ''
HYPER_PARAM_PATH = ''
CORPUS = None
WORD_SCORE = None
VOCAB = None
