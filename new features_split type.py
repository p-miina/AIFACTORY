import random
import pandas as pd
import numpy as np
import os
import time
import librosa
import librosa.display
import IPython.display as ipd
from tqdm.auto import tqdm
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.style.use('seaborn-white')

# 구글 드라이브 마운트
from google.colab import drive
drive.mount('/content/drive')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
# Seed 고정
seed_everything(42)

# 데이터셋 불러오기
train = pd.read_csv('/content/drive/MyDrive/AIFactory/train_outlier_0405_NF.csv')
test = pd.read_csv('/content/drive/MyDrive/AIFactory/test_outlier_0405_NF.csv')

# 새로운 피쳐 생성
hp_dict = {0: 30, 1: 20, 2: 10, 3: 50, 4:30, 5:30, 6:30, 7:30}
train['hp'] = train['type'].replace(hp_dict)
test['hp'] = test['type'].replace(hp_dict)

train['failure'] = (train['motor_temp']*train['motor_vibe']) / train['hp']
train['air_density'] = train['air_inflow']/ ((8314.4621 / 28.9644) * (train['air_end_temp'] + 273.15))
train['efficiency'] = (train['out_pressure'] * train['air_inflow'])/ (train['motor_current'] * train['motor_rpm']*0.001)
train['massflowrate'] = train['air_inflow']*train['air_density']
train['fluidpressure'] = train['out_pressure']/ ((8314.4621 / 28.9644) * (train['air_end_temp'] + 273.15))
train['load'] = (train['motor_current'] * train['motor_rpm'] * 0.001) / train['hp']

test['failure'] = (test['motor_temp']*test['motor_vibe']) / test['hp']
test['air_density'] = test['air_inflow']/ ((8314.4621 / 28.9644) * (test['air_end_temp'] + 273.15))
test['efficiency'] = (test['out_pressure'] * test['air_inflow'])/ (test['motor_current'] * test['motor_rpm']*0.001)
test['massflowrate'] = test['air_inflow']*test['air_density']
test['fluidpressure'] = test['out_pressure']/ ((8314.4621 / 28.9644) * (test['air_end_temp'] + 273.15))
test['load'] = (test['motor_current'] * test['motor_rpm'] * 0.001) / test['hp']

train.drop(['out_pressure', 'hp'],axis=1, inplace=True)
test.drop(['out_pressure', 'hp'],axis=1, inplace=True)
# train

# 데이터 타입별 나누기
train_t0 = train.loc[train['type']==0].drop('type',axis=1)
train_t1 = train.loc[train['type']==1].drop('type',axis=1)
train_t2 = train.loc[train['type']==2].drop('type',axis=1)
train_t3 = train.loc[train['type']==3].drop('type',axis=1)
train_t4 = train.loc[train['type']==4].drop('type',axis=1)
train_t5 = train.loc[train['type']==5].drop('type',axis=1)
train_t6 = train.loc[train['type']==6].drop('type',axis=1)
train_t7 = train.loc[train['type']==7].drop('type',axis=1)

test_t0 = test.loc[test['type']==0].drop('type',axis=1)
test_t1 = test.loc[test['type']==1].drop('type',axis=1)
test_t2 = test.loc[test['type']==2].drop('type',axis=1)
test_t3 = test.loc[test['type']==3].drop('type',axis=1)
test_t4 = test.loc[test['type']==4].drop('type',axis=1)
test_t5 = test.loc[test['type']==5].drop('type',axis=1)
test_t6 = test.loc[test['type']==6].drop('type',axis=1)
test_t7 = test.loc[test['type']==7].drop('type',axis=1)
