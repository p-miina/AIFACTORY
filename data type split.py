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
