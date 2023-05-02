import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터셋 불러오기
train = pd.read_csv('/content/drive/MyDrive/AIFactory/train_data.csv')
test = pd.read_csv('/content/drive/MyDrive/AIFactory/test_data.csv')

# type별 나누기
train_t0 = train.loc[train['type']==0]
train_t1 = train.loc[train['type']==1]
train_t2 = train.loc[train['type']==2]
train_t3 = train.loc[train['type']==3]
train_t4 = train.loc[train['type']==4]
train_t5 = train.loc[train['type']==5]
train_t6 = train.loc[train['type']==6]
train_t7 = train.loc[train['type']==7]

test_t0 = test.loc[test['type']==0]
test_t1 = test.loc[test['type']==1]
test_t2 = test.loc[test['type']==2]
test_t3 = test.loc[test['type']==3]
test_t4 = test.loc[test['type']==4]
test_t5 = test.loc[test['type']==5]
test_t6 = test.loc[test['type']==6]
test_t7 = test.loc[test['type']==7]

# type별 데이터 시각화
data = ['train_t0','train_t1','train_t2','train_t3',
        'train_t4','train_t5','train_t6','train_t7',
        'test_t0','test_t1','test_t2','test_t3',
        'test_t4','test_t5','test_t6','test_t7']

for d in data:
  print('*'*120)
  print(d)
  df = locals()[d]
  for col in df.columns:
    df[col].plot.line()
    plt.xlabel(col)
    plt.show()
    
# 이상 데이터 확인 및 처리
sns.boxplot(data = train_t1['motor_vibe'])

def outliers_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    
    return np.where((data> upper_bound)|(data< lower_bound))
out = outliers_iqr(train_t1['motor_vibe'])[0]

train_t1 = train_t1.drop(out)

train.drop(train[train['type']==1].index, inplace=True)
train.reset_index(drop=True, inplace=True)
train = pd.concat([train,train_t1])
train.reset_index(drop=True, inplace=True)

train.to_csv('train_outlier_0405.csv', index=False)
test.to_csv('test_outlier_0405.csv', index=False)
