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
plt.style.use('seaborn-white')

class CWRU(Dataset):
    def __init__(self, data):

        self.x = data

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx])
        
        return x

    def __len__(self):
        return len(self.x)

# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)

# data를 새롭게 representation하기 위한 AutoEncoder
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
    
        self.fc1 = nn.Linear(12, 6, bias=False)
        self.bn1 = nn.BatchNorm1d(6)        
        self.fc2 = nn.Linear(6, 3, bias=True)
        # self.bn2 = nn.BatchNorm1d(64)
        # self.fc3 = nn.Linear(64, 32, bias=False)
        # self.bn3 = nn.BatchNorm1d(32)
        # self.fc4 = nn.Linear(32, 16, bias=False)
        # self.bn4 = nn.BatchNorm1d(16)
        # self.fc5 = nn.Linear(16, 6, bias=True)
        # self.bn5 = nn.BatchNorm1d(6)

        self.defc1 = nn.Linear(3, 6, bias=True)
        self.defc2 = nn.Linear(6, 12, bias=False)
        # self.defc3 = nn.Linear(12, 64, bias=False)
        # self.defc4 = nn.Linear(64, 128, bias=False)
        # self.defc5 = nn.Linear(128, 33, bias=False)

        self.drop = nn.Dropout(0.2)
        self.swish = nn.SiLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def encoder(self, x):
        x = self.tanh(self.bn1(self.fc1(x)))
        x = self.tanh(self.fc2(x))
        # x = self.tanh(self.bn3(self.fc3(x)))
        # x = self.tanh(self.bn4(self.fc4(x)))
        # x = self.tanh(self.bn5(self.fc5(x)))

        return x

    def decoder(self, x):
        x = self.defc1(x)
        x = self.defc2(x)
        # x = self.defc3(x)
        # x = self.defc4(x)
        # x = self.defc5(x)

        return x

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
        return x_hat, z


# function to get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def train(model, train_loader, optimizer):
    
    model.train()
    
    running_loss = 0.0
    len_data = len(train_loader.dataset)
    
    for x in train_loader:
        x = x.to(device)
        
        x_hat, _ = model(x)
        loss = loss_func(x, x_hat)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        running_loss += loss.item()
    
    return running_loss/len_data

def eval(model, dataloader):
    """Testing the Deep SVDD model"""

    scores = []
    model.eval()
    print('Testing...')
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            x_hat, z = model(x)
            score = torch.mean( torch.abs(x - x_hat) , axis=1)
            scores.extend(score.cpu().numpy())
            
    return np.array(scores), z

def get_pred_label(model_pred, t):
    # (0:정상, 1:불량)로 Label 변환
    model_pred = np.where(model_pred <= t, 0, model_pred)
    model_pred = np.where(model_pred > t, 1, model_pred)
    return model_pred
  
train_data = ['train_t0','train_t1','train_t2','train_t3',
              'train_t4','train_t5','train_t6','train_t7']
test_data = ['test_t0','test_t1','test_t2','test_t3',
              'test_t4','test_t5','test_t6','test_t7']

temp =  ['split0','split1','split2','split3','split4','split5','split6','split7']

n = 8
for i in range(n):
  print('='*120)
  print(i)
  now_train = locals()[train_data[i]]
  now_test = locals()[test_data[i]]

  concat_data = pd.concat([now_train, now_test])

  scaler = RobustScaler()
  train_scaled = scaler.fit_transform(concat_data)
  test_scaled = scaler.transform(now_test)

  Config = {
      "num_epochs" : 500,
      "batch_size" : 16,
      "learning_rate" : 0.0001
  }


  train_dataset=CWRU(train_scaled)
  test_dataset=CWRU(test_scaled)

  train_loader = DataLoader(dataset=train_dataset, batch_size=Config['batch_size'], shuffle=True)
  test_loader = DataLoader(dataset=test_dataset, batch_size=Config['batch_size'], shuffle=False)


  model = AutoEncoder().to(device)
  loss_func = nn.L1Loss()
  opt = optim.Adam(model.parameters(), lr=Config['learning_rate'])

  summary(model, (12,))

  loss_history = {'train': []}
  start_time = time.time()

  for epoch in range(Config['num_epochs']):   
      
      current_lr = get_lr(opt)
      print('Epoch {}/{}, current lr={}'.format(epoch+1, Config['num_epochs'], current_lr))
      
      train_loss = train(model, train_loader, opt)
      loss_history['train'].append(train_loss)

      print('train loss: %.6f, time: %.4f min' %(train_loss, (time.time()-start_time)/60))
      print('-'*10)

  path = '/content/drive/MyDrive/AIFactory/weight/'
  w_path = path + str(i)+'_'+'model_weights.pth'
  torch.save(model.state_dict(), w_path)
  print('Model weights saved at epoch {}'.format(epoch+1))

  plt.plot(np.arange(len(loss_history['train'])), loss_history['train'])
  plt.title("Training loss")
  plt.show()

  with torch.no_grad():
      
      for j,x in enumerate(train_loader):
          x = x.to(device)
          opt.zero_grad()
          output, z = model.forward(x)
          break


  scores, z = eval(model, train_loader)
  # Train data (정상 데이터)에서 발견할 수 있는 score의 최댓값인 t를 임계치로 설정
  # 정상데이터 관찰할 수 있는 관측치 중 가장 큰 값이므로, 임계치 이하의 값은 
  # 정상 데이터일 것이라는 가정
  t=np.percentile(scores, q=95)

  print(scores.shape) 

  scores_, z_ = eval(model, test_loader)

  # t=np.percentile(scores, q=98)

  # print(scores.shape)

  # 히스토그램
  plt.style.use('default')
  plt.rcParams['font.size'] = 12

  plt.hist(scores, bins=50, density=True, alpha=0.7, label='Train data')
  plt.hist(scores_, bins=50, density=True, alpha=0.7, label='Test data')
  plt.axvline(x=t, c='red', linestyle=':', label='Threshold')
  plt.title("Anomaly score")
  plt.legend()
  plt.show()

  train_pred = get_pred_label(scores, t)
  # Counter(train_pred)

  test_pred = get_pred_label(scores_, t)

  test_pred = list(map(int, test_pred))

  
  # Counter(test_pred)

  result_df = pd.DataFrame()

  locals()[temp[i]] = pd.DataFrame()
  locals()[temp[i]]['label']=test_pred
  
label = pd.concat([split0,split1,split2,split3,split4,split5,split6,split7],axis=0)
label.reset_index(drop=True, inplace=True)

AEresult = pd.DataFrame()
test_raw = pd.read_csv('/content/drive/MyDrive/AIFactory/test_data.csv')
AEresult['type'] = test_raw['type']

# result= pd.concat([result, label], axis=1)

AEresult['label']= np.where(label['label']==False, 0, 1)

AEresult['label'].value_counts()
