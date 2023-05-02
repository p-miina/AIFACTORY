import numpy as np
from sklearn.covariance import EllipticEnvelope

train_data = ['train_t0','train_t1','train_t2','train_t3',
              'train_t4','train_t5','train_t6','train_t7']
test_data = ['test_t0','test_t1','test_t2','test_t3',
              'test_t4','test_t5','test_t6','test_t7']
temp =  ['split0','split1','split2','split3','split4','split5','split6','split7']

# train_data = ['train_hp30','train_t1','train_t2','train_t3']
# test_data = ['test_hp30','test_t1','test_t2','test_t3']

# temp =  ['split0','split1','split2','split3']

n = 8
for i in range(n):
  print('='*120)
  print(i)
  now_train = locals()[train_data[i]]
  now_test = locals()[test_data[i]]

  concat_data = pd.concat([now_train, now_test])

  scaler = MinMaxScaler()
  train_scaled = scaler.fit_transform(now_train)
  test_scaled = scaler.transform(now_test)

  model = EllipticEnvelope(contamination = 0.01 , random_state = 42) 
  model.fit(train_scaled)

  y_pred = model.predict(test_scaled)

  results = np.where(y_pred == -1, 1, 0)
  locals()[temp[i]] = pd.DataFrame()
  locals()[temp[i]]['label']= results
  
label = pd.concat([split0,split1,split2,split3,split4,split5,split6,split7],axis=0)
label.reset_index(drop=True, inplace=True)

eeresult = pd.DataFrame()
test_raw = pd.read_csv('/content/drive/MyDrive/AIFactory/test_data.csv')
eeresult['type'] = test_raw['type']

eeresult= pd.concat([eeresult, label], axis=1)
eeresult.value_counts('label')
