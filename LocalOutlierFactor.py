import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import cross_val_score
import optuna

def objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 2, 50)
    algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    leaf_size = trial.suggest_int('leaf_size', 10, 100)
    contamination = trial.suggest_uniform('contamination', 0.0, 0.5)

    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
    score = np.mean(cross_val_score(model, train_scaled, cv=5, scoring='accuracy'))

    if np.isnan(score):
        return float('-inf')
    else:
        return score



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

  scaler = MinMaxScaler()
  train_scaled = scaler.fit_transform(now_train)
  test_scaled = scaler.transform(now_test)

  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=100)

  best_n_neighbors = study.best_params['n_neighbors']
  best_contamination = study.best_params['contamination']

  model = LocalOutlierFactor(n_neighbors=best_n_neighbors, contamination=best_contamination)
  model.fit(train_scaled)

  lof_pred = model.fit_predict(test_scaled)

  # results = np.where(lof_pred == -1, 1, 0)
  locals()[temp[i]] = pd.DataFrame()
  locals()[temp[i]]['label']= results

label = pd.concat([split0,split1,split2,split3,split4,split5,split6,split7],axis=0)
label.reset_index(drop=True, inplace=True)

IFresult = pd.DataFrame()
test_raw = pd.read_csv('/content/drive/MyDrive/AIFactory/test_data.csv')
Iresult['type'] = test_raw['type']

Iresult['label']= np.where(label['label']==False, 0, 1)
Iresult['label'].value_counts()
