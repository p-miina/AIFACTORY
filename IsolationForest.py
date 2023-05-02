import numpy as np
from sklearn.ensemble import IsolationForest

train_data = ['train_t0','train_t1','train_t2','train_t3',
              'train_t4','train_t5','train_t6','train_t7']
test_data = ['test_t0','test_t1','test_t2','test_t3',
              'test_t4','test_t5','test_t6','test_t7']

temp =  ['split0','split1','split2','split3','split4','split5','split6','split7']

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_samples = trial.suggest_uniform('max_samples', 0.1, 1.0)
    max_features = trial.suggest_uniform('max_features', 0.1, 1.0)
    contamination = trial.suggest_uniform('contamination', 0.0, 0.5)
    
    model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                            max_features=max_features, contamination=contamination, random_state=42)
    score = np.mean(cross_val_score(model, train_scaled, cv=5, scoring='accuracy'))

    if np.isnan(score):
        return float('-inf')
    else:
        return score


n = 8
for i in range(n):
  print('='*120)
  print(i)
  now_train = locals()[train_data[i]]
  now_test = locals()[test_data[i]]

  concat_data = pd.concat([now_train, now_test])

  scaler = MinMaxScaler()
  train_scaled = scaler.fit_transform(concat_data)
  test_scaled = scaler.transform(now_test)

  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=100)

  best_n_estimators = study.best_params['n_estimators']
  best_max_samples = study.best_params['max_samples']
  best_max_features = study.best_params['max_features']
  best_contamination = study.best_params['contamination']

  model = IsolationForest(n_estimators=best_n_estimators, max_samples=best_max_samples,
                          max_features=best_max_features, contamination=best_contamination, random_state=42)


  model.fit(train_scaled)
  y_pred = model.predict(test_scaled)

  results = np.where(y_pred == -1, 1, 0)
  locals()[temp[i]] = pd.DataFrame()
  locals()[temp[i]]['label']= results
  
label = pd.concat([split0,split1,split2,split3,split4,split5,split6,split7],axis=0)
label.reset_index(drop=True, inplace=True)

IFresult = pd.DataFrame()
test_raw = pd.read_csv('/content/drive/MyDrive/AIFactory/test_data.csv')
IFresult['type'] = test_raw['type']

IFresult['label']= np.where(label['label']==False, 0, 1)
IFresult['label'].value_counts()
