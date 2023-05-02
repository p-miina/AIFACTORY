from sklearn import svm
import numpy as np
import optuna
from sklearn import svm
from sklearn.model_selection import cross_val_score

train_data = ['train_t0','train_t1','train_t2','train_t3',
              'train_t4','train_t5','train_t6','train_t7']
test_data = ['test_t0','test_t1','test_t2','test_t3',
              'test_t4','test_t5','test_t6','test_t7']

temp =  ['split0','split1','split2','split3','split4','split5','split6','split7']

# n = 8
# for i in range(n):
#   print('='*120)
#   print(i)
#   now_train = locals()[train_data[i]]
#   now_test = locals()[test_data[i]]

#   scaler = RobustScaler()
#   train_scaled = scaler.fit_transform(now_train)
#   test_scaled = scaler.transform(now_test)

#   # One-class SVM 모델 생성 및 학습
#   clf = svm.OneClassSVM(nu=0.01)
#   clf.fit(train_scaled)

#   # 새로운 데이터 이상치 판별
#   y_pred = clf.predict(test_scaled)

#   locals()[temp[i]] = pd.DataFrame()
#   locals()[temp[i]]['label']= np.where(y_pred==-1, 1, 0)
  

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameters to optimize
    kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
    gamma = trial.suggest_loguniform('gamma', 1e-4, 10.0)
    nu = trial.suggest_uniform('nu', 0.05, 0.5)
    
    # Create the One-Class SVM model
    model = svm.OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    
    # Evaluate the model with cross-validation
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

  scaler = MinMaxScaler()
  train_scaled = scaler.fit_transform(now_train)
  test_scaled = scaler.transform(now_test)

  # Run the hyperparameter optimization with Optuna
  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=100)

  # Print the best hyperparameters found by Optuna
  best_kernel = study.best_params['kernel']
  best_gamma = study.best_params['gamma']
  best_nu = study.best_params['nu']
  print(f"Best kernel: {best_kernel}, best gamma: {best_gamma}, best nu: {best_nu}")

  # Train the One-Class SVM model with the best hyperparameters
  best_model = svm.OneClassSVM(kernel=best_kernel, gamma=best_gamma, nu=best_nu)
  best_model.fit(train_scaled)

  svm_pred = best_model.predict(test_scaled)

  locals()[temp[i]] = pd.DataFrame()
  locals()[temp[i]]['label']= np.where(svm_pred==-1, 1, 0)
  
label = pd.concat([split0,split1,split2,split3,split4,split5,split6,split7],axis=0)
label.reset_index(drop=True, inplace=True)

SVMresult = pd.DataFrame()
test_raw = pd.read_csv('/content/drive/MyDrive/AIFactory/test_data.csv')
SVMresult['type'] = test_raw['type']

# result= pd.concat([result, label], axis=1)

SVMresult['label']= np.where(label['label']==False, 0, 1)
SVMresult['label'].value_counts()
