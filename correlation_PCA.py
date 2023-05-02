import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def scatter_plot(data, x, y, hue):

  corr = np.corrcoef(data[x], data[y])[0, 1]
  print(f'\n상관계수 : {corr}\n')

  fig, ax = plt.subplots(figsize=(5, 3))

  sns.scatterplot(data=data, x=x, y=y, hue=hue)
  ax.set_title(f"{x} & {y} Scatter")

  plt.show()
  
train_data = ['train_t0','train_t1','train_t2','train_t3',
              'train_t4','train_t5','train_t6','train_t7']
test_data = ['test_t0','test_t1','test_t2','test_t3',
              'test_t4','test_t5','test_t6','test_t7']

temp =  ['split0','split1','split2','split3','split4','split5','split6','split7']

n = 8
for i in range(n):
  now_train = locals()[train_data[i]]
  now_test = locals()[test_data[i]]

  scaler = RobustScaler()
  X_train = scaler.fit_transform(now_train)
  X_test = scaler.transform(now_test)
  
  # correlation plot
  fig, ax = plt.subplots(figsize=(8, 5))

  sns.heatmap(X_train.corr(), cmap='PuBu')
  ax.set_title("Heat Map")
  plt.show()
  
  col_names = X_train.columns
  for i in range(len(col_names)):
    for j in range(i+1, len(col_names)):
      scatter_plot(data=X_train, x= str(col_names[i]), y=str(col_names[j]), hue=None)
  
  # 
  pca = PCA()
  pca.fit(X_train)

  pca = PCA(n_components=0.8)
  train_pca = pca.fit_transform(X_train)
  test_pca = pca.transform(X_test)

  print(train_data[i], test_data[i])
  print(train_pca.shape, test_pca.shape)
