import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

path = "/Users/lucas/My Repo/KNN-sklearn/csv"
os.chdir(path)

df = pd.read_csv('winequality-white.csv', index_col=0)
#print(df.head())

# Use scaler object to conduct a transforms
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(df.drop('quality', axis=1))
scaled_features = scaler.transform(df.drop('quality',axis=1))

df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
#print(df_feat.head())

# Set the X and ys
X = df_feat
y = df['quality']

# splitting sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

# Create KNN instance
#knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='brute', p=1)
#knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='brute', p=2)

knn = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', p=1)
#knn = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', p=2)

# Fit (i.e. traing) the model
knn.fit(X_train, y_train)

# Use the .predict() method to make predictions from the X_test subset
pred = knn.predict(X_test)

# print knn mean accuracy
print(knn.score(X_test, y_test))

# Print out classification report and confusion matrix
print(classification_report(y_test, pred))

# Print out confusion matrix
cmat = confusion_matrix(y_test, pred)
#df_cm = pd.DataFrame(cmat, range(6),
#                  range(6))

plt.figure(figsize = (10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(cmat, annot=True, fmt="d") # font size

#print(cmat)
print('TP - True Negative {}'.format(cmat[0,0]))
print('FP - False Positive {}'.format(cmat[0,1]))
print('FN - False Negative {}'.format(cmat[1,0]))
print('TP - True Positive {}'.format(cmat[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cmat[0,1],cmat[1,0]]),np.sum(cmat))))

error_rate = []
acc = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    #pred_i = knn.predict(X_test)
    #error_rate.append(np.mean(pred_i != y_test))
    acc.append(knn.score(X_test, y_test))

# Configure and plot error rate over k values

plt.figure(figsize=(10,4))
plt.plot(range(1,40), acc, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. K-Values')
plt.xlabel('K-Values')
plt.ylabel('Accuracy')
plt.show()