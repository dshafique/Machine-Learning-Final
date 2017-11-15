import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv('train.csv').as_matrix()
clf=DecisionTreeClassifier()

#training data
xTrain = data[0:21000, 1:]
train_label = data[0:21000, 0]

clf.fit(xTrain, train_label)

#testing data
xTest = data[21000:, 1:]
actual_label = data[21000:, 0]

p = clf.predict(xTest) #predict

count = 0
for i in range(0, 21000):
    count+=1 if p[i]==actual_label[i] else 0
print("accuracy= ", (count/21000)*100)

