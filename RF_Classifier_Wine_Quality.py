#Import Dependencies
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

%matplotlib inline
import os

from google.colab import drive
drive.mount('/content/gdrive')

wine = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/RedWine/winequality-red.csv')
!cat /content/gdrive/My Drive/Colab Notebooks/RedWine/winequality-red.csv

!pip install fastai   #Run everytime you have a new VM

bins = (2, 6.5, 8)
group_names = ["bad", "good"]
wine["quality"] = pd.cut(wine["quality"], bins = bins, labels = group_names)

label_quality = LabelEncoder()

wine["quality"] = label_quality.fit_transform(wine["quality"])
wine["quality"].value_counts()

sns.countplot(wine["quality"])

X = wine.drop("quality", axis = 1)
y = wine["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
sc = StandardScalar()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
print(X_train)

rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

print(classification_report(y_test, pred_rfc))
