# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Collect a labeled dataset of emails, distinguishing between spam and non-spam.

2.Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.

3.Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF

4.Split the dataset into a training set and a test set.

5.Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.

6.Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.

7.Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.

8.Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.

9.Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed

## Program:

/*
Program to implement the SVM For Spam Mail Detection..
Developed by:Mahalakshmi.k 
RegisterNumber:212222240057  
*/

import chardet

file='/content/spam.csv'

with open(file, 'rb') as rawdata:

  result = chardet.detect(rawdata.read(1000000))
  
result

import pandas as pd

data=pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

x_train=cv.fit_transform(x_train)

x_test=cv.transform(x_test)

from sklearn.svm import SVC

svc=SVC()

svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)

y_pred

from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)

accuracy

## Output:

1.Result output

![Result output](https://github.com/maha712/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121156360/42d5ffe7-ff46-4a2b-9d0e-82b05bd40edc)

2.data.head()

![2 data head()](https://github.com/maha712/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121156360/024830db-947e-4e43-896b-0e4be5de08f3)

3.data.info()

![3 data info()](https://github.com/maha712/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121156360/909ecbca-4d5c-4bc4-8ac3-78e7fa7a60e6)

4.data.isnull().sum()

![4 data isnull() sum()](https://github.com/maha712/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121156360/966564f4-8e53-4bcd-a7bb-908bcb038ae7)

5.Y_prediction value

![5 Y_prediction value](https://github.com/maha712/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121156360/9c7435a1-dae7-4b5e-a73b-0fbcdf15dbbd)

6.Accuracy value

![6 Accuracy value](https://github.com/maha712/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121156360/4c97de69-4c01-438a-a635-48c8424962c8)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
