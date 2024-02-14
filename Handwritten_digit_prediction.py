import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

 
d= load_digits()
x = d.data
y = d.target
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
model = SVC(C=22)
model.fit(X_train,y_train)
p = model.score(X_test,y_test)
print(p)