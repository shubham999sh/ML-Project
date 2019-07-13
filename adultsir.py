import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Desktop/sal.csv', names =['age',
                      'workclass',
                      'fnwgt',
                      'education',
                      'education-num','marital status',
                      'occupation',
                      'relationship',
                      'race',
                      'gender',
                      'capital-gain',
                      'capital loss',
                      'hour-per-week',
                      'native-country',
                      'salary'],na_values = ' ?')


X = dataset.iloc[:,0:14].values
y = dataset.iloc[:,-1].values

dataset.isnull().sum()
temp = pd.DataFrame(X[:,[1,6,13]])
temp[0].value_counts()
temp[1].value_counts()
temp[2].value_counts()


temp[0].fillna(' Private')        #still give a return copy of runnig data
temp[1].fillna(' Prof-specialty')
temp[2].fillna(' United-State')

X[:, [1,6,13]] = temp

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder() 
#lab.fit_transfor

X[:, 1] = lab.fit_transform(X[:, 1].astype(str))

X[:, 3] = lab.fit_transform(X[:, 3].astype(str))
X[:, 5] = lab.fit_transform(X[:, 5].astype(str))
X[:, 6] = lab.fit_transform(X[:, 6].astype(str))
X[:, 7] = lab.fit_transform(X[:, 7].astype(str))
X[:, 8] = lab.fit_transform(X[:, 8].astype(str))
X[:, 9] = lab.fit_transform(X[:, 9].astype(str))
X[:, 13] = lab.fit_transform(X[:, 13].astype(str))


y = lab.fit_transform(y)
lab.classes_

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1,3, 5, 9, 13])
X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# using K nearest neighbour algorithm here


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)



knn.score(X_train, y_train)
knn.score(X_test, y_test)
knn.score(X, y)


ypred = knn.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, ypred)


#Ensembel technique

from sklearn.svm import SVC
svm=SVC()
svm.fit(X,y)
svm.score(X,y)

from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()

from sklearn.naive_bayes import GaussianNB
n_b=GaussianNB()

from sklearn.ensemble import VotingClassifier
vot= VotingClassifier([('LR',log_reg),
                       ('KNN',knn),
                       #('DT',dtf),
                       ('NB',n_b),
                       ('SVM',svm)])
vot.fit(X,y)
vot.score(X,y)

from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(dtf, n_estimators = 5)
bag.fit(X,y)
bag.score(X,y)
