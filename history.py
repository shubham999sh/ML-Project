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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_train, y_train)
knn.score(X_test, y_test)
knn.score(X, y)
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
from sklearn.svm import SVC
svm=SVC()
svm.fit(X,y)
svm.score(X,y)
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
from sklearn.tree import DecisionTreeClassifier
dtf=DecisionTreeClassifier()
from sklearn.naive_bayes import GaussianNB
n_b=GaussianNB()
from sklearn.ensemble import VotingClassifier
vot= VotingClassifier([('LR',log_reg),
                       ('KNN',knn),
                       ('DT',dtf),
                       ('NB',n_b),
                       ('SVM',svm)])
vot.fit(X,y)
vot.score(X,y)
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(dtf, n_estimators = 5)
bag.fit(X,y)
bag.score(X,y)
param_grid={'n_neighbors=5': [1,2,3,4,5,6,7,8,9]}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(knn, param_grid)
grid.fit(X,y)
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(knn, param_grid)
grid.fit(X,y)
param_grid={'n_neighbors=5': [1,2,3,4,5,6,7,8,9]}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(knn, param_grid)
grid.fit(X,y)
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(knn, n_estimators = 5)
bag.fit(X,y)
bag.score(X,y)
param_grid={'n_neighbors=5': [1,2,3,4,5,6,7,8,9]}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(knn, param_grid)
grid.fit(X,y)

## ---(Sat Jul 13 14:59:51 2019)---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
nltk.download('stopwords')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
from sklearn.svm import SVC
svm=SVC()
svm.fit(X,y)
svm.score(X,y)
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
from sklearn.tree import DecisionTreeClassifier
dtf=DecisionTreeClassifier()
from sklearn.tree import DecisionTreeClassifier
dtf=DecisionTreeClassifier()

from sklearn.naive_bayes import GaussianNB
n_b=GaussianNB()
from sklearn.ensemble import VotingClassifier
vot= VotingClassifier([('LR',log_reg),
                       ('KNN',knn),
                       ('DT',dtf),
                       ('NB',n_b),
                       ('SVM',svm)])
vot.fit(X,y)
vot.score(X,y)
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(knn, n_estimators = 5)
bag.fit(X,y)
bag.score(X,y)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X,y)
rf.score(X,y)
param_grid={'n_neighbors=5': [1,2,3,4,5,6,7,8,9]}
param_grid1=[{'criteria'  : ['gin','entropy']},
             {'max_depth' : [1,2,3,4,5,6,7,8,9]}]
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(knn, param_grid)
grid.fit(X,y)
grid.best_estimator_
param_grid={'n_neighbors=5': [1,2,3,4,5,6,7,8,9]}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(knn, param_grid)
grid.fit(X,y)
grid.best_estimator_
param_grid={'n_neighbors=5': [1,2,3,4,5,6,7,8,9]}
param_grid1=[{'criteria'  : ['gin','entropy']},
             {'max_depth' : [1,2,3,4,5,6,7,8,9]}]

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(knn, param_grid)
grid1 = GridSearchCV(dtf, param_grid1)
grid.fit(X,y)
grid.best_estimator_
grid.best_index_
grid.best_params_
grid.best_score_

grid1.fit(X,y)
grid1.best_estimator_
grid1.best_index_
grid1.best_params_
grid1.best_score_
param_grid1=[{'criteria'  : ['gin','entropy']},
             {'max_depth' : [1,2,3,4,5,6,7,8,9]}]
grid1 = GridSearchCV(dtf, param_grid1)
grid1.fit(X,y)