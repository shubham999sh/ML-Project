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
bag = BaggingClassifier(knn, n_estimators = 5)
bag.fit(X,y)
bag.score(X,y)


# it is a random classifier for DT Algorithm
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X,y)
rf.score(X,y)








param_grid={'n_neighbors=5': [1,2,3,4,5,6,7,8,9]}


param_grid1=[{'criteria'  : ['gin','entropy']},
             {'max_depth' : [1,2,3,4,5,6,7,8,9]}]
# grid search
from sklearn.model_selection import GridSearchCV




grid = GridSearchCV(knn, param_grid)

grid1 = GridSearchCV(dtf, param_grid1)
grid.fit(X,y)

grid.best_estimator_
grid.best_index_
grid.best_params_
grid.best_score_















