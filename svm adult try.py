import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Desktop/adult.data')

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


# it is a random classifier for DT Algorithm
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X,y)
rf.score(X,y)
















