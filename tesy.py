import numpy as np
import matplotlib.pyplot as pd
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import re

nltk.download('stopwords')

dataset = pd.read_csv('Desktop/At&T_Data.csv')

dataset['Titles'][0]
dataset['Reviews'][0]

processed_Titles = []
processed_Reviews = []

for i in range(113):
    Titles = re.sub('[^a-zA-Z]', ' ',dataset['Titles'][i])
    Titles = Titles.lower()
    Titles = Titles.split()
    Titles = [ps.stem(token) for token in Titles if not token in stopwords.words('english')]
    Titles = ' '.join(Titles)
    processed_Titles.append(Titles)
    Reviews = re.sub('[^a-zA-Z]', ' ',dataset['Reviews'][i])
    Reviews = Reviews.lower()
    Reviews = Reviews.split()
    Reviews = [ps.stem(token) for token in Reviews if not token in stopwords.words('english')]
    Reviews = ' '.join(Reviews)
    processed_Reviews.append(Reviews)
    

        
from sklearn.feature_extraction.text import CountVectorizer
cv  = CountVectorizer(max_features= 3000)
X = cv.fit_transform(processed_Titles)
X1 = cv.fit_transform(processed_Reviews)

X = X.toarray()
y = dataset['Label'].values

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2)


from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()
n_b.fit(X_train,y_train)
n_b.score(X_train,y_train)


n_b.score(X_test,y_test)

n_b.score(X,y)




    



