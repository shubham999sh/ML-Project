import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')

dataset = pd.read_csv('Desktop/At&T_Data.csv')
dataset['Titles'][0]
dataset['Reviews'][0]

processed_Titles = []
processed_Reviews = []

for i in range(113):
        Titles = re.sub('[^a-zA-Z]',' ', dataset['Titles'] [i])
        Titles = Titles.lower()
        Titles = Titles.split()
        Titles = [ps.stem(token) for token in Titles if not token in stopwords.words('english')]
        Titles = ' '.join(Titles)
        processed_Titles.append(Titles)
        Reviews = re.sub('[^a-zA-Z]',' ', dataset['Reviews'] [i])
        Reviews = Reviews.lower()
        Reviews = Reviews.split()
        Reviews = [ps.stem(token) for token in Reviews if not token in stopwords.words('english')]
        processed_Reviews.append(Reviews)
        
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=4000)

X = cv.fit_transform(processed_Titles)
Y = cv.fit_transform(processed_Reviews)
        
X = X.toarray()
y = dataset['Label'].values

from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()
n_b.fit(X,y)
n_b.score(X,y)

print(cv.get_feature_names())




















