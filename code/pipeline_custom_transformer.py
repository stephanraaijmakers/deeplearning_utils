import sys
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Perceptron
from sklearn.datasets import fetch_20newsgroups
import numpy as np


class MyCaseTransformer(BaseEstimator, TransformerMixin):
    flag=""
    def __init__(self,flag="none"):
        self.flag=flag
    def fit(self,X,y=None):
        return self
    def transform(self, X,y=None):  
        X_copy=X.copy()
        if self.flag=="lower":
            X_copy=[w.lower() for  w in X_copy]
        elif self.flag=="upper":
            X_copy=[w.upper() for  w in X_copy]
        return X_copy


class Check(BaseEstimator, TransformerMixin):
    def __init__(self):
        True
    def fit(self,X,y=None):
        return self
    def transform(self, X,y=None):  
        print(X[0])
        return(X)


def main():

    categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    twenty_test  = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

    pipeline = Pipeline([('transform',MyCaseTransformer(flag="upper")),('check-1',Check()), ('vect', CountVectorizer(lowercase=False)),('check-2', Check()), ('clf', Perceptron())])
    pipeline.fit(twenty_train.data, twenty_train.target)

    X_test=twenty_test.data
    predicted = pipeline.predict(X_test)    
    print("Accuracy:",np.mean(predicted == twenty_test.target))


if __name__=="__main__":
    main()

