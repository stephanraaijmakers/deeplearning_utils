import sys
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def get_data(fname):
    df=pd.read_csv(fname,encoding="ISO-8859-1")
    df=df[:5000]
    df=df.fillna(method='ffill')
    X = df.drop('Tag', axis=1)
    y=df.Tag.values
    vect=DictVectorizer(sparse=False)
    X=vect.fit_transform(X.to_dict('records'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=0)
    return X_train,y_train, X_test, y_test


def main():
    X_train,y_train,X_test,y_test=get_data("ner_dataset.csv")
    pipeline = Pipeline([('clf', Perceptron())])
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)    
    print("Accuracy:",np.mean(predicted == y_test))


if __name__=="__main__":
    main()

