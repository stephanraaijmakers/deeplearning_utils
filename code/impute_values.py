import sys
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from numpy import nan
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def label_encode(y):
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    return to_categorical(encoded_y)

    
def create_keras_model():
    nb_classes=NB_CLASSES
    nb_feat=NB_FEAT
    model=Sequential()
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model



def create_sklearn_wrapper_keras():
    return KerasClassifier(build_fn=create_keras_model, epochs=150, batch_size=10, verbose=0)



global NB_CLASSES
NB_CLASSES=0
global NB_FEAT
NB_FEAT=0

def main():
    global NB_CLASSES
    global NB_FEAT

    # load dataset
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'

    dataframe = read_csv(url, header=None, na_values='?')
    # split into input and output elements

    data = dataframe.values
    ix = [i for i in range(data.shape[1]) if i != 23]
    X, y = data[:, ix], data[:, 23]
    NB_CLASSES=len(set(y))
    NB_FEAT=len(X[0])

    #y=label_encode(y)
    results = list()
    iterations = [str(i) for i in range(1, 3)]
    strategies = ['ascending'] # ,'descending', 'roman', 'arabic', 'random']

    max=0

    
    seed=1234 

    classifier=create_sklearn_wrapper_keras() 
    pipelines = [(s,it, Pipeline(steps=[('i', IterativeImputer(imputation_order=s,max_iter=int(it))), ('m', classifier)])) for s in strategies for it in iterations]


    for (s,it, pipeline) in pipelines:
        cv = StratifiedKFold(n_splits=10,shuffle=True, random_state=seed)
        scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        results.append(scores)
        print('>%s %s %.3f (%.3f)' % (it,s, mean(scores), std(scores)))
        if mean(scores)>max:
            winner_it=it
            winner_strat=s

#    pipeline = Pipeline(steps=[('i', IterativeImputer(imputation_order=winner_strat, max_iter=int(winner_it))), ('m', RandomForestClassifier())])

    imputer=IterativeImputer(imputation_order=winner_strat, max_iter=int(winner_it))
    pipeline = Pipeline(steps=[('i', imputer), ('m', classifier)])   
    pipeline.fit(X, y)
    
    test = [2, 1, 530101, 38.50, 66, 28, 3, 3, nan, 2, 5, 4, 4, nan, nan, nan, 3, 5, 45.00, 8.40, nan, nan, 2, 11300, 00000, 00000, 2]
    yhat = pipeline.predict([test])
    print("PRED:",yhat)
    print("IMPUTED:",imputer.transform([test]))


# plot model performance for comparison
#pyplot.boxplot(results, labels=strategies, showmeans=True)
#pyplot.xticks(rotation=45)
#pyplot.show()


if __name__=="__main__":
    main()

