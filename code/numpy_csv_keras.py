import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import pandas as pd



def create_model(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model


def train_model_from_csv(csv_filename):
    data = np.genfromtxt(csv_filename, delimiter=',',skip_header=1)
    nb_columns=data.shape[1]-1
    model=create_model(nb_columns) # last column = y (label)
    X = data[:, :nb_columns]
    y = data[:, nb_columns]
    model.fit(X,y,epochs=100,validation_split=0.2)

