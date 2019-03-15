1.Use the use case in the class:
 a.Add Dense layers to the existing code and check how the accuracy changes.

from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
dataset = pd.read_csv("C:/Users/pooja/Documents/DeepLearning_Lesson1/DeepLearning_Lesson1/diabetes.csv", header=None).values
# print(dataset)
import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, 0:8], dataset[:, 8],
                                                    test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(250, input_dim=8, activation='relu')) # hidden layer 1
my_first_nn.add(Dense(250, activation='relu')) # hidden layer 2
my_first_nn.add(Dense(20, activation='relu')) # hidden layer 3pip
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam')
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=50, verbose=0,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test, verbose=0))