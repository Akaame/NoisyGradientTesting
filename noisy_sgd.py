
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

from keras.optimizers import SGD
from keras_gradient_noise.gradient_noise import add_gradient_noise

def get_model(shape_input, shape_output):
    m = Sequential()
    m.add(Dense(100,input_shape=shape_input))
    for i in range(25):
        m.add(Dense(50,activation='relu'))
    m.add(Dense(shape_output, activation='softmax'))
    return m

m = get_model((784,),10)

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain = np.reshape(xtrain, [-1,784]).astype(np.float)
xtest = np.reshape(xtest, [-1,784]).astype(np.float)
xtrain /= 255.
xtest /= 255.
ytrain = to_categorical(ytrain, 10)
ytest = to_categorical(ytest,10)

noisy_sgd = add_gradient_noise(SGD)
m.compile(optimizer=noisy_sgd(), loss="categorical_crossentropy", metrics=["accuracy"])
n_epochs = 150
batch_size =128
history = m.fit(xtrain,ytrain,batch_size=batch_size,epochs=n_epochs,verbose=1)
print m.evaluate(xtest, ytest)

import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.show()