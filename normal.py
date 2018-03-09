
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
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

m.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
n_epochs = 150
batch_size =128
history = m.fit(xtrain,ytrain,batch_size=batch_size,epochs=n_epochs,verbose=1)
print m.evaluate(xtest, ytest)