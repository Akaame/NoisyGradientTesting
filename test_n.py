
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from keras_gradient_noise import add_gradient_noise
from keras.optimizers import SGD, Adam, RMSprop

def get_model(shape_input, shape_output):
    m = Sequential()
    m.add(Dense(100,input_shape=shape_input))
    for i in range(25):
        m.add(Dense(100,activation='relu'))
    m.add(Dense(shape_output, activation='softmax'))
    return m

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain = np.reshape(xtrain, [-1,784]).astype(np.float)
xtest = np.reshape(xtest, [-1,784]).astype(np.float)
xtrain /= 255.
xtest /= 255.
ytrain = to_categorical(ytrain, 10)
ytest = to_categorical(ytest,10)

no_tests = 10

history_normal_acc = []
history_normal_loss = []
eval_results_normal = []
history_noisy_acc = []
history_noisy_loss = []
eval_results_noisy = []

for i in range(no_tests):
    m = get_model((784,),10)
    np.random.seed(7)
    normal = Adam()
    m.compile(optimizer=normal, loss="categorical_crossentropy", metrics=["accuracy"])
    n_epochs = 150
    batch_size =256
    history = m.fit(xtrain,ytrain,batch_size=batch_size,epochs=n_epochs,verbose=1)

    res = m.evaluate(xtest, ytest)
    eval_results_normal.append(res)
    m = get_model((784,),10)
    np.random.seed(7)
    noisy = add_gradient_noise(Adam)
    m.compile(optimizer=noisy(), loss="categorical_crossentropy", metrics=["accuracy"])
    history_noisy = m.fit(xtrain,ytrain,batch_size=batch_size,epochs=n_epochs,verbose=1)
    res = m.evaluate(xtest, ytest)
    eval_results_noisy.append(res)

    history_normal_loss.append(history.history["loss"])
    history_noisy_loss.append(history_noisy.history["loss"])
    
    history_normal_acc.append(history.history["acc"])
    history_noisy_acc.append(history_noisy.history["acc"])

print("Normal metodun sonuclari: [Loss, Accuracy]")
print(np.mean(eval_results_normal,axis=0))
print("Gurultulu metodun sonuclari: [Loss, Accuracy]")
print(np.mean(eval_results_noisy,axis=0))

import matplotlib.pyplot as plt
plt.title("Training Loss")
plt.plot(np.mean(history_normal_loss,axis=0))
plt.plot(np.mean(history_noisy_loss,axis=0))
plt.legend(["Normal","Noisy"])
plt.show()

plt.title("Training Accuracy")
plt.plot(np.mean(history_normal_acc,axis=0))
plt.plot(np.mean(history_noisy_acc,axis=0))
plt.legend(["Normal","Noisy"])
plt.show()